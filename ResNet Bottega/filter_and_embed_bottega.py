import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np


def safe_imports():
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
        from PIL import Image  # noqa: F401
        import pandas as pd  # noqa: F401
        from tqdm import tqdm  # noqa: F401
    except ModuleNotFoundError as e:
        missing = str(e).split("No module named ")[-1].strip("'\"")
        print(
            f"Missing dependency: {missing}.\n"
            "Please install required packages: pip install torch torchvision pillow pandas tqdm",
            file=sys.stderr,
        )
        raise


def list_images(folder: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return [p for p in folder.rglob("*") if p.suffix.lower() in exts]


def load_model(device: str = "cpu"):
    import torch
    from torchvision.models import resnet50, ResNet50_Weights
    from torch import nn

    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)
    model.eval()
    model.to(device)

    # Classifier for filtering with ImageNet labels
    preprocess = weights.transforms()
    categories = weights.meta.get("categories", [])

    # Feature extractor (remove final FC)
    feat_model = resnet50(weights=weights)
    feat_model.fc = nn.Identity()
    feat_model.eval()
    feat_model.to(device)

    return model, feat_model, preprocess, categories


def classify_topk(
    model,
    preprocess,
    img_paths: List[Path],
    device: str,
    categories: List[str],
    batch_size: int = 16,
    topk: int = 5,
) -> Tuple[List[List[Tuple[str, float]]], List[np.ndarray]]:
    import torch
    from PIL import Image
    from tqdm import tqdm

    results: List[List[Tuple[str, float]]] = []
    probs_all: List[np.ndarray] = []

    def load_tensor(p: Path):
        with Image.open(p) as im:
            im = im.convert("RGB")
            return preprocess(im)

    tensors = []
    indices = []
    for idx, p in enumerate(img_paths):
        try:
            t = load_tensor(p)
            tensors.append(t)
            indices.append(idx)
        except Exception as e:
            print(f"Warning: failed to read {p}: {e}", file=sys.stderr)
            results.append([])
            probs_all.append(np.array([]))

    if not tensors:
        return results, probs_all

    ds = list(zip(indices, tensors))

    with torch.no_grad():
        for i in tqdm(range(0, len(ds), batch_size), desc="Classifying"):
            batch = ds[i : i + batch_size]
            if not batch:
                continue
            idxs, tens = zip(*batch)
            x = torch.stack(list(tens), dim=0).to(device)
            logits = model(x)
            prob = torch.nn.functional.softmax(logits, dim=1)
            top_p, top_i = prob.topk(topk, dim=1)
            top_p = top_p.cpu().numpy()
            top_i = top_i.cpu().numpy()
            for j in range(top_i.shape[0]):
                labels = [(categories[int(ii)], float(pp)) for ii, pp in zip(top_i[j], top_p[j])]
                pos = idxs[j]
                # ensure list is sized to pos
                while len(results) <= pos:
                    results.append([])
                    probs_all.append(np.array([]))
                results[pos] = labels
                probs_all[pos] = prob[j].cpu().numpy()

    return results, probs_all


def is_sunglasses(labels: List[Tuple[str, float]], keywords: List[str], min_prob: float) -> bool:
    for name, p in labels:
        low = name.lower()
        if any(k in low for k in keywords) and p >= min_prob:
            return True
    return False


def embed_images(
    feat_model,
    preprocess,
    img_paths: List[Path],
    device: str,
    batch_size: int = 16,
) -> np.ndarray:
    import torch
    from PIL import Image
    from tqdm import tqdm

    def load_tensor(p: Path):
        with Image.open(p) as im:
            im = im.convert("RGB")
            return preprocess(im)

    feats = []
    with torch.no_grad():
        buf = []
        idxs = []
        for idx, p in enumerate(img_paths):
            try:
                t = load_tensor(p)
                buf.append(t)
                idxs.append(idx)
            except Exception as e:
                print(f"Warning: failed to read {p}: {e}", file=sys.stderr)
                feats.append(None)

            if len(buf) == batch_size:
                x = torch.stack(buf, dim=0).to(device)
                out = feat_model(x)  # [B, 2048]
                out = out.view(out.size(0), -1).cpu().numpy()
                for j in range(out.shape[0]):
                    feats.append(out[j])
                buf.clear()
                idxs.clear()

        if buf:
            x = torch.stack(buf, dim=0).to(device)
            out = feat_model(x)
            out = out.view(out.size(0), -1).cpu().numpy()
            for j in range(out.shape[0]):
                feats.append(out[j])

    # Filter out None entries if any failures
    feats = [f for f in feats if f is not None]
    return np.vstack(feats) if feats else np.empty((0, 2048), dtype=np.float32)


def main():
    safe_imports()
    import torch
    import pandas as pd

    parser = argparse.ArgumentParser(description="Filter images by sunglasses and embed with ResNet50")
    parser.add_argument("--input-dir", type=str, default="images_bottega", help="Input images directory")
    parser.add_argument("--keep-dir", type=str, default="images_bottega_sunglasses", help="Output dir for sunglasses")
    parser.add_argument(
        "--reject-dir", type=str, default="images_bottega_not_sunglasses", help="Output dir for non-sunglasses"
    )
    parser.add_argument(
        "--keywords",
        type=str,
        nargs="*",
        default=["sunglass", "sunglasses", "shades"],
        help="Keywords to detect sunglasses in ImageNet labels",
    )
    parser.add_argument("--min-prob", type=float, default=0.05, help="Minimum probability for a keyword match")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--no-move", action="store_true", help="Do not move files; just report and embed kept ones")
    parser.add_argument(
        "--embeddings-out",
        type=str,
        default="bottega_resnet50_embeddings.csv",
        help="CSV file to write embeddings",
    )
    parser.add_argument(
        "--predictions-out",
        type=str,
        default="bottega_resnet50_predictions.csv",
        help="CSV file to write top-k predictions for auditing",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    keep_dir = Path(args.keep_dir)
    reject_dir = Path(args.reject_dir)
    keep_dir.mkdir(parents=True, exist_ok=True)
    reject_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"Using device: {device}")

    imgs = list_images(input_dir)
    if not imgs:
        print(f"No images found in {input_dir}")
        return 1

    print(f"Found {len(imgs)} images under {input_dir}")

    model, feat_model, preprocess, categories = load_model(device)

    topk_labels, _ = classify_topk(
        model,
        preprocess,
        imgs,
        device=device,
        categories=categories,
        batch_size=args.batch_size,
        topk=args.topk,
    )

    # Decide which to keep
    keep_mask = [is_sunglasses(lbls, args.keywords, args.min_prob) for lbls in topk_labels]
    keep_paths = [p for p, k in zip(imgs, keep_mask) if k]
    reject_paths = [p for p, k in zip(imgs, keep_mask) if not k]

    print(f"Keeping {len(keep_paths)} images; rejecting {len(reject_paths)} images")

    # Save predictions for auditing
    rows = []
    for p, lbls, k in zip(imgs, topk_labels, keep_mask):
        row = {"path": str(p), "keep": bool(k)}
        for i, (name, prob) in enumerate(lbls):
            row[f"top{i+1}_label"] = name
            row[f"top{i+1}_prob"] = prob
        rows.append(row)
    pd.DataFrame(rows).to_csv(args.predictions_out, index=False)
    print(f"Wrote predictions to {args.predictions_out}")

    # Optionally move files
    if not args.no_move:
        for p in keep_paths:
            dest = keep_dir / p.name
            if dest.exists():
                # Avoid overwrite; add a suffix
                stem, suf = p.stem, p.suffix
                i = 1
                while (keep_dir / f"{stem}_{i}{suf}").exists():
                    i += 1
                dest = keep_dir / f"{stem}_{i}{suf}"
            p.rename(dest)
        for p in reject_paths:
            dest = reject_dir / p.name
            if dest.exists():
                stem, suf = p.stem, p.suffix
                i = 1
                while (reject_dir / f"{stem}_{i}{suf}").exists():
                    i += 1
                dest = reject_dir / f"{stem}_{i}{suf}"
            p.rename(dest)
        print(f"Moved files to {keep_dir} and {reject_dir}")
        # After moving, update keep_paths to new locations
        keep_paths = list_images(keep_dir)

    # Compute embeddings on kept images
    if keep_paths:
        feats = embed_images(
            feat_model,
            preprocess,
            keep_paths,
            device=device,
            batch_size=args.batch_size,
        )
        # Save as CSV with columns: path, f0..f2047
        cols = [f"f{i}" for i in range(feats.shape[1])]
        df = pd.DataFrame(feats, columns=cols)
        df.insert(0, "path", [str(p) for p in keep_paths])
        df.to_csv(args.embeddings_out, index=False)
        print(f"Wrote embeddings for {len(keep_paths)} images to {args.embeddings_out}")
    else:
        print("No images to embed after filtering.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

