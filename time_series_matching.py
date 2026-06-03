#!/usr/bin/env python3
"""
time_series_matching.py — ViT embeddings + cross-season matching for the
F25 / S26 time series. Currently scoped to the two brands we have re-scraped:
Bottega Veneta and Dolce & Gabbana.

Pipeline:

  1. Load all product images for both snapshots, both brands
  2. Embed each image with ViT-Base-MAE (timm) → 768-d vector
  3. L2-normalize the vectors (so cosine similarity = dot product)
  4. Cache embeddings to snapshots/embeddings_{brand}_{snapshot}.pt
  5. Per brand, compute cosine similarity matrix F25 (M) × S26 (N)
  6. Classify:
       persisted  — F25 product whose best S26 match is ≥ threshold
       new_in_s26 — S26 product whose best F25 match is < threshold
       discontinued_from_f25 — F25 product whose best S26 match is < threshold
  7. Emit persistence.json with classifications + cos-sim distribution stats

The README documents 0.92 as the threshold of choice (cos sim ≥ 0.92 ≈
"same product, photo re-shot"); we expose --threshold so we can sweep
later when we have F25↔S26 known-match calibration pairs.

Run:
    python time_series_matching.py --threshold 0.92
"""

import argparse
import json
import time
from pathlib import Path

import torch
import timm
from PIL import Image

REPO_ROOT = Path(__file__).parent
EMBED_DIR = REPO_ROOT / "snapshots" / "embeddings"
EMBED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = REPO_ROOT / "snapshots" / "persistence.json"

# Snapshot → brand → image folder
SNAPSHOTS = {
    "F25": {
        "Bottega Veneta": "images_bottega",
        "Dolce & Gabbana": "images_D&G",
    },
    "S26": {
        "Bottega Veneta": "snapshots/S26/raw/bottega",
        "Dolce & Gabbana": "snapshots/S26/raw/dg",
    },
}

# ViT-MAE was producing cos sim ≥ 0.995 across the board — the MAE CLS token
# (pre-trained with masked reconstruction, not contrastive) doesn't separate
# product instances. DINO (self-distillation, instance-discrimination pretext)
# gives an actual useful distribution where 0.92 lives in the right place.
VIT_MODEL = "vit_base_patch16_224.dino"
BATCH_SIZE = 16
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """RGB → CHW float32 tensor, ImageNet normalized, no numpy involved."""
    img = img.convert("RGB").resize((224, 224), Image.BILINEAR)
    raw = img.tobytes()
    t = torch.frombuffer(bytearray(raw), dtype=torch.uint8).clone()
    t = t.reshape(224, 224, 3).permute(2, 0, 1).float() / 255.0
    return (t - _MEAN) / _STD


@torch.no_grad()
def embed_paths(model, paths: list[Path]) -> torch.Tensor:
    """Return (N, 768) tensor of L2-normalized embeddings."""
    out = []
    for i in range(0, len(paths), BATCH_SIZE):
        batch = []
        for p in paths[i:i + BATCH_SIZE]:
            try:
                batch.append(pil_to_tensor(Image.open(p)))
            except Exception:
                batch.append(torch.zeros(3, 224, 224))
        feats = model(torch.stack(batch).to(DEVICE))
        out.append(feats.cpu())
        print(f"    embedded {min(i + BATCH_SIZE, len(paths))}/{len(paths)}", end="\r")
    print()
    emb = torch.cat(out, dim=0)
    # L2 normalize so cosine sim = dot product
    emb = emb / emb.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return emb


def load_or_embed(model, brand: str, snapshot: str) -> tuple[torch.Tensor, list[Path]]:
    """Embed (or load cached) all images for one (brand, snapshot)."""
    folder = REPO_ROOT / SNAPSHOTS[snapshot][brand]
    paths = sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No images in {folder}")

    # Cache key based on brand + snapshot + image count
    cache_key = f"{brand.replace(' ', '_').replace('&', 'and')}_{snapshot}_n{len(paths)}.pt"
    cache_path = EMBED_DIR / cache_key

    if cache_path.exists():
        emb = torch.load(cache_path, map_location="cpu")
        print(f"  [cache] loaded {emb.shape} from {cache_path.name}")
        return emb, paths

    print(f"  [embed] {brand} {snapshot}: {len(paths)} images …")
    emb = embed_paths(model, paths)
    torch.save(emb, cache_path)
    print(f"  [embed] saved {cache_path.name}")
    return emb, paths


def quantiles(t: torch.Tensor, qs=(0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)) -> dict:
    return {f"p{int(q*100):02d}": round(t.quantile(q).item(), 4) for q in qs}


def match_brand(brand: str, threshold: float, model) -> dict:
    f25_emb, f25_paths = load_or_embed(model, brand, "F25")
    s26_emb, s26_paths = load_or_embed(model, brand, "S26")

    # Cosine similarity matrix (M, N) since both are L2-normalized
    cos = f25_emb @ s26_emb.T

    # Best S26 match for each F25 product
    f25_best_cos, f25_best_idx = cos.max(dim=1)
    # Best F25 match for each S26 product
    s26_best_cos, s26_best_idx = cos.max(dim=0)

    persisted, new_in_s26, discontinued = [], [], []

    for j, (best_cos, best_idx) in enumerate(zip(s26_best_cos.tolist(), s26_best_idx.tolist())):
        if best_cos >= threshold:
            persisted.append({
                "f25_img": f25_paths[best_idx].name,
                "s26_img": s26_paths[j].name,
                "cos_sim": round(best_cos, 4),
            })
        else:
            new_in_s26.append({
                "s26_img": s26_paths[j].name,
                "best_f25_match": f25_paths[best_idx].name,
                "best_cos_sim": round(best_cos, 4),
            })

    for i, (best_cos, best_idx) in enumerate(zip(f25_best_cos.tolist(), f25_best_idx.tolist())):
        if best_cos < threshold:
            discontinued.append({
                "f25_img": f25_paths[i].name,
                "best_s26_match": s26_paths[best_idx].name,
                "best_cos_sim": round(best_cos, 4),
            })

    return {
        "threshold": threshold,
        "f25_count": len(f25_paths),
        "s26_count": len(s26_paths),
        "n_persisted": len(persisted),
        "n_new_in_s26": len(new_in_s26),
        "n_discontinued_from_f25": len(discontinued),
        "persistence_rate_f25": round(1 - len(discontinued) / max(len(f25_paths), 1), 4),
        "novelty_rate_s26": round(len(new_in_s26) / max(len(s26_paths), 1), 4),
        "cos_sim_distribution": {
            "best_per_s26": quantiles(s26_best_cos),
            "best_per_f25": quantiles(f25_best_cos),
        },
        "persisted": persisted,
        "new_in_s26": new_in_s26,
        "discontinued_from_f25": discontinued,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.92,
                    help="Cosine similarity cutoff for 'same product' (default 0.92).")
    args = ap.parse_args()

    print(f"[device] {DEVICE}")
    print(f"[model]  loading {VIT_MODEL} …")
    model = timm.create_model(VIT_MODEL, pretrained=True, num_classes=0).to(DEVICE)
    model.eval()

    t0 = time.time()
    result = {"threshold": args.threshold, "brands": {}}

    for brand in SNAPSHOTS["S26"].keys():  # only brands present in both F25 and S26
        print(f"\n[{brand}] matching F25 ↔ S26 …")
        result["brands"][brand] = match_brand(brand, args.threshold, model)
        b = result["brands"][brand]
        print(f"  F25={b['f25_count']}  S26={b['s26_count']}  threshold={args.threshold}")
        print(f"  persisted={b['n_persisted']}  new={b['n_new_in_s26']}  discontinued={b['n_discontinued_from_f25']}")
        print(f"  persistence_rate(F25)={b['persistence_rate_f25']:.1%}  novelty_rate(S26)={b['novelty_rate_s26']:.1%}")
        q = b["cos_sim_distribution"]["best_per_s26"]
        print(f"  best-cos-sim per S26 product — p10={q['p10']}  p50={q['p50']}  p90={q['p90']}")

    OUTPUT_PATH.write_text(json.dumps(result, indent=2))
    print(f"\n[done] wrote {OUTPUT_PATH}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
