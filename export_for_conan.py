#!/usr/bin/env python3
"""
export_for_conan.py
Runs ViT-Base-MAE embeddings + Ward Agglomerative clustering on the 6 luxury eyewear brands.
Uses pure PyTorch (avoids numpy/scipy conflicts).
Outputs:
  - cluster_data.json  →  conan-insight-hub/public/
  - representative images  →  conan-insight-hub/public/brands/<brand>/<cluster_id>/
"""

import csv
import json
import re
import shutil
from datetime import date
from pathlib import Path

import torch
import torch.nn.functional as F_nn
import timm
from PIL import Image

# ── paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT     = Path(__file__).parent
FRONTEND_ROOT = REPO_ROOT.parent / "conan-insight-hub"
OUT_JSON      = FRONTEND_ROOT / "public" / "cluster_data.json"
OUT_IMAGES    = FRONTEND_ROOT / "public" / "brands"

# ── brand config ──────────────────────────────────────────────────────────────
BRANDS = {
    "Dolce & Gabbana": {
        "folder": "images_D&G",
        "csv":    "images_D&G/dolcegabbana_products.csv",
        "id_col": "Product_ID",
    },
    "Bottega Veneta": {
        "folder": "images_bottega",
        "csv":    "images_bottega/bottega_products.csv",
        "id_col": "Product Name",
    },
    "Cartier": {
        "folder": "images_Cartier",
        "csv":    "images_Cartier/Cartier_products.csv",
        "id_col": "Product Name",
    },
    "Fendi": {
        "folder": "images_Fendi",
        "csv":    "images_Fendi/Fendi_products.csv",
        "id_col": "Product Name",
    },
    "Prada": {
        "folder": "images_Prada",
        "csv":    "images_Prada/prada_products.csv",
        "id_col": "Product Name",
    },
    "YSL": {
        "folder": "images_ysl",
        "csv":    "images_ysl/ysl_products.csv",
        "id_col": "Product Name",
    },
}

K_RANGE          = range(3, 21)   # thesis used range(3, min(21, n-1)) with max(4, best_k)
MIN_K            = 4              # thesis enforces minimum 4 clusters
REPS_PER_CLUSTER = 3
N_COLORS         = 3
PCA_DIMS         = None           # None = skip PCA, use full 768-dim embeddings like the thesis
BATCH_SIZE       = 16


# ── helpers ───────────────────────────────────────────────────────────────────

def parse_price(raw: str) -> float | None:
    m = re.search(r"\d+(?:[.,]\d+)?", str(raw).replace(".", "").replace(",", "."))
    return float(m.group()) if m else None


def img_num(stem: str) -> int | None:
    m = re.search(r"(\d+)$", stem)
    return int(m.group(1)) if m else None


# ── torch math ────────────────────────────────────────────────────────────────

def pca_t(X: torch.Tensor, k: int) -> torch.Tensor:
    _, _, V = torch.pca_lowrank(X, q=k, niter=4)
    return X @ V  # (N, k)


def ward_full_t(X: torch.Tensor) -> tuple:
    """
    Run full Ward agglomerative clustering (pure PyTorch, no scipy).
    Returns (merge_history, cluster_members) where merge_history is a list of
    (cluster_i, cluster_j, new_cluster_id) and cluster_members maps each
    cluster id to the list of original sample indices it contains.
    """
    n = len(X)
    max_c = 2 * n
    centroids = torch.zeros(max_c, X.shape[1])
    centroids[:n] = X.clone()
    sizes = torch.zeros(max_c)
    sizes[:n] = 1.0

    cluster_members = {i: [i] for i in range(n)}
    active = list(range(n))
    next_id = n
    merge_history: list[tuple] = []

    while len(active) > 1:
        m   = len(active)
        ids = torch.tensor(active, dtype=torch.long)
        acts = centroids[ids]          # (m, d)
        szs  = sizes[ids]              # (m,)

        # Ward criterion: n_i * n_j / (n_i + n_j) * ||μ_i - μ_j||²
        diff    = acts.unsqueeze(0) - acts.unsqueeze(1)   # (m, m, d)
        sq_dist = (diff * diff).sum(-1)                   # (m, m)
        si, sj  = szs.unsqueeze(1), szs.unsqueeze(0)
        ward    = si * sj / (si + sj) * sq_dist
        ward.fill_diagonal_(float("inf"))

        flat_idx = ward.argmin().item()
        ii, jj   = flat_idx // m, flat_idx % m
        ci, cj   = active[ii], active[jj]

        ni, nj = sizes[ci].item(), sizes[cj].item()
        centroids[next_id] = (ni * centroids[ci] + nj * centroids[cj]) / (ni + nj)
        sizes[next_id]     = ni + nj
        cluster_members[next_id] = cluster_members[ci] + cluster_members[cj]

        merge_history.append((ci, cj, next_id))
        active = [x for x in active if x != ci and x != cj] + [next_id]
        next_id += 1

    return merge_history, cluster_members


def cut_tree_t(merge_history: list, cluster_members: dict, n: int, k: int) -> torch.Tensor:
    """Cut the dendrogram to obtain k clusters (replay first n-k merges)."""
    active = set(range(n))
    for ci, cj, new_id in merge_history[: n - k]:
        active.discard(ci)
        active.discard(cj)
        active.add(new_id)

    out = torch.zeros(n, dtype=torch.long)
    for label, cid in enumerate(sorted(active)):
        for member in cluster_members[cid]:
            out[member] = label
    return out


def silhouette_t(X: torch.Tensor, lbl: torch.Tensor) -> float:
    unique = lbl.unique()
    if len(unique) < 2:
        return -1.0
    D = torch.cdist(X, X)
    scores = []
    for i in range(len(X)):
        same = (lbl == lbl[i]).clone()
        same[i] = False
        if not same.any():
            scores.append(0.0)
            continue
        a     = D[i, same].mean().item()
        b     = min(D[i, lbl == kk].mean().item() for kk in unique if kk != lbl[i])
        denom = max(a, b)
        scores.append((b - a) / denom if denom > 0 else 0.0)
    return float(sum(scores) / len(scores))


def kmeans_t(X: torch.Tensor, k: int, n_iters: int = 50, n_init: int = 3) -> torch.Tensor:
    """Kept only for dominant-color extraction (pixel clustering)."""
    best_lbl, best_inertia = None, float("inf")
    for _ in range(n_init):
        centers = X[torch.randperm(len(X))[:k]].clone()
        lbl = torch.zeros(len(X), dtype=torch.long)
        for _ in range(n_iters):
            D   = torch.cdist(X, centers)
            lbl = D.argmin(1)
            new_c = torch.stack([
                X[lbl == c].mean(0) if (lbl == c).any() else centers[c]
                for c in range(k)
            ])
            if torch.allclose(centers, new_c):
                break
            centers = new_c
        inertia = sum(torch.norm(X[lbl == c] - centers[c]).item() for c in range(k))
        if inertia < best_inertia:
            best_inertia, best_lbl = inertia, lbl.clone()
    return best_lbl


def dominant_colors_t(paths: list, n: int = 3) -> list[str]:
    pixels = []
    for p in paths[:20]:
        try:
            img = Image.open(p).convert("RGB").resize((30, 30))
            arr = torch.tensor(list(img.getdata()), dtype=torch.float32)
            mask = ~(arr > 230).all(1)
            if mask.any():
                pixels.append(arr[mask])
        except Exception:
            pass
    if not pixels:
        return ["#808080"]
    all_px = torch.cat(pixels, 0)
    lbl    = kmeans_t(all_px, min(n, len(all_px)), n_iters=50, n_init=3)
    colors = []
    for c in range(n):
        if not (lbl == c).any():
            continue
        mean = all_px[lbl == c].mean(0).clamp(0, 255).to(torch.int32)
        colors.append(f"#{mean[0]:02x}{mean[1]:02x}{mean[2]:02x}")
    return colors


# ── ViT embedding ─────────────────────────────────────────────────────────────

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """Convert PIL RGB image to normalized float tensor without touching numpy."""
    img = img.convert("RGB").resize((224, 224), Image.BILINEAR)
    raw = img.tobytes()
    t   = torch.frombuffer(bytearray(raw), dtype=torch.uint8).clone()
    t   = t.reshape(224, 224, 3).permute(2, 0, 1).float() / 255.0
    return (t - _MEAN) / _STD


@torch.no_grad()
def embed_images(model, paths, device) -> torch.Tensor:
    model.eval()
    result = []
    for i in range(0, len(paths), BATCH_SIZE):
        batch = []
        for p in paths[i:i + BATCH_SIZE]:
            try:
                batch.append(pil_to_tensor(Image.open(p)))
            except Exception:
                batch.append(torch.zeros(3, 224, 224))
        feats = model(torch.stack(batch).to(device))
        result.append(feats.cpu())
        print(f"  embedded {min(i+BATCH_SIZE, len(paths))}/{len(paths)}", end="\r")
    print()
    return torch.cat(result, 0)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading ViT-Base-MAE …")
    model = timm.create_model("vit_base_patch16_224.mae", pretrained=True, num_classes=0)
    model = model.to(device)

    OUT_IMAGES.mkdir(parents=True, exist_ok=True)

    output = {
        "generated_at":   str(date.today()),
        "model":          "ViT-Base-patch16-224-MAE",
        "total_products": 0,
        "brands":         {},
    }

    for brand, cfg in BRANDS.items():
        print(f"\n{'─'*55}")
        print(f"  {brand}")

        folder = REPO_ROOT / cfg["folder"]
        paths  = sorted(p for ext in ("jpg", "jpeg", "png", "webp")
                        for p in folder.glob(f"*.{ext}") if p.is_file())
        print(f"  {len(paths)} images")

        # price map { numeric_id → price }
        price_map: dict[int, float] = {}
        with open(REPO_ROOT / cfg["csv"], newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                n = img_num(str(row[cfg["id_col"]]))
                p = parse_price(row["Price"])
                if n is not None and p is not None:
                    price_map[n] = p

        # embeddings (on MPS/CPU)
        print("  Extracting embeddings …")
        X = embed_images(model, paths, device)   # (N, 768) on CPU

        # StandardScaler — thesis applies zero-mean unit-variance normalization to ViT embeddings
        mean = X.mean(0)
        std  = X.std(0).clamp(min=1e-8)
        Xr   = (X - mean) / std   # standardized 768-dim, same as thesis StandardScaler

        # PCA reduce (skip if PCA_DIMS is None, use full embeddings like the thesis)
        if PCA_DIMS is not None:
            d  = min(PCA_DIMS, Xr.shape[0] - 1, Xr.shape[1] - 1)
            Xr = pca_t(Xr, d)

        # Ward agglomerative — run once, cut at each k
        print("  Ward agglomerative clustering …")
        merges, members = ward_full_t(Xr)
        k_upper = min(K_RANGE.stop, len(paths))
        scores_list = []
        for k in range(K_RANGE.start, k_upper):
            lbl = cut_tree_t(merges, members, len(paths), k)
            s   = silhouette_t(Xr, lbl)
            print(f"    k={k}  silhouette={s:.3f}")
            scores_list.append((k, s))

        # Thesis k-selection: pick the first k>4 (in descending silhouette order)
        # that is ≥95% of the max silhouette — mirrors find_optimal_k_agglomerative
        scores_list.sort(key=lambda x: x[1], reverse=True)
        best_s  = scores_list[0][1]
        best_k  = scores_list[0][0]
        for k_val, s in scores_list:
            if k_val > 4 and s / best_s > 0.95:
                best_k = k_val
                break
        best_k  = max(MIN_K, best_k)
        best_lbl = cut_tree_t(merges, members, len(paths), best_k)
        print(f"  → best k={best_k}  silhouette={best_s:.3f}")

        slug      = brand.lower().replace(" ", "_").replace("&", "and")
        brand_dir = OUT_IMAGES / slug
        brand_dir.mkdir(exist_ok=True)

        clusters_out = []
        all_prices   = list(price_map.values())

        for cid in range(best_k):
            mask   = best_lbl == cid
            cpaths = [paths[i] for i in range(len(paths)) if mask[i].item()]
            if not cpaths:
                continue

            # prices
            cprices = [price_map[img_num(p.stem)] for p in cpaths
                       if img_num(p.stem) in price_map]
            price_stats: dict = {}
            if cprices:
                price_stats = {
                    "min":   int(min(cprices)),
                    "mean":  int(sum(cprices) / len(cprices)),
                    "max":   int(max(cprices)),
                    "count": len(cprices),
                }

            # dominant colors
            colors = dominant_colors_t(cpaths, N_COLORS)

            # order all images by distance to centroid (most typical/representative first)
            cembs    = Xr[mask]
            centroid = cembs.mean(0)
            dists    = torch.norm(cembs - centroid, dim=1)
            all_idx  = dists.argsort().tolist()         # all, most-typical first
            all_sorted = [cpaths[i] for i in all_idx]

            # copy ALL images to frontend
            cdir = brand_dir / str(cid)
            cdir.mkdir(exist_ok=True)
            image_refs = []
            for img_path in all_sorted:
                dest = cdir / img_path.name
                shutil.copy2(img_path, dest)
                image_refs.append(f"/brands/{slug}/{cid}/{img_path.name}")

            clusters_out.append({
                "id":              cid,
                "n_products":      int(mask.sum().item()),
                "images":          image_refs[:REPS_PER_CLUSTER],  # first 3 for thumbnail
                "all_images":      image_refs,                      # all for full grid view
                "dominant_colors": colors,
                "price_stats":     price_stats,
            })

        brand_prices = list(price_map.values())
        output["brands"][brand] = {
            "n_products":       len(paths),
            "n_clusters":       best_k,
            "silhouette_score": round(best_s, 4),
            "clusters":         clusters_out,
            "price_stats": {
                "min":  int(min(brand_prices)) if brand_prices else 0,
                "mean": int(sum(brand_prices) / len(brand_prices)) if brand_prices else 0,
                "max":  int(max(brand_prices)) if brand_prices else 0,
            },
        }
        output["total_products"] += len(paths)

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n✓  JSON   → {OUT_JSON}")
    print(f"✓  images → {OUT_IMAGES}")
    print(f"✓  total  → {output['total_products']} products")


if __name__ == "__main__":
    main()
