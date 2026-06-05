#!/usr/bin/env python3
"""
cluster_benchmark.py — head-to-head clustering benchmark.

Grid (per brand × mode):

    backbone  ∈ { MAE (vit_base_patch16_224.mae),
                  DINO (vit_base_patch16_224.dino) }
    prep      ∈ { raw 768-d,
                  PCA(50) }
    algo      ∈ { Ward Agglomerative,
                  K-Means (n_init=10) }

For each (backbone, prep, algo) we sweep k ∈ [3, 12] and report:

    silhouette_score      (higher better, > 0)
    davies_bouldin_score  (lower better, > 0)
    calinski_harabasz     (higher better)
    inertia               (within-cluster sum of squares; lower = tighter)

K is chosen per metric (silhouette argmax) for the "best k" column. We
also compute Adjusted Rand Index (ARI) between Ward and K-Means at the
shared best k — if ARI is high, two structurally different algorithms
agree on the partition, which is the strongest signal that the cluster
structure is real (not an artefact of one algorithm's bias).

Output: snapshots/benchmark.json  (everything, including all k sweeps)
        Stdout: a readable winner-takes-all summary table.

Only Bottega Veneta and Dolce & Gabbana — the two brands with both F25
and S26 catalogues. Modes: joint (F25 ∪ S26), F25-only, S26-only.
"""

import json
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import timm
from PIL import Image

REPO_ROOT = Path(__file__).parent
EMBED_DIR = REPO_ROOT / "snapshots" / "embeddings"
OUT_PATH = REPO_ROOT / "snapshots" / "benchmark.json"

# Where are the actual image folders for each (brand, season)?
PATHS = {
    "Bottega Veneta": {
        "F25": "images_bottega",
        "S26": "snapshots/S26/raw/bottega",
    },
    "Dolce & Gabbana": {
        "F25": "images_D&G",
        "S26": "snapshots/S26/raw/dg",
    },
}

BACKBONES = {
    "MAE":  "vit_base_patch16_224.mae",
    "DINO": "vit_base_patch16_224.dino",
}

K_MIN = 3
K_MAX = 12
PCA_DIMS = 50

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Embedding + PCA helpers
# ─────────────────────────────────────────────────────────────────────────────

def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    img = img.convert("RGB").resize((224, 224), Image.BILINEAR)
    raw = img.tobytes()
    t = torch.frombuffer(bytearray(raw), dtype=torch.uint8).clone()
    t = t.reshape(224, 224, 3).permute(2, 0, 1).float() / 255.0
    return (t - _MEAN) / _STD


@torch.no_grad()
def embed_paths(model, paths: list[Path], batch_size: int = 16) -> torch.Tensor:
    out = []
    for i in range(0, len(paths), batch_size):
        batch = []
        for p in paths[i:i + batch_size]:
            try:
                batch.append(pil_to_tensor(Image.open(p)))
            except Exception:
                batch.append(torch.zeros(3, 224, 224))
        feats = model(torch.stack(batch).to(DEVICE))
        out.append(feats.cpu())
        print(f"    embedded {min(i + batch_size, len(paths))}/{len(paths)}", end="\r")
    print()
    emb = torch.cat(out, dim=0)
    return emb / emb.norm(dim=1, keepdim=True).clamp_min(1e-12)


def load_or_embed(backbone_name: str, model, brand: str, season: str) -> torch.Tensor:
    """Cache .pt files in snapshots/embeddings/<backbone>_<brand>_<season>_n<N>.pt."""
    folder = REPO_ROOT / PATHS[brand][season]
    paths = sorted(folder.glob("*.jpg"))
    if not paths:
        raise FileNotFoundError(folder)
    key = f"{backbone_name}_{brand.replace(' ', '_').replace('&', 'and')}_{season}_n{len(paths)}.pt"
    cache = EMBED_DIR / key
    if cache.exists():
        return torch.load(cache, map_location="cpu")
    print(f"  [embed] {key}")
    emb = embed_paths(model, paths)
    torch.save(emb, cache)
    return emb


def pca_t(X: torch.Tensor, k: int) -> torch.Tensor:
    """Torch low-rank PCA to k dims, projecting X onto its top-k principal axes."""
    _, _, V = torch.pca_lowrank(X.float(), q=k, niter=4)
    return X @ V


# ─────────────────────────────────────────────────────────────────────────────
# Ward + K-Means (pure torch)
# ─────────────────────────────────────────────────────────────────────────────

def ward_full_t(X: torch.Tensor) -> tuple[list, dict]:
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
        m = len(active)
        ids = torch.tensor(active, dtype=torch.long)
        acts = centroids[ids]
        szs = sizes[ids]
        diff = acts.unsqueeze(0) - acts.unsqueeze(1)
        sq_dist = (diff * diff).sum(-1)
        si, sj = szs.unsqueeze(1), szs.unsqueeze(0)
        ward = si * sj / (si + sj) * sq_dist
        ward.fill_diagonal_(float("inf"))
        flat_idx = ward.argmin().item()
        ii, jj = flat_idx // m, flat_idx % m
        ci, cj = active[ii], active[jj]
        ni, nj = sizes[ci].item(), sizes[cj].item()
        centroids[next_id] = (ni * centroids[ci] + nj * centroids[cj]) / (ni + nj)
        sizes[next_id] = ni + nj
        cluster_members[next_id] = cluster_members[ci] + cluster_members[cj]
        merge_history.append((ci, cj, next_id))
        active = [x for x in active if x != ci and x != cj] + [next_id]
        next_id += 1

    return merge_history, cluster_members


def cut_tree_t(merge_history: list, cluster_members: dict, n: int, k: int) -> torch.Tensor:
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


def kmeans_torch(X: torch.Tensor, k: int, n_iters: int = 60, n_init: int = 10) -> torch.Tensor:
    """K-Means via Lloyd's, with k-means++ init, best of n_init by inertia."""
    n = X.shape[0]
    k = min(k, n)
    best_lbl, best_inertia = None, float("inf")
    for _ in range(n_init):
        idx0 = torch.randint(n, (1,)).item()
        centers = [X[idx0]]
        for _ in range(1, k):
            stacked = torch.stack(centers, 0)
            d2 = torch.cdist(X, stacked).min(dim=1).values ** 2
            probs = d2 / d2.sum().clamp_min(1e-12)
            idx = torch.multinomial(probs, 1).item()
            centers.append(X[idx])
        centers = torch.stack(centers, 0)
        for _ in range(n_iters):
            D = torch.cdist(X, centers)
            lbl = D.argmin(1)
            new_centers = torch.stack([
                X[lbl == c].mean(0) if (lbl == c).any() else centers[c]
                for c in range(k)
            ])
            if torch.allclose(centers, new_centers, atol=1e-4):
                centers = new_centers
                break
            centers = new_centers
        D = torch.cdist(X, centers)
        lbl = D.argmin(1)
        inertia = sum(((X[lbl == c] - centers[c]) ** 2).sum().item() for c in range(k))
        if inertia < best_inertia:
            best_inertia, best_lbl = inertia, lbl.clone()
    return best_lbl


# ─────────────────────────────────────────────────────────────────────────────
# Metrics (pure torch)
# ─────────────────────────────────────────────────────────────────────────────

def silhouette_t(X: torch.Tensor, lbl: torch.Tensor) -> float:
    uniq = lbl.unique()
    if len(uniq) < 2:
        return -1.0
    D = torch.cdist(X, X)
    scores = []
    for i in range(len(X)):
        same = (lbl == lbl[i]).clone()
        same[i] = False
        if not same.any():
            scores.append(0.0)
            continue
        a = D[i, same].mean().item()
        b = min(D[i, lbl == kk].mean().item() for kk in uniq if kk != lbl[i])
        denom = max(a, b)
        scores.append((b - a) / denom if denom > 0 else 0.0)
    return float(sum(scores) / len(scores))


def davies_bouldin_t(X: torch.Tensor, lbl: torch.Tensor) -> float:
    uniq = lbl.unique().tolist()
    k = len(uniq)
    if k < 2:
        return float("nan")
    centroids = torch.stack([X[lbl == c].mean(0) for c in uniq])
    scatter = torch.tensor([
        (X[lbl == c] - centroids[i]).norm(dim=1).mean().item() for i, c in enumerate(uniq)
    ])
    D = torch.cdist(centroids, centroids)
    db = 0.0
    for i in range(k):
        ratios = [
            (scatter[i] + scatter[j]) / D[i, j].item()
            for j in range(k) if j != i and D[i, j].item() > 0
        ]
        db += max(ratios) if ratios else 0.0
    return float(db / k)


def calinski_harabasz_t(X: torch.Tensor, lbl: torch.Tensor) -> float:
    uniq = lbl.unique().tolist()
    k = len(uniq)
    n = X.shape[0]
    if k < 2 or k >= n:
        return float("nan")
    overall = X.mean(0)
    centroids = torch.stack([X[lbl == c].mean(0) for c in uniq])
    sizes = torch.tensor([float((lbl == c).sum().item()) for c in uniq])
    bcss = float((sizes * ((centroids - overall) ** 2).sum(1)).sum().item())
    wcss = 0.0
    for i, c in enumerate(uniq):
        diff = X[lbl == c] - centroids[i]
        wcss += float((diff * diff).sum().item())
    if wcss == 0:
        return float("inf")
    return (bcss / (k - 1)) / (wcss / (n - k))


def inertia_t(X: torch.Tensor, lbl: torch.Tensor) -> float:
    uniq = lbl.unique().tolist()
    total = 0.0
    for c in uniq:
        members = X[lbl == c]
        if members.shape[0] == 0:
            continue
        centroid = members.mean(0)
        total += float(((members - centroid) ** 2).sum().item())
    return total


def adjusted_rand_index(a: torch.Tensor, b: torch.Tensor) -> float:
    """Hubert-Arabie ARI via the contingency-table formulation."""
    a_arr, b_arr = a.tolist(), b.tolist()
    pairs = list(zip(a_arr, b_arr))
    contingency = Counter(pairs)
    a_marg = Counter(a_arr)
    b_marg = Counter(b_arr)
    n = len(a_arr)

    def comb2(x): return x * (x - 1) // 2
    sum_ij = sum(comb2(v) for v in contingency.values())
    sum_a  = sum(comb2(v) for v in a_marg.values())
    sum_b  = sum(comb2(v) for v in b_marg.values())
    expected = sum_a * sum_b / comb2(n) if comb2(n) > 0 else 0.0
    max_index = (sum_a + sum_b) / 2.0
    if max_index == expected:
        return 1.0
    return (sum_ij - expected) / (max_index - expected)


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark loop
# ─────────────────────────────────────────────────────────────────────────────

def sweep(X: torch.Tensor, algo: str, k_min: int, k_max: int) -> dict:
    """Run one algorithm across k ∈ [k_min, k_max], return metric series + best k."""
    n = X.shape[0]
    series: dict[int, dict] = {}

    if algo == "ward":
        merge_history, cluster_members = ward_full_t(X)
        for k in range(k_min, min(k_max, n - 1) + 1):
            lbl = cut_tree_t(merge_history, cluster_members, n, k)
            series[k] = labels_to_metrics(X, lbl)
    elif algo == "kmeans":
        for k in range(k_min, min(k_max, n - 1) + 1):
            lbl = kmeans_torch(X, k)
            series[k] = labels_to_metrics(X, lbl)
    else:
        raise ValueError(algo)

    best_k = max(series, key=lambda k: series[k]["silhouette"])
    return {"by_k": series, "best_k_by_silhouette": best_k, "best_metrics": series[best_k]}


def labels_to_metrics(X: torch.Tensor, lbl: torch.Tensor) -> dict:
    return {
        "silhouette":         round(silhouette_t(X, lbl), 4),
        "davies_bouldin":     round(davies_bouldin_t(X, lbl), 4),
        "calinski_harabasz":  round(calinski_harabasz_t(X, lbl), 2),
        "inertia":            round(inertia_t(X, lbl), 4),
        "labels":             lbl.tolist(),
        "cluster_sizes":      sorted(Counter(lbl.tolist()).values(), reverse=True),
    }


def run_one(brand: str, mode: str, backbone_name: str, model, prep: str, algo: str) -> dict:
    """Embed (cached) → prep → algo sweep → metrics."""
    if mode == "joint":
        emb_f = load_or_embed(backbone_name, model, brand, "F25")
        emb_s = load_or_embed(backbone_name, model, brand, "S26")
        emb = torch.cat([emb_f, emb_s], dim=0)
    elif mode == "F25":
        emb = load_or_embed(backbone_name, model, brand, "F25")
    elif mode == "S26":
        emb = load_or_embed(backbone_name, model, brand, "S26")
    else:
        raise ValueError(mode)

    if prep == "pca50":
        emb = pca_t(emb, PCA_DIMS)

    return sweep(emb, algo, K_MIN, K_MAX)


def main():
    print(f"[device] {DEVICE}")
    models: dict[str, torch.nn.Module] = {}
    for backbone_name, model_id in BACKBONES.items():
        print(f"[load] {backbone_name} → {model_id}")
        m = timm.create_model(model_id, pretrained=True, num_classes=0).to(DEVICE)
        m.eval()
        models[backbone_name] = m

    brands = list(PATHS.keys())
    modes = ["joint", "F25", "S26"]
    preps = ["raw", "pca50"]
    algos = ["ward", "kmeans"]

    results: dict = {
        "config": {
            "backbones": BACKBONES,
            "preps": preps,
            "algos": algos,
            "k_min": K_MIN, "k_max": K_MAX, "pca_dims": PCA_DIMS,
        },
        "results": {},
    }

    rows_for_table: list[dict] = []

    for brand in brands:
        for mode in modes:
            print(f"\n=== {brand} · {mode} ===")
            results["results"].setdefault(brand, {}).setdefault(mode, {})
            for backbone_name in BACKBONES:
                for prep in preps:
                    cell = {}
                    for algo in algos:
                        t0 = time.time()
                        cell[algo] = run_one(brand, mode, backbone_name, models[backbone_name], prep, algo)
                        cell[algo]["elapsed_s"] = round(time.time() - t0, 2)
                        best = cell[algo]["best_metrics"]
                        bk = cell[algo]["best_k_by_silhouette"]
                        print(f"  {backbone_name:4s} {prep:5s} {algo:6s} "
                              f"best k={bk:2d}  silh={best['silhouette']:+.3f}  "
                              f"DB={best['davies_bouldin']:.3f}  CH={best['calinski_harabasz']:.1f}  "
                              f"inertia={best['inertia']:.1f}  ({cell[algo]['elapsed_s']:.1f}s)")
                        rows_for_table.append({
                            "brand": brand, "mode": mode,
                            "backbone": backbone_name, "prep": prep, "algo": algo,
                            "best_k": bk,
                            "silhouette": best["silhouette"],
                            "davies_bouldin": best["davies_bouldin"],
                            "calinski_harabasz": best["calinski_harabasz"],
                            "inertia": best["inertia"],
                            "cluster_sizes": best["cluster_sizes"],
                        })

                    # ARI between Ward and K-Means at the chosen best k of Ward
                    bk_ward = cell["ward"]["best_k_by_silhouette"]
                    lbl_ward = torch.tensor(cell["ward"]["by_k"][bk_ward]["labels"])
                    lbl_km   = torch.tensor(cell["kmeans"]["by_k"][bk_ward]["labels"])
                    ari = adjusted_rand_index(lbl_ward, lbl_km)
                    cell["ari_ward_vs_kmeans_at_ward_best_k"] = round(ari, 4)
                    print(f"  {backbone_name:4s} {prep:5s} ARI(ward,kmeans @ k={bk_ward}) = {ari:+.3f}")
                    results["results"][brand][mode].setdefault(backbone_name, {})[prep] = cell

    # Pick the per-(brand, mode) winner — highest silhouette across the grid
    winners = {}
    for brand in brands:
        winners[brand] = {}
        for mode in modes:
            best_row = max(
                (r for r in rows_for_table if r["brand"] == brand and r["mode"] == mode),
                key=lambda r: r["silhouette"],
            )
            winners[brand][mode] = best_row
    results["winners"] = winners

    OUT_PATH.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n[done] wrote {OUT_PATH}")

    # Winners table
    print("\n" + "=" * 100)
    print(f"{'BRAND':18s} {'MODE':6s} {'WINNER (backbone/prep/algo)':32s} {'k':>3s} {'silh':>7s} {'DB↓':>6s} {'CH↑':>8s}")
    print("=" * 100)
    for brand, modes_w in winners.items():
        for mode, w in modes_w.items():
            cfg = f"{w['backbone']}/{w['prep']}/{w['algo']}"
            print(f"{brand:18s} {mode:6s} {cfg:32s} {w['best_k']:>3d} {w['silhouette']:>+7.3f} {w['davies_bouldin']:>6.3f} {w['calinski_harabasz']:>8.2f}")


if __name__ == "__main__":
    main()
