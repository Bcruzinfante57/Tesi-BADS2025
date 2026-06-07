#!/usr/bin/env python3
"""
Extend the silhouette sweep for the winning config (MAE+PCA(50)+Ward)
out to k=20, per (brand, mode), to verify there's no second peak hiding
above the k=12 cap we used in the main benchmark.
"""

import json
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).parent
EMBED_DIR = REPO_ROOT / "snapshots" / "embeddings"

BRANDS = ["Bottega Veneta", "Dolce & Gabbana"]
MODES = ["joint", "F25", "S26"]
EMBED_FILES = {
    "Bottega Veneta": {
        "F25": "MAE_Bottega_Veneta_F25_n123.pt",
        "S26": "MAE_Bottega_Veneta_S26_n162.pt",
    },
    "Dolce & Gabbana": {
        "F25": "MAE_Dolce_and_Gabbana_F25_n161.pt",
        "S26": "MAE_Dolce_and_Gabbana_S26_n114.pt",
    },
}
K_MIN, K_MAX = 3, 20
PCA_DIMS = 50


def pca_t(X: torch.Tensor, k: int) -> torch.Tensor:
    _, _, V = torch.pca_lowrank(X.float(), q=k, niter=4)
    return X @ V


def ward_full_t(X):
    n = len(X)
    max_c = 2 * n
    centroids = torch.zeros(max_c, X.shape[1])
    centroids[:n] = X.clone()
    sizes = torch.zeros(max_c)
    sizes[:n] = 1.0
    cluster_members = {i: [i] for i in range(n)}
    active = list(range(n))
    next_id = n
    merge_history = []
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


def cut_tree_t(merge_history, cluster_members, n, k):
    active = set(range(n))
    for ci, cj, new_id in merge_history[: n - k]:
        active.discard(ci); active.discard(cj); active.add(new_id)
    out = torch.zeros(n, dtype=torch.long)
    for label, cid in enumerate(sorted(active)):
        for member in cluster_members[cid]:
            out[member] = label
    return out


def silhouette_t(X, lbl):
    uniq = lbl.unique()
    if len(uniq) < 2:
        return -1.0
    D = torch.cdist(X, X)
    scores = []
    for i in range(len(X)):
        same = (lbl == lbl[i]).clone()
        same[i] = False
        if not same.any():
            scores.append(0.0); continue
        a = D[i, same].mean().item()
        b = min(D[i, lbl == kk].mean().item() for kk in uniq if kk != lbl[i])
        denom = max(a, b)
        scores.append((b - a) / denom if denom > 0 else 0.0)
    return float(sum(scores) / len(scores))


def cluster_sizes(lbl):
    from collections import Counter
    return sorted(Counter(lbl.tolist()).values(), reverse=True)


def run_one(brand, mode):
    if mode == "joint":
        ef = torch.load(EMBED_DIR / EMBED_FILES[brand]["F25"], map_location="cpu")
        es = torch.load(EMBED_DIR / EMBED_FILES[brand]["S26"], map_location="cpu")
        emb = torch.cat([ef, es], dim=0)
    else:
        emb = torch.load(EMBED_DIR / EMBED_FILES[brand][mode], map_location="cpu")
    emb = pca_t(emb, PCA_DIMS)
    n = emb.shape[0]
    merge_history, cluster_members = ward_full_t(emb)
    out = {}
    for k in range(K_MIN, min(K_MAX, n - 1) + 1):
        lbl = cut_tree_t(merge_history, cluster_members, n, k)
        out[k] = {"silhouette": round(silhouette_t(emb, lbl), 4),
                  "sizes": cluster_sizes(lbl)}
    return out


def main():
    print(f"Extended silhouette sweep — MAE + PCA(50) + Ward, k ∈ [{K_MIN}, {K_MAX}]\n")
    results = {}
    for brand in BRANDS:
        results[brand] = {}
        for mode in MODES:
            print(f"=== {brand} · {mode} ===")
            t0 = time.time()
            sweep = run_one(brand, mode)
            best_k = max(sweep, key=lambda k: sweep[k]["silhouette"])
            print(f"  done in {time.time()-t0:.1f}s, best k by silhouette = {best_k}")
            for k, m in sweep.items():
                marker = " ★" if k == best_k else ""
                print(f"    k={k:>2d}  silh={m['silhouette']:+.4f}  sizes={m['sizes']}{marker}")
            print()
            results[brand][mode] = {"sweep": sweep, "best_k": best_k}

    (REPO_ROOT / "snapshots" / "k_sweep_extended.json").write_text(json.dumps(results, indent=2))
    print("Saved snapshots/k_sweep_extended.json")


if __name__ == "__main__":
    main()
