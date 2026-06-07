#!/usr/bin/env python3
"""
joint_cluster_dino.py — re-clusters Bottega & D&G with DINO embeddings.

For each brand we produce THREE clusterings:
  • joint:   Ward Agglomerative on F25 ∪ S26 — clusters have mixed-season
             membership, season composition recorded per cluster
  • F25:     Ward on F25 products only
  • S26:     Ward on S26 products only

Why this exists
───────────────
The thesis ran Ward on ViT-MAE features. We established this week that
MAE's CLS token gives a near-degenerate feature space for our sunglasses
domain (cos sim ≥ 0.995 across all pairs). Ward on that collapses to
many micro-clusters formed by noise rather than visual identity. DINO
(self-distillation, instance discrimination pretext) gives a real cos
sim distribution and therefore meaningful clusters.

Output: snapshots/cluster_data_dino.json
  {
    "backbone": "vit_base_patch16_224.dino",
    "brands": {
      "<brand>": {
        "joint": { n_products, n_clusters, silhouette_score, clusters: [...] },
        "F25":   { same shape },
        "S26":   { same shape }
      }
    }
  }
"""

import json
import time
from collections import Counter
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).parent
EMBED_DIR = REPO_ROOT / "snapshots" / "embeddings"
OUT_PATH = REPO_ROOT / "snapshots" / "cluster_data_dino.json"

BRANDS = {
    "Bottega Veneta": {
        "f25_pt_dino": "DINO_Bottega_Veneta_F25_n123.pt",
        "s26_pt_dino": "DINO_Bottega_Veneta_S26_n162.pt",
        "f25_pt_mae":  "MAE_Bottega_Veneta_F25_n123.pt",
        "s26_pt_mae":  "MAE_Bottega_Veneta_S26_n162.pt",
        # Legacy filenames (the original joint_cluster_dino run wrote here without the prefix)
        "f25_pt_legacy_dino": "Bottega_Veneta_F25_n123.pt",
        "s26_pt_legacy_dino": "Bottega_Veneta_S26_n162.pt",
        "slug": "bottega",
    },
    "Dolce & Gabbana": {
        "f25_pt_dino": "DINO_Dolce_and_Gabbana_F25_n161.pt",
        "s26_pt_dino": "DINO_Dolce_and_Gabbana_S26_n114.pt",
        "f25_pt_mae":  "MAE_Dolce_and_Gabbana_F25_n161.pt",
        "s26_pt_mae":  "MAE_Dolce_and_Gabbana_S26_n114.pt",
        "f25_pt_legacy_dino": "Dolce_and_Gabbana_F25_n161.pt",
        "s26_pt_legacy_dino": "Dolce_and_Gabbana_S26_n114.pt",
        "slug": "dg",
    },
}

BACKBONES = ["DINO", "MAE"]

K_MIN = 3
K_MAX = 20  # widened to match the thesis pipeline's range (was 12, then 20)
PCA_DIMS = 50
THESIS_RULE_THRESHOLD = 0.95  # follows Main pipeline final.py:553 (score / best_score > 0.95)


def thesis_rule_select_k(silhouettes_by_k: dict[int, float]) -> int:
    """Replicates the thesis's find_optimal_k_agglomerative selection:

      1. Sort (k, silhouette) descending by silhouette.
      2. Iterate the sorted list; the FIRST entry whose k > 4 AND
         silhouette / best_silhouette > 0.95 wins.
      3. Default to the silhouette argmax if no entry qualifies.
      4. Return max(4, chosen_k) — minimum 4 clusters per the thesis.

    The intent is "pick the most granular k whose silhouette is still
    within 95% of the absolute best". For brands whose silhouette curve
    is roughly flat across k (Bottega, Cartier on DINO), this lets the
    rule reach k=10+. For brands with sharp silhouette peak at low k
    (D&G), it correctly stays at the peak.
    """
    scores_list = sorted(silhouettes_by_k.items(), key=lambda kv: -kv[1])
    if not scores_list:
        return 4
    best_score = scores_list[0][1]
    chosen = scores_list[0][0]
    for k_val, score in scores_list:
        if k_val > 4 and score / best_score > THESIS_RULE_THRESHOLD:
            chosen = k_val
            break
    return max(4, chosen)


def pca_t(X: torch.Tensor, k: int) -> torch.Tensor:
    """Torch low-rank PCA to k dims."""
    _, _, V = torch.pca_lowrank(X.float(), q=k, niter=4)
    return X @ V

BUNDLE_V2 = json.loads((REPO_ROOT / "snapshots/cluster_data_v2.json").read_text())


# ─────────────────────────────────────────────────────────────────────────────
# Ward + silhouette (pure torch)
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
        a = D[i, same].mean().item()
        b = min(D[i, lbl == kk].mean().item() for kk in unique if kk != lbl[i])
        denom = max(a, b)
        scores.append((b - a) / denom if denom > 0 else 0.0)
    return float(sum(scores) / len(scores))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_brand_products(brand: str, backbone: str) -> tuple[torch.Tensor, list[dict]]:
    """Concatenate F25 + S26 embeddings + product metadata for one brand × backbone."""
    cfg = BRANDS[brand]
    if backbone == "DINO":
        f25_path = EMBED_DIR / cfg["f25_pt_dino"]
        s26_path = EMBED_DIR / cfg["s26_pt_dino"]
        # Fall back to the legacy DINO filename (no backbone prefix) if the
        # benchmark hasn't been re-run yet
        if not f25_path.exists():
            f25_path = EMBED_DIR / cfg["f25_pt_legacy_dino"]
        if not s26_path.exists():
            s26_path = EMBED_DIR / cfg["s26_pt_legacy_dino"]
    elif backbone == "MAE":
        f25_path = EMBED_DIR / cfg["f25_pt_mae"]
        s26_path = EMBED_DIR / cfg["s26_pt_mae"]
    else:
        raise ValueError(backbone)
    f25_emb = torch.load(f25_path, map_location="cpu")
    s26_emb = torch.load(s26_path, map_location="cpu")

    f25_prods = BUNDLE_V2["brands"][brand]["snapshots"]["F25"]["products"]
    s26_prods = BUNDLE_V2["brands"][brand]["snapshots"]["S26"]["products"]

    if len(f25_prods) != f25_emb.shape[0]:
        raise RuntimeError(f"{brand} F25 size mismatch: {len(f25_prods)} vs {f25_emb.shape[0]}")
    if len(s26_prods) != s26_emb.shape[0]:
        raise RuntimeError(f"{brand} S26 size mismatch: {len(s26_prods)} vs {s26_emb.shape[0]}")

    prods = [{**p, "season": "F25"} for p in f25_prods] + [{**p, "season": "S26"} for p in s26_prods]
    emb = torch.cat([f25_emb, s26_emb], dim=0)
    return emb, prods


def aggregate_colours(member_prods: list[dict], top_n: int = 3) -> list[str]:
    hex_counts: Counter = Counter()
    for p in member_prods:
        for col in (p.get("palette") or [])[:2]:
            hex_counts[col["hex"]] += 1
    return [hex for hex, _ in hex_counts.most_common(top_n)]


def price_summary(arr: list[float]) -> dict:
    if not arr:
        return {}
    return {
        "min": round(min(arr), 2),
        "max": round(max(arr), 2),
        "mean": round(sum(arr) / len(arr), 2),
        "count": len(arr),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Cluster one slice — joint / F25-only / S26-only
# ─────────────────────────────────────────────────────────────────────────────

def cluster_set(label: str, emb_raw: torch.Tensor, prods: list[dict], slug: str,
                k_min: int, k_max: int) -> dict:
    n = emb_raw.shape[0]
    if n < 4:
        print(f"  [{label}] only {n} products — skipping")
        return {"n_products": n, "n_clusters": 0, "silhouette_score": -1.0,
                "silhouettes_by_k": {}, "clusters": []}

    # PCA(50) preprocessing — matches the thesis pipeline. Without it,
    # silhouette is mechanically suppressed by curse-of-dimensionality on
    # 768-d unit vectors, and the thesis rule's 95% threshold catches fewer
    # k values (so we'd lose the granular clusters the editorial display needs).
    emb = pca_t(emb_raw, PCA_DIMS) if n > PCA_DIMS else emb_raw

    t0 = time.time()
    merge_history, cluster_members = ward_full_t(emb)
    ward_secs = time.time() - t0

    silhouettes: dict[int, float] = {}
    for k in range(k_min, min(k_max, n - 1) + 1):
        lbl = cut_tree_t(merge_history, cluster_members, n, k)
        s = silhouette_t(emb, lbl)
        silhouettes[k] = round(s, 4)

    # Thesis-style k selection — see thesis_rule_select_k()
    best_k = thesis_rule_select_k(silhouettes)
    best_silh = silhouettes[best_k]
    silh_argmax_k = max(silhouettes, key=lambda k: silhouettes[k])
    silh_argmax = silhouettes[silh_argmax_k]
    print(f"  [{label}] n={n} Ward {ward_secs:.1f}s  "
          f"argmax k={silh_argmax_k} silh={silh_argmax:.4f}  "
          f"thesis-rule k={best_k} silh={best_silh:.4f}")

    final_lbl = cut_tree_t(merge_history, cluster_members, n, best_k).tolist()
    season_to_dir = {"F25": f"snapshots/F25/{slug}", "S26": f"snapshots/S26/{slug}"}

    clusters_out: list[dict] = []
    for cluster_id in range(best_k):
        members_idx = [i for i, lab in enumerate(final_lbl) if lab == cluster_id]
        member_prods = [prods[i] for i in members_idx]
        member_prods.sort(key=lambda p: (p["season"], p["id"]))

        seasons = {"F25": 0, "S26": 0}
        prices_per_season: dict[str, list[float]] = {"F25": [], "S26": []}
        for p in member_prods:
            seasons[p["season"]] += 1
            prices_per_season[p["season"]].append(p["price_eur"])
        all_prices = prices_per_season["F25"] + prices_per_season["S26"]

        all_images = [f"/{season_to_dir[p['season']]}/{p['image']}" for p in member_prods]

        cluster_centroid = emb[members_idx].mean(dim=0)
        dists = (emb[members_idx] - cluster_centroid).norm(dim=1)
        order = dists.argsort()
        rep_images = [
            f"/{season_to_dir[member_prods[order[j].item()]['season']]}/{member_prods[order[j].item()]['image']}"
            for j in range(min(3, len(members_idx)))
        ]

        clusters_out.append({
            "id": cluster_id,
            "n_products": len(members_idx),
            "season_composition": seasons,
            "dominant_colors": aggregate_colours(member_prods),
            "price_stats": price_summary(all_prices),
            "price_stats_by_season": {
                "F25": price_summary(prices_per_season["F25"]),
                "S26": price_summary(prices_per_season["S26"]),
            },
            "images": rep_images,
            "all_images": all_images,
            "is_new_in_s26": seasons["F25"] == 0 and seasons["S26"] > 0,
            "is_discontinued_from_f25": seasons["F25"] > 0 and seasons["S26"] == 0,
        })

    clusters_out.sort(key=lambda c: -c["n_products"])
    for new_id, c in enumerate(clusters_out):
        c["id"] = new_id

    return {
        "n_products": n,
        "n_clusters": best_k,
        "silhouette_score": round(best_silh, 4),
        "silhouettes_by_k": silhouettes,
        "clusters": clusters_out,
    }


def cluster_brand(brand: str, backbone: str, k_min: int, k_max: int) -> dict:
    """Joint + F25-only + S26-only clusterings for one brand × backbone."""
    print(f"\n=== {brand} · {backbone} ===")
    emb, prods = load_brand_products(brand, backbone)
    slug = BRANDS[brand]["slug"]

    f25_idx = [i for i, p in enumerate(prods) if p["season"] == "F25"]
    s26_idx = [i for i, p in enumerate(prods) if p["season"] == "S26"]
    emb_f25 = emb[f25_idx]
    emb_s26 = emb[s26_idx]
    prods_f25 = [prods[i] for i in f25_idx]
    prods_s26 = [prods[i] for i in s26_idx]

    return {
        "joint": cluster_set(f"joint·{backbone}", emb, prods, slug, k_min, k_max),
        "F25":   cluster_set(f"F25·{backbone}",   emb_f25, prods_f25, slug, k_min, k_max),
        "S26":   cluster_set(f"S26·{backbone}",   emb_s26, prods_s26, slug, k_min, k_max),
    }


def main():
    # Output groups both backbones so the frontend can offer a sub-toggle.
    # schema_version=2 distinguishes from the old flat shape.
    out = {
        "schema_version": 2,
        "backbones": {},
    }
    for backbone in BACKBONES:
        out["backbones"][backbone] = {
            "model_id":
                "vit_base_patch16_224.dino" if backbone == "DINO" else
                "vit_base_patch16_224.mae",
            "prep": f"PCA({PCA_DIMS})",
            "algo": "Ward Agglomerative",
            "selection": f"thesis rule (>{int(THESIS_RULE_THRESHOLD*100)}% of best, k>4, floor 4)",
            "brands": {},
        }
        for brand in BRANDS.keys():
            out["backbones"][backbone]["brands"][brand] = cluster_brand(brand, backbone, K_MIN, K_MAX)

    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"\n[done] wrote {OUT_PATH}")

    print("\nSummary:")
    for backbone, info in out["backbones"].items():
        for brand, modes in info["brands"].items():
            for mode_label, mode in modes.items():
                n = mode.get("n_products", 0)
                k = mode.get("n_clusters", 0)
                silh = mode.get("silhouette_score", -1.0)
                print(f"  {backbone:4s}  {brand:18s} {mode_label:5s}  n={n:3d} k={k:2d} silhouette={silh:.4f}")


if __name__ == "__main__":
    main()
