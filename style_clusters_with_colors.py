#!/usr/bin/env python3
"""
style_clusters_with_colors.py — merge style assignments with per-product
color palettes to produce the JSON the frontend needs to render Style
Clusters with all-color dots ("pelotitas") under each cluster card.

Output schema:
{
  "Bottega Veneta": {
    "F25": {
      "n_total": 123,
      "clusters": [
        {
          "style": "butterfly",
          "count": 24,
          "share_pct": 19.5,
          "delta_vs_other_season": null,         # only for S26
          "products": ["Bottega_40.jpg", ...],   # filenames in this cluster
          "colors": [                            # ALL colors across cluster
                                                 # products, sorted by frequency
            {"hex": "#8c6d5b", "name": "tortoise", "count": 12, "share_pct": 25.0},
            {"hex": "#241e1a", "name": "black",    "count":  8, "share_pct": 16.7},
            ...
          ],
          "experimental": false
        },
        ...
      ]
    },
    "S26": {...}
  },
  "Dolce & Gabbana": {...}
}

The frontend reads this directly. The "colors" list under each cluster is
the set of unique color buckets across all products in that cluster —
useful for the dots row beneath each cluster name.

Run from this directory with the base python (no transformers needed):
    /opt/anaconda3/bin/python style_clusters_with_colors.py
"""

import hashlib
import json
from collections import Counter
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).parent
STYLES_FILE   = REPO_ROOT / "snapshots" / "styles" / "style_assignments.json"
PALETTES_F25  = REPO_ROOT / "palettes_all_brands_v2.json"
PALETTES_S26  = REPO_ROOT / "palettes_S26.json"
EMBED_DIR     = REPO_ROOT / "snapshots" / "embeddings"
OUT_FILE      = REPO_ROOT / "snapshots" / "styles" / "style_clusters.json"

# FashionCLIP embeddings, keyed by (brand, season). Used for the L2 sub-
# clustering pass inside each silhouette.
FCLIP_EMBED_FILES = {
    ("Bottega Veneta",  "F25"): "FashionCLIP_Bottega_Veneta_F25_n123.pt",
    ("Bottega Veneta",  "S26"): "FashionCLIP_Bottega_Veneta_S26_n162.pt",
    ("Dolce & Gabbana", "F25"): "FashionCLIP_Dolce_and_Gabbana_F25_n161.pt",
    ("Dolce & Gabbana", "S26"): "FashionCLIP_Dolce_and_Gabbana_S26_n114.pt",
}

# Silhouettes below this size are kept as a single flat group (no L2
# sub-clustering — too few products to split meaningfully).
SUB_CLUSTER_MIN = 6

# Where the frontend serves images from. Scanned at build time to translate
# a bare filename ("Bottega_107.jpg") into a real URL the browser can load
# (e.g. "/brands/bottega_veneta/7/Bottega_107.jpg"). F25 images are nested
# under the thesis cluster id; S26 are flat.
FRONTEND_PUBLIC = Path("/Users/benja/conan-insight-hub/public")

# Per-product palette covers ALL colors at >= this fraction of the image.
# Below this we consider it noise (e.g., shadows, background bleed).
MIN_COLOR_COVERAGE = 0.05


def load_palettes(brand_label: str, season: str) -> dict[str, list]:
    """Returns {filename: [(hex, name, coverage), ...]} for the given (brand, season)."""
    fpath = PALETTES_F25 if season == "F25" else PALETTES_S26
    raw = json.load(open(fpath))
    # The palettes JSON keys use "Dolce & Gabbana" / "Bottega Veneta" verbatim
    if brand_label not in raw:
        # Try alternative spellings
        candidates = [k for k in raw if brand_label.split()[0] in k]
        if not candidates:
            raise KeyError(f"{brand_label} not found in {fpath.name}; keys = {list(raw.keys())}")
        brand_label = candidates[0]
    out = {}
    for fname, prod in raw[brand_label].items():
        palette = prod.get("palette", [])
        kept = [(c["hex"], c.get("name_bucket", "unknown"), c["coverage"])
                for c in palette if c["coverage"] >= MIN_COLOR_COVERAGE]
        out[fname] = kept
    return out


def build_duplicate_map(brand: str, season: str) -> set[str]:
    """Returns a set of filenames that are byte-duplicates of an earlier file
    (in alphabetical order) within the same (brand, season). The scraper
    occasionally saved the same product photo under two SKU-style names —
    Bottega_161 ≡ Bottega_33 and Bottega_162 ≡ Bottega_42 in S26 — and
    those shouldn't repeat in the cluster's product grid.
    """
    if season == "S26":
        brand_dir = "snapshots/S26/raw/bottega" if "Bottega" in brand else "snapshots/S26/raw/dg"
        folder = Path("/Users/benja/Tesi-BADS2025") / brand_dir
    else:
        brand_dir = "images_bottega" if "Bottega" in brand else "images_D&G"
        folder = Path("/Users/benja/Tesi-BADS2025") / brand_dir
    seen_hashes: dict[str, str] = {}
    drop: set[str] = set()
    for p in sorted(folder.glob("*.jpg")):
        h = hashlib.md5(p.read_bytes()).hexdigest()
        if h in seen_hashes:
            drop.add(p.name)
        else:
            seen_hashes[h] = p.name
    return drop


def build_image_url_map(brand: str, season: str) -> dict[str, str]:
    """Returns {filename → public URL path} by scanning the frontend's public/ tree.

    Conventions inferred from existing layout:
      F25 Bottega → public/brands/bottega_veneta/<thesis-cluster-id>/<file>
      F25 D&G    → public/brands/dolce_and_gabbana/<thesis-cluster-id>/<file>
      S26 Bottega → public/snapshots/S26/bottega/<file>
      S26 D&G    → public/snapshots/S26/dg/<file>
    """
    if season == "S26":
        brand_dir = "bottega" if "Bottega" in brand else "dg"
        folder = FRONTEND_PUBLIC / "snapshots" / "S26" / brand_dir
        return {p.name: f"/snapshots/S26/{brand_dir}/{p.name}" for p in folder.glob("*.jpg")}
    # F25 — walk all cluster subfolders
    brand_dir = "bottega_veneta" if "Bottega" in brand else "dolce_and_gabbana"
    root = FRONTEND_PUBLIC / "brands" / brand_dir
    url_map = {}
    for p in root.rglob("*.jpg"):
        rel = p.relative_to(FRONTEND_PUBLIC)
        url_map[p.name] = "/" + str(rel).replace("\\", "/")
    return url_map


@torch.no_grad()
def kmeans_torch(X: torch.Tensor, k: int, n_iter: int = 30, seed: int = 42
                 ) -> tuple[torch.Tensor, torch.Tensor]:
    """Spherical KMeans on L2-normalised embeddings — assignment by cos sim,
    centroids re-normalised every iteration. Returns (labels, centroids).
    Deterministic for a given seed.
    """
    n, d = X.shape
    if k >= n:
        return torch.arange(n) % k, X[:k].clone()

    g = torch.Generator().manual_seed(seed)
    init_idx = torch.randperm(n, generator=g)[:k]
    centroids = X[init_idx].clone()

    labels = torch.full((n,), -1, dtype=torch.long)
    for _ in range(n_iter):
        sims = X @ centroids.T              # (n, k)
        new_labels = sims.argmax(dim=1)
        if torch.equal(new_labels, labels):
            break
        labels = new_labels
        for c in range(k):
            mask = labels == c
            if mask.sum() == 0:
                continue
            v = X[mask].mean(dim=0)
            centroids[c] = v / v.norm().clamp_min(1e-12)
    return labels, centroids


def pick_subk(n: int) -> int:
    """Per-silhouette k for the L2 sub-clustering pass. Capped at 4 so
    modal sections stay readable; off for clusters with fewer than
    SUB_CLUSTER_MIN products."""
    if n < SUB_CLUSTER_MIN: return 1
    if n < 13:              return 2
    if n < 19:              return 3
    return 4


def build_subclusters(cluster_products: list[dict],
                      img_emb_l2: torch.Tensor,
                      filename_to_idx: dict[str, int],
                      palettes: dict[str, list],
                      ) -> list[dict] | None:
    """For a single silhouette cluster, run spherical KMeans on its members'
    FashionCLIP embeddings and return a list of sub-cluster summaries:
    hero, count, top colours, members. Returns None when the cluster is
    too small or its members aren't all in the embedding index.
    """
    n = len(cluster_products)
    k = pick_subk(n)
    if k <= 1:
        return None

    # Map cluster products → embedding indices (skip any filename we don't
    # have an embedding for — shouldn't happen with dedup but be safe).
    keep_rows: list[tuple[int, dict]] = []
    for p in cluster_products:
        idx = filename_to_idx.get(p["filename"])
        if idx is not None:
            keep_rows.append((idx, p))
    if len(keep_rows) < SUB_CLUSTER_MIN:
        return None

    indices = torch.tensor([r[0] for r in keep_rows])
    members = [r[1] for r in keep_rows]
    sub_emb = img_emb_l2[indices]
    labels, centroids = kmeans_torch(sub_emb, k)

    sub_clusters: list[dict] = []
    for sub_id in range(k):
        mask = labels == sub_id
        if mask.sum() == 0:
            continue                                # very rare edge case
        sub_members = [members[i] for i, m in enumerate(mask.tolist()) if m]
        sub_member_emb = sub_emb[mask]
        # Hero = member with highest cos sim to its centroid.
        sims = sub_member_emb @ centroids[sub_id]
        hero_member_idx = int(sims.argmax().item())
        hero = sub_members[hero_member_idx]
        # Top colours within the sub-cluster
        sub_colors = aggregate_cluster_colors(
            [m["filename"] for m in sub_members], palettes,
        )
        sub_clusters.append({
            "id":              sub_id,
            "count":           len(sub_members),
            "hero_filename":   hero["filename"],
            "hero_url":        hero.get("url", ""),
            "hero_confidence": hero["confidence"],
            "products":        sub_members,
            "colors":          sub_colors,
            "color_summary":   " + ".join(c["name"] for c in sub_colors[:2]),
        })
    # Largest sub-cluster first, so the modal opens with the dominant
    # variant.
    sub_clusters.sort(key=lambda s: -s["count"])
    return sub_clusters


def aggregate_cluster_colors(products: list[str], palettes: dict[str, list]) -> list[dict]:
    """Return one row per unique (hex bucket) across cluster products, sorted
    by frequency. Two products with the same color bucket count as 2."""
    # Bucket by name_bucket so visually similar tortoise tones merge into
    # one dot. We pick the most common exact hex within each bucket as the
    # representative color.
    bucket_count: Counter[str] = Counter()
    bucket_hex_examples: dict[str, Counter[str]] = {}
    for fname in products:
        seen_buckets_in_this_product = set()
        for hex_, name, _cov in palettes.get(fname, []):
            if name in seen_buckets_in_this_product:
                continue
            seen_buckets_in_this_product.add(name)
            bucket_count[name] += 1
            bucket_hex_examples.setdefault(name, Counter())[hex_] += 1
    n_products = len(products) or 1
    rows = []
    for name, cnt in bucket_count.most_common():
        # representative hex = most-common exact hex within this bucket
        rep_hex = bucket_hex_examples[name].most_common(1)[0][0]
        rows.append({
            "hex":       rep_hex,
            "name":      name,
            "count":     cnt,
            "share_pct": round(cnt / n_products * 100, 1),
        })
    return rows


def build_clusters(assignments: dict, brand: str, season: str) -> list[dict]:
    palettes  = load_palettes(brand, season)
    url_map   = build_image_url_map(brand, season)
    drop_set  = build_duplicate_map(brand, season)
    # Drop byte-duplicate filenames from the product list before counting,
    # picking a hero, or aggregating colours.
    products  = [p for p in assignments[brand][season] if p["filename"] not in drop_set]

    # FashionCLIP embeddings + filename → row index, for the L2 sub-clustering
    # pass below. The .pt files are saved in the same alphabetical order as
    # sorted(folder.glob("*.jpg")), so we recreate that mapping here.
    emb_name = FCLIP_EMBED_FILES.get((brand, season))
    img_emb_l2: torch.Tensor | None = None
    filename_to_idx: dict[str, int] = {}
    if emb_name is not None:
        emb_path = EMBED_DIR / emb_name
        if emb_path.exists():
            raw = torch.load(emb_path, map_location="cpu")
            img_emb_l2 = raw / raw.norm(dim=1, keepdim=True).clamp_min(1e-12)
            folder = ("snapshots/S26/raw/bottega" if season == "S26" and "Bottega" in brand
                      else "snapshots/S26/raw/dg" if season == "S26"
                      else "images_bottega" if "Bottega" in brand
                      else "images_D&G")
            for i, p in enumerate(sorted((REPO_ROOT / folder).glob("*.jpg"))):
                filename_to_idx[p.name] = i

    # Hero pick rules:
    #   • Named silhouettes: highest top-1 softmax confidence — the most
    #     "textbook" example of the silhouette.
    #   • Signature bucket: highest intra-season rarity — the product
    #     least similar to anything else in the maison's own catalogue,
    #     i.e., the boldest editorial signature.
    by_style: dict[str, list[dict]] = {}
    for prod in products:
        by_style.setdefault(prod["style"], []).append(prod)

    n_total = len(products)
    clusters = []
    for style, prods in by_style.items():
        is_signature = style == "signature"
        if is_signature:
            sorted_prods = sorted(prods, key=lambda p: p.get("rarity", 0), reverse=True)
        else:
            sorted_prods = sorted(prods, key=lambda p: p["confidence"], reverse=True)
        hero = sorted_prods[0]
        product_rows = [
            {
                "filename":   p["filename"],
                "url":        url_map.get(p["filename"], ""),
                "confidence": p["confidence"],
                "rarity":     p.get("rarity", 0),
            }
            for p in sorted_prods
        ]
        # Sub-cluster non-signature silhouettes via spherical KMeans on the
        # FashionCLIP embeddings. The signature bucket is intentionally
        # left flat because its members already have nothing in common.
        sub_clusters = None
        if not is_signature and img_emb_l2 is not None and len(product_rows) >= SUB_CLUSTER_MIN:
            sub_clusters = build_subclusters(product_rows, img_emb_l2, filename_to_idx, palettes)

        clusters.append({
            "style":           style,
            "count":           len(prods),
            "share_pct":       round(len(prods) / n_total * 100, 1),
            "hero_filename":   hero["filename"],
            "hero_url":        url_map.get(hero["filename"], ""),
            "hero_confidence": hero["confidence"],
            "hero_rarity":     hero.get("rarity", 0),
            "products":        product_rows,
            "colors":          aggregate_cluster_colors([p["filename"] for p in prods], palettes),
            "signature":       is_signature,
            "sub_clusters":    sub_clusters,
        })
    # Sort by count desc; pin "signature" to the end so the editorial
    # reading is "main silhouettes first, then the maison's distinctive
    # pieces as a closing showcase".
    clusters.sort(key=lambda c: (c["signature"], -c["count"]))
    return clusters


def main():
    assignments = json.load(open(STYLES_FILE))

    out: dict = {}
    for brand in assignments:
        out[brand] = {}
        for season in assignments[brand]:
            try:
                clusters = build_clusters(assignments, brand, season)
            except KeyError as e:
                print(f"  [skip] {brand} {season}: {e}")
                continue
            n_total = sum(c["count"] for c in clusters)
            out[brand][season] = {
                "n_total":  n_total,
                "clusters": clusters,
            }
            preview = ", ".join(f"{c['style']}({c['count']})" for c in clusters[:5])
            print(f"  [{brand:18s} {season}] n={n_total:3d}  top: {preview}")

    # Compute S26 delta vs F25 per style (only for brands with both seasons)
    for brand in out:
        if "F25" not in out[brand] or "S26" not in out[brand]:
            continue
        f25_counts = {c["style"]: c["count"] for c in out[brand]["F25"]["clusters"]}
        for c in out[brand]["S26"]["clusters"]:
            c["delta_vs_F25"] = c["count"] - f25_counts.get(c["style"], 0)

    OUT_FILE.write_text(json.dumps(out, indent=2))
    size_kb = OUT_FILE.stat().st_size // 1024
    print(f"\n[done] wrote {OUT_FILE.relative_to(REPO_ROOT)} ({size_kb} KB)")


if __name__ == "__main__":
    main()
