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

import json
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).parent
STYLES_FILE   = REPO_ROOT / "snapshots" / "styles" / "style_assignments.json"
PALETTES_F25  = REPO_ROOT / "palettes_all_brands_v2.json"
PALETTES_S26  = REPO_ROOT / "palettes_S26.json"
OUT_FILE      = REPO_ROOT / "snapshots" / "styles" / "style_clusters.json"

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
    palettes = load_palettes(brand, season)
    url_map  = build_image_url_map(brand, season)
    products = assignments[brand][season]

    # Per-style: list of {filename, confidence}.  Confidence is FashionCLIP's
    # top-1 softmax probability, so the most-confident member of each cluster
    # is the most "textbook" example of that silhouette — the right hero
    # image. For "experimental" we flip and pick the LEAST confident, because
    # those are the most editorially interesting outliers.
    by_style: dict[str, list[dict]] = {}
    for prod in products:
        by_style.setdefault(prod["style"], []).append(prod)

    n_total = len(products)
    clusters = []
    for style, prods in by_style.items():
        experimental = style == "experimental"
        sorted_prods = sorted(
            prods,
            key=lambda p: p["confidence"],
            reverse=not experimental,
        )
        hero = sorted_prods[0]
        product_rows = [
            {
                "filename":   p["filename"],
                "url":        url_map.get(p["filename"], ""),
                "confidence": p["confidence"],
            }
            for p in sorted_prods
        ]
        clusters.append({
            "style":           style,
            "count":           len(prods),
            "share_pct":       round(len(prods) / n_total * 100, 1),
            "hero_filename":   hero["filename"],
            "hero_url":        url_map.get(hero["filename"], ""),
            "hero_confidence": hero["confidence"],
            "products":        product_rows,
            "colors":          aggregate_cluster_colors([p["filename"] for p in prods], palettes),
            "experimental":    experimental,
        })
    # Sort by count desc, but pin "experimental" to the end so the editorial
    # reading is "main styles first, then weird ones".
    clusters.sort(key=lambda c: (c["experimental"], -c["count"]))
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
