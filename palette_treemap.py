#!/usr/bin/env python3
"""
palette_treemap.py — build the per-(brand × season) colour-bucket
aggregate the frontend uses to render its palette treemap.

For each (brand, season) we collapse every product's palette down to
one row per name_bucket (tortoise / black / orange / etc.), counting
how many products in the catalogue contain that colour bucket. The
representative hex is the most frequent exact hex within the bucket.
Each row also carries the list of products that present the colour,
so a click on a treemap cell can open a modal grid of thumbnails.

Output schema (snapshots/palette_treemap.json):

  {
    "Bottega Veneta": {
      "F25": {
        "n_total": 123,
        "is_new_season": false,
        "colours": [
          {
            "name": "tortoise",
            "hex": "#8c6d5b",
            "count": 89,
            "share_pct": 72.4,
            "delta_vs_other_season": null,   # set on the S26 entry
            "products": [
              {"filename": "Bottega_1.jpg",
               "url": "/brands/bottega_veneta/0/Bottega_1.jpg"},
              ...
            ]
          },
          ...
        ]
      },
      "S26": {
        ...
        "colours": [
          { ..., "delta_vs_other_season": +12 },   # 12 more products with
                                                    # tortoise vs F25
          ...
        ]
      }
    },
    "Dolce & Gabbana": {...}
  }

Run with the base conda python (no transformers needed):
    /opt/anaconda3/bin/python palette_treemap.py
"""

import hashlib
import json
from collections import Counter
from pathlib import Path

REPO_ROOT       = Path(__file__).parent
PALETTES_F25    = REPO_ROOT / "palettes_all_brands_v2.json"
PALETTES_S26    = REPO_ROOT / "palettes_S26.json"
OUT_FILE        = REPO_ROOT / "snapshots" / "palette_treemap.json"
FRONTEND_PUBLIC = Path("/Users/benja/conan-insight-hub/public")

# Same threshold as style_clusters_with_colors — only count a colour as
# "present" in a product if it covers at least this fraction of the image.
MIN_COVERAGE    = 0.05

SOURCES = [
    ("Bottega Veneta", "F25", "images_bottega"),
    ("Bottega Veneta", "S26", "snapshots/S26/raw/bottega"),
    ("Dolce & Gabbana", "F25", "images_D&G"),
    ("Dolce & Gabbana", "S26", "snapshots/S26/raw/dg"),
]


def build_duplicate_set(folder: Path) -> set[str]:
    """Mirror the dedup rule used in style_clusters_with_colors so counts
    here match what the user sees in every other section.
    """
    seen: dict[str, str] = {}
    drop: set[str] = set()
    for p in sorted(folder.glob("*.jpg")):
        h = hashlib.md5(p.read_bytes()).hexdigest()
        if h in seen:
            drop.add(p.name)
        else:
            seen[h] = p.name
    return drop


def build_image_url_map(brand: str, season: str) -> dict[str, str]:
    if season == "S26":
        brand_dir = "bottega" if "Bottega" in brand else "dg"
        folder = FRONTEND_PUBLIC / "snapshots" / "S26" / brand_dir
        return {p.name: f"/snapshots/S26/{brand_dir}/{p.name}" for p in folder.glob("*.jpg")}
    brand_dir = "bottega_veneta" if "Bottega" in brand else "dolce_and_gabbana"
    root = FRONTEND_PUBLIC / "brands" / brand_dir
    return {p.name: "/" + str(p.relative_to(FRONTEND_PUBLIC)).replace("\\", "/")
            for p in root.rglob("*.jpg")}


def load_palettes(brand: str, season: str) -> dict:
    fpath = PALETTES_F25 if season == "F25" else PALETTES_S26
    raw = json.load(open(fpath))
    if brand not in raw:
        cands = [k for k in raw if brand.split()[0] in k]
        if not cands:
            return {}
        brand = cands[0]
    return raw[brand]


def build_brand_season(brand: str, season: str, folder_rel: str,
                       url_map: dict[str, str]) -> dict:
    palettes = load_palettes(brand, season)
    drop_set = build_duplicate_set(REPO_ROOT / folder_rel)

    bucket_products: dict[str, list[str]] = {}
    bucket_hexes:    dict[str, Counter[str]] = {}

    n_total = 0
    for fname, prod in palettes.items():
        if fname in drop_set:
            continue
        n_total += 1
        seen_buckets: set[str] = set()
        for c in prod.get("palette", []):
            if c.get("coverage", 0) < MIN_COVERAGE:
                continue
            name = c.get("name_bucket", "unknown")
            if name in seen_buckets:
                continue            # one vote per product per bucket
            seen_buckets.add(name)
            bucket_products.setdefault(name, []).append(fname)
            bucket_hexes.setdefault(name, Counter())[c["hex"]] += 1

    colours = []
    for name, fnames in bucket_products.items():
        rep_hex = bucket_hexes[name].most_common(1)[0][0]
        products = [{"filename": f, "url": url_map.get(f, "")} for f in sorted(fnames)]
        colours.append({
            "name":     name,
            "hex":      rep_hex,
            "count":    len(fnames),
            "share_pct": round(len(fnames) / n_total * 100, 1) if n_total else 0,
            "products": products,
        })
    colours.sort(key=lambda c: -c["count"])
    return {"n_total": n_total, "colours": colours}


def main():
    # First pass: build raw per-(brand, season) blocks
    out: dict = {}
    for brand, season, folder_rel in SOURCES:
        url_map = build_image_url_map(brand, season)
        block   = build_brand_season(brand, season, folder_rel, url_map)
        out.setdefault(brand, {})[season] = block

    # Second pass: stamp delta_vs_other_season on S26
    for brand, blocks in out.items():
        if "F25" not in blocks or "S26" not in blocks:
            continue
        f25 = {c["name"]: c["count"] for c in blocks["F25"]["colours"]}
        s26_counts = {c["name"]: c["count"] for c in blocks["S26"]["colours"]}
        for c in blocks["S26"]["colours"]:
            c["delta_vs_other_season"] = c["count"] - f25.get(c["name"], 0)
        # Also stamp on F25 for symmetry (S26 reference)
        for c in blocks["F25"]["colours"]:
            c["delta_vs_other_season"] = s26_counts.get(c["name"], 0) - c["count"]

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(out, indent=2))
    size_kb = OUT_FILE.stat().st_size // 1024
    for brand, blocks in out.items():
        for season, block in blocks.items():
            top = ", ".join(f"{c['name']}={c['count']}" for c in block["colours"][:5])
            print(f"  [{brand:18s} {season}] n_total={block['n_total']:3d}  "
                  f"{len(block['colours'])} buckets  top: {top}")
    print(f"\n[done] wrote {OUT_FILE.relative_to(REPO_ROOT)} ({size_kb} KB)")


if __name__ == "__main__":
    main()
