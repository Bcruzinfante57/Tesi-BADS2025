#!/usr/bin/env python3
"""
bundle_v2.py — assemble cluster_data_v2.json from the phase-1 artifacts.

Reads:
  - palettes_all_brands_v2.json       (F25 palettes from earlier run)
  - palettes_S26.json                 (S26 palettes from --snapshot S26 run)
  - snapshots/persistence.json        (ViT cross-season matching)
  - snapshots/kde.json                (price KDE per brand × snapshot)
  - images_<brand>/<brand>_products.csv  (F25 prices, per row)
  - snapshots/S26/raw/<brand>/<brand>_products.csv (S26 prices, per row)

Emits:
  - cluster_data_v2.json (consumed by the frontend)

Schema follows the Fase 4 spec we documented in conan-insight-hub/README.md.
"""

import csv
import json
import re
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).parent
OUT_PATH = REPO_ROOT / "snapshots" / "cluster_data_v2.json"

SNAPSHOTS = ["F25", "S26"]
BRANDS = ["Bottega Veneta", "Dolce & Gabbana"]

PALETTE_PATHS = {
    "F25": REPO_ROOT / "palettes_all_brands_v2.json",
    "S26": REPO_ROOT / "palettes_S26.json",
}

CSV_PATHS = {
    "F25": {
        "Bottega Veneta":  REPO_ROOT / "images_bottega" / "bottega_products.csv",
        "Dolce & Gabbana": REPO_ROOT / "images_D&G" / "dolcegabbana_products.csv",
    },
    "S26": {
        "Bottega Veneta":  REPO_ROOT / "snapshots/S26/raw/bottega/bottega_products.csv",
        "Dolce & Gabbana": REPO_ROOT / "snapshots/S26/raw/dg/dolcegabbana_products.csv",
    },
}


def parse_price(s: str) -> float | None:
    """Return numeric price if `s` looks like a currency value, else None."""
    if "€" not in s and "$" not in s and "£" not in s:
        return None
    m = re.search(r"([0-9]+(?:[.,][0-9]+)?)", s)
    if not m:
        return None
    try:
        v = float(m.group(1).replace(",", "."))
        return v if v >= 50 else None
    except ValueError:
        return None


def load_products(csv_path: Path, image_dir: Path) -> list[dict]:
    """Read products from CSV, attach image filenames for joining with palettes."""
    if not csv_path.exists():
        return []
    products = []
    with open(csv_path) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if not row:
                continue
            name = row[0].strip()
            price = parse_price(row[-1])
            if price is None:
                continue
            # Map "Bottega_1" → "Bottega_1.jpg" if the file actually exists
            img_name = f"{name}.jpg"
            if not (image_dir / img_name).exists():
                continue
            entry = {"id": name, "image": img_name, "price_eur": price}
            # Optional pid column (S26 Bottega + D&G capture it)
            if len(row) >= 3 and row[1] and row[1] != row[0]:
                entry["external_pid"] = row[1].strip()
            products.append(entry)
    return products


def color_histogram(palette_entries: list[dict]) -> Counter:
    """Histogram of name buckets, weighted by coverage."""
    c = Counter()
    for entry in palette_entries:
        c[entry["name_bucket"]] += 1
    return c


def main():
    palettes = {snap: json.loads(PALETTE_PATHS[snap].read_text()) for snap in SNAPSHOTS}
    persistence = json.loads((REPO_ROOT / "snapshots/persistence.json").read_text())
    kde = json.loads((REPO_ROOT / "snapshots/kde.json").read_text())

    out: dict = {
        "schema_version": 2,
        "snapshots": SNAPSHOTS,
        "brands": {},
        "kde": {
            "log_x": kde["log_x"],
            "x_euros": kde["x_euros"],
            "brands": kde["brands"],
        },
    }

    for brand in BRANDS:
        out["brands"][brand] = {"snapshots": {}}

        # Per-snapshot data
        for snap in SNAPSHOTS:
            csv_path = CSV_PATHS[snap][brand]
            # Resolve image dir from CSV's parent
            image_dir = csv_path.parent
            products = load_products(csv_path, image_dir)
            brand_palettes = palettes[snap].get(brand, {})

            # Attach palette + color histogram per product
            for p in products:
                pinfo = brand_palettes.get(p["image"], {})
                p["palette"] = pinfo.get("palette", [])

            # Histogram = "how many PRODUCTS show this colour bucket at least once".
            # NOT "how many colour detections fell in this bucket" — that older
            # semantics over-counted (each product has 3-7 palette centroids and
            # a single product can have several shades of grey, etc), producing
            # bucket totals larger than n_products which is misleading.
            color_hist = Counter()
            for p in products:
                seen = set(col["name_bucket"] for col in p["palette"])
                for bucket in seen:
                    color_hist[bucket] += 1

            prices = [p["price_eur"] for p in products]
            out["brands"][brand]["snapshots"][snap] = {
                "n_products": len(products),
                "price_stats": {
                    "min": min(prices) if prices else None,
                    "max": max(prices) if prices else None,
                    "mean": round(sum(prices) / len(prices), 2) if prices else None,
                    "median": round(sorted(prices)[len(prices) // 2], 2) if prices else None,
                },
                "color_histogram": dict(color_hist.most_common()),
                "products": products,
            }

        # Cross-season matching (from persistence.json)
        if brand in persistence["brands"]:
            pmatch = persistence["brands"][brand]
            out["brands"][brand]["matching"] = {
                "threshold": pmatch["threshold"],
                "n_persisted": pmatch["n_persisted"],
                "n_new_in_s26": pmatch["n_new_in_s26"],
                "n_discontinued_from_f25": pmatch["n_discontinued_from_f25"],
                "persistence_rate_f25": pmatch["persistence_rate_f25"],
                "novelty_rate_s26": pmatch["novelty_rate_s26"],
                "cos_sim_distribution": pmatch["cos_sim_distribution"],
                "persisted_pairs": pmatch["persisted"],
                "new_s26_products": pmatch["new_in_s26"],
                "discontinued_f25_products": pmatch["discontinued_from_f25"],
            }

    OUT_PATH.write_text(json.dumps(out, indent=2))
    size_kb = OUT_PATH.stat().st_size / 1024
    print(f"[done] wrote {OUT_PATH}  ({size_kb:.1f} KB)")

    # Quick summary
    for brand in BRANDS:
        b = out["brands"][brand]
        print(f"\n=== {brand} ===")
        for snap in SNAPSHOTS:
            s = b["snapshots"][snap]
            ps = s["price_stats"]
            print(f"  {snap}: n={s['n_products']:3d}  "
                  f"€{ps['min']:.0f}-{ps['max']:.0f}  "
                  f"mean €{ps['mean']:.0f}  "
                  f"colors={sum(s['color_histogram'].values())}")
        if "matching" in b:
            m = b["matching"]
            print(f"  matching: persisted={m['n_persisted']}  "
                  f"new={m['n_new_in_s26']}  "
                  f"discontinued={m['n_discontinued_from_f25']}  "
                  f"(threshold={m['threshold']})")


if __name__ == "__main__":
    main()
