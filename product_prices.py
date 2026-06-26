#!/usr/bin/env python3
"""
product_prices.py — flat manifest of every product's price.

Output (snapshots/product_prices.json) is a single dict mapping every
product filename (e.g. "Bottega_45.jpg") to its EUR price as float.
The frontend imports this so every place that shows a product
thumbnail can attach the price next to it.

Data source: the per-brand scraped CSVs in images_<brand>/*.csv.
Re-uses the parse_price helper from ticket_analysis.py so price-string
normalisation stays consistent (€345, "1.250,00€", "450 €" all parse
to 450.0).

Run with the base conda python:
    /opt/anaconda3/bin/python product_prices.py
"""

import csv
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).parent
OUT_FILE  = REPO_ROOT / "snapshots" / "product_prices.json"

BRAND_CSVS = {
    "Bottega Veneta":   "images_bottega/bottega_products.csv",
    "Cartier":          "images_Cartier/Cartier_products.csv",
    "Dolce & Gabbana":  "images_D&G/dolcegabbana_products.csv",
    "Fendi":            "images_Fendi/Fendi_products.csv",
    "Prada":            "images_Prada/prada_products.csv",
    "YSL":              "images_ysl/ysl_products.csv",
}

PRICE_RE = re.compile(r"([0-9][0-9.,]*)")


def parse_price(raw: str) -> float | None:
    if not raw:
        return None
    m = PRICE_RE.search(raw)
    if not m:
        return None
    s = m.group(1).replace(".", "").replace(",", ".")
    try:
        v = float(s)
    except ValueError:
        return None
    return v if v > 0 else None


def main() -> None:
    out: dict[str, float] = {}
    for brand, csv_rel in BRAND_CSVS.items():
        csv_path = REPO_ROOT / csv_rel
        added = 0
        with csv_path.open(encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = (
                    row.get("Product Name")
                    or row.get("Product_ID")
                    or row.get("Name")
                    or ""
                ).strip()
                price = parse_price(row.get("Price", ""))
                if not name or price is None:
                    continue
                fname = f"{name}.jpg"
                out[fname] = round(price, 0)
                added += 1
        print(f"  [{brand:18s}] +{added:>3d} products")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(out, indent=2))
    size_kb = OUT_FILE.stat().st_size // 1024
    print(f"\n[done] wrote {OUT_FILE.relative_to(REPO_ROOT)} "
          f"({size_kb} KB, {len(out)} products total)")


if __name__ == "__main__":
    main()
