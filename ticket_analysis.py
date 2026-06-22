#!/usr/bin/env python3
"""
ticket_analysis.py — per-brand ticket distribution for the chat's rich
response card.

Each brand gets: global stats (mean/median/min/max/p25/p75/std), a KDE
curve over its price distribution, and 4 anchor products positioned at
percentiles {25, 50, 75, 95}. The anchors are the editorially-meaningful
"touch points" of the curve — the cheap-end signature, the median piece,
the upper-quartile flagship, and the catalogue outlier. Each anchor
includes the product's filename, URL and parent cluster id so the
frontend can render them as clickable thumbnails on top of the curve.

Output (snapshots/ticket_analysis.json) is consumed by the chat panel
when CONAN emits the [[ticket-analysis:BRAND]] token, replacing the
boring markdown price table the agent was producing before.

Run with the base conda python (just numpy + stdlib):
    /opt/anaconda3/bin/python ticket_analysis.py
"""

import csv
import json
import re
from pathlib import Path

import numpy as np

REPO_ROOT       = Path(__file__).parent
CLUSTER_FILE    = Path("/Users/benja/conan-insight-hub/public/cluster_data.json")
OUT_FILE        = REPO_ROOT / "snapshots" / "ticket_analysis.json"
FRONTEND_PUBLIC = Path("/Users/benja/conan-insight-hub/public")

# Per-brand CSV with rows: "Product Name,Price". The price column comes
# scraped as "450 €", with the euro sign and varying whitespace.
BRAND_CSVS = {
    "Bottega Veneta":   "images_bottega/bottega_products.csv",
    "Cartier":          "images_Cartier/Cartier_products.csv",
    "Dolce & Gabbana":  "images_D&G/dolcegabbana_products.csv",
    "Fendi":            "images_Fendi/Fendi_products.csv",
    "Prada":            "images_Prada/prada_products.csv",
    "YSL":              "images_ysl/ysl_products.csv",
}

# How the brand's image folder is named inside the frontend public/brands
# tree. Used to resolve anchor URLs.
BRAND_DIRS = {
    "Bottega Veneta":   "bottega_veneta",
    "Cartier":          "cartier",
    "Dolce & Gabbana":  "dolce_and_gabbana",
    "Fendi":            "fendi",
    "Prada":            "prada",
    "YSL":              "ysl",
}

# Anchor kinds — four statistical landmarks of the price distribution
# that the frontend KdeCard surfaces as a row of clickable thumbnails
# in the empty top-right space, with dashed lines connecting each to
# its actual position on the x-axis.
#
# Replaces the older percentile-based anchors (p10/p30/p50/p70/p99).
# The previous percentile scheme worked for skewed distributions like
# Cartier but produced overlapping thumbs in tight distributions
# (Fendi, Prada — where median ≈ p10 ≈ p70 ≈ p90). The four-stat
# scheme — min / median / mean / max — always renders four distinct
# touchpoints whose semantics every visitor immediately understands.
#
# For each kind we record the target value (the statistic) and the
# actual_price of the product nearest it.
ANCHOR_KINDS = ["min", "median", "mean", "max"]

# Gaussian KDE bandwidth uses Silverman's rule by default. Resolution
# of the output curve, in number of x-axis samples.
#
# Bumped 100 → 300 so the curve has enough density samples to render
# smoothly in the lower-price region when the frontend uses a log-scale
# x-axis. With 100 linear samples and a range like Cartier's €0–€19,927,
# only ~5 samples fall into the dense €450–€1,500 band where the catalogue
# actually lives — producing a visible "plateau" near the peak instead of
# a smooth bell. 300 samples give ~15 samples in that same band, which is
# enough for the bezier-smoothed path on the frontend to render a clean
# bell-shape.
KDE_RESOLUTION = 300


# ─────────────────────────────────────────────────────────────────────────────
# Price parsing
# ─────────────────────────────────────────────────────────────────────────────

PRICE_RE = re.compile(r"([0-9][0-9.,]*)")


def parse_price(raw: str) -> float | None:
    """'450 €' → 450.0   ·   '1.250,00€' → 1250.0   ·   'N/A' → None."""
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
    if v <= 0:
        return None
    return v


def load_brand_products(brand: str) -> list[tuple[str, float]]:
    """Return [(product_name, price), …] for the brand. The scrapers used
    a few different column conventions over time — "Product Name",
    "Product_ID", "Name" — so we try them all. Rows with no parseable
    price are dropped silently."""
    csv_path = REPO_ROOT / BRAND_CSVS[brand]
    out: list[tuple[str, float]] = []
    with csv_path.open(encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = (
                row.get("Product Name")
                or row.get("Product_ID")
                or row.get("Name")
                or ""
            )
            price = parse_price(row.get("Price", ""))
            if not name or price is None:
                continue
            out.append((name.strip(), price))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Stats + KDE
# ─────────────────────────────────────────────────────────────────────────────


def percentile(arr: np.ndarray, p: float) -> float:
    return float(np.percentile(arr, p))


def compute_stats(prices: np.ndarray) -> dict:
    return {
        "mean":   round(float(prices.mean()), 0),
        "median": round(percentile(prices, 50), 0),
        "min":    round(float(prices.min()), 0),
        "max":    round(float(prices.max()), 0),
        "p25":    round(percentile(prices, 25), 0),
        "p75":    round(percentile(prices, 75), 0),
        "std":    round(float(prices.std()), 0),
        "n":      int(prices.size),
    }


def kde_curve(prices: np.ndarray) -> dict:
    """Log-binned smoothed histogram of catalogue prices.

    Replaces the previous gaussian KDE. The user's mental model of the
    chart is "show me where products actually cluster" — a count-based
    histogram — not "estimate the probability of a product at any
    price" — which is what a true KDE does. A gaussian KDE smears
    density beyond the data extrema (the gaussian centered at €450 has
    a tail that extends to €300, €200, even €0), and rendering that
    outside-the-data tail confuses viewers who reasonably expect the
    curve to vanish where the catalogue does.

    The histogram is strictly bounded by [prices.min(), prices.max()]:

      1. Bins are log-spaced between min and max (KDE_RESOLUTION bins).
         Log spacing gives equal sample density across the chart when
         the frontend renders on a log-scale x-axis — without it, the
         lower-price region (where most luxury catalogues peak) gets
         too few bins to render smoothly.

      2. Counts per bin are gaussian-smoothed with zero-edge padding so
         the smoothed curve tapers cleanly to zero at the boundaries.
         The padding is what makes the boundary go to zero — without
         it the boundary bin would smooth to its own count divided by
         the kernel partial sum, which is non-zero.

    Output schema kept identical to the old KDE so the frontend doesn't
    need to know the underlying algorithm changed.
    """
    if prices.size < 2:
        x = np.array([float(prices[0]), float(prices[0]) + 1])
        return {"x_euros": x.tolist(), "density": [1.0, 1.0]}

    lo, hi = float(prices.min()), float(prices.max())
    log_lo, log_hi = float(np.log(lo)), float(np.log(hi))

    n_bins = KDE_RESOLUTION
    log_edges = np.linspace(log_lo, log_hi, n_bins + 1)
    edges = np.exp(log_edges)
    counts, _ = np.histogram(prices, bins=edges)
    bin_centers = np.sqrt(edges[:-1] * edges[1:])  # geometric mean in log space

    # Gaussian smoothing with zero edge padding. Sigma is in "bins"
    # units; scaled with n_bins so the smoothing kernel always covers
    # the same fraction of the chart regardless of resolution.
    sigma = max(2.0, n_bins / 40.0)
    pad = max(int(np.ceil(3 * sigma)), 3)
    padded = np.concatenate([np.zeros(pad), counts.astype(float), np.zeros(pad)])

    kx = np.arange(-pad, pad + 1)
    kernel = np.exp(-0.5 * (kx / sigma) ** 2)
    kernel /= kernel.sum()

    smoothed_padded = np.convolve(padded, kernel, mode="same")
    smoothed = smoothed_padded[pad:-pad]

    return {
        "x_euros": [round(v, 1) for v in bin_centers.tolist()],
        "density": [round(v, 6) for v in smoothed.tolist()],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Anchor selection — the editorial touch points on the curve
# ─────────────────────────────────────────────────────────────────────────────


def url_for(brand_dir: str, name_stem: str) -> str:
    """Find the public URL for a product whose name stem is something like
    'Cartier_42'. Returns "" if no jpg matches."""
    root = FRONTEND_PUBLIC / "brands" / brand_dir
    # name_stem like "Cartier_42" → filename "Cartier_42.jpg"
    target = f"{name_stem}.jpg"
    for p in root.rglob(target):
        return "/" + str(p.relative_to(FRONTEND_PUBLIC)).replace("\\", "/")
    return ""


def cluster_for_product(brand_block: dict, name_stem: str) -> int | None:
    """Lookup the cluster id this product belongs to, from cluster_data.json's
    per-cluster `all_images` field."""
    target = f"{name_stem}.jpg"
    for c in brand_block.get("clusters", []):
        for img in c.get("all_images", []) + c.get("images", []):
            if img.endswith(target):
                return int(c["id"])
    return None


def pick_anchors(brand: str, products: list[tuple[str, float]], brand_block: dict) -> list[dict]:
    """For each anchor kind (min / median / mean / max), pick the product
    whose price is closest to that statistic and resolve its URL +
    cluster. Ensures the four returned anchors are four distinct
    products, even when (e.g.) median == mean.
    """
    prices = np.array([p for _, p in products])
    targets: dict[str, float] = {
        "min":    float(prices.min()),
        "median": float(np.median(prices)),
        "mean":   float(prices.mean()),
        "max":    float(prices.max()),
    }
    anchors: list[dict] = []
    seen_names: set[str] = set()
    brand_dir = BRAND_DIRS[brand]
    for kind in ANCHOR_KINDS:
        target_price = targets[kind]
        # Iterate candidates by distance to the target until we find one
        # not already used elsewhere in this anchor set.
        order = np.argsort(np.abs(prices - target_price))
        chosen = None
        for idx in order:
            name = products[idx][0]
            if name in seen_names:
                continue
            chosen = (idx, name, products[idx][1])
            break
        if chosen is None:
            continue
        idx, name, price = chosen
        seen_names.add(name)
        url = url_for(brand_dir, name)
        cluster_id = cluster_for_product(brand_block, name)
        anchors.append({
            "kind":         kind,
            "target_price": round(target_price, 0),
            "actual_price": round(price, 0),
            "filename":     f"{name}.jpg",
            "url":          url,
            "cluster_id":   cluster_id,
        })
    return anchors


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main():
    cluster_data = json.load(open(CLUSTER_FILE))
    out: dict = {}

    for brand, csv_rel in BRAND_CSVS.items():
        products = load_brand_products(brand)
        if not products:
            print(f"  [skip] {brand}: no priced products")
            continue
        prices = np.array([p for _, p in products])
        stats = compute_stats(prices)
        kde = kde_curve(prices)
        brand_block = cluster_data.get("brands", {}).get(brand, {})
        anchors = pick_anchors(brand, products, brand_block)
        out[brand] = {
            "stats":   stats,
            "kde":     kde,
            "anchors": anchors,
        }
        spread = stats["max"] - stats["min"]
        print(f"  [{brand:18s}] n={stats['n']:3d}  "
              f"€{int(stats['min']):,}–€{int(stats['max']):,}  "
              f"median €{int(stats['median']):,}  spread €{int(spread):,}  "
              f"anchors={len(anchors)}")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(out, indent=2))
    size_kb = OUT_FILE.stat().st_size // 1024
    print(f"\n[done] wrote {OUT_FILE.relative_to(REPO_ROOT)} ({size_kb} KB)")


if __name__ == "__main__":
    main()
