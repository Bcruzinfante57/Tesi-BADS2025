#!/usr/bin/env python3
"""
price_kde.py — Kernel Density Estimates of product prices in log10 space,
per brand × snapshot. Designed for the comparative price-curve view in
the frontend.

Why log10:
  Luxury prices are log-normally distributed (e.g. Cartier mixes €450
  jewellery with €19,000 watches). KDE on the raw scale gets crushed by
  the long tail; KDE on log10 is shape-honest and lets us compare
  brands of wildly different absolute scale on the same axis.

Why Gaussian:
  Smooth, no parametric assumption about the shape (so bimodality survives).
  Bandwidth set by Scott's rule: h = σ × n^(-1/5).

Output (`snapshots/kde.json`) is a dict
  { brand: { snapshot: { x: [...], density: [...], summary: { ... } } } }
where `x` is in log10(price) and `density` integrates to 1 over the x range.

Run:
    python price_kde.py
"""

import csv
import json
import math
import re
from pathlib import Path
from statistics import mean, median, stdev

import torch

REPO_ROOT = Path(__file__).parent
OUTPUT = REPO_ROOT / "snapshots" / "kde.json"

CSV_PATHS = {
    "F25": {
        "Bottega Veneta":  "images_bottega/bottega_products.csv",
        "Dolce & Gabbana": "images_D&G/dolcegabbana_products.csv",
    },
    "S26": {
        "Bottega Veneta":  "snapshots/S26/raw/bottega/bottega_products.csv",
        "Dolce & Gabbana": "snapshots/S26/raw/dg/dolcegabbana_products.csv",
    },
}

# Common log-price grid (€100 – €100,000) so brands & seasons share an x-axis
LOG_X_MIN = 2.0   # 10^2 = €100
LOG_X_MAX = 5.0   # 10^5 = €100,000
N_POINTS  = 200


def load_prices(path: Path) -> list[float]:
    """Parse the last cell of each row as a price; skip rows with no price.

    Defensive: the price cell must contain '€' or '$' (or be the only cell
    that's purely numeric and ≥ 50). Some scrapers leave the price empty
    when a product card failed to load — we previously fell back to the
    Product Name column which contains things like 'Bottega_42', so '42'
    was being read as a €42 price.
    """
    if not path.exists():
        print(f"  [warn] missing {path}")
        return []
    prices = []
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)  # header
        for row in reader:
            last = row[-1] if row else ""
            # Require a currency symbol to count as a real price
            if "€" not in last and "$" not in last and "£" not in last:
                continue
            m = re.search(r"([0-9]+(?:[.,][0-9]+)?)", last)
            if not m:
                continue
            try:
                v = float(m.group(1).replace(",", "."))
                if v >= 50:  # luxury eyewear floor sanity-check
                    prices.append(v)
            except ValueError:
                continue
    return prices


def scott_bandwidth(samples: torch.Tensor) -> float:
    """Scott's rule: h = σ × n^(-1/5)."""
    sigma = samples.std(unbiased=True).item()
    n = samples.numel()
    return sigma * (n ** (-1 / 5))


def gaussian_kde(samples: torch.Tensor, grid: torch.Tensor, h: float) -> torch.Tensor:
    """Density at each x in `grid`, evaluated as the average Gaussian kernel.

    KDE(x) = (1 / n h) Σ_i  φ((x - x_i) / h)
    where φ(z) = (1/√(2π)) exp(-z²/2) is the standard normal pdf.
    """
    # (n_samples, 1) vs (1, n_grid) → (n_samples, n_grid)
    z = (grid.unsqueeze(0) - samples.unsqueeze(1)) / h
    kernel = torch.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
    return kernel.mean(dim=0) / h


def main():
    grid = torch.linspace(LOG_X_MIN, LOG_X_MAX, N_POINTS)
    out: dict = {
        "log_x": [round(x, 4) for x in grid.tolist()],
        "x_euros": [round(10 ** x, 2) for x in grid.tolist()],
        "brands": {},
    }

    for brand in CSV_PATHS["S26"].keys():
        out["brands"][brand] = {}
        for snapshot in ("F25", "S26"):
            prices = load_prices(REPO_ROOT / CSV_PATHS[snapshot][brand])
            if len(prices) < 5:
                print(f"  [skip] {brand} {snapshot}: only {len(prices)} prices")
                continue

            log_prices = torch.tensor([math.log10(p) for p in prices])
            h = scott_bandwidth(log_prices)
            density = gaussian_kde(log_prices, grid, h)

            out["brands"][brand][snapshot] = {
                "n": len(prices),
                "bandwidth": round(h, 5),
                "density": [round(d, 6) for d in density.tolist()],
                "summary": {
                    "min": round(min(prices), 2),
                    "max": round(max(prices), 2),
                    "mean": round(mean(prices), 2),
                    "median": round(median(prices), 2),
                    "std": round(stdev(prices), 2) if len(prices) > 1 else 0,
                    "log_median": round(log_prices.median().item(), 4),
                    # IQR in log-space for shaded band rendering
                    "log_q25": round(log_prices.quantile(0.25).item(), 4),
                    "log_q75": round(log_prices.quantile(0.75).item(), 4),
                },
            }
            s = out["brands"][brand][snapshot]["summary"]
            print(f"  {brand:18s} {snapshot}  n={len(prices):3d}  "
                  f"€{s['min']:.0f}-{s['max']:.0f}  median €{s['median']:.0f}  bw={h:.3f}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(out, indent=2))
    print(f"\n[done] wrote {OUTPUT}")


if __name__ == "__main__":
    main()
