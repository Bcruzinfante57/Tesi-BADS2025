#!/usr/bin/env python3
"""
novelty_pipeline.py — cross-season novelty + discontinuation + hit scoring
on FashionCLIP embeddings.

Produces snapshots/novelty.json with this shape per brand:

  {
    "Bottega Veneta": {
      "matching": {
        "threshold": 0.86,                    # data-driven cutoff
        "threshold_method": "histogram-valley",
        "n_f25": 123, "n_s26": 160,
        "n_new": 47, "n_hit": 12,
        "n_discontinued": 41
      },
      "new_products": [          # S26 with no close F25 match,
                                 # sorted by novelty desc
        {
          "filename": "Bottega_140.jpg",
          "url": "/snapshots/S26/bottega/Bottega_140.jpg",
          "novelty_score": 0.42,             # 1 - best_match_cos_sim
          "best_match_filename": "Bottega_82.jpg",
          "best_match_url": "/brands/bottega_veneta/.../Bottega_82.jpg",
          "best_match_cos": 0.58,
          "intra_season_rarity": 0.27,       # how unique within own S26
          "hit_score": 0.36,                 # composite
          "is_hit": true,
          "style": "square",                 # silhouette label
          "colors": [{hex, name, share}, ...]
        },
        ...
      ],
      "discontinued_products": [ # F25 with no close S26 match,
                                 # sorted by discontinuation desc
        ...same shape, fields rename to discontinuation_score etc...
      ]
    },
    "Dolce & Gabbana": {...}
  }

The discontinuation list highlights pieces the maison closed a chapter
on — F25 products that have nothing visually similar in the S26 drop.

Run with the base conda python (no transformers needed — embeddings
are cached):
    /opt/anaconda3/bin/python novelty_pipeline.py
"""

import hashlib
import json
from pathlib import Path

import torch

REPO_ROOT       = Path(__file__).parent
EMBED_DIR       = REPO_ROOT / "snapshots" / "embeddings"
PALETTES_F25    = REPO_ROOT / "palettes_all_brands_v2.json"
PALETTES_S26    = REPO_ROOT / "palettes_S26.json"
ASSIGN_FILE     = REPO_ROOT / "snapshots" / "styles" / "style_assignments.json"
PRICES_FILE     = REPO_ROOT / "snapshots" / "cluster_data_v2.json"
OUT_FILE        = REPO_ROOT / "snapshots" / "novelty.json"
FRONTEND_PUBLIC = Path("/Users/benja/conan-insight-hub/public")

# Cross-season match distributions are strongly unimodal (most products
# carry over) so valley detection on the histogram doesn't find anything
# robust. Use a per-brand percentile instead: the lowest-N % of best-match
# cos sim are flagged as new (or discontinued, on the mirror direction).
# 20 % gives ~25–35 entries per maison — a meaningful carousel without
# noise.
NEW_PERCENTILE_CUTOFF   = 0.20

# Hits are defined as the top fraction of NEW products by composite score
# (60 % novelty vs F25 + 40 % intra-S26 rarity). Per-brand selection
# guarantees the hit set is always a meaningful subset of new — neither
# everything nor nothing — regardless of the absolute distribution.
HIT_PERCENTILE_OF_NEW   = 0.25

# Sources: (brand, F25 embedding file, F25 image folder,
#                  S26 embedding file, S26 image folder)
SOURCES = [
    (
        "Bottega Veneta",
        "FashionCLIP_Bottega_Veneta_F25_n123.pt", "images_bottega",
        "FashionCLIP_Bottega_Veneta_S26_n162.pt", "snapshots/S26/raw/bottega",
    ),
    (
        "Dolce & Gabbana",
        "FashionCLIP_Dolce_and_Gabbana_F25_n161.pt", "images_D&G",
        "FashionCLIP_Dolce_and_Gabbana_S26_n114.pt", "snapshots/S26/raw/dg",
    ),
    # Prada was added briefly (2026-06-24) but the F25 baseline
    # (thesis 2025 scrape) wasn't actually clean eyewear — the
    # cross-season match against the 2026 re-scrape produced
    # unreliable hits/discontinued, so Prada is held back until we
    # have a proper paired F25 + S26 catalogue. Embeddings stay on
    # disk (FashionCLIP_Prada_F25_n95.pt, ..._S26_n225.pt) for the
    # next attempt.
]


def l2norm(X: torch.Tensor) -> torch.Tensor:
    return X / X.norm(dim=1, keepdim=True).clamp_min(1e-12)


def percentile_cutoff(values: torch.Tensor, percentile: float) -> float:
    """Return the value at the given percentile of best-match cos sim. Below
    this is "new" (or discontinued). Brand-aware by construction — each
    maison's distribution gives its own cutoff."""
    return float(values.quantile(percentile).item())


def find_unique_files(folder: Path) -> tuple[list[Path], list[int]]:
    """Return (unique_paths, indices_into_sorted_glob) — the alphabetically-
    first occurrence of each MD5 hash. Mirrors the dedup behaviour of the
    style_clusters merge so the cross-season matching stays consistent
    with what the user sees in the Style Mix modal grids.
    """
    paths = sorted(folder.glob("*.jpg"))
    seen: dict[str, int] = {}
    unique_paths: list[Path] = []
    unique_idx: list[int] = []
    for i, p in enumerate(paths):
        h = hashlib.md5(p.read_bytes()).hexdigest()
        if h in seen:
            continue
        seen[h] = i
        unique_paths.append(p)
        unique_idx.append(i)
    return unique_paths, unique_idx


def build_image_url_map(brand: str, season: str) -> dict[str, str]:
    """Map filename → public URL the frontend can fetch.

    Per-brand folder layout (mirrors the conan-insight-hub public/ tree):
      Bottega Veneta   F25  /brands/bottega_veneta/N/Bottega_X.jpg
                       S26  /snapshots/S26/bottega/Bottega_X.jpg
      Dolce & Gabbana  F25  /brands/dolce_and_gabbana/N/D&G_X.jpg
                       S26  /snapshots/S26/dg/D&G_X.jpg
      Prada            F25  /brands/prada/N/Prada_X.jpg            (thesis 2025)
                       S26  /snapshots/2026/prada/Prada_X.jpg      (2026 re-scrape)
    """
    if season == "S26":
        if "Bottega" in brand:
            brand_dir = "bottega"
            folder = FRONTEND_PUBLIC / "snapshots" / "S26" / brand_dir
            prefix = f"/snapshots/S26/{brand_dir}"
        elif "Dolce" in brand or "D&G" in brand:
            brand_dir = "dg"
            folder = FRONTEND_PUBLIC / "snapshots" / "S26" / brand_dir
            prefix = f"/snapshots/S26/{brand_dir}"
        elif "Prada" in brand:
            # Prada's "S26" is the 2026 re-scrape — frontend keeps it
            # under snapshots/2026/prada/ to leave the S26 folder
            # reserved for the Bottega + D&G AW26 reference run.
            brand_dir = "prada"
            folder = FRONTEND_PUBLIC / "snapshots" / "2026" / brand_dir
            prefix = f"/snapshots/2026/{brand_dir}"
        else:
            return {}
        return {p.name: f"{prefix}/{p.name}" for p in folder.glob("*.jpg")}

    # F25 — every brand lives under public/brands/<slug>/<cluster_id>/…jpg
    brand_dir = {
        "Bottega Veneta":   "bottega_veneta",
        "Dolce & Gabbana":  "dolce_and_gabbana",
        "Prada":            "prada",
    }.get(brand)
    if brand_dir is None:
        return {}
    root = FRONTEND_PUBLIC / "brands" / brand_dir
    return {p.name: "/" + str(p.relative_to(FRONTEND_PUBLIC)).replace("\\", "/")
            for p in root.rglob("*.jpg")}


def load_palette_for(brand: str, season: str) -> dict:
    fpath = PALETTES_F25 if season == "F25" else PALETTES_S26
    raw = json.load(open(fpath))
    if brand not in raw:
        cands = [k for k in raw if brand.split()[0] in k]
        if not cands:
            return {}
        brand = cands[0]
    return raw[brand]


def palette_for(fname: str, palette_dict: dict, top_n: int = 5) -> list[dict]:
    p = palette_dict.get(fname, {}).get("palette", [])
    return [
        {"hex": c["hex"], "name": c.get("name_bucket", "unknown"),
         "coverage": round(c["coverage"], 4)}
        for c in p[:top_n]
    ]


def style_for_file(assigns: dict, brand: str, season: str, fname: str) -> str:
    for prod in assigns.get(brand, {}).get(season, []):
        if prod["filename"] == fname:
            return prod["style"]
    return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Per-product prices — loaded once from cluster_data_v2.json and indexed
# by (brand, season, filename) for O(1) lookup during the per-row enrichment.
# ─────────────────────────────────────────────────────────────────────────────


def load_price_index() -> dict[tuple[str, str, str], float]:
    """{(brand, season, filename): price_eur}."""
    if not PRICES_FILE.exists():
        return {}
    raw = json.load(open(PRICES_FILE))
    index: dict[tuple[str, str, str], float] = {}
    for brand, block in raw.get("brands", {}).items():
        for season, snap in block.get("snapshots", {}).items():
            for p in snap.get("products", []):
                price = p.get("price_eur")
                if price is None:
                    continue
                fname = p.get("image") or p.get("filename")
                if not fname:
                    continue
                index[(brand, season, fname)] = float(price)
    return index


def price_for(price_index: dict, brand: str, season: str, fname: str) -> float | None:
    return price_index.get((brand, season, fname))


def intra_season_rarity(emb_l2: torch.Tensor) -> torch.Tensor:
    sims = emb_l2 @ emb_l2.T
    n = sims.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool)
    return 1.0 - (sims * mask).sum(dim=1) / mask.sum(dim=1)


def main():
    assigns = json.load(open(ASSIGN_FILE))
    price_index = load_price_index()
    out: dict = {}

    for brand, f25_emb_file, f25_folder, s26_emb_file, s26_folder in SOURCES:
        # Load embeddings (full, in original order — matches sorted glob)
        f25_full = l2norm(torch.load(EMBED_DIR / f25_emb_file, map_location="cpu"))
        s26_full = l2norm(torch.load(EMBED_DIR / s26_emb_file, map_location="cpu"))

        # Dedup F25 and S26 by MD5 (same rule as style_clusters merge)
        f25_paths, f25_keep = find_unique_files(REPO_ROOT / f25_folder)
        s26_paths, s26_keep = find_unique_files(REPO_ROOT / s26_folder)
        f25 = f25_full[f25_keep]
        s26 = s26_full[s26_keep]

        f25_url = build_image_url_map(brand, "F25")
        s26_url = build_image_url_map(brand, "S26")
        f25_pal = load_palette_for(brand, "F25")
        s26_pal = load_palette_for(brand, "S26")

        s26_rarity = intra_season_rarity(s26)
        f25_rarity = intra_season_rarity(f25)

        # Cross-season cos sim: (M_F25, N_S26)
        sim = f25 @ s26.T

        # Per-S26 best match in F25
        s26_best_val, s26_best_idx = sim.max(dim=0)
        # Per-F25 best match in S26
        f25_best_val, f25_best_idx = sim.max(dim=1)

        # Per-direction percentile thresholds. The S26→F25 cutoff defines
        # "new"; the F25→S26 cutoff defines "discontinued". Each side has
        # its own distribution and gets its own threshold.
        new_threshold  = percentile_cutoff(s26_best_val, NEW_PERCENTILE_CUTOFF)
        disc_threshold = percentile_cutoff(f25_best_val, NEW_PERCENTILE_CUTOFF)

        # Build new_products
        new_products = []
        n_s26 = s26.shape[0]
        for i in range(n_s26):
            bm_cos = float(s26_best_val[i].item())
            if bm_cos >= new_threshold:
                continue                  # close match in F25, not new
            fname = s26_paths[i].name
            novelty = 1.0 - bm_cos
            rarity  = float(s26_rarity[i].item())
            hit_score = 0.6 * novelty + 0.4 * rarity
            best_f25_path = f25_paths[int(s26_best_idx[i].item())]
            new_products.append({
                "filename":            fname,
                "url":                 s26_url.get(fname, ""),
                "novelty_score":       round(novelty, 4),
                "best_match_filename": best_f25_path.name,
                "best_match_url":      f25_url.get(best_f25_path.name, ""),
                "best_match_cos":      round(bm_cos, 4),
                "intra_season_rarity": round(rarity, 4),
                "hit_score":           round(hit_score, 4),
                "is_hit":              False,
                "style":               style_for_file(assigns, brand, "S26", fname),
                "colors":              palette_for(fname, s26_pal),
                "price_eur":           price_for(price_index, brand, "S26", fname),
            })

        # Promote top X% of new by hit_score to is_hit. Always at least 1
        # if there are any new products so the visual reading isn't empty.
        n_hits_target = max(1, int(round(len(new_products) * HIT_PERCENTILE_OF_NEW))) \
                        if new_products else 0
        for p in sorted(new_products, key=lambda x: -x["hit_score"])[:n_hits_target]:
            p["is_hit"] = True

        new_products.sort(key=lambda x: (-x["is_hit"], -x["novelty_score"]))

        # Build discontinued_products (mirror, F25 → S26 direction)
        disc_products = []
        n_f25 = f25.shape[0]
        for i in range(n_f25):
            bm_cos = float(f25_best_val[i].item())
            if bm_cos >= disc_threshold:
                continue
            fname = f25_paths[i].name
            disc_score = 1.0 - bm_cos
            rarity = float(f25_rarity[i].item())
            best_s26_path = s26_paths[int(f25_best_idx[i].item())]
            disc_products.append({
                "filename":              fname,
                "url":                   f25_url.get(fname, ""),
                "discontinuation_score": round(disc_score, 4),
                "best_match_filename":   best_s26_path.name,
                "best_match_url":        s26_url.get(best_s26_path.name, ""),
                "best_match_cos":        round(bm_cos, 4),
                "intra_season_rarity":   round(rarity, 4),
                "style":                 style_for_file(assigns, brand, "F25", fname),
                "colors":                palette_for(fname, f25_pal),
                "price_eur":             price_for(price_index, brand, "F25", fname),
            })
        disc_products.sort(key=lambda x: -x["discontinuation_score"])

        n_hit = sum(1 for p in new_products if p["is_hit"])
        out[brand] = {
            "matching": {
                "new_threshold":          round(new_threshold,  4),
                "discontinued_threshold": round(disc_threshold, 4),
                "threshold_method":       f"per-brand p{int(NEW_PERCENTILE_CUTOFF*100):02d} of best-match cos sim",
                "n_f25":                  n_f25,
                "n_s26":                  n_s26,
                "n_new":                  len(new_products),
                "n_hit":                  n_hit,
                "n_discontinued":         len(disc_products),
            },
            "new_products":          new_products,
            "discontinued_products": disc_products,
        }
        print(f"  [{brand}]  new<{new_threshold:.3f}: {len(new_products):3d} (hits={n_hit:2d})  |  "
              f"discontinued<{disc_threshold:.3f}: {len(disc_products):3d}  "
              f"of F25={n_f25} S26={n_s26}")

    OUT_FILE.write_text(json.dumps(out, indent=2))
    size_kb = OUT_FILE.stat().st_size // 1024
    print(f"\n[done] wrote {OUT_FILE.relative_to(REPO_ROOT)} ({size_kb} KB)")


if __name__ == "__main__":
    main()
