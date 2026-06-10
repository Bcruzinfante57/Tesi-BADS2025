#!/usr/bin/env python3
"""
compare_backbones.py — head-to-head comparison of DINO vs FashionCLIP on
the matching task (cross-season cosine similarity distribution).

Loads cached embeddings, L2-normalizes them, computes cos sim matrix
between F25 and S26 per brand, and reports percentile statistics so we
can tell whether FashionCLIP's "spread" is more useful than DINO's for
identifying genuinely persisted products vs novelties.

Run from this repo with the conda base python (no transformers needed
here — embeddings are already cached):
    /opt/anaconda3/bin/python compare_backbones.py
"""

import json
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).parent
EMBED_DIR = REPO_ROOT / "snapshots" / "embeddings"

BRANDS = {
    "Bottega Veneta": {
        "DINO":        ("DINO_Bottega_Veneta_F25_n123.pt",        "DINO_Bottega_Veneta_S26_n162.pt"),
        "MAE":         ("MAE_Bottega_Veneta_F25_n123.pt",         "MAE_Bottega_Veneta_S26_n162.pt"),
        "FashionCLIP": ("FashionCLIP_Bottega_Veneta_F25_n123.pt", "FashionCLIP_Bottega_Veneta_S26_n162.pt"),
    },
    "Dolce & Gabbana": {
        "DINO":        ("DINO_Dolce_and_Gabbana_F25_n161.pt",        "DINO_Dolce_and_Gabbana_S26_n114.pt"),
        "MAE":         ("MAE_Dolce_and_Gabbana_F25_n161.pt",         "MAE_Dolce_and_Gabbana_S26_n114.pt"),
        "FashionCLIP": ("FashionCLIP_Dolce_and_Gabbana_F25_n161.pt", "FashionCLIP_Dolce_and_Gabbana_S26_n114.pt"),
    },
}


def l2norm(X: torch.Tensor) -> torch.Tensor:
    return X / X.norm(dim=1, keepdim=True).clamp_min(1e-12)


def cossim_stats(f25: torch.Tensor, s26: torch.Tensor) -> dict:
    f25, s26 = l2norm(f25), l2norm(s26)
    cos = f25 @ s26.T  # (M, N)
    # Per-S26 best match in F25
    s26_best = cos.max(dim=0).values
    # Per-F25 best match in S26
    f25_best = cos.max(dim=1).values
    qs = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    return {
        "n_f25": f25.shape[0],
        "n_s26": s26.shape[0],
        "feature_dim": f25.shape[1],
        "best_per_s26": {f"p{int(q*100):02d}": round(s26_best.quantile(q).item(), 4) for q in qs},
        "best_per_f25": {f"p{int(q*100):02d}": round(f25_best.quantile(q).item(), 4) for q in qs},
        # Threshold sweeps — what % would be classified as "persisted" at each cutoff
        "persistence_rate_at_thresh": {
            f"{thr:.2f}": round((f25_best >= thr).float().mean().item(), 4)
            for thr in [0.70, 0.80, 0.85, 0.90, 0.92, 0.95]
        },
        "novelty_rate_at_thresh": {
            f"{thr:.2f}": round(1 - (s26_best >= thr).float().mean().item(), 4)
            for thr in [0.70, 0.80, 0.85, 0.90, 0.92, 0.95]
        },
    }


def main():
    out = {}
    for brand, backbones in BRANDS.items():
        out[brand] = {}
        for backbone, (f25_name, s26_name) in backbones.items():
            f25_path = EMBED_DIR / f25_name
            s26_path = EMBED_DIR / s26_name
            if not f25_path.exists() or not s26_path.exists():
                print(f"  [skip] {brand} · {backbone}: missing embeddings")
                continue
            f25 = torch.load(f25_path, map_location="cpu")
            s26 = torch.load(s26_path, map_location="cpu")
            out[brand][backbone] = cossim_stats(f25, s26)

    # Print a compact comparison
    print(f"{'Brand':18s} {'Backbone':12s} {'dim':>4s}  "
          f"{'p05':>6s} {'p50':>6s} {'p95':>6s}  "
          f"{'@0.85':>6s} {'@0.92':>6s}")
    print("=" * 90)
    for brand, bbs in out.items():
        for backbone, s in bbs.items():
            q = s["best_per_s26"]
            print(f"{brand:18s} {backbone:12s} {s['feature_dim']:>4d}  "
                  f"{q['p05']:>6.3f} {q['p50']:>6.3f} {q['p95']:>6.3f}  "
                  f"{s['novelty_rate_at_thresh']['0.85']:>6.2%} {s['novelty_rate_at_thresh']['0.92']:>6.2%}")
        print()

    (REPO_ROOT / "snapshots" / "backbone_comparison.json").write_text(json.dumps(out, indent=2))
    print(f"\n[done] wrote snapshots/backbone_comparison.json")


if __name__ == "__main__":
    main()
