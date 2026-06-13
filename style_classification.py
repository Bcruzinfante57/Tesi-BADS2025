#!/usr/bin/env python3
"""
style_classification.py — zero-shot eyewear style tagging via FashionCLIP.

For each product image we compute cos sim against a taxonomy of style
prompts in the FashionCLIP joint image-text space, and assign the top
label. No training, no labels needed.

The taxonomy and prompts are designed for sunglasses specifically — they
mimic Farfetch product description style (which is how FashionCLIP was
trained), so no "a photo of" prefix.

Run from this directory inside the dedicated venv:
    .venv-fclip/bin/python style_classification.py

Outputs to snapshots/styles/:
    style_assignments.json   — per-product label + top3 + confidence
    style_aggregates.json    — counts + mix % + S26 delta vs F25
    style_validation.json    — top-5 most-confident products per style
                               (eyeball this to validate the taxonomy)
"""

import json
from pathlib import Path

import torch
from transformers import CLIPModel, CLIPProcessor

REPO_ROOT = Path(__file__).parent
EMBED_DIR = REPO_ROOT / "snapshots" / "embeddings"
OUT_DIR = REPO_ROOT / "snapshots" / "styles"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ID = "patrickjohncyh/fashion-clip"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# Eyewear style taxonomy. Key = label for the frontend, value = FashionCLIP
# prompt. We split "geometric" into hexagonal/octagonal (the model can
# distinguish them) and add browline/mask/navigator for luxury-eyewear
# vocabulary that's editorially distinct.
#
# "butterfly" was tried but didn't survive a visual audit — FashionCLIP
# uses it as a soft attractor for "wide feminine shapes that aren't strictly
# cat-eye or oval", so the resulting bucket mixes heart-shaped, oversized
# rounded, and brand-unique pieces. Removing it sends genuine wide-rounded
# pieces to oval/oversized and unique pieces to the experimental bucket —
# editorially cleaner.
STYLE_PROMPTS = {
    "cat-eye":      "cat eye sunglasses",
    "aviator":      "aviator sunglasses",
    "square":       "square frame sunglasses",
    "rectangular":  "rectangular frame sunglasses",
    "round":        "round frame sunglasses",
    "oval":         "oval frame sunglasses",
    "hexagonal":    "hexagonal frame sunglasses",
    "octagonal":    "octagonal frame sunglasses",
    "shield":       "shield wraparound sunglasses",
    "mask":         "single lens visor mask sunglasses",
    "browline":     "browline clubmaster sunglasses",
    "navigator":    "navigator rounded square sunglasses",
    "oversized":    "oversized sunglasses",
    "rimless":      "rimless metal sunglasses",
}

# A product goes to the "signature" bucket — the maison's distinctive
# pieces, things only this house would make — if ANY of these are true:
#
#   1. top-1 softmax confidence < SIGNATURE_CONF_THRESHOLD: the silhouette
#      classifier has no strong opinion at all, so the product doesn't fit
#      any named silhouette template cleanly.
#   2. top1 – top2 < SIGNATURE_MARGIN_THRESHOLD: a technical tie between
#      two silhouettes, i.e. a cross-style hybrid.
#   3. intra-season rarity (1 – mean cos sim against own brand·season
#      catalogue) above the per-catalogue P{RARITY_PERCENTILE} cutoff: the
#      product looks unlike anything else THIS maison is shipping this
#      season. This is the rule that catches Bottega's heart-shaped pieces
#      and floral-embellished frames — silhouette-wise they look like
#      "oversized" or "hexagonal" to FashionCLIP, but they are visually
#      isolated within Bottega's own catalogue, which is exactly what
#      "exclusive to the maison" means editorially.
#
# Together rules 1 + 2 catch ambiguous classifications; rule 3 catches
# brand-signature pieces the silhouette taxonomy can't see.
SIGNATURE_CONF_THRESHOLD   = 0.20  # below 3× random baseline (1/14 ≈ 0.07)
SIGNATURE_MARGIN_THRESHOLD = 0.05  # virtual tie between top-1 and top-2
RARITY_PERCENTILE          = 0.93  # top ~7% most visually isolated
# Rarity is treated as a SECONDARY signal — it only redirects a product
# to "signature" if the silhouette classifier is also unsure (top-1 below
# this cutoff). Above it, a confidently labelled aviator stays an aviator
# regardless of how rare its colourway is. Without this gate the rarity
# rule was pulling obvious aviators / cat-eyes / hexagonals into signature
# because their colour was uncommon — editorially wrong.
RARITY_CONF_GATE           = 0.50

# (brand_label, season_label, fashionclip_embed_filename, image_folder)
SOURCES = [
    ("Bottega Veneta",  "F25", "FashionCLIP_Bottega_Veneta_F25_n123.pt",        "images_bottega"),
    ("Bottega Veneta",  "S26", "FashionCLIP_Bottega_Veneta_S26_n162.pt",        "snapshots/S26/raw/bottega"),
    ("Dolce & Gabbana", "F25", "FashionCLIP_Dolce_and_Gabbana_F25_n161.pt",     "images_D&G"),
    ("Dolce & Gabbana", "S26", "FashionCLIP_Dolce_and_Gabbana_S26_n114.pt",     "snapshots/S26/raw/dg"),
]


def l2norm(X: torch.Tensor) -> torch.Tensor:
    return X / X.norm(dim=1, keepdim=True).clamp_min(1e-12)


@torch.no_grad()
def encode_text(model, processor, prompts: list[str]) -> torch.Tensor:
    inputs = processor(text=prompts, return_tensors="pt", padding=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    text_out = model.text_model(**inputs)
    pooled = text_out.pooler_output
    feats = model.text_projection(pooled)
    return feats.detach().cpu()


def intra_season_rarity(img_emb_l2: torch.Tensor) -> torch.Tensor:
    """1 - average cos sim against every OTHER product in the same season.

    img_emb_l2 must already be L2-normalised, shape (N, D). The output is
    shape (N,): higher values mean the product is visually further from
    everything else in its own catalogue — i.e., a brand-signature outlier.
    """
    sims = img_emb_l2 @ img_emb_l2.T  # (N, N)
    n = sims.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool)
    mean_sim = (sims * mask).sum(dim=1) / mask.sum(dim=1)
    return 1.0 - mean_sim


def main():
    print(f"[device] {DEVICE}")
    print(f"[model]  loading {MODEL_ID} …")
    model = CLIPModel.from_pretrained(MODEL_ID).to(DEVICE)
    processor = CLIPProcessor.from_pretrained(MODEL_ID)
    model.eval()

    style_keys = list(STYLE_PROMPTS.keys())
    prompts    = [STYLE_PROMPTS[k] for k in style_keys]
    text_emb   = l2norm(encode_text(model, processor, prompts))
    print(f"[text]  encoded {len(prompts)} style prompts → {tuple(text_emb.shape)}")
    print()

    assignments: dict[str, dict[str, list]] = {}
    aggregates:  dict[str, dict[str, dict]] = {}

    for brand, season, emb_name, img_folder in SOURCES:
        emb_path = EMBED_DIR / emb_name
        img_emb  = l2norm(torch.load(emb_path, map_location="cpu"))
        img_paths = sorted((REPO_ROOT / img_folder).glob("*.jpg"))
        assert len(img_paths) == img_emb.shape[0], (
            f"{brand} {season}: {len(img_paths)} files vs {img_emb.shape[0]} embeddings"
        )

        # Cos sim → softmax over taxonomy at CLIP's standard T=100
        sims  = img_emb @ text_emb.T
        probs = torch.softmax(sims * 100, dim=1)

        top_vals, top_idx = probs.topk(3, dim=1)

        # Intra-season rarity: how visually distant is each product from the
        # rest of its OWN brand·season catalogue. Top 10% by this score get
        # promoted to "signature" regardless of how confidently the silhouette
        # classifier wanted to label them.
        rarity      = intra_season_rarity(img_emb)
        rarity_cut  = rarity.quantile(RARITY_PERCENTILE).item()

        products = []
        for i, p in enumerate(img_paths):
            top3 = [(style_keys[j], round(top_vals[i, k].item(), 4))
                    for k, j in enumerate(top_idx[i].tolist())]
            top1_conf = top3[0][1]
            top2_conf = top3[1][1] if len(top3) > 1 else 0.0
            margin    = top1_conf - top2_conf
            rare_i    = rarity[i].item()

            low_conf     = top1_conf < SIGNATURE_CONF_THRESHOLD
            tied         = margin    < SIGNATURE_MARGIN_THRESHOLD
            # Rarity-driven signature: visually isolated AND silhouette
            # classifier not confident. The conf-gate prevents confidently
            # labelled aviators / cat-eyes from getting pulled into
            # signature just because their colourway is unusual.
            isolated     = rare_i > rarity_cut and top1_conf < RARITY_CONF_GATE

            if low_conf or tied or isolated:
                assigned = "signature"
            else:
                assigned = top3[0][0]

            products.append({
                "filename":   p.name,
                "style":      assigned,
                "confidence": top3[0][1],
                "raw_cos":    round(sims[i, top_idx[i, 0]].item(), 4),
                "rarity":     round(rare_i, 4),
                "top3":       top3,
            })
        assignments.setdefault(brand, {})[season] = products

        counts = {k: 0 for k in style_keys + ["signature"]}
        for prod in products:
            counts[prod["style"]] += 1
        n = len(products)
        aggregates.setdefault(brand, {})[season] = {
            "n_total":  n,
            "by_style": counts,
            "mix_pct":  {k: round(v / n * 100, 1) for k, v in counts.items()},
        }

        nonzero = {k: v for k, v in counts.items() if v}
        print(f"  [{brand:18s} {season}] n={n:3d} → "
              + "  ".join(f"{k}={v}" for k, v in sorted(nonzero.items(),
                                                         key=lambda x: -x[1])))

    print()
    # S26 vs F25 absolute count delta per brand
    all_buckets = style_keys + ["signature"]
    for brand in aggregates:
        if "F25" in aggregates[brand] and "S26" in aggregates[brand]:
            f25 = aggregates[brand]["F25"]["by_style"]
            s26 = aggregates[brand]["S26"]["by_style"]
            delta = {k: s26[k] - f25[k] for k in all_buckets}
            aggregates[brand]["S26"]["delta_vs_F25"] = delta
            growth = sorted(delta.items(), key=lambda x: -x[1])
            print(f"  [{brand} S26 Δ vs F25] "
                  + "  ".join(f"{k}{v:+d}" for k, v in growth if v))

    # Validation aid — top-5 most confident assignments per style, per
    # (brand, season). Eyeball this to confirm prompts are sensible. The
    # "signature" bucket is sorted by rarity (most isolated first) since
    # that is what makes them editorially interesting — the most distinctive
    # member of the maison's signature appears first.
    validation: dict = {}
    for brand in assignments:
        validation[brand] = {}
        for season in assignments[brand]:
            per_style = {k: [] for k in style_keys + ["signature"]}
            for prod in assignments[brand][season]:
                per_style[prod["style"]].append(prod)
            for style, prods in per_style.items():
                if style == "signature":
                    prods.sort(key=lambda x: x.get("rarity", 0), reverse=True)
                else:
                    prods.sort(key=lambda x: x["confidence"], reverse=True)
                per_style[style] = prods[:5]
            validation[brand][season] = per_style

    (OUT_DIR / "style_assignments.json").write_text(json.dumps(assignments, indent=2))
    (OUT_DIR / "style_aggregates.json").write_text(json.dumps(aggregates,  indent=2))
    (OUT_DIR / "style_validation.json").write_text(json.dumps(validation,  indent=2))
    print()
    print(f"[done] wrote 3 files to {OUT_DIR}/")


if __name__ == "__main__":
    main()
