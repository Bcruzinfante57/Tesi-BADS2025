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

# Eyewear style taxonomy — 15 buckets. Key = label for the frontend,
# value = FashionCLIP prompt. We split "geometric" into hexagonal/octagonal
# (the model can distinguish them) and add butterfly/browline/mask/navigator
# for luxury-eyewear vocabulary that's editorially distinct.
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
    "butterfly":    "butterfly frame sunglasses",
    "browline":     "browline clubmaster sunglasses",
    "navigator":    "navigator rounded square sunglasses",
    "oversized":    "oversized sunglasses",
    "rimless":      "rimless metal sunglasses",
}

# Below this top-1 softmax confidence, the product is marked as
# "experimental" — i.e., the model is not sure it belongs to any of
# the named buckets, which is exactly the "weird ones cluster" the user
# wants to surface separately.
EXPERIMENTAL_CONF_THRESHOLD = 0.30

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

        products = []
        for i, p in enumerate(img_paths):
            top3 = [(style_keys[j], round(top_vals[i, k].item(), 4))
                    for k, j in enumerate(top_idx[i].tolist())]
            assigned = top3[0][0] if top3[0][1] >= EXPERIMENTAL_CONF_THRESHOLD else "experimental"
            products.append({
                "filename":   p.name,
                "style":      assigned,
                "confidence": top3[0][1],
                "raw_cos":    round(sims[i, top_idx[i, 0]].item(), 4),
                "top3":       top3,
            })
        assignments.setdefault(brand, {})[season] = products

        counts = {k: 0 for k in style_keys + ["experimental"]}
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
    all_buckets = style_keys + ["experimental"]
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
    # (brand, season). Eyeball this to confirm prompts are sensible.
    # Experimental gets LOWEST-confidence members instead (they're the
    # actual ambiguous cases — most editorially interesting).
    validation: dict = {}
    for brand in assignments:
        validation[brand] = {}
        for season in assignments[brand]:
            per_style = {k: [] for k in style_keys + ["experimental"]}
            for prod in assignments[brand][season]:
                per_style[prod["style"]].append(prod)
            for style, prods in per_style.items():
                if style == "experimental":
                    prods.sort(key=lambda x: x["confidence"])
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
