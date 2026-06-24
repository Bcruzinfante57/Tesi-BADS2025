#!/usr/bin/env python3
"""
fashion_clip_embeddings.py — extract FashionCLIP image embeddings for the
Bottega + D&G F25/S26 catalogues.

FashionCLIP is a CLIP-ViT-B-32 fine-tuned on ~800K Farfetch fashion image-text
pairs (Chia et al. 2022). HuggingFace model id: patrickjohncyh/fashion-clip.

We use only the image encoder. Output dim is 512 (vs 768 for DINO/MAE).
Embeddings are saved unnormalized so the downstream clustering pipeline
applies StandardScaler the same way it does for DINO/MAE.

Run from this directory inside the dedicated venv:
    .venv-fclip/bin/python fashion_clip_embeddings.py

Outputs go to snapshots/embeddings/FashionCLIP_<brand>_<season>_n<N>.pt
"""

import time
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

REPO_ROOT = Path(__file__).parent
EMBED_DIR = REPO_ROOT / "snapshots" / "embeddings"
EMBED_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ID = "patrickjohncyh/fashion-clip"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BATCH_SIZE = 16

PATHS = {
    "Bottega Veneta": {
        "F25": "images_bottega",
        "S26": "snapshots/S26/raw/bottega",
    },
    "Dolce & Gabbana": {
        "F25": "images_D&G",
        "S26": "snapshots/S26/raw/dg",
    },
    # Prada added 2026-06-24. F25 baseline = the original thesis catalogue
    # (95 sunglasses scraped in 2025). S26 = the 2026 eyewear re-scrape
    # (225 sunglasses, women + men + Linea Rossa, deduped by image URL).
    "Prada": {
        "F25": "images_Prada",
        "S26": "images_Prada_2026",
    },
}


def slug(brand: str) -> str:
    return brand.replace(" ", "_").replace("&", "and")


@torch.no_grad()
def embed_brand_season(model, processor, brand: str, season: str) -> tuple[torch.Tensor, list[Path]]:
    folder = REPO_ROOT / PATHS[brand][season]
    paths = sorted(folder.glob("*.jpg"))
    if not paths:
        raise FileNotFoundError(folder)

    out = []
    for i in range(0, len(paths), BATCH_SIZE):
        chunk = paths[i:i + BATCH_SIZE]
        imgs = []
        for p in chunk:
            try:
                imgs.append(Image.open(p).convert("RGB"))
            except Exception:
                imgs.append(Image.new("RGB", (224, 224), "white"))
        inputs = processor(images=imgs, return_tensors="pt", padding=True)
        pixel_values = inputs["pixel_values"].to(DEVICE)
        # transformers v5 changed get_image_features return type; build the
        # embedding manually instead to stay version-agnostic:
        #   vision_model → pooler_output → visual_projection → image embedding
        vision_out = model.vision_model(pixel_values=pixel_values)
        pooled = vision_out.pooler_output  # (B, 768)
        feats = model.visual_projection(pooled)  # (B, 512) — projection dim
        out.append(feats.detach().cpu())
        print(f"    {brand} {season}: {min(i + BATCH_SIZE, len(paths))}/{len(paths)}", end="\r")
    print()
    emb = torch.cat(out, dim=0)
    return emb, paths


def main():
    print(f"[device] {DEVICE}")
    print(f"[model]  loading {MODEL_ID} …")
    model = CLIPModel.from_pretrained(MODEL_ID).to(DEVICE)
    processor = CLIPProcessor.from_pretrained(MODEL_ID)
    model.eval()
    print(f"[model]  image encoder dim: {model.config.projection_dim}")

    t0 = time.time()
    for brand, seasons in PATHS.items():
        for season in seasons:
            # Skip if a matching embedding file already exists (any n).
            # Avoids re-computing Bottega + D&G every time we add a new
            # brand to the PATHS dict.
            existing = list(
                EMBED_DIR.glob(f"FashionCLIP_{slug(brand)}_{season}_n*.pt")
            )
            if existing:
                print(f"  [{brand} {season}] cached: {existing[0].name} — skipping")
                continue
            t_start = time.time()
            emb, paths = embed_brand_season(model, processor, brand, season)
            cache = EMBED_DIR / f"FashionCLIP_{slug(brand)}_{season}_n{len(paths)}.pt"
            torch.save(emb, cache)
            print(f"  [{brand} {season}] {emb.shape} → {cache.name}  ({time.time()-t_start:.1f}s)")

    print(f"\n[done] total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
