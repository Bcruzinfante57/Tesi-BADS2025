#!/usr/bin/env python3
"""
select_highlights.py — pick the editorial highlights from the F25/S26 time
series and copy the relevant image files into the frontend repo's public/
folder so the magazine section can render them.

Selection per brand:
  - 8 "novedades destacadas" : products in new_in_s26 with the LOWEST
    best_cos_sim (most distinct from anything in F25 = most genuinely new)
  - 8 "discontinuados"        : products in discontinued_from_f25 with the
    LOWEST best_cos_sim (most absent in S26 = most removed)

Also copies the cluster_data_v2.json bundle into public/ and writes a
small highlights.json so the frontend can render the curated set without
having to compute rankings client-side.
"""

import json
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).parent
FRONTEND_PUBLIC = REPO_ROOT.parent / "conan-insight-hub" / "public"
OUT_DIR = FRONTEND_PUBLIC / "snapshots"

# Brand → (snapshot → source dir)
IMAGE_DIRS = {
    "Bottega Veneta": {
        "F25": REPO_ROOT / "images_bottega",
        "S26": REPO_ROOT / "snapshots/S26/raw/bottega",
    },
    "Dolce & Gabbana": {
        "F25": REPO_ROOT / "images_D&G",
        "S26": REPO_ROOT / "snapshots/S26/raw/dg",
    },
}

# URL-safe folder slugs the frontend will use
BRAND_SLUGS = {
    "Bottega Veneta":  "bottega",
    "Dolce & Gabbana": "dg",
}

N_HIGHLIGHTS = 8

BUNDLE_SRC = REPO_ROOT / "snapshots" / "cluster_data_v2.json"


def main():
    bundle = json.loads(BUNDLE_SRC.read_text())
    highlights: dict = {"brands": {}}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    total_copied = 0

    for brand, snaps in IMAGE_DIRS.items():
        slug = BRAND_SLUGS[brand]
        brand_block = bundle["brands"][brand]
        match = brand_block["matching"]

        # Novelties — sort by best_cos_sim ASC (most distinct first)
        novelties = sorted(match["new_s26_products"], key=lambda x: x["best_cos_sim"])[:N_HIGHLIGHTS]
        # Discontinued — sort by best_cos_sim ASC (most absent first)
        discontinued = sorted(match["discontinued_f25_products"], key=lambda x: x["best_cos_sim"])[:N_HIGHLIGHTS]

        # Index S26 / F25 products by image filename for quick lookup of palette + price
        s26_index = {p["image"]: p for p in brand_block["snapshots"]["S26"]["products"]}
        f25_index = {p["image"]: p for p in brand_block["snapshots"]["F25"]["products"]}

        brand_highlights = {
            "novelties": [],
            "discontinued": [],
            "approach": brand_block.get("approach"),  # placeholder — filled in below per brand
        }

        # Copy novelty images (from S26)
        for n in novelties:
            img_name = n["s26_img"]
            src = snaps["S26"] / img_name
            dst_dir = OUT_DIR / "S26" / slug
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / img_name
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)
                total_copied += 1
            prod = s26_index.get(img_name, {})
            brand_highlights["novelties"].append({
                "image": f"snapshots/S26/{slug}/{img_name}",
                "filename": img_name,
                "price_eur": prod.get("price_eur"),
                "palette": prod.get("palette", [])[:5],
                "best_cos_sim": n["best_cos_sim"],
            })

        # Copy discontinued images (from F25)
        for d in discontinued:
            img_name = d["f25_img"]
            src = snaps["F25"] / img_name
            dst_dir = OUT_DIR / "F25" / slug
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / img_name
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)
                total_copied += 1
            prod = f25_index.get(img_name, {})
            brand_highlights["discontinued"].append({
                "image": f"snapshots/F25/{slug}/{img_name}",
                "filename": img_name,
                "price_eur": prod.get("price_eur"),
                "palette": prod.get("palette", [])[:5],
                "best_cos_sim": d["best_cos_sim"],
            })

        highlights["brands"][brand] = brand_highlights

    # Write highlights.json next to the bundle in public/
    (FRONTEND_PUBLIC / "highlights.json").write_text(json.dumps(highlights, indent=2))

    # Also copy the full bundle into public/ for the frontend to consume directly
    shutil.copy2(BUNDLE_SRC, FRONTEND_PUBLIC / "cluster_data_v2.json")

    print(f"[done] copied {total_copied} highlight images → {OUT_DIR}")
    print(f"[done] wrote {FRONTEND_PUBLIC / 'highlights.json'}")
    print(f"[done] copied bundle → {FRONTEND_PUBLIC / 'cluster_data_v2.json'}")

    # Sanity summary
    for brand, h in highlights["brands"].items():
        print(f"\n=== {brand} ===")
        print(f"  novelties: {len(h['novelties'])}")
        for n in h["novelties"][:3]:
            colors = ", ".join(c["name_bucket"] for c in n["palette"][:3])
            print(f"    {n['filename']:20s} €{n['price_eur'] or '?':>4}  colors: {colors}")
        print(f"  discontinued: {len(h['discontinued'])}")
        for d in h["discontinued"][:3]:
            colors = ", ".join(c["name_bucket"] for c in d["palette"][:3])
            print(f"    {d['filename']:20s} €{d['price_eur'] or '?':>4}  colors: {colors}")


if __name__ == "__main__":
    main()
