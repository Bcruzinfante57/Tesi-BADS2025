#!/usr/bin/env python3
"""
color_pipeline_v2.py — Per-product palette extraction with U²-Net + LAB + KMeans.

Replaces the v1 dominant_colors_t which ran KMeans k=3 in RGB on a 30x30 thumbnail
of the cluster's representative images. v1 collapsed minority colors (e.g. blue
lenses on a metal frame) into a neighbor centroid. v2 fixes this with:

  1. Background removal via U²-Net (ONNX, uses cached ~/.u2net/u2net.onnx)
  2. KMeans in LAB color space (perceptually uniform — ΔE distances)
  3. k=12 then filter coverage < 1.5% and merge centroids with ΔE_CIE76 < 5
  4. Per-product (not per-cluster) — every product contributes its full palette

Run:
    python color_pipeline_v2.py --brand Prada --limit 5
    python color_pipeline_v2.py --brand all --limit 0  --output palettes_full.json

Output JSON shape:
    {
      "<brand>": {
        "<image_filename>": {
          "palette": [
            {"hex": "#1a1a1a", "rgb": [26, 26, 26], "lab": [10.5, 0.1, -0.2],
             "coverage": 0.42, "name_bucket": "black"},
            ...
          ]
        }
      }
    }
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image, ImageFilter

REPO_ROOT = Path(__file__).parent

BRANDS = {
    "Dolce & Gabbana": "images_D&G",
    "Bottega Veneta":  "images_bottega",
    "Cartier":         "images_Cartier",
    "Fendi":           "images_Fendi",
    "Prada":           "images_Prada",
    "YSL":             "images_ysl",
}

# Pipeline hyperparameters
PROCESS_SIZE       = 256     # resize for color analysis (preserves minority regions)
U2NET_SIZE         = 320     # U²-Net trained input size
KMEANS_K           = 12      # over-segment then merge
KMEANS_ITERS       = 30
KMEANS_INITS       = 3
ALPHA_THRESHOLD    = 0.7     # foreground if mask > this (0.7 cuts edge bleed from bilinear resize)
MASK_ERODE_PX      = 0       # morphological erosion — disabled; ate into thin frames more than into edge bleed
MIN_COVERAGE       = 0.015   # drop centroids covering < 1.5% of product pixels
DELTA_E_MERGE      = 5.0     # merge centroids with ΔE_CIE76 < this (perceptually identical)

# Background-residue filter: drop centroids that are essentially pure white in LAB.
# Luxury eyewear frames are almost never near-pure-white, so these are reliably bg
# leaking through the U²-Net mask. Catches Prada_20-style failures.
BG_WHITE_L_MIN     = 92.0    # CIE LAB L*; pure white = 100
BG_WHITE_AB_MAX    = 4.0     # |a*|, |b*| chromaticity bound

U2NET_MODEL_PATH   = Path.home() / ".u2net" / "u2net.onnx"

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# RGB <-> LAB conversion (torch implementation, no skimage)
# Uses sRGB D65 white point. Standard CIE 1976 LAB formulas.
# ─────────────────────────────────────────────────────────────────────────────

def srgb_to_linear(rgb: torch.Tensor) -> torch.Tensor:
    """sRGB gamma correction. Input in [0, 1]."""
    return torch.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(lin: torch.Tensor) -> torch.Tensor:
    return torch.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin.clamp_min(1e-12) ** (1 / 2.4) - 0.055)


# sRGB → XYZ (D65) matrix
_M_RGB2XYZ = torch.tensor([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
], dtype=torch.float32)

_M_XYZ2RGB = torch.tensor([
    [ 3.2404542, -1.5371385, -0.4985314],
    [-0.9692660,  1.8760108,  0.0415560],
    [ 0.0556434, -0.2040259,  1.0572252],
], dtype=torch.float32)

# D65 reference white
_XN, _YN, _ZN = 0.95047, 1.0, 1.08883


def _lab_f(t: torch.Tensor) -> torch.Tensor:
    delta = 6 / 29
    return torch.where(t > delta ** 3, t.clamp_min(1e-12) ** (1 / 3), t / (3 * delta ** 2) + 4 / 29)


def _lab_f_inv(t: torch.Tensor) -> torch.Tensor:
    delta = 6 / 29
    return torch.where(t > delta, t ** 3, 3 * delta ** 2 * (t - 4 / 29))


def rgb_to_lab(rgb: torch.Tensor) -> torch.Tensor:
    """rgb: (..., 3) in [0, 255]. Returns (..., 3) LAB (L in [0,100], a/b ~ [-128,127])."""
    rgb = rgb.float() / 255.0
    lin = srgb_to_linear(rgb)
    M = _M_RGB2XYZ.to(rgb.device, rgb.dtype)
    xyz = lin @ M.T
    fx = _lab_f(xyz[..., 0] / _XN)
    fy = _lab_f(xyz[..., 1] / _YN)
    fz = _lab_f(xyz[..., 2] / _ZN)
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return torch.stack([L, a, b], dim=-1)


def lab_to_rgb(lab: torch.Tensor) -> torch.Tensor:
    """lab: (..., 3). Returns (..., 3) RGB in [0, 255] (int-friendly)."""
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16) / 116
    fx = a / 500 + fy
    fz = fy - b / 200
    xyz = torch.stack([_XN * _lab_f_inv(fx), _YN * _lab_f_inv(fy), _ZN * _lab_f_inv(fz)], dim=-1)
    M = _M_XYZ2RGB.to(lab.device, lab.dtype)
    lin = xyz @ M.T
    rgb = linear_to_srgb(lin.clamp(0, None))
    return (rgb.clamp(0, 1) * 255).round()


# ─────────────────────────────────────────────────────────────────────────────
# U²-Net background removal via ONNX runtime
# ─────────────────────────────────────────────────────────────────────────────

_ort_session = None


def _get_u2net():
    global _ort_session
    if _ort_session is not None:
        return _ort_session
    if not U2NET_MODEL_PATH.exists():
        raise FileNotFoundError(f"u2net.onnx not found at {U2NET_MODEL_PATH}")
    import onnxruntime as ort
    providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"] \
        if "CoreMLExecutionProvider" in ort.get_available_providers() \
        else ["CPUExecutionProvider"]
    _ort_session = ort.InferenceSession(str(U2NET_MODEL_PATH), providers=providers)
    return _ort_session


def remove_background(img: Image.Image) -> torch.Tensor:
    """Returns RGBA tensor (H, W, 4) at PROCESS_SIZE × PROCESS_SIZE."""
    sess = _get_u2net()
    work = img.convert("RGB").resize((U2NET_SIZE, U2NET_SIZE), Image.BILINEAR)
    arr = torch.tensor(list(work.getdata()), dtype=torch.float32).view(U2NET_SIZE, U2NET_SIZE, 3) / 255.0

    # U²-Net standard normalization
    mean = torch.tensor([0.485, 0.456, 0.406])
    std  = torch.tensor([0.229, 0.224, 0.225])
    inp = (arr - mean) / std
    inp = inp.permute(2, 0, 1).unsqueeze(0).numpy()  # (1, 3, 320, 320)

    out = sess.run(None, {sess.get_inputs()[0].name: inp})[0]  # (1, 1, 320, 320)
    mask = torch.tensor(out[0, 0])  # (320, 320)
    # Normalize mask to [0, 1] (model may output unnormalized scores)
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)

    # Build RGBA at PROCESS_SIZE.  Avoid numpy.astype (broken under scipy fallout)
    # by routing the mask through PIL natively.
    base = img.convert("RGB").resize((PROCESS_SIZE, PROCESS_SIZE), Image.BILINEAR)
    rgb = torch.tensor(list(base.getdata()), dtype=torch.float32).view(PROCESS_SIZE, PROCESS_SIZE, 3)
    mask_u8 = (mask * 255).clamp(0, 255).to(torch.uint8).contiguous()
    mask_pil = Image.frombytes("L", (U2NET_SIZE, U2NET_SIZE), bytes(mask_u8.flatten().tolist()))
    mask_resized = mask_pil.resize((PROCESS_SIZE, PROCESS_SIZE), Image.BILINEAR)

    # Threshold to binary, then erode to push the boundary away from edge bleed.
    # MinFilter with size (2*N+1) erodes by N pixels in each direction.
    threshold_u8 = int(ALPHA_THRESHOLD * 255)
    binary = mask_resized.point(lambda p: 255 if p >= threshold_u8 else 0, mode="L")
    if MASK_ERODE_PX > 0:
        binary = binary.filter(ImageFilter.MinFilter(2 * MASK_ERODE_PX + 1))

    alpha = torch.tensor(list(binary.getdata()), dtype=torch.float32).view(PROCESS_SIZE, PROCESS_SIZE) / 255.0
    return torch.cat([rgb, alpha.unsqueeze(-1) * 255], dim=-1)  # (H, W, 4)


# ─────────────────────────────────────────────────────────────────────────────
# KMeans in LAB
# ─────────────────────────────────────────────────────────────────────────────

def kmeans_lab(pixels: torch.Tensor, k: int, n_iters: int = KMEANS_ITERS,
               n_inits: int = KMEANS_INITS) -> tuple[torch.Tensor, torch.Tensor]:
    """pixels: (N, 3) LAB. Returns (centroids (k, 3), labels (N,))."""
    n = pixels.shape[0]
    k = min(k, n)
    best_lbl, best_centers, best_inertia = None, None, float("inf")

    for _ in range(n_inits):
        # k-means++ init
        idx0 = torch.randint(n, (1,)).item()
        centers = [pixels[idx0]]
        for _ in range(1, k):
            stacked = torch.stack(centers, dim=0)        # (m, 3)
            dists = torch.cdist(pixels, stacked).min(dim=1).values  # (N,)
            probs = (dists ** 2) / (dists ** 2).sum().clamp_min(1e-12)
            idx = torch.multinomial(probs, 1).item()
            centers.append(pixels[idx])
        centers = torch.stack(centers, dim=0)

        for _ in range(n_iters):
            D = torch.cdist(pixels, centers)
            lbl = D.argmin(1)
            new_centers = torch.stack([
                pixels[lbl == c].mean(0) if (lbl == c).any() else centers[c]
                for c in range(k)
            ])
            if torch.allclose(centers, new_centers, atol=1e-3):
                centers = new_centers
                break
            centers = new_centers

        # Inertia
        D = torch.cdist(pixels, centers)
        lbl = D.argmin(1)
        inertia = sum(((pixels[lbl == c] - centers[c]) ** 2).sum().item() for c in range(k))
        if inertia < best_inertia:
            best_inertia, best_lbl, best_centers = inertia, lbl.clone(), centers.clone()

    return best_centers, best_lbl


# ─────────────────────────────────────────────────────────────────────────────
# Merge perceptually identical centroids
# ─────────────────────────────────────────────────────────────────────────────

def merge_close_centroids(centroids: torch.Tensor, coverage: torch.Tensor,
                          delta_e: float = DELTA_E_MERGE) -> tuple[torch.Tensor, torch.Tensor]:
    """Iteratively merge the two closest (in LAB) centroids if ΔE < delta_e.
    Coverage is added; new centroid is the coverage-weighted average."""
    centers = centroids.clone()
    cov = coverage.clone()
    while centers.shape[0] > 1:
        D = torch.cdist(centers, centers)
        # Mask diagonal
        D.fill_diagonal_(float("inf"))
        min_val, min_idx = D.view(-1).min(0)
        if min_val.item() >= delta_e:
            break
        i, j = (min_idx // D.shape[0]).item(), (min_idx % D.shape[0]).item()
        if i > j:
            i, j = j, i
        # Weighted merge
        w_i, w_j = cov[i].item(), cov[j].item()
        merged = (centers[i] * w_i + centers[j] * w_j) / (w_i + w_j)
        centers[i] = merged
        cov[i] = w_i + w_j
        # Drop j
        keep = torch.ones(centers.shape[0], dtype=torch.bool)
        keep[j] = False
        centers = centers[keep]
        cov = cov[keep]
    return centers, cov


# ─────────────────────────────────────────────────────────────────────────────
# Color name buckets (subset of frontend hexToColorName mapping, sufficient for
# cross-cluster counting). Maps LAB centroid → english bucket name.
# ─────────────────────────────────────────────────────────────────────────────

def name_color_bucket(rgb: tuple[int, int, int]) -> str:
    r, g, bl = (c / 255 for c in rgb)
    mx, mn = max(r, g, bl), min(r, g, bl)
    L = (mx + mn) / 2
    d = mx - mn

    if d < 0.10:
        if L < 0.10: return "black"
        if L < 0.40: return "dark_grey"
        if L < 0.65: return "grey"
        if L < 0.85: return "light_grey"
        return "white"

    if mx == r:
        h = ((g - bl) / d + (6 if g < bl else 0))
    elif mx == g:
        h = (bl - r) / d + 2
    else:
        h = (r - g) / d + 4
    h *= 60

    if L < 0.12: return "black"
    if h < 18 or h >= 340:
        return "burgundy" if L < 0.30 else ("pink" if L > 0.65 else "red")
    if h < 45:
        if L < 0.32: return "brown"
        if L > 0.72: return "peach"
        return "tortoise" if L < 0.50 else "orange"
    if h < 70:  return "gold" if L < 0.50 else "yellow"
    if h < 150: return "dark_green" if L < 0.32 else "green"
    if h < 185: return "turquoise"
    if h < 250: return "navy" if L < 0.30 else "blue"
    if h < 290: return "violet"
    return "magenta"


# ─────────────────────────────────────────────────────────────────────────────
# Main per-image pipeline
# ─────────────────────────────────────────────────────────────────────────────

def extract_palette(img_path: Path) -> list[dict]:
    img = Image.open(img_path)
    rgba = remove_background(img)  # (H, W, 4) RGB + alpha 0-255

    rgb_px = rgba[..., :3].reshape(-1, 3)
    alpha  = rgba[..., 3].reshape(-1)
    fg_mask = alpha > (ALPHA_THRESHOLD * 255)

    if fg_mask.sum().item() < 100:
        return []  # foreground too small / mask failed

    product_rgb = rgb_px[fg_mask]
    # Subsample to keep KMeans tractable
    if product_rgb.shape[0] > 8000:
        idx = torch.randperm(product_rgb.shape[0])[:8000]
        product_rgb = product_rgb[idx]

    lab_px = rgb_to_lab(product_rgb)
    centers, labels = kmeans_lab(lab_px, KMEANS_K)

    # Coverage per centroid
    n_total = labels.shape[0]
    coverage = torch.tensor([(labels == c).sum().item() / n_total for c in range(centers.shape[0])])

    # Drop low-coverage
    keep = coverage >= MIN_COVERAGE
    if keep.sum().item() == 0:
        return []
    centers = centers[keep]
    coverage = coverage[keep]

    # Drop near-pure-white centroids (almost certainly background residue, not product).
    L = centers[:, 0]
    a = centers[:, 1]
    b = centers[:, 2]
    not_bg = ~((L >= BG_WHITE_L_MIN) & (a.abs() <= BG_WHITE_AB_MAX) & (b.abs() <= BG_WHITE_AB_MAX))
    if not_bg.sum().item() == 0:
        return []
    centers = centers[not_bg]
    coverage = coverage[not_bg]

    coverage = coverage / coverage.sum()  # renormalize after drops

    # Merge perceptually identical
    centers, coverage = merge_close_centroids(centers, coverage, DELTA_E_MERGE)

    # Sort by coverage desc
    order = torch.argsort(coverage, descending=True)
    centers = centers[order]
    coverage = coverage[order]

    palette = []
    rgb_centers = lab_to_rgb(centers).to(torch.int32)
    for i in range(centers.shape[0]):
        r, g, b = rgb_centers[i].tolist()
        L_, a_, b_ = centers[i].tolist()
        palette.append({
            "hex": f"#{r:02x}{g:02x}{b:02x}",
            "rgb": [int(r), int(g), int(b)],
            "lab": [round(L_, 2), round(a_, 2), round(b_, 2)],
            "coverage": round(coverage[i].item(), 4),
            "name_bucket": name_color_bucket((r, g, b)),
        })
    return palette


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brand", default="all", help="Brand name or 'all' (default)")
    ap.add_argument("--limit", type=int, default=5, help="Max images per brand (0 = all)")
    ap.add_argument("--output", default="palettes_v2.json", help="Output JSON path")
    args = ap.parse_args()

    brands = BRANDS if args.brand == "all" else {args.brand: BRANDS[args.brand]}

    out = {}
    t_start = time.time()
    for brand, folder in brands.items():
        folder_path = REPO_ROOT / folder
        if not folder_path.exists():
            print(f"[skip] {brand}: folder not found at {folder_path}", file=sys.stderr)
            continue
        images = sorted(p for p in folder_path.glob("*.jpg")) + sorted(p for p in folder_path.glob("*.png"))
        if args.limit > 0:
            images = images[: args.limit]
        print(f"[{brand}] processing {len(images)} images …")
        out[brand] = {}
        for i, p in enumerate(images, 1):
            t0 = time.time()
            try:
                palette = extract_palette(p)
                out[brand][p.name] = {"palette": palette}
                print(f"  [{i}/{len(images)}] {p.name}: {len(palette)} colors  ({time.time()-t0:.2f}s)")
                for c in palette:
                    print(f"      {c['hex']}  {c['coverage']*100:5.1f}%  {c['name_bucket']}")
            except Exception as e:
                print(f"  [err] {p.name}: {e}", file=sys.stderr)
                out[brand][p.name] = {"palette": [], "error": str(e)}

    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"\n✓ wrote {args.output}  ({time.time()-t_start:.1f}s total)")


if __name__ == "__main__":
    main()
