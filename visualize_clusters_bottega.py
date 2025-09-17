import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def safe_imports():
    try:
        import pandas as pd  # noqa: F401
        from PIL import Image, ImageDraw, ImageFont  # noqa: F401
        import math  # noqa: F401
        import colorsys  # noqa: F401
    except ModuleNotFoundError as e:
        missing = str(e).split("No module named ")[-1].strip("'\"")
        print(
            f"Missing dependency: {missing}.\n"
            "Please install: pip install pandas pillow",
        )
        raise


def load_assignments(assign_csv: Path) -> List[Tuple[str, int]]:
    import pandas as pd

    df = pd.read_csv(assign_csv)
    if "path" not in df.columns or "cluster" not in df.columns:
        raise ValueError("Expected columns 'path' and 'cluster' in assignments CSV")
    pairs = [(str(p), int(c)) for p, c in zip(df["path"], df["cluster"])]
    # filter out noise
    pairs = [pc for pc in pairs if pc[1] >= 0]
    return pairs


def build_palette(n: int) -> List[Tuple[int, int, int]]:
    # Generate n distinct colors using HSV space
    cols = []
    for i in range(max(1, n)):
        h = (i / max(1, n)) % 1.0
        s = 0.65
        v = 0.95
        r, g, b = [int(255 * c) for c in __import__("colorsys").hsv_to_rgb(h, s, v)]
        cols.append((r, g, b))
    return cols


def readable_text_color(rgb: Tuple[int, int, int]) -> Tuple[int, int, int]:
    r, g, b = rgb
    # Perceived luminance
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return (0, 0, 0) if lum > 186 else (255, 255, 255)


def make_cluster_panel(
    image_paths: List[Path],
    title: str,
    color: Tuple[int, int, int],
    thumb_size: Tuple[int, int],
    cols: int,
    pad: int = 10,
    header_h: int = 36,
    thumb_border: int = 4,
) -> Optional["Image.Image"]:
    from PIL import Image, ImageDraw, ImageFont
    import math

    if not image_paths:
        return None

    w, h = thumb_size
    rows = math.ceil(len(image_paths) / cols)
    panel_w = cols * (w + pad) + pad
    panel_h = header_h + rows * (h + pad) + pad
    panel = Image.new("RGB", (panel_w, panel_h), (255, 255, 255))
    draw = ImageDraw.Draw(panel)

    # Header bar
    draw.rectangle([0, 0, panel_w, header_h], fill=color)
    txt_col = readable_text_color(color)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    draw.text((8, 6), title, fill=txt_col, font=font)

    # Paste thumbnails with colored borders per cluster
    x0, y0 = pad, header_h + pad
    c = 0
    for p in image_paths:
        try:
            with Image.open(p) as im:
                im = im.convert("RGB")
                im.thumbnail((w, h))
                # Center within slot
                slot = Image.new("RGB", (w, h), (250, 250, 250))
                sx = (w - im.width) // 2
                sy = (h - im.height) // 2
                slot.paste(im, (sx, sy))
        except Exception:
            # Placeholder on error
            slot = Image.new("RGB", (w, h), (220, 220, 220))
            d2 = ImageDraw.Draw(slot)
            d2.line([(0, 0), (w, h)], fill=(150, 150, 150), width=2)
            d2.line([(0, h), (w, 0)], fill=(150, 150, 150), width=2)

        cx = c % cols
        cy = c // cols
        px = x0 + cx * (w + pad)
        py = y0 + cy * (h + pad)
        panel.paste(slot, (px, py))
        # Draw colored border around each thumbnail for clear cluster differentiation
        if thumb_border > 0:
            for b in range(thumb_border):
                draw.rectangle(
                    [px - b, py - b, px + w + b - 1, py + h + b - 1],
                    outline=color,
                    width=1,
                )
        c += 1

    # Colored border around panel
    draw.rectangle([0, 0, panel_w - 1, panel_h - 1], outline=color, width=3)
    return panel


def arrange_panels(panels: List["Image.Image"], per_row: int, pad: int = 12) -> Optional["Image.Image"]:
    from PIL import Image
    if not panels:
        return None
    widths = [p.width for p in panels]
    heights = [p.height for p in panels]
    max_w = max(widths)
    max_h = max(heights)

    rows = (len(panels) + per_row - 1) // per_row
    out_w = per_row * (max_w + pad) + pad
    out_h = rows * (max_h + pad) + pad
    canvas = Image.new("RGB", (out_w, out_h), (255, 255, 255))

    for i, p in enumerate(panels):
        r = i // per_row
        c = i % per_row
        x = pad + c * (max_w + pad)
        y = pad + r * (max_h + pad)
        # Center panel in its slot
        ox = x + (max_w - p.width) // 2
        oy = y + (max_h - p.height) // 2
        canvas.paste(p, (ox, oy))
    return canvas


def main():
    safe_imports()
    import math
    from PIL import Image

    parser = argparse.ArgumentParser(description="Visualize clusters as colored panels of thumbnails")
    parser.add_argument("--assignments", type=str, nargs="*", default=[], help="One or more cluster assignments CSVs")
    parser.add_argument("--auto", action="store_true", help="Auto-discover clusters_best*.csv in current directory")
    parser.add_argument("--out", type=str, default="", help="Output image path (used only if one input file)")
    parser.add_argument("--thumb-w", type=int, default=180)
    parser.add_argument("--thumb-h", type=int, default=180)
    parser.add_argument("--thumb-border", type=int, default=4, help="Border width around each thumbnail (cluster color)")
    parser.add_argument("--cols", type=int, default=8, help="Thumbnails per row inside a cluster panel")
    parser.add_argument("--per-row", type=int, default=3, help="Cluster panels per row in final image")
    parser.add_argument("--max-per-cluster", type=int, default=0, help="Limit images per cluster (0 = no limit)")
    args = parser.parse_args()

    files: List[Path] = [Path(p) for p in args.assignments]
    if args.auto or not files:
        # discover default outputs from clustering script
        here = Path.cwd()
        auto = sorted(here.glob("clusters_best*.csv"))
        files = auto if auto else files
    if not files:
        raise SystemExit("No assignment CSVs provided or discovered.")

    multiple = len(files) > 1
    for f in files:
        pairs = load_assignments(f)
        if not pairs:
            print(f"Skip {f}: no assignments (or all noise)")
            continue

        # Group by cluster
        by_cluster: Dict[int, List[Path]] = {}
        for p, c in pairs:
            by_cluster.setdefault(c, []).append(Path(p))
        clusters = sorted(by_cluster.keys())

        colors = build_palette(len(clusters))
        panels = []
        for idx, c in enumerate(clusters):
            paths = by_cluster[c]
            if args.max_per_cluster and len(paths) > args.max_per_cluster:
                paths = paths[: args.max_per_cluster]
            title = f"{f.stem} | Cluster {c}  (n={len(by_cluster[c])})"
            panel = make_cluster_panel(
                paths,
                title=title,
                color=colors[idx % len(colors)],
                thumb_size=(args.thumb_w, args.thumb_h),
                cols=args.cols,
                thumb_border=max(0, int(args.thumb_border)),
            )
            if panel is not None:
                panels.append(panel)

        grid = arrange_panels(panels, per_row=max(1, args.per_row))
        if grid is None:
            print(f"Skip {f}: nothing to draw.")
            continue

        if args.out and not multiple:
            out = Path(args.out)
        else:
            # derive output per file
            out = f.with_name(f"clusters_montage_{f.stem}.jpg")
        grid.save(out, format="JPEG")
        print(f"Saved montage to {out}")


if __name__ == "__main__":
    main()
