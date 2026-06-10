"""CLI: generate synthetic example images for testing the inpainting pipeline.

These are simple PIL-drawn images — sufficient to verify the pipeline runs
end-to-end.  Replace them with real photos for meaningful visual results.

Output (written to ./examples/ by default):
  crowd_scene.png  — 512×512 crowd of simple figures
  figure.png       — 256×384 single reference figure (red jacket)
  mask.png         — 512×512 mask with a gap in the crowd for the figure
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _draw_stick_figure(
    draw: ImageDraw.ImageDraw,
    cx: int,
    cy: int,
    scale: float,
    body_color: tuple[int, int, int],
    skin_color: tuple[int, int, int],
) -> None:
    """Draw a minimal person: head + torso oval at (cx, cy)."""
    s = scale
    # torso
    draw.ellipse(
        [cx - int(9 * s), cy - int(18 * s), cx + int(9 * s), cy + int(12 * s)],
        fill=body_color,
    )
    # head
    draw.ellipse(
        [cx - int(6 * s), cy - int(30 * s), cx + int(6 * s), cy - int(18 * s)],
        fill=skin_color,
    )


def make_crowd_scene(width: int = 512, height: int = 512, seed: int = 0) -> Image.Image:
    """Return a synthetic crowd scene image."""
    rng = random.Random(seed)
    img = Image.new("RGB", (width, height), (185, 165, 130))  # warm sandy background
    draw = ImageDraw.Draw(img)

    # Sky gradient strip
    for y in range(height // 3):
        v = int(160 + y * 0.3)
        draw.line([(0, y), (width, y)], fill=(v, v + 20, v + 40))

    shirt_palette = [
        (210, 80, 60), (60, 120, 200), (240, 200, 60), (80, 160, 80),
        (180, 80, 160), (60, 180, 180), (220, 140, 60), (100, 100, 200),
    ]
    skin_tones = [
        (220, 180, 140), (200, 155, 115), (170, 120, 80), (230, 195, 160),
    ]

    n_rows = 7
    for row in range(n_rows):
        # Figures get smaller toward the top (perspective).
        t = row / (n_rows - 1)
        scale = 0.5 + 0.55 * t
        y_base = int(height * (0.38 + 0.55 * t))
        n_cols = int(12 + 6 * t)
        spacing = width / n_cols

        for col in range(n_cols):
            cx = int(spacing * (col + 0.5) + rng.uniform(-spacing * 0.15, spacing * 0.15))
            cy = int(y_base + rng.uniform(-8, 8) * scale)
            _draw_stick_figure(
                draw, cx, cy, scale,
                body_color=rng.choice(shirt_palette),
                skin_color=rng.choice(skin_tones),
            )

    return img


def make_figure(width: int = 256, height: int = 384) -> Image.Image:
    """Return a synthetic single-figure reference image (red jacket)."""
    img = Image.new("RGB", (width, height), (200, 220, 195))  # light green bg
    draw = ImageDraw.Draw(img)

    cx = width // 2
    # Legs
    draw.rectangle([cx - 22, height // 2 + 10, cx - 6, height - 40], fill=(40, 40, 120))
    draw.rectangle([cx + 6, height // 2 + 10, cx + 22, height - 40], fill=(40, 40, 120))
    # Feet
    draw.ellipse([cx - 26, height - 50, cx - 2, height - 35], fill=(60, 40, 30))
    draw.ellipse([cx + 2, height - 50, cx + 26, height - 35], fill=(60, 40, 30))
    # Torso (red jacket)
    draw.rectangle([cx - 35, height // 2 - 60, cx + 35, height // 2 + 15], fill=(200, 50, 50))
    # Arms
    draw.rectangle([cx - 55, height // 2 - 55, cx - 32, height // 2 + 10], fill=(200, 50, 50))
    draw.rectangle([cx + 32, height // 2 - 55, cx + 55, height // 2 + 10], fill=(200, 50, 50))
    # Neck
    draw.rectangle([cx - 8, height // 2 - 85, cx + 8, height // 2 - 60], fill=(220, 180, 140))
    # Head
    draw.ellipse([cx - 35, height // 2 - 145, cx + 35, height // 2 - 80], fill=(220, 180, 140))
    # Hair
    draw.ellipse([cx - 35, height // 2 - 145, cx + 35, height // 2 - 110], fill=(80, 50, 30))

    return img


def make_mask(
    scene_width: int,
    scene_height: int,
    bbox: tuple[int, int, int, int],
) -> Image.Image:
    """Return a black mask with a white rectangle at *bbox* (x, y, w, h)."""
    mask = Image.new("L", (scene_width, scene_height), 0)
    x, y, w, h = bbox
    ImageDraw.Draw(mask).rectangle([x, y, x + w, y + h], fill=255)
    return mask


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="find-alan-prepare-examples",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--out-dir",
        default="examples",
        metavar="DIR",
        help="Directory to write example images into. Default: ./examples/",
    )
    p.add_argument("--seed", type=int, default=0, help="RNG seed for crowd layout.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    scene_w, scene_h = 512, 512
    # Leave a gap near the bottom-centre for the inserted figure.
    bbox = (210, 330, 90, 150)

    scene = make_crowd_scene(scene_w, scene_h, seed=args.seed)
    figure = make_figure()
    mask = make_mask(scene_w, scene_h, bbox)

    scene.save(out / "crowd_scene.png")
    figure.save(out / "figure.png")
    mask.save(out / "mask.png")

    bx, by, bw, bh = bbox
    print(f"Written to {out}/")
    print(f"  crowd_scene.png  ({scene_w}×{scene_h})")
    print(f"  figure.png       ({figure.width}×{figure.height})")
    print(f"  mask.png         (gap at x={bx} y={by} w={bw} h={bh})")
    print()
    print("Run inpainting with:")
    print(
        f"  find-alan-inpaint "
        f"--scene {out}/crowd_scene.png "
        f"--figure {out}/figure.png "
        f"--mask {out}/mask.png "
        f"--output {out}/result.png"
    )
    print()
    print("Or with a bounding box instead of mask:")
    print(
        f"  find-alan-inpaint "
        f"--scene {out}/crowd_scene.png "
        f"--figure {out}/figure.png "
        f"--bbox {bx} {by} {bw} {bh} "
        f"--output {out}/result.png"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
