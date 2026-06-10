"""CLI: inpaint a figure into a scene using IP-Adapter + FLUX.1-Fill."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from find_alan.inpaint import (
    DEFAULT_PROMPT,
    load_pipeline,
    run_inpainting,
)
from find_alan.mask import load_mask


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="find-alan-inpaint",
        description=(
            "Inpaint a figure into a scene using IP-Adapter +"
            " FLUX.1-Fill.\n\n"
            "Example (bounding box):\n"
            "  find-alan-inpaint --scene crowd.png --figure person.png"
            " \\\n"
            "    --bbox 220 180 60 120 --output result.png\n\n"
            "Example (explicit mask):\n"
            "  find-alan-inpaint --scene crowd.png --figure person.png"
            " \\\n"
            "    --mask mask.png --output result.png"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument(
        "--scene", required=True, metavar="PATH",
        help="Crowd/background scene image.",
    )
    p.add_argument(
        "--figure", required=True, metavar="PATH",
        help="Reference figure to insert.",
    )
    p.add_argument(
        "--output", required=True, metavar="PATH",
        help="Where to save the result.",
    )

    region = p.add_mutually_exclusive_group(required=True)
    region.add_argument(
        "--mask", metavar="PATH",
        help="Mask image (white=inpaint, black=keep).",
    )
    region.add_argument(
        "--bbox",
        nargs=4,
        type=int,
        metavar=("X", "Y", "W", "H"),
        help="Bounding box of the insertion region in pixels.",
    )

    p.add_argument("--prompt", default=DEFAULT_PROMPT, metavar="TEXT")
    p.add_argument(
        "--ip-scale",
        type=float,
        default=0.6,
        metavar="FLOAT",
        help=(
            "IP-Adapter influence (0=ignore reference, 1=copy reference)."
            " Default: 0.6."
        ),
    )
    p.add_argument(
        "--steps", type=int, default=50, metavar="INT",
        help="Inference steps. Default: 50.",
    )
    p.add_argument(
        "--guidance-scale", type=float, default=30.0, metavar="FLOAT",
        help="CFG scale. FLUX.1-Fill works well at 30. Default: 30.0.",
    )
    p.add_argument(
        "--seed", type=int, default=None, metavar="INT",
        help="Reproducibility seed.",
    )
    p.add_argument(
        "--device",
        default=None,
        metavar="DEVICE",
        help="cuda | mps | cpu  (auto-detected if omitted).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    scene = Image.open(args.scene).convert("RGB")
    figure = Image.open(args.figure).convert("RGB")
    mask = load_mask(
        mask_path=args.mask,
        bbox=tuple(args.bbox) if args.bbox else None,
        image_size=scene.size,
    )

    print("Loading pipeline (FLUX.1-dev ~24 GB, downloaded on first run)...")
    pipe = load_pipeline(device=args.device)

    print("Running inpainting...")
    result = run_inpainting(
        pipe=pipe,
        scene=scene,
        figure=figure,
        mask=mask,
        prompt=args.prompt,
        ip_adapter_scale=args.ip_scale,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path)
    print(f"Saved → {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
