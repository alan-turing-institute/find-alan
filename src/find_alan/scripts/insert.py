"""CLI: insert a figure into a scene using FLUX.2-Klein (no mask needed)."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from find_alan.insert import DEFAULT_PROMPT, load_pipeline, run_insertion


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="find-alan-insert",
        description=(
            "Insert a reference figure into a crowd scene using FLUX.2-Klein.\n\n"
            "No mask required — the model places the figure based on the prompt\n"
            "and the scene context.\n\n"
            "Example:\n"
            "  find-alan-insert --scene crowd.png --figure person.png"
            " --output result.png"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument(
        "--scene", required=True, metavar="PATH",
        help="Crowd/background scene image.",
    )
    p.add_argument(
        "--figure", required=True, metavar="PATH",
        help="Reference figure image to insert.",
    )
    p.add_argument(
        "--output", required=True, metavar="PATH",
        help="Where to save the result.",
    )

    p.add_argument(
        "--prompt", default=DEFAULT_PROMPT, metavar="TEXT",
        help="Text prompt guiding figure placement and blending.",
    )
    p.add_argument(
        "--strength", type=float, default=0.85, metavar="FLOAT",
        help=(
            "How much to modify the scene (0–1). "
            "Lower values preserve more of the original. Default: 0.85."
        ),
    )
    p.add_argument(
        "--steps", type=int, default=50, metavar="INT",
        help="Inference steps. Default: 50.",
    )
    p.add_argument(
        "--guidance-scale", type=float, default=8.0, metavar="FLOAT",
        help="Guidance scale. Default: 8.0.",
    )
    p.add_argument(
        "--seed", type=int, default=None, metavar="INT",
        help="Reproducibility seed.",
    )
    p.add_argument(
        "--device", default=None, metavar="DEVICE",
        help="cuda | mps | cpu  (auto-detected if omitted).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    scene = Image.open(args.scene).convert("RGB")
    figure = Image.open(args.figure).convert("RGB")

    print(
        "Loading pipeline (FLUX.2-Klein ~13 GB, downloaded on first run)..."
    )
    pipe = load_pipeline(device=args.device)

    print("Running insertion...")
    result = run_insertion(
        pipe=pipe,
        scene=scene,
        figure=figure,
        prompt=args.prompt,
        strength=args.strength,
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
