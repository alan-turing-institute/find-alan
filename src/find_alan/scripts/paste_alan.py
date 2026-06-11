"""CLI: paste Alan over a detected person and refine the seam.

Detects people in a finished crowd scene, replaces one (random by default)
with the Alan figure scaled to that person, and gently inpaints the seam so the
crowd closes around him without disturbing the wider scene. Writes the
composite plus a JSON sidecar of the geometry to the output directory.

Example:
  find-alan-paste \\
      --input outputs/improve/conference_refined.png \\
      --figure assets/images/Alan.png \\
      --output-dir outputs/finished --seed 7 --save-debug
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys

from find_alan.paste import (
    DEFAULT_FIGURE_PATH,
    DEFAULT_FLUX_MODEL_ID,
    DEFAULT_INPUT_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PASTE_NEGATIVE_PROMPT,
    DEFAULT_PASTE_PROMPT,
    MissingMLDependencies,
    PasteAlanConfig,
    run_paste_alan,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="find-alan-paste",
        description=(
            "Paste the Alan figure over a detected person and refine the seam "
            "with a gentle masked Flux inpaint."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH, help="Scene to paste into.")
    p.add_argument("--figure", type=Path, default=DEFAULT_FIGURE_PATH, help="RGBA figure to paste.")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Where to write results.")
    p.add_argument("--output-name", default="conference_alan", help="Basename for the PNG and JSON.")

    p.add_argument(
        "--strategy",
        default="random",
        choices=["random", "largest", "smallest", "center"],
        help="Which detected person to replace. Default: random.",
    )
    p.add_argument("--conf", type=float, default=0.3, help="YOLO confidence threshold. Default: 0.3.")
    p.add_argument("--yolo-model", default="yolov8n", help="YOLOv8 variant. Default: yolov8n.")
    p.add_argument("--seed", type=int, default=42, help="Seed for target pick and generator. Default: 42.")

    p.add_argument("--gap-padding", type=float, default=0.4, help="Writable gap ring, fraction of figure. Default: 0.4.")
    p.add_argument("--border-padding", type=float, default=0.2, help="Frozen border ring, fraction. Default: 0.2.")
    p.add_argument("--crop-size", type=int, default=512, help="Inference resolution (mult. of 16). Default: 512.")

    p.add_argument("--protect-fraction", type=float, default=0.6, help="Top fraction of the figure kept frozen. Default: 0.6.")
    p.add_argument("--alpha-threshold", type=int, default=128, help="Alpha cutoff for the silhouette. Default: 128.")
    p.add_argument("--dilate", type=int, default=8, help="Grow protected silhouette before feathering. Default: 8.")
    p.add_argument("--feather", type=int, default=8, help="Gaussian feather on the mask. Default: 8.")

    p.add_argument("--model-id", default=DEFAULT_FLUX_MODEL_ID, help="Flux inpaint model id.")
    p.add_argument("--prompt", default=None, help="Override the paste prompt.")
    p.add_argument("--negative-prompt", default=None, help="Override the negative prompt.")
    p.add_argument("--strength", type=float, default=0.4, help="Inpaint strength (gentle). Default: 0.4.")
    p.add_argument("--steps", type=int, default=28, help="Inference steps. Default: 28.")
    p.add_argument("--guidance-scale", type=float, default=3.5, help="CFG scale. Default: 3.5.")
    p.add_argument("--device", default="cuda:0", help="Torch device. Default: cuda:0.")
    p.add_argument("--torch-dtype", default="bfloat16", help="Torch dtype. Default: bfloat16.")
    p.add_argument("--save-debug", action="store_true", help="Also save crop/mask debug images.")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = PasteAlanConfig(
        input_path=args.input,
        figure_path=args.figure,
        output_dir=args.output_dir,
        output_name=args.output_name,
        strategy=args.strategy,
        conf_threshold=args.conf,
        yolo_model=args.yolo_model,
        seed=args.seed,
        gap_padding=args.gap_padding,
        border_padding=args.border_padding,
        crop_size=args.crop_size,
        protect_fraction=args.protect_fraction,
        alpha_threshold=args.alpha_threshold,
        dilate=args.dilate,
        feather=args.feather,
        model_id=args.model_id,
        prompt=args.prompt or DEFAULT_PASTE_PROMPT,
        negative_prompt=args.negative_prompt or DEFAULT_PASTE_NEGATIVE_PROMPT,
        strength=args.strength,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        device=args.device,
        torch_dtype=args.torch_dtype,
        save_debug=args.save_debug,
    )

    try:
        result = run_paste_alan(config)
    except MissingMLDependencies as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(result.output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
