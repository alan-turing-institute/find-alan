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
    MissingMLDependencies,
    PasteAlanConfig,
    run_paste_alan,
)

# Single source of truth for every default: the dataclass. The CLI reads its
# defaults from here, so editing PasteAlanConfig changes the CLI too — no more
# argparse defaults silently shadowing the dataclass.
_DEFAULTS = PasteAlanConfig()


class _Formatter(
    argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter
):
    """Show each argument's default and keep the raw description layout."""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="find-alan-paste",
        description=(
            "Paste the Alan figure over a detected person and refine the seam "
            "with a gentle masked Flux inpaint."
        ),
        formatter_class=_Formatter,
    )
    p.add_argument("--input", type=Path, default=_DEFAULTS.input_path, help="Scene to paste into.")
    p.add_argument("--figure", type=Path, default=_DEFAULTS.figure_path, help="RGBA figure to paste.")
    p.add_argument("--output-dir", type=Path, default=_DEFAULTS.output_dir, help="Where to write results.")
    p.add_argument("--output-name", default=_DEFAULTS.output_name, help="Basename for the PNG and JSON.")

    p.add_argument(
        "--strategy",
        default=_DEFAULTS.strategy,
        choices=["random", "largest", "smallest", "center"],
        help="Which detected person to replace.",
    )
    p.add_argument("--conf", type=float, default=_DEFAULTS.conf_threshold, help="YOLO confidence threshold.")
    p.add_argument("--yolo-model", default=_DEFAULTS.yolo_model, help="YOLOv8 variant (x needed for tiny crowd figures).")
    p.add_argument("--seed", type=int, default=_DEFAULTS.seed, help="Seed for target pick and generator. Omit for a random figure each run.")
    p.add_argument("--edge-margin", type=float, default=_DEFAULTS.edge_margin, help="Edge fraction where the figure may not land.")
    p.add_argument("--logo-zone", type=float, default=_DEFAULTS.logo_zone, help="Top-left fraction reserved for a logo.")

    p.add_argument("--figure-scale", type=float, default=_DEFAULTS.figure_scale, help="Scale Alan relative to the bbox height (boxes clip).")
    p.add_argument("--gap-padding", type=float, default=_DEFAULTS.gap_padding, help="Writable gap ring, fraction of figure.")
    p.add_argument("--border-padding", type=float, default=_DEFAULTS.border_padding, help="Frozen border ring, fraction.")
    p.add_argument("--crop-size", type=int, default=_DEFAULTS.crop_size, help="Inference resolution (mult. of 16).")

    p.add_argument("--protect-fraction", type=float, default=_DEFAULTS.protect_fraction, help="Fraction of the figure (from top) kept frozen; 1.0 = whole silhouette incl. legs.")
    p.add_argument("--alpha-threshold", type=int, default=_DEFAULTS.alpha_threshold, help="Alpha cutoff for the silhouette.")
    p.add_argument("--dilate", type=int, default=_DEFAULTS.dilate, help="Grow protected silhouette before feathering.")
    p.add_argument("--feather", type=int, default=_DEFAULTS.feather, help="Gaussian feather on the mask.")

    p.add_argument("--model-id", default=_DEFAULTS.model_id, help="Flux inpaint model id.")
    p.add_argument("--prompt", default=_DEFAULTS.prompt, help="Override the paste prompt.")
    p.add_argument("--negative-prompt", default=_DEFAULTS.negative_prompt, help="Override the negative prompt.")
    p.add_argument("--strength", type=float, default=_DEFAULTS.strength, help="Inpaint strength = noise fraction in the writable region.")
    p.add_argument("--steps", type=int, default=_DEFAULTS.steps, help="Inference steps.")
    p.add_argument("--refine-passes", type=int, default=_DEFAULTS.refine_passes, help="Repeat the noise/denoise refinement this many times.")
    p.add_argument("--guidance-scale", type=float, default=_DEFAULTS.guidance_scale, help="CFG scale; lower follows surroundings more.")
    p.add_argument("--gap-blur", type=float, default=_DEFAULTS.gap_blur, help="Pre-blur the writable gap before refining (0 = off). Try ~8 with higher strength.")
    p.add_argument("--device", default=_DEFAULTS.device, help="Torch device.")
    p.add_argument("--torch-dtype", default=_DEFAULTS.torch_dtype, help="Torch dtype.")
    p.add_argument("--save-debug", action="store_true", default=_DEFAULTS.save_debug, help="Also save crop/mask debug images.")
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
        edge_margin=args.edge_margin,
        logo_zone=args.logo_zone,
        figure_scale=args.figure_scale,
        gap_padding=args.gap_padding,
        border_padding=args.border_padding,
        crop_size=args.crop_size,
        protect_fraction=args.protect_fraction,
        alpha_threshold=args.alpha_threshold,
        dilate=args.dilate,
        feather=args.feather,
        model_id=args.model_id,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        strength=args.strength,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        refine_passes=args.refine_passes,
        gap_blur=args.gap_blur,
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
