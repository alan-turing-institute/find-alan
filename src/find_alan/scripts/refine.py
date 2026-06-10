"""Run tiled image refinement scripts."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import os
from pathlib import Path
import sys

from find_alan.refinement import (
    DEFAULT_FLUX_MODEL_ID,
    DEFAULT_INPUT_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_REFINEMENT_NEGATIVE_PROMPT,
    DEFAULT_REFINEMENT_PROMPT,
    MissingMLDependencies,
    TiledRefinementConfig,
    run_tiled_refinement,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="find-alan-refine",
        description="Refine an image with full-coverage Flux inpainting tiles.",
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=_env_path("INPUT_IMAGE", DEFAULT_INPUT_PATH),
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        type=Path,
        default=_env_path("OUTPUT_DIR", DEFAULT_OUTPUT_DIR),
    )
    parser.add_argument("--model-id", default=DEFAULT_FLUX_MODEL_ID)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cuda:0"))
    parser.add_argument("--torch-dtype", default=os.environ.get("TORCH_DTYPE", "bfloat16"))
    parser.add_argument("--outer-size", type=int, default=_env_int("OUTER_SIZE", 512))
    parser.add_argument(
        "--inner-ratio",
        type=float,
        default=_env_float("INNER_RATIO", 0.5),
    )
    parser.add_argument("--feather", type=int, default=_env_int("FEATHER", 4))
    parser.add_argument(
        "--iterations",
        "--num-iters",
        dest="iterations",
        type=int,
        default=_env_int("NUM_ITERS", 4),
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=_env_int("MAX_BATCH_SIZE", 12),
    )
    parser.add_argument("--strength", type=float, default=_env_float("STRENGTH", 0.2))
    parser.add_argument("--steps", type=int, default=_env_int("NUM_STEPS", 28))
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=_env_float("GUIDANCE_SCALE", 3.5),
    )
    parser.add_argument("--seed", type=int, default=_env_int("BASE_SEED", 42))
    parser.add_argument(
        "--gif-frame-duration",
        type=int,
        default=_env_int("GIF_FRAME_DURATION", 400),
    )
    parser.add_argument(
        "--visualization-width",
        type=int,
        default=_env_int("VIZ_WIDTH", 800),
    )
    parser.add_argument("--no-gif", action="store_true")
    parser.add_argument("--no-comparison", action="store_true")
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on the first failed mini-batch instead of continuing.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = TiledRefinementConfig(
        input_path=args.input,
        output_dir=args.output_dir,
        model_id=args.model_id,
        prompt=args.prompt or DEFAULT_REFINEMENT_PROMPT,
        negative_prompt=args.negative_prompt or DEFAULT_REFINEMENT_NEGATIVE_PROMPT,
        outer_size=args.outer_size,
        inner_ratio=args.inner_ratio,
        feather=args.feather,
        iterations=args.iterations,
        max_batch_size=args.max_batch_size,
        strength=args.strength,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        device=args.device,
        torch_dtype=args.torch_dtype,
        gif_frame_duration=args.gif_frame_duration,
        visualization_width=args.visualization_width,
        save_gif=not args.no_gif,
        save_comparison=not args.no_comparison,
        continue_on_batch_error=not args.fail_fast,
    )

    try:
        result = run_tiled_refinement(config)
    except MissingMLDependencies as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(result.final_path)
    return 0


def _env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default)))


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


if __name__ == "__main__":
    raise SystemExit(main())
