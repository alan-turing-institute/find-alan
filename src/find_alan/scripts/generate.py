"""CLI entry point for tiled Waldo-style scene generation.

Usage examples::

    # Fallback prompts (no LLM needed)
    find-alan-generate

    # Free-form theme, LLM-generated sub-scene prompts
    find-alan-generate --theme "a medieval jousting tournament" \\
        --llm-url http://localhost:8000/v1

    # Different grid, bigger tiles, custom output dir
    find-alan-generate --theme "a busy airport" \\
        --llm-url http://localhost:8000/v1 \\
        --grid 3x2 --tile-size 1024 --out outputs/airport

    # Reproducible run
    find-alan-generate --theme "a christmas market" --seed 99

All settings can also be supplied via environment variables (see --help).
"""

from __future__ import annotations

import argparse
import logging
import os
from collections.abc import Sequence
from pathlib import Path
import sys

from find_alan.scene_generate import (
    DEFAULT_FLUX_MODEL_ID,
    DEFAULT_LLM_MODEL,
    DEFAULT_OUTPUT_DIR,
    MissingMLDependencies,
    SceneGenerationConfig,
    generate_scene,
)


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="find-alan-generate",
        description="Generate a tiled Where's Wally style scene with Flux.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Scene / prompt ───────────────────────────────────────────────────────
    parser.add_argument(
        "--theme",
        default=_env("SCENE_THEME", "a busy tech conference"),
        help="High-level scene description passed to the LLM.  "
             "Ignored when --llm-url is not set (fallback prompts are used). "
             "Env: SCENE_THEME",
    )

    # ── LLM ─────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--llm-url",
        default=_env("LLM_BASE_URL", ""),
        metavar="URL",
        help="Base URL of an OpenAI-compatible LLM endpoint "
             "(e.g. http://localhost:8000/v1).  "
             "Omit (or leave blank) to use hard-coded fallback prompts.  "
             "Env: LLM_BASE_URL",
    )
    parser.add_argument(
        "--llm-model",
        default=_env("LLM_MODEL", DEFAULT_LLM_MODEL),
        help="Model name sent in the LLM request body.  Env: LLM_MODEL",
    )
    parser.add_argument(
        "--llm-timeout",
        type=float,
        default=_env_float("LLM_TIMEOUT", 60.0),
        metavar="SECS",
        help="HTTP timeout for the LLM request.  Env: LLM_TIMEOUT",
    )

    # ── Grid / generation ────────────────────────────────────────────────────
    parser.add_argument(
        "--grid",
        default=_env("TILE_GRID", "2x2"),
        metavar="COLSxROWS",
        help='Tile grid, e.g. "2x2" or "3x2".  Env: TILE_GRID',
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=_env_int("TILE_SIZE", 2048),
        metavar="PX",
        help="Width and height of each tile in pixels.  Env: TILE_SIZE",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=_env_int("NUM_STEPS", 28),
        help="Flux denoising steps.  Env: NUM_STEPS",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=_env_float("GUIDANCE_SCALE", 3.5),
        help="Flux guidance scale.  Env: GUIDANCE_SCALE",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=_env_int("SEED", 42),
        help="Base random seed (each tile uses seed+i).  Env: SEED",
    )

    # ── Model / hardware ─────────────────────────────────────────────────────
    parser.add_argument(
        "--model-id",
        default=_env("FLUX_MODEL_ID", DEFAULT_FLUX_MODEL_ID),
        help="HuggingFace repo ID for the Flux model.  Env: FLUX_MODEL_ID",
    )
    parser.add_argument(
        "--device-map",
        default=_env("DEVICE_MAP", ""),
        help='Passed to from_pretrained as device_map.  Leave blank (default) '
             'for single-GPU via enable_model_cpu_offload.  '
             'Use "balanced" to spread across multiple GPUs.  '
             "Env: DEVICE_MAP",
    )
    parser.add_argument(
        "--generator-device",
        default=_env("GENERATOR_DEVICE", "cuda:0"),
        help="Device used for the torch.Generator seed.  Env: GENERATOR_DEVICE",
    )
    parser.add_argument(
        "--torch-dtype",
        default=_env("TORCH_DTYPE", "bfloat16"),
        choices=["bfloat16", "float16"],
        help="Floating-point dtype for the model.  Env: TORCH_DTYPE",
    )

    # ── Output ───────────────────────────────────────────────────────────────
    parser.add_argument(
        "--out",
        type=Path,
        default=_env_path("OUTPUT_DIR", DEFAULT_OUTPUT_DIR),
        dest="output_dir",
        metavar="DIR",
        help="Directory to write tiles and the final stitched image.  Env: OUTPUT_DIR",
    )
    parser.add_argument(
        "--no-save-tiles",
        action="store_true",
        help="Do not save individual tile PNGs (only the final stitched image).",
    )

    # ── Logging ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG logging.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    # Parse grid
    try:
        cols_str, rows_str = args.grid.lower().split("x")
        cols, rows = int(cols_str), int(rows_str)
        if cols < 1 or rows < 1:
            raise ValueError
    except ValueError:
        print(
            f"error: --grid must be COLSxROWS with positive integers, got {args.grid!r}",
            file=sys.stderr,
        )
        return 2

    llm_base_url: str | None = args.llm_url.strip() or None
    device_map: str | None = args.device_map.strip() or None

    config = SceneGenerationConfig(
        theme=args.theme,
        output_dir=args.output_dir,
        flux_model_id=args.model_id,
        device_map=device_map,
        generator_device=args.generator_device,
        torch_dtype_str=args.torch_dtype,
        cols=cols,
        rows=rows,
        tile_size=args.tile_size,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        save_tiles=not args.no_save_tiles,
        llm_base_url=llm_base_url,
        llm_model=args.llm_model,
        llm_timeout=args.llm_timeout,
    )

    try:
        result = generate_scene(config)
    except MissingMLDependencies as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(result.final_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
