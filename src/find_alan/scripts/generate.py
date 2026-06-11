"""CLI entry point for Waldo-style scene generation.

Single-image mode (recommended) — generate one image from asset prompt files::

    find-alan-generate \\
        --scene-file  assets/prompts/scenes/beach.txt \\
        --style-file  assets/prompts/styles/1-waldo-cartoon.txt \\
        --negative-file assets/prompts/NEGATIVE.txt \\
        --out outputs/beach.png

Tiled mode — stitch multiple tiles using an LLM or fallback prompts::

    find-alan-generate --theme "a medieval jousting tournament" \\
        --llm-url http://localhost:8000/v1 --grid 2x2 --out outputs/market/
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
    DEFAULT_OUTPUT_PATH,
    MissingMLDependencies,
    PromptSet,
    SceneGenerationConfig,
    SingleImageConfig,
    generate_image,
    generate_scene,
    load_negative,
    load_scene_prompt,
    load_style,
    resolve_tile_prompts,
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
        description="Generate a Where's Wally style scene with Flux.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Prompt files (single-image mode) ────────────────────────────────────
    parser.add_argument(
        "--scene-file",
        type=Path,
        default=None,
        metavar="FILE",
        help="Path to a scene description text file (assets/prompts/scenes/*.txt). "
             "When provided, generates a single image instead of a tiled grid.",
    )
    parser.add_argument(
        "--style-file",
        type=Path,
        default=None,
        metavar="FILE",
        help="Path to a style suffix text file (assets/prompts/styles/*.txt). "
             "Defaults to the built-in Waldo cartoon style.",
    )
    parser.add_argument(
        "--negative-file",
        type=Path,
        default=None,
        metavar="FILE",
        help="Path to a negative prompt text file (e.g. assets/prompts/NEGATIVE.txt). "
             "Defaults to the built-in negative prompt.",
    )

    # ── Tiled-mode theme / LLM ───────────────────────────────────────────────
    parser.add_argument(
        "--theme",
        default=_env("SCENE_THEME", "a busy tech conference"),
        help="High-level scene description for LLM-based tile prompts (tiled mode only). "
             "Env: SCENE_THEME",
    )
    parser.add_argument(
        "--llm-url",
        default=_env("LLM_BASE_URL", ""),
        metavar="URL",
        help="Base URL of an OpenAI-compatible LLM endpoint for tiled mode. "
             "Omit to use hard-coded fallback prompts.  Env: LLM_BASE_URL",
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

    # ── Generation knobs ─────────────────────────────────────────────────────
    parser.add_argument(
        "--width",
        type=int,
        default=_env_int("IMAGE_WIDTH", 2048),
        metavar="PX",
        help="Image width in pixels (single-image mode).  Env: IMAGE_WIDTH",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=_env_int("IMAGE_HEIGHT", 2048),
        metavar="PX",
        help="Image height in pixels (single-image mode).  Env: IMAGE_HEIGHT",
    )
    parser.add_argument(
        "--grid",
        default=_env("TILE_GRID", "2x2"),
        metavar="COLSxROWS",
        help='Tile grid for tiled mode, e.g. "2x2".  Env: TILE_GRID',
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=_env_int("TILE_SIZE", 2048),
        metavar="PX",
        help="Width and height of each tile in pixels (tiled mode).  Env: TILE_SIZE",
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
        help="Random seed.  Env: SEED",
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
        help='Passed to from_pretrained as device_map.  Leave blank for '
             'single-GPU via enable_model_cpu_offload.  Env: DEVICE_MAP',
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
        default=None,
        metavar="PATH",
        help="Output path.  For single-image mode: a .png file path "
             f"(default: {DEFAULT_OUTPUT_PATH}).  "
             f"For tiled mode: a directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--no-save-tiles",
        action="store_true",
        help="Do not save individual tile PNGs (tiled mode only).",
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

    device_map: str | None = args.device_map.strip() or None

    # ── Single-image mode ────────────────────────────────────────────────────
    if args.scene_file is not None:
        scene_prompt = load_scene_prompt(args.scene_file)
        style_suffix = load_style(args.style_file) if args.style_file else None
        negative_prompt = load_negative(args.negative_file) if args.negative_file else None

        output_path = args.out if args.out is not None else DEFAULT_OUTPUT_PATH

        cfg_kwargs: dict = dict(
            scene_prompt=scene_prompt,
            output_path=output_path,
            flux_model_id=args.model_id,
            device_map=device_map,
            generator_device=args.generator_device,
            torch_dtype_str=args.torch_dtype,
            width=args.width,
            height=args.height,
            steps=args.steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
        )
        if style_suffix is not None:
            cfg_kwargs["style_suffix"] = style_suffix
        if negative_prompt is not None:
            cfg_kwargs["negative_prompt"] = negative_prompt

        config = SingleImageConfig(**cfg_kwargs)

        try:
            path = generate_image(config)
        except MissingMLDependencies as exc:
            print(str(exc), file=sys.stderr)
            return 2

        print(path)
        return 0

    # ── Tiled mode ───────────────────────────────────────────────────────────
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

    subscenes = resolve_tile_prompts(
        theme=args.theme,
        cols=cols,
        rows=rows,
        llm_base_url=llm_base_url,
        llm_model=args.llm_model,
        llm_timeout=args.llm_timeout,
    )
    prompt_set = PromptSet(
        subscenes=subscenes,
        source="llm" if llm_base_url else "fallback",
    )

    output_dir = args.out if args.out is not None else DEFAULT_OUTPUT_DIR

    tiled_config = SceneGenerationConfig(
        prompt_set=prompt_set,
        output_dir=output_dir,
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
    )

    try:
        result = generate_scene(tiled_config)
    except MissingMLDependencies as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(result.final_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
