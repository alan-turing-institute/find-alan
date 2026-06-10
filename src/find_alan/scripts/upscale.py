"""Run image upscaling scripts."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys

from find_alan.upscale import (
    DEFAULT_FLUX2_MODEL_ID,
    DEFAULT_MOD_MODEL_ID,
    DEFAULT_MOD_CONTROLNET_ID,
    DEFAULT_MULTIDIFFUSION_CONTROLNET_ID,
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_PROMPT,
    DiffusionUpscaleConfig,
    MissingMLDependencies,
    run_diffusion_upscale,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="find-alan-upscale",
        description="Upscale an image with tiled diffusion, MultiDiffusion, or Flux.2 tiles.",
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--engine",
        choices=("mod-tile", "multidiffusion", "flux2-tile"),
        default="mod-tile",
    )
    parser.add_argument("--scale", type=float, default=4.0)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--controlnet-id", default=None)
    parser.add_argument("--vae-id", default="madebyollin/sdxl-vae-fp16-fix")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--steps", type=int, default=35)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--denoising-strength", type=float, default=None)
    parser.add_argument(
        "--strong-denoise",
        action="store_true",
        help="Use a stronger default denoising strength when --denoising-strength is omitted.",
    )
    parser.add_argument("--controlnet-strength", type=float, default=1.0)
    parser.add_argument("--max-tile-size", type=int, default=1024)
    parser.add_argument("--tile-gaussian-sigma", type=float, default=0.3)
    parser.add_argument("--overlap", type=int, default=None)
    parser.add_argument("--md-tile-size", type=int, default=1024)
    parser.add_argument("--md-overlap", type=int, default=512)
    parser.add_argument("--md-jitter", type=int, default=None)
    parser.add_argument("--md-view-batch-size", type=int, default=1)
    parser.add_argument("--flux2-tile-size", type=int, default=1024)
    parser.add_argument("--flux2-overlap", type=int, default=256)
    parser.add_argument("--flux2-jitter", type=int, default=None)
    parser.add_argument(
        "--flux2-pipeline",
        choices=("auto", "dev", "klein", "klein-kv"),
        default="auto",
    )
    parser.add_argument("--flux2-max-sequence-length", type=int, default=512)
    parser.add_argument(
        "--flux2-caption-upsample-temperature",
        type=float,
        default=None,
    )
    parser.add_argument("--no-cpu-offload", action="store_true")
    return parser


def _model_id(args: argparse.Namespace) -> str:
    if args.model_id:
        return args.model_id
    if args.engine == "flux2-tile":
        return DEFAULT_FLUX2_MODEL_ID
    return DEFAULT_MOD_MODEL_ID


def _controlnet_id(args: argparse.Namespace) -> str:
    if args.controlnet_id:
        return args.controlnet_id
    if args.engine == "multidiffusion":
        return DEFAULT_MULTIDIFFUSION_CONTROLNET_ID
    return DEFAULT_MOD_CONTROLNET_ID


def _denoising_strength(args: argparse.Namespace) -> float:
    if args.denoising_strength is not None:
        return args.denoising_strength
    if args.strong_denoise:
        return 0.72
    if args.engine == "multidiffusion":
        return 0.62
    return 0.45


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = DiffusionUpscaleConfig(
        input_path=args.input,
        output_path=args.output,
        scale=args.scale,
        prompt=args.prompt or DEFAULT_PROMPT,
        negative_prompt=args.negative_prompt or DEFAULT_NEGATIVE_PROMPT,
        model_id=_model_id(args),
        controlnet_id=_controlnet_id(args),
        vae_id=args.vae_id,
        engine=args.engine,
        device=args.device,
        seed=args.seed,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        denoising_strength=_denoising_strength(args),
        controlnet_strength=args.controlnet_strength,
        max_tile_size=args.max_tile_size,
        tile_gaussian_sigma=args.tile_gaussian_sigma,
        overlap=args.overlap,
        cpu_offload=not args.no_cpu_offload,
        multidiffusion_tile_size=args.md_tile_size,
        multidiffusion_overlap=args.md_overlap,
        multidiffusion_jitter=args.md_jitter,
        multidiffusion_view_batch_size=args.md_view_batch_size,
        flux2_tile_size=args.flux2_tile_size,
        flux2_overlap=args.flux2_overlap,
        flux2_jitter=args.flux2_jitter,
        flux2_pipeline=args.flux2_pipeline,
        flux2_max_sequence_length=args.flux2_max_sequence_length,
        flux2_caption_upsample_temperature=args.flux2_caption_upsample_temperature,
    )

    try:
        output_path = run_diffusion_upscale(config)
    except MissingMLDependencies as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
