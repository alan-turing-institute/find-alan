"""Run the diffusion upscaler script."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys

from find_alan.upscale import (
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_PROMPT,
    DiffusionUpscaleConfig,
    MissingMLDependencies,
    run_diffusion_upscale,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="find-alan-upscale",
        description="Upscale an image with the Diffusers MoD ControlNet Tile SR SDXL pipeline.",
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--scale", type=float, default=4.0)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--model-id", default="SG161222/RealVisXL_V5.0")
    parser.add_argument("--controlnet-id", default="brad-twinkl/controlnet-union-sdxl-1.0-promax")
    parser.add_argument("--vae-id", default="madebyollin/sdxl-vae-fp16-fix")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--steps", type=int, default=35)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--denoising-strength", type=float, default=0.45)
    parser.add_argument("--controlnet-strength", type=float, default=1.0)
    parser.add_argument("--max-tile-size", type=int, default=1024)
    parser.add_argument("--tile-gaussian-sigma", type=float, default=0.3)
    parser.add_argument("--overlap", type=int, default=None)
    parser.add_argument("--no-cpu-offload", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = DiffusionUpscaleConfig(
        input_path=args.input,
        output_path=args.output,
        scale=args.scale,
        prompt=args.prompt or DEFAULT_PROMPT,
        negative_prompt=args.negative_prompt or DEFAULT_NEGATIVE_PROMPT,
        model_id=args.model_id,
        controlnet_id=args.controlnet_id,
        vae_id=args.vae_id,
        device=args.device,
        seed=args.seed,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        denoising_strength=args.denoising_strength,
        controlnet_strength=args.controlnet_strength,
        max_tile_size=args.max_tile_size,
        tile_gaussian_sigma=args.tile_gaussian_sigma,
        overlap=args.overlap,
        cpu_offload=not args.no_cpu_offload,
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
