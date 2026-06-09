"""Diffusion upscaling starter pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from diffusers import (
    AutoencoderKL,
    ControlNetUnionModel,
    DiffusionPipeline,
    UniPCMultistepScheduler,
)
from PIL import Image


DEFAULT_PROMPT = (
    "dense illustrated crowd scene, crisp ink linework, natural faces, coherent clothing, "
    "high frequency detail, clean edges, richly detailed background"
)
DEFAULT_NEGATIVE_PROMPT = (
    "blurry, pixelated, low resolution, smeared faces, duplicate faces, malformed eyes, "
    "text artifacts, watermark, oversharpened halos"
)


@dataclass(frozen=True)
class DiffusionUpscaleConfig:
    input_path: Path
    output_path: Path
    scale: float = 4.0
    prompt: str = DEFAULT_PROMPT
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT
    model_id: str = "SG161222/RealVisXL_V5.0"
    controlnet_id: str = "OzzyGT/controlnet-union-promax-sdxl-1.0"
    vae_id: str = "madebyollin/sdxl-vae-fp16-fix"
    custom_pipeline: str = "mod_controlnet_tile_sr_sdxl"
    device: str | None = None
    seed: int | None = 1337
    steps: int = 35
    guidance_scale: float = 4.0
    denoising_strength: float = 0.45
    controlnet_strength: float = 1.0
    max_tile_size: int = 1024
    tile_gaussian_sigma: float = 0.3
    overlap: int | None = None
    cpu_offload: bool = True


def _ceil_to_multiple(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def target_size(width: int, height: int, scale: float, multiple: int = 8) -> tuple[int, int]:
    if scale <= 0:
        raise ValueError("scale must be positive")

    return (
        _ceil_to_multiple(round(width * scale), multiple),
        _ceil_to_multiple(round(height * scale), multiple),
    )


def _tile_weighting_method(pipe: Any) -> str:
    method_enum = getattr(pipe, "TileWeightingMethod", None)
    cosine = getattr(method_enum, "COSINE", None) if method_enum else None
    return getattr(cosine, "value", "cosine")


def _overlaps(pipe: Any, width: int, height: int, requested: int | None) -> tuple[int, int]:
    if requested is not None:
        return requested, requested

    calculate_overlap = getattr(pipe, "calculate_overlap", None)
    if calculate_overlap is None:
        return 192, 192

    return calculate_overlap(width, height)


def run_diffusion_upscale(config: DiffusionUpscaleConfig) -> Path:
    device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.startswith("cuda") else torch.float32

    control_image = Image.open(config.input_path).convert("RGB")
    original_width, original_height = control_image.size
    resized_width, resized_height = target_size(original_width, original_height, config.scale)
    image = control_image.resize((resized_width, resized_height), Image.Resampling.LANCZOS)

    controlnet = ControlNetUnionModel.from_pretrained(
        config.controlnet_id,
        torch_dtype=dtype,
        variant="fp16",
        use_safetensors=True,
    ).to(device=device)
    vae = AutoencoderKL.from_pretrained(config.vae_id, torch_dtype=dtype, use_safetensors=True).to(device=device)

    pipe = DiffusionPipeline.from_pretrained(
        config.model_id,
        torch_dtype=dtype,
        vae=vae,
        controlnet=controlnet,
        custom_pipeline=config.custom_pipeline,
        use_safetensors=True,
        variant="fp16" if dtype is torch.float16 else None,
    )

    if config.cpu_offload and device.startswith("cuda") and hasattr(pipe, "enable_model_cpu_offload"):
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)

    if hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    generator = None
    if config.seed is not None:
        generator = torch.Generator(device=device).manual_seed(config.seed)

    normal_overlap, border_overlap = _overlaps(pipe, resized_width, resized_height, config.overlap)

    result = pipe(
        image=image,
        control_image=control_image,
        control_mode=[6],
        controlnet_conditioning_scale=float(config.controlnet_strength),
        prompt=config.prompt,
        negative_prompt=config.negative_prompt,
        normal_tile_overlap=normal_overlap,
        border_tile_overlap=border_overlap,
        height=resized_height,
        width=resized_width,
        original_size=(original_width, original_height),
        target_size=(resized_width, resized_height),
        guidance_scale=float(config.guidance_scale),
        strength=float(config.denoising_strength),
        tile_weighting_method=_tile_weighting_method(pipe),
        max_tile_size=config.max_tile_size,
        tile_gaussian_sigma=float(config.tile_gaussian_sigma),
        num_inference_steps=config.steps,
        generator=generator,
    )

    images = result["images"] if isinstance(result, dict) else result.images
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(config.output_path)
    return config.output_path
