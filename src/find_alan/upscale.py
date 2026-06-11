"""Diffusion upscaling entry points."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .constants import UPSCALE_NEGATIVE_PROMPT_01, UPSCALE_PROMPT_01


DEFAULT_PROMPT = UPSCALE_PROMPT_01
DEFAULT_NEGATIVE_PROMPT = UPSCALE_NEGATIVE_PROMPT_01
DEFAULT_MOD_MODEL_ID = "SG161222/RealVisXL_V5.0"
DEFAULT_FLUX2_MODEL_ID = "black-forest-labs/FLUX.2-dev"
DEFAULT_SD3_MODEL_ID = "stabilityai/stable-diffusion-3.5-large"
DEFAULT_MOD_CONTROLNET_ID = "OzzyGT/controlnet-union-promax-sdxl-1.0"
DEFAULT_MULTIDIFFUSION_CONTROLNET_ID = "xinsir/controlnet-tile-sdxl-1.0"
DEFAULT_SD3_CONTROLNET_ID = "stabilityai/stable-diffusion-3.5-large-controlnet-blur"


class MissingMLDependencies(RuntimeError):
    """Raised when optional ML dependencies are not installed."""


@dataclass(frozen=True)
class DiffusionUpscaleConfig:
    input_path: Path
    output_path: Path
    scale: float = 4.0
    prompt: str = DEFAULT_PROMPT
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT
    model_id: str = DEFAULT_MOD_MODEL_ID
    controlnet_id: str = DEFAULT_MOD_CONTROLNET_ID
    vae_id: str = "madebyollin/sdxl-vae-fp16-fix"
    custom_pipeline: str = "mod_controlnet_tile_sr_sdxl"
    engine: str = "mod-tile"
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
    multidiffusion_tile_size: int = 1024
    multidiffusion_overlap: int = 512
    multidiffusion_jitter: int | None = None
    multidiffusion_view_batch_size: int = 1
    flux2_tile_size: int = 1024
    flux2_overlap: int = 256
    flux2_jitter: int | None = None
    flux2_md_fusion: str = "weighted"
    flux2_md_anneal_fraction: float = 0.35
    flux2_pipeline: str = "auto"
    flux2_max_sequence_length: int = 512
    flux2_caption_upsample_temperature: float | None = None
    sd3_tile_size: int = 1024
    sd3_overlap: int = 256
    sd3_jitter: int | None = None
    sd3_control_guidance_start: float = 0.0
    sd3_control_guidance_end: float = 1.0
    sd3_max_sequence_length: int = 256


def _ceil_to_multiple(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def target_size(
    width: int, height: int, scale: float, multiple: int = 8
) -> tuple[int, int]:
    if scale <= 0:
        raise ValueError("scale must be positive")

    return (
        _ceil_to_multiple(round(width * scale), multiple),
        _ceil_to_multiple(round(height * scale), multiple),
    )


def _import_mod_pipeline() -> dict[str, Any]:
    try:
        import torch
        from diffusers import (
            AutoencoderKL,
            ControlNetUnionModel,
            DiffusionPipeline,
            UniPCMultistepScheduler,
        )
        from PIL import Image
    except ImportError as exc:
        raise MissingMLDependencies(
            "Install the optional image stack with `uv sync --extra ml`. "
            "For CUDA-specific PyTorch wheels, install torch from the PyTorch index first."
        ) from exc

    return {
        "torch": torch,
        "AutoencoderKL": AutoencoderKL,
        "ControlNetUnionModel": ControlNetUnionModel,
        "DiffusionPipeline": DiffusionPipeline,
        "UniPCMultistepScheduler": UniPCMultistepScheduler,
        "Image": Image,
    }


def _tile_weighting_method(pipe: Any) -> str:
    method_enum = getattr(pipe, "TileWeightingMethod", None)
    cosine = getattr(method_enum, "COSINE", None) if method_enum else None
    return getattr(cosine, "value", "cosine")


def _overlaps(
    pipe: Any, width: int, height: int, requested: int | None
) -> tuple[int, int]:
    if requested is not None:
        return requested, requested

    calculate_overlap = getattr(pipe, "calculate_overlap", None)
    if calculate_overlap is None:
        return 192, 192

    return calculate_overlap(width, height)


def _run_mod_tile_upscale(config: DiffusionUpscaleConfig) -> Path:
    ml = _import_mod_pipeline()
    torch = ml["torch"]
    Image = ml["Image"]
    AutoencoderKL = ml["AutoencoderKL"]
    ControlNetUnionModel = ml["ControlNetUnionModel"]
    DiffusionPipeline = ml["DiffusionPipeline"]
    UniPCMultistepScheduler = ml["UniPCMultistepScheduler"]

    device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.startswith("cuda") else torch.float32

    control_image = Image.open(config.input_path).convert("RGB")
    original_width, original_height = control_image.size
    resized_width, resized_height = target_size(
        original_width, original_height, config.scale
    )
    image = control_image.resize(
        (resized_width, resized_height), Image.Resampling.LANCZOS
    )

    controlnet = ControlNetUnionModel.from_pretrained(
        config.controlnet_id,
        torch_dtype=dtype,
        variant="fp16" if dtype is torch.float16 else None,
        use_safetensors=True,
    ).to(device=device)
    vae = AutoencoderKL.from_pretrained(
        config.vae_id, torch_dtype=dtype, use_safetensors=True
    ).to(device=device)

    pipe = DiffusionPipeline.from_pretrained(
        config.model_id,
        torch_dtype=dtype,
        vae=vae,
        controlnet=controlnet,
        custom_pipeline=config.custom_pipeline,
        use_safetensors=True,
        variant="fp16" if dtype is torch.float16 else None,
    )

    if (
        config.cpu_offload
        and device.startswith("cuda")
        and hasattr(pipe, "enable_model_cpu_offload")
    ):
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)

    if hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()
    elif hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()
    if hasattr(pipe.vae, "enable_slicing"):
        pipe.vae.enable_slicing()
    elif hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    generator = None
    if config.seed is not None:
        generator = torch.Generator(device=device).manual_seed(config.seed)

    normal_overlap, border_overlap = _overlaps(
        pipe, resized_width, resized_height, config.overlap
    )

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


def run_diffusion_upscale(config: DiffusionUpscaleConfig) -> Path:
    if config.engine == "mod-tile":
        return _run_mod_tile_upscale(config)
    if config.engine == "multidiffusion":
        from .experimental_multidiffusion import run_multidiffusion_upscale

        return run_multidiffusion_upscale(config)
    if config.engine == "flux2-tile":
        from .experimental_flux2_tile import run_flux2_tile_upscale

        return run_flux2_tile_upscale(config)
    if config.engine == "flux2-multidiffusion":
        from .experimental_flux2_multidiffusion import run_flux2_multidiffusion_upscale

        return run_flux2_multidiffusion_upscale(config)
    if config.engine == "sd3-tile":
        from .experimental_sd3_tile import run_sd3_tile_upscale

        return run_sd3_tile_upscale(config)

    raise ValueError(f"Unknown upscale engine: {config.engine}")
