"""Experimental Stable Diffusion 3 tiled ControlNet upscaler."""

from __future__ import annotations

import random
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .tiling import Crop, jittered_grid
from .upscale import MissingMLDependencies, target_size

if TYPE_CHECKING:
    from .upscale import DiffusionUpscaleConfig


def _import_sd3() -> dict[str, Any]:
    try:
        import numpy as np
        import torch
        from diffusers import SD3ControlNetModel, StableDiffusion3ControlNetPipeline
        from PIL import Image
    except ImportError as exc:
        raise MissingMLDependencies(
            "Install the optional image stack with `uv sync --extra ml`. "
            "The SD3 tiled engine also downloads SD3 and ControlNet weights on first run."
        ) from exc

    return {
        "np": np,
        "torch": torch,
        "SD3ControlNetModel": SD3ControlNetModel,
        "StableDiffusion3ControlNetPipeline": StableDiffusion3ControlNetPipeline,
        "Image": Image,
    }


def _ceil_to_multiple(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _sd3_prompt(config: "DiffusionUpscaleConfig") -> str:
    prompt = (
        f"{config.prompt}. Faithfully upscale and redraw the provided reference crop as "
        "one seamless high-resolution tile. Preserve the exact composition, object "
        "placement, viewpoint, linework, color palette, and crowd layout."
    )
    return prompt


def _gaussian_weight(height: int, width: int, sigma: float, np: Any) -> Any:
    if sigma <= 0:
        return np.ones((height, width, 1), dtype=np.float32)

    y = np.linspace(-1, 1, height, dtype=np.float32).reshape(height, 1)
    x = np.linspace(-1, 1, width, dtype=np.float32).reshape(1, width)
    distance = x * x + y * y
    weight = np.exp(-distance / (2 * sigma * sigma)).astype(np.float32)
    return np.maximum(weight, 1e-3)[..., None]


def _progress_bar(total: int) -> Any | None:
    if total <= 0:
        return None

    try:
        from tqdm.auto import tqdm
    except ImportError:
        return None

    return tqdm(total=total, desc="SD3 tiles", unit="tile", dynamic_ncols=True)


def _tile_grid(
    width: int,
    height: int,
    tile_size: int,
    overlap: int,
    seed: int | None,
    jitter: int | None,
) -> list[Crop]:
    rng = random.Random(seed)
    return jittered_grid(
        width=width,
        height=height,
        tile_size=tile_size,
        overlap=overlap,
        rng=rng,
        jitter=jitter,
    )


def _save_blended_canvas(
    canvas: Any,
    weights: Any,
    output_path: Path,
    Image: Any,
    np: Any,
) -> Path:
    image_array = canvas / np.maximum(weights, 1e-6)
    image_array = np.clip(image_array * 255.0, 0, 255).astype(np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_array).save(output_path)
    return output_path


def run_sd3_tile_upscale(config: "DiffusionUpscaleConfig") -> Path:
    ml = _import_sd3()
    np = ml["np"]
    torch = ml["torch"]
    Image = ml["Image"]

    device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.startswith("cuda") else torch.float32

    source_image = Image.open(config.input_path).convert("RGB")
    original_width, original_height = source_image.size
    resized_width, resized_height = target_size(
        original_width,
        original_height,
        config.scale,
        multiple=16,
    )
    resized_image = source_image.resize(
        (resized_width, resized_height),
        Image.Resampling.LANCZOS,
    )

    tile_size = _ceil_to_multiple(max(16, config.sd3_tile_size), 16)
    overlap = min(max(0, config.sd3_overlap), tile_size - 1)
    crops = _tile_grid(
        width=resized_width,
        height=resized_height,
        tile_size=tile_size,
        overlap=overlap,
        seed=config.seed,
        jitter=config.sd3_jitter,
    )

    Pipeline = ml["StableDiffusion3ControlNetPipeline"]
    if config.controlnet_id and config.controlnet_id != config.model_id:
        ControlNetModel = ml["SD3ControlNetModel"]
        controlnet = ControlNetModel.from_pretrained(
            config.controlnet_id,
            torch_dtype=dtype,
            use_safetensors=True,
        )
        pipe = Pipeline.from_pretrained(
            config.model_id,
            controlnet=controlnet,
            torch_dtype=dtype,
            use_safetensors=True,
            variant="fp16" if dtype is torch.float16 else None,
        )
    else:
        pipe = Pipeline.from_pretrained(
            config.model_id,
            torch_dtype=dtype,
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

    if hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()
    elif hasattr(getattr(pipe, "vae", None), "enable_tiling"):
        pipe.vae.enable_tiling()
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    elif hasattr(getattr(pipe, "vae", None), "enable_slicing"):
        pipe.vae.enable_slicing()

    generator = None
    if config.seed is not None:
        generator = torch.Generator(device=device).manual_seed(config.seed)

    prompt = _sd3_prompt(config)
    canvas = np.zeros((resized_height, resized_width, 3), dtype=np.float32)
    weights = np.zeros((resized_height, resized_width, 1), dtype=np.float32)
    progress = _progress_bar(len(crops))

    try:
        for crop in crops:
            reference_tile = resized_image.crop(
                (crop.x, crop.y, crop.right, crop.bottom)
            )
            result = pipe(
                prompt=prompt,
                negative_prompt=config.negative_prompt,
                control_image=reference_tile,
                controlnet_conditioning_scale=float(config.controlnet_strength),
                height=crop.height,
                width=crop.width,
                num_inference_steps=config.steps,
                max_sequence_length=config.sd3_max_sequence_length,
                generator=generator,
                output_type="pil",
            )
            images = result["images"] if isinstance(result, dict) else result.images
            tile = images[0].convert("RGB")
            if tile.size != (crop.width, crop.height):
                tile = tile.resize((crop.width, crop.height), Image.Resampling.LANCZOS)

            tile_array = np.asarray(tile, dtype=np.float32) / 255.0
            weight = _gaussian_weight(
                crop.height,
                crop.width,
                config.tile_gaussian_sigma,
                np,
            )
            canvas[crop.y : crop.bottom, crop.x : crop.right] += tile_array * weight
            weights[crop.y : crop.bottom, crop.x : crop.right] += weight

            if progress is not None:
                progress.update(1)
    finally:
        if progress is not None:
            progress.close()

    return _save_blended_canvas(canvas, weights, config.output_path, Image, np)
