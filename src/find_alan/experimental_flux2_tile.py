"""Experimental Flux.2 tiled reference upscaler."""

from __future__ import annotations

import inspect
from pathlib import Path
import random
from typing import TYPE_CHECKING, Any

from .tiling import Crop, jittered_grid
from .upscale import (
    DEFAULT_FLUX2_MODEL_ID,
    DEFAULT_MOD_MODEL_ID,
    MissingMLDependencies,
    target_size,
)

if TYPE_CHECKING:
    from .upscale import DiffusionUpscaleConfig


def _import_flux2() -> dict[str, Any]:
    try:
        import numpy as np
        import torch
        from diffusers import Flux2KleinKVPipeline, Flux2KleinPipeline, Flux2Pipeline
        from PIL import Image
    except ImportError as exc:
        raise MissingMLDependencies(
            "Install the optional image stack with `uv sync --extra ml`. "
            "The flux2-tile engine also downloads FLUX.2 weights on first run."
        ) from exc

    return {
        "np": np,
        "torch": torch,
        "Flux2KleinKVPipeline": Flux2KleinKVPipeline,
        "Flux2KleinPipeline": Flux2KleinPipeline,
        "Flux2Pipeline": Flux2Pipeline,
        "Image": Image,
    }


def _ceil_to_multiple(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _flux2_model_id(config: "DiffusionUpscaleConfig") -> str:
    if config.model_id == DEFAULT_MOD_MODEL_ID:
        return DEFAULT_FLUX2_MODEL_ID
    return config.model_id


def _pipeline_class(ml: dict[str, Any], model_id: str, requested: str) -> Any:
    if requested not in {"auto", "dev", "klein", "klein-kv"}:
        raise ValueError("--flux2-pipeline must be one of: auto, dev, klein, klein-kv")

    if requested == "dev":
        return ml["Flux2Pipeline"]
    if requested == "klein":
        return ml["Flux2KleinPipeline"]
    if requested == "klein-kv":
        return ml["Flux2KleinKVPipeline"]

    normalized_model_id = model_id.lower()
    if "klein" in normalized_model_id and "kv" in normalized_model_id:
        return ml["Flux2KleinKVPipeline"]
    if "klein" in normalized_model_id:
        return ml["Flux2KleinPipeline"]
    return ml["Flux2Pipeline"]


def _pipeline_call_parameters(pipe: Any) -> set[str]:
    return set(inspect.signature(pipe.__call__).parameters)


def _flux2_prompt(config: "DiffusionUpscaleConfig") -> str:
    prompt = (
        f"{config.prompt}. Faithfully upscale and redraw the provided reference crop as "
        "one seamless high-resolution tile. Preserve the exact composition, object "
        "placement, viewpoint, linework, color palette, and crowd layout. Do not add "
        "borders, frames, captions, watermarks, or extra text."
    )
    if config.negative_prompt:
        prompt = f"{prompt} Avoid: {config.negative_prompt}."
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

    return tqdm(total=total, desc="Flux.2 tiles", unit="tile", dynamic_ncols=True)


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


def run_flux2_tile_upscale(config: "DiffusionUpscaleConfig") -> Path:
    ml = _import_flux2()
    np = ml["np"]
    torch = ml["torch"]
    Image = ml["Image"]

    device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    model_id = _flux2_model_id(config)
    Pipeline = _pipeline_class(ml, model_id, config.flux2_pipeline)

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

    tile_size = _ceil_to_multiple(max(16, config.flux2_tile_size), 16)
    overlap = min(max(0, config.flux2_overlap), tile_size - 1)
    crops = _tile_grid(
        width=resized_width,
        height=resized_height,
        tile_size=tile_size,
        overlap=overlap,
        seed=config.seed,
        jitter=config.flux2_jitter,
    )

    pipe = Pipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        use_safetensors=True,
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

    prompt = _flux2_prompt(config)
    call_parameters = _pipeline_call_parameters(pipe)
    canvas = np.zeros((resized_height, resized_width, 3), dtype=np.float32)
    weights = np.zeros((resized_height, resized_width, 1), dtype=np.float32)
    progress = _progress_bar(len(crops))

    try:
        for crop in crops:
            reference_tile = resized_image.crop(
                (crop.x, crop.y, crop.right, crop.bottom)
            )
            call_kwargs: dict[str, Any] = {
                "image": reference_tile,
                "prompt": prompt,
                "height": crop.height,
                "width": crop.width,
                "num_inference_steps": config.steps,
                "generator": generator,
                "max_sequence_length": config.flux2_max_sequence_length,
                "output_type": "pil",
            }
            if "guidance_scale" in call_parameters:
                call_kwargs["guidance_scale"] = float(config.guidance_scale)
            if (
                "caption_upsample_temperature" in call_parameters
                and config.flux2_caption_upsample_temperature is not None
            ):
                call_kwargs["caption_upsample_temperature"] = float(
                    config.flux2_caption_upsample_temperature
                )

            result = pipe(**call_kwargs)
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
