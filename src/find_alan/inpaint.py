"""IP-Adapter + Stable Diffusion inpainting pipeline."""

from __future__ import annotations

import torch
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline

_SD_INPAINT_MODEL = "runwayml/stable-diffusion-inpainting"
_IP_ADAPTER_REPO = "h94/IP-Adapter"

# "plus" uses ViT-H — more faithful to the reference image.
# Use IP_ADAPTER_WEIGHTS_BASE for more creative freedom.
IP_ADAPTER_WEIGHTS_PLUS = "ip-adapter-plus_sd15.bin"
IP_ADAPTER_WEIGHTS_BASE = "ip-adapter_sd15.bin"

DEFAULT_PROMPT = (
    "a person standing in a crowd, same scale and size as surrounding"
    " figures, natural perspective, fitting into the scene, consistent"
    " lighting and style"
)
DEFAULT_NEGATIVE_PROMPT = (
    "blurry, distorted, deformed, bad anatomy, watermark, low quality"
)

# SD 1.5 was trained at 512×512; multiples of 64 work well up to ~768.
MODEL_RESOLUTION = 512


def _auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_pipeline(
    device: str | None = None,
    ip_adapter_weights: str = IP_ADAPTER_WEIGHTS_PLUS,
) -> StableDiffusionInpaintPipeline:
    """Download (first run) and return a ready-to-use inpainting pipeline."""
    if device is None:
        device = _auto_device()

    dtype = torch.float32 if device == "cpu" else torch.float16

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        _SD_INPAINT_MODEL,
        torch_dtype=dtype,
        safety_checker=None,
    )
    pipe.load_ip_adapter(
        _IP_ADAPTER_REPO,
        subfolder="models",
        weight_name=ip_adapter_weights,
    )
    pipe = pipe.to(device)
    return pipe


def run_inpainting(
    pipe: StableDiffusionInpaintPipeline,
    scene: Image.Image,
    figure: Image.Image,
    mask: Image.Image,
    prompt: str = DEFAULT_PROMPT,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    ip_adapter_scale: float = 0.7,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    seed: int | None = None,
) -> Image.Image:
    """
    Inpaint *figure* into *scene* within *mask* (white=inpaint, black=keep).

    Returns an image at the same size as *scene*.
    """
    pipe.set_ip_adapter_scale(ip_adapter_scale)

    generator = None
    if seed is not None:
        generator = torch.Generator(
            device=pipe.device.type
        ).manual_seed(seed)

    original_size = scene.size

    model_scene = scene.convert("RGB").resize(
        (MODEL_RESOLUTION, MODEL_RESOLUTION)
    )
    model_mask = mask.convert("L").resize(
        (MODEL_RESOLUTION, MODEL_RESOLUTION)
    )

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=model_scene,
        mask_image=model_mask,
        ip_adapter_image=figure.convert("RGB"),
        height=MODEL_RESOLUTION,
        width=MODEL_RESOLUTION,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images[0]

    # Composite result back onto the original-resolution scene so that the
    # non-masked area is pixel-identical to the input.
    result_full = result.resize(original_size, Image.LANCZOS)
    mask_full = mask.convert("L").resize(original_size, Image.LANCZOS)
    return Image.composite(result_full, scene.convert("RGB"), mask_full)
