"""IP-Adapter + FLUX.1 inpainting pipeline."""

from __future__ import annotations

import torch
from PIL import Image
from diffusers import FluxInpaintPipeline

_FLUX_MODEL = "black-forest-labs/FLUX.1-dev"
_IP_ADAPTER_REPO = "InstantX/FLUX.1-dev-IP-Adapter"
_IP_ADAPTER_WEIGHTS = "ip-adapter.bin"

DEFAULT_PROMPT = (
    "a person standing in a crowd, same scale and size as surrounding"
    " figures, natural perspective, fitting into the scene, consistent"
    " lighting and style"
)


# FLUX.1 is trained at higher resolutions than SD 1.5.
# 1024 gives good quality; use 512 if memory is tight.
MODEL_RESOLUTION = 1024


def _auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_pipeline(
    device: str | None = None,
) -> FluxInpaintPipeline:
    """Download (first run) and return a ready-to-use inpainting pipeline.

    FLUX.1-dev is ~24 GB in bfloat16. Requires 24 GB+ VRAM (GPU)
    or unified memory (Apple Silicon).
    """
    if device is None:
        device = _auto_device()

    pipe = FluxInpaintPipeline.from_pretrained(
        _FLUX_MODEL,
        torch_dtype=torch.bfloat16,
    )
    pipe.load_ip_adapter(
        _IP_ADAPTER_REPO,
        weight_name=_IP_ADAPTER_WEIGHTS,
    )
    pipe = pipe.to(device)
    return pipe


def run_inpainting(
    pipe: FluxInpaintPipeline,
    scene: Image.Image,
    figure: Image.Image,
    mask: Image.Image,
    prompt: str = DEFAULT_PROMPT,
    ip_adapter_scale: float = 0.6,
    num_inference_steps: int = 50,
    guidance_scale: float = 30.0,
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

    # Round to nearest multiple of 16 (FLUX requirement).
    w = (original_size[0] // 16) * 16 or MODEL_RESOLUTION
    h = (original_size[1] // 16) * 16 or MODEL_RESOLUTION
    model_scene = scene.convert("RGB").resize((w, h))
    model_mask = mask.convert("L").resize((w, h))

    result = pipe(
        prompt=prompt,
        image=model_scene,
        mask_image=model_mask,
        ip_adapter_image=figure.convert("RGB"),
        height=h,
        width=w,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images[0]

    # Composite result back at original resolution so non-masked pixels
    # are pixel-identical to the input.
    result_full = result.resize(original_size, Image.LANCZOS)
    mask_full = mask.convert("L").resize(original_size, Image.LANCZOS)
    return Image.composite(result_full, scene.convert("RGB"), mask_full)
