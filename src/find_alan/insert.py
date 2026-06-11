"""FLUX.2-Klein figure insertion pipeline.

Uses Flux2KleinInpaintPipeline with an image_reference parameter — no mask
required. The model decides where to place the figure based on the prompt
and the scene context.
"""

from __future__ import annotations

import torch
from diffusers.pipelines.flux2.pipeline_flux2_klein_inpaint import (
    Flux2KleinInpaintPipeline,
)
from PIL import Image

_KLEIN_MODEL = "black-forest-labs/FLUX.2-klein-4B"

DEFAULT_PROMPT = (
    "A crowd scene in which one of the background figures is the person"
    " from the reference image. The person is the same apparent height and"
    " body size as the people standing directly beside them — not a"
    " foreground subject, not larger than the surrounding crowd."
    " Their feet are planted on the ground at the same depth plane as"
    " nearby figures. Adjacent crowd members overlap them slightly in a"
    " natural way. Every other part of"
    " the image — all other crowd members, the background, the overall"
    " composition — is identical to the original."
)


def _auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _pad_to_square(img: Image.Image) -> Image.Image:
    """Pad to square so the reference encoder doesn't squish the figure."""
    w, h = img.size
    if w == h:
        return img
    size = max(w, h)
    out = Image.new("RGB", (size, size), (255, 255, 255))
    out.paste(img, ((size - w) // 2, (size - h) // 2))
    return out


def load_pipeline(device: str | None = None) -> Flux2KleinInpaintPipeline:
    """Download (first run) and return the FLUX.2-Klein inpaint pipeline.

    Requires ~13 GB VRAM. Model is gated — accept the licence at:
      https://huggingface.co/black-forest-labs/FLUX.2-klein-4B
    then run: huggingface-cli login
    """
    if device is None:
        device = _auto_device()

    pipe = Flux2KleinInpaintPipeline.from_pretrained(
        _KLEIN_MODEL,
        torch_dtype=torch.bfloat16,
    )
    pipe.enable_sequential_cpu_offload()
    return pipe


def run_insertion(
    pipe: Flux2KleinInpaintPipeline,
    scene: Image.Image,
    figure: Image.Image,
    prompt: str = DEFAULT_PROMPT,
    strength: float = 0.75,
    num_inference_steps: int = 50,
    guidance_scale: float = 10.0,
    seed: int | None = None,
) -> Image.Image:
    """
    Insert *figure* into *scene*.

    The model infers placement from the prompt and scene context.
    *strength* controls how much the scene is allowed to change (0–1);
    lower values preserve more of the original.
    """
    generator = None
    if seed is not None:
        generator = torch.Generator(device=pipe._execution_device).manual_seed(seed)

    w, h = scene.size
    full_mask = Image.new("L", (w, h), 255)

    result = pipe(
        prompt=prompt,
        image=scene.convert("RGB"),
        mask_image=full_mask,
        image_reference=_pad_to_square(figure.convert("RGB")),
        height=h,
        width=w,
        strength=strength,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images[0]

    return result
