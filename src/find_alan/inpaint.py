"""FLUX.1-Redux + FLUX.1-Fill inpainting pipeline.

FLUX Redux encodes a reference image into visual token embeddings that FLUX
conditions on — the same mechanism ComfyUI uses for reference-image injection.
FluxFillPipeline then uses those embeddings for scene-aware inpainting.

Text prompt tokens are concatenated with the Redux visual tokens so you can
guide scale, pose, and blending via text while Redux handles appearance.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from PIL import Image
from diffusers import FluxFillPipeline, FluxPriorReduxPipeline

_REDUX_MODEL = "black-forest-labs/FLUX.1-Redux-dev"
_FILL_MODEL = "black-forest-labs/FLUX.1-Fill-dev"

DEFAULT_PROMPT = (
    "a person fitting naturally into the crowd scene, scaled to match the"
    " size of surrounding figures, consistent perspective and lighting,"
    " full body preserved, blending seamlessly with the background"
)


def _auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class Pipelines:
    prior: FluxPriorReduxPipeline
    fill: FluxFillPipeline


def load_pipeline(device: str | None = None) -> Pipelines:
    """Download (first run) and return both FLUX pipelines.

    FLUX.1-Fill-dev + FLUX.1-Redux-dev together are ~25 GB in bfloat16.
    Both models are gated — accept the licence at:
      https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev
      https://huggingface.co/black-forest-labs/FLUX.1-Redux-dev
    then run: huggingface-cli login
    """
    if device is None:
        device = _auto_device()

    prior = FluxPriorReduxPipeline.from_pretrained(
        _REDUX_MODEL,
        torch_dtype=torch.bfloat16,
    ).to(device)

    fill = FluxFillPipeline.from_pretrained(
        _FILL_MODEL,
        torch_dtype=torch.bfloat16,
    ).to(device)

    return Pipelines(prior=prior, fill=fill)


def _combine_redux_and_text(
    pipelines: Pipelines,
    figure: Image.Image,
    prompt: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode the reference figure and text prompt, return combined embeddings.

    Redux visual tokens are prepended to the T5 text tokens so the model
    conditions on both appearance (Redux) and layout/sizing instructions
    (text). The Redux pooled embedding is used for CLIP conditioning.
    """
    prior_output = pipelines.prior(figure.convert("RGB"))

    if prompt:
        text_embeds, _, _ = pipelines.fill.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=256,
        )
        # Visual tokens first, then text tokens.
        prompt_embeds = torch.cat(
            [prior_output.prompt_embeds, text_embeds], dim=1
        )
    else:
        prompt_embeds = prior_output.prompt_embeds

    return prompt_embeds, prior_output.pooled_prompt_embeds


def run_inpainting(
    pipelines: Pipelines,
    scene: Image.Image,
    figure: Image.Image,
    mask: Image.Image,
    prompt: str = DEFAULT_PROMPT,
    num_inference_steps: int = 50,
    guidance_scale: float = 15.0,
    seed: int | None = None,
) -> Image.Image:
    """
    Inpaint the masked region of *scene*, conditioned on *figure*.

    Returns an image at the same size as *scene*.
    """
    device = pipelines.fill.device

    generator = None
    if seed is not None:
        generator = torch.Generator(device=device.type).manual_seed(seed)

    original_size = scene.size

    # Round to nearest multiple of 16 (FLUX requirement).
    w = max((original_size[0] // 16) * 16, 16)
    h = max((original_size[1] // 16) * 16, 16)
    model_scene = scene.convert("RGB").resize((w, h))
    model_mask = mask.convert("L").resize((w, h))

    prompt_embeds, pooled_prompt_embeds = _combine_redux_and_text(
        pipelines, figure, prompt, device
    )

    result = pipelines.fill(
        image=model_scene,
        mask_image=model_mask,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        height=h,
        width=w,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images[0]

    result_full = result.resize(original_size, Image.LANCZOS)
    mask_full = mask.convert("L").resize(original_size, Image.LANCZOS)
    return Image.composite(result_full, scene.convert("RGB"), mask_full)
