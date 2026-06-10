"""Experimental Flux.2 latent-space MultiDiffusion upscaler."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .experimental_flux2_tile import (
    _ceil_to_multiple,
    _flux2_model_id,
    _flux2_prompt,
    _import_flux2,
    _pipeline_class,
)
from .tiling import Crop, jittered_grid
from .upscale import target_size

if TYPE_CHECKING:
    from .upscale import DiffusionUpscaleConfig


@dataclass(frozen=True)
class _ViewContext:
    crop: Crop
    indices: Any
    latent_ids: Any
    image_latents: Any
    image_latent_ids: Any
    weight: Any


def _progress_bar(total: int, desc: str) -> Any | None:
    if total <= 0:
        return None

    try:
        from tqdm.auto import tqdm
    except ImportError:
        return None

    return tqdm(total=total, desc=desc, unit="crop", dynamic_ncols=True)


def _gaussian_token_weight(height: int, width: int, sigma: float, torch: Any, device: Any, dtype: Any) -> Any:
    if sigma <= 0:
        return torch.ones((1, height * width, 1), device=device, dtype=dtype)

    y = torch.linspace(-1, 1, height, device=device, dtype=dtype).view(height, 1)
    x = torch.linspace(-1, 1, width, device=device, dtype=dtype).view(1, width)
    distance = x.square() + y.square()
    weight = torch.exp(-distance / (2 * sigma * sigma)).clamp_min(1e-3)
    return weight.reshape(1, height * width, 1)


def _crop_indices(crop: Crop, latent_width: int, torch: Any, device: Any) -> Any:
    rows = torch.arange(crop.y, crop.bottom, device=device, dtype=torch.long)
    cols = torch.arange(crop.x, crop.right, device=device, dtype=torch.long)
    return (rows[:, None] * latent_width + cols[None, :]).reshape(-1)


def _preprocess_reference_image(pipe: Any, image: Any) -> Any:
    image_width, image_height = image.size
    if image_width * image_height > 1024 * 1024:
        image = pipe.image_processor._resize_to_target_area(image, 1024 * 1024)
        image_width, image_height = image.size

    multiple_of = pipe.vae_scale_factor * 2
    image_width = max(multiple_of, (image_width // multiple_of) * multiple_of)
    image_height = max(multiple_of, (image_height // multiple_of) * multiple_of)
    return pipe.image_processor.preprocess(
        image,
        height=image_height,
        width=image_width,
        resize_mode="crop",
    )


def _text_encoder_out_layers(pipe: Any) -> tuple[int, ...]:
    import inspect

    parameter = inspect.signature(pipe.__call__).parameters.get("text_encoder_out_layers")
    if parameter is None or parameter.default is inspect.Parameter.empty:
        return (9, 18, 27)
    return tuple(parameter.default)


def _retrieve_timesteps(pipe: Any, np: Any, num_steps: int, image_seq_len: int, device: Any) -> tuple[Any, int]:
    try:
        from diffusers.pipelines.flux2.pipeline_flux2_klein import (
            compute_empirical_mu,
            retrieve_timesteps,
        )
    except ImportError:
        from diffusers.pipelines.flux2.pipeline_flux2 import (
            compute_empirical_mu,
            retrieve_timesteps,
        )

    sigmas = np.linspace(1.0, 1 / num_steps, num_steps)
    if getattr(pipe.scheduler.config, "use_flow_sigmas", False):
        sigmas = None
    mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=num_steps)
    return retrieve_timesteps(pipe.scheduler, num_steps, device, sigmas=sigmas, mu=mu)


def _build_views(
    *,
    pipe: Any,
    resized_image: Any,
    resized_width: int,
    resized_height: int,
    config: "DiffusionUpscaleConfig",
    latents: Any,
    latent_ids: Any,
    generator: Any,
    torch: Any,
    device: Any,
) -> list[_ViewContext]:
    pixel_per_token = pipe.vae_scale_factor * 2
    latent_width = resized_width // pixel_per_token
    latent_height = resized_height // pixel_per_token

    tile_size = _ceil_to_multiple(max(pixel_per_token, config.flux2_tile_size), pixel_per_token)
    overlap = min(max(0, config.flux2_overlap), tile_size - pixel_per_token)
    token_tile_size = max(1, tile_size // pixel_per_token)
    token_overlap = min(max(0, overlap // pixel_per_token), token_tile_size - 1)
    token_jitter = None
    if config.flux2_jitter is not None:
        token_jitter = max(0, config.flux2_jitter // pixel_per_token)

    crops = jittered_grid(
        width=latent_width,
        height=latent_height,
        tile_size=token_tile_size,
        overlap=token_overlap,
        rng=__import__("random").Random(config.seed),
        jitter=token_jitter,
    )

    contexts: list[_ViewContext] = []
    for crop in crops:
        indices = _crop_indices(crop, latent_width, torch, device)
        crop_latent_ids = latent_ids.index_select(1, indices)

        pixel_box = (
            crop.x * pixel_per_token,
            crop.y * pixel_per_token,
            crop.right * pixel_per_token,
            crop.bottom * pixel_per_token,
        )
        reference_tile = resized_image.crop(pixel_box)
        condition_image = _preprocess_reference_image(pipe, reference_tile)
        with torch.no_grad():
            image_latents, image_latent_ids = pipe.prepare_image_latents(
                images=[condition_image],
                batch_size=latents.shape[0],
                generator=generator,
                device=device,
                dtype=pipe.vae.dtype,
            )
        image_latents = image_latents.detach()
        image_latent_ids = image_latent_ids.detach().clone()
        image_latent_ids[..., 1] += crop.y
        image_latent_ids[..., 2] += crop.x

        weight = _gaussian_token_weight(
            crop.height,
            crop.width,
            float(config.tile_gaussian_sigma),
            torch,
            device,
            latents.dtype,
        )
        contexts.append(
            _ViewContext(
                crop=crop,
                indices=indices,
                latent_ids=crop_latent_ids,
                image_latents=image_latents,
                image_latent_ids=image_latent_ids,
                weight=weight,
            )
        )

    return contexts


def _decode_latents(pipe: Any, latents: Any, latent_ids: Any, height: int, width: int, output_path: Path) -> Path:
    latent_height = 2 * (int(height) // (pipe.vae_scale_factor * 2))
    latent_width = 2 * (int(width) // (pipe.vae_scale_factor * 2))
    latents = pipe._unpack_latents_with_ids(latents, latent_ids, latent_height // 2, latent_width // 2)

    latents_bn_mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
    latents_bn_std = __import__("torch").sqrt(
        pipe.vae.bn.running_var.view(1, -1, 1, 1) + pipe.vae.config.batch_norm_eps
    ).to(latents.device, latents.dtype)
    latents = latents * latents_bn_std + latents_bn_mean
    latents = pipe._unpatchify_latents(latents)
    with __import__("torch").no_grad():
        image = pipe.vae.decode(latents, return_dict=False)[0]
    image = pipe.image_processor.postprocess(image.detach(), output_type="pil")[0]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path


def run_flux2_multidiffusion_upscale(config: "DiffusionUpscaleConfig") -> Path:
    if config.multidiffusion_view_batch_size != 1:
        raise ValueError("Experimental flux2-multidiffusion currently supports --md-view-batch-size 1 only")

    ml = _import_flux2()
    np = ml["np"]
    torch = ml["torch"]
    Image = ml["Image"]

    requested_device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if requested_device.startswith("cuda") else torch.float32
    model_id = _flux2_model_id(config)
    Pipeline = _pipeline_class(ml, model_id, config.flux2_pipeline)

    pipe = Pipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        use_safetensors=True,
    )
    if (
        config.cpu_offload
        and requested_device.startswith("cuda")
        and hasattr(pipe, "enable_model_cpu_offload")
    ):
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(requested_device)

    if hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()
    elif hasattr(getattr(pipe, "vae", None), "enable_tiling"):
        pipe.vae.enable_tiling()
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    elif hasattr(getattr(pipe, "vae", None), "enable_slicing"):
        pipe.vae.enable_slicing()

    device = pipe._execution_device
    source_image = Image.open(config.input_path).convert("RGB")
    original_width, original_height = source_image.size
    pixel_per_token = pipe.vae_scale_factor * 2
    resized_width, resized_height = target_size(
        original_width,
        original_height,
        config.scale,
        multiple=pixel_per_token,
    )
    resized_image = source_image.resize((resized_width, resized_height), Image.Resampling.LANCZOS)

    generator = None
    if config.seed is not None:
        generator = torch.Generator(device=device).manual_seed(config.seed)

    prompt = _flux2_prompt(config)
    pipe._guidance_scale = float(config.guidance_scale)
    pipe._attention_kwargs = None
    pipe._current_timestep = None
    pipe._interrupt = False

    with torch.no_grad():
        prompt_embeds, text_ids = pipe.encode_prompt(
            prompt=prompt,
            prompt_embeds=None,
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=config.flux2_max_sequence_length,
            text_encoder_out_layers=_text_encoder_out_layers(pipe),
        )
    prompt_embeds = prompt_embeds.detach()
    text_ids = text_ids.detach()

    negative_prompt_embeds = None
    negative_text_ids = None
    if pipe.do_classifier_free_guidance:
        with torch.no_grad():
            negative_prompt_embeds, negative_text_ids = pipe.encode_prompt(
                prompt=config.negative_prompt or "",
                prompt_embeds=None,
                device=device,
                num_images_per_prompt=1,
                max_sequence_length=config.flux2_max_sequence_length,
                text_encoder_out_layers=_text_encoder_out_layers(pipe),
            )
        negative_prompt_embeds = negative_prompt_embeds.detach()
        negative_text_ids = negative_text_ids.detach()

    num_channels_latents = pipe.transformer.config.in_channels // 4
    latents, latent_ids = pipe.prepare_latents(
        batch_size=1,
        num_latents_channels=num_channels_latents,
        height=resized_height,
        width=resized_width,
        dtype=prompt_embeds.dtype,
        device=device,
        generator=generator,
        latents=None,
    )

    views = _build_views(
        pipe=pipe,
        resized_image=resized_image,
        resized_width=resized_width,
        resized_height=resized_height,
        config=config,
        latents=latents,
        latent_ids=latent_ids,
        generator=generator,
        torch=torch,
        device=device,
    )

    timesteps, num_inference_steps = _retrieve_timesteps(
        pipe,
        np,
        config.steps,
        latents.shape[1],
        device,
    )
    pipe.scheduler.set_begin_index(0)

    progress = _progress_bar(num_inference_steps * len(views), "Flux.2 MD crops")
    try:
        with torch.no_grad():
            for step_index, timestep_value in enumerate(timesteps):
                if progress is not None:
                    progress.set_postfix(step=f"{step_index + 1}/{num_inference_steps}", views=len(views), refresh=False)

                pipe._current_timestep = timestep_value
                timestep = timestep_value.expand(latents.shape[0]).to(latents.dtype)
                noise_value = torch.zeros_like(latents)
                count = torch.zeros((latents.shape[0], latents.shape[1], 1), device=device, dtype=latents.dtype)

                for view in views:
                    latent_crop = latents.index_select(1, view.indices)
                    latent_model_input = torch.cat([latent_crop, view.image_latents], dim=1).to(pipe.transformer.dtype)
                    latent_image_ids = torch.cat([view.latent_ids, view.image_latent_ids], dim=1)

                    noise_pred = pipe.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep / 1000,
                        guidance=None,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=pipe._attention_kwargs,
                        return_dict=False,
                    )[0]
                    noise_pred = noise_pred[:, : latent_crop.size(1), :]

                    if pipe.do_classifier_free_guidance:
                        neg_noise_pred = pipe.transformer(
                            hidden_states=latent_model_input,
                            timestep=timestep / 1000,
                            guidance=None,
                            encoder_hidden_states=negative_prompt_embeds,
                            txt_ids=negative_text_ids,
                            img_ids=latent_image_ids,
                            joint_attention_kwargs=pipe._attention_kwargs,
                            return_dict=False,
                        )[0]
                        neg_noise_pred = neg_noise_pred[:, : latent_crop.size(1), :]
                        noise_pred = neg_noise_pred + float(config.guidance_scale) * (noise_pred - neg_noise_pred)

                    weight = view.weight.to(dtype=noise_pred.dtype)
                    noise_value.index_add_(1, view.indices, noise_pred.to(latents.dtype) * weight.to(latents.dtype))
                    count.index_add_(1, view.indices, weight.to(latents.dtype))

                    if progress is not None:
                        progress.update(1)

                fused_noise = noise_value / count.clamp_min(1e-6)
                latents_dtype = latents.dtype
                latents = pipe.scheduler.step(fused_noise, timestep_value, latents, return_dict=False)[0]
                if latents.dtype != latents_dtype:
                    latents = latents.to(latents_dtype)
    finally:
        if progress is not None:
            progress.close()

    pipe._current_timestep = None
    output_path = _decode_latents(pipe, latents, latent_ids, resized_height, resized_width, config.output_path)
    if hasattr(pipe, "maybe_free_model_hooks"):
        pipe.maybe_free_model_hooks()
    return output_path
