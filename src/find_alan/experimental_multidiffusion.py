"""Experimental latent-space MultiDiffusion upscaler."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from .tiling import Crop, jittered_crop_schedule
from .upscale import MissingMLDependencies, target_size

if TYPE_CHECKING:
    from .upscale import DiffusionUpscaleConfig


def _import_diffusers() -> dict[str, Any]:
    try:
        import torch
        from diffusers import (
            AutoencoderKL,
            ControlNetModel,
            StableDiffusionXLControlNetImg2ImgPipeline,
            UniPCMultistepScheduler,
        )
        from PIL import Image
    except ImportError as exc:
        raise MissingMLDependencies(
            "Install the optional image stack with `uv sync --extra ml`. "
            "The experimental multidiffusion engine also downloads SDXL, VAE, "
            "and ControlNet Tile weights on first run."
        ) from exc

    return {
        "torch": torch,
        "AutoencoderKL": AutoencoderKL,
        "ControlNetModel": ControlNetModel,
        "StableDiffusionXLControlNetImg2ImgPipeline": StableDiffusionXLControlNetImg2ImgPipeline,
        "UniPCMultistepScheduler": UniPCMultistepScheduler,
        "Image": Image,
    }


def _vae_scale_factor(pipe: Any) -> int:
    if hasattr(pipe, "vae_scale_factor"):
        return int(pipe.vae_scale_factor)

    block_channels = getattr(pipe.vae.config, "block_out_channels", [1, 1, 1, 1])
    return 2 ** (len(block_channels) - 1)


def _preprocess_control(pipe: Any, image: Any, height: int, width: int, device: str, dtype: Any) -> Any:
    processor = getattr(pipe, "control_image_processor", pipe.image_processor)
    try:
        control = processor.preprocess(image, height=height, width=width)
    except TypeError:
        control = processor.preprocess(image)
    return control.to(device=device, dtype=dtype)


def _prepare_latents(
    pipe: Any,
    init_image: Any,
    latent_timestep: Any,
    dtype: Any,
    device: str,
    generator: Any,
) -> Any:
    torch = __import__("torch")
    _remove_accelerate_hook(pipe.vae)

    seed = generator.initial_seed() if generator is not None else None
    cpu_generator = None
    if seed is not None:
        cpu_generator = torch.Generator(device="cpu").manual_seed(seed)

    pipe.vae.to(device="cpu", dtype=torch.float32)
    init_image = init_image.detach().to(device="cpu", dtype=torch.float32)
    latent_timestep = latent_timestep.to(device="cpu")

    with torch.no_grad():
        latent_dist = pipe.vae.encode(init_image).latent_dist
        latents = latent_dist.sample(generator=cpu_generator)
        latents = latents * pipe.vae.config.scaling_factor
        noise = torch.randn(
            latents.shape,
            generator=cpu_generator,
            device="cpu",
            dtype=latents.dtype,
        )
        latents = pipe.scheduler.add_noise(latents, noise, latent_timestep)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return latents.to(device=device, dtype=dtype)


def _randn_like(tensor: Any, generator: Any) -> Any:
    torch = __import__("torch")
    try:
        return torch.randn_like(tensor, generator=generator)
    except TypeError:
        return torch.randn(tensor.shape, generator=generator, device=tensor.device, dtype=tensor.dtype)


def _timesteps(pipe: Any, steps: int, strength: float, device: str) -> Any:
    pipe.scheduler.set_timesteps(steps, device=device)
    if hasattr(pipe, "get_timesteps"):
        timesteps, _ = pipe.get_timesteps(steps, strength, device)
        return timesteps

    init_timestep = min(int(steps * strength), steps)
    t_start = max(steps - init_timestep, 0)
    return pipe.scheduler.timesteps[t_start:]


def _extra_step_kwargs(pipe: Any, generator: Any) -> dict[str, Any]:
    if hasattr(pipe, "prepare_extra_step_kwargs"):
        return pipe.prepare_extra_step_kwargs(generator, eta=0.0)
    return {}


def _encode_prompt(pipe: Any, config: "DiffusionUpscaleConfig", device: str, do_cfg: bool) -> tuple[Any, Any]:
    prompt_outputs = pipe.encode_prompt(
        prompt=config.prompt,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=do_cfg,
        negative_prompt=config.negative_prompt,
    )
    prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = prompt_outputs[:4]

    if do_cfg:
        torch = __import__("torch")
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds])

    return prompt_embeds, pooled_prompt_embeds


def _time_ids(
    pipe: Any,
    *,
    original_size: tuple[int, int],
    crop_top_left: tuple[int, int],
    target: tuple[int, int],
    dtype: Any,
    device: str,
    do_cfg: bool,
) -> Any:
    projection_dim = getattr(getattr(pipe, "text_encoder_2", None), "config", None)
    projection_dim = getattr(projection_dim, "projection_dim", None)

    try:
        time_ids = pipe._get_add_time_ids(
            original_size,
            crop_top_left,
            target,
            dtype=dtype,
            text_encoder_projection_dim=projection_dim,
        )
    except (AttributeError, TypeError):
        try:
            time_ids = pipe._get_add_time_ids(original_size, crop_top_left, target, dtype=dtype)
        except (AttributeError, TypeError):
            torch = __import__("torch")
            values = list(original_size + crop_top_left + target)
            time_ids = torch.tensor([values], dtype=dtype)

    time_ids = time_ids.to(device=device, dtype=dtype)
    if do_cfg:
        time_ids = __import__("torch").cat([time_ids, time_ids])
    return time_ids


def _gaussian_weight(height: int, width: int, sigma: float, device: str, dtype: Any) -> Any:
    torch = __import__("torch")
    if sigma <= 0:
        return torch.ones((1, 1, height, width), device=device, dtype=dtype)

    y = torch.linspace(-1, 1, height, device=device, dtype=dtype).view(1, 1, height, 1)
    x = torch.linspace(-1, 1, width, device=device, dtype=dtype).view(1, 1, 1, width)
    distance = x.square() + y.square()
    weight = torch.exp(-distance / (2 * sigma * sigma))
    return weight.clamp_min(1e-3)


def _latent_views(
    latent_width: int,
    latent_height: int,
    config: "DiffusionUpscaleConfig",
    vae_scale_factor: int,
    steps: int,
) -> list[list[Crop]]:
    tile_size = max(1, config.multidiffusion_tile_size // vae_scale_factor)
    overlap = max(0, config.multidiffusion_overlap // vae_scale_factor)
    overlap = min(overlap, tile_size - 1)
    jitter = None
    if config.multidiffusion_jitter is not None:
        jitter = max(0, config.multidiffusion_jitter // vae_scale_factor)

    return list(
        jittered_crop_schedule(
            width=latent_width,
            height=latent_height,
            tile_size=tile_size,
            overlap=overlap,
            steps=steps,
            seed=config.seed,
            jitter=jitter,
        )
    )


def _remove_accelerate_hook(module: Any) -> None:
    try:
        from accelerate.hooks import remove_hook_from_module
    except ImportError:
        return

    remove_hook_from_module(module, recurse=True)


def _offload_before_decode(pipe: Any, torch: Any) -> None:
    if hasattr(pipe, "maybe_free_model_hooks"):
        pipe.maybe_free_model_hooks()

    for module_name in ("unet", "controlnet", "text_encoder", "text_encoder_2", "vae"):
        module = getattr(pipe, module_name, None)
        if module is not None:
            _remove_accelerate_hook(module)
            if hasattr(module, "to"):
                module.to("cpu")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _progress_bar(total: int, *, desc: str, unit: str) -> Any | None:
    if total <= 0:
        return None

    try:
        from tqdm.auto import tqdm
    except ImportError:
        return None

    return tqdm(total=total, desc=desc, unit=unit, dynamic_ncols=True)


def _decode(pipe: Any, latents: Any, output_path: Path) -> Path:
    torch = __import__("torch")
    _offload_before_decode(pipe, torch)

    vae_latents = latents.detach().to(device="cpu", dtype=torch.float32)
    del latents
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    pipe.vae.to(device="cpu", dtype=torch.float32)
    vae_latents = vae_latents / pipe.vae.config.scaling_factor
    with torch.no_grad():
        image_tensor = pipe.vae.decode(vae_latents, return_dict=False)[0].detach()

    image = pipe.image_processor.postprocess(image_tensor, output_type="pil")[0]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path


def run_multidiffusion_upscale(config: "DiffusionUpscaleConfig") -> Path:
    ml = _import_diffusers()
    torch = ml["torch"]
    Image = ml["Image"]
    AutoencoderKL = ml["AutoencoderKL"]
    ControlNetModel = ml["ControlNetModel"]
    Pipeline = ml["StableDiffusionXLControlNetImg2ImgPipeline"]
    UniPCMultistepScheduler = ml["UniPCMultistepScheduler"]

    if config.multidiffusion_view_batch_size != 1:
        raise ValueError("Experimental multidiffusion currently supports --md-view-batch-size 1 only")

    device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.startswith("cuda") else torch.float32

    control_image = Image.open(config.input_path).convert("RGB")
    original_width, original_height = control_image.size
    resized_width, resized_height = target_size(original_width, original_height, config.scale)
    init_image = control_image.resize((resized_width, resized_height), Image.Resampling.LANCZOS)

    controlnet = ControlNetModel.from_pretrained(
        config.controlnet_id,
        torch_dtype=dtype,
        use_safetensors=True,
    )
    vae = AutoencoderKL.from_pretrained(config.vae_id, torch_dtype=dtype, use_safetensors=True)
    pipe = Pipeline.from_pretrained(
        config.model_id,
        controlnet=controlnet,
        vae=vae,
        torch_dtype=dtype,
        use_safetensors=True,
        variant="fp16" if dtype is torch.float16 else None,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    if config.cpu_offload and device.startswith("cuda") and hasattr(pipe, "enable_model_cpu_offload"):
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

    generator = None
    if config.seed is not None:
        generator = torch.Generator(device=device).manual_seed(config.seed)

    do_cfg = config.guidance_scale > 1.0
    prompt_embeds, pooled_prompt_embeds = _encode_prompt(pipe, config, device, do_cfg)
    init_tensor = pipe.image_processor.preprocess(
        init_image, height=resized_height, width=resized_width
    ).to(device=device, dtype=dtype)
    control_tensor = _preprocess_control(
        pipe, init_image, resized_height, resized_width, device=device, dtype=dtype
    )

    timesteps = _timesteps(pipe, config.steps, config.denoising_strength, device)
    latent_timestep = timesteps[:1].repeat(1)
    latents = _prepare_latents(pipe, init_tensor, latent_timestep, dtype, device, generator)
    extra_step_kwargs = _extra_step_kwargs(pipe, generator)
    scale_factor = _vae_scale_factor(pipe)

    _, _, latent_height, latent_width = latents.shape
    view_schedule = _latent_views(
        latent_width=latent_width,
        latent_height=latent_height,
        config=config,
        vae_scale_factor=scale_factor,
        steps=len(timesteps),
    )

    total_views = sum(len(views) for views in view_schedule)
    progress = _progress_bar(total_views, desc="MultiDiffusion crops", unit="crop")
    try:
        with torch.no_grad():
            for step_index, timestep in enumerate(timesteps):
                step_views = view_schedule[step_index]
                if progress is not None:
                    progress.set_postfix(
                        step=f"{step_index + 1}/{len(timesteps)}",
                        views=len(step_views),
                        refresh=False,
                    )

                noise_value = torch.zeros_like(latents)
                count = torch.zeros(
                    (latents.shape[0], 1, latents.shape[2], latents.shape[3]),
                    device=latents.device,
                    dtype=latents.dtype,
                )

                for view in step_views:
                    y0, y1 = view.y, view.bottom
                    x0, x1 = view.x, view.right
                    latent_crop = latents[:, :, y0:y1, x0:x1]
                    latent_model_input = torch.cat([latent_crop, latent_crop]) if do_cfg else latent_crop
                    latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, timestep)

                    pixel_y0, pixel_y1 = y0 * scale_factor, y1 * scale_factor
                    pixel_x0, pixel_x1 = x0 * scale_factor, x1 * scale_factor
                    control_crop = control_tensor[:, :, pixel_y0:pixel_y1, pixel_x0:pixel_x1]
                    if do_cfg:
                        control_crop = torch.cat([control_crop, control_crop])

                    time_ids = _time_ids(
                        pipe,
                        original_size=(original_width, original_height),
                        crop_top_left=(pixel_x0, pixel_y0),
                        target=(resized_width, resized_height),
                        dtype=dtype,
                        device=device,
                        do_cfg=do_cfg,
                    )
                    added_cond_kwargs = {
                        "text_embeds": pooled_prompt_embeds,
                        "time_ids": time_ids,
                    }

                    down_samples, mid_sample = pipe.controlnet(
                        latent_model_input,
                        timestep,
                        encoder_hidden_states=prompt_embeds,
                        controlnet_cond=control_crop,
                        conditioning_scale=float(config.controlnet_strength),
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )
                    noise_pred = pipe.unet(
                        latent_model_input,
                        timestep,
                        encoder_hidden_states=prompt_embeds,
                        down_block_additional_residuals=down_samples,
                        mid_block_additional_residual=mid_sample,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0]

                    if do_cfg:
                        noise_uncond, noise_text = noise_pred.chunk(2)
                        noise_pred = noise_uncond + config.guidance_scale * (noise_text - noise_uncond)

                    weight = _gaussian_weight(
                        noise_pred.shape[-2],
                        noise_pred.shape[-1],
                        config.tile_gaussian_sigma,
                        device=device,
                        dtype=noise_pred.dtype,
                    )
                    noise_value[:, :, y0:y1, x0:x1] += noise_pred * weight
                    count[:, :, y0:y1, x0:x1] += weight

                    if progress is not None:
                        progress.update(1)

                fused_noise = noise_value / count.clamp_min(1e-6)
                latents = pipe.scheduler.step(
                    fused_noise,
                    timestep,
                    latents,
                    **extra_step_kwargs,
                    return_dict=False,
                )[0]
    finally:
        if progress is not None:
            progress.close()

    return _decode(pipe, latents, config.output_path)
