# find-alan

Experiments for diffusion upscaling low-resolution images with tiled pipelines and a custom MultiDiffusion path.

## Setup

```sh
uv sync
uv sync --extra ml
```

The first real generation downloads model weights from Hugging Face unless they are already cached. Set `HF_TOKEN` for higher Hub rate limits.

## Commands

```sh
uv run find-alan-upscale --help
uv run find-alan-crop-plan --help
```

## How The Upscaling Works

The upscaler has three separate jobs that are easy to mix together:

1. **Image-to-image denoising** starts from a resized version of the low-resolution input. `--denoising-strength` controls how much noise is added before the model redraws it. Higher values give the model more room to invent new detail.
2. **ControlNet** feeds the source image back into the model as a spatial constraint. It says: keep this composition, these edges, this local texture, and this rough object placement. `--controlnet-strength` controls how strongly that constraint is enforced.
3. **Tiling/MultiDiffusion** is about scale and seams. Large images are too big to denoise in one pass, so the model denoises overlapping latent crops and blends them into one canvas.

ControlNet and MultiDiffusion solve different problems. ControlNet controls **what the generated image should stay aligned to**. MultiDiffusion controls **how many overlapping windows are combined into a seamless large image**.

For more hallucinated detail, raise `--denoising-strength` and lower `--controlnet-strength`. For a faithful upscale, lower denoising and raise ControlNet strength.

## Pipelines

### `mod-tile`

Default engine. Uses the Diffusers community tiled super-resolution pipeline with SDXL and ControlNet Tile/Union.

The flow is:

1. Resize the input image to the target scale.
2. Use ControlNet Tile/Union to keep the resized image structure visible to SDXL.
3. Let the community tiled SR pipeline split work into tiles and blend the output.

Best for: quick baselines, stable 4x results, preserving the source layout. It is the safer first pass when you want to check whether the source and prompt are reasonable.

```sh
uv run find-alan-upscale input.png output.png --scale 4
```

### `multidiffusion`

Experimental engine. Runs overlapping latent crops at each denoising step, fuses their noise predictions, and advances the whole latent canvas once per step. This is closer to the original MultiDiffusion idea.

The flow is:

1. Resize the input image to the target scale.
2. Encode that resized image into one large latent canvas.
3. For each denoising step, generate a crop grid over the latent canvas.
4. Run ControlNet and the UNet on every overlapping crop.
5. Blend the predicted noise from all crops with soft weights.
6. Advance the whole latent canvas once with the fused noise prediction.
7. Decode the final latent canvas back to pixels.

ControlNet still runs inside each crop. That means the custom MultiDiffusion engine can still be source-faithful if `--controlnet-strength` is high. The MultiDiffusion part makes the crop fusion more coherent; it does not, by itself, make the model more imaginative.

Best for: testing stronger hallucinated detail, jittered crop fusion, and high-overlap seamlessness. It is much slower than `mod-tile`.

```sh
uv run find-alan-upscale input.png output.png \
  --engine multidiffusion \
  --scale 2 \
  --steps 28 \
  --denoising-strength 0.92 \
  --controlnet-strength 0.45 \
  --guidance-scale 6 \
  --md-tile-size 768 \
  --md-overlap 384 \
  --md-jitter 256
```

### `crop-plan`

Debug helper. Prints the jittered crop grids used by the custom MultiDiffusion scheduler.

```sh
uv run find-alan-crop-plan --width 320 --height 240 --scale 10 --steps 4
```

## Main Parameters

`--denoising-strength`: higher means more imagined changes; lower means more faithful to the upscaled input.

`--controlnet-strength`: higher pins structure and local texture to the source; lower gives the model more freedom. Yes, ControlNet is one of the main reasons outputs stay similar.

`--guidance-scale`: higher follows the prompt harder, but can overcook details.

`--md-overlap`: higher improves seam consistency but increases runtime.

`--md-jitter`: changes crop alignment between denoising steps, which can reduce repeated tile artifacts.

## Useful Recipes

Faithful baseline:

```sh
uv run find-alan-upscale input.png output.png --engine mod-tile --scale 4 --denoising-strength 0.45
```

More imagined 2x MultiDiffusion:

```sh
uv run find-alan-upscale input.png output.png \
  --engine multidiffusion \
  --scale 2 \
  --steps 28 \
  --denoising-strength 0.92 \
  --controlnet-strength 0.45 \
  --guidance-scale 6 \
  --md-tile-size 768 \
  --md-overlap 384 \
  --md-jitter 256
```

Detailed 4x MultiDiffusion trial:

```sh
uv run find-alan-upscale input.png output.png \
  --engine multidiffusion \
  --scale 4 \
  --steps 24 \
  --denoising-strength 0.85 \
  --controlnet-strength 0.75 \
  --guidance-scale 5 \
  --md-tile-size 1024 \
  --md-overlap 512 \
  --md-jitter 256
```

Use a separate masked inpaint pass for final hidden-character and face corrections.

## Build

```sh
uv build
```
