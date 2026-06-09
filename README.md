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

## Pipelines

### `mod-tile`

Default engine. Uses the Diffusers community tiled super-resolution pipeline with SDXL and ControlNet Tile/Union.

Best for: quick baselines, stable 4x results, preserving the source layout.

```sh
uv run find-alan-upscale input.png output.png --scale 4
```

### `multidiffusion`

Experimental engine. Runs overlapping latent crops at each denoising step, fuses their noise predictions, and advances the whole latent canvas once per step. This is closer to the original MultiDiffusion idea.

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
