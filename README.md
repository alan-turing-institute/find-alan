# find-alan

Python package skeleton managed with `uv`.

## Setup

Install the package environment:

```sh
uv sync
```

## Inpainting

Insert a figure into a crowd scene using IP-Adapter + FLUX.1 inpainting.

### Step 1 — Generate example images

Creates synthetic test images (`crowd_scene.png`, `figure.png`, `mask.png`) in `./examples/`:

```sh
uv run find-alan-prepare-examples
```

Replace these with real images for meaningful results.

### Step 2 — Run inpainting

Models (~24 GB) are downloaded from Hugging Face on first run. Requires 24 GB+ VRAM or unified memory (Apple Silicon).

```sh
uv run find-alan-inpaint \
  --scene examples/crowd_scene.png \
  --figure examples/figure.png \
  --mask examples/mask.png \
  --output examples/result.png \
  --seed 42
```

You can also specify a bounding box instead of a mask file:

```sh
uv run find-alan-inpaint \
  --scene examples/crowd_scene.png \
  --figure examples/figure.png \
  --bbox 210 330 90 150 \
  --output examples/result.png
```

See all options:

```sh
uv run find-alan-inpaint --help
```
