# find-alan

Python package skeleton managed with `uv`.

## Setup

Install the package environment:

```sh
uv sync
```

## Inpainting

Insert a figure into a crowd scene using FLUX.1-Redux + FLUX.1-Fill inpainting.

### Step 1 — Generate example images

Creates synthetic test images (`crowd_scene.png`, `figure.png`, `mask.png`) in `./examples/`:

```sh
uv run find-alan-prepare-examples
```

Replace these with real images for meaningful results.

### Step 2 — Run inpainting

Models (~25 GB) are downloaded from Hugging Face on first run. Both models are gated — accept the licence at huggingface.co/black-forest-labs/FLUX.1-Fill-dev and huggingface.co/black-forest-labs/FLUX.1-Redux-dev, then run `huggingface-cli login`.

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
