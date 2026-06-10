# find-alan

Python package skeleton managed with `uv`.

## Setup

Install the package environment:

```sh
uv sync
```

## Usage

There are two ways to insert a figure into a crowd scene, differing in model requirements and how much control you want over placement.

### Option A — FLUX.2-Klein: maskless insertion (recommended)

Uses `FLUX.2-klein-4B` (~13 GB) with a dedicated `image_reference` parameter. No mask needed — the model places the figure based on the prompt and scene context.

#### Step 1 — Accept the model licence and log in

Accept the licence at huggingface.co/black-forest-labs/FLUX.2-klein-4B, then:

```sh
huggingface-cli login
```

#### Step 2 — Generate example images

```sh
uv run find-alan-prepare-examples
```

#### Step 3 — Run insertion

```sh
uv run find-alan-insert \
  --scene examples/crowd_scene.png \
  --figure examples/figure.png \
  --output examples/result.png \
  --seed 42
```

Key options:

| Flag | Default | Notes |
| --- | --- | --- |
| `--strength` | `0.85` | How much the scene is allowed to change (0–1). Lower = preserve more. |
| `--guidance-scale` | `8.0` | Prompt adherence. |
| `--steps` | `50` | Inference steps. |
| `--prompt` | *(see code)* | Text describing placement and blending. |

```sh
uv run find-alan-insert --help
```

---

### Option B — FLUX.1-Redux + FLUX.1-Fill: mask-based inpainting

Uses `FLUX.1-Redux-dev` + `FLUX.1-Fill-dev` (~25 GB combined). You supply a mask that marks exactly where the figure is inserted; the Redux prior encodes the reference figure as visual tokens that condition the fill.

Both models are gated — accept the licence at huggingface.co/black-forest-labs/FLUX.1-Fill-dev and huggingface.co/black-forest-labs/FLUX.1-Redux-dev, then run `huggingface-cli login`.

#### Step 1 — Generate example images

```sh
uv run find-alan-prepare-examples
```

#### Step 2 — Run inpainting

With a mask file (white = inpaint, black = keep):

```sh
uv run find-alan-inpaint \
  --scene examples/crowd_scene.png \
  --figure examples/figure.png \
  --mask examples/mask.png \
  --output examples/result.png \
  --seed 42
```

Or with a bounding box instead:

```sh
uv run find-alan-inpaint \
  --scene examples/crowd_scene.png \
  --figure examples/figure.png \
  --bbox 210 330 90 150 \
  --output examples/result.png
```

```sh
uv run find-alan-inpaint --help
```
