# find-alan

Python package skeleton managed with `uv`.

## Setup

Install the package environment:

```sh
uv sync
```

## Usage

There are three ways to insert a figure into a crowd scene, differing in model requirements and how much control you want over placement.

### Option A — FLUX.2-Klein: maskless insertion (recommended)

Uses `FLUX.2-klein-4B` with a dedicated `image_reference` parameter. No mask needed — the model places the figure based on the prompt and scene context.

Before running, accept the model licence at huggingface.co/black-forest-labs/FLUX.2-klein-4B.

####  Run insertion

```sh
uv run find-alan-insert \
  --scene <base image>.png \
  --figure <figure image>.png \
  --output <result filename>.png \
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

Uses `FLUX.1-Redux-dev` + `FLUX.1-Fill-dev. You supply a mask that marks exactly where the figure is inserted; the Redux prior encodes the reference figure as visual tokens that condition the fill.

Both models are gated — accept the licence at huggingface.co/black-forest-labs/FLUX.1-Fill-dev and huggingface.co/black-forest-labs/FLUX.1-Redux-dev.


#### Run inpainting

With a mask file (white = inpaint, black = keep):

```sh
uv run find-alan-inpaint \
  --scene <base image>.png \
  --figure <figure image>.png \
  --mask <mask image>.png \
  --output <result filename>.png \
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

---

### Option C — YOLOv8 + FLUX.1-Redux/Fill: detection-guided inpainting

Uses `YOLOv8` to detect people in the scene, selects one as the target, then inpaints the reference figure into that region with `FLUX.1-Redux-dev` + `FLUX.1-Fill-dev`. Because the mask is sized to a real crowd member the inserted figure automatically matches the correct scale and perspective.

Both FLUX models are gated — accept the licences at huggingface.co/black-forest-labs/FLUX.1-Fill-dev and huggingface.co/black-forest-labs/FLUX.1-Redux-dev before running.

#### Run detection + inpainting

```sh
uv run find-alan-insert-detected \
  --scene <base image>.png \
  --figure <figure image>.png \
  --output <result filename>.png \
  --seed 42
```

Key options:

| Flag | Default | Notes |
| --- | --- | --- |
| `--strategy` | `random` | Which detected person to replace: `random`, `largest`, `smallest`, `center`. |
| `--conf` | `0.3` | YOLO confidence threshold — lower to detect more people. |
| `--yolo-model` | `yolov8n` | YOLOv8 variant (`yolov8n/s/m/l/x`). Larger = more accurate, slower. |
| `--padding` | `0.15` | Fraction to expand the detected bbox for edge blending. |
| `--guidance-scale` | `30.0` | CFG scale. |
| `--steps` | `50` | Inference steps. |
| `--save-mask` | *(none)* | Optional path to save the generated mask for inspection. |

```sh
uv run find-alan-insert-detected --help
```
