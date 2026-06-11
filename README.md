# find-alan

# Upscaling

Experiments for diffusion upscaling low-resolution images with tiled pipelines, a custom MultiDiffusion path, and Flux.2 reference-conditioned tiles.

See [upscalers.md](upscalers.md) for the current upscaler architecture guide,
engine tradeoffs, and result-informed next experiments.

## Setup

```sh
uv sync
uv sync --extra ml
```

The first real generation downloads model weights from Hugging Face unless they are already cached. Set `HF_TOKEN` for higher Hub rate limits.

## Commands

```sh
uv run find-alan-refine --help
```

## Tiled Refinement

`find-alan-refine` runs the full-coverage iterative refinement pass from `full_coverage_v5.py`, packaged as an importable module and CLI.

The refinement pass:

1. Opens an existing image.
2. Builds shifted patch grids with edge-aware writable masks.
3. Randomly packs non-overlapping patches into mini-batches.
4. Runs Flux inpainting over each mini-batch.
5. Blends the refined patch interiors back into the working image.
6. Saves per-iteration outputs, a final image, a before/after comparison, and a patch progression GIF.

```sh
uv run find-alan-refine input.png outputs/refined \
  --iterations 4 \
  --max-batch-size 12 \
  --strength 0.2
```

The old environment variables still work as defaults for the CLI:

- `INPUT_IMAGE`
- `OUTPUT_DIR`
- `NUM_ITERS`
- `MAX_BATCH_SIZE`
- `STRENGTH`

For scripts that combine stages, use the package API:

```python
from pathlib import Path

from find_alan.refinement import TiledRefinementConfig, run_tiled_refinement

result = run_tiled_refinement(
    TiledRefinementConfig(
        input_path=Path("input.png"),
        output_dir=Path("outputs/refined"),
    )
)
print(result.final_path)
```

## Build

Run static checks when dev dependencies are installed:

```sh
uv run ty check
uv run find-alan-upscale --help
uv run find-alan-crop-plan --help
```

## How The Upscaling Works

The upscaler has a few separate jobs that are easy to mix together:

1. **Image-to-image denoising** starts from a resized version of the low-resolution input. `--denoising-strength` controls how much noise is added before the model redraws it. Higher values give the model more room to invent new detail.
2. **ControlNet** feeds the source image back into the model as a spatial constraint. It says: keep this composition, these edges, this local texture, and this rough object placement. `--controlnet-strength` controls how strongly that constraint is enforced.
3. **Tiling/MultiDiffusion** is about scale and seams. Large images are too big to denoise in one pass, so the model denoises overlapping latent crops and blends them into one canvas.
4. **Flux.2 reference conditioning** is different from ControlNet. Flux.2 accepts image inputs as reference/context tokens, so the `flux2-tile` engine gives each target crop to Flux.2 as a reference image and asks it to redraw that crop at the same pixel size.

ControlNet and MultiDiffusion solve different problems. ControlNet controls **what the generated image should stay aligned to**. MultiDiffusion controls **how many overlapping windows are combined into a seamless large image**.

For more hallucinated detail, raise `--denoising-strength` and lower `--controlnet-strength`. For a faithful upscale, lower denoising and raise ControlNet strength.

For `flux2-tile`, there is no SDXL ControlNet and no global latent canvas. Faithfulness comes from the per-tile reference image plus the prompt. Seam reduction comes from overlap and gaussian blending.

## Pipeline Diagrams

```mermaid
flowchart TD
    Base["Low-resolution crowd image"] --> Resize["Resize to target scale"]
    Resize --> Engine{"Choose upscale engine"}
    Engine --> ModTile["mod-tile"]
    Engine --> Multi["multidiffusion"]
    Engine --> Flux["flux2-tile"]
    Engine --> FluxMD["flux2-multidiffusion"]
    Engine --> SD3["sd3-tile"]
    ModTile --> Blend["Tile, condition, and blend"]
    Multi --> Blend
    Flux --> Blend
    FluxMD --> Blend
    SD3 --> Blend
    Blend --> BaseOut["Upscaled crowd base"]
    BaseOut --> Review["Review seams, detail, and layout"]
    Review --> Local["Optional local object insertion or repair"]
    Local --> Final["Final image"]
```

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

### `flux2-tile`

Experimental engine for Flux.2. It does not reuse the SDXL ControlNet or latent MultiDiffusion loop, because Flux.2 is a DiT pipeline with image/reference conditioning. Instead, it:

1. Resizes the input image to the target scale.
2. Splits the resized image into overlapping pixel crops.
3. Sends each crop to Flux.2 as the reference image.
4. Prompts Flux.2 to faithfully redraw that reference crop.
5. Blends the generated crops back together with gaussian weights.

Best for: A100 trials where Flux.2 quality is more important than strict SDXL ControlNet-style fidelity. `--denoising-strength` and `--controlnet-strength` are not used by this engine.

Detailed flow:

1. Open the source image as RGB.
2. Compute the scaled output size and round it to a multiple of 16, matching Flux.2/VAE packing constraints.
3. Resize the source image to that final output size with Lanczos filtering.
4. Round `--flux2-tile-size` up to a multiple of 16 and clamp `--flux2-overlap` so it is smaller than the tile.
5. Build a full-cover crop grid. The grid always includes the top-left and bottom-right bounds, so edge pixels are covered even when the image size is not an exact multiple of the stride.
6. For each crop, cut the resized image and pass that crop to Flux.2 as `image=...` with `height` and `width` set to the crop size.
7. Add a faithfulness instruction to the user prompt, including a reminder to preserve composition, linework, colors, viewpoint, and crowd layout.
8. Generate one tile independently. The engine uses bf16 on CUDA and fp32 on CPU.
9. Convert the generated tile to float RGB and multiply it by a gaussian weight map. The center of the tile contributes more strongly than the edges.
10. Accumulate weighted tile pixels into one output canvas and accumulate the matching weights.
11. Normalize `canvas / weights`, clip to RGB, and save the final image.

The tradeoff is important: because Flux.2 tiles are sampled independently, this engine is simpler than latent MultiDiffusion but has less global coordination. Increase `--flux2-overlap` when seams are visible. Increase `--flux2-tile-size` when objects need more surrounding context. Use `--flux2-jitter` to change the crop alignment in a deterministic seed-controlled way.

Model selection:

- Default: `black-forest-labs/FLUX.2-dev`.
- `--flux2-pipeline auto` selects the Diffusers pipeline from the model id.
- Use `--flux2-pipeline dev` for `Flux2Pipeline`.
- Use `--flux2-pipeline klein` for `Flux2KleinPipeline`.
- Use `--flux2-pipeline klein-kv` for `Flux2KleinKVPipeline`.

```sh
uv run find-alan-upscale input.png output.png \
  --engine flux2-tile \
  --scale 2 \
  --steps 50 \
  --guidance-scale 4 \
  --flux2-tile-size 1024 \
  --flux2-overlap 256 \
  --no-cpu-offload
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

`--flux2-tile-size`: pixel crop size for `flux2-tile`. Larger tiles give Flux.2 more context, but require more VRAM and time.

`--flux2-overlap`: pixel overlap between Flux.2 tiles. Higher overlap gives the gaussian blend more room to hide seams.

`--flux2-jitter`: optional maximum random tile-grid offset for Flux.2. This is deterministic with `--seed`.

`--flux2-pipeline`: selects the Diffusers Flux.2 pipeline class. `auto` is usually enough unless the model id does not clearly name the variant.

`--flux2-caption-upsample-temperature`: optional Flux.2 prompt upsampling temperature. Leave unset for the local prompt as written.

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
Use a separate local pass for final object insertion and local corrections.

## Object Insertion

The current upscaling engines should stay focused on making a strong base image. Do not put object-specific language into the global upscale prompt, because it can create false positives or repeated motifs across the crowd.

The object insertion stage should stay separate from global upscaling, but the exact approach is intentionally not fixed yet. It might use masked inpainting, local img2img repair, compositing, or a small engine-specific workflow once the base image quality is clear.

For now, treat the global output as an engine-flexible base image. After choosing the best base, use a local pass around the target region so the object can be inserted or repaired without changing the whole crowd scene.

Avoid running a full-image upscale or redraw after inserting the object, because that could smear it, duplicate it, or change the hiding location.

## Current Upscale Experiment Settings

### Flux2 source images, SDXL MultiDiffusion 3x display batch

The Flux2-style source images in `data/examples/lr/flux2` are being upscaled for large-screen review with SDXL ControlNet MultiDiffusion, not Flux2. The queued batch writes to `data/examples/out/flux2`.

Queued batch suffix:

```text
c31d3a8a_rb826e0f
```

Output naming pattern:

```text
data/examples/out/flux2/<source_stem>_sdxl_md_3x_upscaleprompt_d075_c035_tile1024_c31d3a8a_rb826e0f.png
```

Settings:

```sh
uv run find-alan-upscale input.png output.png \
  --engine multidiffusion \
  --scale 3 \
  --steps 28 \
  --denoising-strength 0.75 \
  --controlnet-strength 0.35 \
  --guidance-scale 4.5 \
  --md-tile-size 1024 \
  --md-overlap 512 \
  --md-jitter 256
```

Rationale:

- `3x` turns the `1920x1072` sources into roughly `5760x3216`, which is better suited to large-screen display than `2x`.
- `1024` tiles with `512` overlap keep the setting consistent with the best conference SDXL MultiDiffusion runs and prioritize seam control.
- `denoising-strength 0.75` and `controlnet-strength 0.35` are the current preferred balance for adding detail while keeping the crowd layout anchored.
- Refinement is intentionally not queued for this batch yet; inspect the 3x bases first, then refine selected outputs.

### Refinement pass after base upscale

Use `find-alan-refine` only after a base upscale has been selected or when a specific comparison needs polishing. The refinement stage is a tiled Flux inpaint pass: each patch sees a `512x512` context window and writes only the inner `256x256` region when `--inner-ratio 0.5` is used.

Default comparison refinement settings:

```sh
uv run find-alan-refine base.png output_dir \
  --iterations 4 \
  --strength 0.2 \
  --steps 28 \
  --guidance-scale 3.5 \
  --outer-size 512 \
  --inner-ratio 0.5 \
  --feather 4 \
  --max-batch-size 12
```

Output directory naming pattern:

```text
data/examples/out/<set>/refined/<base_stem>_refine_default_i4_s020_steps28_c<commit7>_r<run7>
```

For a heavier refinement stress test, increase iterations while keeping strength fixed:

```sh
uv run find-alan-refine base.png output_dir \
  --iterations 12 \
  --strength 0.2 \
  --steps 28 \
  --guidance-scale 3.5 \
  --outer-size 512 \
  --inner-ratio 0.5 \
  --feather 4 \
  --max-batch-size 12
```

`--max-batch-size 24` can improve throughput on an otherwise empty 80 GB GPU, but it is more fragile. Use `12` as the reliable default and only raise it for throughput experiments.

Operational notes:

- Keep `pueue parallel 1` for these runs unless jobs are pinned to separate GPUs.
- If refinement fails with CUDA OOM while loading the pipeline, lowering `--max-batch-size` usually will not help; that failure happens before patch batches start and usually means another process is already occupying VRAM.
- If refinement fails during patch processing, retry with a smaller batch size such as `--max-batch-size 6`.
- For 3x Flux2-source bases, inspect the base upscales first and refine only selected outputs.

# Inpainting

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


## Final find-alan-insert-detected pipeline

The command detects people in a scene image using YOLO, then uses FLUX (FLUX.2-Klein) inpainting to replace one of them with a given figure image.

For example, to generate the Alan in Venice example:

```bash
uv run find-alan-insert-detected --scene examples/final_result.png --figure examples/alan_cartoon.png --yolo-model yolov8s-worldv2  --detection-classes person --seed 111
```

What this specific command does:

```bash
--scene  examples/final_result.png    # the background/scene to modify
--figure examples/alan_cartoon.png    # the person/figure to insert
--yolo-model yolov8s-worldv2          # YOLO model for detection
--detection-classes person            # detect people```

Pipeline steps:

1. Detect — runs YOLOv8 (yolov8s-worldv2) on the scene to find all bounding boxes matching person
2. Select — picks one target person (at random)
3. Pad bbox — expands the bounding box by 20% on all sides to give inpainting context
4. Crop — extracts that padded region from the scene
5. Inpaint — loads FLUX.2-Klein (~13 GB) and runs inpainting on the crop, using the reference image to condition what gets inserted
6. Composite — resizes the inpainted crop back and blends it into the original scene with a feathered mask at the edges for smooth transitions
7. Save — writes the final image to `examples/<figure>_<scene>_<seed>.png`


Net effect: one person in final_result.png is swapped out for Alan (the cartoon figure), seamlessly composited back into the original scene.
