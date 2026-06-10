# find-alan

Experiments for composing image generation and refinement scripts.

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
```

```sh
uv build
```
