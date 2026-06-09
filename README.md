# find-alan

Python scripts package managed with `uv`.

## Setup

Install the runnable scripts:

```sh
uv sync
```

Install the optional diffusion/image stack:

```sh
uv sync --extra ml
```

For an NVIDIA CUDA setup, install the PyTorch wheels that match your driver/CUDA runtime before syncing the ML extra.

## Usage

Show script options:

```sh
uv run find-alan-crop-plan --help
uv run find-alan-upscale --help
```

Preview jittered crop grids for a custom MultiDiffusion loop:

```sh
uv run find-alan-crop-plan --width 320 --height 240 --scale 10 --steps 4
```

Run the starter tiled diffusion upscaler:

```sh
uv run find-alan-upscale input.png output.png --scale 4
```

Add more runnable scripts by creating modules under `src/find_alan/scripts/`
with a `main()` function, then adding them to `[project.scripts]` in
`pyproject.toml`:

```toml
[project.scripts]
find-alan-crop-plan = "find_alan.scripts.crop_plan:main"
find-alan-upscale = "find_alan.scripts.upscale:main"
find-alan-new-script = "find_alan.scripts.new_script:main"
```

Try lower `--denoising-strength` values when preserving the original scene matters. Use a separate masked inpaint pass for the final hidden character/face corrections.

Build the package:

```sh
uv build
```
