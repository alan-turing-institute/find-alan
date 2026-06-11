"""Full-coverage tiled refinement with Flux inpainting."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
import random
import time
import traceback
from typing import Any

from find_alan.constants import NEGATIVE_PROMPT_01, POSITIVE_PROMPT_01


DEFAULT_INPUT_PATH = Path(
    "/lus/lfs1aip2/projects/u6ge/greder/bare-coordinate/outputs/upscaled/image_refined.png"
)
DEFAULT_OUTPUT_DIR = Path(
    "/lus/lfs1aip2/projects/u6ge/greder/bare-coordinate/outputs/full_coverage_v5"
)
DEFAULT_FLUX_MODEL_ID = "black-forest-labs/FLUX.1-dev"

DEFAULT_REFINEMENT_PROMPT = POSITIVE_PROMPT_01
DEFAULT_REFINEMENT_NEGATIVE_PROMPT = NEGATIVE_PROMPT_01


class MissingMLDependencies(RuntimeError):
    """Raised when optional ML dependencies are not installed."""


@dataclass(frozen=True)
class RefinementPatch:
    """A refinement patch with both context and writable inner bounds."""

    patch_box: tuple[int, int, int, int]
    inner_box: tuple[int, int, int, int]
    at_left: bool
    at_right: bool
    at_top: bool
    at_bottom: bool


@dataclass(frozen=True)
class RefinementShift:
    """A named pass-grid shift in image pixel coordinates."""

    x: int
    y: int
    name: str


@dataclass(frozen=True)
class TiledRefinementConfig:
    """Settings for a full-coverage tiled refinement run."""

    input_path: Path = DEFAULT_INPUT_PATH
    output_dir: Path = DEFAULT_OUTPUT_DIR
    model_id: str = DEFAULT_FLUX_MODEL_ID
    prompt: str = DEFAULT_REFINEMENT_PROMPT
    negative_prompt: str = DEFAULT_REFINEMENT_NEGATIVE_PROMPT
    outer_size: int = 512
    inner_ratio: float = 0.5
    feather: int = 4
    iterations: int = 4
    max_batch_size: int = 12
    strength: float = 0.2
    steps: int = 28
    guidance_scale: float = 3.5
    seed: int = 42
    device: str = "cuda:0"
    torch_dtype: str = "bfloat16"
    gif_frame_duration: int = 400
    visualization_width: int = 800
    save_gif: bool = True
    save_comparison: bool = True
    continue_on_batch_error: bool = True


@dataclass(frozen=True)
class TiledRefinementResult:
    """Paths written by a tiled refinement run."""

    output_dir: Path
    final_path: Path
    comparison_path: Path | None
    gif_path: Path | None
    iteration_paths: tuple[Path, ...]


def refinement_geometry(outer_size: int, inner_ratio: float) -> tuple[int, int]:
    """Return the inner patch size and context offset."""

    if outer_size <= 0:
        raise ValueError("outer_size must be positive")
    if inner_ratio <= 0:
        raise ValueError("inner_ratio must be positive")

    inner_size = int(outer_size * inner_ratio)
    if inner_size <= 0 or inner_size > outer_size:
        raise ValueError("inner_ratio must produce an inner size within outer_size")

    offset = (outer_size - inner_size) // 2
    return inner_size, offset


def default_refinement_shifts(inner_size: int) -> tuple[RefinementShift, ...]:
    """Return the four pass-grid shifts used by the original refinement script."""

    half_inner = inner_size // 2
    return (
        RefinementShift(0, 0, "aligned"),
        RefinementShift(half_inner, 0, "shift_right"),
        RefinementShift(0, half_inner, "shift_down"),
        RefinementShift(half_inner, half_inner, "shift_diagonal"),
    )


def build_refinement_patches(
    width: int,
    height: int,
    outer_size: int,
    inner_size: int,
    offset: int,
    shift_x: int = 0,
    shift_y: int = 0,
) -> list[RefinementPatch]:
    """Generate edge-aware refinement patches for one shifted grid."""

    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")

    patches: list[RefinementPatch] = []
    y = shift_y
    while y < height:
        x = shift_x
        while x < width:
            inner_x2 = min(x + inner_size, width)
            inner_y2 = min(y + inner_size, height)

            px1 = max(0, x - offset)
            py1 = max(0, y - offset)
            px2 = min(width, inner_x2 + offset)
            py2 = min(height, inner_y2 + offset)

            patches.append(
                RefinementPatch(
                    patch_box=(px1, py1, px2, py2),
                    inner_box=(x, y, inner_x2, inner_y2),
                    at_left=px1 == 0,
                    at_right=px2 == width,
                    at_top=py1 == 0,
                    at_bottom=py2 == height,
                )
            )
            x += inner_size
        y += inner_size

    return patches


def boxes_overlap(
    first: tuple[int, int, int, int], second: tuple[int, int, int, int]
) -> bool:
    """Return True when two boxes overlap."""

    return (
        first[0] < second[2]
        and first[2] > second[0]
        and first[1] < second[3]
        and first[3] > second[1]
    )


def build_random_batches(
    patches: list[RefinementPatch], max_batch_size: int, rng: random.Random
) -> list[list[RefinementPatch]]:
    """Shuffle patches and greedily pack non-overlapping mini-batches."""

    if max_batch_size <= 0:
        raise ValueError("max_batch_size must be positive")

    shuffled = patches.copy()
    rng.shuffle(shuffled)

    batches: list[list[RefinementPatch]] = []
    remaining = list(shuffled)

    while remaining:
        batch: list[RefinementPatch] = []
        still_remaining: list[RefinementPatch] = []

        for index, patch in enumerate(remaining):
            if not any(boxes_overlap(patch.patch_box, item.patch_box) for item in batch):
                batch.append(patch)
                if len(batch) >= max_batch_size:
                    still_remaining.extend(remaining[index + 1 :])
                    break
            else:
                still_remaining.append(patch)

        batches.append(batch)
        remaining = still_remaining

    return batches


def make_refinement_mask(
    patch: RefinementPatch,
    outer_size: int,
    inner_size: int,
    offset: int,
    feather: int,
    *,
    image_module: Any | None = None,
    image_draw_module: Any | None = None,
    image_filter_module: Any | None = None,
) -> Any:
    """Create the edge-aware writable mask for one refinement patch."""

    if image_module is None or image_draw_module is None or image_filter_module is None:
        image_module, image_draw_module, image_filter_module = _import_image_stack()

    x1 = 0 if patch.at_left else offset
    y1 = 0 if patch.at_top else offset
    x2 = outer_size if patch.at_right else offset + inner_size
    y2 = outer_size if patch.at_bottom else offset + inner_size

    mask = image_module.new("L", (outer_size, outer_size), 0)
    draw = image_draw_module.Draw(mask)
    draw.rectangle([x1, y1, x2, y2], fill=255)
    if feather > 0:
        mask = mask.filter(image_filter_module.GaussianBlur(feather))
    return mask


def run_tiled_refinement(
    config: TiledRefinementConfig,
    *,
    pipe: Any | None = None,
    progress: Callable[[str], None] | None = print,
) -> TiledRefinementResult:
    """Run iterative tiled refinement and return the generated artifact paths."""

    _validate_config(config)
    ml = _import_refinement_stack()
    np = ml["np"]
    torch = ml["torch"]
    Image = ml["Image"]
    ImageDraw = ml["ImageDraw"]
    ImageFilter = ml["ImageFilter"]
    FluxInpaintPipeline = ml["FluxInpaintPipeline"]
    lanczos = _lanczos(Image)

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    _emit(progress, f"Loading: {config.input_path}")
    image = Image.open(config.input_path).convert("RGB")
    width, height = image.size
    _emit(progress, f"Image size: {width}x{height}")

    current_image = image.copy()
    inner_size, offset = refinement_geometry(config.outer_size, config.inner_ratio)
    shifts = default_refinement_shifts(inner_size)
    _emit(progress, f"Outer={config.outer_size} Inner={inner_size} Offset={offset}")

    _emit(progress, "")
    _emit(progress, "Verifying coverage per grid...")
    for shift in shifts:
        patches = build_refinement_patches(
            width,
            height,
            config.outer_size,
            inner_size,
            offset,
            shift.x,
            shift.y,
        )
        uncovered = _uncovered_pixel_count(width, height, patches, np)
        mark = "ok" if uncovered == 0 else "missing"
        _emit(progress, f"  {shift.name}: {uncovered} uncovered pixels {mark}")

    if pipe is None:
        dtype = _torch_dtype(torch, config.torch_dtype)
        _emit(progress, "")
        _emit(progress, f"Loading FluxInpaintPipeline ({config.model_id})...")
        pipe = FluxInpaintPipeline.from_pretrained(
            config.model_id,
            torch_dtype=dtype,
        ).to(config.device)
        _emit(progress, "Pipeline loaded")

    _emit(progress, "")
    _emit(progress, "=" * 60)
    _emit(progress, "Full coverage v5 - random mini-batch sampling")
    _emit(progress, f"  Iterations:    {config.iterations}")
    _emit(progress, f"  Max batch:     {config.max_batch_size}")
    _emit(progress, f"  Strength:      {config.strength}")
    _emit(progress, f"  Grid per iter: cycles through {len(shifts)} offsets")
    _emit(progress, "  Edge-aware:    yes (corners/edges extend to boundary)")
    _emit(progress, "  Bias:          none (random shuffle each iteration)")
    _emit(progress, "=" * 60)
    _emit(progress, "")

    viz_frames: list[Any] | None = [] if config.save_gif else None
    iteration_paths: list[Path] = []
    current_image.save(output_dir / "iter_000_original.png")

    if viz_frames is not None:
        patches_init = build_refinement_patches(
            width, height, config.outer_size, inner_size, offset, 0, 0
        )
        viz_frames.append(
            _make_viz_frame(
                current_image,
                patches_init,
                [],
                [],
                0,
                0,
                0,
                "start",
                width,
                height,
                config,
                Image,
                ImageDraw,
                lanczos,
            )
        )

    for iter_idx in range(config.iterations):
        iter_num = iter_idx + 1
        shift = shifts[iter_idx % len(shifts)]
        rng = random.Random(config.seed + iter_idx)

        patches = build_refinement_patches(
            width,
            height,
            config.outer_size,
            inner_size,
            offset,
            shift.x,
            shift.y,
        )
        n_patches = len(patches)
        batches = build_random_batches(patches, config.max_batch_size, rng)
        n_batches = len(batches)

        _emit(progress, "")
        _emit(progress, "-" * 50)
        _emit(
            progress,
            f"Iter {iter_num}/{config.iterations}: grid={shift.name} "
            f"{n_patches} patches -> {n_batches} mini-batches",
        )
        _emit(progress, f"  Batch sizes: {[len(batch) for batch in batches]}")
        _emit(progress, "-" * 50)

        iter_dir = output_dir / f"iter{iter_num:02d}_{shift.name}"
        iter_dir.mkdir(exist_ok=True)

        done_indices: list[int] = []
        started_at = time.time()

        for batch_idx, batch in enumerate(batches):
            batch_seed = config.seed + iter_idx * 1000 + batch_idx
            generator = torch.Generator(device=config.device).manual_seed(batch_seed)
            batch_started_at = time.time()
            patch_id_map = {id(patch): index for index, patch in enumerate(patches)}
            active_indices = [patch_id_map[id(patch)] for patch in batch]

            if viz_frames is not None:
                viz_frames.append(
                    _make_viz_frame(
                        current_image,
                        patches,
                        done_indices,
                        active_indices,
                        iter_num,
                        batch_idx + 1,
                        n_batches,
                        shift.name,
                        width,
                        height,
                        config,
                        Image,
                        ImageDraw,
                        lanczos,
                    )
                )

            patch_images = []
            for patch in batch:
                patch_image = current_image.crop(patch.patch_box)
                if patch_image.size != (config.outer_size, config.outer_size):
                    patch_image = patch_image.resize(
                        (config.outer_size, config.outer_size), lanczos
                    )
                patch_images.append(patch_image)

            masks = [
                make_refinement_mask(
                    patch,
                    config.outer_size,
                    inner_size,
                    offset,
                    config.feather,
                    image_module=Image,
                    image_draw_module=ImageDraw,
                    image_filter_module=ImageFilter,
                )
                for patch in batch
            ]

            try:
                results = pipe(
                    prompt=[config.prompt] * len(batch),
                    negative_prompt=[config.negative_prompt] * len(batch),
                    image=patch_images,
                    mask_image=masks,
                    height=config.outer_size,
                    width=config.outer_size,
                    strength=config.strength,
                    num_inference_steps=config.steps,
                    guidance_scale=config.guidance_scale,
                    generator=generator,
                ).images

                current_image = _write_back_batch(
                    current_image,
                    results,
                    batch,
                    config.outer_size,
                    inner_size,
                    offset,
                    config.feather,
                    Image,
                    ImageDraw,
                    ImageFilter,
                    np,
                    lanczos,
                )
                done_indices.extend(active_indices)

            except Exception:
                _emit(progress, f"  Batch {batch_idx + 1} failed")
                traceback.print_exc()
                done_indices.extend(active_indices)
                if not config.continue_on_batch_error:
                    raise
                continue

            elapsed = time.time() - batch_started_at
            _emit(
                progress,
                f"  Batch {batch_idx + 1:3d}/{n_batches}  "
                f"size={len(batch):2d}  {elapsed:.1f}s",
            )

        iter_path = output_dir / f"iter{iter_num:02d}_{shift.name}_result.png"
        current_image.save(iter_path)
        iteration_paths.append(iter_path)
        _emit(
            progress,
            f"  Iteration {iter_num} done in {time.time() - started_at:.1f}s "
            f"-> {iter_path.name}",
        )

        if viz_frames is not None:
            viz_frames.append(
                _make_viz_frame(
                    current_image,
                    patches,
                    list(range(n_patches)),
                    [],
                    iter_num,
                    n_batches,
                    n_batches,
                    shift.name,
                    width,
                    height,
                    config,
                    Image,
                    ImageDraw,
                    lanczos,
                )
            )

    final_path = output_dir / "final_result.png"
    current_image.save(final_path)
    _emit(progress, "")
    _emit(progress, f"Final -> {final_path}")

    comparison_path = None
    if config.save_comparison:
        comparison_path = output_dir / "before_after.png"
        _save_before_after(
            image,
            current_image,
            comparison_path,
            config.iterations,
            Image,
            ImageDraw,
            lanczos,
        )

    gif_path = None
    if viz_frames:
        _emit(progress, "")
        _emit(progress, f"Generating GIF ({len(viz_frames)} frames)...")
        gif_path = output_dir / "patch_progression.gif"
        viz_frames[0].save(
            gif_path,
            save_all=True,
            append_images=viz_frames[1:],
            duration=config.gif_frame_duration,
            loop=0,
            optimize=True,
        )
        _emit(progress, f"GIF -> {gif_path}")

    _emit(progress, "")
    _emit(progress, "-- Done ----------------------------------------------------")
    _emit(progress, "  final_result.png       <- final image")
    if comparison_path is not None:
        _emit(progress, "  before_after.png       <- comparison")
    if gif_path is not None:
        _emit(progress, "  patch_progression.gif  <- animated viz (random batch order)")
    _emit(progress, "  iter*/                 <- per-iteration saves")
    _emit(progress, "")
    _emit(progress, "Tuning:")
    _emit(progress, "  --iterations 8         -> more refinement passes")
    _emit(progress, "  --max-batch-size 8     -> smaller batches, more randomness")
    _emit(progress, "  --strength 0.15        -> even more subtle refinement")

    return TiledRefinementResult(
        output_dir=output_dir,
        final_path=final_path,
        comparison_path=comparison_path,
        gif_path=gif_path,
        iteration_paths=tuple(iteration_paths),
    )


def _validate_config(config: TiledRefinementConfig) -> None:
    refinement_geometry(config.outer_size, config.inner_ratio)
    if config.feather < 0:
        raise ValueError("feather must be non-negative")
    if config.iterations <= 0:
        raise ValueError("iterations must be positive")
    if config.max_batch_size <= 0:
        raise ValueError("max_batch_size must be positive")
    if config.steps <= 0:
        raise ValueError("steps must be positive")
    if config.gif_frame_duration <= 0:
        raise ValueError("gif_frame_duration must be positive")
    if config.visualization_width <= 0:
        raise ValueError("visualization_width must be positive")


def _import_image_stack() -> tuple[Any, Any, Any]:
    try:
        Image = import_module("PIL.Image")
        ImageDraw = import_module("PIL.ImageDraw")
        ImageFilter = import_module("PIL.ImageFilter")
    except ImportError as exc:
        raise MissingMLDependencies(
            "Install the optional image stack with `uv sync --extra ml`."
        ) from exc
    return Image, ImageDraw, ImageFilter


def _import_refinement_stack() -> dict[str, Any]:
    try:
        np = import_module("numpy")
        torch = import_module("torch")
        diffusers = import_module("diffusers")
        Image = import_module("PIL.Image")
        ImageDraw = import_module("PIL.ImageDraw")
        ImageFilter = import_module("PIL.ImageFilter")
    except ImportError as exc:
        raise MissingMLDependencies(
            "Install the optional image stack with `uv sync --extra ml`. "
            "For CUDA-specific PyTorch wheels, install torch from the PyTorch index first."
        ) from exc

    return {
        "np": np,
        "torch": torch,
        "FluxInpaintPipeline": getattr(diffusers, "FluxInpaintPipeline"),
        "Image": Image,
        "ImageDraw": ImageDraw,
        "ImageFilter": ImageFilter,
    }


def _torch_dtype(torch: Any, dtype_name: str) -> Any:
    dtype = getattr(torch, dtype_name, None)
    if dtype is None:
        raise ValueError(f"Unknown torch dtype: {dtype_name}")
    return dtype


def _lanczos(Image: Any) -> Any:
    resampling = getattr(Image, "Resampling", None)
    if resampling is not None:
        return resampling.LANCZOS
    return Image.LANCZOS


def _emit(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)


def _uncovered_pixel_count(
    width: int, height: int, patches: list[RefinementPatch], np: Any
) -> int:
    covered = np.zeros((height, width), dtype=np.int32)
    for patch in patches:
        ix1, iy1, ix2, iy2 = patch.inner_box
        covered[iy1:iy2, ix1:ix2] += 1
    return int((covered == 0).sum())


def _write_back_batch(
    current_image: Any,
    results: list[Any],
    batch_patches: list[RefinementPatch],
    outer_size: int,
    inner_size: int,
    offset: int,
    feather: int,
    Image: Any,
    ImageDraw: Any,
    ImageFilter: Any,
    np: Any,
    lanczos: Any,
) -> Any:
    new_image = current_image.copy()
    for result, patch in zip(results, batch_patches, strict=False):
        px1, py1, px2, py2 = patch.patch_box
        actual_w = px2 - px1
        actual_h = py2 - py1

        mask = make_refinement_mask(
            patch,
            outer_size,
            inner_size,
            offset,
            feather,
            image_module=Image,
            image_draw_module=ImageDraw,
            image_filter_module=ImageFilter,
        )

        if result.size != (actual_w, actual_h):
            result = result.resize((actual_w, actual_h), lanczos)
            mask = mask.resize((actual_w, actual_h), lanczos)

        orig_arr = np.array(current_image.crop(patch.patch_box)).astype(float)
        result_arr = np.array(result).astype(float)
        alpha = np.array(mask).astype(float)[:, :, np.newaxis] / 255.0
        blended = (orig_arr * (1 - alpha) + result_arr * alpha).astype(np.uint8)
        new_image.paste(Image.fromarray(blended), (px1, py1))

    return new_image


def _make_viz_frame(
    current_image: Any,
    all_patches: list[RefinementPatch],
    done_indices: list[int],
    active_indices: list[int],
    iter_num: int,
    batch_num: int,
    total_batches: int,
    shift_name: str,
    width: int,
    height: int,
    config: TiledRefinementConfig,
    Image: Any,
    ImageDraw: Any,
    lanczos: Any,
) -> Any:
    scale = config.visualization_width / max(width, height)
    viz_w = int(width * scale)
    viz_h = int(height * scale)

    thumb = current_image.resize((viz_w, viz_h), lanczos).convert("RGBA")
    overlay = Image.new("RGBA", (viz_w, viz_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    active_set = set(active_indices)
    done_set = set(done_indices)

    for index, patch in enumerate(all_patches):
        ix1, iy1, ix2, iy2 = patch.inner_box
        vix1 = int(ix1 * scale)
        viy1 = int(iy1 * scale)
        vix2 = int(ix2 * scale)
        viy2 = int(iy2 * scale)

        if index in active_set:
            draw.rectangle(
                [vix1, viy1, vix2, viy2],
                fill=(255, 50, 50, 200),
                outline=(255, 50, 50, 255),
                width=2,
            )
        elif index in done_set:
            draw.rectangle([vix1, viy1, vix2, viy2], fill=(50, 200, 50, 60))
        else:
            draw.rectangle(
                [vix1, viy1, vix2, viy2],
                outline=(150, 150, 255, 160),
                width=1,
            )

    viz = Image.alpha_composite(thumb, overlay).convert("RGB")
    bar = Image.new("RGB", (viz_w, 38), (20, 20, 20))
    bar_draw = ImageDraw.Draw(bar)
    bar_draw.text(
        (6, 4),
        f"Iter {iter_num}/{config.iterations}  Grid: {shift_name}  "
        f"Batch {batch_num}/{total_batches}",
        fill=(255, 220, 100),
    )
    bar_draw.text(
        (6, 20),
        "Red=active  Green=done  Blue=pending  "
        f"strength={config.strength}  batch_size<={config.max_batch_size}",
        fill=(160, 160, 160),
    )

    frame = Image.new("RGB", (viz_w, viz_h + 38))
    frame.paste(bar, (0, 0))
    frame.paste(viz, (0, 38))
    return frame


def _save_before_after(
    original: Any,
    refined: Any,
    output_path: Path,
    iterations: int,
    Image: Any,
    ImageDraw: Any,
    lanczos: Any,
) -> None:
    width, height = original.size
    comp_w = 1024
    comp_h = int(height * comp_w / width)
    comp = Image.new("RGB", (comp_w * 2 + 10, comp_h + 30), (25, 25, 25))
    comp.paste(original.resize((comp_w, comp_h), lanczos), (0, 30))
    comp.paste(refined.resize((comp_w, comp_h), lanczos), (comp_w + 10, 30))
    draw = ImageDraw.Draw(comp)
    draw.text((comp_w // 2 - 40, 8), "ORIGINAL", fill=(255, 255, 255))
    draw.text(
        (comp_w + 10 + comp_w // 2 - 50, 8),
        f"AFTER {iterations} ITERS",
        fill=(255, 255, 255),
    )
    comp.save(output_path)
