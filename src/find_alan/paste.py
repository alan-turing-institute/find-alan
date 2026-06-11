"""Paste a reference figure (Alan) over a detected person and refine the seam.

Pipeline:
    1. detect people in the scene and pick one (random by default);
    2. crop a square region around them with a writable gap ring and a frozen
       border ring;
    3. paste the figure over the person, scaled to the person's bbox height and
       alpha-composited;
    4. build a mask whose writable area is the gap ring plus the figure below
       the upper thigh, while the figure's alpha silhouette above the thigh and
       the outer border ring stay frozen;
    5. gently inpaint only the writable area with ``FluxInpaintPipeline`` so the
       background closes around the figure without disturbing him or the wider
       scene;
    6. composite the refined crop back into the full scene (frozen pixels stay
       bit-identical) and write the result plus a JSON sidecar of the geometry.

Geometry and mask construction are pure PIL and importable without the ML
stack; only :func:`run_paste_alan` pulls in torch/diffusers (lazily).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from importlib import import_module
import json
from pathlib import Path
import random
from typing import Any

from PIL import Image, ImageFilter

Box = tuple[int, int, int, int]  # (x0, y0, x1, y1), x1/y1 exclusive


DEFAULT_INPUT_PATH = Path("outputs/improve/conference_refined.png")
DEFAULT_FIGURE_PATH = Path("assets/images/Alan.png")
DEFAULT_OUTPUT_DIR = Path("outputs/finished")
DEFAULT_FLUX_MODEL_ID = "black-forest-labs/FLUX.1-dev"

# A paste-specific prompt: the goal is to close the crowd seamlessly around a
# single inserted figure, not to restyle the whole scene.
DEFAULT_PASTE_PROMPT = (
    "Where's Wally illustration style, flat cartoon art with bold black "
    "outlines, a single character standing naturally within a dense crowd of "
    "tiny distinct people, the background crowd and floor closing seamlessly "
    "around the character, clean linework, limited bright palette of red "
    "yellow blue beige, no merged figures, no blurry regions"
)

DEFAULT_PASTE_NEGATIVE_PROMPT = (
    "blurry, smeared, merged figures, indistinct blobs, artifacts, duplicated "
    "person, distorted limbs, melted shapes, halo, photorealistic, 3d render"
)


class MissingMLDependencies(RuntimeError):
    """Raised when optional ML dependencies are not installed."""


@dataclass(frozen=True)
class PasteAlanConfig:
    """Settings for a single paste-and-refine run."""

    input_path: Path = DEFAULT_INPUT_PATH
    figure_path: Path = DEFAULT_FIGURE_PATH
    output_dir: Path = DEFAULT_OUTPUT_DIR
    output_name: str = "conference_alan"

    # Detection / target selection.
    strategy: str = "random"
    conf_threshold: float = 0.3
    yolo_model: str = "yolov8n"
    seed: int = 42

    # Crop geometry (fractions of the figure box on each side).
    gap_padding: float = 0.4
    border_padding: float = 0.2
    crop_size: int = 512  # inference resolution (multiple of 16)

    # Mask shape.
    protect_fraction: float = 0.6  # top fraction of the figure kept frozen
    alpha_threshold: int = 128
    dilate: int = 8  # grow protected silhouette before feathering
    feather: int = 8

    # Refinement (kept gentle so neighbouring figures don't pick up artifacts).
    model_id: str = DEFAULT_FLUX_MODEL_ID
    prompt: str = DEFAULT_PASTE_PROMPT
    negative_prompt: str = DEFAULT_PASTE_NEGATIVE_PROMPT
    strength: float = 0.4
    steps: int = 28
    guidance_scale: float = 3.5
    device: str = "cuda:0"
    torch_dtype: str = "bfloat16"

    save_debug: bool = False


@dataclass(frozen=True)
class PasteAlanResult:
    """Paths and geometry produced by a paste-and-refine run."""

    output_path: Path
    sidecar_path: Path
    person_bbox: Box
    paste_box: Box
    gap_box: Box
    crop_box: Box
    debug_paths: dict[str, Path] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Pure geometry / mask helpers (no ML imports, fully unit-testable).
# --------------------------------------------------------------------------- #


def compute_paste_box(bbox: Box, figure_size: tuple[int, int]) -> Box:
    """Place *figure* over *bbox*: fit bbox height, keep aspect, bottom-aligned.

    *bbox* is (x, y, w, h). Returns (x0, y0, x1, y1). The figure is centred
    horizontally on the bbox and its base sits on the bbox base, so for a
    figure scaled to the bbox height the top edge coincides with the bbox top.
    """
    x, y, w, h = bbox
    fw, fh = figure_size
    if fh <= 0 or h <= 0:
        raise ValueError("bbox height and figure height must be positive")

    paste_h = h
    paste_w = max(1, round(fw * h / fh))
    cx = x + w / 2
    px0 = round(cx - paste_w / 2)
    py0 = y + h - paste_h  # == y when paste_h == h
    return (px0, py0, px0 + paste_w, py0 + paste_h)


def _union(a: Box, b: Box) -> Box:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _expand(box: Box, frac_x: float, frac_y: float) -> Box:
    x0, y0, x1, y1 = box
    dx = (x1 - x0) * frac_x
    dy = (y1 - y0) * frac_y
    return (round(x0 - dx), round(y0 - dy), round(x1 + dx), round(y1 + dy))


def _squareify(box: Box) -> Box:
    """Grow the shorter side of *box* symmetrically so it becomes square."""
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    side = max(w, h)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return (
        round(cx - side / 2),
        round(cy - side / 2),
        round(cx + side / 2),
        round(cy + side / 2),
    )


def _shift_into_bounds(box: Box, image_size: tuple[int, int]) -> Box:
    """Translate *box* to lie within the image where possible, then clamp."""
    iw, ih = image_size
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0

    if x0 < 0:
        x0, x1 = 0, w
    if x1 > iw:
        x0, x1 = iw - w, iw
    if y0 < 0:
        y0, y1 = 0, h
    if y1 > ih:
        y0, y1 = ih - h, ih

    # If the box is larger than the image in a dimension, clamp hard.
    return (max(0, x0), max(0, y0), min(iw, x1), min(ih, y1))


def compute_crop_box(
    bbox: Box,
    paste_box: Box,
    image_size: tuple[int, int],
    gap_padding: float,
    border_padding: float,
) -> tuple[Box, Box]:
    """Return ``(crop_box, gap_box)`` around the figure.

    The *gap box* (figure + gap ring) is the writable interior; expanding it by
    *border_padding* gives the *crop box*, whose outer ring stays frozen. The
    crop is made square and shifted into the image bounds; the gap box is
    clamped to the crop so callers can translate it into crop coordinates.
    """
    figure_box = _union(bbox_to_xyxy(bbox), paste_box)
    gap_box = _expand(figure_box, gap_padding, gap_padding)
    crop_box = _expand(gap_box, border_padding, border_padding)
    crop_box = _shift_into_bounds(_squareify(crop_box), image_size)

    # Clamp the gap box inside the realised crop so it is always representable
    # in crop-local coordinates.
    gx0 = max(gap_box[0], crop_box[0])
    gy0 = max(gap_box[1], crop_box[1])
    gx1 = min(gap_box[2], crop_box[2])
    gy1 = min(gap_box[3], crop_box[3])
    return crop_box, (gx0, gy0, gx1, gy1)


def bbox_to_xyxy(bbox: Box) -> Box:
    """Convert an (x, y, w, h) bbox to (x0, y0, x1, y1)."""
    x, y, w, h = bbox
    return (x, y, x + w, y + h)


def build_writable_mask(
    crop_box: Box,
    gap_box: Box,
    paste_box: Box,
    figure_rgba: Image.Image,
    *,
    protect_fraction: float,
    alpha_threshold: int,
    dilate: int,
    feather: int,
) -> Image.Image:
    """Build the feathered writable mask (white = may change) in crop coords.

    Writable = the gap box interior, minus the figure's alpha silhouette above
    the upper-thigh line. The protected silhouette is dilated before being cut
    out so the subsequent feather does not eat into the figure's true edge.
    *figure_rgba* must already be scaled to the paste-box size.
    """
    cx0, cy0, cx1, cy1 = crop_box
    crop_w, crop_h = cx1 - cx0, cy1 - cy0

    # Start with the gap ring writable, the border ring frozen.
    mask = Image.new("L", (crop_w, crop_h), 0)
    gx0, gy0, gx1, gy1 = gap_box
    _rect(mask, (gx0 - cx0, gy0 - cy0, gx1 - cx0, gy1 - cy0), 255)

    # Protected silhouette: the figure's alpha above the thigh cut line.
    protect = _protected_silhouette(
        figure_rgba,
        paste_box,
        crop_box,
        protect_fraction=protect_fraction,
        alpha_threshold=alpha_threshold,
        dilate=dilate,
    )
    # Subtract protected pixels from the writable mask.
    from PIL import ImageChops

    mask = ImageChops.subtract(mask, protect)

    if feather > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(feather))
    return mask


def _protected_silhouette(
    figure_rgba: Image.Image,
    paste_box: Box,
    crop_box: Box,
    *,
    protect_fraction: float,
    alpha_threshold: int,
    dilate: int,
) -> Image.Image:
    """White where the figure must stay frozen, in crop coordinates."""
    cx0, cy0, cx1, cy1 = crop_box
    crop_w, crop_h = cx1 - cx0, cy1 - cy0
    px0, py0, px1, py1 = paste_box
    paste_w, paste_h = px1 - px0, py1 - py0

    alpha = figure_rgba.split()[-1]
    if alpha.size != (paste_w, paste_h):
        alpha = alpha.resize((paste_w, paste_h), Image.LANCZOS)

    # Threshold to a hard silhouette, then drop everything below the thigh line.
    silhouette = alpha.point(lambda v: 255 if v >= alpha_threshold else 0)
    cut = max(0, min(paste_h, round(paste_h * protect_fraction)))
    if cut < paste_h:
        _rect(silhouette, (0, cut, paste_w, paste_h), 0)

    if dilate > 0:
        silhouette = silhouette.filter(ImageFilter.MaxFilter(2 * dilate + 1))

    # Paste the silhouette into a crop-sized canvas at the figure's offset.
    protect = Image.new("L", (crop_w, crop_h), 0)
    protect.paste(silhouette, (px0 - cx0, py0 - cy0))
    return protect


def _rect(image: Image.Image, box: Box, fill: int) -> None:
    from PIL import ImageDraw

    ImageDraw.Draw(image).rectangle([box[0], box[1], box[2], box[3]], fill=fill)


def paste_figure(crop: Image.Image, figure_rgba: Image.Image, paste_box: Box, crop_box: Box) -> Image.Image:
    """Alpha-composite *figure* onto *crop* at the paste box (crop coords)."""
    px0, py0, px1, py1 = paste_box
    cx0, cy0, _, _ = crop_box
    scaled = figure_rgba
    if scaled.size != (px1 - px0, py1 - py0):
        scaled = scaled.resize((px1 - px0, py1 - py0), Image.LANCZOS)
    out = crop.convert("RGB").copy()
    out.paste(scaled, (px0 - cx0, py0 - cy0), scaled)
    return out


# --------------------------------------------------------------------------- #
# Orchestration (pulls in detection + the ML stack).
# --------------------------------------------------------------------------- #


def run_paste_alan(
    config: PasteAlanConfig,
    *,
    pipe: Any | None = None,
    progress: Any | None = print,
) -> PasteAlanResult:
    """Run the full paste-and-refine pipeline and return the written paths."""
    from find_alan.detect import detect_people, load_detector, pick_target

    ml = _import_ml_stack()
    torch = ml["torch"]
    FluxInpaintPipeline = ml["FluxInpaintPipeline"]

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    _emit(progress, f"Loading scene: {config.input_path}")
    scene = Image.open(config.input_path).convert("RGB")
    figure = Image.open(config.figure_path).convert("RGBA")

    _emit(progress, f"Detecting people (YOLO {config.yolo_model}, conf>={config.conf_threshold})...")
    detector = load_detector(config.yolo_model)
    bboxes = detect_people(detector, scene, conf_threshold=config.conf_threshold)
    if not bboxes:
        raise ValueError("No people detected; try lowering conf_threshold.")
    _emit(progress, f"Found {len(bboxes)} person(s); picking with strategy='{config.strategy}'")

    rng = random.Random(config.seed)
    person_bbox = pick_target(bboxes, strategy=config.strategy, scene_size=scene.size, rng=rng)

    paste_box = compute_paste_box(person_bbox, figure.size)
    crop_box, gap_box = compute_crop_box(
        person_bbox, paste_box, scene.size, config.gap_padding, config.border_padding
    )
    _emit(progress, f"person={person_bbox} paste={paste_box} crop={crop_box}")

    original_crop = scene.crop(crop_box)
    pasted_crop = paste_figure(original_crop, figure, paste_box, crop_box)
    mask = build_writable_mask(
        crop_box,
        gap_box,
        paste_box,
        figure,
        protect_fraction=config.protect_fraction,
        alpha_threshold=config.alpha_threshold,
        dilate=config.dilate,
        feather=config.feather,
    )

    if pipe is None:
        dtype = _torch_dtype(torch, config.torch_dtype)
        _emit(progress, f"Loading FluxInpaintPipeline ({config.model_id})...")
        pipe = FluxInpaintPipeline.from_pretrained(config.model_id, torch_dtype=dtype).to(config.device)

    refined_crop = _refine_crop(pasted_crop, mask, config, pipe, ml)

    # Composite the refined crop back over the pasted crop using the feathered
    # mask, so frozen pixels stay bit-identical, then drop it into the scene.
    blended_crop = Image.composite(refined_crop, pasted_crop, mask)
    result = scene.copy()
    result.paste(blended_crop, (crop_box[0], crop_box[1]))

    output_path = output_dir / f"{config.output_name}.png"
    result.save(output_path)
    _emit(progress, f"Saved -> {output_path}")

    debug_paths: dict[str, Path] = {}
    if config.save_debug:
        debug_paths = _save_debug(output_dir, config.output_name, original_crop, pasted_crop, mask, refined_crop)

    sidecar_path = output_dir / f"{config.output_name}.json"
    _write_sidecar(sidecar_path, config, person_bbox, paste_box, gap_box, crop_box, scene.size)

    return PasteAlanResult(
        output_path=output_path,
        sidecar_path=sidecar_path,
        person_bbox=person_bbox,
        paste_box=paste_box,
        gap_box=gap_box,
        crop_box=crop_box,
        debug_paths=debug_paths,
    )


def _refine_crop(
    pasted_crop: Image.Image,
    mask: Image.Image,
    config: PasteAlanConfig,
    pipe: Any,
    ml: dict[str, Any],
) -> Image.Image:
    """Gentle masked inpaint at the model resolution, returned at crop size."""
    torch = ml["torch"]
    size = config.crop_size
    crop_size = pasted_crop.size

    model_image = pasted_crop.resize((size, size), Image.LANCZOS)
    model_mask = mask.resize((size, size), Image.LANCZOS)

    generator = torch.Generator(device=config.device).manual_seed(config.seed)
    result = pipe(
        prompt=config.prompt,
        negative_prompt=config.negative_prompt,
        image=model_image,
        mask_image=model_mask,
        height=size,
        width=size,
        strength=config.strength,
        num_inference_steps=config.steps,
        guidance_scale=config.guidance_scale,
        generator=generator,
    ).images[0]
    return result.resize(crop_size, Image.LANCZOS)


def _save_debug(
    output_dir: Path,
    name: str,
    original_crop: Image.Image,
    pasted_crop: Image.Image,
    mask: Image.Image,
    refined_crop: Image.Image,
) -> dict[str, Path]:
    paths = {
        "crop_original": output_dir / f"{name}_debug_crop_original.png",
        "crop_pasted": output_dir / f"{name}_debug_crop_pasted.png",
        "mask": output_dir / f"{name}_debug_mask.png",
        "crop_refined": output_dir / f"{name}_debug_crop_refined.png",
    }
    original_crop.save(paths["crop_original"])
    pasted_crop.save(paths["crop_pasted"])
    mask.save(paths["mask"])
    refined_crop.save(paths["crop_refined"])
    return paths


def _write_sidecar(
    path: Path,
    config: PasteAlanConfig,
    person_bbox: Box,
    paste_box: Box,
    gap_box: Box,
    crop_box: Box,
    scene_size: tuple[int, int],
) -> None:
    params = asdict(config)
    for key, value in params.items():
        if isinstance(value, Path):
            params[key] = str(value)
    payload = {
        "scene_size": list(scene_size),
        "person_bbox_xywh": list(person_bbox),
        "paste_box_xyxy": list(paste_box),
        "gap_box_xyxy": list(gap_box),
        "crop_box_xyxy": list(crop_box),
        "params": params,
    }
    path.write_text(json.dumps(payload, indent=2))


def _import_ml_stack() -> dict[str, Any]:
    try:
        torch = import_module("torch")
        diffusers = import_module("diffusers")
    except ImportError as exc:
        raise MissingMLDependencies(
            "Install the optional ML stack with `uv sync --extra ml`. "
            "For CUDA wheels, install torch from the PyTorch index first."
        ) from exc
    return {"torch": torch, "FluxInpaintPipeline": getattr(diffusers, "FluxInpaintPipeline")}


def _torch_dtype(torch: Any, dtype_name: str) -> Any:
    dtype = getattr(torch, dtype_name, None)
    if dtype is None:
        raise ValueError(f"Unknown torch dtype: {dtype_name}")
    return dtype


def _emit(progress: Any | None, message: str) -> None:
    if progress is not None:
        progress(message)
