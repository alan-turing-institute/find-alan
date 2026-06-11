"""Person detection using YOLOv8.

Detects people in a crowd scene and returns bounding boxes that can be
passed directly to the inpaint pipeline as insertion targets.

Pass model_size ending in "-seg" (e.g. "yolov8n-seg") to
load_detector() to enable segmentation mask output from
detect_people_with_masks().

The models (~6 MB each) are downloaded automatically on first use.
"""

from __future__ import annotations

import random as _random
from typing import Literal

from PIL import Image, ImageDraw, ImageFilter

_PERSON_CLASS = 0  # COCO class index for "person"

Strategy = Literal["random", "largest", "smallest", "center"]


def load_detector(
    model_size: str = "yolov8n",
    classes: list[str] | None = None,
):
    """Return a YOLOv8 / YOLO-World detector. Downloads weights on first call.

    Pass *classes* for open-vocabulary YOLO-World models, e.g. ``["cartoon person"]``.
    Leave as None for standard YOLOv8 to use the built-in COCO person class.
    """
    from ultralytics import YOLO  # import here so ultralytics is optional at import time

    model = YOLO(f"{model_size}.pt")
    if classes is not None:
        if not hasattr(model, "set_classes"):
            raise ValueError(
                f"Model '{model_size}' does not support open-vocabulary classes. "
                "Use a YOLO-World variant (e.g. yolov8s-worldv2) for --detection-classes."
            )
        model.set_classes(classes)
    return model


def detect_people(
    detector,
    scene: Image.Image,
    conf_threshold: float = 0.3,
) -> list[tuple[int, int, int, int]]:
    """Return (x, y, w, h) bboxes for every person detected in *scene*.

    Boxes are in pixel coordinates relative to *scene*.
    """
    results = detector(scene, classes=[_PERSON_CLASS], verbose=False)
    bboxes: list[tuple[int, int, int, int]] = []
    for r in results:
        for box in r.boxes:
            if float(box.conf[0]) >= conf_threshold:
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                bboxes.append((x1, y1, x2 - x1, y2 - y1))
    return bboxes


def detect_people_with_masks(
    detector,
    scene: Image.Image,
    conf_threshold: float = 0.3,
) -> list[tuple[tuple[int, int, int, int], Image.Image]]:
    """Detect people and return (bbox, mask) pairs.

    Requires a segmentation model (e.g. yolov8n-seg).  Each mask is a
    greyscale image the same size as *scene* — white where the person is,
    black elsewhere.  Falls back to a filled rectangle if the model returns
    no polygon for a detection.
    """
    results = detector(scene, classes=[_PERSON_CLASS], verbose=False)
    detections: list[tuple[tuple[int, int, int, int], Image.Image]] = []
    w, h = scene.size

    for r in results:
        masks_data = getattr(r, "masks", None)
        for i, box in enumerate(r.boxes):
            if float(box.conf[0]) < conf_threshold:
                continue
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            bbox = (x1, y1, x2 - x1, y2 - y1)

            if masks_data is not None and i < len(masks_data):
                # xy gives a list of (N,2) polygon arrays in image coordinates
                poly_coords = masks_data.xy[i]
                mask = Image.new("L", (w, h), 0)
                if len(poly_coords) >= 3:
                    pts = [(float(p[0]), float(p[1])) for p in poly_coords]
                    ImageDraw.Draw(mask).polygon(pts, fill=255)
                else:
                    ImageDraw.Draw(mask).rectangle(
                        [x1, y1, x2, y2], fill=255
                    )
            else:
                mask = Image.new("L", (w, h), 0)
                ImageDraw.Draw(mask).rectangle([x1, y1, x2, y2], fill=255)

            detections.append((bbox, mask))

    return detections


def dilate_mask(
    mask: Image.Image, radius: int = 10
) -> Image.Image:
    """Expand a binary mask outward by *radius* pixels.

    Uses PIL MaxFilter (morphological dilation) then softens the edge
    with a Gaussian feather so the composite boundary isn't hard.
    """
    # MaxFilter size must be odd; PIL clamps large sizes internally.
    size = 2 * radius + 1
    dilated = mask.filter(ImageFilter.MaxFilter(size=size))
    return dilated.filter(ImageFilter.GaussianBlur(radius=radius // 2 or 1))


def pick_target(
    bboxes: list[tuple[int, int, int, int]],
    strategy: Strategy = "random",
    scene_size: tuple[int, int] | None = None,
    rng: _random.Random | None = None,
) -> tuple[int, int, int, int]:
    """Choose one bounding box from *bboxes* according to *strategy*.

    Strategies:
        random   – any detected person (default)
        largest  – highest area; gives the model most inpainting context
        smallest – lowest area; least disruptive to the scene composition
        center   – person closest to the image centre

    *scene_size* is (width, height) and is required for the "center" strategy.
    *rng* allows reproducible random selection.
    """
    if not bboxes:
        raise ValueError("No people detected in the scene.")

    if strategy == "random":
        r = rng or _random.Random()
        return r.choice(bboxes)

    if strategy == "largest":
        return max(bboxes, key=lambda b: b[2] * b[3])

    if strategy == "smallest":
        return min(bboxes, key=lambda b: b[2] * b[3])

    if strategy == "center":
        if scene_size is None:
            raise ValueError("scene_size is required for the 'center' strategy.")
        cx, cy = scene_size[0] / 2, scene_size[1] / 2
        return min(
            bboxes,
            key=lambda b: (b[0] + b[2] / 2 - cx) ** 2 + (b[1] + b[3] / 2 - cy) ** 2,
        )

    raise ValueError(f"Unknown strategy: {strategy!r}")


def pad_bbox(
    bbox: tuple[int, int, int, int],
    image_size: tuple[int, int],
    padding: float = 0.15,
) -> tuple[int, int, int, int]:
    """Expand *bbox* by *padding* fraction on each side, clamped to image bounds.

    Padding gives the inpaint model context pixels at the mask boundary so
    it can blend edges naturally.
    """
    x, y, w, h = bbox
    img_w, img_h = image_size
    dx = int(w * padding)
    dy = int(h * padding)
    x0 = max(0, x - dx)
    y0 = max(0, y - dy)
    x1 = min(img_w, x + w + dx)
    y1 = min(img_h, y + h + dy)
    return x0, y0, x1 - x0, y1 - y0
