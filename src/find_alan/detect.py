"""Person detection using YOLOv8.

Detects people in a crowd scene and returns bounding boxes that can be
passed directly to the inpaint pipeline as insertion targets.

The model (yolov8n.pt, ~6 MB) is downloaded automatically on first use.
"""

from __future__ import annotations

import random as _random
from typing import Literal

from PIL import Image

_PERSON_CLASS = 0  # COCO class index for "person"

Strategy = Literal["random", "largest", "smallest", "center"]


def load_detector(model_size: str = "yolov8n"):
    """Return a YOLOv8 detector. Downloads the weights on first call."""
    from ultralytics import YOLO  # import here so ultralytics is optional at import time

    return YOLO(f"{model_size}.pt")


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
