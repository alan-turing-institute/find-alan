"""Mask creation utilities for inpainting."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw


def bbox_to_mask(image_size: tuple[int, int], bbox: tuple[int, int, int, int]) -> Image.Image:
    """Return a black image with a white rectangle at *bbox* (x, y, w, h)."""
    mask = Image.new("L", image_size, 0)
    x, y, w, h = bbox
    ImageDraw.Draw(mask).rectangle([x, y, x + w, y + h], fill=255)
    return mask


def load_mask(
    mask_path: str | Path | None,
    bbox: tuple[int, int, int, int] | None,
    image_size: tuple[int, int],
) -> Image.Image:
    """Load a mask from a file path or generate one from a bounding box."""
    if mask_path is not None:
        return Image.open(mask_path).convert("L")
    if bbox is not None:
        return bbox_to_mask(image_size, bbox)
    raise ValueError("Provide either mask_path or bbox.")
