"""Unit tests for the pure-geometry / mask helpers in find_alan.paste.

These exercise crop-box, paste-box and mask construction without touching the
ML stack, so they run on CPU with only Pillow installed.
"""

from __future__ import annotations

from PIL import Image
import pytest

from find_alan.paste import (
    bbox_to_xyxy,
    build_writable_mask,
    compute_crop_box,
    compute_paste_box,
    filter_selectable,
    paste_figure,
    selection_zones,
)


def _alan(width: int = 100, height: int = 200) -> Image.Image:
    """A simple RGBA figure: opaque vertical bar down the centre, rest clear."""
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    bar = Image.new("RGBA", (width // 2, height), (200, 50, 50, 255))
    img.paste(bar, (width // 4, 0))
    return img


# --------------------------------------------------------------------------- #
# compute_paste_box
# --------------------------------------------------------------------------- #


def test_paste_box_fits_bbox_height_and_preserves_aspect():
    # figure 100x200 (aspect 0.5), bbox height 400 -> paste 200x400.
    box = compute_paste_box((1000, 500, 80, 400), (100, 200))
    x0, y0, x1, y1 = box
    assert (x1 - x0, y1 - y0) == (200, 400)


def test_paste_box_is_centered_and_bottom_aligned():
    # bbox centre x = 1000 + 80/2 = 1040; paste width 200 -> x0 = 940.
    # paste height == bbox height, so top aligns with bbox top (y0 == 500).
    box = compute_paste_box((1000, 500, 80, 400), (100, 200))
    assert box == (940, 500, 1140, 900)


def test_paste_box_scale_grows_upward_from_bbox_base():
    # scale 1.10 on a bbox of height 400 -> paste height 440, base unchanged.
    base = compute_paste_box((1000, 500, 80, 400), (100, 200))
    scaled = compute_paste_box((1000, 500, 80, 400), (100, 200), 1.10)
    assert (scaled[3] - scaled[1]) == 440  # 10% taller
    assert scaled[3] == base[3] == 900  # base (bottom) stays on the bbox base
    assert scaled[1] < base[1]  # grows upward
    # aspect preserved: width scales with height.
    assert (scaled[2] - scaled[0]) == round(100 * 440 / 200)


def test_paste_box_rejects_degenerate_input():
    with pytest.raises(ValueError):
        compute_paste_box((0, 0, 10, 0), (100, 200))


# --------------------------------------------------------------------------- #
# compute_crop_box
# --------------------------------------------------------------------------- #


def test_crop_box_contains_figure_and_is_square():
    bbox = (1000, 500, 80, 400)
    paste = compute_paste_box(bbox, (100, 200))  # wider than bbox
    crop, gap = compute_crop_box(bbox, paste, (2048, 2048), 0.4, 0.2)

    # Square.
    assert (crop[2] - crop[0]) == (crop[3] - crop[1])
    # Contains the whole figure (union of bbox and paste box).
    fig = (
        min(paste[0], bbox[0]),
        min(paste[1], bbox[1]),
        max(paste[2], bbox[0] + bbox[2]),
        max(paste[3], bbox[1] + bbox[3]),
    )
    assert crop[0] <= fig[0] and crop[1] <= fig[1]
    assert crop[2] >= fig[2] and crop[3] >= fig[3]
    # Gap box sits inside the crop and outside (or on) the figure.
    assert crop[0] <= gap[0] and crop[2] >= gap[2]


def test_crop_box_stays_within_image_bounds():
    # Person near the top-left corner: crop must clamp to the image.
    bbox = (5, 5, 60, 300)
    paste = compute_paste_box(bbox, (100, 200))
    crop, _ = compute_crop_box(bbox, paste, (2048, 2048), 0.4, 0.2)
    assert crop[0] >= 0 and crop[1] >= 0
    assert crop[2] <= 2048 and crop[3] <= 2048


def test_bbox_to_xyxy():
    assert bbox_to_xyxy((10, 20, 30, 40)) == (10, 20, 40, 60)


# --------------------------------------------------------------------------- #
# build_writable_mask
# --------------------------------------------------------------------------- #


def test_mask_freezes_border_and_protects_upper_figure():
    figure = _alan(100, 200)
    bbox = (1000, 500, 80, 400)
    paste = compute_paste_box(bbox, figure.size)
    crop, gap = compute_crop_box(bbox, paste, (2048, 2048), 0.4, 0.2)

    mask = build_writable_mask(
        crop,
        gap,
        paste,
        figure,
        protect_fraction=0.6,
        alpha_threshold=128,
        dilate=8,
        feather=8,
    )
    assert mask.size == (crop[2] - crop[0], crop[3] - crop[1])

    # The very corner of the crop is in the frozen border ring -> black.
    assert mask.getpixel((1, 1)) == 0

    # A point on the figure in its upper (protected) third should be frozen.
    upper_cx = (paste[0] + paste[2]) // 2 - crop[0]
    upper_cy = paste[1] + (paste[3] - paste[1]) // 5 - crop[1]
    assert mask.getpixel((upper_cx, upper_cy)) < 40

    # Midway through the gap ring beside the figure should be writable (bright).
    gap_x = (gap[0] + paste[0]) // 2 - crop[0]
    gap_y = (gap[1] + gap[3]) // 2 - crop[1]
    assert mask.getpixel((gap_x, gap_y)) > 200


def test_mask_leaves_lower_figure_writable():
    figure = _alan(100, 200)
    bbox = (1000, 500, 80, 400)
    paste = compute_paste_box(bbox, figure.size)
    crop, gap = compute_crop_box(bbox, paste, (2048, 2048), 0.4, 0.2)

    mask = build_writable_mask(
        crop, gap, paste, figure,
        protect_fraction=0.6, alpha_threshold=128, dilate=8, feather=8,
    )
    # A point on the figure low down (well below the 0.6 thigh line) is writable.
    low_cx = (paste[0] + paste[2]) // 2 - crop[0]
    low_cy = paste[1] + int((paste[3] - paste[1]) * 0.9) - crop[1]
    assert mask.getpixel((low_cx, low_cy)) > 200


# --------------------------------------------------------------------------- #
# selection_zones / filter_selectable
# --------------------------------------------------------------------------- #


def test_selection_zones_geometry():
    safe, logo = selection_zones((2048, 2048), 0.05, 0.10)
    assert safe == (102, 102, 1946, 1946)
    assert logo == (0, 0, 205, 205)


def test_filter_keeps_central_person():
    # Figure 100x200; central person well inside the safe area, away from logo.
    bbox = (1000, 1000, 80, 300)
    kept = filter_selectable([bbox], (100, 200), (2048, 2048), 0.05, 0.10)
    assert kept == [bbox]


def test_filter_rejects_person_against_bottom_edge():
    # The originally-chosen person: bottom at 2046, inside the 5% margin.
    bbox = (1215, 1862, 95, 184)
    kept = filter_selectable([bbox], (1754, 2480), (2048, 2048), 0.05, 0.10)
    assert kept == []


def test_filter_rejects_person_in_logo_block():
    # A person whose figure box sits in the top-left logo zone.
    bbox = (40, 40, 60, 120)
    kept = filter_selectable([bbox], (100, 200), (2048, 2048), 0.05, 0.10)
    assert kept == []


def test_filter_accounts_for_alan_horizontal_overhang():
    # bbox clears the right margin, but a wide figure's centred paste box would
    # overhang past it into the margin -> rejected.
    safe, _ = selection_zones((2048, 2048), 0.05, 0.10)
    # Place the bbox so its right edge is just inside safe, then use a very wide
    # figure so the paste box (centred, wider) pushes past the safe edge.
    bbox = (safe[2] - 100, 1000, 90, 300)  # right edge at safe[2]-10
    kept = filter_selectable([bbox], (400, 200), (2048, 2048), 0.05, 0.10)
    assert kept == []


# --------------------------------------------------------------------------- #
# paste_figure
# --------------------------------------------------------------------------- #


def test_paste_figure_places_opaque_pixels_over_crop():
    figure = _alan(100, 200)
    bbox = (1000, 500, 80, 400)
    paste = compute_paste_box(bbox, figure.size)
    crop, _ = compute_crop_box(bbox, paste, (2048, 2048), 0.4, 0.2)

    crop_img = Image.new("RGB", (crop[2] - crop[0], crop[3] - crop[1]), (10, 120, 10))
    out = paste_figure(crop_img, figure, paste, crop)
    assert out.size == crop_img.size

    # Centre of the figure should now show the figure's red, not the green bg.
    cx = (paste[0] + paste[2]) // 2 - crop[0]
    cy = (paste[1] + paste[3]) // 2 - crop[1]
    r, g, b = out.getpixel((cx, cy))
    assert r > g and r > b
