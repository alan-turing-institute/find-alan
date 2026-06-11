"""Overlay the Find Alan logo onto the top-left corner of an image.

The logo is alpha-composited (it is partly transparent) at the very top-left,
scaled to a fraction of the base image width with its aspect ratio preserved.

Pure PIL — no ML dependencies — so the geometry helpers are importable and
unit-testable on their own.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

DEFAULT_LOGO_PATH = Path("assets/images/FindAlanLogo.png")
DEFAULT_WIDTH_FRACTION = 0.20
DEFAULT_MARGIN = 0


@dataclass(frozen=True)
class LogoOverlayConfig:
    """Settings for placing the logo on an image."""

    input_path: Path | None = None
    logo_path: Path = DEFAULT_LOGO_PATH
    output_path: Path | None = None  # defaults to "<input stem>_logo.png" beside input
    width_fraction: float = DEFAULT_WIDTH_FRACTION
    margin: int = DEFAULT_MARGIN  # pixels in from the top-left corner


def logo_size(
    logo_size: tuple[int, int], base_width: int, width_fraction: float
) -> tuple[int, int]:
    """Return the (width, height) for the logo at *width_fraction* of *base_width*.

    The logo's aspect ratio is preserved.
    """
    lw, lh = logo_size
    if lw <= 0 or lh <= 0:
        raise ValueError("logo dimensions must be positive")
    if base_width <= 0:
        raise ValueError("base_width must be positive")
    if width_fraction <= 0:
        raise ValueError("width_fraction must be positive")

    target_w = max(1, round(base_width * width_fraction))
    target_h = max(1, round(lh * target_w / lw))
    return target_w, target_h


def overlay_logo(
    base: Image.Image,
    logo: Image.Image,
    width_fraction: float = DEFAULT_WIDTH_FRACTION,
    margin: int = DEFAULT_MARGIN,
) -> Image.Image:
    """Composite *logo* onto the top-left of *base*, returning a new RGB image.

    *logo* is scaled to *width_fraction* of the base width (aspect preserved)
    and alpha-composited at (*margin*, *margin*) so its transparency is honoured.
    """
    target_w, target_h = logo_size(logo.size, base.width, width_fraction)
    scaled = logo.convert("RGBA").resize((target_w, target_h), Image.LANCZOS)

    out = base.convert("RGBA")
    out.alpha_composite(scaled, (margin, margin))
    return out.convert("RGB")


def run_add_logo(config: LogoOverlayConfig) -> Path:
    """Load, overlay, and save. Returns the output path written."""
    if config.input_path is None:
        raise ValueError("input_path is required")

    base = Image.open(config.input_path)
    logo = Image.open(config.logo_path)
    result = overlay_logo(base, logo, config.width_fraction, config.margin)

    output_path = config.output_path
    if output_path is None:
        output_path = config.input_path.with_name(f"{config.input_path.stem}_logo.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path)
    return output_path
