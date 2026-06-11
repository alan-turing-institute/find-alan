"""CLI: overlay the Find Alan logo on the top-left corner of an image.

Example:
  find-alan-logo outputs/finished/conference_alan.png
  find-alan-logo scene.png --output scene_branded.png --width-fraction 0.12
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from find_alan.logo import LogoOverlayConfig, run_add_logo

# Single source of truth for defaults: the dataclass.
_DEFAULTS = LogoOverlayConfig()


class _Formatter(
    argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter
):
    """Show each argument's default and keep the raw description layout."""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="find-alan-logo",
        description="Composite the partly-transparent Find Alan logo onto the top-left corner.",
        formatter_class=_Formatter,
    )
    p.add_argument("input", type=Path, help="Image to add the logo to.")
    p.add_argument("--logo", type=Path, default=_DEFAULTS.logo_path, help="Logo PNG (with transparency).")
    p.add_argument(
        "--output",
        type=Path,
        default=_DEFAULTS.output_path,
        help="Output path. Defaults to '<input stem>_logo.png' beside the input.",
    )
    p.add_argument(
        "--width-fraction",
        type=float,
        default=_DEFAULTS.width_fraction,
        help="Logo width as a fraction of the image width (aspect preserved).",
    )
    p.add_argument(
        "--margin",
        type=int,
        default=_DEFAULTS.margin,
        help="Pixels in from the top-left corner.",
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = LogoOverlayConfig(
        input_path=args.input,
        logo_path=args.logo,
        output_path=args.output,
        width_fraction=args.width_fraction,
        margin=args.margin,
    )
    output_path = run_add_logo(config)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
