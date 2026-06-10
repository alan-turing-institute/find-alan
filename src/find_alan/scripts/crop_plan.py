"""Print jittered crop grids for custom MultiDiffusion experiments."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json

from find_alan.tiling import jittered_crop_schedule


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="find-alan-crop-plan",
        description="Print jittered full-cover crop grids for custom MultiDiffusion experiments.",
    )
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--scale", type=float, default=10.0)
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=192)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--jitter", type=int, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    target_width = round(args.width * args.scale)
    target_height = round(args.height * args.scale)
    schedule = [
        [crop.to_dict() for crop in crops]
        for crops in jittered_crop_schedule(
            width=target_width,
            height=target_height,
            tile_size=args.tile_size,
            overlap=args.overlap,
            steps=args.steps,
            seed=args.seed,
            jitter=args.jitter,
        )
    ]
    print(
        json.dumps(
            {
                "target_width": target_width,
                "target_height": target_height,
                "tile_size": args.tile_size,
                "overlap": args.overlap,
                "steps": schedule,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
