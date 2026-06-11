"""Run plakat ML upscalers tile-by-tile for large inputs."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import subprocess
import tempfile

from PIL import Image


_METHOD_SCALES = {
    "real-esrgan-x2": 2,
    "real-esrgan-x4": 4,
}


def _starts(length: int, tile_size: int, overlap: int) -> list[int]:
    if length <= tile_size:
        return [0]

    stride = tile_size - overlap
    starts = list(range(0, max(1, length - tile_size + 1), stride))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return sorted(set(starts))


def _paste_bounds(
    starts: list[int], index: int, length: int, tile_size: int
) -> tuple[int, int]:
    start = starts[index]
    if index == 0:
        left = 0
    else:
        previous = starts[index - 1]
        left = (previous + tile_size + start) // 2

    if index == len(starts) - 1:
        right = length
    else:
        next_start = starts[index + 1]
        right = (start + tile_size + next_start) // 2

    return left, right


def _run_plakat_tile(
    input_path: Path,
    output_path: Path,
    method: str,
    device: str,
) -> None:
    subprocess.run(
        [
            "plakat",
            "upscale",
            "--in",
            str(input_path),
            "--out",
            str(output_path),
            "--method",
            method,
            "--device",
            device,
        ],
        check=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="find-alan-plakat",
        description="Run plakat Real-ESRGAN methods on overlapping tiles and stitch the result.",
    )
    parser.add_argument("--in", dest="input_path", type=Path, required=True)
    parser.add_argument("--out", dest="output_path", type=Path, required=True)
    parser.add_argument(
        "--method",
        choices=tuple(_METHOD_SCALES),
        default="real-esrgan-x4",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=64)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.tile_size <= 0:
        raise ValueError("--tile-size must be positive")
    if args.overlap < 0 or args.overlap >= args.tile_size:
        raise ValueError("--overlap must be non-negative and smaller than --tile-size")

    scale = _METHOD_SCALES[args.method]
    image = Image.open(args.input_path).convert("RGB")
    width, height = image.size
    tile_size = min(args.tile_size, width, height)
    overlap = min(args.overlap, tile_size - 1)
    xs = _starts(width, tile_size, overlap)
    ys = _starts(height, tile_size, overlap)

    output = Image.new("RGB", (width * scale, height * scale))
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="plakat_realesrgan_tiles_") as temp_dir:
        temp_path = Path(temp_dir)
        total = len(xs) * len(ys)
        completed = 0
        for y_index, y in enumerate(ys):
            for x_index, x in enumerate(xs):
                completed += 1
                tile_input = temp_path / f"tile_y{y_index}_x{x_index}.png"
                tile_output = temp_path / f"tile_y{y_index}_x{x_index}_upscaled.png"
                image.crop((x, y, x + tile_size, y + tile_size)).save(tile_input)

                print(
                    f"[{completed:02d}/{total}] {args.method} tile x={x} y={y}",
                    flush=True,
                )
                _run_plakat_tile(tile_input, tile_output, args.method, args.device)

                upscaled = Image.open(tile_output).convert("RGB")
                paste_x0, paste_x1 = _paste_bounds(xs, x_index, width, tile_size)
                paste_y0, paste_y1 = _paste_bounds(ys, y_index, height, tile_size)
                crop = upscaled.crop(
                    (
                        (paste_x0 - x) * scale,
                        (paste_y0 - y) * scale,
                        (paste_x1 - x) * scale,
                        (paste_y1 - y) * scale,
                    )
                )
                output.paste(crop, (paste_x0 * scale, paste_y0 * scale))

    output.save(args.output_path)
    print(f"wrote {args.output_path} ({output.size[0]}x{output.size[1]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
