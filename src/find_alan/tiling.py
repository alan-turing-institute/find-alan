"""Tile planning helpers for diffusion upscaling experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from collections.abc import Iterator


@dataclass(frozen=True)
class Crop:
    """A rectangular crop in target-image pixel coordinates."""

    x: int
    y: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height

    def to_dict(self) -> dict[str, int]:
        data = asdict(self)
        data["right"] = self.right
        data["bottom"] = self.bottom
        return data


def _axis_starts(length: int, tile_size: int, stride: int, offset: int) -> list[int]:
    if length <= 0:
        raise ValueError("length must be positive")
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if stride <= 0:
        raise ValueError("overlap must be smaller than tile_size")
    if length <= tile_size:
        return [0]

    max_start = length - tile_size
    starts = {0, max(0, min(offset, max_start)), max_start}
    current = max(0, min(offset, max_start))

    while current < max_start:
        current = min(current + stride, max_start)
        starts.add(current)

    return sorted(starts)


def jittered_grid(
    width: int,
    height: int,
    tile_size: int,
    overlap: int,
    *,
    rng: random.Random,
    jitter: int | None = None,
) -> list[Crop]:
    """Return a full-cover tile grid with a random per-step alignment."""

    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    if not 0 <= overlap < tile_size:
        raise ValueError("overlap must satisfy 0 <= overlap < tile_size")

    stride = tile_size - overlap
    max_jitter = min(stride - 1, jitter if jitter is not None else stride - 1)
    offset_x = rng.randint(0, max_jitter) if max_jitter > 0 else 0
    offset_y = rng.randint(0, max_jitter) if max_jitter > 0 else 0

    x_starts = _axis_starts(width, tile_size, stride, offset_x)
    y_starts = _axis_starts(height, tile_size, stride, offset_y)

    return [
        Crop(
            x=x,
            y=y,
            width=min(tile_size, width - x),
            height=min(tile_size, height - y),
        )
        for y in y_starts
        for x in x_starts
    ]


def jittered_crop_schedule(
    width: int,
    height: int,
    tile_size: int,
    overlap: int,
    steps: int,
    *,
    seed: int | None = None,
    jitter: int | None = None,
) -> Iterator[list[Crop]]:
    """Yield full-cover crop grids with different alignments across steps."""

    if steps <= 0:
        raise ValueError("steps must be positive")

    rng = random.Random(seed)
    for _ in range(steps):
        yield jittered_grid(
            width=width,
            height=height,
            tile_size=tile_size,
            overlap=overlap,
            rng=rng,
            jitter=jitter,
        )
