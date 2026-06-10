import time
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw

GENERATE_DIR = Path("outputs/generate")
IMPROVE_DIR = Path("outputs/improve")

COLORS = [
    (70, 90, 120),
    (90, 70, 120),
    (70, 120, 90),
    (120, 90, 70),
]


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, max_width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        test = f"{current} {word}".strip()
        if draw.textlength(test) <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _draw_centred_text(draw: ImageDraw.ImageDraw, lines: list[str], size: int, colour: tuple) -> None:
    line_height = 20
    total_height = len(lines) * line_height
    y = (size - total_height) // 2
    for line in lines:
        w = draw.textlength(line)
        draw.text(((size - w) / 2, y), line, fill=colour)
        y += line_height


def _make_image(prompt: str, index: int, timestamp: str) -> str:
    img = Image.new("RGB", (512, 512), color=COLORS[index])
    draw = ImageDraw.Draw(img)
    lines = _wrap_text(draw, f"[{index + 1}/4] {prompt}", 512 - 40)
    _draw_centred_text(draw, lines, 512, (200, 220, 255))
    filename = GENERATE_DIR / f"{timestamp}_{index + 1}.png"
    img.save(filename)
    return str(filename)


def generate_image(prompt: str) -> list[str]:
    time.sleep(3)
    GENERATE_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return [_make_image(prompt, i, timestamp) for i in range(4)]


def improve_image(image_path: str) -> str:
    time.sleep(3)
    IMPROVE_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    src = Image.open(image_path)
    img = src.copy()
    draw = ImageDraw.Draw(img)
    # Gold border to mark as improved
    border = 8
    draw.rectangle([border, border, 511 - border, 511 - border], outline=(255, 200, 0), width=border)
    lines = _wrap_text(draw, f"[improved] {Path(image_path).stem}", 512 - 40)
    _draw_centred_text(draw, lines, 512, (255, 200, 0))
    filename = IMPROVE_DIR / f"{timestamp}_improved.png"
    img.save(filename)
    return str(filename)
