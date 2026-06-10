from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw

OUTPUTS_DIR = Path("outputs")

COLORS = [
    (70, 90, 120),
    (90, 70, 120),
    (70, 120, 90),
    (120, 90, 70),
]


def _make_image(prompt: str, index: int, timestamp: str) -> str:
    img = Image.new("RGB", (512, 512), color=COLORS[index])
    draw = ImageDraw.Draw(img)

    label = f"[{index + 1}/4] {prompt}"
    margin = 20
    max_width = 512 - 2 * margin
    words = label.split()
    lines = []
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

    line_height = 20
    total_height = len(lines) * line_height
    y = (512 - total_height) // 2
    for line in lines:
        w = draw.textlength(line)
        draw.text(((512 - w) / 2, y), line, fill=(200, 220, 255))
        y += line_height

    filename = OUTPUTS_DIR / f"{timestamp}_{index + 1}.png"
    img.save(filename)
    return str(filename)


def generate_image(prompt: str) -> list[str]:
    OUTPUTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return [_make_image(prompt, i, timestamp) for i in range(4)]
