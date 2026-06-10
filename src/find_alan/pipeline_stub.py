from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUTPUTS_DIR = Path("outputs")


def generate_image(prompt: str) -> str:
    OUTPUTS_DIR.mkdir(exist_ok=True)

    img = Image.new("RGB", (512, 512), color=(70, 90, 120))
    draw = ImageDraw.Draw(img)

    margin = 20
    max_width = 512 - 2 * margin
    words = prompt.split()
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

    filename = OUTPUTS_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    img.save(filename)
    return str(filename)
