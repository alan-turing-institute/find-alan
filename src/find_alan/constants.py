from __future__ import annotations

import ast
from pathlib import Path


_PROMPTS_DIR = Path(__file__).resolve().parents[2] / "assets" / "prompts"


def read_prompt_file(filename: str) -> str:
    source = (_PROMPTS_DIR / filename).read_text(encoding="utf-8").strip()

    try:
        parsed = ast.literal_eval(f"({source})")
    except (SyntaxError, ValueError):
        parsed = source

    if not isinstance(parsed, str):
        raise TypeError(f"{filename} must parse to a string")

    return " ".join(parsed.split())


POSTIVE_PROMPT_01 = read_prompt_file("styles/1-waldo-cartoon.txt")
NEGATIVE_PROMPT_01 = read_prompt_file("NEGATIVE.txt")
