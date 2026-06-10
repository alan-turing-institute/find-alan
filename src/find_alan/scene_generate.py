"""Waldo-style tiled scene generation using Flux.

Two prompt modes:
- LLM mode:  a locally-deployed OpenAI-compatible LLM (e.g. vLLM serving Gemma)
             is queried to decompose a free-form theme into per-tile sub-scene prompts.
- Fallback:  hard-coded conference sub-scene prompts are used when no LLM endpoint
             is configured or reachable.  The Wally styling suffix is identical in
             both modes so visual output is consistent regardless of prompt source.

Typical library use::

    from find_alan.scene_generate import SceneGenerationConfig, generate_scene
    cfg = SceneGenerationConfig(
        theme="a Victorian street market",
        llm_base_url="http://localhost:8000/v1",
        output_dir=Path("outputs/market"),
    )
    result = generate_scene(cfg)
    print(result.final_path)

As a script::

    find-alan-generate --theme "a Victorian street market" \\
        --llm-url http://localhost:8000/v1 \\
        --grid 2x2 --tile-size 2048 --out outputs/market
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Styling constants — identical regardless of prompt source
# ---------------------------------------------------------------------------

STYLE_SUFFIX: str = (
    "Where's Wally illustration style, flat cartoon art, bold black outlines, "
    "limited bright colour palette of red yellow blue and beige, hundreds of tiny "
    "detailed figures, isometric bird's eye view, dense crowd, Martin Handford style, "
    "highly detailed, no text"
)

NEGATIVE_PROMPT: str = (
    "photorealistic, 3d render, blurry, dark, moody, few people, empty spaces, "
    "ugly, watermark, signature"
)

# ---------------------------------------------------------------------------
# Fallback sub-scene prompts (conference theme)
# Used when no LLM endpoint is available.
# ---------------------------------------------------------------------------

FALLBACK_TILE_PROMPTS: tuple[str, ...] = (
    (
        "conference entrance lobby packed with tiny people checking in at registration desks, "
        "banner stands, lanyards, queues of attendees, welcome signage"
    ),
    (
        "main stage auditorium filled with tiny audience members watching a keynote presenter, "
        "spotlights, giant projection screens, rows of seats, live-streaming cameras"
    ),
    (
        "exhibition floor dense with tiny people visiting colourful sponsor booths, "
        "product demos, handshakes, branded pop-ups, swag tables, networking clusters"
    ),
    (
        "networking area and food court crammed with tiny people chatting over coffee, "
        "eating lunch, exchanging business cards, casual breakout clusters, catering stands"
    ),
)

# ---------------------------------------------------------------------------
# LLM prompt-generation helpers
# ---------------------------------------------------------------------------

_LLM_SYSTEM_PROMPT = """\
You are a creative director generating prompts for a Where's Wally style illustration.
Given a scene description and a grid size, generate exactly {n} sub-scene prompts.
Each tile should depict a different area of the overall scene with dense crowds of tiny people.
Respond ONLY with a JSON array of {n} strings, no other text, no markdown fences."""

_LLM_USER_TEMPLATE = """\
Overall theme: {theme}

Generate {n} tile prompts for a {cols}x{rows} grid.
Assign each tile to a distinct sub-area of the theme (e.g. entrance, main stage, expo floor, food court).
Each prompt must describe a dense crowd of tiny illustrated people doing activities specific to that area.
Be specific and vivid. Keep each prompt under 100 words."""


def _query_llm(
    base_url: str,
    model: str,
    theme: str,
    cols: int,
    rows: int,
    timeout: float = 60.0,
) -> list[str]:
    """Call an OpenAI-compatible /v1/chat/completions endpoint and return tile prompts.

    Raises ``RuntimeError`` on any network, HTTP, or parse failure so callers
    can catch it and fall back to hard-coded prompts.
    """
    try:
        import urllib.request
        import urllib.error
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("urllib not available") from exc

    n = cols * rows
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": _LLM_SYSTEM_PROMPT.format(n=n),
            },
            {
                "role": "user",
                "content": _LLM_USER_TEMPLATE.format(
                    theme=theme, n=n, cols=cols, rows=rows
                ),
            },
        ],
        "temperature": 0.7,
        "max_tokens": 1024,
    }
    url = base_url.rstrip("/") + "/chat/completions"
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode())
    except urllib.error.URLError as exc:
        raise RuntimeError(f"LLM request failed: {exc}") from exc

    try:
        content: str = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise RuntimeError(f"Unexpected LLM response shape: {body}") from exc

    # Strip optional markdown fences the model may emit despite instructions
    content = content.strip()
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]

    start = content.find("[")
    end = content.rfind("]") + 1
    if start == -1 or end == 0:
        raise RuntimeError(f"No JSON array found in LLM response: {content!r}")

    prompts: list[str] = json.loads(content[start:end])
    if not isinstance(prompts, list) or len(prompts) != n:
        raise RuntimeError(
            f"Expected {n} prompts, got {len(prompts) if isinstance(prompts, list) else type(prompts)}"
        )
    return [str(p) for p in prompts]


def resolve_tile_prompts(
    theme: str,
    cols: int,
    rows: int,
    llm_base_url: str | None,
    llm_model: str,
    llm_timeout: float,
) -> tuple[list[str], str]:
    """Return ``(tile_prompts, source)`` where *source* is ``"llm"`` or ``"fallback"``."""
    n = cols * rows

    if llm_base_url:
        try:
            prompts = _query_llm(
                llm_base_url, llm_model, theme, cols, rows, llm_timeout
            )
            logger.info("LLM generated %d tile prompts for theme %r", n, theme)
            return prompts, "llm"
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "LLM prompt generation failed (%s); using fallback prompts.", exc
            )

    # Fallback: repeat / trim the conference prompts to fit the requested grid
    base = list(FALLBACK_TILE_PROMPTS)
    prompts = (base * ((n // len(base)) + 1))[:n]
    logger.info("Using fallback tile prompts (n=%d)", n)
    return prompts, "fallback"


# ---------------------------------------------------------------------------
# ML pipeline helpers  (heavy imports deferred so the module is importable
# without torch/diffusers installed)
# ---------------------------------------------------------------------------


class MissingMLDependencies(RuntimeError):
    """Raised when torch / diffusers are not installed."""


def _require_ml() -> Any:
    """Return the ``torch`` module or raise ``MissingMLDependencies``."""
    try:
        import torch  # noqa: PLC0415
        return torch
    except ImportError as exc:
        raise MissingMLDependencies(
            "ML dependencies are not installed.  Install the [ml] extra:\n"
            "  uv pip install 'find-alan[ml]'"
        ) from exc


def _load_flux_pipeline(model_id: str, device_map: str | None, torch_dtype: Any) -> Any:
    from diffusers import FluxPipeline  # noqa: PLC0415

    logger.info("Loading Flux pipeline from %r …", model_id)
    kwargs: dict[str, Any] = {"torch_dtype": torch_dtype}
    if device_map:
        kwargs["device_map"] = device_map
    pipe = FluxPipeline.from_pretrained(model_id, **kwargs)
    return pipe


def _generate_tile(
    pipe: Any,
    prompt: str,
    seed: int,
    tile_size: int,
    steps: int,
    guidance_scale: float,
    generator_device: str,
) -> Any:
    import torch  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    generator = torch.Generator(device=generator_device).manual_seed(seed)
    result = pipe(
        prompt=prompt,
        height=tile_size,
        width=tile_size,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )
    image: Image.Image = result.images[0]
    return image


def _stitch_tiles(
    tiles: list[Any],
    cols: int,
    rows: int,
    tile_size: int,
) -> Any:
    from PIL import Image  # noqa: PLC0415

    final = Image.new("RGB", (tile_size * cols, tile_size * rows))
    for idx, tile in enumerate(tiles):
        row = idx // cols
        col = idx % cols
        final.paste(tile, (col * tile_size, row * tile_size))
    return final


# ---------------------------------------------------------------------------
# Public config / result dataclasses
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT_DIR = Path("outputs/scene_generate")
DEFAULT_FLUX_MODEL_ID = "black-forest-labs/FLUX.2-dev"
DEFAULT_LLM_MODEL = "google/gemma-3-27b-it"


@dataclass(frozen=True)
class SceneGenerationConfig:
    """Settings for a tiled Waldo-style scene generation run.

    Parameters
    ----------
    theme:
        Free-form description of the overall scene (e.g.
        ``"a busy tech conference"``).  Passed to the LLM to produce
        per-tile sub-scene prompts; ignored when using fallback prompts.
    output_dir:
        Directory where tiles and the final stitched image are saved.
    flux_model_id:
        HuggingFace repo ID for the Flux model.
    device_map:
        Passed directly to ``from_pretrained`` when set (e.g. ``"balanced"``
        spreads across multiple GPUs).  ``None`` (default) uses a single GPU
        via ``enable_model_cpu_offload``.
    generator_device:
        Device used to create the ``torch.Generator`` for seeding.
    torch_dtype_str:
        ``"bfloat16"`` or ``"float16"``.
    cols, rows:
        Grid dimensions; total tiles = ``cols * rows``.
    tile_size:
        Height and width of each individual tile in pixels.
    steps:
        Flux denoising steps.
    guidance_scale:
        Flux guidance scale.
    seed:
        Base seed; each tile uses ``seed + tile_index``.
    save_tiles:
        Whether to save individual tile PNGs alongside the final image.
    llm_base_url:
        Base URL of an OpenAI-compatible LLM endpoint
        (e.g. ``"http://localhost:8000/v1"``).  ``None`` skips LLM and
        uses the hard-coded fallback prompts.
    llm_model:
        Model name passed in the LLM request body.
    llm_timeout:
        HTTP timeout in seconds for the LLM request.
    """

    theme: str = "a busy tech conference"
    output_dir: Path = DEFAULT_OUTPUT_DIR
    flux_model_id: str = DEFAULT_FLUX_MODEL_ID
    device_map: str | None = None  # None → enable_model_cpu_offload (single GPU); "balanced" → multi-GPU
    generator_device: str = "cuda:0"
    torch_dtype_str: str = "bfloat16"
    cols: int = 2
    rows: int = 2
    tile_size: int = 2048
    steps: int = 28
    guidance_scale: float = 3.5
    seed: int = 42
    save_tiles: bool = True
    llm_base_url: str | None = None
    llm_model: str = DEFAULT_LLM_MODEL
    llm_timeout: float = 60.0


@dataclass(frozen=True)
class SceneGenerationResult:
    """Paths written by a scene generation run."""

    output_dir: Path
    final_path: Path
    tile_paths: tuple[Path, ...]
    prompt_source: str  # "llm" or "fallback"
    tile_prompts: tuple[str, ...]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def generate_scene(config: SceneGenerationConfig) -> SceneGenerationResult:
    """Run the full tiled generation pipeline and return a result object.

    This is the primary library API.  Import and call it directly from
    other pipeline stages::

        result = generate_scene(cfg)
        next_stage(result.final_path)
    """
    torch = _require_ml()

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    torch_dtype = dtype_map.get(config.torch_dtype_str, torch.bfloat16)

    config.output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Resolve tile prompts ──────────────────────────────────────────────
    raw_prompts, prompt_source = resolve_tile_prompts(
        theme=config.theme,
        cols=config.cols,
        rows=config.rows,
        llm_base_url=config.llm_base_url,
        llm_model=config.llm_model,
        llm_timeout=config.llm_timeout,
    )
    full_prompts = [f"{p}, {STYLE_SUFFIX}" for p in raw_prompts]

    logger.info(
        "Prompt source: %s  |  theme: %r  |  grid: %dx%d",
        prompt_source,
        config.theme,
        config.cols,
        config.rows,
    )
    for i, p in enumerate(full_prompts):
        logger.debug("Tile %d prompt: %s", i, p[:120])

    # ── 2. Load Flux ─────────────────────────────────────────────────────────
    pipe = _load_flux_pipeline(config.flux_model_id, config.device_map, torch_dtype)
    if config.device_map is None:
        pipe.enable_model_cpu_offload()

    # ── 3. Generate tiles ────────────────────────────────────────────────────
    n = config.cols * config.rows
    positions = [
        f"r{r}c{c}"
        for r in range(config.rows)
        for c in range(config.cols)
    ]
    tiles = []
    tile_paths: list[Path] = []

    for i, (prompt, position) in enumerate(zip(full_prompts, positions)):
        tile_seed = config.seed + i
        logger.info("Generating tile %d/%d (%s) seed=%d …", i + 1, n, position, tile_seed)
        tile = _generate_tile(
            pipe=pipe,
            prompt=prompt,
            seed=tile_seed,
            tile_size=config.tile_size,
            steps=config.steps,
            guidance_scale=config.guidance_scale,
            generator_device=config.generator_device,
        )
        tiles.append(tile)
        if config.save_tiles:
            tile_path = config.output_dir / f"tile_{i + 1:02d}_{position}.png"
            tile.save(tile_path)
            tile_paths.append(tile_path)
            logger.info("  Saved tile → %s", tile_path)

    # ── 4. Stitch ────────────────────────────────────────────────────────────
    logger.info(
        "Stitching %dx%d grid → %dx%d px …",
        config.cols,
        config.rows,
        config.tile_size * config.cols,
        config.tile_size * config.rows,
    )
    final_image = _stitch_tiles(tiles, config.cols, config.rows, config.tile_size)
    final_path = config.output_dir / "scene_final.png"
    final_image.save(final_path)
    logger.info("Final image saved → %s", final_path)

    return SceneGenerationResult(
        output_dir=config.output_dir,
        final_path=final_path,
        tile_paths=tuple(tile_paths),
        prompt_source=prompt_source,
        tile_prompts=tuple(raw_prompts),
    )
