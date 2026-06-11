"""Waldo-style scene generation using Flux.

Prompt resolution and image generation are intentionally decoupled:

1. **Prompt resolution** — load prompts from the ``assets/prompts/`` text files
   (scene, style, negative), or generate subscene prompts via an LLM.
   This step has no dependency on torch / diffusers.

2. **Image generation** — :func:`generate_image` takes a
   :class:`SingleImageConfig` and produces one PNG.  For tiled/stitched output
   use :func:`generate_scene` / :func:`generate_scene_stream` with a
   :class:`SceneGenerationConfig`.

Typical single-image use::

    from find_alan.scene_generate import (
        SingleImageConfig, generate_image,
        load_scene_prompt, load_style, load_negative,
    )

    cfg = SingleImageConfig(
        scene_prompt=load_scene_prompt(Path("assets/prompts/scenes/beach.txt")),
        style_suffix=load_style(Path("assets/prompts/styles/1-waldo-cartoon.txt")),
        negative_prompt=load_negative(Path("assets/prompts/NEGATIVE.txt")),
        output_path=Path("outputs/beach.png"),
    )
    generate_image(cfg)

As a script::

    find-alan-generate \\
        --scene-file  assets/prompts/scenes/beach.txt \\
        --style-file  assets/prompts/styles/1-waldo-cartoon.txt \\
        --negative-file assets/prompts/NEGATIVE.txt \\
        --out outputs/beach.png
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Built-in defaults  (used when no file is supplied)
# ---------------------------------------------------------------------------

DEFAULT_STYLE_SUFFIX: str = (
    "Where's Wally illustration style, flat cartoon art, bold black outlines, "
    "limited bright colour palette of red yellow blue and beige, hundreds of tiny "
    "detailed figures, isometric bird's eye view, dense crowd, Martin Handford style, "
    "highly detailed, no text"
)

DEFAULT_NEGATIVE_PROMPT: str = (
    "photorealistic, 3d render, blurry, dark, moody, few people, empty spaces, "
    "ugly, watermark, signature"
)

# Fallback subscene prompts used when neither a file nor an LLM is available.
FALLBACK_SUBSCENE_PROMPTS: tuple[str, ...] = (
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
# File loaders
# ---------------------------------------------------------------------------


def _load_quoted_prompt(path: Path) -> str:
    """Load a prompt from a file that uses Python quoted-string-literal format.

    Each non-blank, non-comment line should be a double-quoted string, e.g.::

        "beach scene with sunbathers, lifeguards, ice cream sellers, "
        "volleyball players, sandcastles"

    Lines are stripped of their outer quotes then concatenated.  If the
    previous part already ends with a space (``", "`` pattern), the next part
    is appended directly; otherwise ``, `` is inserted.  Plain-text lines
    (no surrounding quotes) are also accepted and joined with ``, ``.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    non_blank = [l for l in lines if l.strip() and not l.strip().startswith("#")]
    if not non_blank:
        raise ValueError(f"No prompt content found in {path}")

    parts: list[str] = []
    for line in non_blank:
        s = line.strip()
        if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
            s = s[1:-1]
        parts.append(s)

    result = ""
    for i, part in enumerate(parts):
        if i == 0:
            result = part
        elif result.endswith(" "):
            result += part
        else:
            result += ", " + part
    return result


def load_scene_prompt(path: Path) -> str:
    """Load a scene description from an ``assets/prompts/scenes/`` file.

    Returns a single string ready to be combined with a style suffix.
    """
    return _load_quoted_prompt(path)


def load_subscenes(path: Path) -> list[str]:
    """Read a plain-text file and return one subscene prompt per non-blank line.

    Lines starting with ``#`` are treated as comments and ignored.

    Example file::

        conference entrance lobby packed with tiny people checking in ...
        main stage auditorium filled with tiny audience members ...
        # this line is ignored
        exhibition floor dense with tiny people visiting booths ...
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    prompts = [l.strip() for l in lines if l.strip() and not l.strip().startswith("#")]
    if not prompts:
        raise ValueError(f"No subscene prompts found in {path}")
    return prompts


def load_style(path: Path) -> str:
    """Load a style suffix from an ``assets/prompts/styles/`` file.

    Accepts both the quoted-string format used in the bundled assets and plain
    multi-line text (lines joined with ``", "``).
    """
    return _load_quoted_prompt(path)


def load_negative(path: Path) -> str:
    """Load a negative prompt from an ``assets/prompts/NEGATIVE.txt`` file.

    Accepts both the quoted-string format used in the bundled assets and plain
    multi-line text (lines joined with ``", "``).
    """
    return _load_quoted_prompt(path)


# ---------------------------------------------------------------------------
# PromptSet — the resolved prompt bundle handed to image generation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PromptSet:
    """Fully-resolved prompts for a scene generation run.

    Parameters
    ----------
    subscenes:
        One prompt string per tile, describing the crowd/activity in that area.
        The style suffix is appended at generation time and should *not* be
        included here.
    style_suffix:
        Appended to every subscene prompt before passing to Flux.
        Defaults to :data:`DEFAULT_STYLE_SUFFIX`.
    negative_prompt:
        Passed to Flux as the negative conditioning string.
        Defaults to :data:`DEFAULT_NEGATIVE_PROMPT`.
    source:
        Human-readable label describing where the subscenes came from
        (e.g. ``"file"``, ``"llm"``, ``"fallback"``).  Stored in the result
        for provenance but not used during generation.
    """

    subscenes: list[str]
    style_suffix: str = DEFAULT_STYLE_SUFFIX
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT
    source: str = "unknown"

    def full_prompts(self) -> list[str]:
        """Return ``subscenes`` with the style suffix appended to each."""
        return [f"{p}, {self.style_suffix}" for p in self.subscenes]


# ---------------------------------------------------------------------------
# LLM-based subscene resolution (unchanged logic, now a standalone helper)
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
    import urllib.error
    import urllib.request

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
            f"Expected {n} prompts, got "
            f"{len(prompts) if isinstance(prompts, list) else type(prompts)}"
        )
    return [str(p) for p in prompts]


DEFAULT_LLM_MODEL = "google/gemma-3-27b-it"


def resolve_tile_prompts(
    theme: str,
    cols: int,
    rows: int,
    llm_base_url: str | None = None,
    llm_model: str = DEFAULT_LLM_MODEL,
    llm_timeout: float = 60.0,
) -> list[str]:
    """Return a list of subscene prompt strings for a ``cols x rows`` grid.

    Tries the LLM first (if *llm_base_url* is set), then falls back to the
    built-in conference prompts.  The returned strings are raw subscene
    descriptions — the style suffix is **not** included.

    To wrap the result in a :class:`PromptSet`::

        prompts = resolve_tile_prompts("a street market", cols=2, rows=2,
                                       llm_base_url="http://localhost:8000/v1")
        prompt_set = PromptSet(subscenes=prompts, source="llm")
    """
    n = cols * rows

    if llm_base_url:
        try:
            prompts = _query_llm(llm_base_url, llm_model, theme, cols, rows, llm_timeout)
            logger.info("LLM generated %d tile prompts for theme %r", n, theme)
            return prompts
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "LLM prompt generation failed (%s); using fallback prompts.", exc
            )

    base = list(FALLBACK_SUBSCENE_PROMPTS)
    prompts = (base * ((n // len(base)) + 1))[:n]
    logger.info("Using fallback subscene prompts (n=%d)", n)
    return prompts


# ---------------------------------------------------------------------------
# ML pipeline helpers
# ---------------------------------------------------------------------------


class MissingMLDependencies(RuntimeError):
    """Raised when torch / diffusers are not installed."""


def _require_ml() -> Any:
    try:
        import torch  # noqa: PLC0415
        return torch
    except ImportError as exc:
        raise MissingMLDependencies(
            "ML dependencies are not installed.  Install the [ml] extra:\n"
            "  uv pip install 'find-alan[ml]'"
        ) from exc


def _load_flux_pipeline(
    model_id: str,
    device_map: str | None,
    torch_dtype: Any,
    lora_weights: str | None = None,
    lora_weight_name: str | None = None,
) -> Any:
    from diffusers import AutoPipelineForText2Image  # noqa: PLC0415

    logger.info("Loading pipeline from %r …", model_id)
    kwargs: dict[str, Any] = {"torch_dtype": torch_dtype}
    if device_map:
        kwargs["device_map"] = device_map
    pipe = AutoPipelineForText2Image.from_pretrained(model_id, **kwargs)
    if lora_weights:
        logger.info("Loading LoRA weights from %r …", lora_weights)
        lora_kwargs: dict[str, Any] = {}
        if lora_weight_name:
            lora_kwargs["weight_name"] = lora_weight_name
        pipe.load_lora_weights(lora_weights, **lora_kwargs)
    return pipe


def _generate_tile(
    pipe: Any,
    prompt: str,
    negative_prompt: str,
    seed: int,
    tile_size: int,
    steps: int,
    guidance_scale: float,
    generator_device: str,
    custom_sigmas: tuple[float, ...] | None = None,
) -> Any:
    import torch  # noqa: PLC0415

    generator = torch.Generator(device=generator_device).manual_seed(seed)
    extra: dict[str, Any] = {}
    if custom_sigmas is not None:
        extra["sigmas"] = list(custom_sigmas)
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=tile_size,
        width=tile_size,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
        **extra,
    )
    return result.images[0]


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


@dataclass(frozen=True)
class SceneGenerationConfig:
    """Settings for a tiled Waldo-style scene generation run.

    Prompt content lives entirely in *prompt_set* — this config only holds
    generation-time knobs (model, grid, hardware, seeds, etc.).

    Parameters
    ----------
    prompt_set:
        Fully-resolved prompts for this run.  Build one from files::

            PromptSet(
                subscenes=load_subscenes(Path("prompts/subscenes.txt")),
                style_suffix=load_style(Path("prompts/style.txt")),
                negative_prompt=load_negative(Path("prompts/negative.txt")),
            )

        Or from the LLM helper::

            PromptSet(subscenes=resolve_tile_prompts("a street market", 2, 2,
                                                      llm_base_url="http://..."))
    output_dir:
        Directory where tiles and the final stitched image are saved.
    flux_model_id:
        HuggingFace repo ID for the Flux model.
    device_map:
        Passed to ``from_pretrained`` when set (e.g. ``"balanced"`` spreads
        across multiple GPUs).  ``None`` uses ``enable_model_cpu_offload``.
    generator_device:
        Device used to create the ``torch.Generator`` for seeding.
    torch_dtype_str:
        ``"bfloat16"`` or ``"float16"``.
    cols, rows:
        Grid dimensions; must match ``len(prompt_set.subscenes)``.
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
    lora_weights:
        HF repo ID for a LoRA adapter (e.g. ``"fal/FLUX.2-dev-Turbo"``).
    lora_weight_name:
        Specific safetensors file within the LoRA repo.
    custom_sigmas:
        Override the inference sigma schedule.
    """

    prompt_set: PromptSet
    output_dir: Path = DEFAULT_OUTPUT_DIR
    flux_model_id: str = DEFAULT_FLUX_MODEL_ID
    device_map: str | None = None
    generator_device: str = "cuda:0"
    torch_dtype_str: str = "bfloat16"
    cols: int = 2
    rows: int = 2
    tile_size: int = 2048
    steps: int = 28
    guidance_scale: float = 3.5
    seed: int = 42
    save_tiles: bool = True
    lora_weights: str | None = None
    lora_weight_name: str | None = None
    custom_sigmas: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        expected = self.cols * self.rows
        actual = len(self.prompt_set.subscenes)
        if actual != expected:
            raise ValueError(
                f"prompt_set has {actual} subscene(s) but grid is "
                f"{self.cols}x{self.rows} ({expected} tiles)"
            )


@dataclass(frozen=True)
class SceneGenerationResult:
    """Paths written by a scene generation run."""

    output_dir: Path
    final_path: Path
    tile_paths: tuple[Path, ...]
    prompt_set: PromptSet  # full provenance — subscenes, style, negative, source


DEFAULT_OUTPUT_PATH = Path("outputs/scene.png")


@dataclass(frozen=True)
class SingleImageConfig:
    """Settings for generating a single Waldo-style scene image.

    Parameters
    ----------
    scene_prompt:
        Scene description loaded from a ``scenes/`` file via
        :func:`load_scene_prompt`.
    style_suffix:
        Appended to ``scene_prompt`` before passing to Flux.
        Defaults to :data:`DEFAULT_STYLE_SUFFIX`.
    negative_prompt:
        Passed to Flux as negative conditioning.
        Defaults to :data:`DEFAULT_NEGATIVE_PROMPT`.
    output_path:
        Where the generated PNG is saved.
    width, height:
        Image dimensions in pixels.
    """

    scene_prompt: str
    style_suffix: str = DEFAULT_STYLE_SUFFIX
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT
    output_path: Path = DEFAULT_OUTPUT_PATH
    flux_model_id: str = DEFAULT_FLUX_MODEL_ID
    device_map: str | None = None
    generator_device: str = "cuda:0"
    torch_dtype_str: str = "bfloat16"
    width: int = 2048
    # width: int = 4096 # try for double width
    height: int = 2048
    steps: int = 28
    guidance_scale: float = 3.5
    seed: int = 42
    lora_weights: str | None = None
    lora_weight_name: str | None = None
    custom_sigmas: tuple[float, ...] | None = None

    @property
    def full_prompt(self) -> str:
        """Scene prompt with style suffix appended."""
        return f"{self.scene_prompt}, {self.style_suffix}"


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------


def generate_image(config: SingleImageConfig) -> Path:
    """Generate a single scene image and save it to ``config.output_path``.

    Returns the path of the saved PNG.
    """
    torch = _require_ml()
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    torch_dtype = dtype_map.get(config.torch_dtype_str, torch.bfloat16)

    config.output_path.parent.mkdir(parents=True, exist_ok=True)

    pipe = _load_flux_pipeline(
        config.flux_model_id,
        config.device_map,
        torch_dtype,
        lora_weights=config.lora_weights,
        lora_weight_name=config.lora_weight_name,
    )
    if config.device_map is None:
        pipe.enable_model_cpu_offload()

    logger.info("Prompt: %s", config.full_prompt[:120])

    generator = torch.Generator(device=config.generator_device).manual_seed(config.seed)
    extra: dict[str, Any] = {}
    if config.custom_sigmas is not None:
        extra["sigmas"] = list(config.custom_sigmas)

    result = pipe(
        prompt=config.full_prompt,
        # negative_prompt=config.negative_prompt,  # Flux2Pipeline does not support negative_prompt
        height=config.height,
        width=config.width,
        num_inference_steps=config.steps,
        guidance_scale=config.guidance_scale,
        generator=generator,
        **extra,
    )
    image = result.images[0]
    image.save(config.output_path)
    logger.info("Image saved → %s", config.output_path)
    return config.output_path


def generate_scene_stream(
    config: SceneGenerationConfig,
) -> Iterator[tuple[int, Path | None] | tuple[None, SceneGenerationResult]]:
    """Generate a tiled scene, yielding progress after each tile.

    Yields ``(tile_index, tile_path)`` after each tile (``tile_path`` is
    ``None`` when ``config.save_tiles`` is ``False``), then
    ``(None, SceneGenerationResult)`` as the final item.

    Typical use::

        for idx, value in generate_scene_stream(cfg):
            if idx is None:
                result = value   # SceneGenerationResult
            else:
                print(f"tile {idx} → {value}")
    """
    torch = _require_ml()

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    torch_dtype = dtype_map.get(config.torch_dtype_str, torch.bfloat16)

    config.output_dir.mkdir(parents=True, exist_ok=True)

    full_prompts = config.prompt_set.full_prompts()
    negative_prompt = config.prompt_set.negative_prompt
    n = config.cols * config.rows

    logger.info(
        "Prompt source: %s  |  grid: %dx%d",
        config.prompt_set.source,
        config.cols,
        config.rows,
    )
    for i, p in enumerate(full_prompts):
        logger.debug("Tile %d prompt: %s", i, p[:120])

    # ── 1. Load Flux ─────────────────────────────────────────────────────────
    pipe = _load_flux_pipeline(
        config.flux_model_id,
        config.device_map,
        torch_dtype,
        lora_weights=config.lora_weights,
        lora_weight_name=config.lora_weight_name,
    )
    if config.device_map is None:
        pipe.enable_model_cpu_offload()

    # ── 2. Generate tiles ────────────────────────────────────────────────────
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
            negative_prompt=negative_prompt,
            seed=tile_seed,
            tile_size=config.tile_size,
            steps=config.steps,
            guidance_scale=config.guidance_scale,
            generator_device=config.generator_device,
            custom_sigmas=config.custom_sigmas,
        )
        tiles.append(tile)
        tile_path: Path | None = None
        if config.save_tiles:
            tile_path = config.output_dir / f"tile_{i + 1:02d}_{position}.png"
            tile.save(tile_path)
            tile_paths.append(tile_path)
            logger.info("  Saved tile → %s", tile_path)
        yield i, tile_path

    # ── 3. Stitch ────────────────────────────────────────────────────────────
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

    yield None, SceneGenerationResult(
        output_dir=config.output_dir,
        final_path=final_path,
        tile_paths=tuple(tile_paths),
        prompt_set=config.prompt_set,
    )


def generate_scene(config: SceneGenerationConfig) -> SceneGenerationResult:
    """Run the full tiled generation pipeline and return a result object.

    This is the primary library API for non-streaming use (e.g. CLI scripts).
    For GUIs or contexts that want per-tile progress, use
    :func:`generate_scene_stream` instead.
    """
    result: SceneGenerationResult | None = None
    for idx, value in generate_scene_stream(config):
        if idx is None:
            result = value  # type: ignore[assignment]
    assert result is not None
    return result