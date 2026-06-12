"""
Where's Wally image generation pipeline using HuggingFace Diffusers.

Generates one or more tiles independently, stitches them into a grid,
then optionally uses FluxInpaintPipeline to blend the seams.

Usage:
    # Single image
    python wheres_wally_pipeline.py --scene beach

    # 3x3 tiled panorama with seam blending
    python wheres_wally_pipeline.py --tiles 3x3 --seam-fix --scene christmas

    # With LoRA
    python wheres_wally_pipeline.py --lora ./disney_lora.safetensors --lora-strength 0.7
"""

import argparse
import random
from pathlib import Path

STYLE_PREFIX = (
    "where's wally illustration, martin handford style, "
    "isometric birds eye view, flat 2D, no perspective, "
    "crowd of distinct people, bold outlines, bright colours, clean linework, "
    "children's book art, busy scene"
)

NEGATIVE_PROMPT = (
    "photorealistic, 3d render, blurry, low quality, nsfw, violence, "
    "dark, monochrome, simple background, few people, missing limbs, deformed, bad anatomy, "
    "extra limbs, extra fingers, fused fingers, mutated hands, malformed, disfigured, "
    "ugly, poorly drawn hands, poorly drawn face, mutation, text, watermark, "
    "close-up, portrait, zoomed in, macro, few characters, empty space, "
    "sky, horizon, clouds, perspective, vanishing point, depth of field, "
    "street level view, eye level, foreground background, depth, 3d space, "
    "blurry, lowres, bad composition, overexposed, underexposed, out of focus"
)

SCENE_PROMPTS: dict[str, str] = {
    "waterfalls": "birds eye view, isometric view, lush jungle scene with multiple waterfalls, tropical plants, exotic animals, "
                  "tourists with umbrellas, misty atmosphere, rainbow, hanging vines, rocks, "
                  "birds, colorful flowers, hidden paths",
    "beach":      "sandy beach, sunbathers, swimmers, ice cream stand, bbq, "
                  "beach umbrellas, sandcastles, volleyball, beach huts, rocks",
    "fairground": "isometric view, birds eye view,fairground with rides, candy floss stalls, clowns, "
                  "acrobats, ticket booths, food stalls, burger stands, popcorn carts, families with children, balloons, carousel, ferris wheel",
    "museum":     "natural history museum interior with dinosaur skeletons, "
                  "tourists, school children, gift shop",
    "airport":    "birds eye view, isometric view, busy international airport terminal, travellers with luggage, "
                  "departure boards, security queues,",
    "city":       "birds eye view, isometric view, medieval city street festival, market stalls, jesters, "
                  "knights, townspeople",
    "markets":    "isometric view,bustling outdoor market, colourful stalls, fresh fish, spices, fruits, "
                  "vegetables, street food, crabs, hanging meats, bread, flowers, "
                  "shoppers haggling, lanterns, baskets of produce, street vendors",
    "lantern_festival": "birds eye view, isometric view, traditional chinese temples, shrines, bridges over water, lantern festival at night, glowing lanterns, "
                        "lanterns floating on water, vibrant colors, festive atmosphere, food stands,"
                        "festival decorations, shops selling lanterns, shops selling trinkets, cocoa stands"
                        "performers with fire, people releasing lanterns, night sky, reflections in water,",
    "christmas":  "isometric view, christmas market, festive stalls, christmas trees, snow, "
                  "people in winter clothing, twinkling lights, santa hats, carol singers, "
                  "hot cocoa stands, snowmen, ice skaters, wrapped presents, wreaths",
    "halloween":  "birds eye view, isometric view, moonlit sky, bats, creepy trees, trick-or-treaters in costumes, "
                  "witches, ghosts, skeletons, pumpkins, jack-o-lanterns, "
                  "haunted houses, spooky decorations, black cats, bats, gravestones",
    "canal":      "birds eye view, isometric view, venice canal scene with gondolas, bridges, tourists, street musicians, "
                  "waterways, historic buildings, colorful facades, outdoor cafes",
    "library":    "isometric view, grand library interior with towering bookshelves, ladders, readers, "
                  "multiple floors, balconies, ornate architecture, librarian desks, "
                  "stained glass windows, warm lighting, diverse characters",
}

DEFAULT_CHECKPOINT = "black-forest-labs/FLUX.1-dev"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def is_flux(checkpoint: str) -> bool:
    return "flux" in checkpoint.lower()


def is_xl(checkpoint: str) -> bool:
    return "xl" in checkpoint.lower()


def load_pipeline(checkpoint: str, lora: str | None, device: str,
                  lora_strength: float = 0.8, inpaint: bool = False):
    import torch
    from diffusers import (
        FluxInpaintPipeline,
        FluxPipeline,
        StableDiffusionInpaintPipeline,
        StableDiffusionPipeline,
        StableDiffusionXLInpaintPipeline,
        StableDiffusionXLPipeline,
    )

    print(f"  Loading {'inpaint ' if inpaint else ''}pipeline: {checkpoint}")

    if is_flux(checkpoint):
        dtype = torch.bfloat16
        cls = FluxInpaintPipeline if inpaint else FluxPipeline
        pipe = cls.from_pretrained(checkpoint, torch_dtype=dtype)
        pipe.enable_model_cpu_offload()
    else:
        dtype = torch.float16 if device != "cpu" else torch.float32
        if is_xl(checkpoint):
            cls = StableDiffusionXLInpaintPipeline if inpaint else StableDiffusionXLPipeline
        else:
            cls = StableDiffusionInpaintPipeline if inpaint else StableDiffusionPipeline
        cp = Path(checkpoint)
        if checkpoint.startswith("http") or cp.suffix in {".safetensors", ".ckpt"}:
            pipe = cls.from_single_file(checkpoint, torch_dtype=dtype)
        else:
            pipe = cls.from_pretrained(checkpoint, torch_dtype=dtype)
        pipe = pipe.to(device)
        pipe.enable_attention_slicing()

    if lora:
        print(f"  Loading LoRA: {lora} (strength: {lora_strength})")
        lora_path = Path(lora).resolve()
        if lora_path.exists():
            pipe.load_lora_weights(str(lora_path.parent), weight_name=lora_path.name)
        else:
            pipe.load_lora_weights(lora)
        pipe.fuse_lora(lora_scale=lora_strength)

    return pipe


# ---------------------------------------------------------------------------
# Tile generation
# ---------------------------------------------------------------------------

def generate_tile(pipe, prompt: str, seed: int, width: int, height: int,
                  steps: int, cfg: float, device: str):
    import torch
    from diffusers import FluxPipeline
    generator = torch.Generator(device=device).manual_seed(seed)
    kwargs = dict(
        prompt=prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=cfg,
        generator=generator,
    )
    if not isinstance(pipe, FluxPipeline):
        kwargs["negative_prompt"] = NEGATIVE_PROMPT
    return pipe(**kwargs).images[0]


# ---------------------------------------------------------------------------
# Stitching
# ---------------------------------------------------------------------------

def stitch_tiles(tiles: list, cols: int, rows: int, tile_w: int, tile_h: int):
    """Hard-cut stitch — place tiles edge to edge with no blending.
    Seam fix handles the blending separately.
    Final size: cols * tile_w  x  rows * tile_h
    """
    from PIL import Image
    canvas = Image.new("RGB", (cols * tile_w, rows * tile_h))
    for idx, tile in enumerate(tiles):
        row = idx // cols
        col = idx % cols
        canvas.paste(tile, (col * tile_w, row * tile_h))
    return canvas


# ---------------------------------------------------------------------------
# Seam fix
# ---------------------------------------------------------------------------

def _align16(x: int) -> int:
    """Round up to nearest multiple of 16 (required by Flux)."""
    return ((x + 15) // 16) * 16


def _run_inpaint(inpaint_pipe, image, mask, x0, y0, w, h, prompt, seed, steps, cfg):
    """Run inpaint on a region and paste back into image."""
    import torch
    from PIL import Image as PILImage
    patch = image.crop((x0, y0, x0 + w, y0 + h))
    # Flux requires dimensions divisible by 16
    w16 = _align16(w)
    h16 = _align16(h)
    if w16 != w or h16 != h:
        patch = patch.resize((w16, h16))
        mask = mask.resize((w16, h16))
    generator = torch.Generator().manual_seed(seed)
    out = inpaint_pipe(
        prompt=prompt,
        image=patch,
        mask_image=mask,
        width=w16,
        height=h16,
        num_inference_steps=steps,
        guidance_scale=cfg,
        strength=0.80,
        generator=generator,
    ).images[0]
    # Resize back to original patch size before pasting
    if out.size != (w, h):
        from PIL import Image as PILImage
        out = out.resize((w, h), PILImage.LANCZOS)
    image.paste(out, (x0, y0))


def _make_mask(w, h, direction, centre_frac=0.40):
    """Build an inpaint mask.
    direction: 'vertical'  → inpaint centre strip horizontally
               'horizontal' → inpaint centre strip vertically
               'corner'     → inpaint centre block
    Edges (context) = 0 (keep). Centre = 255 (inpaint).
    """
    import numpy as np
    from PIL import Image
    m = np.zeros((h, w), dtype=np.uint8)
    if direction == "vertical":
        cw = int(w * centre_frac)
        cx = (w - cw) // 2
        m[:, cx:cx + cw] = 255
    elif direction == "horizontal":
        ch = int(h * centre_frac)
        cy = (h - ch) // 2
        m[cy:cy + ch, :] = 255
    elif direction == "corner":
        cw = int(w * centre_frac)
        ch = int(h * centre_frac)
        cx = (w - cw) // 2
        cy = (h - ch) // 2
        m[cy:cy + ch, cx:cx + cw] = 255
    return Image.fromarray(m)


def fix_seams(stitched, cols: int, rows: int, tile_w: int, tile_h: int,
              seam_context: int, inpaint_pipe, prompt: str, seed: int,
              steps: int, cfg: float):
    """
    Three-pass seam fix on a hard-cut stitched image.

    seam_context: px from each side of the seam boundary included in the strip.
                  Larger = more context for the model. e.g. 512 means each strip
                  is 1024px wide (512px from each tile).

    Pass 1 — Vertical seams (between columns):
        Strip: seam_context px from tile A right edge + seam_context px from tile B left edge
        Mask:  centre 40% inpainted, edges kept as fixed context
        Chunks: tile_h tall so each chunk is a reasonable size for Flux

    Pass 2 — Horizontal seams (between rows):
        Same logic, horizontal orientation
        Chunks: tile_w wide

    Pass 3 — Corner patches (where seams intersect):
        Done last so corners see already-blended vertical + horizontal seams
        Strip: seam_context px from each of the 4 surrounding tiles
        Mask:  centre 40% in both directions
    """
    result = stitched.copy()
    final_w, final_h = result.size
    half = min(seam_context, tile_w // 2, tile_h // 2)  # can't exceed half a tile

    # Pass 1: vertical seams
    print(f"  Pass 1: {cols - 1} vertical seam(s), strip width={half * 2}px")
    for col in range(1, cols):
        x = col * tile_w                        # exact seam position
        x0 = max(0, x - half)
        x1 = min(final_w, x + half)
        sw = x1 - x0
        for row in range(rows):
            pad = 8
            y0 = max(0, row * tile_h - pad)
            y1 = min(final_h, (row + 1) * tile_h + pad)
            sh = y1 - y0
            mask = _make_mask(sw, sh, "vertical")
            _run_inpaint(inpaint_pipe, result, mask, x0, y0, sw, sh,
                         prompt, seed + col * 100 + row, steps, cfg)

    # Pass 2: horizontal seams
    print(f"  Pass 2: {rows - 1} horizontal seam(s), strip height={half * 2}px")
    for row in range(1, rows):
        y = row * tile_h
        y0 = max(0, y - half)
        y1 = min(final_h, y + half)
        sh = y1 - y0
        for col in range(cols):
            # Add 8px padding on each side so resize artifacts don't land on tile boundaries
            pad = 8
            x0 = max(0, col * tile_w - pad)
            x1 = min(final_w, (col + 1) * tile_w + pad)
            sw = x1 - x0
            mask = _make_mask(sw, sh, "horizontal")
            _run_inpaint(inpaint_pipe, result, mask, x0, y0, sw, sh,
                         prompt, seed + row * 1000 + col, steps, cfg)

    # Pass 3: corners
    n_corners = (cols - 1) * (rows - 1)
    print(f"  Pass 3: {n_corners} corner(s), patch={half * 2}x{half * 2}px")
    for col in range(1, cols):
        for row in range(1, rows):
            x = col * tile_w
            y = row * tile_h
            x0 = max(0, x - half)
            x1 = min(final_w, x + half)
            y0 = max(0, y - half)
            y1 = min(final_h, y + half)
            sw = x1 - x0
            sh = y1 - y0
            mask = _make_mask(sw, sh, "corner")
            _run_inpaint(inpaint_pipe, result, mask, x0, y0, sw, sh,
                         prompt, seed + row * 10000 + col * 1000, steps, cfg)

    return result


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    scene: str = "beach",
    custom_prompt: str | None = None,
    seed: int | None = None,
    out_path: str | None = None,
    width: int = 1024,
    height: int = 768,
    checkpoint: str = DEFAULT_CHECKPOINT,
    lora: str | None = None,
    lora_strength: float = 0.8,
    steps: int = 30,
    cfg: float = 7.5,
    device: str = "cuda",
    tiles: str | None = None,
    seam_context: int = 512,
    seam_fix: bool = False,
) -> Path:
    import torch

    seed = seed if seed is not None else random.randint(0, 2**32 - 1)
    scene_detail = SCENE_PROMPTS.get(scene, scene)
    full_prompt = f"{STYLE_PREFIX}, {custom_prompt or scene_detail}"

    print(f"Scene   : {scene}")
    print(f"Seed    : {seed}")
    print(f"Device  : {device}")
    print(f"Prompt  : {full_prompt[:120]}…")

    if tiles:
        cols, rows = (int(x) for x in tiles.lower().split("x"))
        total = cols * rows
        print(f"Tiles   : {cols}x{rows} = {total} tiles, final {cols*width}x{rows*height}px")

        pipe = load_pipeline(checkpoint, lora, device, lora_strength, inpaint=False)
        inpaint_pipe = load_pipeline(checkpoint, lora, device, lora_strength, inpaint=True) if seam_fix else None

        generated = []
        for row in range(rows):
            for col in range(cols):
                i = row * cols + col
                tile_seed = seed + i
                print(f"  Generating tile ({row},{col}) [{i+1}/{total}] seed={tile_seed} …")
                tile = generate_tile(pipe, full_prompt, tile_seed, width, height, steps, cfg, device)
                generated.append(tile)

        print("  Stitching …")
        image = stitch_tiles(generated, cols, rows, width, height)
        print(f"  Stitched: {image.size[0]}x{image.size[1]}px")

        if seam_fix:
            print(f"  Seam fix (seam_context={seam_context}px per side) …")
            image = fix_seams(image, cols, rows, width, height,
                              seam_context, inpaint_pipe, full_prompt, seed, steps, cfg)
    else:
        pipe = load_pipeline(checkpoint, lora, device, lora_strength)
        print(f"Size    : {width}x{height}")
        print("  Generating …")
        image = generate_tile(pipe, full_prompt, seed, width, height, steps, cfg, device)

    out = Path(out_path) if out_path else Path(f"wally_{scene}_{seed}.png")
    image.save(out)
    print(f"  Saved → {out}")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Where's Wally image pipeline (diffusers)")
    parser.add_argument("--scene", default="beach",
                        choices=list(SCENE_PROMPTS.keys()) + ["custom"],
                        help="Preset scene (default: beach)")
    parser.add_argument("--prompt", default=None, help="Custom scene description")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", default=None, help="Output PNG path")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                        help="HuggingFace repo ID or local .safetensors path")
    parser.add_argument("--lora", default=None,
                        help="HuggingFace repo ID or local .safetensors path for a LoRA")
    parser.add_argument("--lora-strength", type=float, default=0.8,
                        help="LoRA weight 0.0–1.0 (default: 0.8)")
    parser.add_argument("--tiles", default=None,
                        help="Tile grid e.g. 3x2 (cols x rows). Generates multiple tiles and stitches.")
    parser.add_argument("--seam-fix", action="store_true",
                        help="After tiling, blend seams using FluxInpaintPipeline (requires --tiles)")
    parser.add_argument("--seam-context", type=int, default=512,
                        help="Pixels from each side of seam included in inpaint strip (default: 512). "
                             "Larger = more context for the model.")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--cfg", type=float, default=7.5)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"])
    args = parser.parse_args()

    run_pipeline(
        scene=args.scene,
        custom_prompt=args.prompt,
        seed=args.seed,
        out_path=args.out,
        width=args.width,
        height=args.height,
        checkpoint=args.checkpoint,
        lora=args.lora,
        lora_strength=args.lora_strength,
        steps=args.steps,
        cfg=args.cfg,
        device=args.device,
        tiles=args.tiles,
        seam_context=args.seam_context,
        seam_fix=args.seam_fix,
    )


if __name__ == "__main__":
    main()
