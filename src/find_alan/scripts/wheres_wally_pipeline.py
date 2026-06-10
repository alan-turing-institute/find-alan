"""
Where's Wally image generation pipeline using HuggingFace Diffusers.

Works with local .safetensors checkpoints or HuggingFace repo IDs.
Optionally applies a LoRA on top.

Usage:
    python wheres_wally_pipeline.py
    python wheres_wally_pipeline.py --scene fairground --seed 42
    python wheres_wally_pipeline.py --checkpoint RunDiffusion/Juggernaut-XL-v9 --lora pytorch_lora_weights.safetensors
    python wheres_wally_pipeline.py --checkpoint ./DreamShaper_8_pruned.safetensors --scene city --out city.png
"""

import argparse
import random
from pathlib import Path

STYLE_PREFIX = (
    "flat 2D illustration, cartoon comic book style, line art, "
    "where's wally art style, where's waldo art style, where's walter art style, "
    "extremely detailed crowd scene, "
    "lots of people doing various activities, busy and chaotic, "
    "small tiny people, flat cartoon art, bright colours, "
    "illustrated book style, isometric view, wide angle, zoomed out, bird's eye view, "
    "hand-drawn look, highly detailed, colourful, full scene visible"
)

NEGATIVE_PROMPT = (
    "photorealistic, 3d render, blurry, low quality, nsfw, violence, "
    "dark, monochrome, simple background, few people, missing limbs, deformed, bad anatomy, "
    "extra limbs, extra fingers, fused fingers, mutated hands, malformed, disfigured, "
    "ugly, poorly drawn hands, poorly drawn face, mutation, text, watermark, "
    "close-up, portrait, zoomed in, macro, few characters, empty space, "
    "blurry, lowres, bad composition, overexposed, underexposed, out of focus"
)

SCENE_PROMPTS: dict[str, str] = {
    "beach":      "beach scene with sunbathers, lifeguards, ice cream sellers, "
                  "volleyball players, sandcastles",
    "fairground": "fairground with rides, candy floss stalls, clowns, "
                  "acrobats, ticket booths",
    "museum":     "natural history museum interior with dinosaur skeletons, "
                  "tourists, school children, gift shop",
    "airport":    "busy international airport terminal, travellers with luggage, "
                  "departure boards, security queues",
    "city":       "medieval city street festival, market stalls, jesters, "
                  "knights, townspeople",
    "markets":    "bustling outdoor market, colourful stalls, fresh fish, spices, fruits, "
                  "vegetables, street food, crabs, hanging meats, bread, flowers, "
                  "shoppers haggling, lanterns, baskets of produce, street vendors",
                  
    "lantern_festival": "lantern festival at night, crowds of people, glowing lanterns into the sky, "
                        "lanterns floating on water, lanterns hanging from trees, "
                        "lanterns of various shapes and sizes, vibrant colors, "
                        "festive atmosphere, reflections of lanterns in water, "
                        "people releasing lanterns, lanterns illuminating faces, "
                        "lanterns drifting in the night sky, lanterns creating a magical ambiance"
                        "body of water, trees, night sky, reflections, crowds of people,",
                            
    "christmas":  "christmas market scene with festive stalls, christmas trees, "
                    "people in winter clothing, snow, twinkling lights, santa hats, carol singers, hot cocoa stands, holiday decorations,"
                    "snowmen, ice skaters, wrapped presents, wreaths, gingerbread houses, festive atmosphere,"
                    "snow-covered ground, cozy cabins, christmas ornaments, holiday cheer,"
                    "families enjoying the market, christmas music, seasonal treats, warm scarves and mittens,"
                    "christmas lights illuminating the scene, joyful crowds, festive spirit,",
        
    "halloween":   "halloween night scene with trick-or-treaters in costumes, "
                   "pumpkin patches, haunted houses, witches, ghosts, skeletons, "
                   "spooky decorations, black cats, bats, moonlit sky",

    "canal":       "venice canal scene with gondolas, bridges, tourists, street musicians, boats"
                    "waterways, historic buildings, colorful facades, outdoor cafes, reflections in water, " 
                   "water, boats, historic architecture, bustling atmosphere,",
                   
    "library":     "grand library interior with towering bookshelves, ladders, readers, "
                    "multiple floors, balconies, ornate architecture, quiet study areas, hidden nooks, librarian desks, "
                    "librarians, study tables, globes, ancient tomes"
                    "warm lighting, cozy atmosphere, diverse characters, quiet study areas, "
                    "stained glass windows, ornate woodwork, hidden nooks, banker lamps,"
                    }
                        
                        
DEFAULT_CHECKPOINT = "RunDiffusion/Juggernaut-XL-v9"


def is_flux(checkpoint: str) -> bool:
    return "flux" in checkpoint.lower()


def is_xl(checkpoint: str) -> bool:
    return "xl" in checkpoint.lower()


def load_pipeline(checkpoint: str, lora: str | None, device: str, lora_strength: float = 0.8, inpaint: bool = False):
    import torch
    from diffusers import (
        FluxPipeline,
        StableDiffusionPipeline,
        StableDiffusionXLPipeline,
        StableDiffusionXLInpaintPipeline,
        StableDiffusionInpaintPipeline,
    )

    print(f"  Loading checkpoint: {checkpoint}{' (inpaint)' if inpaint else ''}")

    if is_flux(checkpoint):
        dtype = torch.bfloat16
        if inpaint:
            from diffusers import FluxInpaintPipeline
            pipe = FluxInpaintPipeline.from_pretrained(checkpoint, torch_dtype=dtype)
        else:
            pipe = FluxPipeline.from_pretrained(checkpoint, torch_dtype=dtype)
        pipe.enable_model_cpu_offload()
    else:
        dtype = torch.float16 if device != "cpu" else torch.float32
        if is_xl(checkpoint):
            PipelineClass = StableDiffusionXLInpaintPipeline if inpaint else StableDiffusionXLPipeline
        else:
            PipelineClass = StableDiffusionInpaintPipeline if inpaint else StableDiffusionPipeline
        cp = Path(checkpoint)
        is_single_file = checkpoint.startswith("http") or cp.suffix in {".safetensors", ".ckpt"}
        if is_single_file:
            pipe = PipelineClass.from_single_file(checkpoint, torch_dtype=dtype)
        else:
            pipe = PipelineClass.from_pretrained(checkpoint, torch_dtype=dtype)
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


def generate_tile(pipe, prompt: str, seed: int, width: int, height: int, steps: int, cfg: float, device: str):
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
    # Flux doesn't support negative_prompt
    if not isinstance(pipe, FluxPipeline):
        kwargs["negative_prompt"] = NEGATIVE_PROMPT
    result = pipe(**kwargs)
    return result.images[0]


def outpaint_tile(pipe, inpaint_pipe, prev_tile, prompt: str, seed: int, width: int, height: int,
                  steps: int, cfg: float, device: str, overlap: int = 256, direction: str = "right"):
    """Extend prev_tile using inpainting (SDXL) or img2img (Flux)."""
    from PIL import Image
    import numpy as np
    import torch

    # Build canvas with the known edge pasted in
    canvas = Image.new("RGB", (width, height), (128, 128, 128))
    # SDXL mask: 255=inpaint, 0=keep
    # FluxFill mask: 255=inpaint, 0=keep (same convention)
    mask   = Image.new("L", (width, height), 255)

    if direction == "right":
        mirrored = prev_tile.transpose(Image.FLIP_LEFT_RIGHT).resize((width, height))
        canvas.paste(mirrored, (0, 0))
        edge = prev_tile.crop((prev_tile.width - overlap, 0, prev_tile.width, prev_tile.height))
        canvas.paste(edge, (0, 0))
        mask_arr = np.array(mask)
        mask_arr[:, :overlap] = 0  # keep the left edge strip
        mask = Image.fromarray(mask_arr)
    elif direction == "down":
        mirrored = prev_tile.transpose(Image.FLIP_TOP_BOTTOM).resize((width, height))
        canvas.paste(mirrored, (0, 0))
        edge = prev_tile.crop((0, prev_tile.height - overlap, prev_tile.width, prev_tile.height))
        canvas.paste(edge, (0, 0))
        mask_arr = np.array(mask)
        mask_arr[:overlap, :] = 0  # keep the top edge strip
        mask = Image.fromarray(mask_arr)

    generator = torch.Generator(device=device).manual_seed(seed)

    from diffusers import FluxInpaintPipeline
    is_flux_inpaint = isinstance(inpaint_pipe, FluxInpaintPipeline)

    if is_flux_inpaint:
        result = inpaint_pipe(
            prompt=prompt,
            image=canvas,
            mask_image=mask,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=cfg,
            strength=0.4,
            generator=generator,
        )
    elif inpaint_pipe is not None:
        result = inpaint_pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            image=canvas,
            mask_image=mask,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=cfg,
            generator=generator,
        )
    else:
        raise RuntimeError("No inpaint pipeline loaded.")

    out = result.images[0]
    if out.size != (width, height):
        out = out.resize((width, height))
    return out


def fix_seams(stitched, tiles: list, cols: int, rows: int, tile_w: int, tile_h: int,
              overlap: int, inpaint_pipe, prompt: str, seed: int, steps: int, cfg: float):
    """Run FLUX.1-Fill over just the seam strips to blend tile joins."""
    from PIL import Image
    import numpy as np
    import torch

    from diffusers import FluxInpaintPipeline
    is_flux_inpaint = isinstance(inpaint_pipe, FluxInpaintPipeline)
    fill_cfg = cfg

    result = stitched.copy()
    step_x = tile_w - overlap
    step_y = tile_h - overlap

    # Fix vertical seams (between columns)
    for col in range(1, cols):
        x = col * step_x + overlap // 2  # true seam centre
        x0 = max(0, x - overlap)
        x1 = min(stitched.width, x + overlap)

        # Crop the seam strip from stitched image
        seam_strip = result.crop((x0, 0, x1, stitched.height))
        strip_w = x1 - x0
        strip_h = stitched.height

        # Mask: inpaint only a narrow strip at the centre seam
        seam_half = max(32, overlap // 8)
        mask = Image.new("L", (strip_w, strip_h), 0)
        mask_arr = np.array(mask)
        centre = strip_w // 2
        mask_arr[:, centre - seam_half:centre + seam_half] = 255
        mask = Image.fromarray(mask_arr)

        generator = torch.Generator().manual_seed(seed + col)
        print(f"  Fixing vertical seam at x={x} …")
        out = inpaint_pipe(
            prompt=prompt,
            image=seam_strip,
            mask_image=mask,
            width=strip_w,
            height=strip_h,
            num_inference_steps=steps,
            guidance_scale=fill_cfg,
            strength=0.4,
            generator=generator,
        ).images[0]

        if out.size != (strip_w, strip_h):
            out = out.resize((strip_w, strip_h))
        result.paste(out, (x0, 0))

    # Fix horizontal seams (between rows)
    for row in range(1, rows):
        y = row * step_y + overlap // 2  # true seam centre
        y0 = max(0, y - overlap)
        y1 = min(stitched.height, y + overlap)

        seam_strip = result.crop((0, y0, stitched.width, y1))
        strip_w = stitched.width
        strip_h = y1 - y0

        seam_half = max(32, overlap // 8)
        mask = Image.new("L", (strip_w, strip_h), 0)
        mask_arr = np.array(mask)
        centre = strip_h // 2
        mask_arr[centre - seam_half:centre + seam_half, :] = 255
        mask = Image.fromarray(mask_arr)

        generator = torch.Generator().manual_seed(seed + cols + row)
        print(f"  Fixing horizontal seam at y={y} …")
        out = inpaint_pipe(
            prompt=prompt,
            image=seam_strip,
            mask_image=mask,
            width=strip_w,
            height=strip_h,
            num_inference_steps=steps,
            guidance_scale=fill_cfg,
            strength=0.4,
            generator=generator,
        ).images[0]

        if out.size != (strip_w, strip_h):
            out = out.resize((strip_w, strip_h))
        result.paste(out, (0, y0))

    return result


def stitch_tiles_hard(tiles: list, cols: int, rows: int, tile_w: int, tile_h: int, overlap: int = 0):
    """Stitch tiles — skip the duplicate overlap strip, blend just 8px at the cut to hide the seam line."""
    from PIL import Image
    import numpy as np

    feather = 8  # narrow blend just to hide the 1px seam artifact
    step_x = tile_w - overlap
    step_y = tile_h - overlap
    final_w = tile_w + step_x * (cols - 1)
    final_h = tile_h + step_y * (rows - 1)
    canvas = Image.new("RGB", (final_w, final_h))

    for idx, tile in enumerate(tiles):
        row = idx // cols
        col = idx % cols
        x = col * step_x
        y = row * step_y
        if col > 0:
            # Paste everything after the overlap strip
            tile_cropped = tile.crop((overlap + feather, 0, tile_w, tile_h))
            canvas.paste(tile_cropped, (x + overlap + feather, y))
            # Feather blend just the first `feather` pixels after the cut
            canvas_arr = np.array(canvas).astype(np.float32)
            tile_arr = np.array(tile).astype(np.float32)
            for f in range(feather):
                alpha = f / feather
                px = x + overlap + f
                canvas_arr[y:y+tile_h, px] = (
                    (1 - alpha) * canvas_arr[y:y+tile_h, px] +
                    alpha * tile_arr[:, overlap + f]
                )
            canvas = Image.fromarray(canvas_arr.astype(np.uint8))
        else:
            canvas.paste(tile, (x, y))

    return canvas


def stitch_tiles(tiles: list, cols: int, rows: int, tile_w: int, tile_h: int, overlap: int = 128):
    """Stitch tiles into a panorama with feathered blending at edges."""
    from PIL import Image
    import numpy as np

    step_x = tile_w - overlap
    step_y = tile_h - overlap
    final_w = tile_w + step_x * (cols - 1)
    final_h = tile_h + step_y * (rows - 1)

    canvas = np.zeros((final_h, final_w, 3), dtype=np.float32)
    weight  = np.zeros((final_h, final_w, 1), dtype=np.float32)

    # Build a feathered weight mask for one tile
    mask = np.ones((tile_h, tile_w), dtype=np.float32)
    if overlap > 0:
        ramp = np.linspace(0, 1, overlap)
        mask[:, :overlap]  *= ramp
        mask[:, -overlap:] *= ramp[::-1]
        mask[:overlap, :]  *= ramp[:, None]
        mask[-overlap:, :] *= ramp[::-1, None]

    for idx, tile in enumerate(tiles):
        row = idx // cols
        col = idx % cols
        x = col * step_x
        y = row * step_y
        arr = np.array(tile).astype(np.float32)
        w = mask[:, :, None]
        canvas[y:y+tile_h, x:x+tile_w] += arr * w
        weight[y:y+tile_h, x:x+tile_w] += w

    stitched = np.clip(canvas / np.maximum(weight, 1e-6), 0, 255).astype(np.uint8)
    return Image.fromarray(stitched)


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
    overlap: int = 128,
    outpaint: bool = False,
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

    if tiles and outpaint:
        cols, rows = (int(x) for x in tiles.lower().split("x"))
        print(f"Mode    : outpaint {cols}x{rows}")
        pipe = load_pipeline(checkpoint, lora, device, lora_strength, inpaint=False)
        inpaint_pipe = load_pipeline(checkpoint, lora, device, lora_strength, inpaint=True)

        all_tiles = []
        for row in range(rows):
            row_tiles = []
            for col in range(cols):
                tile_seed = seed + row * cols + col
                if col == 0 and row == 0:
                    print(f"  Generating seed tile (0,0) …")
                    tile = generate_tile(pipe, full_prompt, tile_seed, width, height, steps, cfg, device)
                elif col > 0:
                    prev = row_tiles[-1]
                    print(f"  Outpainting right → tile ({row},{col}) …")
                    tile = outpaint_tile(pipe, inpaint_pipe, prev, full_prompt, tile_seed,
                                        width, height, steps, cfg, device, overlap, "right")
                else:
                    prev = all_tiles[-1][0]
                    print(f"  Outpainting down → tile ({row},{col}) …")
                    tile = outpaint_tile(pipe, inpaint_pipe, prev, full_prompt, tile_seed,
                                        width, height, steps, cfg, device, overlap, "down")
                row_tiles.append(tile)
            all_tiles.append(row_tiles)

        flat_tiles = [t for row_tiles in all_tiles for t in row_tiles]
        print("  Stitching …")
        # Outpainted tiles share edges via the overlap context — stitch with overlap so edges blend
        image = stitch_tiles(flat_tiles, cols, rows, width, height, overlap // 2)

    elif tiles:
        cols, rows = (int(x) for x in tiles.lower().split("x"))
        total = cols * rows
        print(f"Tiles   : {cols}x{rows} = {total} tiles")
        pipe = load_pipeline(checkpoint, lora, device, lora_strength, inpaint=False)
        inpaint_pipe = load_pipeline(checkpoint, lora, device, lora_strength, inpaint=True)
        generated = []
        tile_variations = [
            "left section", "centre-left section", "centre section",
            "centre-right section", "right section", "far right section",
            "top-left corner", "top-right corner", "bottom-left corner",
        ]
        for i in range(total):
            tile_seed = seed + i
            if i == 0:
                # First tile — generate normally
                print(f"  Generating tile 1/{total} (seed {tile_seed}) …")
                tile = generate_tile(pipe, full_prompt, tile_seed, width, height, steps, cfg, device)
            else:
                # Subsequent tiles — outpaint from previous tile's right edge using inpainting
                from PIL import Image
                import numpy as np
                import torch
                prev = generated[-1]
                edge = prev.crop((prev.width - overlap, 0, prev.width, prev.height))
                # Canvas: real edge on left, black on right (model fills the black area)
                canvas = Image.new("RGB", (width, height), (0, 0, 0))
                canvas.paste(edge, (0, 0))
                # Mask: 0 = keep edge, 255 = generate new content
                mask = Image.new("L", (width, height), 255)
                mask_arr = np.array(mask)
                mask_arr[:, :overlap] = 0
                mask = Image.fromarray(mask_arr)
                generator = torch.Generator(device=device).manual_seed(tile_seed)
                print(f"  Generating tile {i+1}/{total} from edge (seed {tile_seed}) …")
                result = inpaint_pipe(
                    prompt=full_prompt,
                    image=canvas,
                    mask_image=mask,
                    width=width,
                    height=height,
                    num_inference_steps=steps,
                    guidance_scale=cfg,
                    strength=1.0,  # only noise the masked area, edge stays pixel-perfect
                    generator=generator,
                )
                tile = result.images[0]
                if tile.size != (width, height):
                    tile = tile.resize((width, height))
            generated.append(tile)
        print("  Stitching …")
        # Tile 2+ were generated with tile 1's edge as left anchor, so stitch with
        # overlap offset but NO feathered blending — tile edges already match
        image = stitch_tiles_hard(generated, cols, rows, width, height, overlap)
        final_w = width + (width - overlap) * (cols - 1)
        final_h = height + (height - overlap) * (rows - 1)
        print(f"  Final size: {final_w}x{final_h}")
        if seam_fix:
            print("  Fixing seams …")
            image = fix_seams(image, generated, cols, rows, width, height,
                              overlap, inpaint_pipe, full_prompt, seed, steps, cfg)
    else:
        pipe = load_pipeline(checkpoint, lora, device, lora_strength)
        print(f"Size    : {width}x{height}")
        print("  Generating …")
        image = generate_tile(pipe, full_prompt, seed, width, height, steps, cfg, device)

    out = Path(out_path) if out_path else Path(f"wally_{scene}_{seed}.png")
    image.save(out)
    print(f"  Saved → {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Where's Wally image pipeline (diffusers)")
    parser.add_argument(
        "--scene",
        default="beach",
        choices=list(SCENE_PROMPTS.keys()) + ["custom"],
        help="Preset scene (default: beach)",
    )
    parser.add_argument("--prompt", default=None, help="Custom scene description")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", default=None, help="Output PNG path")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help="HuggingFace repo ID or local .safetensors path",
    )
    parser.add_argument(
        "--lora",
        default=None,
        help="HuggingFace repo ID or local .safetensors path for a LoRA",
    )
    parser.add_argument("--lora-strength", type=float, default=0.8, help="LoRA weight 0.0–1.0 (default: 0.8)")
    parser.add_argument("--tiles", default=None, help="Grid of tiles e.g. 3x2 (cols x rows)")
    parser.add_argument("--overlap", type=int, default=256, help="Overlap in pixels for tile blending (default: 256)")
    parser.add_argument("--outpaint", action="store_true", help="Use outpainting to extend the scene from tile edges (requires --tiles)")
    parser.add_argument("--seam-fix", action="store_true", help="After tiling, use FLUX.1-Fill to blend seams (requires --tiles)")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--cfg", type=float, default=7.5)
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu", "mps"],
        help="Device to run on (default: cuda)",
    )
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
        overlap=args.overlap,
        outpaint=args.outpaint,
        seam_fix=args.seam_fix,
    )


if __name__ == "__main__":
    main()
