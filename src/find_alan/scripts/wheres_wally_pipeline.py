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
    "flat 2D illustration, cartoon comic book style, line art"
    "where's wally art style, where's waldo art style, where's walter art style,"
    "extremely detailed crowd scene, "
    "lots of people doing various activities, busy and chaotic, crowd "
    "small people, flat cartoon art, bright colours, "
    "illustrated book style, top-down view, "
    "hand-drawn look, highly detailed, colourful"
)

NEGATIVE_PROMPT = (
    "photorealistic, 3d render, blurry, low quality, nsfw, violence, "
    "dark, monochrome, simple background, few people, missing limbs, deformed, bad anatomy, "
    "extra limbs, extra fingers, fused fingers, mutated hands, malformed, disfigured, "
    "ugly, poorly drawn hands, poorly drawn face, mutation, text, watermark,"
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
    "markets":    "bustling outdoor market with colourful stalls selling fresh fish,"
                    "spices, fruits, vegetables, textiles, pottery, jewelry, street food, skewers"
                  "whole crabs, exotic fruits, tropical vegetables, spices, "
                  "hanging meats, bread loaves, cheese wheels, olives, dried herbs, flowers, "
                  "handmade clothes, pottery, trinkets, shoppers haggling, street food vendors, "
                  "lanterns lining the markets, baskets overflowing with produce",
                  
    "halloween":   "halloween night scene with trick-or-treaters in costumes, "
                   "pumpkin patches, haunted houses, witches, ghosts, skeletons, "
                   "spooky decorations, black cats, bats, moonlit sky",
}

DEFAULT_CHECKPOINT = "RunDiffusion/Juggernaut-XL-v9"


def is_flux(checkpoint: str) -> bool:
    return "flux" in checkpoint.lower()


def is_xl(checkpoint: str) -> bool:
    return "xl" in checkpoint.lower()


def load_pipeline(checkpoint: str, lora: str | None, device: str, lora_strength: float = 0.8, inpaint: bool = False):
    import torch
    from diffusers import (
        FluxFillPipeline,
        FluxPipeline,
        StableDiffusionInpaintPipeline,
        StableDiffusionPipeline,
        StableDiffusionXLInpaintPipeline,
        StableDiffusionXLPipeline,
    )

    print(f"  Loading checkpoint: {checkpoint}{' (inpaint)' if inpaint else ''}")

    if is_flux(checkpoint):
        dtype = torch.bfloat16
        if inpaint:
            pipe = FluxFillPipeline.from_pretrained("black-forest-labs/FLUX.1-Fill-dev", torch_dtype=dtype)
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
        is_single_file = (
            checkpoint.startswith("http")
            or (cp.suffix in {".safetensors", ".ckpt"})
        )
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
    generator = torch.Generator(device=device).manual_seed(seed)
    result = pipe(
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=cfg,
        generator=generator,
    )
    return result.images[0]


def outpaint_tile(inpaint_pipe, prev_tile, prompt: str, seed: int, width: int, height: int,
                  steps: int, cfg: float, device: str, overlap: int = 256, direction: str = "right"):
    """Extend prev_tile in the given direction using inpainting."""
    from PIL import Image
    import numpy as np
    import torch

    # Build a canvas: paste the known edge strip, mask the rest
    canvas = Image.new("RGB", (width, height), (0, 0, 0))
    mask   = Image.new("L",   (width, height), 255)  # 255 = inpaint this area

    if direction == "right":
        # Take the right `overlap` pixels of prev_tile as the known left edge
        edge = prev_tile.crop((prev_tile.width - overlap, 0, prev_tile.width, prev_tile.height))
        canvas.paste(edge, (0, 0))
        # Known area = left overlap strip → mask = 0 there
        mask_arr = np.array(mask)
        mask_arr[:, :overlap] = 0
        mask = Image.fromarray(mask_arr)
    elif direction == "down":
        edge = prev_tile.crop((0, prev_tile.height - overlap, prev_tile.width, prev_tile.height))
        canvas.paste(edge, (0, 0))
        mask_arr = np.array(mask)
        mask_arr[:overlap, :] = 0
        mask = Image.fromarray(mask_arr)

    generator = torch.Generator(device=device).manual_seed(seed)
    result = inpaint_pipe(
        prompt=prompt,
        image=canvas,
        mask_image=mask,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=cfg,
        generator=generator,
    )
    return result.images[0]


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
        pipe      = load_pipeline(checkpoint, lora, device, lora_strength, inpaint=False)
        inpaint_pipe = load_pipeline(checkpoint, lora, device, lora_strength, inpaint=True)

        # Generate all tiles row by row using outpainting
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
                    tile = outpaint_tile(inpaint_pipe, prev, full_prompt, tile_seed,
                                        width, height, steps, cfg, device, overlap, "right")
                else:
                    prev = all_tiles[-1][0]
                    print(f"  Outpainting down → tile ({row},{col}) …")
                    tile = outpaint_tile(inpaint_pipe, prev, full_prompt, tile_seed,
                                        width, height, steps, cfg, device, overlap, "down")
                row_tiles.append(tile)
            all_tiles.append(row_tiles)

        flat_tiles = [t for row_tiles in all_tiles for t in row_tiles]
        print("  Stitching …")
        image = stitch_tiles(flat_tiles, cols, rows, width, height, overlap)

    elif tiles:
        cols, rows = (int(x) for x in tiles.lower().split("x"))
        total = cols * rows
        print(f"Tiles   : {cols}x{rows} = {total} tiles")
        pipe = load_pipeline(checkpoint, lora, device, lora_strength)
        generated = []
        tile_variations = [
            "left section", "centre-left section", "centre section",
            "centre-right section", "right section", "far right section",
            "top-left corner", "top-right corner", "bottom-left corner",
        ]
        for i in range(total):
            variation = tile_variations[i % len(tile_variations)]
            tile_prompt = f"{full_prompt}, {variation} of the scene"
            tile_seed = seed + i
            print(f"  Generating tile {i+1}/{total} ({variation}, seed {tile_seed}) …")
            tile = generate_tile(pipe, tile_prompt, tile_seed, width, height, steps, cfg, device)
            generated.append(tile)
        print("  Stitching …")
        image = stitch_tiles(generated, cols, rows, width, height, overlap)
        final_w = width + (width - overlap) * (cols - 1)
        final_h = height + (height - overlap) * (rows - 1)
        print(f"  Final size: {final_w}x{final_h}")
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
    )


if __name__ == "__main__":
    main()
