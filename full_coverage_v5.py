"""
Full coverage iterative refinement - random mini-batch sampling.

Each iteration:
  - Uses a different offset grid (cycles through 4) for full pixel coverage
  - Shuffles all patches randomly
  - Greedily packs non-overlapping patches into mini-batches (MAX_BATCH_SIZE)
  - Processes each mini-batch, writes back immediately
  - Continues until all patches in this iteration are processed

This eliminates top-left bias from fixed checkerboard ordering.
Edge/corner patches use edge-aware masks automatically.
Complete pixel coverage guaranteed across 4 iterations (one per offset grid).

Env vars:
  INPUT_IMAGE, OUTPUT_DIR
  NUM_ITERS       total iterations (default 4)
  MAX_BATCH_SIZE  max patches per mini-batch (default 12)
  STRENGTH        inpainting strength (default 0.2)
  ITERS_PER_PASS  kept for compatibility but ignored (use NUM_ITERS)
"""

import torch
import numpy as np
import os
import time
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter
from diffusers import FluxInpaintPipeline

# ── Config ────────────────────────────────────────────────────────────────────

INPUT_IMAGE = Path(os.environ.get(
    "INPUT_IMAGE",
    "/lus/lfs1aip2/projects/u6ge/greder/bare-coordinate/outputs/upscaled/image_refined.png"
))

OUTPUT_DIR = Path(os.environ.get(
    "OUTPUT_DIR",
    "/lus/lfs1aip2/projects/u6ge/greder/bare-coordinate/outputs/full_coverage_v5"
))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FLUX_MODEL = "black-forest-labs/FLUX.1-dev"

OUTER_SIZE = 512
INNER_RATIO = 0.5
FEATHER = 4

NUM_ITERS      = int(os.environ.get("NUM_ITERS", 4))
MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", 12))
STRENGTH       = float(os.environ.get("STRENGTH", 0.2))
NUM_STEPS      = 28
GUIDANCE       = 3.5
BASE_SEED      = 42

GIF_FRAME_DURATION = 400

PROMPT = (
    "Where's Wally illustration style, flat cartoon art, bold black outlines, "
    "tiny distinct human figures each clearly separated with visible outlines, "
    "every person has a clear silhouette and individual identity, "
    "crisp clean linework, no merged figures, no blurry regions, "
    "limited bright color palette red yellow blue beige, "
    "isometric bird's eye view, dense crowd, Martin Handford style"
)

NEGATIVE_PROMPT = (
    "blurry, smeared, merged figures, indistinct blobs, artifacts, "
    "distorted limbs, melted shapes, incoherent crowd, fuzzy, "
    "photorealistic, 3d render, dark, moody"
)

# ── Load image ────────────────────────────────────────────────────────────────

print(f"Loading: {INPUT_IMAGE}")
image = Image.open(INPUT_IMAGE).convert("RGB")
width, height = image.size
print(f"Image size: {width}x{height}")

current_image = image.copy()

inner_size = int(OUTER_SIZE * INNER_RATIO)   # 256
offset     = (OUTER_SIZE - inner_size) // 2  # 128

print(f"Outer={OUTER_SIZE} Inner={inner_size} Offset={offset}")

# ── Mask factory ──────────────────────────────────────────────────────────────

def make_mask(at_left, at_right, at_top, at_bottom):
    """Edge-aware PIL mask. Inner region extends to image boundary on clamped sides."""
    x1 = 0          if at_left   else offset
    y1 = 0          if at_top    else offset
    x2 = OUTER_SIZE if at_right  else offset + inner_size
    y2 = OUTER_SIZE if at_bottom else offset + inner_size
    mask = Image.new("L", (OUTER_SIZE, OUTER_SIZE), 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle([x1, y1, x2, y2], fill=255)
    if FEATHER > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(FEATHER))
    return mask

# ── Patch generation ──────────────────────────────────────────────────────────

def get_patches(width, height, outer_size, inner_size, offset, shift_x=0, shift_y=0):
    """
    Generate patches with edge-to-edge inner coverage.
    Edge/corner patches automatically detected and flagged.
    """
    patches = []
    y = shift_y
    while y < height:
        x = shift_x
        while x < width:
            inner_x2 = min(x + inner_size, width)
            inner_y2 = min(y + inner_size, height)

            px1 = max(0, x - offset)
            py1 = max(0, y - offset)
            px2 = min(width,  inner_x2 + offset)
            py2 = min(height, inner_y2 + offset)

            patches.append({
                "patch_box": (px1, py1, px2, py2),
                "inner_box": (x, y, inner_x2, inner_y2),
                "at_left":   px1 == 0,
                "at_right":  px2 == width,
                "at_top":    py1 == 0,
                "at_bottom": py2 == height,
            })
            x += inner_size
        y += inner_size
    return patches

# 4 offset grids for full coverage — cycled across iterations
HALF_INNER = inner_size // 2
PASS_SHIFTS = [
    (0,          0,          "aligned"),
    (HALF_INNER, 0,          "shift_right"),
    (0,          HALF_INNER, "shift_down"),
    (HALF_INNER, HALF_INNER, "shift_diagonal"),
]

# Verify each grid gives full coverage
print("\nVerifying coverage per grid...")
for sx, sy, name in PASS_SHIFTS:
    covered = np.zeros((height, width), dtype=np.int32)
    for p in get_patches(width, height, OUTER_SIZE, inner_size, offset, sx, sy):
        ix1, iy1, ix2, iy2 = p["inner_box"]
        covered[iy1:iy2, ix1:ix2] += 1
    uncovered = int((covered == 0).sum())
    print(f"  {name}: {uncovered} uncovered pixels {'✓' if uncovered == 0 else '✗'}")

# ── Non-overlap check ─────────────────────────────────────────────────────────

def boxes_overlap(b1, b2):
    """True if two patch_boxes overlap."""
    return (b1[0] < b2[2] and b1[2] > b2[0] and
            b1[1] < b2[3] and b1[3] > b2[1])

def build_random_batches(patches, max_batch_size, rng):
    """
    Shuffle patches randomly, then greedily pack non-overlapping patches
    into mini-batches of up to max_batch_size.
    Returns list of batches (each batch is a list of patches).
    """
    shuffled = patches.copy()
    rng.shuffle(shuffled)

    batches = []
    remaining = list(shuffled)

    while remaining:
        batch = []
        still_remaining = []

        for p in remaining:
            # Check this patch doesn't overlap anything already in batch
            if not any(boxes_overlap(p["patch_box"], b["patch_box"]) for b in batch):
                batch.append(p)
                if len(batch) >= max_batch_size:
                    # Put the rest back
                    still_remaining.extend(remaining[remaining.index(p)+1:])
                    break
            else:
                still_remaining.append(p)

        batches.append(batch)
        remaining = still_remaining

    return batches

# ── Load pipeline ─────────────────────────────────────────────────────────────

print(f"\nLoading FluxInpaintPipeline (FLUX.1-dev)...")
pipe = FluxInpaintPipeline.from_pretrained(
    FLUX_MODEL,
    torch_dtype=torch.bfloat16,
).to("cuda:0")
print("Pipeline loaded\n")

# ── Write-back helper ─────────────────────────────────────────────────────────

def write_back_batch(current_image, results, batch_patches):
    new_image = current_image.copy()
    for result, p in zip(results, batch_patches):
        px1, py1, px2, py2 = p["patch_box"]
        actual_w, actual_h = px2-px1, py2-py1

        if result.size != (actual_w, actual_h):
            result = result.resize((actual_w, actual_h), Image.LANCZOS)
            mask_pil = make_mask(
                p["at_left"], p["at_right"], p["at_top"], p["at_bottom"]
            ).resize((actual_w, actual_h), Image.LANCZOS)
        else:
            mask_pil = make_mask(
                p["at_left"], p["at_right"], p["at_top"], p["at_bottom"]
            )

        orig_arr   = np.array(current_image.crop(p["patch_box"])).astype(float)
        result_arr = np.array(result).astype(float)
        alpha      = np.array(mask_pil).astype(float)[:,:,np.newaxis] / 255.0
        blended    = (orig_arr*(1-alpha) + result_arr*alpha).astype(np.uint8)
        new_image.paste(Image.fromarray(blended), (px1, py1))
    return new_image

# ── Visualization ─────────────────────────────────────────────────────────────

VIZ_W = 800
viz_scale = VIZ_W / max(width, height)
viz_w = int(width  * viz_scale)
viz_h = int(height * viz_scale)

ACTIVE_RGBA = (255, 50,  50,  200)
DONE_RGBA   = (50,  200, 50,  60)
PENDING_RGBA= (150, 150, 255, 80)

def make_viz_frame(current_img, all_patches, done_indices, active_indices,
                   iter_num, batch_num, total_batches, shift_name):
    thumb = current_img.resize((viz_w, viz_h), Image.LANCZOS).convert("RGBA")
    overlay = Image.new("RGBA", (viz_w, viz_h), (0,0,0,0))
    draw = ImageDraw.Draw(overlay)

    active_set = set(active_indices)
    done_set   = set(done_indices)

    for i, p in enumerate(all_patches):
        ix1, iy1, ix2, iy2 = p["inner_box"]
        vix1 = int(ix1*viz_scale); viy1 = int(iy1*viz_scale)
        vix2 = int(ix2*viz_scale); viy2 = int(iy2*viz_scale)

        if i in active_set:
            draw.rectangle([vix1,viy1,vix2,viy2],
                           fill=ACTIVE_RGBA, outline=(255,50,50,255), width=2)
        elif i in done_set:
            draw.rectangle([vix1,viy1,vix2,viy2], fill=DONE_RGBA)
        else:
            draw.rectangle([vix1,viy1,vix2,viy2],
                           outline=(150,150,255,160), width=1)

    viz = Image.alpha_composite(thumb, overlay).convert("RGB")
    bar = Image.new("RGB", (viz_w, 38), (20,20,20))
    bd  = ImageDraw.Draw(bar)
    bd.text((6, 4),  f"Iter {iter_num}/{NUM_ITERS}  Grid: {shift_name}  "
                     f"Batch {batch_num}/{total_batches}", fill=(255,220,100))
    bd.text((6, 20), f"Red=active  Green=done  Blue=pending  "
                     f"strength={STRENGTH}  batch_size≤{MAX_BATCH_SIZE}", fill=(160,160,160))
    frame = Image.new("RGB", (viz_w, viz_h+38))
    frame.paste(bar, (0,0))
    frame.paste(viz, (0,38))
    return frame

# ── Main loop ─────────────────────────────────────────────────────────────────

print(f"{'='*60}")
print(f"Full coverage v5 — random mini-batch sampling")
print(f"  Iterations:    {NUM_ITERS}")
print(f"  Max batch:     {MAX_BATCH_SIZE}")
print(f"  Strength:      {STRENGTH}")
print(f"  Grid per iter: cycles through {len(PASS_SHIFTS)} offsets")
print(f"  Edge-aware:    yes (corners/edges extend to boundary)")
print(f"  Bias:          none (random shuffle each iteration)")
print(f"{'='*60}\n")

viz_frames = []
current_image.save(OUTPUT_DIR / "iter_000_original.png")

# Add initial viz frame
patches_init = get_patches(width, height, OUTER_SIZE, inner_size, offset, 0, 0)
viz_frames.append(make_viz_frame(current_image, patches_init, [], [], 0, 0, 0, "start"))

for iter_idx in range(NUM_ITERS):
    iter_num = iter_idx + 1
    sx, sy, shift_name = PASS_SHIFTS[iter_idx % 4]
    rng = random.Random(BASE_SEED + iter_idx)

    patches = get_patches(width, height, OUTER_SIZE, inner_size, offset, sx, sy)
    n_patches = len(patches)

    # Build random batches for this iteration
    batches = build_random_batches(patches, MAX_BATCH_SIZE, rng)
    n_batches = len(batches)

    print(f"\n{'─'*50}")
    print(f"Iter {iter_num}/{NUM_ITERS}: grid={shift_name}  "
          f"{n_patches} patches → {n_batches} mini-batches")
    print(f"  Batch sizes: {[len(b) for b in batches]}")
    print(f"{'─'*50}")

    iter_dir = OUTPUT_DIR / f"iter{iter_num:02d}_{shift_name}"
    iter_dir.mkdir(exist_ok=True)

    done_indices = []
    t_iter = time.time()

    for batch_idx, batch in enumerate(batches):
        seed = BASE_SEED + iter_idx * 1000 + batch_idx
        generator = torch.Generator(device="cuda:0").manual_seed(seed)
        t0 = time.time()

        # Get patch indices for visualization
        patch_id_map = {id(p): i for i, p in enumerate(patches)}
        active_indices = [patch_id_map[id(p)] for p in batch]

        # Viz frame — show this batch as active
        viz_frames.append(make_viz_frame(
            current_image, patches, done_indices, active_indices,
            iter_num, batch_idx+1, n_batches, shift_name
        ))

        # Extract patches from current image
        patch_images = []
        for p in batch:
            patch = current_image.crop(p["patch_box"])
            if patch.size != (OUTER_SIZE, OUTER_SIZE):
                patch = patch.resize((OUTER_SIZE, OUTER_SIZE), Image.LANCZOS)
            patch_images.append(patch)

        masks = [make_mask(p["at_left"], p["at_right"],
                           p["at_top"],  p["at_bottom"]) for p in batch]

        try:
            results = pipe(
                prompt=[PROMPT] * len(batch),
                negative_prompt=[NEGATIVE_PROMPT] * len(batch),
                image=patch_images,
                mask_image=masks,
                height=OUTER_SIZE,
                width=OUTER_SIZE,
                strength=STRENGTH,
                num_inference_steps=NUM_STEPS,
                guidance_scale=GUIDANCE,
                generator=generator,
            ).images

            current_image = write_back_batch(current_image, results, batch)
            done_indices.extend(active_indices)

        except Exception as e:
            print(f"  Batch {batch_idx+1} failed: {e}")
            import traceback; traceback.print_exc()
            done_indices.extend(active_indices)  # mark as done anyway
            continue

        elapsed = time.time() - t0
        print(f"  Batch {batch_idx+1:3d}/{n_batches}  "
              f"size={len(batch):2d}  {elapsed:.1f}s")

    # Save after each iteration
    iter_path = OUTPUT_DIR / f"iter{iter_num:02d}_{shift_name}_result.png"
    current_image.save(iter_path)
    print(f"  Iteration {iter_num} done in {time.time()-t_iter:.1f}s → {iter_path.name}")

    # End-of-iteration viz frame
    viz_frames.append(make_viz_frame(
        current_image, patches, list(range(n_patches)), [],
        iter_num, n_batches, n_batches, shift_name
    ))

# ── Final outputs ─────────────────────────────────────────────────────────────

final_path = OUTPUT_DIR / "final_result.png"
current_image.save(final_path)
print(f"\nFinal → {final_path}")

comp_w = 1024
comp_h = int(height * comp_w / width)
comp = Image.new("RGB", (comp_w*2+10, comp_h+30), (25,25,25))
comp.paste(image.resize((comp_w, comp_h)), (0, 30))
comp.paste(current_image.resize((comp_w, comp_h)), (comp_w+10, 30))
draw = ImageDraw.Draw(comp)
draw.text((comp_w//2-40, 8), "ORIGINAL", fill=(255,255,255))
draw.text((comp_w+10+comp_w//2-50, 8), f"AFTER {NUM_ITERS} ITERS", fill=(255,255,255))
comp.save(OUTPUT_DIR / "before_after.png")

# GIF
print(f"\nGenerating GIF ({len(viz_frames)} frames)...")
gif_path = OUTPUT_DIR / "patch_progression.gif"
viz_frames[0].save(
    gif_path, save_all=True, append_images=viz_frames[1:],
    duration=GIF_FRAME_DURATION, loop=0, optimize=True,
)
print(f"GIF → {gif_path}")

print(f"\n── Done ──────────────────────────────────────────────────────")
print(f"  final_result.png       ← final image")
print(f"  before_after.png       ← comparison")
print(f"  patch_progression.gif  ← animated viz (random batch order)")
print(f"  iter*/                 ← per-iteration saves")
print(f"\nTuning:")
print(f"  NUM_ITERS=8       → more refinement passes")
print(f"  MAX_BATCH_SIZE=8  → smaller batches, more randomness")
print(f"  STRENGTH=0.15     → even more subtle refinement")