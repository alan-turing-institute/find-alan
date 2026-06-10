"""CLI: detect a person in the scene and replace them with a reference figure.

Uses YOLOv8 to find people, selects one according to a strategy, then
inpaints the reference figure into that region using FLUX.1-Redux + Fill.
This gives the model a correctly-scaled insertion region so the output
figure matches the crowd's perspective and size.

Example:
  find-alan-insert-detected --scene crowd.png --figure person.png \\
      --output result.png --strategy largest
"""

from __future__ import annotations

import argparse
import random

from PIL import Image

from find_alan.detect import detect_people, load_detector, pad_bbox, pick_target
from find_alan.inpaint import DEFAULT_PROMPT, load_pipeline, run_inpainting
from find_alan.mask import bbox_to_mask

_DETECTION_PROMPT = (
    "Replace the person in the masked area with the reference figure."
    " Match the scale, pose, perspective, and lighting of the surrounding"
    " crowd exactly. Preserve the full body of the reference figure."
    " Blend the edges naturally with the background."
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="find-alan-insert-detected",
        description=(
            "Detect a person in the scene with YOLOv8 and replace them with"
            " the reference figure using FLUX.1-Redux + FLUX.1-Fill.\n\n"
            "Scale is correct by construction: the inpaint region is sized to"
            " a real crowd member, so the model fills it at the right size.\n\n"
            "Example:\n"
            "  find-alan-insert-detected --scene crowd.png --figure alan.png"
            " \\\n"
            "    --output result.png --strategy largest"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument("--scene", required=True, metavar="PATH", help="Crowd/background scene image.")
    p.add_argument("--figure", required=True, metavar="PATH", help="Reference figure to insert.")
    p.add_argument("--output", required=True, metavar="PATH", help="Where to save the result.")

    p.add_argument(
        "--strategy",
        default="random",
        choices=["random", "largest", "smallest", "center"],
        help=(
            "Which detected person to replace. "
            "'random' picks any; 'largest' picks the biggest (most context for blending); "
            "'smallest' picks the least prominent; 'center' picks the one nearest the image centre. "
            "Default: random."
        ),
    )
    p.add_argument(
        "--padding",
        type=float,
        default=0.15,
        metavar="FLOAT",
        help="Fraction of bbox size to expand the mask for edge blending. Default: 0.15.",
    )
    p.add_argument(
        "--conf",
        type=float,
        default=0.3,
        metavar="FLOAT",
        help="YOLO confidence threshold for person detection. Default: 0.3.",
    )
    p.add_argument(
        "--yolo-model",
        default="yolov8n",
        metavar="NAME",
        help="YOLOv8 model variant (yolov8n/s/m/l/x). Default: yolov8n.",
    )
    p.add_argument(
        "--prompt",
        default=_DETECTION_PROMPT,
        metavar="TEXT",
        help="Text appended to Redux visual tokens to guide blending.",
    )
    p.add_argument("--steps", type=int, default=50, metavar="INT", help="Inference steps. Default: 50.")
    p.add_argument(
        "--guidance-scale",
        type=float,
        default=30.0,
        metavar="FLOAT",
        help="CFG scale. Default: 30.0.",
    )
    p.add_argument("--seed", type=int, default=None, metavar="INT", help="Reproducibility seed.")
    p.add_argument(
        "--device", default=None, metavar="DEVICE", help="cuda | mps | cpu  (auto-detected if omitted)."
    )
    p.add_argument(
        "--save-mask",
        metavar="PATH",
        default=None,
        help="Optionally save the generated mask image for inspection.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    scene = Image.open(args.scene).convert("RGB")
    figure = Image.open(args.figure).convert("RGB")

    print(f"Detecting people in scene (YOLO {args.yolo_model}, conf≥{args.conf})...")
    detector = load_detector(args.yolo_model)
    bboxes = detect_people(detector, scene, conf_threshold=args.conf)

    if not bboxes:
        print("No people detected. Try lowering --conf or check the scene image.")
        return 1

    print(f"Found {len(bboxes)} person(s). Picking target with strategy='{args.strategy}'...")

    rng = random.Random(args.seed) if args.seed is not None else None
    target_bbox = pick_target(
        bboxes,
        strategy=args.strategy,
        scene_size=scene.size,
        rng=rng,
    )
    x, y, w, h = target_bbox
    print(f"Target bbox (x={x}, y={y}, w={w}, h={h})")

    padded_bbox = pad_bbox(target_bbox, scene.size, padding=args.padding)
    px, py, pw, ph = padded_bbox
    print(f"Padded bbox (x={px}, y={py}, w={pw}, h={ph})")

    mask = bbox_to_mask(scene.size, padded_bbox)

    if args.save_mask:
        from pathlib import Path
        Path(args.save_mask).parent.mkdir(parents=True, exist_ok=True)
        mask.save(args.save_mask)
        print(f"Mask saved → {args.save_mask}")

    print("Loading pipelines (FLUX.1-Redux + FLUX.1-Fill, ~25 GB, downloaded on first run)...")
    pipelines = load_pipeline(device=args.device)

    print("Running inpainting...")
    result = run_inpainting(
        pipelines=pipelines,
        scene=scene,
        figure=figure,
        mask=mask,
        prompt=args.prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
    )

    from pathlib import Path
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path)
    print(f"Saved → {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
