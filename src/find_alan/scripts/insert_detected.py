"""CLI: detect a person in the scene and replace them with a reference figure.

Uses YOLOv8 to find people, selects one according to a strategy, then
inpaints the reference figure into that region using FLUX.2-Klein.
This gives the model a correctly-scaled insertion region so the output
figure matches the crowd's perspective and size.

Example:
  find-alan-insert-detected --scene crowd.png --figure person.png \\
      --output result.png --strategy largest
"""

from __future__ import annotations

import argparse
import random

from PIL import Image, ImageDraw, ImageFilter

from find_alan.detect import (
    detect_people,
    detect_people_with_masks,
    dilate_mask,
    load_detector,
    pad_bbox,
    pick_target,
)
from find_alan.insert import load_pipeline, run_insertion
from find_alan.mask import bbox_to_mask

_DETECTION_PROMPT = (
    "Replace the person in this image with the person" " from the reference image."
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="find-alan-insert-detected",
        description=(
            "Detect a person in the scene with YOLOv8 and replace them"
            " with the reference figure using FLUX.2-Klein.\n\n"
            "The detected region is cropped and passed to FLUX.2-Klein as"
            " the full scene, so the model fills it at the right scale.\n\n"
            "Example:\n"
            "  find-alan-insert-detected --scene crowd.png"
            " --figure alan.png \\\n"
            "    --output result.png --strategy largest"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument(
        "--scene",
        required=True,
        metavar="PATH",
        help="Crowd/background scene image.",
    )
    p.add_argument(
        "--figure",
        required=True,
        metavar="PATH",
        help="Reference figure to insert.",
    )
    p.add_argument(
        "--output",
        required=True,
        metavar="PATH",
        help="Where to save the result.",
    )

    p.add_argument(
        "--strategy",
        default="random",
        choices=["random", "largest", "smallest", "center"],
        help=(
            "Which detected person to replace. "
            "'random' picks any; 'largest' picks the biggest; "
            "'smallest' picks the least prominent; "
            "'center' picks the one nearest the image centre. "
            "Default: random."
        ),
    )
    p.add_argument(
        "--padding",
        type=float,
        default=0.2,
        metavar="FLOAT",
        help=(
            "Fraction of bbox size to add around the crop on each side."
            " Larger values give FLUX.2-Klein more scene context."
            " Default: 0.2."
        ),
    )
    p.add_argument(
        "--feather",
        type=int,
        default=30,
        metavar="PIXELS",
        help=(
            "Radius (px) of the Gaussian blur applied to the composite"
            " mask edges. Softens the boundary between the generated"
            " region and the original scene. Default: 30."
        ),
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
        default="yolov8s-worldv2",
        metavar="NAME",
        help=(
            "YOLOv8 / YOLO-World model variant. Use a '-worldv2' variant"
            " (e.g. yolov8s-worldv2) for open-vocabulary detection with"
            " --detection-classes; use a '-seg' variant (e.g. yolov8n-seg)"
            " to enable segmentation masks. Default: yolov8s-worldv2."
        ),
    )
    p.add_argument(
        "--detection-classes",
        nargs="+",
        default=["cartoon person"],
        metavar="CLASS",
        help=(
            "One or more text class prompts for open-vocabulary detection"
            " (YOLO-World). Ignored for standard YOLOv8 models."
            " Default: 'cartoon person'."
        ),
    )
    p.add_argument(
        "--segmentation",
        action="store_true",
        help=(
            "Use person-shaped segmentation mask for compositing instead"
            " of a feathered rectangle. Requires a '-seg' YOLO model."
        ),
    )
    p.add_argument(
        "--prompt",
        default=_DETECTION_PROMPT,
        metavar="TEXT",
        help="Text prompt to guide the insertion.",
    )
    p.add_argument(
        "--strength",
        type=float,
        default=0.99,
        metavar="FLOAT",
        help="How much the crop can change (0–1). Default: 0.99.",
    )
    p.add_argument(
        "--steps",
        type=int,
        default=50,
        metavar="INT",
        help="Inference steps. Default: 50.",
    )
    p.add_argument(
        "--guidance-scale",
        type=float,
        default=10.0,
        metavar="FLOAT",
        help="CFG scale (ignored on distilled models). Default: 10.0.",
    )
    p.add_argument(
        "--reference-repeats",
        type=int,
        default=1,
        metavar="INT",
        help=(
            "How many times to repeat the reference image in the conditioning"
            " sequence. More repeats = stronger reference fidelity."
            " Try 2–4; beyond ~6 quality degrades. Default: 1."
        ),
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="INT",
        help="Reproducibility seed.",
    )
    p.add_argument(
        "--device",
        default=None,
        metavar="DEVICE",
        help="cuda | mps | cpu  (auto-detected if omitted).",
    )
    p.add_argument(
        "--save-mask",
        metavar="PATH",
        default=None,
        help="Optionally save the bbox mask image for inspection.",
    )
    p.add_argument(
        "--test",
        action="store_true",
        help=(
            "Draw a red outline around the inpainted crop region in the"
            " final image for quick visual verification."
        ),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    scene = Image.open(args.scene).convert("RGB")
    figure = Image.open(args.figure).convert("RGB")

    print(
        f"Detecting people in scene" f" (YOLO {args.yolo_model}, conf≥{args.conf})..."
    )
    detector = load_detector(args.yolo_model, classes=args.detection_classes)

    seg_mask: Image.Image | None = None
    if args.segmentation:
        detections = detect_people_with_masks(detector, scene, conf_threshold=args.conf)
        if not detections:
            print(
                "No people detected. Try lowering --conf or check the" " scene image."
            )
            return 1
        bboxes = [bbox for bbox, _ in detections]
        seg_masks = {bbox: mask for bbox, mask in detections}
    else:
        bboxes = detect_people(detector, scene, conf_threshold=args.conf)
        seg_masks = {}
        if not bboxes:
            print(
                "No people detected. Try lowering --conf or check the" " scene image."
            )
            return 1

    print(
        f"Found {len(bboxes)} person(s)."
        f" Picking target with strategy='{args.strategy}'..."
    )

    rng = random.Random(args.seed) if args.seed is not None else None
    target_bbox = pick_target(
        bboxes,
        strategy=args.strategy,
        scene_size=scene.size,
        rng=rng,
    )
    x, y, w, h = target_bbox
    print(f"Target bbox (x={x}, y={y}, w={w}, h={h})")

    if args.segmentation and target_bbox in seg_masks:
        seg_mask = seg_masks[target_bbox]

    padded_bbox = pad_bbox(target_bbox, scene.size, padding=args.padding)
    px, py, pw, ph = padded_bbox
    print(f"Padded bbox (x={px}, y={py}, w={pw}, h={ph})")

    if args.save_mask:
        from pathlib import Path

        save = (
            seg_mask if seg_mask is not None else bbox_to_mask(scene.size, padded_bbox)
        )
        Path(args.save_mask).parent.mkdir(parents=True, exist_ok=True)
        save.save(args.save_mask)
        print(f"Mask saved → {args.save_mask}")

    # Crop the scene to the padded detection region and run FLUX.2-Klein on
    # that crop.  Because the crop IS the entire scene the model sees, the
    # reference figure must appear somewhere within it rather than being
    # placed at an arbitrary location in the full image.  We then paste the
    # result back into the original scene at the detection coordinates.
    crop_box = (px, py, px + pw, py + ph)
    scene_crop = scene.crop(crop_box)
    print(f"Cropped scene to detection region: {scene_crop.size}")

    print("Loading pipeline (FLUX.2-Klein, ~13 GB, downloaded on first run)...")
    pipe = load_pipeline(device=args.device)

    print("Running FLUX.2-Klein on detection crop...")
    result_crop = run_insertion(
        pipe=pipe,
        scene=scene_crop,
        figure=figure,
        prompt=args.prompt,
        strength=args.strength,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        reference_repeats=args.reference_repeats,
    )

    result_sized = result_crop.resize((pw, ph), Image.LANCZOS)

    if seg_mask is not None:
        # Person-shaped composite: dilate the segmentation mask slightly so
        # the very edge of the figure doesn't get hard-clipped, then crop it
        # to the padded bbox region for pasting.
        dilation = max(4, min(pw, ph) // 20)
        dilated = dilate_mask(seg_mask, radius=dilation)
        paste_mask = dilated.crop(crop_box)
    else:
        # Feathered rectangle: scale feather relative to crop size so it
        # doesn't swamp a small crop.
        feather = min(args.feather, max(4, min(pw, ph) // 8))
        paste_mask = Image.new("L", (pw, ph), 0)
        inner = [feather, feather, pw - feather, ph - feather]
        ImageDraw.Draw(paste_mask).rectangle(inner, fill=255)
        paste_mask = paste_mask.filter(ImageFilter.GaussianBlur(radius=feather))

    result = scene.copy()
    result.paste(result_sized, (px, py), mask=paste_mask)

    if args.test:
        # Visual debug aid: show where the inpainted crop was pasted.
        line_width = max(2, min(scene.size) // 300)
        ImageDraw.Draw(result).rectangle(
            (px, py, px + pw - 1, py + ph - 1),
            outline=(255, 0, 0),
            width=line_width,
        )
        print(
            "[test] Drew red outline around inpainted region "
            f"(x={px}, y={py}, w={pw}, h={ph})"
        )

    from pathlib import Path

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path)
    print(f"Saved → {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
