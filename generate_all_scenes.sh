#!/usr/bin/env bash
# Generate one image per scene file in assets/prompts/scenes/, spreading jobs
# across two GPUs.  A new job only starts on a GPU once its previous job is done.

set -euo pipefail

SCENES_DIR="assets/prompts/scenes"
STYLE_FILE="assets/prompts/styles/1-waldo-cartoon.txt"
NEGATIVE_FILE="assets/prompts/NEGATIVE.txt"
OUTPUT_DIR="${1:-outputs/scenes}"

mkdir -p "$OUTPUT_DIR"

declare -A gpu_pid
gpu_pid[0]=0
gpu_pid[1]=0

slot=0

for scene_file in "$SCENES_DIR"/*.txt; do
    scene_name=$(basename "$scene_file" .txt)

    # Wait for the previous job on this GPU slot before reusing it
    if [ "${gpu_pid[$slot]}" -ne 0 ]; then
        echo "Waiting for GPU $slot to finish before starting $scene_name …"
        wait "${gpu_pid[$slot]}"
    fi

    echo "[GPU $slot] $scene_name → $OUTPUT_DIR/${scene_name}.png"
    CUDA_VISIBLE_DEVICES=$slot uv run find-alan-generate \
        --scene-file  "$scene_file" \
        --style-file  "$STYLE_FILE" \
        --negative-file "$NEGATIVE_FILE" \
        --height 1080 \
        --width 1920 \
        --out "$OUTPUT_DIR/flux2-dev_style1_${scene_name}.png" &

    gpu_pid[$slot]=$!
    slot=$(( (slot + 1) % 2 ))
done

# Wait for whichever GPU is still running a final job
wait "${gpu_pid[0]}" 2>/dev/null || true
wait "${gpu_pid[1]}" 2>/dev/null || true

echo "Done. Images in $OUTPUT_DIR/"
