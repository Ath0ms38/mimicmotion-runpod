#!/bin/bash
set -e

echo "============================================"
echo "  StableAnimator - RunPod Container"
echo "============================================"
echo ""

# Print GPU info
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
    echo "WARNING: nvidia-smi not found, no GPU detected"
fi
echo ""

# Download model weights (idempotent, skips existing)
echo "Checking model weights..."
python /app/download_models.py
echo ""

# Look for input files on persistent volume
IMAGE="${SA_IMAGE:-/workspace/input/image.png}"
VIDEO="${SA_VIDEO:-/workspace/input/video.mp4}"
OUTPUT="${SA_OUTPUT:-/workspace/output.mp4}"

if [ ! -f "$IMAGE" ]; then
    echo "ERROR: Reference image not found: $IMAGE"
    echo ""
    echo "Upload your files to /workspace/input/ and set env vars:"
    echo "  SA_IMAGE=/workspace/input/image.png"
    echo "  SA_VIDEO=/workspace/input/video.mp4"
    echo "  SA_OUTPUT=/workspace/output.mp4"
    echo ""
    echo "Or run manually:"
    echo "  python /app/run.py --image <path> --video <path>"
    exit 1
fi

if [ ! -f "$VIDEO" ]; then
    echo "ERROR: Motion video not found: $VIDEO"
    echo ""
    echo "Upload your files to /workspace/input/ and set env vars:"
    echo "  SA_IMAGE=/workspace/input/image.png"
    echo "  SA_VIDEO=/workspace/input/video.mp4"
    echo "  SA_OUTPUT=/workspace/output.mp4"
    echo ""
    echo "Or run manually:"
    echo "  python /app/run.py --image <path> --video <path>"
    exit 1
fi

echo "============================================"
echo "  Running StableAnimator inference"
echo "============================================"
echo "  Image:  $IMAGE"
echo "  Video:  $VIDEO"
echo "  Output: $OUTPUT"
echo ""

# Pass through any extra args via SA_ARGS env var
# e.g. SA_ARGS="--width 576 --height 1024 --steps 30"
exec python /app/run.py \
    --image "$IMAGE" \
    --video "$VIDEO" \
    --output "$OUTPUT" \
    ${SA_ARGS:-}
