#!/bin/bash
set -e

echo "============================================"
echo "  MimicMotion - RunPod Container"
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

# Create symlinks from MimicMotion repo to weight locations
mkdir -p /app/MimicMotion/models
ln -sfn /workspace/models/DWPose /app/MimicMotion/models/DWPose
ln -sfn /workspace/models/MimicMotion_1-1.pth /app/MimicMotion/models/MimicMotion_1-1.pth
echo "Model symlinks created."

echo ""
echo "============================================"
echo "  Starting Gradio UI on port 7860"
echo "============================================"
echo ""
echo "CLI usage:"
echo "  python /app/run.py --image /workspace/image.jpg --video /workspace/video.mp4"
echo ""

# Launch Gradio UI
exec python /app/app.py
