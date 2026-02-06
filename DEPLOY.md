# Deploying MimicMotion on RunPod

## Prerequisites

- A [RunPod](https://www.runpod.io/) account with credits
- A [HuggingFace](https://huggingface.co/) account with:
  - Access granted to [stabilityai/stable-video-diffusion-img2vid-xt-1-1](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1) (click "Agree and access repository")
  - An access token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
- The Docker image pushed to GitHub Container Registry (see [Building](#building) below)

## GPU Recommendation

| GPU | VRAM | Price/hr | Notes |
|-----|------|----------|-------|
| **RTX 5090** | 32 GB | ~$1.04 | Best performance. Blackwell architecture, ~30-40% faster than 4090. |
| **RTX 4090** | 24 GB | ~$0.69-0.77 | Best value. ~20 min for 72 frames at 576x1024. |
| RTX A5000 | 24 GB | ~$0.49 | Budget option, slightly slower. |

MimicMotion requires **16 GB+ VRAM** minimum. 24-32 GB recommended for comfortable headroom.

## Building

### Build the Docker image

```bash
cd /path/to/dance
docker build -t ghcr.io/<your-github-user>/mimicmotion-runpod:latest .
```

Build takes ~10-15 minutes (mostly downloading CUDA base images and PyTorch wheels).

### Push to GitHub Container Registry

```bash
# Create a GitHub personal access token with `write:packages` scope
# at https://github.com/settings/tokens

echo $GITHUB_TOKEN | docker login ghcr.io -u <your-github-user> --password-stdin
docker push ghcr.io/<your-github-user>/mimicmotion-runpod:latest
```

Make sure the package visibility is set to **Public** on GitHub (Settings > Packages) so RunPod can pull it without authentication.

## Deploying on RunPod

### Step 1: Create a Pod Template

1. Go to [RunPod Console](https://www.runpod.io/console/pods)
2. Click **Templates** in the sidebar, then **New Template**
3. Fill in:

| Field | Value |
|-------|-------|
| Template Name | `MimicMotion` |
| Container Image | `ghcr.io/<your-github-user>/mimicmotion-runpod:latest` |
| Container Disk | `20 GB` |
| Volume Disk | `20 GB` |
| Volume Mount Path | `/workspace` |
| Expose HTTP Ports | `7860` |
| Environment Variables | `HF_TOKEN` = `hf_your_token_here` |

> **Important:** The volume at `/workspace` persists model weights (~9 GB) across pod restarts. Without it, models re-download every time the pod starts.

### Step 2: Launch a Pod

1. Go to **Pods** > **Deploy**
2. Select your GPU (RTX 4090 or RTX 5090 recommended)
3. Choose your **MimicMotion** template
4. Click **Deploy**

### Step 3: Wait for Startup

The first boot takes **3-8 minutes** depending on network speed:

```
============================================
  MimicMotion - RunPod Container
============================================

NVIDIA GeForce RTX 4090, 24564 MiB, 550.xx.xx

Checking model weights...
[1/3] DWPose models...          (~350 MB)
[2/3] MimicMotion checkpoint... (~3.0 GB)
[3/3] SVD model...              (~5.0 GB)

All models ready!

Starting Gradio UI on port 7860
```

Subsequent restarts skip the downloads since weights are on the persistent volume.

### Step 4: Access the UI

1. In the RunPod pod page, click **Connect**
2. Click the **HTTP Port 7860** link (or find it under "Connect via HTTP")
3. The Gradio UI opens in your browser

### Step 5: Generate

1. **Upload a reference image** — the person/character you want to animate
2. **Upload a motion video** — the dance or movement to transfer
3. **Set the time range** — use the Start/End Time sliders to select which portion of the video to process
4. **Auto-chunk** is on by default — long videos are automatically split into ~4.8s chunks, processed one by one, and assembled with crossfade blending
5. (Optional) Adjust settings in the **Settings** accordion
6. Click **Generate**
7. Watch the progress bar — shows which chunk is being processed and ETA per chunk
8. View the output video and inspect individual frames in the **Frame Preview** gallery
9. Download the result

**Time estimates (RTX 4090):** ~20 min per chunk. A 30-second video = ~6 chunks = ~2 hours.

## Using the CLI

You can also SSH into the pod and run inference from the command line:

```bash
# Process first 30 seconds of a video
python /app/run.py \
    --image /workspace/your_image.jpg \
    --video /workspace/your_video.mp4 \
    --output /workspace/output.mp4 \
    --start_time 0 --end_time 30

# Process just one chunk (no auto-assembly)
python /app/run.py \
    --image /workspace/your_image.jpg \
    --video /workspace/your_video.mp4 \
    --no_auto_chunk
```

## Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `num_frames` | 72 | 16-72 | Frames per chunk (MimicMotion limit). |
| `resolution` | 576 | 256-576 | Output height in pixels. Width auto-set to 16:9 ratio. |
| `steps` | 25 | 10-50 | Denoising steps. More = higher quality but slower. |
| `guidance_scale` | 2.0 | 1.0-5.0 | How closely to follow the pose. Higher = more rigid. |
| `noise_aug_strength` | 0.0 | 0.0-1.0 | Noise augmentation. 0 = cleanest output. |
| `sample_stride` | 2 | 1-4 | Frame sampling from input video. 2 = use every other frame. |
| `fps` | 15 | 7-30 | Output video framerate. |
| `seed` | 42 | any int | Random seed for reproducibility. |
| `frames_overlap` | 6 | 2-12 | Overlap between tiles for smooth transitions. |
| `start_time` | 0 | 0-duration | Start time in seconds for video range. |
| `end_time` | None | 0-duration | End time in seconds (None = full video). |
| `--no_auto_chunk` | off | flag | Disable auto-chunking, only process first chunk. |

## Auto-Chunking

MimicMotion can only generate 72 frames (~4.8s at 15fps) per pass. For longer videos, the system automatically:

1. **Detects** the video duration and calculates how many chunks are needed
2. **Trims** the video into overlapping segments using ffmpeg
3. **Runs** MimicMotion on each chunk (pipeline is loaded once and reused)
4. **Blends** chunks with 8-frame linear crossfade for smooth transitions
5. **Assembles** the final video

Example: a 30-second dance video at stride=2 produces ~6 chunks, each taking ~20 min on RTX 4090 (~2 hours total).

## Tips

- **Image aspect ratio**: The input image is automatically center-cropped to 9:16 and resized to 576x1024. Portrait photos work best.
- **Time range**: Use `--start_time` and `--end_time` to process specific parts of a long video instead of the whole thing.
- **VRAM usage**: If you get OOM errors, try reducing `resolution` to 384 or `num_frames` to 48.
- **Speed**: Reducing `steps` from 25 to 15 cuts inference time by ~40% with a small quality trade-off.
- **Persistent storage**: Always use a volume mount at `/workspace` to avoid re-downloading 9 GB of models on every restart.

## Troubleshooting

### "Cannot access gated repo" error
You need to accept the SVD model terms and provide your HuggingFace token:
1. Visit [stabilityai/stable-video-diffusion-img2vid-xt-1-1](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1) and click "Agree"
2. Set `HF_TOKEN` in your pod's environment variables

### OOM (Out of Memory)
Your GPU doesn't have enough VRAM. Try:
- Lower `resolution` (384 instead of 576)
- Fewer `num_frames` (48 instead of 72)
- Use a GPU with more VRAM (RTX 4090 or 5090)

### Slow first startup
Model weights (~9 GB) download on first boot. Use a persistent volume at `/workspace` so this only happens once.

### Gradio UI not loading
- Check that port 7860 is exposed in your template
- Wait for the "Starting Gradio UI" message in the pod logs
- Try the direct URL: `https://<pod-id>-7860.proxy.runpod.net`
