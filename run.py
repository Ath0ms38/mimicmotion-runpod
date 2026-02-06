"""
run.py - MimicMotion motion transfer core inference logic.

Supports full-length video processing by automatically splitting the input
video into chunks, running MimicMotion on each chunk, and assembling the
results with crossfade blending.

Usage (CLI):
    python run.py --image /path/to/image.jpg --video /path/to/video.mp4 \
                  --output /workspace/output.mp4

Can also be imported and called from app.py (Gradio UI).
"""

import argparse
import math
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from torchvision.datasets.folder import pil_loader
from torchvision.transforms.functional import pil_to_tensor, resize, center_crop, to_pil_image

# Add MimicMotion to path
MIMICMOTION_DIR = os.environ.get("MIMICMOTION_DIR", "/app/MimicMotion")
if MIMICMOTION_DIR not in sys.path:
    sys.path.insert(0, MIMICMOTION_DIR)

# Apply geglu patch before any MimicMotion imports (as in their inference.py)
from mimicmotion.utils.geglu_patch import patch_geglu_inplace
patch_geglu_inplace()

from mimicmotion_patch import ProgressTracker, apply_patches


MODELS_DIR = Path(os.environ.get("MODELS_DIR", "/workspace/models"))
DEFAULT_CONFIG = {
    "base_model_path": str(MODELS_DIR / "stable-video-diffusion-img2vid-xt-1-1"),
    "ckpt_path": str(MODELS_DIR / "MimicMotion_1-1.pth"),
}

# MimicMotion hard limit per pass
MAX_FRAMES_PER_CHUNK = 72
# Overlap between chunks for crossfade blending
CHUNK_BLEND_FRAMES = 8


def _get_video_info(video_path):
    """Get video duration, fps, and frame count using ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_streams", "-select_streams", "v:0", str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return None

    import json
    data = json.loads(result.stdout)
    stream = data["streams"][0]

    fps_parts = stream.get("r_frame_rate", "25/1").split("/")
    fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else 25.0
    duration = float(stream.get("duration", 0))
    nb_frames = int(stream.get("nb_frames", 0))

    return {"fps": fps, "duration": duration, "nb_frames": nb_frames}


def _trim_video(video_path, start_sec, end_sec, output_path):
    """Trim a video to a specific time range using ffmpeg."""
    cmd = [
        "ffmpeg", "-y", "-ss", str(start_sec), "-to", str(end_sec),
        "-i", str(video_path), "-c", "copy", str(output_path),
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return output_path


def _build_infer_config(
    num_frames=72,
    resolution=576,
    steps=25,
    guidance_scale=2.0,
    noise_aug_strength=0.0,
    sample_stride=2,
    fps=15,
    seed=42,
    frames_overlap=6,
):
    """Build an OmegaConf config matching MimicMotion's expected format."""
    cfg = OmegaConf.create(
        {
            "base_model_path": DEFAULT_CONFIG["base_model_path"],
            "ckpt_path": DEFAULT_CONFIG["ckpt_path"],
            "test_case": [
                {
                    "num_frames": num_frames,
                    "resolution": resolution,
                    "num_inference_steps": steps,
                    "guidance_scale": guidance_scale,
                    "noise_aug_strength": noise_aug_strength,
                    "sample_stride": sample_stride,
                    "fps": fps,
                    "seed": seed,
                    "frames_overlap": frames_overlap,
                }
            ],
        }
    )
    return cfg


def _preprocess(video_path, image_path, resolution, sample_stride, tracker=None):
    """Preprocess inputs — matches MimicMotion's inference.py exactly."""
    from mimicmotion.dwpose.preprocess import get_image_pose, get_video_pose

    ASPECT_RATIO = 9 / 16

    if tracker:
        tracker.log(f"Loading reference image: {image_path}")

    # Load and resize image (exactly as in MimicMotion's inference.py)
    image_pixels = pil_loader(image_path)
    image_pixels = pil_to_tensor(image_pixels)  # (c, h, w)
    h, w = image_pixels.shape[-2:]

    # Compute target h/w according to original aspect ratio
    if h > w:
        w_target, h_target = resolution, int(resolution / ASPECT_RATIO // 64) * 64
    else:
        w_target, h_target = int(resolution / ASPECT_RATIO // 64) * 64, resolution

    h_w_ratio = float(h) / float(w)
    if h_w_ratio < h_target / w_target:
        h_resize, w_resize = h_target, math.ceil(h_target / h_w_ratio)
    else:
        h_resize, w_resize = math.ceil(w_target * h_w_ratio), w_target

    image_pixels = resize(image_pixels, [h_resize, w_resize], antialias=None)
    image_pixels = center_crop(image_pixels, [h_target, w_target])
    image_pixels = image_pixels.permute((1, 2, 0)).numpy()  # back to HWC numpy

    if tracker:
        tracker.log(f"Image resized to {w_target}x{h_target}")
        tracker.log("Extracting poses from video...")

    # Get ref image pose and video poses
    image_pose = get_image_pose(image_pixels)
    video_pose = get_video_pose(video_path, image_pixels, sample_stride=sample_stride)

    if tracker:
        tracker.log(f"Extracted {video_pose.shape[0]} pose frames")

    # Prepend ref image pose to video poses (as in MimicMotion's inference.py)
    pose_pixels = np.concatenate([np.expand_dims(image_pose, 0), video_pose])

    # image_pixels: HWC -> NCHW
    image_pixels_nchw = np.transpose(np.expand_dims(image_pixels, 0), (0, 3, 1, 2))

    # Convert to tensors normalized to [-1, 1]
    pose_tensor = torch.from_numpy(pose_pixels.copy()) / 127.5 - 1
    image_tensor = torch.from_numpy(image_pixels_nchw) / 127.5 - 1

    # Keep a PIL ref image for previews
    ref_image_pil = Image.fromarray(image_pixels)

    return pose_tensor, image_tensor, ref_image_pil


def _run_pipeline(pipeline, image_pixels, pose_pixels, device, task_config, tracker=None):
    """Run the MimicMotion pipeline — matches inference.py's run_pipeline exactly."""
    # Convert image tensor back to list of PIL images (as MimicMotion expects)
    image_pil_list = [
        to_pil_image(img.to(torch.uint8))
        for img in (image_pixels + 1.0) * 127.5
    ]

    generator = torch.Generator(device=device)
    generator.manual_seed(task_config.seed)

    # Clamp tile_size to actual frame count (avoids empty indices in pipeline tiling)
    num_frames = pose_pixels.size(0)
    tile_size = min(task_config.num_frames, num_frames)
    tile_overlap = min(task_config.frames_overlap, tile_size - 1)

    if tracker:
        indices_count = len(
            list(range(0, num_frames - tile_size + 1, tile_size - tile_overlap))
        )
        if indices_count == 0:
            indices_count = 1
        total_iters = task_config.num_inference_steps * indices_count
        tracker.log(
            f"Starting inference: {task_config.num_inference_steps} steps, "
            f"~{indices_count} tiles, ~{total_iters} iterations"
        )

    frames = pipeline(
        image_pil_list,
        image_pose=pose_pixels,
        num_frames=pose_pixels.size(0),
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        height=pose_pixels.shape[-2],
        width=pose_pixels.shape[-1],
        fps=7,
        noise_aug_strength=task_config.noise_aug_strength,
        num_inference_steps=task_config.num_inference_steps,
        generator=generator,
        min_guidance_scale=task_config.guidance_scale,
        max_guidance_scale=task_config.guidance_scale,
        decode_chunk_size=8,
        output_type="pt",
        device=device,
    ).frames.cpu()

    # Convert to uint8, skip first frame (ref image) — exactly as in inference.py
    video_frames = (frames * 255.0).to(torch.uint8)
    for vid_idx in range(video_frames.shape[0]):
        _video_frames = video_frames[vid_idx, 1:]

    if tracker:
        tracker.log(f"Generated {_video_frames.shape[0]} output frames")

    return _video_frames


def _blend_chunks(chunks, blend_frames):
    """Blend overlapping chunks with linear crossfade.

    Args:
        chunks: list of uint8 tensors, each (F, C, H, W)
        blend_frames: number of overlapping frames to crossfade

    Returns:
        Single concatenated uint8 tensor (F, C, H, W)
    """
    if len(chunks) == 1:
        return chunks[0]

    # If any chunk is too small for blending, just concatenate without blending
    min_chunk_len = min(c.shape[0] for c in chunks)
    if blend_frames <= 0 or min_chunk_len <= blend_frames:
        return torch.cat(chunks, dim=0)

    result_parts = []
    for i, chunk in enumerate(chunks):
        if i == 0:
            # First chunk: keep everything except the last blend_frames
            result_parts.append(chunk[:-blend_frames])
        else:
            # Blend overlap region with previous chunk's tail
            prev_tail = chunks[i - 1][-blend_frames:]
            curr_head = chunk[:blend_frames]

            weights = torch.linspace(1.0, 0.0, blend_frames).view(-1, 1, 1, 1)
            blended = (prev_tail.float() * weights + curr_head.float() * (1 - weights)).to(torch.uint8)
            result_parts.append(blended)

            if i == len(chunks) - 1:
                # Last chunk: append everything after the blend region
                result_parts.append(chunk[blend_frames:])
            else:
                # Middle chunk: append middle, hold back tail for next blend
                result_parts.append(chunk[blend_frames:-blend_frames])

    return torch.cat(result_parts, dim=0)


def run_mimicmotion(
    image_path,
    video_path,
    output_path=None,
    num_frames=72,
    resolution=576,
    steps=25,
    guidance_scale=2.0,
    noise_aug_strength=0.0,
    sample_stride=2,
    fps=15,
    seed=42,
    frames_overlap=6,
    start_time=0.0,
    end_time=None,
    auto_chunk=True,
    gradio_progress=None,
):
    """Run MimicMotion motion transfer with auto-chunking for long videos.

    Args:
        image_path: Path to reference image
        video_path: Path to motion source video
        output_path: Path for output video (auto-generated if None)
        start_time: Start time in seconds (default: 0)
        end_time: End time in seconds (default: None = end of video)
        auto_chunk: If True, automatically split long videos into chunks
        gradio_progress: Optional Gradio progress callback

    Returns:
        dict with 'output_path', 'elapsed', 'num_frames', 'resolution',
              'logs', 'preview_frames', 'chunks_processed'
    """
    from mimicmotion.utils.loader import create_pipeline
    from mimicmotion.utils.utils import save_to_mp4

    # Use float16 for inference (as in MimicMotion's inference.py)
    original_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float16)

    start_clock = time.time()

    if output_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"/workspace/output_{ts}.mp4"

    output_path = str(output_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get video info
    video_info = _get_video_info(video_path)
    video_duration = video_info["duration"] if video_info else 0
    video_fps = video_info["fps"] if video_info else 25.0

    if video_duration <= 0:
        raise ValueError(
            f"Could not determine video duration for {video_path}. "
            "The file may be corrupt or in an unsupported format."
        )

    # Clamp time range
    start_time = max(0.0, float(start_time))
    if end_time is None or end_time <= 0:
        end_time = video_duration
    end_time = min(float(end_time), video_duration)
    selected_duration = end_time - start_time

    if selected_duration <= 0:
        raise ValueError(
            f"Invalid time range: start={start_time:.1f}s, end={end_time:.1f}s "
            f"(video is {video_duration:.1f}s)"
        )

    # Calculate how many source frames MimicMotion uses per chunk
    # Each chunk produces num_frames output frames from num_frames*sample_stride source frames
    source_frames_per_chunk = num_frames * sample_stride
    seconds_per_chunk = source_frames_per_chunk / video_fps

    # Determine chunks needed
    # With blending, each chunk after the first overlaps by CHUNK_BLEND_FRAMES output frames
    # which means we need to extend the source overlap too
    blend_source_sec = (CHUNK_BLEND_FRAMES * sample_stride) / video_fps

    if not auto_chunk or selected_duration <= seconds_per_chunk:
        num_chunks = 1
    else:
        effective_per_chunk = seconds_per_chunk - blend_source_sec
        num_chunks = max(1, int(np.ceil((selected_duration - blend_source_sec) / effective_per_chunk)))

    # Build config
    infer_config = _build_infer_config(
        num_frames=num_frames,
        resolution=resolution,
        steps=steps,
        guidance_scale=guidance_scale,
        noise_aug_strength=noise_aug_strength,
        sample_stride=sample_stride,
        fps=fps,
        seed=seed,
        frames_overlap=frames_overlap,
    )
    task_config = infer_config.test_case[0]

    # Create pipeline (only once, reused across all chunks)
    tracker = ProgressTracker(gradio_progress=gradio_progress)
    tracker.log(f"Device: {device}")
    tracker.log(f"Video: {video_path} ({video_duration:.1f}s, {video_fps:.1f}fps)")
    tracker.log(f"Selected range: {start_time:.1f}s - {end_time:.1f}s ({selected_duration:.1f}s)")
    tracker.log(f"Chunks: {num_chunks} x {num_frames} frames (blend: {CHUNK_BLEND_FRAMES})")
    tracker.log(f"Config: {resolution}p, {steps} steps, guidance={guidance_scale}, "
                f"stride={sample_stride}, fps={fps}")

    if gradio_progress:
        gradio_progress(0.0, desc="Loading pipeline...")

    pipeline = create_pipeline(infer_config, device)
    apply_patches(pipeline, gradio_progress=gradio_progress)

    # Process each chunk (saved to /workspace/ for crash safety + persistence)
    all_chunk_paths = []
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    chunks_dir = f"/workspace/chunks_{ts}"
    os.makedirs(chunks_dir, exist_ok=True)
    tmpdir = tempfile.mkdtemp(prefix="mimicmotion_")

    for chunk_idx in range(num_chunks):
        chunk_label = f"[Chunk {chunk_idx + 1}/{num_chunks}]"
        tracker.log(f"\n{'='*50}")
        tracker.log(f"{chunk_label} Starting...")

        # Calculate time range for this chunk
        effective_per_chunk = seconds_per_chunk - blend_source_sec
        chunk_start = start_time + chunk_idx * effective_per_chunk
        chunk_end = min(chunk_start + seconds_per_chunk, end_time)

        tracker.log(f"{chunk_label} Video segment: {chunk_start:.1f}s - {chunk_end:.1f}s")

        if gradio_progress:
            chunk_progress = chunk_idx / num_chunks
            gradio_progress(chunk_progress, desc=f"Chunk {chunk_idx+1}/{num_chunks}: trimming video...")

        # Trim video to chunk range
        chunk_video = os.path.join(tmpdir, f"chunk_{chunk_idx}.mp4")
        _trim_video(video_path, chunk_start, chunk_end, chunk_video)

        # Preprocess (extract poses for this chunk)
        tracker.log(f"{chunk_label} Extracting poses...")
        pose_pixels, image_pixels, ref_image = _preprocess(
            chunk_video, image_path, resolution, sample_stride, tracker
        )

        # Limit total pose frames (includes 1 prepended ref image pose)
        # Pipeline needs num_frames+1 poses to output num_frames frames (first is ref)
        max_pose_frames = num_frames + 1
        if pose_pixels.size(0) > max_pose_frames:
            pose_pixels = pose_pixels[:max_pose_frames]
        elif pose_pixels.size(0) < 16:
            tracker.log(f"{chunk_label} Too few frames ({pose_pixels.size(0)}), skipping")
            continue

        tracker.log(f"{chunk_label} Running inference on {pose_pixels.size(0) - 1} frames (+ ref)...")

        if gradio_progress:
            chunk_progress = (chunk_idx + 0.1) / num_chunks
            gradio_progress(chunk_progress, desc=f"Chunk {chunk_idx+1}/{num_chunks}: denoising...")

        # Re-apply patches for fresh progress bar per chunk
        apply_patches(pipeline, gradio_progress=gradio_progress)

        # Run inference
        chunk_frames = _run_pipeline(
            pipeline, image_pixels, pose_pixels, device, task_config, tracker
        )

        tracker.log(f"{chunk_label} Generated {chunk_frames.shape[0]} frames")

        # Save chunk to /workspace/ immediately (survives crashes + restarts)
        chunk_path = os.path.join(chunks_dir, f"chunk_{chunk_idx:03d}.pt")
        torch.save(chunk_frames, chunk_path)
        all_chunk_paths.append(chunk_path)
        tracker.log(f"{chunk_label} Saved to {chunk_path}")

        # Free GPU + CPU memory
        del chunk_frames
        torch.cuda.empty_cache()

    if not all_chunk_paths:
        raise RuntimeError("No chunks were processed successfully")

    # Load chunks back and blend
    tracker.log(f"\n{'='*50}")
    all_chunks = [torch.load(p, weights_only=True) for p in all_chunk_paths]

    if len(all_chunks) > 1:
        tracker.log(f"Blending {len(all_chunks)} chunks with {CHUNK_BLEND_FRAMES}-frame crossfade...")
        if gradio_progress:
            gradio_progress(0.95, desc="Blending chunks...")
        video_frames = _blend_chunks(all_chunks, CHUNK_BLEND_FRAMES)
    else:
        video_frames = all_chunks[0]

    # Save output — frames are (F, C, H, W), save_to_mp4 handles permutation
    tracker.log(f"Saving {video_frames.shape[0]} frames to {output_path}")
    save_to_mp4(video_frames, output_path, fps=fps)

    # Extract preview frames (evenly spaced, up to 12)
    total_frames = video_frames.shape[0]
    max_previews = min(12, total_frames)
    preview_indices = [int(i * (total_frames - 1) / (max_previews - 1)) for i in range(max_previews)] if max_previews > 1 else [0]
    preview_frames = [
        to_pil_image(video_frames[idx])
        for idx in preview_indices
    ]

    elapsed = time.time() - start_clock
    output_duration = total_frames / fps
    tracker.log(f"Done! {total_frames} frames, {output_duration:.1f}s video, "
                f"total time: {int(elapsed // 60)}m {int(elapsed % 60)}s")

    # Restore default dtype
    torch.set_default_dtype(original_dtype)

    # Cleanup temp files
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)

    return {
        "output_path": output_path,
        "elapsed": elapsed,
        "num_frames": total_frames,
        "resolution": f"{video_frames.shape[3]}x{video_frames.shape[2]}",  # FCHW -> WxH
        "logs": tracker.get_logs(),
        "preview_frames": preview_frames,
        "chunks_processed": len(all_chunks),
    }


def main():
    parser = argparse.ArgumentParser(description="MimicMotion motion transfer")
    parser.add_argument("--image", required=True, help="Path to reference image")
    parser.add_argument("--video", required=True, help="Path to motion source video")
    parser.add_argument("--output", default=None, help="Output video path")
    parser.add_argument("--num_frames", type=int, default=72)
    parser.add_argument("--resolution", type=int, default=576)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--guidance_scale", type=float, default=2.0)
    parser.add_argument("--noise_aug_strength", type=float, default=0.0)
    parser.add_argument("--sample_stride", type=int, default=2)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frames_overlap", type=int, default=6)
    parser.add_argument("--start_time", type=float, default=0.0,
                        help="Start time in seconds")
    parser.add_argument("--end_time", type=float, default=None,
                        help="End time in seconds (default: full video)")
    parser.add_argument("--no_auto_chunk", action="store_true",
                        help="Disable auto-chunking (only process first chunk)")
    args = parser.parse_args()

    result = run_mimicmotion(
        image_path=args.image,
        video_path=args.video,
        output_path=args.output,
        num_frames=args.num_frames,
        resolution=args.resolution,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        noise_aug_strength=args.noise_aug_strength,
        sample_stride=args.sample_stride,
        fps=args.fps,
        seed=args.seed,
        frames_overlap=args.frames_overlap,
        start_time=args.start_time,
        end_time=args.end_time,
        auto_chunk=not args.no_auto_chunk,
    )

    print()
    print("=" * 60)
    print(f"Output: {result['output_path']}")
    print(f"Frames: {result['num_frames']}")
    print(f"Resolution: {result['resolution']}")
    print(f"Chunks: {result['chunks_processed']}")
    print(f"Time: {int(result['elapsed'] // 60)}m {int(result['elapsed'] % 60)}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
