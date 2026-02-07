"""
StableAnimator inference pipeline for RunPod.

Accepts a reference image + motion video, extracts poses via DWPose,
runs StableAnimator diffusion with ID-preserving face adapter,
and outputs an animated MP4 video.

Callable from CLI (argparse) or start.sh entrypoint.
"""

import argparse
import gc
import os
import random
import shutil
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------

class ProgressTracker:
    """Simple log collector that optionally prints to stdout."""

    def __init__(self):
        self.logs: list[str] = []

    def log(self, msg: str):
        self.logs.append(msg)
        print(msg, flush=True)


# ---------------------------------------------------------------------------
# Video helpers
# ---------------------------------------------------------------------------

def extract_frames_from_video(
    video_path: str,
    output_dir: str,
    start_time: float = 0,
    end_time: float = 0,
    sample_stride: int = 1,
    tracker: ProgressTracker | None = None,
) -> list[str]:
    """Extract frames from video using decord, save as PNGs.

    Skips extraction if output_dir already contains frames (resume support).
    """
    import decord
    decord.bridge.set_bridge("native")

    if tracker is None:
        tracker = ProgressTracker()

    # Check if frames were already extracted (crash recovery)
    if os.path.isdir(output_dir):
        existing = sorted(
            [f for f in os.listdir(output_dir) if f.endswith(".png")],
            key=lambda x: int(x.split("_")[1].split(".")[0]),
        )
        if existing:
            tracker.log(f"Frames already extracted ({len(existing)} files), skipping")
            return [os.path.join(output_dir, f) for f in existing]

    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    fps = vr.get_avg_fps()
    total_frames = len(vr)

    start_frame = int(start_time * fps) if start_time > 0 else 0
    end_frame = int(end_time * fps) if end_time > 0 else total_frames
    end_frame = min(end_frame, total_frames)

    frame_indices = list(range(start_frame, end_frame, sample_stride))
    tracker.log(f"Extracting {len(frame_indices)} frames from video "
                f"(frames {start_frame}-{end_frame}, stride={sample_stride})")

    os.makedirs(output_dir, exist_ok=True)
    frames = vr.get_batch(frame_indices).asnumpy()

    paths = []
    for i, frame in enumerate(frames):
        path = os.path.join(output_dir, f"frame_{i}.png")
        cv2.imwrite(path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        paths.append(path)

    tracker.log(f"Saved {len(paths)} frames to {output_dir}")
    return paths


# ---------------------------------------------------------------------------
# DWPose extraction (adapted from StableAnimator/DWPose/skeleton_extraction.py)
# Note: skeleton_extraction.py uses 'from dwpose_utils...' which breaks when
# imported as a package. We inline the logic here with proper import paths.
# ---------------------------------------------------------------------------

def _init_dwpose_detector():
    """Lazily import and return the DWPose detector.

    Reuses the module-level instance from dwpose_detector.py to avoid
    loading ONNX models twice (~350MB).
    """
    from DWPose.dwpose_utils.dwpose_detector import dwpose_detector_aligned
    return dwpose_detector_aligned


def _draw_pose(pose, H, W, ref_w=2160):
    """Draw pose skeleton on canvas (from skeleton_extraction.py)."""
    import math
    import matplotlib

    bodies = pose["bodies"]
    faces = pose["faces"]
    hands = pose["hands"]
    candidate = bodies["candidate"]
    subset = bodies["subset"]

    sz = min(H, W)
    sr = (ref_w / sz) if sz != ref_w else 1
    canvas = np.zeros(shape=(int(H * sr), int(W * sr), 3), dtype=np.uint8)

    # Draw body
    stickwidth = 4
    limbSeq = [
        [2,3],[2,6],[3,4],[4,5],[6,7],[7,8],[2,9],[9,10],
        [10,11],[2,12],[12,13],[13,14],[2,1],[1,15],[15,17],
        [1,16],[16,18],[3,17],[6,18],
    ]
    colors = [
        [255,0,0],[255,85,0],[255,170,0],[255,255,0],[170,255,0],[85,255,0],
        [0,255,0],[0,255,85],[0,255,170],[0,255,255],[0,170,255],[0,85,255],
        [0,0,255],[85,0,255],[170,0,255],[255,0,255],[255,0,170],[255,0,85],
    ]

    score = bodies["score"]
    cH, cW = canvas.shape[:2]

    for i in range(17):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i]) - 1]
            conf = score[n][np.array(limbSeq[i]) - 1]
            if conf[0] < 0.3 or conf[1] < 0.3:
                continue
            Y = candidate[index.astype(int), 0] * float(cW)
            X = candidate[index.astype(int), 1] * float(cH)
            mX, mY = np.mean(X), np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            c = [int(v * conf[0] * conf[1]) for v in colors[i]]
            cv2.fillConvexPoly(canvas, polygon, c)

    canvas = (canvas * 0.6).astype(np.uint8)
    for i in range(18):
        for n in range(len(subset)):
            index = int(subset[n][i])
            if index == -1:
                continue
            x, y = candidate[index][0:2]
            conf = score[n][i]
            x, y = int(x * cW), int(y * cH)
            c = [int(v * conf) for v in colors[i]]
            cv2.circle(canvas, (x, y), 4, c, thickness=-1)

    # Draw hands
    edges = [
        [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],
        [10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20],
    ]
    for peaks, scores_h in zip(hands, pose["hands_score"]):
        for ie, e in enumerate(edges):
            x1, y1 = peaks[e[0]]
            x2, y2 = peaks[e[1]]
            x1, y1 = int(x1 * cW), int(y1 * cH)
            x2, y2 = int(x2 * cW), int(y2 * cH)
            s = int(scores_h[e[0]] * scores_h[e[1]] * 255)
            eps = 0.01
            if x1 > eps and y1 > eps and x2 > eps and y2 > eps:
                color = matplotlib.colors.hsv_to_rgb([ie / float(len(edges)), 1.0, 1.0]) * s
                cv2.line(canvas, (x1, y1), (x2, y2), color.tolist(), thickness=2)
        for i, kp in enumerate(peaks):
            x, y = int(kp[0] * cW), int(kp[1] * cH)
            s = int(scores_h[i] * 255)
            if x > eps and y > eps:
                cv2.circle(canvas, (x, y), 4, (0, 0, s), thickness=-1)

    # Draw face
    for lmks, scores_f in zip(faces, pose["faces_score"]):
        for lmk, s in zip(lmks, scores_f):
            x, y = int(lmk[0] * cW), int(lmk[1] * cH)
            conf = int(s * 255)
            if x > 0.01 and y > 0.01:
                cv2.circle(canvas, (x, y), 3, (conf, conf, conf), thickness=-1)

    return cv2.cvtColor(cv2.resize(canvas, (W, H)), cv2.COLOR_BGR2RGB).transpose(2, 0, 1)


def extract_poses(
    video_frames_dir: str,
    ref_image_path: str,
    poses_dir: str,
    tracker: ProgressTracker | None = None,
) -> list[str]:
    """Extract DWPose skeletons from video frames, aligned to reference image.

    Skips extraction if poses_dir already contains frames (resume support).
    """
    if tracker is None:
        tracker = ProgressTracker()

    # Check if poses were already extracted (crash recovery)
    if os.path.isdir(poses_dir):
        existing = sorted(
            [f for f in os.listdir(poses_dir) if f.endswith(".png")],
            key=lambda x: int(x.split("_")[1].split(".")[0]),
        )
        if existing:
            tracker.log(f"Poses already extracted ({len(existing)} files), skipping")
            return [os.path.join(poses_dir, f) for f in existing]

    tracker.log("Extracting poses with DWPose...")

    detector = _init_dwpose_detector()

    # Process reference image
    ref_image = cv2.imread(ref_image_path)
    ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
    height, width, _ = ref_image.shape
    ref_pose = detector(ref_image)
    ref_keypoint_id = [0, 1, 2, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    ref_keypoint_id = [
        i for i in ref_keypoint_id
        if len(ref_pose["bodies"]["subset"]) > 0 and ref_pose["bodies"]["subset"][0][i] >= 0.0
    ]
    ref_body = ref_pose["bodies"]["candidate"][ref_keypoint_id]

    # Process video frames
    os.makedirs(poses_dir, exist_ok=True)
    files = sorted(
        [f for f in os.listdir(video_frames_dir) if f.endswith(".png")],
        key=lambda x: int(x.split("_")[1].split(".")[0]),
    )

    # Read first frame to get video dimensions (may differ from reference)
    first_frame = cv2.imread(os.path.join(video_frames_dir, files[0]))
    fh, fw = first_frame.shape[:2]

    detected_poses = []
    for fname in tqdm(files, desc="DWPose"):
        frame = cv2.imread(os.path.join(video_frames_dir, fname))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detected_poses.append(detector(frame))

    # Note: we reuse the module-level detector instance across runs,
    # so we do NOT call release_memory() here.

    # Compute rescale parameters
    detected_bodies = np.stack(
        [p["bodies"]["candidate"] for p in detected_poses
         if p["bodies"]["candidate"].shape[0] == 18]
    )[:, ref_keypoint_id]
    ay, by = np.polyfit(
        detected_bodies[:, :, 1].flatten(),
        np.tile(ref_body[:, 1], len(detected_bodies)), 1,
    )
    # Adjust for aspect ratio difference between video frames and reference image
    ax = ay / (fh / fw / height * width)
    bx = np.mean(
        np.tile(ref_body[:, 0], len(detected_bodies))
        - detected_bodies[:, :, 0].flatten() * ax
    )
    a = np.array([ax, ay])
    b = np.array([bx, by])

    # Rescale poses and draw
    paths = []
    for i, detected_pose in enumerate(detected_poses):
        detected_pose["bodies"]["candidate"] = detected_pose["bodies"]["candidate"] * a + b
        detected_pose["faces"] = detected_pose["faces"] * a + b
        detected_pose["hands"] = detected_pose["hands"] * a + b
        im = _draw_pose(detected_pose, height, width)
        pose_image = np.transpose(im, (1, 2, 0))
        pose_path = os.path.join(poses_dir, f"frame_{i}.png")
        cv2.imwrite(pose_path, pose_image)
        paths.append(pose_path)

    tracker.log(f"Extracted {len(paths)} pose frames")
    return paths


# ---------------------------------------------------------------------------
# Face embedding extraction
# ---------------------------------------------------------------------------

def extract_face_embedding(
    image_path: str,
    face_model,
    tracker: ProgressTracker | None = None,
) -> np.ndarray:
    """Extract 512-dim face ID embedding from reference image."""
    if tracker is None:
        tracker = ProgressTracker()

    tracker.log("Extracting face embedding...")

    face_model.face_helper.clean_all()
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    face_info = face_model.app.get(image_rgb)
    if len(face_info) > 0:
        face_info = sorted(
            face_info,
            key=lambda x: (x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1]),
        )[-1]
        embedding = face_info["embedding"]
        tracker.log("Face detected via InsightFace")
        return embedding

    # Fallback: use FaceRestoreHelper
    face_model.face_helper.read_image(image_rgb)
    face_model.face_helper.get_face_landmarks_5(only_center_face=True)
    face_model.face_helper.align_warp_face()

    if len(face_model.face_helper.cropped_faces) == 0:
        tracker.log("WARNING: No face detected, using zero embedding")
        return np.zeros((512,))

    align_face = face_model.face_helper.cropped_faces[0]
    embedding = face_model.handler_ante.get_feat(align_face)
    tracker.log("Face detected via FaceRestoreHelper fallback")
    return embedding


# ---------------------------------------------------------------------------
# Pipeline loading & caching
# ---------------------------------------------------------------------------

_cached_pipeline = None
_cached_face_model = None


def _get_dtype():
    """Auto-detect best dtype for current GPU."""
    if torch.cuda.get_device_capability()[0] >= 8:
        return torch.bfloat16
    return torch.float16


def load_pipeline(tracker: ProgressTracker | None = None):
    """Load and cache StableAnimator pipeline + face model."""
    global _cached_pipeline, _cached_face_model

    if _cached_pipeline is not None:
        if tracker:
            tracker.log("Using cached pipeline")
        return _cached_pipeline, _cached_face_model

    if tracker is None:
        tracker = ProgressTracker()

    from diffusers import AutoencoderKLTemporalDecoder, EulerDiscreteScheduler
    from diffusers.models.attention_processor import XFormersAttnProcessor
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

    from animation.modules.attention_processor import AnimationAttnProcessor
    from animation.modules.attention_processor_normalized import AnimationIDAttnNormalizedProcessor
    from animation.modules.face_model import FaceModel
    from animation.modules.id_encoder import FusionFaceId
    from animation.modules.pose_net import PoseNet
    from animation.modules.unet import UNetSpatioTemporalConditionModel
    from animation.pipelines.inference_pipeline_animation import InferenceAnimationPipeline

    svd_path = "checkpoints/stable-video-diffusion-img2vid-xt"
    posenet_path = "checkpoints/Animation/pose_net.pth"
    face_encoder_path = "checkpoints/Animation/face_encoder.pth"
    unet_path = "checkpoints/Animation/unet.pth"

    dtype = _get_dtype()
    tracker.log(f"Loading pipeline (dtype={dtype})...")

    # Load base components
    tracker.log("  Loading CLIP image processor...")
    feature_extractor = CLIPImageProcessor.from_pretrained(svd_path, subfolder="feature_extractor")

    tracker.log("  Loading noise scheduler...")
    noise_scheduler = EulerDiscreteScheduler.from_pretrained(svd_path, subfolder="scheduler")

    tracker.log("  Loading CLIP image encoder...")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(svd_path, subfolder="image_encoder")

    tracker.log("  Loading VAE...")
    vae = AutoencoderKLTemporalDecoder.from_pretrained(svd_path, subfolder="vae")

    tracker.log("  Loading UNet...")
    unet = UNetSpatioTemporalConditionModel.from_pretrained(svd_path, subfolder="unet", low_cpu_mem_usage=True)

    # StableAnimator modules
    pose_net = PoseNet(noise_latent_channels=unet.config.block_out_channels[0])
    face_encoder = FusionFaceId(
        cross_attention_dim=1024,
        id_embeddings_dim=512,
        clip_embeddings_dim=1024,
        num_tokens=4,
    )
    face_model = FaceModel()

    # Set up LoRA attention processors
    tracker.log("  Setting up attention processors...")
    lora_rank = 128
    attn_procs = {}
    unet_svd = unet.state_dict()

    for name in unet.attn_processors.keys():
        if "transformer_blocks" in name and "temporal_transformer_blocks" not in name:
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = AnimationAttnProcessor(
                    hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=lora_rank,
                )
            else:
                layer_name = name.split(".processor")[0]
                weights = {
                    "to_k_ip.weight": unet_svd[layer_name + ".to_k.weight"],
                    "to_v_ip.weight": unet_svd[layer_name + ".to_v.weight"],
                }
                attn_procs[name] = AnimationIDAttnNormalizedProcessor(
                    hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=lora_rank,
                )
                attn_procs[name].load_state_dict(weights, strict=False)
        elif "temporal_transformer_blocks" in name:
            attn_procs[name] = XFormersAttnProcessor()

    unet.set_attn_processor(attn_procs)

    # Load StableAnimator weights
    tracker.log("  Loading StableAnimator weights...")
    pose_net.load_state_dict(torch.load(posenet_path, map_location="cpu", weights_only=True), strict=True)
    face_encoder.load_state_dict(torch.load(face_encoder_path, map_location="cpu", weights_only=True), strict=True)
    unet.load_state_dict(torch.load(unet_path, map_location="cpu", weights_only=True), strict=True)

    # Freeze all
    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    pose_net.requires_grad_(False)
    face_encoder.requires_grad_(False)

    # Build pipeline
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = InferenceAnimationPipeline(
        vae=vae,
        image_encoder=image_encoder,
        unet=unet,
        scheduler=noise_scheduler,
        feature_extractor=feature_extractor,
        pose_net=pose_net,
        face_encoder=face_encoder,
    ).to(device=device, dtype=dtype)

    tracker.log("Pipeline loaded and cached")

    _cached_pipeline = pipeline
    _cached_face_model = face_model
    return pipeline, face_model


# ---------------------------------------------------------------------------
# Save output video
# ---------------------------------------------------------------------------

def save_frames_as_mp4(frames: list[np.ndarray], output_path: str, fps: int = 8):
    """Save list of numpy frames (RGB) as MP4."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    h, w, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


# ---------------------------------------------------------------------------
# Auto-chunking
# ---------------------------------------------------------------------------

def _crossfade(chunk_a: list[np.ndarray], chunk_b: list[np.ndarray], overlap: int) -> list[np.ndarray]:
    """Crossfade overlapping frames between two chunks."""
    if overlap <= 0:
        return chunk_a + chunk_b

    blended = []
    for i in range(overlap):
        alpha = i / overlap
        a = chunk_a[-(overlap - i)].astype(np.float32)
        b = chunk_b[i].astype(np.float32)
        blended.append((a * (1 - alpha) + b * alpha).astype(np.uint8))

    return chunk_a[:-overlap] + blended + chunk_b[overlap:]


# ---------------------------------------------------------------------------
# Main inference function
# ---------------------------------------------------------------------------

def run_stableanimator(
    image_path: str,
    video_path: str,
    output_path: str | None = None,
    width: int = 512,
    height: int = 512,
    guidance_scale: float = 3.0,
    num_inference_steps: int = 25,
    tile_size: int = 16,
    frames_overlap: int = 4,
    noise_aug_strength: float = 0.02,
    decode_chunk_size: int = 4,
    fps: int = 8,
    seed: int = -1,
    sample_stride: int = 1,
    start_time: float = 0,
    end_time: float = 0,
    max_frames_per_chunk: int = 64,
) -> dict:
    """
    End-to-end StableAnimator inference.

    Args:
        image_path: Reference image (person whose identity to preserve).
        video_path: Motion source video (dance, etc.).
        output_path: Where to save the output MP4. Auto-generated if None.
        width, height: Output resolution (512x512 or 576x1024).
        guidance_scale: Classifier-free guidance scale.
        num_inference_steps: Denoising steps.
        tile_size: Temporal tile size for UNet.
        frames_overlap: Overlap between temporal tiles.
        noise_aug_strength: Noise augmentation on input image.
        decode_chunk_size: Frames decoded at once by VAE.
        fps: Output video FPS.
        seed: Random seed (-1 for random).
        sample_stride: Sample every Nth frame from video.
        start_time, end_time: Video time range (0 = full video).
        max_frames_per_chunk: Auto-chunk threshold.

    Returns:
        dict with keys: output_path, num_frames, duration, elapsed, logs
    """
    t0 = time.time()
    tracker = ProgressTracker()

    if output_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"/workspace/output_{ts}.mp4"

    if seed == -1:
        seed = random.randint(1, 2**20 - 1)
    tracker.log(f"Seed: {seed}")
    tracker.log(f"Resolution: {width}x{height}")

    # Create working directory on persistent volume so intermediate
    # data (frames, poses, chunks) survives pod restarts / crashes.
    job_id = Path(output_path).stem  # e.g. "output_20250207_143012"
    work_dir = os.path.join("/workspace", "jobs", job_id)
    frames_dir = os.path.join(work_dir, "frames")
    poses_dir = os.path.join(work_dir, "poses")
    chunks_dir = os.path.join(work_dir, "chunks")
    os.makedirs(work_dir, exist_ok=True)

    try:
        # Step 1: Extract frames from video
        tracker.log("")
        tracker.log("=" * 40)
        tracker.log("Step 1: Extracting video frames")
        tracker.log("=" * 40)
        extract_frames_from_video(
            video_path, frames_dir,
            start_time=start_time, end_time=end_time,
            sample_stride=sample_stride, tracker=tracker,
        )

        # Count extracted frames
        frame_files = sorted(
            [f for f in os.listdir(frames_dir) if f.endswith(".png")],
            key=lambda x: int(x.split("_")[1].split(".")[0]),
        )
        total_frames = len(frame_files)
        tracker.log(f"Total frames: {total_frames}")

        # Step 2: Extract poses
        tracker.log("")
        tracker.log("=" * 40)
        tracker.log("Step 2: DWPose skeleton extraction")
        tracker.log("=" * 40)
        extract_poses(frames_dir, image_path, poses_dir, tracker=tracker)

        # Step 3: Load pipeline
        tracker.log("")
        tracker.log("=" * 40)
        tracker.log("Step 3: Loading pipeline")
        tracker.log("=" * 40)
        pipeline, face_model = load_pipeline(tracker=tracker)

        # Step 4: Extract face embedding
        tracker.log("")
        tracker.log("=" * 40)
        tracker.log("Step 4: Face embedding")
        tracker.log("=" * 40)
        face_embedding = extract_face_embedding(image_path, face_model, tracker=tracker)

        # Step 5: Run inference (with auto-chunking)
        tracker.log("")
        tracker.log("=" * 40)
        tracker.log("Step 5: Running inference")
        tracker.log("=" * 40)

        # Load pose images
        pose_files = sorted(
            [f for f in os.listdir(poses_dir) if f.endswith(".png")],
            key=lambda x: int(x.split("_")[1].split(".")[0]),
        )

        if total_frames <= max_frames_per_chunk:
            # Single chunk
            all_output_frames = _run_single_chunk(
                pipeline, image_path, poses_dir, pose_files,
                face_embedding, width, height, guidance_scale,
                num_inference_steps, tile_size, frames_overlap,
                noise_aug_strength, decode_chunk_size, seed,
                tracker,
            )
        else:
            # Auto-chunking with persistence for crash recovery
            chunk_overlap = 8
            all_output_frames = _run_chunked(
                pipeline, image_path, poses_dir, pose_files,
                face_embedding, width, height, guidance_scale,
                num_inference_steps, tile_size, frames_overlap,
                noise_aug_strength, decode_chunk_size, seed,
                max_frames_per_chunk, chunk_overlap, chunks_dir,
                tracker,
            )

        # Step 6: Save output
        tracker.log("")
        tracker.log("=" * 40)
        tracker.log("Step 6: Saving output")
        tracker.log("=" * 40)

        save_frames_as_mp4(all_output_frames, output_path, fps)
        duration = len(all_output_frames) / fps
        elapsed = time.time() - t0

        tracker.log(f"Output: {output_path}")
        tracker.log(f"Frames: {len(all_output_frames)}, Duration: {duration:.1f}s, FPS: {fps}")
        tracker.log(f"Total time: {elapsed:.1f}s")

        # Clean up job directory after successful save
        shutil.rmtree(work_dir, ignore_errors=True)
        tracker.log(f"Cleaned up working directory: {work_dir}")

        return {
            "output_path": output_path,
            "num_frames": len(all_output_frames),
            "duration": duration,
            "elapsed": elapsed,
            "logs": "\n".join(tracker.logs),
        }

    finally:
        # Always free GPU memory; job dir is kept on failure for resume
        gc.collect()
        torch.cuda.empty_cache()


def _run_single_chunk(
    pipeline, image_path, poses_dir, pose_files,
    face_embedding, width, height, guidance_scale,
    num_inference_steps, tile_size, frames_overlap,
    noise_aug_strength, decode_chunk_size, seed,
    tracker,
) -> list[np.ndarray]:
    """Run inference on a single chunk of pose frames."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=device).manual_seed(seed)

    # Load pose images as PIL
    pose_images = []
    for pf in pose_files:
        img = Image.open(os.path.join(poses_dir, pf)).convert("RGB")
        img = img.resize((width, height))
        pose_images.append(img)

    num_frames = len(pose_images)
    tracker.log(f"Processing {num_frames} frames in single chunk")

    validation_image = Image.open(image_path).convert("RGB")

    video_frames = pipeline(
        image=validation_image,
        image_pose=pose_images,
        height=height,
        width=width,
        num_frames=num_frames,
        tile_size=tile_size,
        tile_overlap=frames_overlap,
        decode_chunk_size=decode_chunk_size,
        motion_bucket_id=127.0,
        fps=7,
        min_guidance_scale=guidance_scale,
        max_guidance_scale=guidance_scale,
        noise_aug_strength=noise_aug_strength,
        num_inference_steps=num_inference_steps,
        generator=generator,
        output_type="pil",
        validation_image_id_ante_embedding=face_embedding,
    ).frames[0]

    output_frames = [np.array(f) for f in video_frames]
    tracker.log(f"Generated {len(output_frames)} frames")
    return output_frames


def _run_chunked(
    pipeline, image_path, poses_dir, pose_files,
    face_embedding, width, height, guidance_scale,
    num_inference_steps, tile_size, frames_overlap,
    noise_aug_strength, decode_chunk_size, seed,
    max_frames, chunk_overlap, chunks_dir, tracker,
) -> list[np.ndarray]:
    """Run inference with auto-chunking for long videos.

    Each chunk is saved to chunks_dir as chunk_NNN.npy immediately after
    generation. On resume (e.g. after a crash), existing chunk files are
    loaded instead of re-generated.
    """
    os.makedirs(chunks_dir, exist_ok=True)

    total = len(pose_files)
    stride = max_frames - chunk_overlap
    chunks = []
    start = 0
    while start < total:
        end = min(start + max_frames, total)
        chunks.append((start, end))
        if end >= total:
            break
        start += stride

    tracker.log(f"Auto-chunking: {len(chunks)} chunks "
                f"(max {max_frames} frames/chunk, {chunk_overlap} overlap)")

    # Generate (or load from disk) each chunk
    chunk_frame_lists: list[list[np.ndarray]] = []
    for ci, (start, end) in enumerate(chunks):
        chunk_path = os.path.join(chunks_dir, f"chunk_{ci:03d}.npy")
        tracker.log(f"")
        tracker.log(f"--- Chunk {ci+1}/{len(chunks)} (frames {start}-{end-1}) ---")

        if os.path.exists(chunk_path):
            tracker.log(f"  Loaded from cache: {chunk_path}")
            chunk_frames = [f for f in np.load(chunk_path)]
        else:
            chunk_pose_files = pose_files[start:end]
            chunk_frames = _run_single_chunk(
                pipeline, image_path, poses_dir, chunk_pose_files,
                face_embedding, width, height, guidance_scale,
                num_inference_steps, tile_size, frames_overlap,
                noise_aug_strength, decode_chunk_size, seed + ci,
                tracker,
            )
            # Save to persistent volume immediately
            np.save(chunk_path, np.stack(chunk_frames))
            tracker.log(f"  Saved to: {chunk_path}")

        chunk_frame_lists.append(chunk_frames)

        # Free memory between chunks
        del chunk_frames
        gc.collect()
        torch.cuda.empty_cache()

    # Stitch all chunks with crossfade
    tracker.log("")
    tracker.log("Stitching chunks with crossfade...")
    all_frames = chunk_frame_lists[0]
    for ci in range(1, len(chunk_frame_lists)):
        all_frames = _crossfade(all_frames, chunk_frame_lists[ci], chunk_overlap)

    return all_frames


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="StableAnimator inference")
    parser.add_argument("--image", required=True, help="Reference image path")
    parser.add_argument("--video", required=True, help="Motion video path")
    parser.add_argument("--output", default=None, help="Output MP4 path")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--tile-size", type=int, default=16)
    parser.add_argument("--frames-overlap", type=int, default=4)
    parser.add_argument("--noise-aug", type=float, default=0.02)
    parser.add_argument("--decode-chunk-size", type=int, default=4)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--start-time", type=float, default=0)
    parser.add_argument("--end-time", type=float, default=0)
    parser.add_argument("--max-frames", type=int, default=64)
    args = parser.parse_args()

    result = run_stableanimator(
        image_path=args.image,
        video_path=args.video,
        output_path=args.output,
        width=args.width,
        height=args.height,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.steps,
        tile_size=args.tile_size,
        frames_overlap=args.frames_overlap,
        noise_aug_strength=args.noise_aug,
        decode_chunk_size=args.decode_chunk_size,
        fps=args.fps,
        seed=args.seed,
        sample_stride=args.sample_stride,
        start_time=args.start_time,
        end_time=args.end_time,
        max_frames_per_chunk=args.max_frames,
    )
    print(f"\nDone! Output: {result['output_path']}")


if __name__ == "__main__":
    main()
