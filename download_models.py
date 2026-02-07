"""
Idempotent model downloader for StableAnimator on RunPod.

Downloads all required weights to /workspace/models/ (persistent volume)
and creates symlinks so StableAnimator's hardcoded relative paths resolve.

Models from HuggingFace: FrancisRing/StableAnimator
"""

import os
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

HF_REPO = "FrancisRing/StableAnimator"
MODELS_DIR = Path(os.environ.get("MODELS_DIR", "/workspace/models"))
APP_DIR = Path("/app")


def download_svd():
    """Download Stable Video Diffusion base model."""
    svd_dir = MODELS_DIR / "SVD" / "stable-video-diffusion-img2vid-xt"
    if svd_dir.exists() and (svd_dir / "model_index.json").exists():
        print("  [skip] SVD base model already exists", flush=True)
        return

    print("  [download] SVD base model (~5 GB)...", flush=True)
    snapshot_download(
        repo_id="stabilityai/stable-video-diffusion-img2vid-xt",
        local_dir=str(svd_dir),
    )
    print("  [done] SVD base model", flush=True)


def download_animation_weights():
    """Download StableAnimator-specific weights (pose_net, face_encoder, unet)."""
    anim_dir = MODELS_DIR / "Animation"
    anim_dir.mkdir(parents=True, exist_ok=True)

    files = [
        ("Animation/pose_net.pth", "pose_net.pth"),
        ("Animation/face_encoder.pth", "face_encoder.pth"),
        ("Animation/unet.pth", "unet.pth"),
    ]

    for hf_path, local_name in files:
        local_path = anim_dir / local_name
        if local_path.exists():
            print(f"  [skip] {local_name} already exists", flush=True)
            continue
        print(f"  [download] {local_name}...", flush=True)
        hf_hub_download(
            repo_id=HF_REPO,
            filename=hf_path,
            local_dir=str(MODELS_DIR),
        )
        print(f"  [done] {local_name}", flush=True)


def download_dwpose():
    """Download DWPose ONNX models."""
    dwpose_dir = MODELS_DIR / "DWPose"
    dwpose_dir.mkdir(parents=True, exist_ok=True)

    files = [
        ("DWPose/yolox_l.onnx", "yolox_l.onnx"),
        ("DWPose/dw-ll_ucoco_384.onnx", "dw-ll_ucoco_384.onnx"),
    ]

    for hf_path, local_name in files:
        local_path = dwpose_dir / local_name
        if local_path.exists():
            print(f"  [skip] {local_name} already exists", flush=True)
            continue
        print(f"  [download] {local_name}...", flush=True)
        hf_hub_download(
            repo_id=HF_REPO,
            filename=hf_path,
            local_dir=str(MODELS_DIR),
        )
        print(f"  [done] {local_name}", flush=True)


def download_insightface():
    """Download InsightFace antelopev2 models."""
    ante_dir = MODELS_DIR / "antelopev2"
    ante_dir.mkdir(parents=True, exist_ok=True)

    files = [
        "models/antelopev2/1k3d68.onnx",
        "models/antelopev2/2d106det.onnx",
        "models/antelopev2/genderage.onnx",
        "models/antelopev2/glintr100.onnx",
        "models/antelopev2/scrfd_10g_bnkps.onnx",
    ]

    for hf_path in files:
        local_name = Path(hf_path).name
        local_path = ante_dir / local_name
        if local_path.exists():
            print(f"  [skip] {local_name} already exists", flush=True)
            continue
        print(f"  [download] {local_name}...", flush=True)
        hf_hub_download(
            repo_id=HF_REPO,
            filename=hf_path,
            local_dir=str(MODELS_DIR),
        )
        # HF downloads to MODELS_DIR/models/antelopev2/ but we want MODELS_DIR/antelopev2/
        hf_downloaded = MODELS_DIR / hf_path
        if hf_downloaded.exists() and hf_downloaded != local_path:
            shutil.move(str(hf_downloaded), str(local_path))
        print(f"  [done] {local_name}", flush=True)

    # Clean up the nested models/ dir if it exists and is empty
    nested = MODELS_DIR / "models" / "antelopev2"
    if nested.exists() and not any(nested.iterdir()):
        nested.rmdir()
    nested_parent = MODELS_DIR / "models"
    if nested_parent.exists() and not any(nested_parent.iterdir()):
        nested_parent.rmdir()


def create_symlinks():
    """Create symlinks so StableAnimator's hardcoded relative paths resolve.

    StableAnimator expects (relative to CWD=/app):
      checkpoints/stable-video-diffusion-img2vid-xt/  -> SVD base
      checkpoints/Animation/                          -> Animation weights
      checkpoints/DWPose/                             -> DWPose ONNX models
      models/antelopev2/                              -> InsightFace models
    """
    print("Creating symlinks...", flush=True)

    links = [
        (MODELS_DIR / "SVD" / "stable-video-diffusion-img2vid-xt",
         APP_DIR / "checkpoints" / "stable-video-diffusion-img2vid-xt"),
        (MODELS_DIR / "Animation",
         APP_DIR / "checkpoints" / "Animation"),
        (MODELS_DIR / "DWPose",
         APP_DIR / "checkpoints" / "DWPose"),
        (MODELS_DIR / "antelopev2",
         APP_DIR / "models" / "antelopev2"),
    ]

    for src, dst in links:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.is_symlink():
            dst.unlink()
        elif dst.exists():
            if dst.is_dir():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        if src.exists():
            dst.symlink_to(src)
            print(f"  [link] {dst} -> {src}", flush=True)
        else:
            print(f"  [warn] source not found: {src}", flush=True)


def main():
    print("=" * 60, flush=True)
    print("StableAnimator Model Downloader", flush=True)
    print(f"  Models dir: {MODELS_DIR}", flush=True)
    print(f"  HF repo:    {HF_REPO}", flush=True)
    print("=" * 60, flush=True)
    print(flush=True)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("1/4 SVD base model", flush=True)
    download_svd()
    print(flush=True)

    print("2/4 StableAnimator weights", flush=True)
    download_animation_weights()
    print(flush=True)

    print("3/4 DWPose models", flush=True)
    download_dwpose()
    print(flush=True)

    print("4/4 InsightFace antelopev2", flush=True)
    download_insightface()
    print(flush=True)

    create_symlinks()
    print(flush=True)

    print("All models ready!", flush=True)


if __name__ == "__main__":
    main()
