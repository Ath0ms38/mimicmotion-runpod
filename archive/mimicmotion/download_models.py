"""
download_models.py - Download MimicMotion model weights to /workspace/models/

Downloads:
  1. DWPose ONNX models (yolox_l.onnx, dw-ll_ucoco_384.onnx) from yzd-v/DWPose
  2. MimicMotion_1-1.pth checkpoint from tencent/MimicMotion
  3. stabilityai/stable-video-diffusion-img2vid-xt-1-1 (full model via snapshot_download)

All downloads are idempotent -- files that already exist are skipped.
"""

import os
import sys
import time
from pathlib import Path

import huggingface_hub
from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.errors import GatedRepoError
from tqdm import tqdm

# Force tqdm progress bars to show in Docker
huggingface_hub.utils.tqdm.are_progress_bars_disabled = lambda: False

HF_TOKEN = os.environ.get("HF_TOKEN")

MODELS_DIR = Path(os.environ.get("MODELS_DIR", "/workspace/models"))
MIMICMOTION_MODELS_DIR = Path(
    os.environ.get("MIMICMOTION_MODELS_DIR", "/app/MimicMotion/models")
)

# Expected downloads with approximate sizes for the overall progress
DOWNLOADS = [
    ("DWPose: yolox_l.onnx", "~217 MB"),
    ("DWPose: dw-ll_ucoco_384.onnx", "~134 MB"),
    ("MimicMotion_1-1.pth", "~3.0 GB"),
    ("SVD model (multiple files)", "~5.0 GB"),
]


def _format_size(size_bytes):
    """Format bytes into human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def download_dwpose_models(overall_bar):
    """Download DWPose ONNX models for pose detection."""
    dwpose_dir = MODELS_DIR / "DWPose"
    dwpose_dir.mkdir(parents=True, exist_ok=True)

    files = [
        ("yzd-v/DWPose", "yolox_l.onnx"),
        ("yzd-v/DWPose", "dw-ll_ucoco_384.onnx"),
    ]

    for repo_id, filename in files:
        target = dwpose_dir / filename
        if target.exists():
            size = _format_size(target.stat().st_size)
            print(f"  [skip] {filename} ({size}) already exists", flush=True)
            overall_bar.update(1)
            continue

        print(f"  Downloading {filename} from {repo_id}...", flush=True)
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(dwpose_dir),
            token=HF_TOKEN,
        )
        size = _format_size((dwpose_dir / filename).stat().st_size)
        print(f"  [done] {filename} ({size})", flush=True)
        overall_bar.update(1)


def download_mimicmotion_checkpoint(overall_bar):
    """Download MimicMotion_1-1.pth checkpoint."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    target = MODELS_DIR / "MimicMotion_1-1.pth"

    if target.exists():
        size = _format_size(target.stat().st_size)
        print(f"  [skip] MimicMotion_1-1.pth ({size}) already exists", flush=True)
        overall_bar.update(1)
        return

    print("  Downloading MimicMotion_1-1.pth from tencent/MimicMotion...", flush=True)
    hf_hub_download(
        repo_id="tencent/MimicMotion",
        filename="MimicMotion_1-1.pth",
        local_dir=str(MODELS_DIR),
        token=HF_TOKEN,
    )
    size = _format_size(target.stat().st_size)
    print(f"  [done] MimicMotion_1-1.pth ({size})", flush=True)
    overall_bar.update(1)


def download_svd_model(overall_bar):
    """Download Stable Video Diffusion model."""
    svd_dir = MODELS_DIR / "stable-video-diffusion-img2vid-xt-1-1"

    if svd_dir.exists() and any(svd_dir.iterdir()):
        total = sum(f.stat().st_size for f in svd_dir.rglob("*") if f.is_file())
        print(f"  [skip] SVD model ({_format_size(total)}) already exists", flush=True)
        overall_bar.update(1)
        return

    print(
        "  Downloading stabilityai/stable-video-diffusion-img2vid-xt-1-1...",
        flush=True,
    )
    try:
        snapshot_download(
            repo_id="stabilityai/stable-video-diffusion-img2vid-xt-1-1",
            local_dir=str(svd_dir),
            token=HF_TOKEN,
        )
    except GatedRepoError:
        print(
            "\n"
            "  ERROR: The SVD model is a gated repo. To fix this:\n"
            "  1. Accept the terms at:\n"
            "     https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1\n"
            "  2. Create a token at: https://huggingface.co/settings/tokens\n"
            "  3. Run with: docker run -e HF_TOKEN=hf_xxx ...\n",
            flush=True,
        )
        raise SystemExit(1)
    total = sum(f.stat().st_size for f in svd_dir.rglob("*") if f.is_file())
    print(f"  [done] SVD model ({_format_size(total)})", flush=True)
    overall_bar.update(1)


def create_symlinks():
    """Create symlinks from MimicMotion repo models/ to /workspace/models/."""
    MIMICMOTION_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    links = [
        (MODELS_DIR / "DWPose", MIMICMOTION_MODELS_DIR / "DWPose"),
        (
            MODELS_DIR / "MimicMotion_1-1.pth",
            MIMICMOTION_MODELS_DIR / "MimicMotion_1-1.pth",
        ),
    ]

    for src, dst in links:
        if dst.is_symlink():
            dst.unlink()
        elif dst.exists():
            import shutil
            if dst.is_dir():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        if src.exists():
            dst.symlink_to(src)
            print(f"  [link] {dst} -> {src}", flush=True)


def main():
    print("=" * 60, flush=True)
    print("MimicMotion Model Downloader", flush=True)
    print("=" * 60, flush=True)
    print(f"Models directory: {MODELS_DIR}", flush=True)
    print(flush=True)

    start = time.time()

    with tqdm(
        total=4,
        desc="Overall",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        file=sys.stderr,
    ) as overall_bar:
        print("[1/3] DWPose models...", flush=True)
        download_dwpose_models(overall_bar)
        print(flush=True)

        print("[2/3] MimicMotion checkpoint...", flush=True)
        download_mimicmotion_checkpoint(overall_bar)
        print(flush=True)

        print("[3/3] Stable Video Diffusion model...", flush=True)
        download_svd_model(overall_bar)
        print(flush=True)

    elapsed = time.time() - start
    print(f"Downloads completed in {int(elapsed // 60)}m {int(elapsed % 60)}s", flush=True)
    print(flush=True)

    print("Creating symlinks...", flush=True)
    create_symlinks()
    print(flush=True)

    print("All models ready!", flush=True)


if __name__ == "__main__":
    main()
