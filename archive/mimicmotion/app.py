"""
app.py - Gradio web UI for MimicMotion motion transfer.

Provides drag-and-drop upload for reference image and motion video,
settings sliders, start/end time controls for video range selection,
auto-chunking for full-length videos, real-time progress bar with ETA,
output video preview, and a frame gallery.
"""

import os
import subprocess
import json

import gradio as gr
from run import run_mimicmotion

SAMPLES_DIR = "/app/samples"
DEFAULT_IMAGE = os.path.join(SAMPLES_DIR, "IMG_8502.jpg")
DEFAULT_VIDEO = os.path.join(SAMPLES_DIR, "caramell_dansen.mp4")


def get_video_duration(video_path):
    """Get video duration in seconds."""
    if not video_path:
        return 0
    try:
        cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_streams", "-select_streams", "v:0", str(video_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        data = json.loads(result.stdout)
        return float(data["streams"][0].get("duration", 0))
    except Exception:
        return 0


def on_video_upload(video):
    """Update time sliders when a video is uploaded."""
    duration = get_video_duration(video)
    if duration <= 0:
        return gr.update(), gr.update(), ""

    info = f"Video duration: {int(duration // 60)}m {int(duration % 60)}s ({duration:.1f}s)"

    # Estimate chunks
    seconds_per_chunk = 72 * 2 / 25  # num_frames * stride / assumed_fps
    num_chunks = max(1, int(duration / seconds_per_chunk) + 1)
    info += f" | ~{num_chunks} chunks for full video"

    return (
        gr.update(maximum=duration, value=0),
        gr.update(maximum=duration, value=duration),
        info,
    )


def generate(
    image,
    video,
    start_time,
    end_time,
    auto_chunk,
    num_frames,
    resolution,
    steps,
    guidance_scale,
    noise_aug_strength,
    sample_stride,
    fps,
    seed,
    frames_overlap,
    progress=gr.Progress(track_tqdm=True),
):
    """Run MimicMotion inference with Gradio progress tracking."""
    if image is None:
        raise gr.Error("Please upload a reference image.")
    if video is None:
        raise gr.Error("Please upload a motion source video.")

    result = run_mimicmotion(
        image_path=image,
        video_path=video,
        num_frames=int(num_frames),
        resolution=int(resolution),
        steps=int(steps),
        guidance_scale=float(guidance_scale),
        noise_aug_strength=float(noise_aug_strength),
        sample_stride=int(sample_stride),
        fps=int(fps),
        seed=int(seed),
        frames_overlap=int(frames_overlap),
        start_time=float(start_time),
        end_time=float(end_time) if end_time > 0 else None,
        auto_chunk=bool(auto_chunk),
        gradio_progress=progress,
    )

    elapsed = result["elapsed"]
    chunks = result.get("chunks_processed", 1)
    output_duration = result["num_frames"] / int(fps)
    summary = (
        f"Done in {int(elapsed // 60)}m {int(elapsed % 60)}s | "
        f"{result['num_frames']} frames ({output_duration:.1f}s) | "
        f"{chunks} chunk{'s' if chunks > 1 else ''} | {result['resolution']}"
    )

    # Build gallery with frame labels
    n = len(result["preview_frames"])
    total = result["num_frames"]
    gallery_items = []
    for i, frame in enumerate(result["preview_frames"]):
        frame_idx = int(i * (total - 1) / (n - 1)) if n > 1 else 0
        gallery_items.append((frame, f"Frame {frame_idx + 1}/{total}"))

    return result["output_path"], gallery_items, result["logs"], summary


with gr.Blocks(title="MimicMotion", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # MimicMotion - Motion Transfer
        Upload a **reference image** (the person to animate) and a **motion video**
        (the dance/movement to transfer). Set the time range, adjust settings, and
        click **Generate**. Long videos are automatically split into chunks and
        assembled with crossfade blending.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                label="Reference Image",
                type="filepath",
                value=DEFAULT_IMAGE if os.path.exists(DEFAULT_IMAGE) else None,
                height=400,
            )
            video_input = gr.Video(
                label="Motion Source Video",
                value=DEFAULT_VIDEO if os.path.exists(DEFAULT_VIDEO) else None,
                height=400,
            )
            video_info = gr.Textbox(
                label="Video Info", interactive=False, lines=1,
            )

        with gr.Column(scale=1):
            output_video = gr.Video(label="Output Video", height=400)
            status_text = gr.Textbox(label="Status", interactive=False, lines=1)

    with gr.Row():
        start_time = gr.Slider(
            minimum=0, maximum=300, value=0, step=0.5,
            label="Start Time (seconds)",
        )
        end_time = gr.Slider(
            minimum=0, maximum=300, value=0, step=0.5,
            label="End Time (seconds, 0 = full video)",
        )
        auto_chunk = gr.Checkbox(
            value=True,
            label="Auto-chunk (process full range)",
        )

    # Update sliders when video is uploaded
    video_input.change(
        fn=on_video_upload,
        inputs=[video_input],
        outputs=[start_time, end_time, video_info],
    )

    with gr.Accordion("Frame Preview", open=True):
        frame_gallery = gr.Gallery(
            label="Output Frames",
            columns=6,
            rows=2,
            height="auto",
            object_fit="contain",
        )

    with gr.Accordion("Settings", open=False):
        with gr.Row():
            num_frames = gr.Slider(
                minimum=16, maximum=72, value=72, step=1,
                label="Frames per Chunk",
            )
            resolution = gr.Slider(
                minimum=256, maximum=576, value=576, step=64,
                label="Resolution (height)",
            )
            steps = gr.Slider(
                minimum=10, maximum=50, value=25, step=1,
                label="Inference Steps",
            )
        with gr.Row():
            guidance_scale = gr.Slider(
                minimum=1.0, maximum=5.0, value=2.0, step=0.1,
                label="Guidance Scale",
            )
            noise_aug_strength = gr.Slider(
                minimum=0.0, maximum=1.0, value=0.0, step=0.05,
                label="Noise Augmentation",
            )
            sample_stride = gr.Slider(
                minimum=1, maximum=4, value=2, step=1,
                label="Sample Stride",
            )
        with gr.Row():
            fps_slider = gr.Slider(
                minimum=7, maximum=30, value=15, step=1,
                label="Output FPS",
            )
            seed = gr.Number(value=42, label="Seed", precision=0)
            frames_overlap = gr.Slider(
                minimum=2, maximum=12, value=6, step=1,
                label="Frames Overlap",
            )

    generate_btn = gr.Button("Generate", variant="primary", size="lg")

    with gr.Accordion("Console Log", open=False):
        log_output = gr.Textbox(
            label="Logs",
            interactive=False,
            lines=15,
            max_lines=50,
        )

    generate_btn.click(
        fn=generate,
        inputs=[
            image_input,
            video_input,
            start_time,
            end_time,
            auto_chunk,
            num_frames,
            resolution,
            steps,
            guidance_scale,
            noise_aug_strength,
            sample_stride,
            fps_slider,
            seed,
            frames_overlap,
        ],
        outputs=[output_video, frame_gallery, log_output, status_text],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, allowed_paths=["/workspace"])
