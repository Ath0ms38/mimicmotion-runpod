"""
mimicmotion_patch.py - Monkey-patch MimicMotion pipeline for rich progress.

Replaces the diffusers DiffusionPipeline.progress_bar() context manager
with a custom one showing ETA and step counts. Also provides a Gradio-
compatible progress callback for UI updates.
"""

import sys
import time
import types
from contextlib import contextmanager

from tqdm import tqdm


class ProgressTracker:
    """Tracks progress across preprocessing and denoising phases."""

    def __init__(self, gradio_progress=None):
        self.gradio_progress = gradio_progress
        self.phase = ""
        self.total_steps = 0
        self.current_step = 0
        self.start_time = None
        self.logs = []

    def log(self, msg):
        self.logs.append(msg)
        print(msg, flush=True)

    def get_logs(self):
        return "\n".join(self.logs)


def create_progress_bar_patch(tracker: ProgressTracker):
    """Create a patched progress_bar method for the pipeline."""

    @contextmanager
    def patched_progress_bar(self, total=None):
        tracker.phase = "Denoising"
        tracker.total_steps = total or 0
        tracker.current_step = 0
        tracker.start_time = time.time()
        tracker.log(f"Denoising: {total} total iterations (steps x tiles)")

        bar = tqdm(
            total=total,
            desc="Denoising",
            bar_format=(
                "{l_bar}{bar}| {n_fmt}/{total_fmt} "
                "[{elapsed}<{remaining}, {rate_fmt}]"
            ),
            dynamic_ncols=True,
            file=sys.stderr,
        )

        original_update = bar.update

        def tracked_update(n=1):
            tracker.current_step += n
            if tracker.gradio_progress is not None:
                progress_frac = tracker.current_step / tracker.total_steps
                elapsed = time.time() - tracker.start_time
                if tracker.current_step > 0:
                    eta = elapsed / tracker.current_step * (
                        tracker.total_steps - tracker.current_step
                    )
                    eta_str = _format_time(eta)
                else:
                    eta_str = "?"
                tracker.gradio_progress(
                    progress_frac,
                    desc=f"Denoising {tracker.current_step}/{tracker.total_steps} "
                    f"(ETA: {eta_str})",
                )
            return original_update(n)

        bar.update = tracked_update

        try:
            yield bar
        finally:
            bar.close()
            elapsed = time.time() - tracker.start_time
            tracker.log(f"Denoising completed in {_format_time(elapsed)}")

    return patched_progress_bar


def create_step_callback(tracker: ProgressTracker):
    """Create a callback_on_step_end function for per-timestep logging."""

    def step_callback(pipeline, step_index, timestep, callback_kwargs):
        tracker.log(
            f"  Step {step_index + 1}/{pipeline._num_timesteps} "
            f"(timestep={timestep:.1f})"
        )
        return callback_kwargs

    return step_callback


def apply_patches(pipeline, gradio_progress=None):
    """Apply progress bar patches to a MimicMotionPipeline instance.

    Args:
        pipeline: MimicMotionPipeline instance
        gradio_progress: Optional Gradio progress callback (gr.Progress)

    Returns:
        ProgressTracker instance for monitoring progress
    """
    tracker = ProgressTracker(gradio_progress=gradio_progress)

    patched = create_progress_bar_patch(tracker)
    pipeline.progress_bar = types.MethodType(patched, pipeline)

    return tracker


def _format_time(seconds):
    """Format seconds into mm:ss or hh:mm:ss."""
    seconds = int(seconds)
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60}s"
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours}h {minutes}m {secs}s"
