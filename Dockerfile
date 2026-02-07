# ============================================================
# Stage 1: Build environment with UV
# ============================================================
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV UV_SYSTEM_PYTHON=1
ENV UV_COMPILE_BYTECODE=1
ENV UV_HTTP_TIMEOUT=300

# System deps for building Python packages with C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3.11-venv python3-pip \
    git wget curl \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Install UV
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# Clone StableAnimator and keep only source code (no assets/weights)
RUN git clone --depth 1 https://github.com/Francis-Rings/StableAnimator.git /tmp/StableAnimator \
    && mkdir -p /app/StableAnimator \
    && mv /tmp/StableAnimator/animation /app/StableAnimator/animation \
    && mv /tmp/StableAnimator/DWPose /app/StableAnimator/DWPose \
    && rm -rf /tmp/StableAnimator

# Copy dependency spec and install
COPY pyproject.toml /app/
RUN uv sync --no-install-project

# Copy application code
COPY run.py download_models.py start.sh /app/

# Bundle sample image and video
COPY IMG_8502.jpg caramell_dansen.mp4 /app/samples/

# ============================================================
# Stage 2: Runtime image
# ============================================================
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 \
    ffmpeg wget curl git \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Copy UV binary
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Copy built environment and application from builder
COPY --from=builder /app /app

WORKDIR /app

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/StableAnimator"
ENV MODELS_DIR="/workspace/models"
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
ENV PYTHONUNBUFFERED=1

# Persistent volume mount point (RunPod convention)
RUN mkdir -p /workspace

# Create directories for symlinks (populated at runtime by download_models.py)
RUN mkdir -p /app/checkpoints /app/models

RUN chmod +x /app/start.sh

CMD ["/app/start.sh"]
