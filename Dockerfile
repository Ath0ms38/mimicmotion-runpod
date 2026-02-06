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

# Copy local MimicMotion repo (with loader.py fix)
COPY MimicMotion/ /app/MimicMotion/

# Copy dependency spec and install
COPY pyproject.toml /app/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-install-project

# Copy application code
COPY run.py download_models.py mimicmotion_patch.py app.py start.sh /app/

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
ENV PYTHONPATH="/app/MimicMotion"
ENV MIMICMOTION_DIR="/app/MimicMotion"
ENV MODELS_DIR="/workspace/models"
ENV MIMICMOTION_MODELS_DIR="/app/MimicMotion/models"
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
ENV PYTHONUNBUFFERED=1

# Gradio UI port
EXPOSE 7860

# Persistent volume mount point (RunPod convention)
RUN mkdir -p /workspace

# DWPose uses relative path "models/DWPose/" from workdir /app
RUN ln -s /app/MimicMotion/models /app/models

RUN chmod +x /app/start.sh

CMD ["/app/start.sh"]
