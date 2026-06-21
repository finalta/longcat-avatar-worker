# ── Base image ─────────────────────────────────────────────────────────────
# CUDA 12.4 + Python 3.10 — matches LongCat's torch==2.6.0+cu124 requirement
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# ── System deps ────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    ffmpeg \
    git \
    wget \
    curl \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3

# ── PyTorch (CUDA 12.4) ────────────────────────────────────────────────────
RUN pip install --no-cache-dir \
    torch==2.6.0+cu124 \
    torchvision==0.21.0+cu124 \
    torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

# ── Flash Attention 2 (must come after torch) ─────────────────────────────
RUN pip install --no-cache-dir ninja psutil packaging && \
    pip install --no-cache-dir flash_attn==2.7.4.post1 \
    --no-build-isolation

# ── HuggingFace transfer for fast weight downloads ────────────────────────
RUN pip install --no-cache-dir "huggingface_hub[hf_transfer,cli]"

# ── Clone LongCat-Video repo ───────────────────────────────────────────────
RUN git clone --single-branch --branch main \
    https://github.com/meituan-longcat/LongCat-Video.git \
    /LongCat-Video

WORKDIR /LongCat-Video

# ── LongCat base requirements ─────────────────────────────────────────────
RUN pip install --no-cache-dir -r requirements.txt

# ── LongCat avatar requirements ────────────────────────────────────────────
RUN pip install --no-cache-dir -r requirements_avatar.txt

# ── librosa via pip (no conda in Docker) ──────────────────────────────────
RUN pip install --no-cache-dir librosa soundfile

# ── RunPod + AWS (for R2 uploads) ─────────────────────────────────────────
RUN pip install --no-cache-dir runpod boto3

# ── Copy handler ──────────────────────────────────────────────────────────
COPY handler.py /handler.py
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# NOTE: Model weights are NOT baked into this image.
# Mount a RunPod Network Volume at /runpod-volume with:
#   /runpod-volume/weights/LongCat-Video/          ← base model
#   /runpod-volume/weights/LongCat-Video-Avatar-1.5/ ← avatar model
#
# To download weights onto the volume (run once on a pod):
#   huggingface-cli download meituan-longcat/LongCat-Video \
#       --local-dir /runpod-volume/weights/LongCat-Video
#   huggingface-cli download meituan-longcat/LongCat-Video-Avatar-1.5 \
#       --local-dir /runpod-volume/weights/LongCat-Video-Avatar-1.5

CMD ["/entrypoint.sh"]
