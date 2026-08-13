# see-through (upstream PyTorch) — linux/amd64 image for RunPod Serverless.
#
# Fork-local file: not present upstream. Kept minimal so `gh repo sync` and
# upstream merges stay low-friction.
#
# Torch pin comes from requirements.txt's documented setup (2.8.0+cu128), so
# the CUDA base is 12.8 to match. Weights are NOT baked in -- mount a RunPod
# network volume at /models.

FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.12 python3.12-venv python3-pip git ca-certificates \
      libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv
RUN python3.12 -m venv "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /app

# Torch first, from the cu128 index, exactly as requirements.txt documents.
# Separate layer so the (large) torch download is cached across app changes.
RUN pip install --no-cache-dir \
      torch==2.8.0+cu128 torchvision==0.23.0+cu128 torchaudio==2.8.0+cu128 \
      --index-url https://download.pytorch.org/whl/cu128

# requirements.txt has `-e ./common` and `-e ./annotators`, so the source
# tree must be present before it is installed.
COPY . .
RUN pip install --no-cache-dir -r requirements.txt

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV SEETHROUGH_MODELS=/models
VOLUME ["/models"]

# Variants available under inference/scripts/:
#   inference_psd.py             baseline
#   inference_psd_blockswap.py   CUDA VRAM offload, for <24GB cards
#   inference_psd_quantized.py   quantized
ENTRYPOINT ["python", "inference/scripts/inference_psd.py"]
CMD ["--help"]
