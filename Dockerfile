# Base image with nano-vllm's dependencies (no source code).
# Build and push:
#   docker build --platform linux/amd64 -t ghcr.io/<user>/nanovllm:latest .
#   docker push ghcr.io/<user>/nanovllm:latest
# Run locally with the repo mounted:
#   docker run --gpus all -v $PWD:/workspace -w /workspace ghcr.io/<user>/nanovllm:latest python example.py
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-dev python3-pip git build-essential \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip \
    && python -m pip install torch==2.6.0 triton "transformers>=4.56" xxhash huggingface_hub ninja packaging wheel

RUN python -m pip install flash-attn==2.7.4.post1 --no-build-isolation
