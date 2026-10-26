FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PIP_BREAK_SYSTEM_PACKAGES=1
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip python3-dev ffmpeg libsndfile1 libgl1 libglib2.0-0 git \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
RUN pip install torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
COPY requirements.txt .
RUN pip install -r requirements.txt && pip install "onnxruntime-gpu[cuda,cudnn]"
COPY pyproject.toml .
COPY src src
COPY configs configs
COPY scripts scripts
RUN pip install -e .
CMD ["python3", "scripts/run_pipeline.py"]
