FROM python:3.9-slim

ARG DEBIAN_FRONTEND=noninteractive

# Pacchetti base
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential curl wget pkg-config python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir torch torchvision torchaudio jupyter matplotlib pandas numpy tqdm timm scikit-learn \
    && pip install --no-cache-dir deepproblog

WORKDIR /workspace
CMD ["bash"]
