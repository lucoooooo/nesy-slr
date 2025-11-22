FROM python:3.9-slim

ARG DEBIAN_FRONTEND=noninteractive

# Pacchetti base
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential curl wget pkg-config python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir torch torchvision torchaudio numpy scikit-learn tqdm timm matplotlib pandas 


RUN pip install --no-cache-dir LTNtorch

RUN pip install --upgrade pip

WORKDIR /workspace
CMD ["bash"]

