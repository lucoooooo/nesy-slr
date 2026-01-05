FROM python:3.9-slim

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    git swi-prolog build-essential python3-dev vim \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt
COPY ./deepstochlog /opt/deepstochlog
WORKDIR /opt/deepstochlog

RUN pip install --no-cache-dir torch torchvision torchaudio jupyter matplotlib pandas numpy tqdm timm scikit-learn 

RUN pip install --no-cache-dir -e .

WORKDIR /workspace
CMD ["bash"]
