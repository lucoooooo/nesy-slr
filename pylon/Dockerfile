FROM python:3.9-slim

ARG DEBIAN_FRONTEND=noninteractive

# Pacchetti base
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential curl wget pkg-config vim python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY ./pylon /app/pylon
WORKDIR /app/pylon

RUN python3 -m pip install --upgrade pip
RUN pip install flake8 pytest
RUN pip install --no-cache-dir torch torchvision torchaudio jupyter matplotlib pandas numpy tqdm timm scikit-learn datasets transformers pySDD pulp astor
RUN pip install -e .
#RUN pip install -r requirements.txt --no-deps


WORKDIR /workspace
CMD ["bash"]
