FROM python:3.6-slim

ARG DEBIAN_FRONTEND=noninteractive

# Pacchetti base
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential curl wget pkg-config python3-dev vim\
    && rm -rf /var/lib/apt/lists/*

COPY ./SLASH /opt/SLASH
WORKDIR /opt/SLASH

RUN pip install --upgrade pip
RUN python3 -m pip install torch torchaudio torchvision numpy jupyter matplotlib pandas numpy tqdm timm scikit-learn 
RUN python3 -m pip install --ignore-installed -r requirements.txt

ENV PYTHONPATH="/opt/SLASH:/opt/SLASH/src/SLASH:${PYTHONPATH:-}"

WORKDIR /workspace
CMD ["bash"]
