FROM python:3.9-slim

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential curl wget pkg-config python3-dev vim\
    && rm -rf /var/lib/apt/lists/*

COPY ./NeurASP ./opt/neurasp
WORKDIR /opt/neurasp

RUN pip install --upgrade pip
RUN pip install clingo
RUN pip install torch torchaudio torchvision numpy jupyter matplotlib pandas numpy tqdm timm scikit-learn 

ENV PYTHONPATH="/opt/neurasp:/opt/neurasp/:${PYTHONPATH:-}"

WORKDIR /workspace
CMD ["bash"]

