FROM python:3.10-slim

ARG DEBIAN_FRONTEND=noninteractive

# Pacchetti base
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential curl wget pkg-config vim python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip
RUN pip install "edward2 @ git+https://github.com/google/edward2.git" tensorflow tensorflow-probability tf-keras
RUN pip install --no-cache-dir jupyter matplotlib pandas numpy tqdm tenacity


WORKDIR /workspace
CMD ["bash"]
