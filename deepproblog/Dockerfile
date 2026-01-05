FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive

# Pacchetti base
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential curl wget pkg-config python3 python3-pip python3-dev swi-prolog=8.4* swi-prolog-nox=8.4* swi-prolog-x=8.4* \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

RUN pip install --no-cache-dir torch torchvision torchaudio jupyter matplotlib pandas numpy tqdm timm scikit-learn \
    && pip install --no-cache-dir deepproblog && pip install git+https://github.com/ML-KULeuven/pyswip
# pip install --no-cache-dir deepproblog && pip install git+https://github.com/ML-KULeuven/pyswip && apt-get install swi-prolog=8.4* swi-prolog-nox=8.4* swi-prolog-x=8.4* && pip install sympy==1.12.0
WORKDIR /workspace
CMD ["bash"]
