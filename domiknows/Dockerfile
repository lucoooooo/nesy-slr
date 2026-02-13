FROM python:3.12-slim

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential curl wget pkg-config vim python3-dev procps\
    && rm -rf /var/lib/apt/lists/*

#WORKDIR /opt
#RUN git clone https://github.com/HLR/DomiKnowS.git domiknows

#COPY ./api_key/gurobi.lic /opt/gurobi/gurobi.lic
RUN python3 -m pip install --upgrade pip
RUN pip install --no-cache-dir torch torchvision torchaudio jupyter matplotlib pandas numpy tqdm timm scikit-learn datasets gurobipy spacy domiknows

WORKDIR /workspace
CMD ["bash"]
