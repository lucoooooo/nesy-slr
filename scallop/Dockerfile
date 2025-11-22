FROM python:3.9-slim

ARG DEBIAN_FRONTEND=noninteractive


RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates git build-essential pkg-config python3-dev \
    && rm -rf /var/lib/apt/lists/*


#RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
#    && /root/.cargo/bin/rustup default nightly
#ENV PATH="/root/.cargo/bin:${PATH}"


#WORKDIR /opt
#RUN git clone --depth 1 https://github.com/scallop-lang/scallop.git
#WORKDIR /opt/scallop

#RUN pip install --no-cache-dir scallopy@git+https://github.com/scallop-lang/scallopy.git

#RUN make install-scli && make install-sclc && make install-sclrepl

#RUN pip install --no-cache-dir maturin && make install-scallopy 

RUN apt update && apt upgrade -y
RUN apt-get -y install git

RUN apt update && apt upgrade -y
RUN apt-get -y install curl

RUN apt update && apt upgrade -y
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
RUN /bin/bash -c "source '$HOME/.cargo/env' ; rustup default nightly"

RUN apt update && apt upgrade -y
RUN apt install -y make

RUN apt update && apt upgrade -y
RUN apt install -y build-essential


WORKDIR /root
RUN git clone https://github.com/scallop-lang/scallop.git
WORKDIR /root/scallop

ENV SCALLOPDIR=/root/scallop


RUN /bin/bash -c "source '$HOME/.cargo/env' ; make install-scli"
RUN /bin/bash -c "source '$HOME/.cargo/env' ; make install-sclc"
RUN /bin/bash -c "source '$HOME/.cargo/env' ; make install-sclrepl"


RUN pip install maturin
RUN /bin/bash -c "source '$HOME/.cargo/env' ; make install-scallopy"

RUN pip install --no-cache-dir torch torchvision torchaudio jupyter matplotlib pandas numpy tqdm timm scikit-learn


WORKDIR /workspace
CMD ["bash"]

