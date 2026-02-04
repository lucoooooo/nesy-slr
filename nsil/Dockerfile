FROM ubuntu:22.04

# Set timezone
RUN ln -snf /usr/share/zoneinfo/$CONTAINER_TIMEZONE /etc/localtime && echo $CONTAINER_TIMEZONE > /etc/timezone

# Install dependencies
RUN apt-get update
RUN apt-get -y install g++ build-essential cmake
RUN apt-get -y install flex bison
RUN apt-get -y install libboost-dev libboost-regex-dev libboost-program-options-dev
RUN apt-get -y install git wget gringo unzip software-properties-common curl
RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN apt-get -y install python3.8
RUN apt-get -y install python3.9-dev
RUN apt-get -y install python3-pip
RUN apt-get install -y libjpeg-dev zlib1g-dev
RUN ln -s /usr/bin/python3 /usr/bin/python

# Copy NSIL code
WORKDIR /opt/
RUN git clone https://github.com/DanCunnington/NSIL.git
#COPY ./NSIL ./opt/NSIL
WORKDIR /opt/NSIL

RUN touch /opt/NSIL/__init__.py /opt/NSIL/src/__init__.py

# Run NSIL setup
RUN pip3 install --upgrade pip setuptools wheel
#RUN pip install "notebook<7.0.0" "traitlets<5.9.0" "jupyter_client<8.0.0" "jupyter_server<2.0.0"
RUN pip3 install -r requirements.txt
RUN pip3 install timm
RUN chmod +x ./download_data.sh
RUN ./download_data.sh
RUN mv paper_experiments/ijcai_2023/scripts/container_setup.sh paper_experiments/ijcai_2023/scripts/setup.sh
RUN cp paper_experiments/ijcai_2023/scripts/setup.sh paper_experiments/ijcai_2023/scripts/naive_baselines/setup.sh
RUN find paper_experiments/ijcai_2023/scripts -type f -exec chmod +x {} \;
RUN find paper_experiments/ijcai_2023/scripts/naive_baselines -type f -exec chmod +x {} \;
RUN mv LAS_binaries/ILASP_ubuntu ILASP
RUN mv LAS_binaries/FastLAS_ubuntu FastLAS
ENV PYTHONPATH="/opt/:/opt/NSIL:/opt/NSIL/src/"

# Start web servers
WORKDIR /workspace
CMD ["bash"]
