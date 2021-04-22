FROM nvidia/cuda:11.1-cudnn8-devel-ubuntu18.04

LABEL maintainer="K. Li"

# 1. Install basic packages
RUN apt-get update --fix-missing && apt-get upgrade -y && \
    apt-get install -y wget bzip2 ca-certificates curl git python3 python3-pip libboost-all-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 2. Install pytorch
RUN pip3 install -U pip && pip install numpy && \
    pip install --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/cu110/torch_nightly.html

# 3. Install cmake
RUN wget --quiet https://github.com/Kitware/CMake/releases/download/v3.19.1/cmake-3.19.1-Linux-x86_64.sh -O ~/cmake.sh && \
    mkdir /opt/cmake && /bin/sh ~/cmake.sh --skip-license --prefix=/opt/cmake

ENV PATH /opt/cmake/bin:$PATH

# 4. Install spconv
RUN cd /root && \
    git clone --depth 1 --recursive https://www.github.com/traveller59/spconv.git && \
    cd ./spconv && \
    SPCONV_FORCE_BUILD_CUDA=1 python3 setup.py install

# 5. Install pcdet
WORKDIR /code
COPY . /code/
RUN pip install -r requirements.txt && python3 setup.py develop

WORKDIR /src
