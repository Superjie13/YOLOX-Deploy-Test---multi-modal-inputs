####################################################################################################################################################
# This Dockerfile is used to build an image containing necessary dependencies to develop and run mmdeploy 
# (only for the usage of converting PyTorch model to ONNX model)
# @Author: Sijie Hu
# @Reference:
#   -https://github.com/open-mmlab/mmdeploy/blob/main/docker/Base/Dockerfile\
####################################################################################################################################################

FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04


ARG TORCH_VERSION=1.13.1
ARG TORCHVISION_VERSION=0.14.1

# backends
ARG ONNXRUNTIME_VERSION=1.8.1

# flags
ARG INSTALL_ONNXRUNTIME=true
ARG INSTALL_TENSORRT=false

# tensorrt tar file url
ARG TENSORRT_URL

USER root
WORKDIR /root/workspace

ENV DEBIAN_FRONTEND=nointeractive
ENV FORCE_CUDA="1"

RUN apt-get update && apt-get install -y --no-install-recommends \
    apt-utils \
    ca-certificates \
    gcc-7 \
    g++-7 \
    git \
    vim \
    wget \
    libopencv-dev \
    libprotobuf-dev protobuf-compiler \
    unzip \
    python3-dev python3-pip \
    && rm -rf /var/lib/apt/lists/*

ENV CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# install onnxruntime, and other python packages
RUN wget https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz &&\
    tar -xzvf onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz && rm onnxruntime-*.tgz &&\
    wget https://github.com/Kitware/CMake/releases/download/v3.25.2/cmake-3.25.2-linux-x86_64.tar.gz &&\
    tar -xzvf cmake-3.25.2-linux-x86_64.tar.gz && rm cmake-*.tar.gz && mv cmake-* cmake &&\
    ln -sf $(pwd)/cmake/bin/* /usr/bin/
    
ENV ONNXRUNTIME_DIR=/root/workspace/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}
ENV LD_LIBRARY_PATH=${ONNXRUNTIME_DIR}/lib:$LD_LIBRARY_PATH