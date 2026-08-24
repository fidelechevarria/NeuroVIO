# Multi-stage Dockerfile for NeuroVIO
# Stage 1: Build base environment with LLVM/Clang 18, GTSAM, OpenCV, and C++ dependencies
FROM ubuntu:24.04 AS dev-base

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install essential system tools and compilers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    git \
    wget \
    curl \
    pkg-config \
    ca-certificates \
    gnupg \
    lsb-release \
    software-properties-common \
    clang-18 \
    clang-tools-18 \
    clang-format-18 \
    clang-tidy-18 \
    lldb-18 \
    lld-18 \
    libomp-18-dev \
    libeigen3-dev \
    libopencv-dev \
    libboost-all-dev \
    libtbb-dev \
    libfmt-dev \
    libspdlog-dev \
    libgoogle-glog-dev \
    python3 \
    python3-pip \
    python3-dev \
    python3-numpy \
    && rm -rf /var/lib/apt/lists/*

# Set Clang 18 as default compiler
RUN update-alternatives --install /usr/bin/clang clang /usr/bin/clang-18 100 && \
    update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-18 100 && \
    update-alternatives --install /usr/bin/lld lld /usr/bin/lld-18 100 && \
    update-alternatives --install /usr/bin/clang-format clang-format /usr/bin/clang-format-18 100 && \
    update-alternatives --install /usr/bin/clang-tidy clang-tidy /usr/bin/clang-tidy-18 100

# Install GTSAM 4.2+
WORKDIR /opt
RUN git clone --depth 1 --branch 4.2.0 https://github.com/borglab/gtsam.git && \
    cmake -B gtsam/build -S gtsam -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=clang \
      -DCMAKE_CXX_COMPILER=clang++ \
      -DGTSAM_USE_SYSTEM_EIGEN=ON \
      -DGTSAM_BUILD_TESTS=OFF \
      -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF \
      -DGTSAM_BUILD_UNSTABLE=ON \
      -DGTSAM_ROT3_EXPMAP=ON \
      -DGTSAM_ENABLE_BOOST_SERIALIZATION=OFF && \
    cmake --build gtsam/build --target install && \
    rm -rf /opt/gtsam

# Install evo trajectory evaluation package
RUN pip3 install --break-system-packages --no-cache-dir evo

WORKDIR /workspace
ENV CC=clang
ENV CXX=clang++

CMD ["/bin/bash"]
