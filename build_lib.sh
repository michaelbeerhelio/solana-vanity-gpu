#!/bin/bash
export PATH=/usr/local/cuda/bin:$PATH

# Build original project
make -j$(nproc)

# Build shared library for Ray
nvcc -Xcompiler -fPIC -shared \
    -o src/release/libcuda-ed25519-vanity.so \
    src/cuda-ecc-ed25519/vanity.cu \
    -I./src/cuda-headers \
    -I./src/cuda-sha256 \
    --gpu-architecture=compute_89 \
    --gpu-code=sm_89,compute_89