#!/bin/bash
set -ex  # Exit on error and print commands

export PATH=/usr/local/cuda/bin:$PATH

# Clean previous builds
rm -f ./src/release/cuda_ed25519_vanity
rm -f ./src/release/ecc_scan.o
rm -f ./src/release/libcuda-ed25519-vanity.so

# Build shared library for Ray
nvcc -Xcompiler -fPIC -shared \
    -o src/release/libcuda-ed25519-vanity.so \
    src/cuda-ecc-ed25519/vanity.cu \
    -I./src/cuda-headers \
    -I./src/cuda-sha256 \
    --gpu-architecture=compute_89 \
    --gpu-code=sm_89,compute_89 \
    -DENDIAN_NEUTRAL -DLTC_NO_ASM

# Verify library was created
ls -l src/release/libcuda-ed25519-vanity.so