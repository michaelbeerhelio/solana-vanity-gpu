#!/bin/bash

# Clean previous builds
rm -f ./src/release/cuda_ed25519_vanity
rm -f ./src/release/ecc_scan.o
rm -f ./src/release/libcuda-ed25519-vanity.so

# Ensure CUDA is in PATH
export PATH=/usr/local/cuda/bin:$PATH

# Build original binary
make -j

# Build shared library for Ray
make cuda_ed25519_vanity_shared

# Verify builds
if [ ! -f ./src/release/cuda_ed25519_vanity ]; then
    echo "Error: Failed to build cuda_ed25519_vanity binary"
    exit 1
fi

if [ ! -f ./src/release/libcuda-ed25519-vanity.so ]; then
    echo "Error: Failed to build libcuda-ed25519-vanity.so shared library"
    exit 1
fi

echo "Build completed successfully!"