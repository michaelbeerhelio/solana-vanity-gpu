#!/bin/bash
set -e  # Exit on error

# Clean previous builds
rm -f ./src/release/cuda_ed25519_vanity
rm -f ./src/release/ecc_scan.o
rm -f ./src/release/libcuda-ed25519-vanity.so

# Ensure CUDA is in PATH
if ! command -v nvcc &> /dev/null; then
    echo "CUDA not found, adding to PATH"
    export PATH=/usr/local/cuda/bin:$PATH
fi

# Check CUDA is available
if ! command -v nvcc &> /dev/null; then
    echo "Error: CUDA (nvcc) not found in PATH"
    exit 1
fi

# Print CUDA version
nvcc --version

# Build original binary
make -j

# Build shared library
make cuda_ed25519_vanity_shared

echo "Build completed successfully!"