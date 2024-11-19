#!/bin/bash
cd "$(dirname "$0")"  # Change to script directory

# Build the library
./build_lib.sh

# Copy to all Ray nodes using ray rsync_files
ray rsync_files src/release/libcuda-ed25519-vanity.so /home/ray/solana-vanity-gpu/src/release/

export LD_LIBRARY_PATH=/tmp:$(pwd)/src/release:$LD_LIBRARY_PATH
python ray_vanity.py