#!/bin/bash
cd "$(dirname "$0")"  # Change to script directory

# Build the library
./build_lib.sh

# Create directory on all nodes
for node in $(ray get-worker-ips); do
    ssh ray@$node "mkdir -p /home/ray/solana-vanity-gpu/src/release"
    scp src/release/libcuda-ed25519-vanity.so ray@$node:/home/ray/solana-vanity-gpu/src/release/
done

export LD_LIBRARY_PATH=$(pwd)/src/release:$LD_LIBRARY_PATH
python ray_vanity.py