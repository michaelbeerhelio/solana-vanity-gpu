#!/bin/bash
cd "$(dirname "$0")"  # Change to script directory

# Copy library to all Ray nodes
for node in $(ray get-worker-ips); do
    ssh $node "mkdir -p /home/ray/solana-vanity-gpu/src/release"
    scp src/release/libcuda-ed25519-vanity.so $node:/home/ray/solana-vanity-gpu/src/release/
done

export LD_LIBRARY_PATH=$(pwd)/src/release:$LD_LIBRARY_PATH
python ray_vanity.py