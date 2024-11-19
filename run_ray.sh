#!/bin/bash
cd "$(dirname "$0")"  # Change to script directory

# Build the library
./build_lib.sh

# Get Ray worker IPs from Ray's internal state
worker_ips=$(ray status | grep "NodeName:" | awk '{print $2}' | cut -d@ -f2 | cut -d: -f1)

# Create directory and copy library to all nodes
for ip in $worker_ips; do
    echo "Copying library to $ip..."
    ssh ray@$ip "mkdir -p /home/ray/solana-vanity-gpu/src/release"
    scp src/release/libcuda-ed25519-vanity.so ray@$ip:/home/ray/solana-vanity-gpu/src/release/
done

# Initialize Ray and run the program
export LD_LIBRARY_PATH=$(pwd)/src/release:$LD_LIBRARY_PATH
python ray_vanity.py