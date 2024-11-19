#!/bin/bash
cd "$(dirname "$0")"  # Change to script directory

# Build the library
./build_lib.sh

# Get Ray worker IPs using Python
worker_ips=$(python3 -c '
import ray
ray.init(address="auto")
nodes = ray.nodes()
ips = [node["NodeManagerAddress"] for node in nodes]
print("\n".join(ips))
ray.shutdown()
')

echo "Found worker IPs:"
echo "$worker_ips"

# Create directory and copy library to all nodes
for ip in $worker_ips; do
    echo "Copying library to $ip..."
    ssh -o StrictHostKeyChecking=no ray@$ip "mkdir -p /home/ray/solana-vanity-gpu/src/release"
    scp -o StrictHostKeyChecking=no src/release/libcuda-ed25519-vanity.so ray@$ip:/home/ray/solana-vanity-gpu/src/release/
done

# Run the program
export LD_LIBRARY_PATH=$(pwd)/src/release:$LD_LIBRARY_PATH
python ray_vanity.py