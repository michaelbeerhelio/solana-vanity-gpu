#!/bin/bash
cd "$(dirname "$0")"  # Change to script directory

# Build the library
./build_lib.sh

# Create a clean directory structure for Ray
rm -rf ./ray_package
mkdir -p ./ray_package/src/release
cp src/release/libcuda-ed25519-vanity.so ./ray_package/src/release/
cp ray_vanity.py ./ray_package/

# Run from the package directory
cd ray_package
python ray_vanity.py