#!/bin/bash
cd "$(dirname "$0")"  # Change to script directory
export LD_LIBRARY_PATH=$(pwd)/src/release:$LD_LIBRARY_PATH

# Ensure library is built
./build_lib.sh

python ray_vanity.py