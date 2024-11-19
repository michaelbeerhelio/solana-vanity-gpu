#!/bin/bash
cd "$(dirname "$0")"  # Change to script directory

# Build the library
./build_lib.sh

# Run the program
export LD_LIBRARY_PATH=$(pwd)/src/release:$LD_LIBRARY_PATH
python ray_vanity.py