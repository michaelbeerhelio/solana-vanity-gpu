#!/bin/bash
cd "$(dirname "$0")"  # Change to script directory
export LD_LIBRARY_PATH=$(pwd)/src/release:$LD_LIBRARY_PATH
python ray_vanity.py