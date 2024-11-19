#!/bin/bash
export LD_LIBRARY_PATH=./src/release:$LD_LIBRARY_PATH
python ray_vanity.py