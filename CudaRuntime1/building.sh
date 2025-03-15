#!/bin/bash
cd ~/Desktop/GOST/GOST/CudaRuntime
rm -rf ./_build
cmake -S . -B _build
cmake --build _build
