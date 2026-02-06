#!/bin/bash
set -e
cd output
rm -rf *
cd ..
cd build
rm -rf *
cmake ..
make -j$(nproc)
cd ..
g++ main.cpp -o pipeline_test     -I./include     -L./build/lib -lAickTensorrt     `pkg-config --cflags --libs opencv4`     -Wl,-rpath,./build/lib
[ -f "./test.jpg" ] && ./pipeline_test ./test.jpg || echo "Build Done."
