#!/bin/bash

echo "=== TESTING DIFFERENT COMPILER OPTIMIZATION LEVELS ==="
echo ""

echo "Building with -O2:"
make clean
make CXXFLAGS="-std=c++11 -O2 -march=native -mtune=native -Wall -Wextra -Wpedantic -I. -Icommon -mcpu=apple-m1 -ffast-math -pthread" flash-compare

echo ""
echo "Building with -O3 (default):"
make clean
make flash-compare

echo ""
echo "Building with -O3 -funroll-loops:"
make clean
make CXXFLAGS="-std=c++11 -O3 -march=native -mtune=native -Wall -Wextra -Wpedantic -I. -Icommon -mcpu=apple-m1 -ffast-math -funroll-loops -pthread" flash-compare