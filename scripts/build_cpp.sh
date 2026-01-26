#!/bin/bash
# Build script for C++ entropy engine

set -e

echo "=== Building Entropy C++ Engine ==="

# Get project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build"

# Clean previous build
if [ -d "$BUILD_DIR" ]; then
    echo "Cleaning previous build..."
    rm -rf "$BUILD_DIR"
fi

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure with CMake
echo "Configuring with CMake..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTS=ON \
    -DENABLE_OPENMP=ON

# Build
echo "Building..."
cmake --build . -j$(nproc)

# Run tests
echo ""
echo "=== Running Tests ==="
ctest --output-on-failure

echo ""
echo "✓ Build successful!"
echo "Python module location: ${PROJECT_ROOT}/src/python/entropy_cpp*.so"
