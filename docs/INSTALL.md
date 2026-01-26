# Build System Requirements

## System Dependencies

### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install -y \
    cmake \
    g++ \
    libeigen3-dev \
    python3-dev \
    python3-pip \
    git
```

### macOS
```bash
brew install cmake eigen python3
```

### Windows
- Install Visual Studio 2019 or later with C++ support
- Install CMake from https://cmake.org/download/
- Install Eigen3 via vcpkg: `vcpkg install eigen3`
- Install Python 3.8+ from https://www.python.org/

## Python Environment

Recommended: Use a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Verification

Check that all dependencies are available:

```bash
# CMake
cmake --version  # Should be >= 3.15

# C++ compiler
g++ --version    # GCC >= 7 or Clang >= 5

# Eigen3
echo '#include <Eigen/Dense>' | g++ -x c++ -c - -I/usr/include/eigen3

# Python
python --version  # Should be >= 3.8

# Python packages
python -c "import numpy, scipy, pandas, pybind11; print('OK')"
```

## Optional: OpenMP

For parallel execution, ensure OpenMP is available:

```bash
echo '#include <omp.h>' | g++ -fopenmp -x c++ -c -

# Test OpenMP
echo 'int main() { return omp_get_max_threads(); }' | \
  g++ -fopenmp -x c++ - -o /tmp/test_omp && /tmp/test_omp
```

## Troubleshooting

### Eigen3 not found
If CMake cannot find Eigen3, specify the path manually:
```bash
cmake .. -DEigen3_DIR=/usr/share/eigen3/cmake
```

### pybind11 not found
Install pybind11 development files:
```bash
pip install "pybind11[global]"
```

### OpenMP not available
Disable OpenMP in CMake:
```bash
cmake .. -DENABLE_OPENMP=OFF
```
