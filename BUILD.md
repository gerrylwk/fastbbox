# Building FastBBox

This document describes how to build the FastBBox package with either Cython or nanobind backends.

## Overview

FastBBox supports two build systems:

1. **Cython (Stable)**: Mature, battle-tested implementation using Cython
   - Package name: `fastbbox`
   - Python support: 3.8+
   - Build system: setuptools + Cython

2. **Nanobind (Experimental)**: Modern, optimized implementation using nanobind
   - Package name: `fastbbox-nanobind`
   - Python support: 3.9+
   - Build system: CMake + scikit-build-core

Both implementations provide identical Python APIs and can be used interchangeably.

## Prerequisites

### Common Requirements
- Python 3.8+ (Cython) or 3.9+ (nanobind)
- NumPy >= 1.19.0
- C++ compiler (MSVC on Windows, GCC/Clang on Linux/macOS)

### Cython Build Requirements
```bash
pip install setuptools wheel cython numpy
```

### Nanobind Build Requirements
```bash
pip install scikit-build-core nanobind cmake numpy
```

You'll also need CMake 3.15+ installed on your system:
- **Windows**: Download from https://cmake.org/download/
- **Linux**: `sudo apt-get install cmake` or `sudo yum install cmake`
- **macOS**: `brew install cmake`

## Local Development Builds

### Building with Cython (Stable)

**Editable install** (recommended for development):
```bash
pip install -e .
```

**Standard install**:
```bash
pip install .
```

**Build wheel**:
```bash
pip install build
python -m build
```
Wheels will be in `dist/` directory.

### Building with Nanobind (Experimental)

**Editable install**:
```bash
pip install -e . -C--config-file=pyproject.nanobind.toml
```

**Standard install**:
```bash
pip install . -C--config-file=pyproject.nanobind.toml
```

**Build wheel**:
```bash
python -m build -C--config-file=pyproject.nanobind.toml
```

### Manual CMake Build (nanobind only)

For low-level development and debugging:

```bash
# Create build directory
mkdir build
cd build

# Configure
cmake ..

# Build
cmake --build . --config Release

# The built modules will be in the build directory
# You can copy them to fastbbox/ for testing
```

On Windows, you may need to specify the generator:
```bash
cmake .. -G "Visual Studio 17 2022"
```

## Testing Your Build

After building either version, verify it works:

```bash
# Check which backend is loaded
python -c "import fastbbox; print(f'Backend: {fastbbox.__backend__}')"

# Run basic functionality test
python -c "
import fastbbox
import numpy as np

boxes = np.array([[0, 0, 10, 10], [5, 5, 15, 15]], dtype=np.float32)
iou = fastbbox.bbox_overlaps(boxes, boxes)
print(f'IoU shape: {iou.shape}')
print(f'IoU diagonal: {iou.diagonal()}')
assert iou[0, 0] > 0.99, 'Self-IoU should be ~1.0'
print('✓ Basic test passed')
"

# Run full test suite
python test_all_iou.py
python test_obb_iou.py

# Run benchmarks
python benchmark_comparison.py
```

## CI/CD Wheel Building

### GitHub Actions Workflows

Two workflows build wheels automatically:

1. **`.github/workflows/build.yml`**: Builds Cython wheels (stable)
   - Triggered on: push to main, tags matching `v*`, pull requests
   - Produces: `fastbbox` wheels for Python 3.8-3.12

2. **`.github/workflows/build-nanobind.yml`**: Builds nanobind wheels (experimental)
   - Triggered on: push to main, tags matching `v*-nb`, pull requests
   - Produces: `fastbbox-nanobind` wheels for Python 3.9-3.12

### Building Wheels Locally with cibuildwheel

To build wheels for multiple Python versions locally:

**Cython**:
```bash
pip install cibuildwheel
cibuildwheel --platform linux  # or windows, macos
```

**Nanobind**:
```bash
CIBW_CONFIG_SETTINGS="--global-option=--config-file=pyproject.nanobind.toml" cibuildwheel --platform linux
```

## Comparing Performance

To compare Cython vs nanobind performance:

1. Build and install both versions in separate virtual environments:

```bash
# Cython version
python -m venv venv-cython
source venv-cython/bin/activate  # or venv-cython\Scripts\activate on Windows
pip install .
python benchmark_comparison.py > results-cython.txt
deactivate

# Nanobind version
python -m venv venv-nanobind
source venv-nanobind/bin/activate
pip install . -C--config-file=pyproject.nanobind.toml
python benchmark_comparison.py > results-nanobind.txt
deactivate
```

2. Compare the results files

## Troubleshooting

### Cython Build Issues

**"cython: command not found"**
```bash
pip install cython
```

**"numpy/arrayobject.h: No such file or directory"**
```bash
pip install numpy
```

### Nanobind Build Issues

**"nanobind not found"**
```bash
pip install nanobind
```

**"CMake not found"**
- Install CMake from your package manager or https://cmake.org/download/

**"No CMAKE_CXX_COMPILER could be found"**
- Install a C++ compiler:
  - **Windows**: Install Visual Studio 2019 or newer with C++ tools
  - **Linux**: `sudo apt-get install g++` or `sudo yum install gcc-c++`
  - **macOS**: `xcode-select --install`

**Import error: "module 'fastbbox.bbox' has no attribute 'bbox_overlaps'"**
- Make sure CMake installed the modules: check for `bbox.*.so` or `bbox.*.pyd` in `fastbbox/`
- Try rebuilding with: `pip install --force-reinstall --no-deps .`

**Tests fail with backend detection**
- This is expected if module detection fails
- The functionality should still work; only the `__backend__` attribute may be incorrect

## Package Distribution

### Publishing to PyPI

**Cython version** (tag with `v*`):
```bash
git tag v0.1.0
git push origin v0.1.0
```

**Nanobind version** (tag with `v*-nb`):
```bash
git tag v0.1.0-nb
git push origin v0.1.0-nb
```

The GitHub Actions workflows will automatically build and publish to PyPI.

### Manual PyPI Upload

```bash
pip install twine

# Upload Cython build
python -m build
twine upload dist/*

# Upload nanobind build
python -m build -C--config-file=pyproject.nanobind.toml
twine upload dist/*
```

## Source File Reference

### Cython Implementation
- `fastbbox/bbox.pyx` - Axis-aligned bounding box operations
- `fastbbox/obb_bbox.pyx` - Oriented bounding box operations
- `setup.py` - Build configuration
- `pyproject.toml` - Package metadata and cibuildwheel config

### Nanobind Implementation
- `fastbbox/bbox_nb.cpp` - Axis-aligned bounding box operations
- `fastbbox/obb_bbox_nb.cpp` - Oriented bounding box operations
- `CMakeLists.txt` - CMake build configuration
- `pyproject.nanobind.toml` - Package metadata and scikit-build-core config

### Shared Files
- `fastbbox/__init__.py` - Package initialization (works with both)
- `test_all_iou.py` - Test suite for axis-aligned boxes
- `test_obb_iou.py` - Test suite for oriented boxes
- `benchmark_comparison.py` - Performance benchmarks
