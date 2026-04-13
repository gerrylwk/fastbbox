# FastBBox

Fast IoU/overlap computations for axis-aligned and oriented bounding boxes, powered by nanobind C++ extensions.

## Benchmark
```
Function Python (ms)    FastBBox (ms)  Speedup
--------------------------------------------------
IoU           1994.19          4.19     475.7x
GIoU          3028.40          5.99     505.8x
DIoU          4104.21          6.21     661.3x
CIoU          6436.51         17.55     366.7x
EIoU          5211.51          7.04     740.5x
NWD           2427.36         13.31     182.4x
OBB          21729.42         31.29     694.5x
--------------------------------------------------
TOTAL        44931.61         85.58     525.1x
```

## Installation

```bash
pip install fastbbox
```

### Building from Source

Requires CMake 3.15+ and a C++17 compiler.

```bash
git clone https://github.com/gerrylwk/fastbbox
cd fastbbox
pip install .
```

## Usage

### Axis-Aligned Bounding Boxes

```python
import numpy as np
from fastbbox import (bbox_overlaps, generalized_iou, distance_iou,
                      complete_iou, efficient_iou, normalized_wasserstein_distance)

# Bounding boxes in [x1, y1, x2, y2] format
boxes = np.array([
    [0, 0, 10, 10],
    [5, 5, 15, 15],
    [20, 20, 30, 30]
], dtype=np.float32)

query_boxes = np.array([
    [0, 0, 10, 10],
    [12, 12, 22, 22]
], dtype=np.float32)

iou = bbox_overlaps(boxes, query_boxes)
giou = generalized_iou(boxes, query_boxes)
diou = distance_iou(boxes, query_boxes)
ciou = complete_iou(boxes, query_boxes)
eiou = efficient_iou(boxes, query_boxes)
nwd = normalized_wasserstein_distance(boxes, query_boxes)
```

### Oriented Bounding Boxes (OBB)

```python
import numpy as np
import math
from fastbbox import bbox_overlaps_obb

# Oriented bounding boxes in [center_x, center_y, width, height, angle_radians] format
obb_boxes = np.array([
    [0, 0, 4, 2, 0],
    [1, 0, 4, 2, math.pi/4],
    [0, 0, 2, 2, math.pi/6],
], dtype=np.float32)

obb_query_boxes = np.array([
    [0, 0, 4, 2, 0],
    [2, 2, 2, 2, 0],
], dtype=np.float32)

obb_iou = bbox_overlaps_obb(obb_boxes, obb_query_boxes)
```

## Features

### Axis-Aligned Bounding Box IoU Variants

All variants accept boxes in `[x1, y1, x2, y2]` top-left-bottom-right format:

- **Standard IoU**: Classic Intersection over Union
- **Generalized IoU (GIoU)**: Considers enclosing area; values in `[-1, 1]`
  - Reference: [Generalized Intersection over Union](https://arxiv.org/abs/1902.09630)
- **Distance IoU (DIoU)**: Penalizes center point distance
  - Reference: [Distance-IoU Loss](https://arxiv.org/abs/1911.08287)
- **Complete IoU (CIoU)**: Adds aspect ratio consistency penalty to DIoU
  - Reference: [Distance-IoU Loss](https://arxiv.org/abs/1911.08287)
- **Efficient IoU (EIoU)**: Separate width/height penalties for faster convergence
  - Reference: [Focal and Efficient IOU Loss](https://arxiv.org/abs/2101.08158)
- **Normalized Wasserstein Distance (NWD)**: Optimal transport-based similarity, values in `[0, 1]`
  - Effective for tiny object detection
  - Reference: [Normalized Gaussian Wasserstein Distance](https://arxiv.org/abs/2110.13389)

### Oriented Bounding Box (OBB) IoU

Accepts boxes in `[center_x, center_y, width, height, angle_radians]` format:

- Exact calculation for axis-aligned boxes (angle = 0)
- Approximation-based calculation for rotated boxes using AABB intersection with angle-based scaling
- Batch processing support

## Performance

FastBBox provides significant speedup over pure Python implementations, especially for large numbers of bounding boxes.

Run `python benchmark_fastbbox.py` to benchmark on your system.

### When to Use Each Variant

- **IoU**: Standard evaluation, NMS (Non-Maximum Suppression)
- **GIoU**: Training loss for non-overlapping boxes
- **DIoU**: Object detection training where center distance matters
- **CIoU**: Bounding box regression with aspect ratio consistency
- **EIoU**: Fast convergence with separate width/height penalties
- **NWD**: Tiny object detection
- **OBB IoU**: Rotated object detection, text detection, aerial imagery

## OBB Implementation Details

The OBB IoU implementation uses different strategies based on box orientations:

1. **Axis-Aligned Boxes** (angle = 0): Exact intersection calculation
2. **Rotated Boxes**: AABB intersection with scaling factor `cos(|angle1|) * cos(|angle2|)`, minimum 50% retention

For exact rotated box IoU, consider implementing Sutherland-Hodgman clipping or the Separating Axes Theorem (SAT).

## Development

### Building Locally

```bash
pip install build scikit-build-core cmake nanobind
python -m build
```

### Testing Your Build

```bash
# Correctness tests (compares fastbbox against Python reference implementations)
python test_fastbbox.py              # Summary output
python test_fastbbox.py --verbose    # Detailed output with values
python test_fastbbox.py -f iou giou  # Test specific functions

# Performance benchmarks
python benchmark_fastbbox.py              # Summary output
python benchmark_fastbbox.py --verbose    # Detailed timing per run
python benchmark_fastbbox.py --size 1000  # Test with 1000 boxes
python benchmark_fastbbox.py --runs 10    # 10 iterations
```

#### Test File Options

**test_fastbbox.py** - Correctness validation:
- `--verbose, -v`: Show detailed output with expected/actual values
- `--function, -f`: Test specific functions (iou, giou, diou, ciou, eiou, nwd, obb)
- `--tolerance, -t`: Tolerance threshold (default: 1e-5)
- `--obb-tolerance`: OBB-specific tolerance (default: 1e-3)
- `--size, -s`: Number of test boxes (default: 100)

**benchmark_fastbbox.py** - Performance benchmarks:
- `--verbose, -v`: Show individual run times and statistics
- `--function, -f`: Benchmark specific functions
- `--size, -s`: Number of boxes (default: 500)
- `--runs, -r`: Number of iterations (default: 5)

## Requirements

- Python 3.9+
- NumPy >= 1.19.0

## License

MIT License - see LICENSE file for details.
