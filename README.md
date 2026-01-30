# FastBBox

Fast IoU/overlap computations for axis-aligned and oriented bounding boxes using Cython.

## Installation

### From PyPI (recommended, not uploaded yet)

```bash
pip install fastbbox
```

### From source

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

# Create some example bounding boxes [x1, y1, x2, y2]
boxes = np.array([
    [0, 0, 10, 10],
    [5, 5, 15, 15],
    [20, 20, 30, 30]
], dtype=np.float32)

query_boxes = np.array([
    [0, 0, 10, 10],
    [12, 12, 22, 22]
], dtype=np.float32)

# Compute all IoU variants and distance metrics
iou = bbox_overlaps(boxes, query_boxes)
giou = generalized_iou(boxes, query_boxes)
diou = distance_iou(boxes, query_boxes)
ciou = complete_iou(boxes, query_boxes)
eiou = efficient_iou(boxes, query_boxes)
nwd = normalized_wasserstein_distance(boxes, query_boxes)

print("IoU: ", iou)
print("GIoU:", giou)
print("DIoU:", diou)
print("CIoU:", ciou)
print("EIoU:", eiou)
print("NWD: ", nwd)
```

### Oriented Bounding Boxes (OBB)

```python
import numpy as np
import math
from fastbbox import bbox_overlaps_obb

# Create oriented bounding boxes [center_x, center_y, width, height, angle_radians]
obb_boxes = np.array([
    [0, 0, 4, 2, 0],                    # Axis-aligned box
    [1, 0, 4, 2, math.pi/4],            # 45° rotated box
    [0, 0, 2, 2, math.pi/6],            # 30° rotated box
], dtype=np.float32)

obb_query_boxes = np.array([
    [0, 0, 4, 2, 0],                    # Same as first box
    [2, 2, 2, 2, 0],                    # Offset box
], dtype=np.float32)

# Calculate OBB IoU matrix
obb_iou = bbox_overlaps_obb(obb_boxes, obb_query_boxes)
print("OBB IoU:", obb_iou)

# Helper function for degrees
def create_obb(cx, cy, width, height, angle_degrees=0):
    """Create OBB with angle in degrees."""
    angle_radians = math.radians(angle_degrees)
    return np.array([cx, cy, width, height, angle_radians], dtype=np.float32)

# Usage with degrees
box = create_obb(0, 0, 4, 2, 45)  # 45 degree rotation
```

## Features

### Axis-Aligned Bounding Box IoU Variants

All variants support standard axis-aligned bounding boxes in `[x1, y1, x2, y2]` format:

- **Standard IoU**: Classic Intersection over Union calculation
- **Generalized IoU (GIoU)**: Addresses limitations of IoU by considering enclosing area
  - Provides meaningful gradients even when boxes don't overlap
  - Values in range [-1, 1] where -1 indicates boxes are far apart
  - Reference: [Generalized Intersection over Union](https://arxiv.org/abs/1902.09630)

- **Distance IoU (DIoU)**: Considers center point distance between boxes
  - Penalizes boxes with far-apart centers even if they overlap
  - Formula: `DIoU = IoU - ρ²(b, b_gt) / c²`
  - Better for object detection training than GIoU
  - Reference: [Distance-IoU Loss](https://arxiv.org/abs/1911.08287)

- **Complete IoU (CIoU)**: Most comprehensive metric considering overlap, center distance, and aspect ratio
  - Adds aspect ratio consistency penalty to DIoU
  - Formula: `CIoU = DIoU - α * v` (where v measures aspect ratio consistency)
  - Best for bounding box regression in object detection
  - Reference: [Distance-IoU Loss](https://arxiv.org/abs/1911.08287)

- **Efficient IoU (EIoU)**: Separates width and height penalties for more efficient training
  - Formula: `EIoU = IoU - ρ²(b, b_gt) / c² - ρ²(w, w_gt) / c_w² - ρ²(h, h_gt) / c_h²`
  - Directly penalizes width and height differences separately
  - Faster convergence than CIoU in many cases
  - Reference: [Focal and Efficient IOU Loss](https://arxiv.org/abs/2101.08158)

- **Normalized Wasserstein Distance (NWD)**: Similarity metric based on optimal transport theory
  - Treats bounding boxes as 2D Gaussian distributions with Σ = diag(w²/4, h²/4)
  - Formula: `NWD = exp(-√(W₂²) / τ)` where W₂² is Wasserstein-2 distance squared
  - Especially effective for tiny object detection
  - Returns values in [0, 1] where 1 = identical boxes, 0 = very different
  - Reference: [Normalized Gaussian Wasserstein Distance](https://arxiv.org/abs/2110.13389)

### Oriented Bounding Box (OBB) IoU

For rotated/oriented bounding boxes in `[center_x, center_y, width, height, angle_radians]` format:

- **OBB IoU**: Intersection over Union for oriented bounding boxes
  - **Exact calculation** for axis-aligned bounding boxes (angle = 0)
  - **Approximation-based calculation** for rotated bounding boxes
  - Uses AABB intersection with angle-based scaling factor
  - Scaling factor: `cos(|angle1|) * cos(|angle2|)` with minimum 50% retention
  - **Batch processing** support for multiple OBBs
  - Particularly useful for text detection, aerial imagery, and rotated object detection
  - **Performance**: ~310x speedup over pure Python implementation

## Performance

This Cython implementation provides significant speedup over pure Python implementations, especially for large numbers of bounding boxes.

### When to Use Each Variant

- **Use IoU** for: Standard evaluation metrics, NMS (Non-Maximum Suppression), simple overlap measurement
- **Use GIoU** for: Training loss when you want to consider spatial relationships of non-overlapping boxes
- **Use DIoU** for: Object detection training where center point distance matters, anchor-free detectors
- **Use CIoU** for: Advanced bounding box regression, when aspect ratio consistency is important
- **Use EIoU** for: Fast training convergence, when you need separate width/height penalties
- **Use NWD** for: Tiny object detection, when traditional IoU variants are insufficient
- **Use OBB IoU** for: Rotated object detection, text detection, aerial imagery analysis, scene text recognition

### Metric Relationships

For any pair of bounding boxes, the following relationships generally hold:
```
Axis-aligned IoU metrics: EIoU ≤ CIoU ≤ DIoU ≤ IoU  (for overlapping boxes)
                          GIoU ≤ IoU

IoU variants ∈ [-1, 1] where 1 = perfect match, 0 = no overlap, negative = penalty for separation

NWD ∈ [0, 1] where 1 = identical boxes, 0 = very different (similarity measure)

OBB IoU ∈ [0, 1] where 1 = perfect match, 0 = no overlap (exact for axis-aligned, approximation for rotated)
```

## OBB Implementation Details

### Algorithm Strategy

The OBB IoU implementation uses different strategies based on box orientations:

1. **Axis-Aligned Boxes** (angle = 0): Exact intersection calculation using rectangle overlap formulas
2. **Rotated Boxes**: Approximation using axis-aligned bounding box (AABB) intersection with angle-based scaling

### Approximation Method

For rotated boxes, the algorithm:
1. Converts OBBs to corner coordinates
2. Finds axis-aligned bounding boxes of both OBBs  
3. Calculates AABB intersection
4. Applies scaling factor: `cos(|angle1|) * cos(|angle2|)`
5. Ensures minimum 50% of AABB intersection is retained

### Performance Characteristics

- ✅ **Exact results** for axis-aligned boxes (angle = 0)
- ✅ **Good approximations** for small rotation angles  
- ✅ **Reasonable approximations** for larger rotations
- ✅ **Fast computation** due to Cython implementation (~310x speedup)

### Limitations

1. **Approximation for rotated boxes**: Not exact for arbitrary rotations
2. **Scaling factor heuristic**: The angle-based scaling is a heuristic approximation
3. **No advanced polygon intersection**: Uses AABB approximation rather than exact polygon clipping

For applications requiring exact rotated box IoU, consider implementing:
- Exact polygon intersection (Sutherland-Hodgman clipping)
- Separating Axes Theorem (SAT) for precise overlap detection

## Development

### Building locally

```bash
pip install build
python -m build
```

### Testing the build

```bash
pip install dist/fastbbox-0.1.0-*.whl
python test_all_iou.py    # Run axis-aligned bounding box test suite
python test_obb_iou.py    # Run oriented bounding box test suite
python benchmark_comparison.py  # Run performance benchmarks
```

## Requirements

- Python 3.8+
- NumPy >= 1.19.0

## License

MIT License - see LICENSE file for details.

