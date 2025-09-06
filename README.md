# FastBBox

Fast IoU/overlap computations for bounding boxes using Cython.

## Installation

### From PyPI (recommended)

```bash
pip install fastbbox
```

### From source

```bash
git clone https://github.com/yourusername/fastbbox
cd fastbbox
pip install .
```

## Usage

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

## Features

### IoU Variants Implemented

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

## Performance

This Cython implementation provides significant speedup over pure Python implementations, especially for large numbers of bounding boxes.

### When to Use Each Variant

- **Use IoU** for: Standard evaluation metrics, NMS (Non-Maximum Suppression), simple overlap measurement
- **Use GIoU** for: Training loss when you want to consider spatial relationships of non-overlapping boxes
- **Use DIoU** for: Object detection training where center point distance matters, anchor-free detectors
- **Use CIoU** for: Advanced bounding box regression, when aspect ratio consistency is important
- **Use EIoU** for: Fast training convergence, when you need separate width/height penalties
- **Use NWD** for: Tiny object detection, when traditional IoU variants are insufficient

### Metric Relationships

For any pair of bounding boxes, the following relationships generally hold:
```
IoU-based metrics: EIoU ≤ CIoU ≤ DIoU ≤ IoU  (for overlapping boxes)
                   GIoU ≤ IoU

IoU variants ∈ [-1, 1] where 1 = perfect match, 0 = no overlap, negative = penalty for separation

NWD ∈ [0, 1] where 1 = identical boxes, 0 = very different (similarity measure)
```

## Development

### Building locally

```bash
pip install build
python -m build
```

### Testing the build

```bash
pip install dist/fastbbox-0.1.0-*.whl
python test_all_iou.py  # Run comprehensive test suite
```

## Requirements

- Python 3.8+
- NumPy >= 1.19.0

## License

MIT License - see LICENSE file for details.

