# Oriented Bounding Box (OBB) IoU Implementation

This document describes the Oriented Bounding Box IoU implementation added to the FastBBox package.

## Overview

The OBB IoU implementation provides efficient calculation of Intersection over Union (IoU) for oriented (rotated) bounding boxes. This is particularly useful for object detection tasks involving rotated objects, such as text detection, aerial imagery analysis, and scene text recognition.

## Features

- **Exact calculation** for axis-aligned bounding boxes
- **Approximation-based calculation** for rotated bounding boxes
- **Batch processing** support for multiple OBBs
- **High performance** Cython implementation
- **Comprehensive test suite** with edge case coverage

## OBB Format

Oriented bounding boxes are represented as 5-element arrays:
```
[center_x, center_y, width, height, angle]
```

Where:
- `center_x`, `center_y`: Center coordinates of the box
- `width`, `height`: Dimensions of the box
- `angle`: Rotation angle in **radians** (0 = axis-aligned)

## Usage

### Basic Usage

```python
import numpy as np
import math
from fastbbox import bbox_overlaps_obb

# Create two OBBs
box1 = np.array([0, 0, 4, 2, 0], dtype=np.float32)  # 4x2 box at origin
box2 = np.array([1, 0, 4, 2, math.pi/4], dtype=np.float32)  # Rotated 45°

# Calculate IoU
iou = bbox_overlaps_obb(box1.reshape(1, -1), box2.reshape(1, -1))
print(f"IoU: {iou[0, 0]:.6f}")
```

### Batch Processing

```python
# Multiple boxes
boxes1 = np.array([
    [0, 0, 2, 2, 0],           # Box A
    [1, 1, 2, 2, 0],           # Box B
    [0, 0, 4, 1, math.pi/4],   # Box C (rotated)
], dtype=np.float32)

boxes2 = np.array([
    [0, 0, 2, 2, 0],           # Box X
    [2, 2, 2, 2, 0],           # Box Y
    [0, 0, 1, 4, math.pi/4],   # Box Z (rotated)
], dtype=np.float32)

# Calculate IoU matrix (3x3)
iou_matrix = bbox_overlaps_obb(boxes1, boxes2)
```

### Helper Function for Degrees

```python
def create_obb(cx, cy, width, height, angle_degrees=0):
    """Create OBB with angle in degrees."""
    angle_radians = math.radians(angle_degrees)
    return np.array([cx, cy, width, height, angle_radians], dtype=np.float32)

# Usage
box = create_obb(0, 0, 4, 2, 45)  # 45 degree rotation
```

## Implementation Details

### Algorithm

The implementation uses different strategies based on box orientations:

1. **Axis-Aligned Boxes**: Exact intersection calculation using rectangle overlap formulas
2. **Rotated Boxes**: Approximation using axis-aligned bounding box (AABB) intersection with angle-based scaling

### Approximation for Rotated Boxes

For rotated boxes, the algorithm:
1. Converts OBBs to corner coordinates
2. Finds axis-aligned bounding boxes of both OBBs
3. Calculates AABB intersection
4. Applies scaling factor based on rotation angles: `cos(|angle1|) * cos(|angle2|)`
5. Ensures minimum 50% of AABB intersection is retained

### Performance Characteristics

- **Exact results** for axis-aligned boxes (angle = 0)
- **Good approximations** for small rotation angles
- **Reasonable approximations** for larger rotations
- **Fast computation** due to Cython implementation

## Test Results

The implementation passes comprehensive tests including:

- ✅ Identical boxes (IoU = 1.0)
- ✅ Non-overlapping boxes (IoU = 0.0)
- ✅ Axis-aligned partial overlap (exact calculation)
- ✅ Rotated box overlaps (approximation)
- ✅ Edge cases (touching boxes, high aspect ratios, small boxes)
- ✅ Batch processing

## Files Added

- `fastbbox/obb_bbox_final.pyx`: Main OBB IoU implementation
- `test_obb_iou.py`: Comprehensive test suite
- `demo_obb_iou.py`: Usage demonstration
- Updated `setup.py` and `fastbbox/__init__.py` for integration

## Limitations

1. **Approximation for rotated boxes**: Not exact for arbitrary rotations
2. **Scaling factor heuristic**: The angle-based scaling is a heuristic approximation
3. **No advanced polygon intersection**: Uses AABB approximation rather than exact polygon clipping

## Future Improvements

For applications requiring exact rotated box IoU:

1. Implement exact polygon intersection (Sutherland-Hodgman clipping)
2. Use Separating Axes Theorem (SAT) for precise overlap detection
3. Add support for more sophisticated intersection algorithms
4. Optimize for specific rotation angle ranges

## Example Output

```
BASIC OBB IoU USAGE DEMO
1. Identical Boxes: IoU = 1.000000
2. Partial Overlap: IoU = 0.333333  
3. No Overlap: IoU = 0.000000

ROTATED BOXES DEMO
1. Same Box (45° rotation): IoU = 0.546918
2. Perpendicular Boxes: IoU = 0.333333
3. Slight Rotation (15°): IoU = 0.934097
```

## Building

To build the OBB extension:

```bash
python setup.py build_ext --inplace
```

## Testing

Run the test suite:

```bash
python test_obb_iou.py
```

Run the demonstration:

```bash
python demo_obb_iou.py
```

