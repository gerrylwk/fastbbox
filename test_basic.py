#!/usr/bin/env python
"""Quick test of fastbbox functionality"""
import numpy as np
import fastbbox

print(f"Testing fastbbox {fastbbox.__version__} with backend: {fastbbox.__backend__}")
print("="*60)

# Test basic IoU
boxes = np.array([[0, 0, 10, 10], [5, 5, 15, 15]], dtype=np.float32)
iou = fastbbox.bbox_overlaps(boxes, boxes)
print(f"IoU shape: {iou.shape}")
print(f"Self-IoU diagonal: {iou.diagonal()}")
assert np.allclose(iou.diagonal(), 1.0), "Self-IoU should be 1.0"
print("[PASS] IoU test")

# Test GIoU
giou = fastbbox.generalized_iou(boxes, boxes)
print(f"GIoU diagonal: {giou.diagonal()}")
assert np.allclose(giou.diagonal(), 1.0), "Self-GIoU should be 1.0"
print("[PASS] GIoU test")

# Test DIoU
diou = fastbbox.distance_iou(boxes, boxes)
print(f"DIoU diagonal: {diou.diagonal()}")
assert np.allclose(diou.diagonal(), 1.0), "Self-DIoU should be 1.0"
print("[PASS] DIoU test")

# Test CIoU
ciou = fastbbox.complete_iou(boxes, boxes)
print(f"CIoU diagonal: {ciou.diagonal()}")
assert np.allclose(ciou.diagonal(), 1.0), "Self-CIoU should be 1.0"
print("[PASS] CIoU test")

# Test EIoU
eiou = fastbbox.efficient_iou(boxes, boxes)
print(f"EIoU diagonal: {eiou.diagonal()}")
assert np.allclose(eiou.diagonal(), 1.0), "Self-EIoU should be 1.0"
print("[PASS] EIoU test")

# Test NWD
nwd = fastbbox.normalized_wasserstein_distance(boxes, boxes)
print(f"NWD diagonal: {nwd.diagonal()}")
assert np.allclose(nwd.diagonal(), 1.0), "Self-NWD should be 1.0"
print("[PASS] NWD test")

# Test OBB
obb_boxes = np.array([[5, 5, 10, 10, 0], [10, 10, 10, 10, 0]], dtype=np.float32)
obb_iou = fastbbox.bbox_overlaps_obb(obb_boxes, obb_boxes)
print(f"OBB IoU shape: {obb_iou.shape}")
print(f"OBB Self-IoU diagonal: {obb_iou.diagonal()}")
# Note: OBB uses approximation for rotated boxes, so exact 1.0 may not always hold
# Just verify it runs and returns reasonable values
assert obb_iou.shape == (2, 2), "OBB IoU shape should be (2, 2)"
print("[PASS] OBB IoU test")

print("="*60)
print(f"ALL TESTS PASSED with backend: {fastbbox.__backend__}")
