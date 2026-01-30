#!/usr/bin/env python
"""Quick benchmark of current build"""
import numpy as np
import time
import fastbbox

print(f"Benchmarking fastbbox {fastbbox.__version__} with backend: {fastbbox.__backend__}")
print("="*60)

# Generate test data
np.random.seed(42)
n_boxes = 1000
boxes = np.random.rand(n_boxes, 4).astype(np.float32) * 1000
boxes[:, 2:] += boxes[:, :2] + 10  # Ensure x2 > x1, y2 > y1
query_boxes = np.random.rand(100, 4).astype(np.float32) * 1000
query_boxes[:, 2:] += query_boxes[:, :2] + 10

functions = [
    ('IoU', fastbbox.bbox_overlaps),
    ('GIoU', fastbbox.generalized_iou),
    ('DIoU', fastbbox.distance_iou),
    ('CIoU', fastbbox.complete_iou),
    ('EIoU', fastbbox.efficient_iou),
    ('NWD', fastbbox.normalized_wasserstein_distance),
]

# Warmup
_ = fastbbox.bbox_overlaps(boxes[:10], query_boxes[:10])

results = []
for name, func in functions:
    start = time.perf_counter()
    result = func(boxes, query_boxes)
    elapsed = time.perf_counter() - start
    
    print(f"{name:6s}: {elapsed*1000:7.2f} ms | shape: {result.shape} | range: [{result.min():.3f}, {result.max():.3f}]")
    results.append((name, elapsed))

# OBB test
obb_boxes = np.random.rand(500, 5).astype(np.float32) * 1000
obb_query = np.random.rand(50, 5).astype(np.float32) * 1000

start = time.perf_counter()
obb_result = fastbbox.bbox_overlaps_obb(obb_boxes, obb_query)
obb_elapsed = time.perf_counter() - start

print(f"OBB   : {obb_elapsed*1000:7.2f} ms | shape: {obb_result.shape} | range: [{obb_result.min():.3f}, {obb_result.max():.3f}]")

print("\n" + "="*60)
print(f"Total time: {sum(e for _, e in results) + obb_elapsed:.3f}s")
print(f"Backend: {fastbbox.__backend__}")
