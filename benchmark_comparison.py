#!/usr/bin/env python3
"""
Benchmark and Accuracy Comparison Script
========================================

Compares Python implementations against fastbbox Cython implementations
for all IoU variants and NWD using hundreds of bounding boxes.

Tests both correctness (numerical accuracy) and performance.
"""

import numpy as np
import time
from typing import Tuple

# Import fastbbox functions
try:
    from fastbbox import (
        bbox_overlaps, 
        generalized_iou, 
        distance_iou, 
        complete_iou, 
        efficient_iou, 
        normalized_wasserstein_distance
    )
    print("✓ Successfully imported fastbbox functions")
except ImportError as e:
    print(f"✗ Failed to import fastbbox: {e}")
    exit(1)


def generate_test_boxes(n_boxes: int, max_coord: float = 1000.0) -> np.ndarray:
    """Generate random bounding boxes for testing."""
    np.random.seed(42)  # For reproducible results
    
    # Generate random coordinates
    x1 = np.random.uniform(0, max_coord * 0.8, n_boxes)
    y1 = np.random.uniform(0, max_coord * 0.8, n_boxes)
    x2 = x1 + np.random.uniform(10, max_coord * 0.2, n_boxes)
    y2 = y1 + np.random.uniform(10, max_coord * 0.2, n_boxes)
    
    boxes = np.column_stack([x1, y1, x2, y2]).astype(np.float32)
    return boxes


# Python reference implementations
def python_bbox_overlaps(boxes: np.ndarray, query_boxes: np.ndarray) -> np.ndarray:
    """Python reference implementation of IoU."""
    n, k = len(boxes), len(query_boxes)
    ious = np.zeros((n, k), dtype=np.float32)
    
    for i in range(n):
        for j in range(k):
            # Calculate intersection
            x1 = max(boxes[i, 0], query_boxes[j, 0])
            y1 = max(boxes[i, 1], query_boxes[j, 1])
            x2 = min(boxes[i, 2], query_boxes[j, 2])
            y2 = min(boxes[i, 3], query_boxes[j, 3])
            
            if x2 > x1 and y2 > y1:
                intersection = (x2 - x1) * (y2 - y1)
            else:
                intersection = 0.0
            
            # Calculate areas
            area1 = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            area2 = (query_boxes[j, 2] - query_boxes[j, 0]) * (query_boxes[j, 3] - query_boxes[j, 1])
            
            # Calculate IoU
            union = area1 + area2 - intersection
            ious[i, j] = intersection / union if union > 0 else 0.0
    
    return ious


def python_generalized_iou(boxes: np.ndarray, query_boxes: np.ndarray) -> np.ndarray:
    """Python reference implementation of GIoU."""
    n, k = len(boxes), len(query_boxes)
    gious = np.zeros((n, k), dtype=np.float32)
    
    for i in range(n):
        for j in range(k):
            # Calculate intersection
            x1 = max(boxes[i, 0], query_boxes[j, 0])
            y1 = max(boxes[i, 1], query_boxes[j, 1])
            x2 = min(boxes[i, 2], query_boxes[j, 2])
            y2 = min(boxes[i, 3], query_boxes[j, 3])
            
            if x2 > x1 and y2 > y1:
                intersection = (x2 - x1) * (y2 - y1)
            else:
                intersection = 0.0
            
            # Calculate areas
            area1 = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            area2 = (query_boxes[j, 2] - query_boxes[j, 0]) * (query_boxes[j, 3] - query_boxes[j, 1])
            union = area1 + area2 - intersection
            
            # Calculate enclosing box
            enc_x1 = min(boxes[i, 0], query_boxes[j, 0])
            enc_y1 = min(boxes[i, 1], query_boxes[j, 1])
            enc_x2 = max(boxes[i, 2], query_boxes[j, 2])
            enc_y2 = max(boxes[i, 3], query_boxes[j, 3])
            enc_area = (enc_x2 - enc_x1) * (enc_y2 - enc_y1)
            
            # Calculate GIoU
            iou = intersection / union if union > 0 else 0.0
            penalty = (enc_area - union) / enc_area if enc_area > 0 else 0.0
            gious[i, j] = iou - penalty
    
    return gious


def python_distance_iou(boxes: np.ndarray, query_boxes: np.ndarray) -> np.ndarray:
    """Python reference implementation of DIoU."""
    n, k = len(boxes), len(query_boxes)
    dious = np.zeros((n, k), dtype=np.float32)
    
    for i in range(n):
        for j in range(k):
            # Calculate IoU first
            x1 = max(boxes[i, 0], query_boxes[j, 0])
            y1 = max(boxes[i, 1], query_boxes[j, 1])
            x2 = min(boxes[i, 2], query_boxes[j, 2])
            y2 = min(boxes[i, 3], query_boxes[j, 3])
            
            if x2 > x1 and y2 > y1:
                intersection = (x2 - x1) * (y2 - y1)
            else:
                intersection = 0.0
            
            area1 = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            area2 = (query_boxes[j, 2] - query_boxes[j, 0]) * (query_boxes[j, 3] - query_boxes[j, 1])
            union = area1 + area2 - intersection
            iou = intersection / union if union > 0 else 0.0
            
            # Calculate centers
            c1_x = (boxes[i, 0] + boxes[i, 2]) / 2.0
            c1_y = (boxes[i, 1] + boxes[i, 3]) / 2.0
            c2_x = (query_boxes[j, 0] + query_boxes[j, 2]) / 2.0
            c2_y = (query_boxes[j, 1] + query_boxes[j, 3]) / 2.0
            
            # Distance between centers
            center_dist_sq = (c1_x - c2_x) ** 2 + (c1_y - c2_y) ** 2
            
            # Diagonal of enclosing box
            enc_x1 = min(boxes[i, 0], query_boxes[j, 0])
            enc_y1 = min(boxes[i, 1], query_boxes[j, 1])
            enc_x2 = max(boxes[i, 2], query_boxes[j, 2])
            enc_y2 = max(boxes[i, 3], query_boxes[j, 3])
            diagonal_sq = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2
            
            # DIoU
            penalty = center_dist_sq / diagonal_sq if diagonal_sq > 0 else 0.0
            dious[i, j] = iou - penalty
    
    return dious


def python_complete_iou(boxes: np.ndarray, query_boxes: np.ndarray) -> np.ndarray:
    """Python reference implementation of CIoU."""
    n, k = len(boxes), len(query_boxes)
    cious = np.zeros((n, k), dtype=np.float32)
    
    for i in range(n):
        for j in range(k):
            # Calculate DIoU first
            x1 = max(boxes[i, 0], query_boxes[j, 0])
            y1 = max(boxes[i, 1], query_boxes[j, 1])
            x2 = min(boxes[i, 2], query_boxes[j, 2])
            y2 = min(boxes[i, 3], query_boxes[j, 3])
            
            if x2 > x1 and y2 > y1:
                intersection = (x2 - x1) * (y2 - y1)
            else:
                intersection = 0.0
            
            area1 = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            area2 = (query_boxes[j, 2] - query_boxes[j, 0]) * (query_boxes[j, 3] - query_boxes[j, 1])
            union = area1 + area2 - intersection
            iou = intersection / union if union > 0 else 0.0
            
            # Centers and distance penalty
            c1_x = (boxes[i, 0] + boxes[i, 2]) / 2.0
            c1_y = (boxes[i, 1] + boxes[i, 3]) / 2.0
            c2_x = (query_boxes[j, 0] + query_boxes[j, 2]) / 2.0
            c2_y = (query_boxes[j, 1] + query_boxes[j, 3]) / 2.0
            
            center_dist_sq = (c1_x - c2_x) ** 2 + (c1_y - c2_y) ** 2
            
            enc_x1 = min(boxes[i, 0], query_boxes[j, 0])
            enc_y1 = min(boxes[i, 1], query_boxes[j, 1])
            enc_x2 = max(boxes[i, 2], query_boxes[j, 2])
            enc_y2 = max(boxes[i, 3], query_boxes[j, 3])
            diagonal_sq = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2
            
            # Aspect ratio penalty
            w1, h1 = boxes[i, 2] - boxes[i, 0], boxes[i, 3] - boxes[i, 1]
            w2, h2 = query_boxes[j, 2] - query_boxes[j, 0], query_boxes[j, 3] - query_boxes[j, 1]
            
            if w1 > 0 and h1 > 0 and w2 > 0 and h2 > 0:
                # Use atan2 for more stable computation - matches Cython implementation
                atan_diff = np.arctan2(w2, h2) - np.arctan2(w1, h1)
                v = (4.0 / (np.pi ** 2)) * (atan_diff ** 2)
            else:
                v = 0.0
            
            # Calculate alpha parameter with numerical stability
            if iou > 0:
                alpha = v / (1 - iou + v + 1e-8)  # Add small epsilon for numerical stability
            else:
                alpha = 0.0
            
            # CIoU
            dist_penalty = center_dist_sq / diagonal_sq if diagonal_sq > 0 else 0.0
            cious[i, j] = iou - dist_penalty - alpha * v
    
    return cious


def python_efficient_iou(boxes: np.ndarray, query_boxes: np.ndarray) -> np.ndarray:
    """Python reference implementation of EIoU."""
    n, k = len(boxes), len(query_boxes)
    eious = np.zeros((n, k), dtype=np.float32)
    
    for i in range(n):
        for j in range(k):
            # Calculate IoU first
            x1 = max(boxes[i, 0], query_boxes[j, 0])
            y1 = max(boxes[i, 1], query_boxes[j, 1])
            x2 = min(boxes[i, 2], query_boxes[j, 2])
            y2 = min(boxes[i, 3], query_boxes[j, 3])
            
            if x2 > x1 and y2 > y1:
                intersection = (x2 - x1) * (y2 - y1)
            else:
                intersection = 0.0
            
            area1 = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            area2 = (query_boxes[j, 2] - query_boxes[j, 0]) * (query_boxes[j, 3] - query_boxes[j, 1])
            union = area1 + area2 - intersection
            iou = intersection / union if union > 0 else 0.0
            
            # Centers and distance penalty
            c1_x = (boxes[i, 0] + boxes[i, 2]) / 2.0
            c1_y = (boxes[i, 1] + boxes[i, 3]) / 2.0
            c2_x = (query_boxes[j, 0] + query_boxes[j, 2]) / 2.0
            c2_y = (query_boxes[j, 1] + query_boxes[j, 3]) / 2.0
            
            center_dist_sq = (c1_x - c2_x) ** 2 + (c1_y - c2_y) ** 2
            
            enc_x1 = min(boxes[i, 0], query_boxes[j, 0])
            enc_y1 = min(boxes[i, 1], query_boxes[j, 1])
            enc_x2 = max(boxes[i, 2], query_boxes[j, 2])
            enc_y2 = max(boxes[i, 3], query_boxes[j, 3])
            diagonal_sq = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2
            
            # Width and height penalties (separated)
            w1, h1 = boxes[i, 2] - boxes[i, 0], boxes[i, 3] - boxes[i, 1]
            w2, h2 = query_boxes[j, 2] - query_boxes[j, 0], query_boxes[j, 3] - query_boxes[j, 1]
            enc_w = enc_x2 - enc_x1
            enc_h = enc_y2 - enc_y1
            
            w_penalty = ((w1 - w2) ** 2) / (enc_w ** 2) if enc_w > 0 else 0.0
            h_penalty = ((h1 - h2) ** 2) / (enc_h ** 2) if enc_h > 0 else 0.0
            
            # EIoU
            dist_penalty = center_dist_sq / diagonal_sq if diagonal_sq > 0 else 0.0
            eious[i, j] = iou - dist_penalty - w_penalty - h_penalty
    
    return eious


def python_normalized_wasserstein_distance(boxes: np.ndarray, query_boxes: np.ndarray, tau: float = 1.0) -> np.ndarray:
    """Python reference implementation of NWD."""
    n, k = len(boxes), len(query_boxes)
    nwd_matrix = np.zeros((n, k), dtype=np.float32)
    
    for i in range(n):
        for j in range(k):
            # Calculate box centers and half-dimensions
            box_cx = (boxes[i, 0] + boxes[i, 2]) / 2.0
            box_cy = (boxes[i, 1] + boxes[i, 3]) / 2.0
            box_w_half = (boxes[i, 2] - boxes[i, 0]) / 2.0
            box_h_half = (boxes[i, 3] - boxes[i, 1]) / 2.0
            
            query_cx = (query_boxes[j, 0] + query_boxes[j, 2]) / 2.0
            query_cy = (query_boxes[j, 1] + query_boxes[j, 3]) / 2.0
            query_w_half = (query_boxes[j, 2] - query_boxes[j, 0]) / 2.0
            query_h_half = (query_boxes[j, 3] - query_boxes[j, 1]) / 2.0
            
            # Mean difference squared
            mean_diff_sq = (box_cx - query_cx) ** 2 + (box_cy - query_cy) ** 2
            
            # Covariance difference squared
            cov_diff_sq = (box_w_half - query_w_half) ** 2 + (box_h_half - query_h_half) ** 2
            
            # Wasserstein-2 distance squared
            wasserstein_sq = mean_diff_sq + cov_diff_sq
            
            # Apply exponential normalization
            nwd_val = np.exp(-np.sqrt(wasserstein_sq) / tau) if wasserstein_sq >= 0 else 1.0
            nwd_matrix[i, j] = nwd_val
    
    return nwd_matrix


def benchmark_function(func, boxes: np.ndarray, query_boxes: np.ndarray, name: str, *args) -> Tuple[np.ndarray, float]:
    """Benchmark a function and return results with timing."""
    print(f"  Running {name}...", end=" ")
    start_time = time.time()
    result = func(boxes, query_boxes, *args)
    elapsed_time = time.time() - start_time
    print(f"{elapsed_time:.4f}s")
    return result, elapsed_time


def compare_results(python_result: np.ndarray, cython_result: np.ndarray, name: str, tolerance: float = 1e-5):
    """Compare Python and Cython results for accuracy."""
    max_diff = np.max(np.abs(python_result - cython_result))
    mean_diff = np.mean(np.abs(python_result - cython_result))
    
    print(f"  {name} - Max diff: {max_diff:.2e}, Mean diff: {mean_diff:.2e}", end=" ")
    
    if max_diff < tolerance:
        print("✓ PASS")
        return True
    else:
        print("✗ FAIL")
        print(f"    Sample differences: {(python_result - cython_result).flat[:5]}")
        return False


def main():
    print("=" * 80)
    print("FASTBBOX BENCHMARK & ACCURACY COMPARISON")
    print("=" * 80)
    
    # Test parameters
    n_boxes = 500
    n_query_boxes = 300
    print(f"Test setup: {n_boxes} boxes vs {n_query_boxes} query boxes")
    print(f"Total comparisons: {n_boxes * n_query_boxes:,}")
    
    # Generate test data
    print("\nGenerating test data...")
    boxes = generate_test_boxes(n_boxes)
    query_boxes = generate_test_boxes(n_query_boxes)
    print(f"✓ Generated {len(boxes)} boxes and {len(query_boxes)} query boxes")
    
    # Test functions
    test_cases = [
        ("IoU", python_bbox_overlaps, bbox_overlaps, []),
        ("GIoU", python_generalized_iou, generalized_iou, []),
        ("DIoU", python_distance_iou, distance_iou, []),
        ("CIoU", python_complete_iou, complete_iou, []),
        ("EIoU", python_efficient_iou, efficient_iou, []),
        ("NWD", python_normalized_wasserstein_distance, normalized_wasserstein_distance, [1.0]),
    ]
    
    print("\n" + "=" * 80)
    print("PERFORMANCE BENCHMARKS")
    print("=" * 80)
    
    all_passed = True
    performance_results = []
    
    for name, python_func, cython_func, extra_args in test_cases:
        print(f"\n{name} ({python_func.__name__}):")
        
        # Benchmark Python implementation
        python_result, python_time = benchmark_function(
            python_func, boxes, query_boxes, "Python", *extra_args
        )
        
        # Benchmark Cython implementation
        cython_result, cython_time = benchmark_function(
            cython_func, boxes, query_boxes, "Cython", *extra_args
        )
        
        # Compare accuracy
        passed = compare_results(python_result, cython_result, name)
        all_passed = all_passed and passed
        
        # Calculate speedup
        speedup = python_time / cython_time if cython_time > 0 else float('inf')
        performance_results.append((name, python_time, cython_time, speedup))
        print(f"  Speedup: {speedup:.1f}x")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print(f"\nAccuracy: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    
    print(f"\nPerformance Summary:")
    print(f"{'Function':<8} {'Python (s)':<12} {'Cython (s)':<12} {'Speedup':<10}")
    print("-" * 50)
    
    total_python_time = 0
    total_cython_time = 0
    
    for name, py_time, cy_time, speedup in performance_results:
        print(f"{name:<8} {py_time:<12.4f} {cy_time:<12.4f} {speedup:<10.1f}x")
        total_python_time += py_time
        total_cython_time += cy_time
    
    print("-" * 50)
    overall_speedup = total_python_time / total_cython_time if total_cython_time > 0 else float('inf')
    print(f"{'TOTAL':<8} {total_python_time:<12.4f} {total_cython_time:<12.4f} {overall_speedup:<10.1f}x")
    
    print(f"\n🚀 Overall Cython speedup: {overall_speedup:.1f}x faster than Python!")
    print(f"📊 Test completed with {n_boxes * n_query_boxes:,} total box comparisons")
    
    if not all_passed:
        print("\n⚠️  Some accuracy tests failed. Please check the implementations.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
