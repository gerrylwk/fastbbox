#!/usr/bin/env python3
"""
FastBBox Test Suite
===================

Comprehensive correctness validation for all fastbbox functions by comparing
against pure Python reference implementations.

Usage:
    python test_fastbbox.py              # Run all tests (summary mode)
    python test_fastbbox.py --verbose    # Run all tests with detailed output
    python test_fastbbox.py --function iou giou  # Test specific functions
    python test_fastbbox.py --tolerance 1e-4     # Custom tolerance threshold
"""

import argparse
import numpy as np
import sys
from typing import Callable, Dict, List, Tuple

# =============================================================================
# Python Reference Implementations
# =============================================================================

def python_iou(boxes: np.ndarray, query_boxes: np.ndarray) -> np.ndarray:
    """Pure Python implementation of IoU."""
    n, k = len(boxes), len(query_boxes)
    result = np.zeros((n, k), dtype=np.float32)
    
    for i in range(n):
        for j in range(k):
            x1 = max(boxes[i, 0], query_boxes[j, 0])
            y1 = max(boxes[i, 1], query_boxes[j, 1])
            x2 = min(boxes[i, 2], query_boxes[j, 2])
            y2 = min(boxes[i, 3], query_boxes[j, 3])
            
            intersection = max(0, x2 - x1) * max(0, y2 - y1)
            area1 = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            area2 = (query_boxes[j, 2] - query_boxes[j, 0]) * (query_boxes[j, 3] - query_boxes[j, 1])
            union = area1 + area2 - intersection
            result[i, j] = intersection / union if union > 0 else 0.0
    
    return result


def python_giou(boxes: np.ndarray, query_boxes: np.ndarray) -> np.ndarray:
    """Pure Python implementation of Generalized IoU."""
    n, k = len(boxes), len(query_boxes)
    result = np.zeros((n, k), dtype=np.float32)
    
    for i in range(n):
        for j in range(k):
            x1 = max(boxes[i, 0], query_boxes[j, 0])
            y1 = max(boxes[i, 1], query_boxes[j, 1])
            x2 = min(boxes[i, 2], query_boxes[j, 2])
            y2 = min(boxes[i, 3], query_boxes[j, 3])
            
            intersection = max(0, x2 - x1) * max(0, y2 - y1)
            area1 = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            area2 = (query_boxes[j, 2] - query_boxes[j, 0]) * (query_boxes[j, 3] - query_boxes[j, 1])
            union = area1 + area2 - intersection
            
            enc_x1 = min(boxes[i, 0], query_boxes[j, 0])
            enc_y1 = min(boxes[i, 1], query_boxes[j, 1])
            enc_x2 = max(boxes[i, 2], query_boxes[j, 2])
            enc_y2 = max(boxes[i, 3], query_boxes[j, 3])
            enc_area = (enc_x2 - enc_x1) * (enc_y2 - enc_y1)
            
            iou = intersection / union if union > 0 else 0.0
            penalty = (enc_area - union) / enc_area if enc_area > 0 else 0.0
            result[i, j] = iou - penalty
    
    return result


def python_diou(boxes: np.ndarray, query_boxes: np.ndarray) -> np.ndarray:
    """Pure Python implementation of Distance IoU."""
    n, k = len(boxes), len(query_boxes)
    result = np.zeros((n, k), dtype=np.float32)
    
    for i in range(n):
        for j in range(k):
            x1 = max(boxes[i, 0], query_boxes[j, 0])
            y1 = max(boxes[i, 1], query_boxes[j, 1])
            x2 = min(boxes[i, 2], query_boxes[j, 2])
            y2 = min(boxes[i, 3], query_boxes[j, 3])
            
            intersection = max(0, x2 - x1) * max(0, y2 - y1)
            area1 = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            area2 = (query_boxes[j, 2] - query_boxes[j, 0]) * (query_boxes[j, 3] - query_boxes[j, 1])
            union = area1 + area2 - intersection
            iou = intersection / union if union > 0 else 0.0
            
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
            
            penalty = center_dist_sq / diagonal_sq if diagonal_sq > 0 else 0.0
            result[i, j] = iou - penalty
    
    return result


def python_ciou(boxes: np.ndarray, query_boxes: np.ndarray) -> np.ndarray:
    """Pure Python implementation of Complete IoU."""
    n, k = len(boxes), len(query_boxes)
    result = np.zeros((n, k), dtype=np.float32)
    
    for i in range(n):
        for j in range(k):
            x1 = max(boxes[i, 0], query_boxes[j, 0])
            y1 = max(boxes[i, 1], query_boxes[j, 1])
            x2 = min(boxes[i, 2], query_boxes[j, 2])
            y2 = min(boxes[i, 3], query_boxes[j, 3])
            
            intersection = max(0, x2 - x1) * max(0, y2 - y1)
            area1 = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            area2 = (query_boxes[j, 2] - query_boxes[j, 0]) * (query_boxes[j, 3] - query_boxes[j, 1])
            union = area1 + area2 - intersection
            iou = intersection / union if union > 0 else 0.0
            
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
            
            w1, h1 = boxes[i, 2] - boxes[i, 0], boxes[i, 3] - boxes[i, 1]
            w2, h2 = query_boxes[j, 2] - query_boxes[j, 0], query_boxes[j, 3] - query_boxes[j, 1]
            
            if w1 > 0 and h1 > 0 and w2 > 0 and h2 > 0:
                atan_diff = np.arctan2(w2, h2) - np.arctan2(w1, h1)
                v = (4.0 / (np.pi ** 2)) * (atan_diff ** 2)
            else:
                v = 0.0
            
            alpha = v / (1 - iou + v + 1e-8) if iou > 0 else 0.0
            dist_penalty = center_dist_sq / diagonal_sq if diagonal_sq > 0 else 0.0
            result[i, j] = iou - dist_penalty - alpha * v
    
    return result


def python_eiou(boxes: np.ndarray, query_boxes: np.ndarray) -> np.ndarray:
    """Pure Python implementation of Efficient IoU."""
    n, k = len(boxes), len(query_boxes)
    result = np.zeros((n, k), dtype=np.float32)
    
    for i in range(n):
        for j in range(k):
            x1 = max(boxes[i, 0], query_boxes[j, 0])
            y1 = max(boxes[i, 1], query_boxes[j, 1])
            x2 = min(boxes[i, 2], query_boxes[j, 2])
            y2 = min(boxes[i, 3], query_boxes[j, 3])
            
            intersection = max(0, x2 - x1) * max(0, y2 - y1)
            area1 = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            area2 = (query_boxes[j, 2] - query_boxes[j, 0]) * (query_boxes[j, 3] - query_boxes[j, 1])
            union = area1 + area2 - intersection
            iou = intersection / union if union > 0 else 0.0
            
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
            
            w1, h1 = boxes[i, 2] - boxes[i, 0], boxes[i, 3] - boxes[i, 1]
            w2, h2 = query_boxes[j, 2] - query_boxes[j, 0], query_boxes[j, 3] - query_boxes[j, 1]
            enc_w, enc_h = enc_x2 - enc_x1, enc_y2 - enc_y1
            
            w_penalty = ((w1 - w2) ** 2) / (enc_w ** 2) if enc_w > 0 else 0.0
            h_penalty = ((h1 - h2) ** 2) / (enc_h ** 2) if enc_h > 0 else 0.0
            dist_penalty = center_dist_sq / diagonal_sq if diagonal_sq > 0 else 0.0
            result[i, j] = iou - dist_penalty - w_penalty - h_penalty
    
    return result


def python_nwd(boxes: np.ndarray, query_boxes: np.ndarray, tau: float = 1.0) -> np.ndarray:
    """Pure Python implementation of Normalized Wasserstein Distance."""
    n, k = len(boxes), len(query_boxes)
    result = np.zeros((n, k), dtype=np.float32)
    
    for i in range(n):
        for j in range(k):
            box_cx = (boxes[i, 0] + boxes[i, 2]) / 2.0
            box_cy = (boxes[i, 1] + boxes[i, 3]) / 2.0
            box_w_half = (boxes[i, 2] - boxes[i, 0]) / 2.0
            box_h_half = (boxes[i, 3] - boxes[i, 1]) / 2.0
            
            query_cx = (query_boxes[j, 0] + query_boxes[j, 2]) / 2.0
            query_cy = (query_boxes[j, 1] + query_boxes[j, 3]) / 2.0
            query_w_half = (query_boxes[j, 2] - query_boxes[j, 0]) / 2.0
            query_h_half = (query_boxes[j, 3] - query_boxes[j, 1]) / 2.0
            
            mean_diff_sq = (box_cx - query_cx) ** 2 + (box_cy - query_cy) ** 2
            cov_diff_sq = (box_w_half - query_w_half) ** 2 + (box_h_half - query_h_half) ** 2
            wasserstein_sq = mean_diff_sq + cov_diff_sq
            
            result[i, j] = np.exp(-np.sqrt(wasserstein_sq) / tau) if wasserstein_sq >= 0 else 1.0
    
    return result


def python_obb_iou(boxes: np.ndarray, query_boxes: np.ndarray) -> np.ndarray:
    """Pure Python implementation of OBB IoU (matches obb_bbox_nb.cpp numerics)."""
    boxes = np.asarray(boxes, dtype=np.float64, order="C")
    query_boxes = np.asarray(query_boxes, dtype=np.float64, order="C")
    n, k = boxes.shape[0], query_boxes.shape[0]
    result = np.zeros((n, k), dtype=np.float64)
    eps_aa = 1e-6  # axis-aligned threshold, same as C++

    def obb_to_corners(cx, cy, width, height, angle):
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        hw = width * 0.5
        hh = height * 0.5
        local_x = np.array([-hw, hw, hw, -hw], dtype=np.float64)
        local_y = np.array([-hh, -hh, hh, hh], dtype=np.float64)
        corners_x = cx + local_x * cos_a - local_y * sin_a
        corners_y = cy + local_x * sin_a + local_y * cos_a
        return corners_x, corners_y

    for i in range(n):
        for j in range(k):
            cx1, cy1, w1, h1, angle1 = boxes[i]
            cx2, cy2, w2, h2, angle2 = query_boxes[j]

            if abs(angle1) < eps_aa and abs(angle2) < eps_aa:
                x1_min = cx1 - w1 * 0.5
                x1_max = cx1 + w1 * 0.5
                y1_min = cy1 - h1 * 0.5
                y1_max = cy1 + h1 * 0.5
                x2_min = cx2 - w2 * 0.5
                x2_max = cx2 + w2 * 0.5
                y2_min = cy2 - h2 * 0.5
                y2_max = cy2 + h2 * 0.5
                inter_x_min = max(x1_min, x2_min)
                inter_x_max = min(x1_max, x2_max)
                inter_y_min = max(y1_min, y2_min)
                inter_y_max = min(y1_max, y2_max)
                if inter_x_min >= inter_x_max or inter_y_min >= inter_y_max:
                    intersection = 0.0
                else:
                    intersection = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
            else:
                corners1_x, corners1_y = obb_to_corners(cx1, cy1, w1, h1, angle1)
                corners2_x, corners2_y = obb_to_corners(cx2, cy2, w2, h2, angle2)
                min_x1 = float(np.min(corners1_x))
                max_x1 = float(np.max(corners1_x))
                min_y1 = float(np.min(corners1_y))
                max_y1 = float(np.max(corners1_y))
                min_x2 = float(np.min(corners2_x))
                max_x2 = float(np.max(corners2_x))
                min_y2 = float(np.min(corners2_y))
                max_y2 = float(np.max(corners2_y))
                inter_x_min = max(min_x1, min_x2)
                inter_x_max = min(max_x1, max_x2)
                inter_y_min = max(min_y1, min_y2)
                inter_y_max = min(max_y1, max_y2)
                if inter_x_min >= inter_x_max or inter_y_min >= inter_y_max:
                    intersection = 0.0
                else:
                    aabb_intersection = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
                    angle_factor = max(
                        0.5, float(np.cos(abs(angle1)) * np.cos(abs(angle2)))
                    )
                    intersection = aabb_intersection * angle_factor

            area1 = float(w1 * h1)
            area2 = float(w2 * h2)
            union = area1 + area2 - intersection
            result[i, j] = intersection / union if union > 0 else 0.0

    return result


# =============================================================================
# Test Data Generation
# =============================================================================

def generate_boxes(n: int, seed: int = 42) -> np.ndarray:
    """Generate random axis-aligned bounding boxes."""
    np.random.seed(seed)
    x1 = np.random.uniform(0, 800, n)
    y1 = np.random.uniform(0, 800, n)
    x2 = x1 + np.random.uniform(10, 200, n)
    y2 = y1 + np.random.uniform(10, 200, n)
    return np.column_stack([x1, y1, x2, y2]).astype(np.float32)


def generate_obb_boxes(n: int, seed: int = 42) -> np.ndarray:
    """Generate random oriented bounding boxes [cx, cy, w, h, angle]."""
    np.random.seed(seed)
    cx = np.random.uniform(50, 950, n)
    cy = np.random.uniform(50, 950, n)
    w = np.random.uniform(10, 200, n)
    h = np.random.uniform(10, 200, n)
    angle = np.random.uniform(0, 2 * np.pi, n)
    return np.column_stack([cx, cy, w, h, angle]).astype(np.float64)


# =============================================================================
# Test Runner
# =============================================================================

class TestResult:
    """Container for test results."""
    def __init__(self, name: str, passed: bool, max_diff: float, mean_diff: float,
                 expected_val: float = None, actual_val: float = None):
        self.name = name
        self.passed = passed
        self.max_diff = max_diff
        self.mean_diff = mean_diff
        self.expected_val = expected_val
        self.actual_val = actual_val


def compare_results(python_result: np.ndarray, fast_result: np.ndarray, 
                    tolerance: float) -> Tuple[bool, float, float]:
    """Compare Python and fastbbox results."""
    diff = np.abs(python_result - fast_result)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    passed = max_diff < tolerance
    return passed, max_diff, mean_diff


def run_function_tests(name: str, python_func: Callable, fast_func: Callable,
                       boxes: np.ndarray, query_boxes: np.ndarray,
                       tolerance: float, verbose: bool, extra_args: tuple = ()) -> List[TestResult]:
    """Run tests for a single function."""
    results = []
    
    # Test 1: Identical boxes
    identical_boxes = np.array([[100, 100, 200, 200]], dtype=np.float32)
    py_result = python_func(identical_boxes, identical_boxes, *extra_args)
    fast_result = fast_func(identical_boxes, identical_boxes, *extra_args)
    passed, max_diff, mean_diff = compare_results(py_result, fast_result, tolerance)
    results.append(TestResult(f"{name}: identical boxes", passed, max_diff, mean_diff,
                              py_result[0, 0], fast_result[0, 0]))
    
    # Test 2: Non-overlapping boxes
    box1 = np.array([[0, 0, 10, 10]], dtype=np.float32)
    box2 = np.array([[100, 100, 110, 110]], dtype=np.float32)
    py_result = python_func(box1, box2, *extra_args)
    fast_result = fast_func(box1, box2, *extra_args)
    passed, max_diff, mean_diff = compare_results(py_result, fast_result, tolerance)
    results.append(TestResult(f"{name}: non-overlapping", passed, max_diff, mean_diff,
                              py_result[0, 0], fast_result[0, 0]))
    
    # Test 3: Partial overlap
    box1 = np.array([[0, 0, 10, 10]], dtype=np.float32)
    box2 = np.array([[5, 5, 15, 15]], dtype=np.float32)
    py_result = python_func(box1, box2, *extra_args)
    fast_result = fast_func(box1, box2, *extra_args)
    passed, max_diff, mean_diff = compare_results(py_result, fast_result, tolerance)
    results.append(TestResult(f"{name}: partial overlap", passed, max_diff, mean_diff,
                              py_result[0, 0], fast_result[0, 0]))
    
    # Test 4: Batch processing (main comparison)
    py_result = python_func(boxes, query_boxes, *extra_args)
    fast_result = fast_func(boxes, query_boxes, *extra_args)
    passed, max_diff, mean_diff = compare_results(py_result, fast_result, tolerance)
    results.append(TestResult(f"{name}: batch ({len(boxes)}x{len(query_boxes)})", 
                              passed, max_diff, mean_diff))
    
    # Test 5: Edge case - tiny boxes
    tiny = np.array([[0, 0, 0.1, 0.1]], dtype=np.float32)
    tiny2 = np.array([[0.05, 0.05, 0.15, 0.15]], dtype=np.float32)
    py_result = python_func(tiny, tiny2, *extra_args)
    fast_result = fast_func(tiny, tiny2, *extra_args)
    passed, max_diff, mean_diff = compare_results(py_result, fast_result, tolerance)
    results.append(TestResult(f"{name}: tiny boxes", passed, max_diff, mean_diff,
                              py_result[0, 0], fast_result[0, 0]))
    
    return results


def run_obb_tests(python_func: Callable, fast_func: Callable,
                  boxes: np.ndarray, query_boxes: np.ndarray,
                  tolerance: float, verbose: bool) -> List[TestResult]:
    """Run tests for OBB IoU function."""
    results = []
    
    # Test 1: Identical OBB boxes
    identical = np.array([[100, 100, 50, 30, 0]], dtype=np.float64)
    py_result = python_func(identical, identical)
    fast_result = fast_func(identical, identical)
    passed, max_diff, mean_diff = compare_results(py_result, fast_result, tolerance)
    results.append(TestResult("OBB: identical boxes", passed, max_diff, mean_diff,
                              py_result[0, 0], fast_result[0, 0]))
    
    # Test 2: Non-overlapping OBB
    box1 = np.array([[0, 0, 10, 10, 0]], dtype=np.float64)
    box2 = np.array([[100, 100, 10, 10, 0]], dtype=np.float64)
    py_result = python_func(box1, box2)
    fast_result = fast_func(box1, box2)
    passed, max_diff, mean_diff = compare_results(py_result, fast_result, tolerance)
    results.append(TestResult("OBB: non-overlapping", passed, max_diff, mean_diff,
                              py_result[0, 0], fast_result[0, 0]))
    
    # Test 3: Rotated boxes
    box1 = np.array([[50, 50, 40, 20, 0]], dtype=np.float64)
    box2 = np.array([[50, 50, 40, 20, np.pi/4]], dtype=np.float64)
    py_result = python_func(box1, box2)
    fast_result = fast_func(box1, box2)
    passed, max_diff, mean_diff = compare_results(py_result, fast_result, tolerance)
    results.append(TestResult("OBB: rotated 45deg", passed, max_diff, mean_diff,
                              py_result[0, 0], fast_result[0, 0]))
    
    # Test 4: Batch processing
    py_result = python_func(boxes, query_boxes)
    fast_result = fast_func(boxes, query_boxes)
    passed, max_diff, mean_diff = compare_results(py_result, fast_result, tolerance)
    results.append(TestResult(f"OBB: batch ({len(boxes)}x{len(query_boxes)})", 
                              passed, max_diff, mean_diff))
    
    return results


def print_results(results: List[TestResult], verbose: bool):
    """Print test results."""
    all_passed = all(r.passed for r in results)
    
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        symbol = "+" if r.passed else "x"
        
        if verbose:
            print(f"  [{symbol}] {r.name}")
            print(f"      Max diff: {r.max_diff:.2e}, Mean diff: {r.mean_diff:.2e}")
            if r.expected_val is not None:
                print(f"      Expected: {r.expected_val:.6f}, Got: {r.actual_val:.6f}")
        else:
            print(f"  [{symbol}] {r.name}: max_diff={r.max_diff:.2e} [{status}]")
    
    return all_passed


def main():
    parser = argparse.ArgumentParser(description="FastBBox Correctness Test Suite")
    parser.add_argument("--verbose", "-v", action="store_true", 
                        help="Show detailed test output")
    parser.add_argument("--function", "-f", nargs="+", 
                        choices=["iou", "giou", "diou", "ciou", "eiou", "nwd", "obb"],
                        help="Test specific function(s)")
    parser.add_argument("--tolerance", "-t", type=float, default=1e-5,
                        help="Tolerance threshold for comparisons (default: 1e-5)")
    parser.add_argument("--obb-tolerance", type=float, default=1e-9,
                        help="Tolerance for OBB tests; reference uses float64 matching C++ (default: 1e-9)")
    parser.add_argument("--size", "-s", type=int, default=1000,
                        help="Number of test boxes (default: 1000)")
    args = parser.parse_args()
    
    print("=" * 70)
    print("FASTBBOX CORRECTNESS TEST SUITE")
    print("=" * 70)
    
    # Import fastbbox
    try:
        from fastbbox import (bbox_overlaps, generalized_iou, distance_iou, 
                              complete_iou, efficient_iou, normalized_wasserstein_distance,
                              bbox_overlaps_obb)
        print(f"Backend: Nanobind")
    except ImportError as e:
        print(f"ERROR: Could not import fastbbox: {e}")
        return 1
    
    print(f"Tolerance: {args.tolerance} (OBB: {args.obb_tolerance})")
    print(f"Test size: {args.size} boxes")
    print("=" * 70)
    
    # Generate test data
    boxes = generate_boxes(args.size, seed=42)
    query_boxes = generate_boxes(args.size // 2, seed=123)
    obb_boxes = generate_obb_boxes(args.size, seed=42)
    obb_query = generate_obb_boxes(args.size // 2, seed=123)
    
    # Define test functions
    test_functions = {
        "iou": ("IoU", python_iou, bbox_overlaps, ()),
        "giou": ("GIoU", python_giou, generalized_iou, ()),
        "diou": ("DIoU", python_diou, distance_iou, ()),
        "ciou": ("CIoU", python_ciou, complete_iou, ()),
        "eiou": ("EIoU", python_eiou, efficient_iou, ()),
        "nwd": ("NWD", python_nwd, normalized_wasserstein_distance, (1.0,)),
    }
    
    # Filter functions if specified
    if args.function:
        test_functions = {k: v for k, v in test_functions.items() if k in args.function}
        run_obb = "obb" in args.function
    else:
        run_obb = True
    
    all_passed = True
    total_tests = 0
    passed_tests = 0
    
    # Run bbox tests
    for key, (name, py_func, fast_func, extra_args) in test_functions.items():
        print(f"\n{name} Tests:")
        print("-" * 40)
        results = run_function_tests(name, py_func, fast_func, boxes, query_boxes,
                                     args.tolerance, args.verbose, extra_args)
        func_passed = print_results(results, args.verbose)
        all_passed = all_passed and func_passed
        total_tests += len(results)
        passed_tests += sum(1 for r in results if r.passed)
    
    # Run OBB tests (uses separate tolerance - OBB is approximation-based)
    if run_obb:
        print(f"\nOBB IoU Tests:")
        print("-" * 40)
        results = run_obb_tests(python_obb_iou, bbox_overlaps_obb, 
                                obb_boxes, obb_query, args.obb_tolerance, args.verbose)
        func_passed = print_results(results, args.verbose)
        all_passed = all_passed and func_passed
        total_tests += len(results)
        passed_tests += sum(1 for r in results if r.passed)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Tests passed: {passed_tests}/{total_tests}")
    
    if all_passed:
        print("\nAll tests PASSED - fastbbox matches Python reference implementations")
        return 0
    else:
        print("\nSome tests FAILED - check implementations")
        return 1


if __name__ == "__main__":
    sys.exit(main())
