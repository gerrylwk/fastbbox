#!/usr/bin/env python3
"""
FastBBox Performance Benchmark
==============================

Benchmark fastbbox functions against pure Python implementations to measure
speedup and performance characteristics.

Usage:
    python benchmark_fastbbox.py              # Run benchmarks (summary mode)
    python benchmark_fastbbox.py --verbose    # Show individual run times
    python benchmark_fastbbox.py --size 1000  # Test with 1000 boxes
    python benchmark_fastbbox.py --runs 10    # Run 10 iterations
    python benchmark_fastbbox.py --function iou giou  # Benchmark specific functions
"""

import argparse
import numpy as np
import sys
import time
from typing import Callable, Dict, List, Tuple

# =============================================================================
# Python Reference Implementations (for benchmark comparison; float64 / C++ aligned)
# =============================================================================

def _xyxy_f64(a: np.ndarray) -> np.ndarray:
    """Coerce (N, 4) XYXY boxes to C-contiguous float64 (matches bbox_nb.cpp)."""
    return np.asarray(a, dtype=np.float64, order="C")


def python_iou(boxes: np.ndarray, query_boxes: np.ndarray) -> np.ndarray:
    """Pure Python implementation of IoU."""
    boxes = _xyxy_f64(boxes)
    query_boxes = _xyxy_f64(query_boxes)
    n, k = boxes.shape[0], query_boxes.shape[0]
    result = np.zeros((n, k), dtype=np.float64)
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
    """Pure Python implementation of GIoU."""
    boxes = _xyxy_f64(boxes)
    query_boxes = _xyxy_f64(query_boxes)
    n, k = boxes.shape[0], query_boxes.shape[0]
    result = np.zeros((n, k), dtype=np.float64)
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
    """Pure Python implementation of DIoU."""
    boxes = _xyxy_f64(boxes)
    query_boxes = _xyxy_f64(query_boxes)
    n, k = boxes.shape[0], query_boxes.shape[0]
    result = np.zeros((n, k), dtype=np.float64)
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
    """Pure Python implementation of CIoU."""
    boxes = _xyxy_f64(boxes)
    query_boxes = _xyxy_f64(query_boxes)
    n, k = boxes.shape[0], query_boxes.shape[0]
    result = np.zeros((n, k), dtype=np.float64)
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
    """Pure Python implementation of EIoU."""
    boxes = _xyxy_f64(boxes)
    query_boxes = _xyxy_f64(query_boxes)
    n, k = boxes.shape[0], query_boxes.shape[0]
    result = np.zeros((n, k), dtype=np.float64)
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
    """Pure Python implementation of NWD."""
    boxes = _xyxy_f64(boxes)
    query_boxes = _xyxy_f64(query_boxes)
    n, k = boxes.shape[0], query_boxes.shape[0]
    result = np.zeros((n, k), dtype=np.float64)
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
    eps_aa = 1e-6

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
    return np.column_stack([x1, y1, x2, y2]).astype(np.float64)


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
# Benchmark Utilities
# =============================================================================

class BenchmarkResult:
    """Container for benchmark results."""
    def __init__(self, name: str, python_times: List[float], fast_times: List[float]):
        self.name = name
        self.python_times = python_times
        self.fast_times = fast_times
        
        self.python_mean = np.mean(python_times)
        self.python_std = np.std(python_times)
        self.fast_mean = np.mean(fast_times)
        self.fast_std = np.std(fast_times)
        self.speedup = self.python_mean / self.fast_mean if self.fast_mean > 0 else float('inf')


def benchmark_function(python_func: Callable, fast_func: Callable,
                       boxes: np.ndarray, query_boxes: np.ndarray,
                       runs: int, extra_args: tuple = ()) -> Tuple[List[float], List[float]]:
    """Benchmark a function over multiple runs."""
    python_times = []
    fast_times = []
    
    # Warmup
    _ = fast_func(boxes[:10], query_boxes[:10], *extra_args)
    
    for _ in range(runs):
        # Python timing
        start = time.perf_counter()
        _ = python_func(boxes, query_boxes, *extra_args)
        python_times.append(time.perf_counter() - start)
        
        # Fastbbox timing
        start = time.perf_counter()
        _ = fast_func(boxes, query_boxes, *extra_args)
        fast_times.append(time.perf_counter() - start)
    
    return python_times, fast_times


def print_results(results: List[BenchmarkResult], verbose: bool):
    """Print benchmark results in table format."""
    print(f"\n{'Function':<8} {'Python (ms)':<14} {'FastBBox (ms)':<14} {'Speedup':<10}")
    print("-" * 50)
    
    total_python = 0
    total_fast = 0
    
    for r in results:
        python_ms = r.python_mean * 1000
        fast_ms = r.fast_mean * 1000
        total_python += r.python_mean
        total_fast += r.fast_mean
        
        if verbose:
            print(f"{r.name:<8} {python_ms:>7.2f} +/- {r.python_std*1000:>4.2f}  "
                  f"{fast_ms:>7.2f} +/- {r.fast_std*1000:>4.2f}  {r.speedup:>7.1f}x")
        else:
            print(f"{r.name:<8} {python_ms:>12.2f}  {fast_ms:>12.2f}  {r.speedup:>8.1f}x")
    
    print("-" * 50)
    overall_speedup = total_python / total_fast if total_fast > 0 else float('inf')
    print(f"{'TOTAL':<8} {total_python*1000:>12.2f}  {total_fast*1000:>12.2f}  {overall_speedup:>8.1f}x")
    
    return overall_speedup


def print_detailed_times(results: List[BenchmarkResult]):
    """Print detailed timing information for each run."""
    print("\nDetailed Run Times:")
    print("=" * 60)
    
    for r in results:
        print(f"\n{r.name}:")
        print(f"  Python runs (ms): {', '.join(f'{t*1000:.2f}' for t in r.python_times)}")
        print(f"  FastBBox runs (ms): {', '.join(f'{t*1000:.2f}' for t in r.fast_times)}")
        print(f"  Python: mean={r.python_mean*1000:.2f}ms, std={r.python_std*1000:.2f}ms")
        print(f"  FastBBox: mean={r.fast_mean*1000:.2f}ms, std={r.fast_std*1000:.2f}ms")


def main():
    parser = argparse.ArgumentParser(description="FastBBox Performance Benchmark")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed timing information")
    parser.add_argument("--function", "-f", nargs="+",
                        choices=["iou", "giou", "diou", "ciou", "eiou", "nwd", "obb"],
                        help="Benchmark specific function(s)")
    parser.add_argument("--size", "-s", type=int, default=1000,
                        help="Number of boxes to test (default: 500)")
    parser.add_argument("--runs", "-r", type=int, default=1,
                        help="Number of benchmark runs (default: 1)")
    args = parser.parse_args()
    
    print("=" * 60)
    print("FASTBBOX PERFORMANCE BENCHMARK")
    print("=" * 60)
    
    # Import fastbbox
    try:
        from fastbbox import (bbox_overlaps, generalized_iou, distance_iou,
                              complete_iou, efficient_iou, normalized_wasserstein_distance,
                              bbox_overlaps_obb)
        print(f"Backend: Nanobind")
    except ImportError as e:
        print(f"ERROR: Could not import fastbbox: {e}")
        return 1
    
    print(f"Test size: {args.size} x {args.size} boxes = {args.size * args.size:,} comparisons (float64)")
    print(f"Runs: {args.runs}")
    print("=" * 60)
    
    # Generate test data
    boxes = generate_boxes(args.size, seed=42)
    query_boxes = generate_boxes(args.size, seed=123)
    obb_boxes = generate_obb_boxes(args.size, seed=42)
    obb_query = generate_obb_boxes(args.size, seed=123)
    
    # Define benchmark functions
    benchmark_functions = {
        "iou": ("IoU", python_iou, bbox_overlaps, (), boxes, query_boxes),
        "giou": ("GIoU", python_giou, generalized_iou, (), boxes, query_boxes),
        "diou": ("DIoU", python_diou, distance_iou, (), boxes, query_boxes),
        "ciou": ("CIoU", python_ciou, complete_iou, (), boxes, query_boxes),
        "eiou": ("EIoU", python_eiou, efficient_iou, (), boxes, query_boxes),
        "nwd": ("NWD", python_nwd, normalized_wasserstein_distance, (1.0,), boxes, query_boxes),
        "obb": ("OBB", python_obb_iou, bbox_overlaps_obb, (), obb_boxes, obb_query),
    }
    
    # Filter functions if specified
    if args.function:
        benchmark_functions = {k: v for k, v in benchmark_functions.items() if k in args.function}
    
    results = []
    
    print("\nRunning benchmarks...")
    for key, (name, py_func, fast_func, extra_args, test_boxes, test_query) in benchmark_functions.items():
        print(f"  {name}...", end=" ", flush=True)
        py_times, fast_times = benchmark_function(
            py_func, fast_func, test_boxes, test_query, args.runs, extra_args
        )
        result = BenchmarkResult(name, py_times, fast_times)
        results.append(result)
        print(f"{result.speedup:.1f}x speedup")
    
    # Print results
    overall_speedup = print_results(results, args.verbose)
    
    if args.verbose:
        print_detailed_times(results)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Overall speedup: {overall_speedup:.1f}x faster than Python")
    print(f"Total comparisons: {args.size * (args.size):,}")
    print(f"Functions benchmarked: {', '.join(r.name for r in results)}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
