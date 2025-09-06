#!/usr/bin/env python3
"""
Comprehensive demonstration of all IoU variants in FastBBox
"""

import numpy as np
from fastbbox import bbox_overlaps, generalized_iou, distance_iou, complete_iou


def demo_all_variants():
    """Compare all IoU variants across different scenarios"""
    print("=== FastBBox: Complete IoU Variants Demo ===\n")
    
    scenarios = [
        {
            "name": "Perfect Overlap",
            "boxes": np.array([[0, 0, 10, 10]], dtype=np.float32),
            "queries": np.array([[0, 0, 10, 10]], dtype=np.float32),
        },
        {
            "name": "Partial Overlap - Same Centers",
            "boxes": np.array([[0, 0, 10, 10]], dtype=np.float32),
            "queries": np.array([[-2, -2, 12, 12]], dtype=np.float32),
        },
        {
            "name": "Partial Overlap - Offset Centers",
            "boxes": np.array([[0, 0, 10, 10]], dtype=np.float32),
            "queries": np.array([[5, 5, 15, 15]], dtype=np.float32),
        },
        {
            "name": "Different Aspect Ratios",
            "boxes": np.array([[0, 0, 10, 10]], dtype=np.float32),  # Square
            "queries": np.array([[0, 0, 20, 5]], dtype=np.float32),   # Wide rectangle
        },
        {
            "name": "Adjacent Boxes",
            "boxes": np.array([[0, 0, 10, 10]], dtype=np.float32),
            "queries": np.array([[10, 0, 20, 10]], dtype=np.float32),
        },
        {
            "name": "Separated Boxes",
            "boxes": np.array([[0, 0, 10, 10]], dtype=np.float32),
            "queries": np.array([[20, 20, 30, 30]], dtype=np.float32),
        },
        {
            "name": "One Inside Another",
            "boxes": np.array([[0, 0, 20, 20]], dtype=np.float32),   # Large
            "queries": np.array([[5, 5, 15, 15]], dtype=np.float32), # Small inside
        }
    ]
    
    print(f"{'Scenario':<25} {'IoU':<8} {'GIoU':<8} {'DIoU':<8} {'CIoU':<8}")
    print("-" * 65)
    
    for scenario in scenarios:
        boxes = scenario["boxes"]
        queries = scenario["queries"]
        
        iou = bbox_overlaps(boxes, queries)[0, 0]
        giou = generalized_iou(boxes, queries)[0, 0]
        diou = distance_iou(boxes, queries)[0, 0]
        ciou = complete_iou(boxes, queries)[0, 0]
        
        print(f"{scenario['name']:<25} {iou:<8.3f} {giou:<8.3f} {diou:<8.3f} {ciou:<8.3f}")
    
    print()


def demo_penalty_analysis():
    """Analyze how different penalties affect the metrics"""
    print("=== Penalty Analysis ===\n")
    
    # Base case: overlapping boxes
    base_box = np.array([[0, 0, 10, 10]], dtype=np.float32)
    
    test_cases = [
        {
            "description": "Same IoU, closer centers",
            "query": np.array([[0, 5, 10, 15]], dtype=np.float32),  # Vertical shift
        },
        {
            "description": "Same IoU, farther centers", 
            "query": np.array([[5, 5, 15, 15]], dtype=np.float32),  # Diagonal shift
        },
        {
            "description": "Same IoU, different aspect ratio",
            "query": np.array([[0, 2.5, 10, 17.5]], dtype=np.float32),  # Tall overlap
        }
    ]
    
    print("Analyzing how center distance and aspect ratio affect metrics:")
    print(f"{'Case':<30} {'IoU':<8} {'DIoU':<8} {'CIoU':<8} {'Center Dist':<12} {'Aspect Diff':<12}")
    print("-" * 90)
    
    for case in test_cases:
        query = case["query"]
        
        iou = bbox_overlaps(base_box, query)[0, 0]
        diou = distance_iou(base_box, query)[0, 0]
        ciou = complete_iou(base_box, query)[0, 0]
        
        # Calculate center distance
        box_center = np.array([5.0, 5.0])  # Center of base_box
        query_center = np.array([(query[0,0] + query[0,2])/2, (query[0,1] + query[0,3])/2])
        center_dist = np.linalg.norm(box_center - query_center)
        
        # Calculate aspect ratio difference
        box_aspect = 10.0 / 10.0  # 1.0
        query_w = query[0,2] - query[0,0]
        query_h = query[0,3] - query[0,1]
        query_aspect = query_w / query_h
        aspect_diff = abs(box_aspect - query_aspect)
        
        print(f"{case['description']:<30} {iou:<8.3f} {diou:<8.3f} {ciou:<8.3f} {center_dist:<12.3f} {aspect_diff:<12.3f}")
    
    print()


def demo_batch_processing():
    """Demonstrate efficient batch processing"""
    print("=== Batch Processing Performance ===\n")
    
    # Create larger dataset
    np.random.seed(42)
    
    # Random detection boxes
    n_detections = 100
    detections = np.random.uniform(0, 100, (n_detections, 4)).astype(np.float32)
    detections[:, 2] += detections[:, 0] + np.random.uniform(5, 20, n_detections)  # Add width
    detections[:, 3] += detections[:, 1] + np.random.uniform(5, 20, n_detections)  # Add height
    
    # Random ground truth boxes
    n_gt = 50
    ground_truth = np.random.uniform(0, 100, (n_gt, 4)).astype(np.float32)
    ground_truth[:, 2] += ground_truth[:, 0] + np.random.uniform(5, 20, n_gt)
    ground_truth[:, 3] += ground_truth[:, 1] + np.random.uniform(5, 20, n_gt)
    
    print(f"Computing IoU variants for {n_detections} detections vs {n_gt} ground truth boxes...")
    print(f"Total comparisons: {n_detections * n_gt:,}")
    
    # Compute all metrics
    iou_matrix = bbox_overlaps(detections, ground_truth)
    giou_matrix = generalized_iou(detections, ground_truth)
    diou_matrix = distance_iou(detections, ground_truth)
    ciou_matrix = complete_iou(detections, ground_truth)
    
    print(f"\nResults summary:")
    print(f"{'Metric':<8} {'Min':<8} {'Max':<8} {'Mean':<8} {'Positive%':<10}")
    print("-" * 50)
    
    for name, matrix in [("IoU", iou_matrix), ("GIoU", giou_matrix), ("DIoU", diou_matrix), ("CIoU", ciou_matrix)]:
        positive_pct = (matrix > 0).sum() / matrix.size * 100
        print(f"{name:<8} {matrix.min():<8.3f} {matrix.max():<8.3f} {matrix.mean():<8.3f} {positive_pct:<10.1f}")
    
    # Find best matches using different metrics
    print(f"\nBest match comparison (first 5 detections):")
    print(f"{'Det':<4} {'IoU Best':<9} {'GIoU Best':<10} {'DIoU Best':<10} {'CIoU Best':<10}")
    print("-" * 50)
    
    for i in range(min(5, n_detections)):
        iou_best = np.argmax(iou_matrix[i])
        giou_best = np.argmax(giou_matrix[i])
        diou_best = np.argmax(diou_matrix[i])
        ciou_best = np.argmax(ciou_matrix[i])
        
        print(f"{i:<4} GT-{iou_best:<6} GT-{giou_best:<7} GT-{diou_best:<7} GT-{ciou_best:<7}")
    
    print()


def demo_practical_applications():
    """Show practical applications of different IoU variants"""
    print("=== Practical Applications ===\n")
    
    # Scenario 1: Object Detection Evaluation
    print("1. Object Detection Evaluation:")
    detections = np.array([
        [10, 10, 50, 40],    # Detection 1
        [25, 25, 65, 55],    # Detection 2 (slight offset)
        [100, 100, 130, 120] # Detection 3
    ], dtype=np.float32)
    
    ground_truth = np.array([
        [12, 12, 48, 38],    # GT 1 (close to Det 1)
        [20, 20, 60, 50],    # GT 2 (between Det 1 & 2)
        [105, 105, 135, 125] # GT 3 (close to Det 3)
    ], dtype=np.float32)
    
    iou_scores = bbox_overlaps(detections, ground_truth)
    
    print("IoU matrix (for evaluation at IoU=0.5 threshold):")
    for i in range(len(detections)):
        for j in range(len(ground_truth)):
            score = iou_scores[i, j]
            status = "✓" if score >= 0.5 else "✗"
            print(f"  Det{i+1} vs GT{j+1}: {score:.3f} {status}")
    
    # Scenario 2: Training Loss Selection
    print(f"\n2. Training Loss Comparison:")
    print("For the same detection vs ground truth pair:")
    
    det_box = detections[0:1]  # First detection
    gt_box = ground_truth[0:1]  # First ground truth
    
    iou = bbox_overlaps(det_box, gt_box)[0, 0]
    giou = generalized_iou(det_box, gt_box)[0, 0]
    diou = distance_iou(det_box, gt_box)[0, 0]
    ciou = complete_iou(det_box, gt_box)[0, 0]
    
    print(f"  IoU Loss:  {1 - iou:.3f} (1 - IoU)")
    print(f"  GIoU Loss: {1 - giou:.3f} (1 - GIoU)")
    print(f"  DIoU Loss: {1 - diou:.3f} (1 - DIoU)")
    print(f"  CIoU Loss: {1 - ciou:.3f} (1 - CIoU)")
    print(f"  → CIoU provides richest gradient information for training")


if __name__ == "__main__":
    demo_all_variants()
    demo_penalty_analysis()
    demo_batch_processing()
    demo_practical_applications()
    print("\n🎉 FastBBox complete IoU suite demonstration finished!")
    print("Ready for production use in object detection, tracking, and evaluation!")
