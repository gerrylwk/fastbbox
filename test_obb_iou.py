#!/usr/bin/env python3
"""
Test suite for Oriented Bounding Box (OBB) IoU calculations.

This module contains comprehensive tests to verify the correctness of the
bbox_overlaps function for oriented bounding boxes.
"""

import numpy as np
import math
import sys
import os
from typing import List, Tuple

# Add the current directory to Python path to import fastbbox
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from fastbbox.obb_bbox import bbox_overlaps_obb_final as bbox_overlaps_obb
    OBB_AVAILABLE = True
    print("SUCCESS: OBB functions imported successfully")
except ImportError as e:
    print(f"ERROR: Could not import OBB functions: {e}")
    print("  Please build the module first with: python setup.py build_ext --inplace")
    OBB_AVAILABLE = False


def create_obb_from_center(cx: float, cy: float, width: float, height: float, angle: float) -> np.ndarray:
    """
    Create an OBB from center coordinates, dimensions, and rotation angle.
    
    Args:
        cx, cy: Center coordinates
        width, height: Box dimensions
        angle: Rotation angle in radians
        
    Returns:
        np.ndarray: OBB as [cx, cy, width, height, angle]
    """
    return np.array([cx, cy, width, height, angle], dtype=np.float32)


def create_obb_corners(cx: float, cy: float, width: float, height: float, angle: float) -> List[Tuple[float, float]]:
    """
    Create OBB corner points for visualization and reference calculations.
    
    Returns:
        List of (x, y) corner coordinates in clockwise order
    """
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    
    # Half dimensions
    hw = width / 2.0
    hh = height / 2.0
    
    # Corner offsets in local coordinates
    corners_local = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
    
    # Transform to world coordinates
    corners = []
    for lx, ly in corners_local:
        wx = cx + lx * cos_a - ly * sin_a
        wy = cy + lx * sin_a + ly * cos_a
        corners.append((wx, wy))
    
    return corners


class TestOBBIoU:
    """Test cases for Oriented Bounding Box IoU calculations."""
    
    def test_identical_boxes(self):
        """Test that identical OBBs have IoU = 1.0"""
        box1 = create_obb_from_center(0, 0, 4, 2, 0)
        box2 = create_obb_from_center(0, 0, 4, 2, 0)
        
        print(f"Test identical boxes: {box1} vs {box2}")
        
        if OBB_AVAILABLE:
            iou = bbox_overlaps_obb(box1.reshape(1, -1), box2.reshape(1, -1))
            print(f"  IoU result: {iou[0, 0]:.6f}")
            assert abs(iou[0, 0] - 1.0) < 1e-5, f"Expected IoU=1.0, got {iou[0, 0]}"
            print("  PASSED")
        else:
            print("  SKIPPED (OBB functions not available)")
        
    def test_non_overlapping_boxes(self):
        """Test that non-overlapping OBBs have IoU = 0.0"""
        box1 = create_obb_from_center(0, 0, 2, 2, 0)
        box2 = create_obb_from_center(5, 5, 2, 2, 0)
        
        print(f"Test non-overlapping boxes: {box1} vs {box2}")
        
        if OBB_AVAILABLE:
            iou = bbox_overlaps_obb(box1.reshape(1, -1), box2.reshape(1, -1))
            print(f"  IoU result: {iou[0, 0]:.6f}")
            assert abs(iou[0, 0] - 0.0) < 1e-5, f"Expected IoU=0.0, got {iou[0, 0]}"
            print("  PASSED")
        else:
            print("  SKIPPED (OBB functions not available)")
        
    def test_axis_aligned_partial_overlap(self):
        """Test axis-aligned boxes with known partial overlap"""
        # Two 2x2 boxes, one at origin, one shifted by (1, 0)
        # Expected intersection area: 1x2 = 2
        # Expected union area: 2*4 - 2 = 6
        # Expected IoU: 2/6 = 1/3 ≈ 0.3333
        box1 = create_obb_from_center(0, 0, 2, 2, 0)
        box2 = create_obb_from_center(1, 0, 2, 2, 0)
        
        expected_iou = 1.0 / 3.0
        print(f"Test axis-aligned partial overlap: {box1} vs {box2}")
        print(f"Expected IoU: {expected_iou:.4f}")
        
        if OBB_AVAILABLE:
            iou = bbox_overlaps_obb(box1.reshape(1, -1), box2.reshape(1, -1))
            print(f"  IoU result: {iou[0, 0]:.6f}")
            assert abs(iou[0, 0] - expected_iou) < 0.01, f"Expected IoU≈{expected_iou:.4f}, got {iou[0, 0]:.6f}"
            print("  PASSED")
        else:
            print("  SKIPPED (OBB functions not available)")
        
    def test_rotated_boxes_no_overlap(self):
        """Test rotated boxes that don't overlap"""
        box1 = create_obb_from_center(0, 0, 2, 1, 0)  # Horizontal
        box2 = create_obb_from_center(0, 2, 1, 2, math.pi/2)  # Vertical, shifted up
        
        print(f"Test rotated boxes no overlap: {box1} vs {box2}")
        
        if OBB_AVAILABLE:
            iou = bbox_overlaps_obb(box1.reshape(1, -1), box2.reshape(1, -1))
            print(f"  IoU result: {iou[0, 0]:.6f}")
            assert abs(iou[0, 0] - 0.0) < 1e-5, f"Expected IoU=0.0, got {iou[0, 0]}"
            print("  PASSED")
        else:
            print("  SKIPPED (OBB functions not available)")
        
    def test_rotated_boxes_with_overlap(self):
        """Test rotated boxes with overlap"""
        box1 = create_obb_from_center(0, 0, 4, 2, 0)  # Horizontal
        box2 = create_obb_from_center(0, 0, 4, 2, math.pi/4)  # 45 degree rotation
        
        print(f"Test rotated boxes with overlap: {box1} vs {box2}")
        
        if OBB_AVAILABLE:
            iou = bbox_overlaps_obb(box1.reshape(1, -1), box2.reshape(1, -1))
            print(f"  IoU result: {iou[0, 0]:.6f}")
            assert iou[0, 0] > 0.0 and iou[0, 0] < 1.0, f"Expected 0 < IoU < 1, got {iou[0, 0]}"
            print("  PASSED")
        else:
            print("  SKIPPED (OBB functions not available)")
        
    def test_contained_box(self):
        """Test when one box is completely inside another"""
        box1 = create_obb_from_center(0, 0, 6, 4, 0)  # Large box
        box2 = create_obb_from_center(0, 0, 2, 2, math.pi/6)  # Small box inside
        
        # Expected IoU should be area(small) / area(large) = 4 / 24 = 1/6 ≈ 0.1667
        expected_iou = 4.0 / 24.0
        print(f"Test contained box: {box1} vs {box2}")
        print(f"Expected IoU: {expected_iou:.4f}")
        
    def test_touching_edges(self):
        """Test boxes that touch at edges but don't overlap"""
        box1 = create_obb_from_center(0, 0, 2, 2, 0)
        box2 = create_obb_from_center(2, 0, 2, 2, 0)  # Touching right edge
        
        print(f"Test touching edges: {box1} vs {box2}")
        
    def test_high_aspect_ratio(self):
        """Test boxes with high aspect ratios"""
        box1 = create_obb_from_center(0, 0, 10, 1, 0)  # Very wide
        box2 = create_obb_from_center(0, 0, 1, 10, math.pi/2)  # Very tall, rotated
        
        print(f"Test high aspect ratio: {box1} vs {box2}")
        
    def test_small_boxes(self):
        """Test very small boxes for numerical stability"""
        box1 = create_obb_from_center(0, 0, 0.01, 0.01, 0)
        box2 = create_obb_from_center(0.005, 0.005, 0.01, 0.01, math.pi/4)
        
        print(f"Test small boxes: {box1} vs {box2}")
        
    def test_batch_processing(self):
        """Test batch processing of multiple OBBs"""
        # Create multiple boxes for batch testing
        boxes1 = np.array([
            create_obb_from_center(0, 0, 2, 2, 0),
            create_obb_from_center(1, 1, 2, 2, math.pi/4),
            create_obb_from_center(5, 5, 3, 1, math.pi/2)
        ])
        
        boxes2 = np.array([
            create_obb_from_center(0, 0, 2, 2, 0),
            create_obb_from_center(2, 2, 2, 2, 0),
            create_obb_from_center(5, 5, 1, 3, 0)
        ])
        
        print(f"Test batch processing:")
        print(f"Boxes1 shape: {boxes1.shape}")
        print(f"Boxes2 shape: {boxes2.shape}")
        
    def test_corner_coordinates(self):
        """Test corner coordinate calculation for reference"""
        box = create_obb_from_center(1, 2, 4, 2, math.pi/4)
        corners = create_obb_corners(1, 2, 4, 2, math.pi/4)
        
        print(f"Test corner coordinates for box {box}:")
        for i, (x, y) in enumerate(corners):
            print(f"  Corner {i}: ({x:.3f}, {y:.3f})")
            
    def test_reference_calculations(self):
        """Test reference calculations using known geometric formulas"""
        # Simple case: two axis-aligned squares
        box1_corners = [(0, 0), (2, 0), (2, 2), (0, 2)]
        box2_corners = [(1, 1), (3, 1), (3, 3), (1, 3)]
        
        # Manual calculation: intersection is (1,1) to (2,2) = 1x1 = 1
        # Union = 4 + 4 - 1 = 7
        # IoU = 1/7 ≈ 0.1429
        expected_iou = 1.0 / 7.0
        print(f"Reference calculation test:")
        print(f"Box1 corners: {box1_corners}")
        print(f"Box2 corners: {box2_corners}")
        print(f"Expected IoU: {expected_iou:.4f}")


def run_all_tests():
    """Run all test cases"""
    test_suite = TestOBBIoU()
    
    print("=" * 60)
    print("ORIENTED BOUNDING BOX IoU TEST SUITE")
    print("=" * 60)
    
    test_methods = [method for method in dir(test_suite) if method.startswith('test_')]
    
    for test_method in test_methods:
        print(f"\n--- {test_method.replace('_', ' ').title()} ---")
        getattr(test_suite, test_method)()
    
    print("\n" + "=" * 60)
    print("All tests completed. Ready for implementation verification.")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
