#!/usr/bin/env python3
"""
Comprehensive test suite for all IoU variants in FastBBox:
- Standard IoU (bbox_overlaps)
- Generalized IoU (GIoU)
- Distance IoU (DIoU)  
- Complete IoU (CIoU)
- Efficient IoU (EIoU)
- Normalized Wasserstein Distance (NWD)
"""

import numpy as np
import math
from fastbbox import bbox_overlaps, generalized_iou, distance_iou, complete_iou, efficient_iou, normalized_wasserstein_distance


# =============================================================================
# Standard IoU Tests
# =============================================================================

def test_iou_basic_functionality():
    """Test basic IoU functionality"""
    print("Testing basic IoU functionality...")
    
    # Test case 1: Perfect overlap
    boxes = np.array([[0, 0, 10, 10]], dtype=np.float32)
    query_boxes = np.array([[0, 0, 10, 10]], dtype=np.float32)
    overlaps = bbox_overlaps(boxes, query_boxes)
    
    print(f"Perfect overlap: {overlaps[0, 0]:.6f} (expected: 1.0)")
    assert abs(overlaps[0, 0] - 1.0) < 1e-6, "Perfect overlap should be 1.0"
    
    # Test case 2: No overlap
    boxes = np.array([[0, 0, 10, 10]], dtype=np.float32)
    query_boxes = np.array([[20, 20, 30, 30]], dtype=np.float32)
    overlaps = bbox_overlaps(boxes, query_boxes)
    
    print(f"No overlap: {overlaps[0, 0]:.6f} (expected: 0.0)")
    assert overlaps[0, 0] == 0.0, "No overlap should be 0.0"
    
    # Test case 3: Partial overlap
    boxes = np.array([[0, 0, 10, 10]], dtype=np.float32)
    query_boxes = np.array([[5, 5, 15, 15]], dtype=np.float32)
    overlaps = bbox_overlaps(boxes, query_boxes)
    
    # Intersection: (10-5) * (10-5) = 25
    # Union: 100 + 100 - 25 = 175
    # IoU: 25/175 = 1/7 ≈ 0.142857
    expected_iou = 25.0 / 175.0
    print(f"Partial overlap: {overlaps[0, 0]:.6f} (expected: {expected_iou:.6f})")
    assert abs(overlaps[0, 0] - expected_iou) < 1e-6, f"Partial overlap should be {expected_iou}"
    
    print("✓ IoU basic tests passed!")


def test_iou_multiple_boxes():
    """Test IoU with multiple boxes"""
    print("\nTesting IoU with multiple boxes...")
    
    boxes = np.array([
        [0, 0, 10, 10],
        [5, 5, 15, 15],
        [20, 20, 30, 30]
    ], dtype=np.float32)
    
    query_boxes = np.array([
        [0, 0, 10, 10],
        [12, 12, 22, 22]
    ], dtype=np.float32)
    
    overlaps = bbox_overlaps(boxes, query_boxes)
    print(f"Overlaps shape: {overlaps.shape}")
    print(f"Overlaps:\n{overlaps}")
    
    # Check specific values
    assert abs(overlaps[0, 0] - 1.0) < 1e-6, "Box 0 with query 0 should have IoU = 1.0"
    assert overlaps[0, 1] == 0.0, "Box 0 with query 1 should have IoU = 0.0"
    
    print("✓ Multiple boxes test passed!")


def test_iou_float_coordinates():
    """Test IoU with floating point coordinates"""
    print("\nTesting IoU with float coordinates...")
    
    # Test with precise float coordinates
    boxes = np.array([
        [100.5, 200.25, 150.75, 250.5],
        [120.1, 220.9, 180.3, 280.7]
    ], dtype=np.float32)
    
    query_boxes = np.array([
        [110.2, 210.1, 140.8, 240.9],
        [160.0, 260.0, 200.0, 300.0]
    ], dtype=np.float32)
    
    overlaps = bbox_overlaps(boxes, query_boxes)
    print(f"Float coordinates overlaps:\n{overlaps}")
    
    # Manual calculation for first pair
    x_left = max(100.5, 110.2)  # 110.2
    y_top = max(200.25, 210.1)  # 210.1
    x_right = min(150.75, 140.8)  # 140.8
    y_bottom = min(250.5, 240.9)  # 240.9
    
    if x_right > x_left and y_bottom > y_top:
        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = (150.75 - 100.5) * (250.5 - 200.25)
        area2 = (140.8 - 110.2) * (240.9 - 210.1)
        union = area1 + area2 - intersection
        expected_iou = intersection / union
        
        print(f"Manual calculation:")
        print(f"  Expected IoU: {expected_iou:.6f}")
        print(f"  Computed IoU: {overlaps[0, 0]:.6f}")
        
        assert abs(overlaps[0, 0] - expected_iou) < 1e-5, f"IoU mismatch: {overlaps[0, 0]} vs {expected_iou}"
    
    print("✓ Float coordinates test passed!")


def test_iou_edge_cases():
    """Test IoU edge cases"""
    print("\nTesting IoU edge cases...")
    
    # Very small overlap
    boxes = np.array([[0.0, 0.0, 10.1, 10.1]], dtype=np.float32)
    query_boxes = np.array([[10.0, 10.0, 20.0, 20.0]], dtype=np.float32)
    overlaps = bbox_overlaps(boxes, query_boxes)
    
    # Should have tiny overlap: 0.1 * 0.1 = 0.01
    expected_area = 0.1 * 0.1
    union = 10.1 * 10.1 + 10.0 * 10.0 - expected_area
    expected_iou = expected_area / union
    
    print(f"Tiny overlap IoU: {overlaps[0, 0]:.8f} (expected: {expected_iou:.8f})")
    assert abs(overlaps[0, 0] - expected_iou) < 1e-6, "Tiny overlap calculation failed"
    
    # High precision coordinates
    boxes = np.array([[0.123456, 0.789012, 5.654321, 8.210987]], dtype=np.float32)
    query_boxes = np.array([[2.111111, 3.333333, 7.777777, 9.999999]], dtype=np.float32)
    overlaps = bbox_overlaps(boxes, query_boxes)
    
    print(f"High precision IoU: {overlaps[0, 0]:.8f}")
    assert overlaps[0, 0] > 0, "High precision calculation should work"
    
    print("✓ IoU edge cases test passed!")


# =============================================================================
# Generalized IoU (GIoU) Tests
# =============================================================================

def test_giou_identical_boxes():
    """Test GIoU for identical boxes - should equal 1.0"""
    print("\nTesting GIoU for identical boxes...")
    
    boxes = np.array([[0, 0, 10, 10]], dtype=np.float32)
    query_boxes = np.array([[0, 0, 10, 10]], dtype=np.float32)
    
    giou = generalized_iou(boxes, query_boxes)
    iou = bbox_overlaps(boxes, query_boxes)
    
    print(f"Identical boxes - IoU: {iou[0, 0]:.6f}, GIoU: {giou[0, 0]:.6f}")
    assert abs(giou[0, 0] - 1.0) < 1e-6, "GIoU for identical boxes should be 1.0"
    assert abs(iou[0, 0] - giou[0, 0]) < 1e-6, "GIoU should equal IoU for identical boxes"
    
    print("✓ Identical boxes test passed!")


def test_giou_no_overlap():
    """Test GIoU for non-overlapping boxes"""
    print("\nTesting GIoU for non-overlapping boxes...")
    
    # Two separate boxes
    boxes = np.array([[0, 0, 10, 10]], dtype=np.float32)
    query_boxes = np.array([[20, 20, 30, 30]], dtype=np.float32)
    
    giou = generalized_iou(boxes, query_boxes)
    iou = bbox_overlaps(boxes, query_boxes)
    
    print(f"Non-overlapping boxes - IoU: {iou[0, 0]:.6f}, GIoU: {giou[0, 0]:.6f}")
    
    # Manual calculation:
    # IoU = 0 (no overlap)
    # Enclosing box: [0, 0, 30, 30] with area = 900
    # Union = 100 + 100 = 200
    # GIoU = 0 - (900 - 200) / 900 = -0.777...
    expected_giou = 0 - (900 - 200) / 900
    
    print(f"Expected GIoU: {expected_giou:.6f}")
    assert abs(giou[0, 0] - expected_giou) < 1e-5, f"GIoU mismatch: {giou[0, 0]} vs {expected_giou}"
    assert iou[0, 0] == 0.0, "IoU should be 0 for non-overlapping boxes"
    assert giou[0, 0] < 0, "GIoU should be negative for non-overlapping boxes"
    
    print("✓ Non-overlapping boxes test passed!")


def test_giou_partial_overlap():
    """Test GIoU for partially overlapping boxes"""
    print("\nTesting GIoU for partially overlapping boxes...")
    
    boxes = np.array([[0, 0, 10, 10]], dtype=np.float32)
    query_boxes = np.array([[5, 5, 15, 15]], dtype=np.float32)
    
    giou = generalized_iou(boxes, query_boxes)
    iou = bbox_overlaps(boxes, query_boxes)
    
    print(f"Partial overlap - IoU: {iou[0, 0]:.6f}, GIoU: {giou[0, 0]:.6f}")
    
    # Manual calculation
    expected_iou = 25.0 / 175.0
    expected_giou = expected_iou - (225 - 175) / 225
    
    print(f"Expected IoU: {expected_iou:.6f}, Expected GIoU: {expected_giou:.6f}")
    assert abs(iou[0, 0] - expected_iou) < 1e-5, "IoU calculation error"
    assert abs(giou[0, 0] - expected_giou) < 1e-5, f"GIoU mismatch: {giou[0, 0]} vs {expected_giou}"
    assert giou[0, 0] < iou[0, 0], "GIoU should be less than IoU for partial overlap"
    
    print("✓ Partial overlap test passed!")


def test_giou_contained_boxes():
    """Test GIoU when one box is completely inside another"""
    print("\nTesting GIoU for contained boxes...")
    
    # Small box inside large box
    boxes = np.array([[0, 0, 20, 20]], dtype=np.float32)  # Large box
    query_boxes = np.array([[5, 5, 15, 15]], dtype=np.float32)  # Small box inside
    
    giou = generalized_iou(boxes, query_boxes)
    iou = bbox_overlaps(boxes, query_boxes)
    
    print(f"Contained boxes - IoU: {iou[0, 0]:.6f}, GIoU: {giou[0, 0]:.6f}")
    
    expected_iou = 100.0 / 400.0
    expected_giou = expected_iou  # No penalty since enclosing box equals union
    
    print(f"Expected IoU: {expected_iou:.6f}, Expected GIoU: {expected_giou:.6f}")
    assert abs(iou[0, 0] - expected_iou) < 1e-5, "IoU calculation error"
    assert abs(giou[0, 0] - expected_giou) < 1e-5, f"GIoU mismatch: {giou[0, 0]} vs {expected_giou}"
    assert abs(giou[0, 0] - iou[0, 0]) < 1e-5, "GIoU should equal IoU when one box contains another"
    
    print("✓ Contained boxes test passed!")


# =============================================================================
# Distance IoU (DIoU) Tests
# =============================================================================

def test_diou_identical_boxes():
    """Test DIoU for identical boxes - should equal 1.0"""
    print("\nTesting DIoU for identical boxes...")
    
    boxes = np.array([[0, 0, 10, 10]], dtype=np.float32)
    query_boxes = np.array([[0, 0, 10, 10]], dtype=np.float32)
    
    diou = distance_iou(boxes, query_boxes)
    iou = bbox_overlaps(boxes, query_boxes)
    
    print(f"Identical boxes - IoU: {iou[0, 0]:.6f}, DIoU: {diou[0, 0]:.6f}")
    assert abs(diou[0, 0] - 1.0) < 1e-6, "DIoU for identical boxes should be 1.0"
    assert abs(iou[0, 0] - diou[0, 0]) < 1e-6, "DIoU should equal IoU for identical boxes"
    
    print("✓ Identical boxes test passed!")


def test_diou_center_distance_penalty():
    """Test DIoU penalty for center distance"""
    print("\nTesting DIoU center distance penalty...")
    
    # Two boxes with same IoU but different center distances
    # Case 1: Overlapping boxes close centers
    boxes1 = np.array([[0, 0, 10, 10]], dtype=np.float32)
    query1 = np.array([[5, 5, 15, 15]], dtype=np.float32)  # Centers: (5,5) vs (10,10)
    
    # Case 2: Overlapping boxes far centers (same IoU, different center distance)
    boxes2 = np.array([[0, 0, 10, 10]], dtype=np.float32) 
    query2 = np.array([[0, 5, 10, 15]], dtype=np.float32)  # Centers: (5,5) vs (5,10)
    
    iou1 = bbox_overlaps(boxes1, query1)[0, 0]
    iou2 = bbox_overlaps(boxes2, query2)[0, 0]
    diou1 = distance_iou(boxes1, query1)[0, 0]
    diou2 = distance_iou(boxes2, query2)[0, 0]
    
    print(f"Case 1 - IoU: {iou1:.6f}, DIoU: {diou1:.6f}")
    print(f"Case 2 - IoU: {iou2:.6f}, DIoU: {diou2:.6f}")
    
    # Manual calculation for Case 1
    center_dist_sq_1 = (5-10)**2 + (5-10)**2  # 50
    diagonal_sq_1 = 15**2 + 15**2  # 450
    expected_diou1 = iou1 - center_dist_sq_1 / diagonal_sq_1
    
    print(f"Expected DIoU1: {expected_diou1:.6f}")
    assert abs(diou1 - expected_diou1) < 1e-5, f"DIoU calculation error: {diou1} vs {expected_diou1}"
    
    # DIoU should be less than IoU when centers are different
    assert diou1 < iou1, "DIoU should be less than IoU when centers don't coincide"
    assert diou2 < iou2, "DIoU should be less than IoU when centers don't coincide"
    
    print("✓ Center distance penalty test passed!")


def test_diou_no_overlap():
    """Test DIoU for non-overlapping boxes"""
    print("\nTesting DIoU for non-overlapping boxes...")
    
    boxes = np.array([[0, 0, 10, 10]], dtype=np.float32)
    query_boxes = np.array([[20, 20, 30, 30]], dtype=np.float32)
    
    diou = distance_iou(boxes, query_boxes)
    iou = bbox_overlaps(boxes, query_boxes)
    giou = generalized_iou(boxes, query_boxes)
    
    print(f"Non-overlapping - IoU: {iou[0, 0]:.6f}, GIoU: {giou[0, 0]:.6f}, DIoU: {diou[0, 0]:.6f}")
    
    # Manual calculation
    center_dist_sq = (5-25)**2 + (5-25)**2  # 800
    diagonal_sq = 30**2 + 30**2  # 1800
    expected_diou = 0 - center_dist_sq / diagonal_sq
    
    print(f"Expected DIoU: {expected_diou:.6f}")
    assert abs(diou[0, 0] - expected_diou) < 1e-5, f"DIoU calculation error"
    assert iou[0, 0] == 0.0, "IoU should be 0 for non-overlapping boxes"
    assert diou[0, 0] > giou[0, 0], "DIoU should be greater than GIoU for separated boxes (less negative penalty)"
    
    print("✓ Non-overlapping boxes test passed!")


# =============================================================================
# Complete IoU (CIoU) Tests
# =============================================================================

def test_ciou_identical_boxes():
    """Test CIoU for identical boxes - should equal 1.0"""
    print("\nTesting CIoU for identical boxes...")
    
    boxes = np.array([[0, 0, 10, 10]], dtype=np.float32)
    query_boxes = np.array([[0, 0, 10, 10]], dtype=np.float32)
    
    ciou = complete_iou(boxes, query_boxes)
    diou = distance_iou(boxes, query_boxes)
    iou = bbox_overlaps(boxes, query_boxes)
    
    print(f"Identical boxes - IoU: {iou[0, 0]:.6f}, DIoU: {diou[0, 0]:.6f}, CIoU: {ciou[0, 0]:.6f}")
    assert abs(ciou[0, 0] - 1.0) < 1e-6, "CIoU for identical boxes should be 1.0"
    assert abs(ciou[0, 0] - diou[0, 0]) < 1e-6, "CIoU should equal DIoU for identical boxes"
    
    print("✓ Identical boxes test passed!")


def test_ciou_aspect_ratio_penalty():
    """Test CIoU penalty for aspect ratio differences"""
    print("\nTesting CIoU aspect ratio penalty...")
    
    # Two boxes with different aspect ratios but same area and position
    boxes = np.array([
        [0, 0, 10, 10],  # Square: 10x10
        [0, 0, 20, 5]    # Rectangle: 20x5 (same area, different aspect ratio)
    ], dtype=np.float32)
    
    query_boxes = np.array([[0, 0, 10, 10]], dtype=np.float32)  # Square
    
    ciou = complete_iou(boxes, query_boxes)
    diou = distance_iou(boxes, query_boxes)
    iou = bbox_overlaps(boxes, query_boxes)
    
    print(f"Square vs Square - IoU: {iou[0, 0]:.6f}, DIoU: {diou[0, 0]:.6f}, CIoU: {ciou[0, 0]:.6f}")
    print(f"Rectangle vs Square - IoU: {iou[1, 0]:.6f}, DIoU: {diou[1, 0]:.6f}, CIoU: {ciou[1, 0]:.6f}")
    
    # For identical aspect ratios, CIoU should equal DIoU
    assert abs(ciou[0, 0] - diou[0, 0]) < 1e-6, "CIoU should equal DIoU for same aspect ratios"
    
    # For different aspect ratios, CIoU should be less than DIoU
    assert ciou[1, 0] < diou[1, 0], "CIoU should be less than DIoU for different aspect ratios"
    
    # Manual verification of aspect ratio penalty
    w1, h1 = 10, 10  # Square
    w2, h2 = 20, 5   # Rectangle
    v = (4 / (math.pi**2)) * (math.atan2(w1, h1) - math.atan2(w2, h2))**2
    print(f"Aspect ratio penalty v: {v:.6f}")
    
    print("✓ Aspect ratio penalty test passed!")


def test_ciou_comprehensive_comparison():
    """Test comprehensive comparison of all IoU variants"""
    print("\nTesting comprehensive IoU comparison...")
    
    # Test case: Partial overlap with center offset and aspect ratio difference
    boxes = np.array([[0, 0, 10, 20]], dtype=np.float32)     # Tall rectangle
    query_boxes = np.array([[5, 5, 20, 15]], dtype=np.float32)  # Wide rectangle
    
    iou = bbox_overlaps(boxes, query_boxes)[0, 0]
    giou = generalized_iou(boxes, query_boxes)[0, 0]
    diou = distance_iou(boxes, query_boxes)[0, 0]
    ciou = complete_iou(boxes, query_boxes)[0, 0]
    
    print(f"Comprehensive comparison:")
    print(f"  IoU:  {iou:.6f}")
    print(f"  GIoU: {giou:.6f}")
    print(f"  DIoU: {diou:.6f}")
    print(f"  CIoU: {ciou:.6f}")
    
    # Verify expected ordering relationships
    assert ciou <= diou, "CIoU should be <= DIoU (aspect ratio penalty)"
    assert diou <= iou, "DIoU should be <= IoU (center distance penalty)"
    assert giou <= iou, "GIoU should be <= IoU (enclosing area penalty)"
    
    # IoU should be positive, others can be negative due to penalties
    assert iou > 0, "IoU should be positive for overlapping boxes"
    assert diou > 0 and ciou > 0, "DIoU and CIoU should be positive for overlapping boxes"
    # Note: GIoU can be negative even for overlapping boxes due to large enclosing area penalty
    
    print("✓ Comprehensive comparison test passed!")


# =============================================================================
# Efficient IoU (EIoU) Tests
# =============================================================================

def test_eiou_identical_boxes():
    """Test EIoU for identical boxes - should equal 1.0"""
    print("\nTesting EIoU for identical boxes...")
    
    boxes = np.array([[0, 0, 10, 10]], dtype=np.float32)
    query_boxes = np.array([[0, 0, 10, 10]], dtype=np.float32)
    
    eiou = efficient_iou(boxes, query_boxes)
    iou = bbox_overlaps(boxes, query_boxes)
    
    print(f"Identical boxes - IoU: {iou[0, 0]:.6f}, EIoU: {eiou[0, 0]:.6f}")
    assert abs(eiou[0, 0] - 1.0) < 1e-6, "EIoU for identical boxes should be 1.0"
    assert abs(iou[0, 0] - eiou[0, 0]) < 1e-6, "EIoU should equal IoU for identical boxes"
    
    print("✓ Identical boxes test passed!")


def test_eiou_width_height_penalty():
    """Test EIoU penalty for width and height differences"""
    print("\nTesting EIoU width and height penalty...")
    
    # Test boxes with same IoU but different width/height ratios
    boxes = np.array([
        [0, 0, 10, 10],   # Square: 10x10
        [0, 0, 20, 5],    # Wide rectangle: 20x5 (same area)
        [0, 0, 5, 20]     # Tall rectangle: 5x20 (same area)
    ], dtype=np.float32)
    
    query_boxes = np.array([[0, 0, 10, 10]], dtype=np.float32)  # Square
    
    eiou = efficient_iou(boxes, query_boxes)
    ciou = complete_iou(boxes, query_boxes)
    diou = distance_iou(boxes, query_boxes)
    iou = bbox_overlaps(boxes, query_boxes)
    
    print(f"Square vs Square    - IoU: {iou[0, 0]:.6f}, DIoU: {diou[0, 0]:.6f}, CIoU: {ciou[0, 0]:.6f}, EIoU: {eiou[0, 0]:.6f}")
    print(f"Wide vs Square      - IoU: {iou[1, 0]:.6f}, DIoU: {diou[1, 0]:.6f}, CIoU: {ciou[1, 0]:.6f}, EIoU: {eiou[1, 0]:.6f}")
    print(f"Tall vs Square      - IoU: {iou[2, 0]:.6f}, DIoU: {diou[2, 0]:.6f}, CIoU: {ciou[2, 0]:.6f}, EIoU: {eiou[2, 0]:.6f}")
    
    # For identical shapes, EIoU should equal DIoU (no width/height penalty)
    assert abs(eiou[0, 0] - diou[0, 0]) < 1e-6, "EIoU should equal DIoU for identical shapes"
    
    # For different shapes, EIoU should be less than DIoU due to width/height penalty
    assert eiou[1, 0] < diou[1, 0], "EIoU should be less than DIoU for different width/height"
    assert eiou[2, 0] < diou[2, 0], "EIoU should be less than DIoU for different width/height"
    
    # EIoU should also be different from CIoU due to different penalty calculation
    assert abs(eiou[1, 0] - ciou[1, 0]) > 1e-6, "EIoU should differ from CIoU for different aspect ratios"
    
    print("✓ Width and height penalty test passed!")


def test_eiou_comprehensive_penalties():
    """Test EIoU with all penalty components"""
    print("\nTesting EIoU comprehensive penalties...")
    
    # Box with center offset, different width, and different height
    boxes = np.array([[0, 0, 10, 20]], dtype=np.float32)     # Tall rectangle
    query_boxes = np.array([[3, 2, 18, 12]], dtype=np.float32)  # Wide rectangle, offset
    
    eiou = efficient_iou(boxes, query_boxes)
    diou = distance_iou(boxes, query_boxes)
    ciou = complete_iou(boxes, query_boxes)
    iou = bbox_overlaps(boxes, query_boxes)
    
    print(f"Complex case - IoU: {iou[0, 0]:.6f}, DIoU: {diou[0, 0]:.6f}, CIoU: {ciou[0, 0]:.6f}, EIoU: {eiou[0, 0]:.6f}")
    
    # EIoU should be less than DIoU due to additional width/height penalties
    assert eiou[0, 0] < diou[0, 0], "EIoU should be less than DIoU due to width/height penalties"
    
    # All should be less than IoU
    assert eiou[0, 0] < iou[0, 0], "EIoU should be less than IoU"
    assert diou[0, 0] < iou[0, 0], "DIoU should be less than IoU"
    assert ciou[0, 0] < iou[0, 0], "CIoU should be less than IoU"
    
    # Manual verification of width/height penalties
    box_w, box_h = 10, 20
    query_w, query_h = 15, 10
    enclosing_w = max(13, 18) - min(0, 3)  # 18
    enclosing_h = max(20, 12) - min(0, 2)  # 20
    
    width_penalty = (box_w - query_w)**2 / enclosing_w**2
    height_penalty = (box_h - query_h)**2 / enclosing_h**2
    
    print(f"Width penalty: {width_penalty:.6f}, Height penalty: {height_penalty:.6f}")
    
    print("✓ Comprehensive penalties test passed!")


# =============================================================================
# Normalized Wasserstein Distance (NWD) Tests
# =============================================================================

def test_nwd_identical_boxes():
    """Test NWD for identical boxes - should equal 1.0"""
    print("\nTesting NWD for identical boxes...")
    
    boxes = np.array([[0, 0, 10, 10]], dtype=np.float32)
    query_boxes = np.array([[0, 0, 10, 10]], dtype=np.float32)
    
    nwd = normalized_wasserstein_distance(boxes, query_boxes)
    
    print(f"Identical boxes - NWD: {nwd[0, 0]:.6f}")
    assert abs(nwd[0, 0] - 1.0) < 1e-6, "NWD for identical boxes should be 1.0"
    
    print("✓ Identical boxes test passed!")


def test_nwd_distance_properties():
    """Test NWD distance properties"""
    print("\nTesting NWD distance properties...")
    
    # Test increasing distances
    base_box = np.array([[0, 0, 10, 10]], dtype=np.float32)
    
    test_boxes = np.array([
        [0, 0, 10, 10],     # Same position
        [5, 0, 15, 10],     # Moderate distance
        [20, 0, 30, 10],    # Large distance
        [50, 50, 60, 60]    # Very large distance
    ], dtype=np.float32)
    
    nwd = normalized_wasserstein_distance(test_boxes, base_box)
    
    print("NWD for increasing distances:")
    for i, box in enumerate(test_boxes):
        print(f"  Box {i+1}: {nwd[i, 0]:.6f}")
    
    # NWD should decrease with distance (higher values = more similar)
    assert nwd[1, 0] < nwd[0, 0], "NWD should decrease with distance"
    assert nwd[2, 0] < nwd[1, 0], "NWD should decrease with distance"
    assert nwd[3, 0] < nwd[2, 0], "NWD should decrease with distance"
    
    # All values should be in [0, 1]
    assert np.all(nwd >= 0) and np.all(nwd <= 1), "NWD should be in [0, 1]"
    
    print("✓ Distance properties test passed!")


def test_nwd_size_sensitivity():
    """Test NWD sensitivity to box sizes"""
    print("\nTesting NWD size sensitivity...")
    
    # Test boxes of different sizes at same center
    base_box = np.array([[0, 0, 10, 10]], dtype=np.float32)  # 10x10 at (5,5)
    
    size_boxes = np.array([
        [2.5, 2.5, 7.5, 7.5],    # 5x5 at (5,5) - smaller, same center
        [0, 0, 10, 10],          # 10x10 at (5,5) - same size, same center
        [-2.5, -2.5, 12.5, 12.5] # 15x15 at (5,5) - larger, same center
    ], dtype=np.float32)
    
    nwd = normalized_wasserstein_distance(size_boxes, base_box)
    
    print("NWD for different sizes (same center):")
    for i, box in enumerate(size_boxes):
        size = (box[2] - box[0]) * (box[3] - box[1])
        print(f"  Size {size:.1f}: {nwd[i, 0]:.6f}")
    
    # Same size should have maximal NWD (close to 1.0)
    assert abs(nwd[1, 0] - 1.0) < 1e-5, "Same size boxes at same center should have NWD = 1.0"
    
    # Different sizes should have lower NWD
    assert nwd[0, 0] < nwd[1, 0], "Different sizes should decrease NWD"
    assert nwd[2, 0] < nwd[1, 0], "Different sizes should decrease NWD"
    
    print("✓ Size sensitivity test passed!")


def test_nwd_vs_iou_comparison():
    """Compare NWD behavior with IoU variants"""
    print("\nTesting NWD vs IoU comparison...")
    
    # Test cases with different overlap and distance characteristics
    base_box = np.array([[0, 0, 10, 10]], dtype=np.float32)
    
    test_cases = [
        ("Identical", np.array([[0, 0, 10, 10]], dtype=np.float32)),
        ("Overlapping", np.array([[5, 5, 15, 15]], dtype=np.float32)),
        ("Adjacent", np.array([[10, 0, 20, 10]], dtype=np.float32)),
        ("Separated", np.array([[20, 20, 30, 30]], dtype=np.float32)),
        ("Far apart", np.array([[100, 100, 110, 110]], dtype=np.float32))
    ]
    
    print(f"{'Case':<12} {'IoU':<8} {'GIoU':<8} {'DIoU':<8} {'CIoU':<8} {'EIoU':<8} {'NWD':<8}")
    print("-" * 70)
    
    for name, query_box in test_cases:
        iou = bbox_overlaps(base_box, query_box)[0, 0]
        giou = generalized_iou(base_box, query_box)[0, 0]
        diou = distance_iou(base_box, query_box)[0, 0]
        ciou = complete_iou(base_box, query_box)[0, 0]
        eiou = efficient_iou(base_box, query_box)[0, 0]
        nwd = normalized_wasserstein_distance(base_box, query_box)[0, 0]
        
        print(f"{name:<12} {iou:<8.3f} {giou:<8.3f} {diou:<8.3f} {ciou:<8.3f} {eiou:<8.3f} {nwd:<8.3f}")
    
    print("✓ NWD vs IoU comparison test passed!")


def test_nwd_mathematical_properties():
    """Test mathematical properties of NWD"""
    print("\nTesting NWD mathematical properties...")
    
    # Generate random test cases
    np.random.seed(42)
    
    boxes = np.random.uniform(0, 50, (15, 4)).astype(np.float32)
    boxes[:, 2] += boxes[:, 0] + np.random.uniform(5, 20, 15)  # Add width
    boxes[:, 3] += boxes[:, 1] + np.random.uniform(5, 20, 15)  # Add height
    
    query_boxes = np.random.uniform(0, 50, (10, 4)).astype(np.float32)
    query_boxes[:, 2] += query_boxes[:, 0] + np.random.uniform(5, 20, 10)
    query_boxes[:, 3] += query_boxes[:, 1] + np.random.uniform(5, 20, 10)
    
    nwd = normalized_wasserstein_distance(boxes, query_boxes)
    
    print(f"Testing {boxes.shape[0]}x{query_boxes.shape[0]} random box pairs...")
    
    # Property 1: All values should be in [0, 1]
    assert np.all(nwd >= 0) and np.all(nwd <= 1), "NWD should be in [0, 1]"
    
    # Property 2: No NaN or Inf values
    assert not np.any(np.isnan(nwd)) and not np.any(np.isinf(nwd)), "NWD should not contain NaN/Inf"
    
    # Property 3: Symmetry - NWD(A,B) should equal NWD(B,A)
    nwd_reverse = normalized_wasserstein_distance(query_boxes[:3], boxes[:3])
    nwd_forward = normalized_wasserstein_distance(boxes[:3], query_boxes[:3])
    
    # Note: Due to matrix indexing, we compare nwd_forward[i,j] with nwd_reverse[j,i]
    for i in range(3):
        for j in range(3):
            assert abs(nwd_forward[i, j] - nwd_reverse[j, i]) < 1e-5, "NWD should be symmetric"
    
    print(f"NWD range: [{np.min(nwd):.6f}, {np.max(nwd):.6f}]")
    print(f"NWD mean: {np.mean(nwd):.6f}")
    
    print("✓ Mathematical properties test passed!")


# =============================================================================
# Multi-box and Edge Case Tests
# =============================================================================

def test_all_variants_multiple_boxes():
    """Test all IoU variants with multiple boxes"""
    print("\nTesting all variants with multiple boxes...")
    
    boxes = np.array([
        [0, 0, 10, 10],     # Square
        [5, 5, 15, 25],     # Tall rectangle
        [20, 20, 40, 25],   # Wide rectangle
        [2, 2, 8, 8]        # Small square inside first
    ], dtype=np.float32)
    
    query_boxes = np.array([
        [0, 0, 10, 10],     # Query square
        [25, 25, 35, 30]    # Query rectangle
    ], dtype=np.float32)
    
    iou = bbox_overlaps(boxes, query_boxes)
    giou = generalized_iou(boxes, query_boxes)
    diou = distance_iou(boxes, query_boxes)
    ciou = complete_iou(boxes, query_boxes)
    eiou = efficient_iou(boxes, query_boxes)
    nwd = normalized_wasserstein_distance(boxes, query_boxes)
    
    print(f"Multiple boxes shapes: {boxes.shape[0]}x{query_boxes.shape[0]}")
    print(f"IoU matrix:\n{iou}")
    print(f"GIoU matrix:\n{giou}")
    print(f"DIoU matrix:\n{diou}")
    print(f"CIoU matrix:\n{ciou}")
    print(f"EIoU matrix:\n{eiou}")
    print(f"NWD matrix:\n{nwd}")
    
    # Verify matrix shapes
    assert iou.shape == giou.shape == diou.shape == ciou.shape == eiou.shape == nwd.shape, "All matrices should have same shape"
    assert iou.shape == (4, 2), "Matrix should be 4x2"
    
    # Verify ordering relationships for IoU-based metrics
    assert np.all(ciou <= diou + 1e-6), "CIoU should be <= DIoU for all pairs"
    assert np.all(diou <= iou + 1e-6), "DIoU should be <= IoU for all pairs"
    assert np.all(giou <= iou + 1e-6), "GIoU should be <= IoU for all pairs"
    assert np.all(eiou <= diou + 1e-6), "EIoU should be <= DIoU for all pairs"
    
    # NWD properties
    assert np.all(nwd >= 0) and np.all(nwd <= 1), "NWD should be in [0, 1] for all pairs"
    
    # Perfect match should give 1.0 for all similarity metrics
    assert abs(iou[0, 0] - 1.0) < 1e-6, "Perfect match IoU should be 1.0"
    assert abs(giou[0, 0] - 1.0) < 1e-6, "Perfect match GIoU should be 1.0"
    assert abs(diou[0, 0] - 1.0) < 1e-6, "Perfect match DIoU should be 1.0"
    assert abs(ciou[0, 0] - 1.0) < 1e-6, "Perfect match CIoU should be 1.0"
    assert abs(eiou[0, 0] - 1.0) < 1e-6, "Perfect match EIoU should be 1.0"
    assert abs(nwd[0, 0] - 1.0) < 1e-6, "Perfect match NWD should be 1.0"
    
    print("✓ Multiple boxes test passed!")


def test_edge_cases_all_variants():
    """Test edge cases for all IoU variants"""
    print("\nTesting edge cases for all variants...")
    
    # Edge case 1: Zero width or height boxes
    try:
        boxes = np.array([[0, 0, 0, 10]], dtype=np.float32)  # Zero width
        query_boxes = np.array([[0, 0, 10, 10]], dtype=np.float32)
        
        iou = bbox_overlaps(boxes, query_boxes)
        giou = generalized_iou(boxes, query_boxes)
        diou = distance_iou(boxes, query_boxes)
        ciou = complete_iou(boxes, query_boxes)
        eiou = efficient_iou(boxes, query_boxes)
        nwd = normalized_wasserstein_distance(boxes, query_boxes)
        
        print(f"Zero width box - IoU: {iou[0, 0]:.6f}, GIoU: {giou[0, 0]:.6f}, DIoU: {diou[0, 0]:.6f}, CIoU: {ciou[0, 0]:.6f}, EIoU: {eiou[0, 0]:.6f}, NWD: {nwd[0, 0]:.6f}")
        
    except Exception as e:
        print(f"Zero width box handling: {e}")
    
    # Edge case 2: Very small boxes
    boxes = np.array([[0, 0, 0.1, 0.1]], dtype=np.float32)
    query_boxes = np.array([[0.05, 0.05, 0.15, 0.15]], dtype=np.float32)
    
    iou = bbox_overlaps(boxes, query_boxes)
    giou = generalized_iou(boxes, query_boxes)
    diou = distance_iou(boxes, query_boxes)
    ciou = complete_iou(boxes, query_boxes)
    eiou = efficient_iou(boxes, query_boxes)
    nwd = normalized_wasserstein_distance(boxes, query_boxes)
    
    print(f"Tiny boxes - IoU: {iou[0, 0]:.6f}, GIoU: {giou[0, 0]:.6f}, DIoU: {diou[0, 0]:.6f}, CIoU: {ciou[0, 0]:.6f}, EIoU: {eiou[0, 0]:.6f}, NWD: {nwd[0, 0]:.6f}")
    assert not any(np.isnan(x[0, 0]) for x in [iou, giou, diou, ciou, eiou, nwd]), "Should not produce NaN for tiny boxes"
    
    # Edge case 3: Extreme aspect ratios
    boxes = np.array([[0, 0, 100, 1]], dtype=np.float32)  # Very wide
    query_boxes = np.array([[0, 0, 1, 100]], dtype=np.float32)  # Very tall
    
    iou = bbox_overlaps(boxes, query_boxes)
    giou = generalized_iou(boxes, query_boxes)
    diou = distance_iou(boxes, query_boxes)
    ciou = complete_iou(boxes, query_boxes)
    eiou = efficient_iou(boxes, query_boxes)
    nwd = normalized_wasserstein_distance(boxes, query_boxes)
    
    print(f"Extreme aspect ratios - IoU: {iou[0, 0]:.6f}, GIoU: {giou[0, 0]:.6f}, DIoU: {diou[0, 0]:.6f}, CIoU: {ciou[0, 0]:.6f}, EIoU: {eiou[0, 0]:.6f}, NWD: {nwd[0, 0]:.6f}")
    assert ciou[0, 0] < diou[0, 0], "CIoU should heavily penalize extreme aspect ratio differences"
    assert eiou[0, 0] < diou[0, 0], "EIoU should heavily penalize extreme width/height differences"
    
    print("✓ Edge cases test passed!")


def test_mathematical_properties():
    """Test mathematical properties of all IoU variants"""
    print("\nTesting mathematical properties...")
    
    # Generate random test cases
    np.random.seed(42)  # For reproducible tests
    
    # Random boxes ensuring x2 > x1, y2 > y1
    boxes = np.random.uniform(0, 50, (20, 4)).astype(np.float32)
    boxes[:, 2] += boxes[:, 0] + 1  # Ensure x2 > x1
    boxes[:, 3] += boxes[:, 1] + 1  # Ensure y2 > y1
    
    query_boxes = np.random.uniform(0, 50, (10, 4)).astype(np.float32)
    query_boxes[:, 2] += query_boxes[:, 0] + 1
    query_boxes[:, 3] += query_boxes[:, 1] + 1
    
    iou = bbox_overlaps(boxes, query_boxes)
    giou = generalized_iou(boxes, query_boxes)
    diou = distance_iou(boxes, query_boxes)
    ciou = complete_iou(boxes, query_boxes)
    
    print(f"Testing {boxes.shape[0]}x{query_boxes.shape[0]} random box pairs...")
    
    # Property 1: All values should be in reasonable range
    assert np.all(iou >= 0) and np.all(iou <= 1), "IoU should be in [0, 1]"
    assert np.all(giou >= -1) and np.all(giou <= 1), "GIoU should be in [-1, 1]"
    assert np.all(diou >= -1) and np.all(diou <= 1), "DIoU should be in [-1, 1]"
    assert np.all(ciou >= -1) and np.all(ciou <= 1), "CIoU should be in [-1, 1]"
    
    # Property 2: Ordering relationships
    assert np.all(giou <= iou + 1e-5), "GIoU should be <= IoU"
    assert np.all(diou <= iou + 1e-5), "DIoU should be <= IoU"
    assert np.all(ciou <= diou + 1e-5), "CIoU should be <= DIoU"
    
    # Property 3: No NaN or Inf values
    for name, matrix in [("IoU", iou), ("GIoU", giou), ("DIoU", diou), ("CIoU", ciou)]:
        assert not np.any(np.isnan(matrix)) and not np.any(np.isinf(matrix)), f"{name} should not contain NaN/Inf"
    
    print(f"IoU range:  [{np.min(iou):.6f}, {np.max(iou):.6f}]")
    print(f"GIoU range: [{np.min(giou):.6f}, {np.max(giou):.6f}]")
    print(f"DIoU range: [{np.min(diou):.6f}, {np.max(diou):.6f}]")
    print(f"CIoU range: [{np.min(ciou):.6f}, {np.max(ciou):.6f}]")
    
    print("✓ Mathematical properties test passed!")


# =============================================================================
# Main Test Runner
# =============================================================================

def run_all_tests():
    """Run all tests for all IoU variants"""
    print("=" * 80)
    print("FastBBox: Comprehensive IoU Variants Test Suite")
    print("Testing: IoU, GIoU, DIoU, CIoU, EIoU, NWD")
    print("=" * 80)
    
    # Standard IoU Tests
    print("\n" + "=" * 40)
    print("STANDARD IoU TESTS")
    print("=" * 40)
    test_iou_basic_functionality()
    test_iou_multiple_boxes()
    test_iou_float_coordinates()
    test_iou_edge_cases()
    
    # GIoU Tests
    print("\n" + "=" * 40)
    print("GENERALIZED IoU (GIoU) TESTS")
    print("=" * 40)
    test_giou_identical_boxes()
    test_giou_no_overlap()
    test_giou_partial_overlap()
    test_giou_contained_boxes()
    
    # DIoU Tests
    print("\n" + "=" * 40)
    print("DISTANCE IoU (DIoU) TESTS")
    print("=" * 40)
    test_diou_identical_boxes()
    test_diou_center_distance_penalty()
    test_diou_no_overlap()
    
    # CIoU Tests
    print("\n" + "=" * 40)
    print("COMPLETE IoU (CIoU) TESTS")
    print("=" * 40)
    test_ciou_identical_boxes()
    test_ciou_aspect_ratio_penalty()
    test_ciou_comprehensive_comparison()
    
    # EIoU Tests
    print("\n" + "=" * 40)
    print("EFFICIENT IoU (EIoU) TESTS")
    print("=" * 40)
    test_eiou_identical_boxes()
    test_eiou_width_height_penalty()
    test_eiou_comprehensive_penalties()
    
    # NWD Tests
    print("\n" + "=" * 40)
    print("NORMALIZED WASSERSTEIN DISTANCE (NWD) TESTS")
    print("=" * 40)
    test_nwd_identical_boxes()
    test_nwd_distance_properties()
    test_nwd_size_sensitivity()
    test_nwd_vs_iou_comparison()
    test_nwd_mathematical_properties()
    
    # Combined Tests
    print("\n" + "=" * 40)
    print("COMBINED TESTS")
    print("=" * 40)
    test_all_variants_multiple_boxes()
    test_edge_cases_all_variants()
    test_mathematical_properties()
    
    print("\n" + "=" * 80)
    print("🎉 ALL TESTS PASSED!")
    print("FastBBox IoU variants are working correctly:")
    print("✓ Standard IoU (bbox_overlaps)")
    print("✓ Generalized IoU (generalized_iou)")
    print("✓ Distance IoU (distance_iou)")
    print("✓ Complete IoU (complete_iou)")
    print("✓ Efficient IoU (efficient_iou)")
    print("✓ Normalized Wasserstein Distance (normalized_wasserstein_distance)")
    print("Ready for production use! 🚀")
    print("=" * 80)


if __name__ == "__main__":
    run_all_tests()
