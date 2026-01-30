#!/usr/bin/env python3
"""
Demo script for Oriented Bounding Box (OBB) IoU calculations.

This script demonstrates the usage of the OBB IoU functions implemented
in the fastbbox package.
"""

import numpy as np
import math
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from fastbbox import bbox_overlaps_obb
    print("SUCCESS: Imported OBB IoU function from fastbbox package")
except ImportError as e:
    print(f"ERROR: Could not import from fastbbox package: {e}")
    print("Trying direct import...")
    try:
        from fastbbox.obb_bbox_final import bbox_overlaps_obb_final as bbox_overlaps_obb
        print("SUCCESS: Direct import successful")
    except ImportError as e2:
        print(f"ERROR: Direct import also failed: {e2}")
        sys.exit(1)


def create_obb(cx, cy, width, height, angle_degrees=0):
    """Create an OBB with angle in degrees (converted to radians internally)."""
    angle_radians = math.radians(angle_degrees)
    return np.array([cx, cy, width, height, angle_radians], dtype=np.float32)


def demo_basic_usage():
    """Demonstrate basic OBB IoU calculations."""
    print("\n" + "="*60)
    print("BASIC OBB IoU USAGE DEMO")
    print("="*60)
    
    # Example 1: Identical boxes
    print("\n1. Identical Boxes:")
    box1 = create_obb(0, 0, 4, 2, 0)  # 4x2 box at origin
    box2 = create_obb(0, 0, 4, 2, 0)  # Same box
    
    iou = bbox_overlaps_obb(box1.reshape(1, -1), box2.reshape(1, -1))
    print(f"   Box1: center=(0,0), size=(4,2), angle=0°")
    print(f"   Box2: center=(0,0), size=(4,2), angle=0°")
    print(f"   IoU: {iou[0,0]:.6f} (Expected: 1.000000)")
    
    # Example 2: Partial overlap
    print("\n2. Partial Overlap (Axis-Aligned):")
    box1 = create_obb(0, 0, 2, 2, 0)    # 2x2 box at origin
    box2 = create_obb(1, 0, 2, 2, 0)    # 2x2 box shifted right by 1
    
    iou = bbox_overlaps_obb(box1.reshape(1, -1), box2.reshape(1, -1))
    print(f"   Box1: center=(0,0), size=(2,2), angle=0°")
    print(f"   Box2: center=(1,0), size=(2,2), angle=0°")
    print(f"   IoU: {iou[0,0]:.6f} (Expected: ~0.333333)")
    
    # Example 3: No overlap
    print("\n3. No Overlap:")
    box1 = create_obb(0, 0, 2, 2, 0)    # 2x2 box at origin
    box2 = create_obb(5, 5, 2, 2, 0)    # 2x2 box far away
    
    iou = bbox_overlaps_obb(box1.reshape(1, -1), box2.reshape(1, -1))
    print(f"   Box1: center=(0,0), size=(2,2), angle=0°")
    print(f"   Box2: center=(5,5), size=(2,2), angle=0°")
    print(f"   IoU: {iou[0,0]:.6f} (Expected: 0.000000)")


def demo_rotated_boxes():
    """Demonstrate OBB IoU with rotated boxes."""
    print("\n" + "="*60)
    print("ROTATED BOXES DEMO")
    print("="*60)
    
    # Example 1: Same box, different rotations
    print("\n1. Same Box at Different Rotations:")
    box1 = create_obb(0, 0, 4, 2, 0)     # Horizontal
    box2 = create_obb(0, 0, 4, 2, 45)    # 45° rotation
    
    iou = bbox_overlaps_obb(box1.reshape(1, -1), box2.reshape(1, -1))
    print(f"   Box1: center=(0,0), size=(4,2), angle=0°")
    print(f"   Box2: center=(0,0), size=(4,2), angle=45°")
    print(f"   IoU: {iou[0,0]:.6f} (Approximation for rotated overlap)")
    
    # Example 2: Perpendicular boxes
    print("\n2. Perpendicular Boxes:")
    box1 = create_obb(0, 0, 4, 1, 0)     # Horizontal thin box
    box2 = create_obb(0, 0, 1, 4, 90)    # Vertical thin box
    
    iou = bbox_overlaps_obb(box1.reshape(1, -1), box2.reshape(1, -1))
    print(f"   Box1: center=(0,0), size=(4,1), angle=0°")
    print(f"   Box2: center=(0,0), size=(1,4), angle=90°")
    print(f"   IoU: {iou[0,0]:.6f}")
    
    # Example 3: Slight rotation
    print("\n3. Slight Rotation:")
    box1 = create_obb(0, 0, 3, 2, 0)     # No rotation
    box2 = create_obb(0, 0, 3, 2, 15)    # 15° rotation
    
    iou = bbox_overlaps_obb(box1.reshape(1, -1), box2.reshape(1, -1))
    print(f"   Box1: center=(0,0), size=(3,2), angle=0°")
    print(f"   Box2: center=(0,0), size=(3,2), angle=15°")
    print(f"   IoU: {iou[0,0]:.6f}")


def demo_batch_processing():
    """Demonstrate batch processing of multiple OBBs."""
    print("\n" + "="*60)
    print("BATCH PROCESSING DEMO")
    print("="*60)
    
    # Create multiple boxes
    boxes_set1 = np.array([
        create_obb(0, 0, 2, 2, 0),      # Box A: 2x2 at origin
        create_obb(1, 1, 2, 2, 0),      # Box B: 2x2 shifted
        create_obb(0, 0, 4, 1, 45),     # Box C: 4x1 rotated 45°
    ])
    
    boxes_set2 = np.array([
        create_obb(0, 0, 2, 2, 0),      # Box X: same as A
        create_obb(2, 2, 2, 2, 0),      # Box Y: 2x2 further shifted
        create_obb(0, 0, 1, 4, 45),     # Box Z: 1x4 rotated 45°
    ])
    
    # Compute IoU matrix
    iou_matrix = bbox_overlaps_obb(boxes_set1, boxes_set2)
    
    print(f"\nSet 1 boxes:")
    for i, box in enumerate(boxes_set1):
        angle_deg = math.degrees(box[4])
        print(f"   Box {chr(65+i)}: center=({box[0]:.1f},{box[1]:.1f}), size=({box[2]:.1f},{box[3]:.1f}), angle={angle_deg:.0f}°")
    
    print(f"\nSet 2 boxes:")
    for i, box in enumerate(boxes_set2):
        angle_deg = math.degrees(box[4])
        print(f"   Box {chr(88+i)}: center=({box[0]:.1f},{box[1]:.1f}), size=({box[2]:.1f},{box[3]:.1f}), angle={angle_deg:.0f}°")
    
    print(f"\nIoU Matrix (Set1 vs Set2):")
    print(f"        X       Y       Z")
    for i in range(3):
        row_label = chr(65+i)
        row_values = "  ".join([f"{iou_matrix[i,j]:.4f}" for j in range(3)])
        print(f"   {row_label}  {row_values}")


def demo_edge_cases():
    """Demonstrate edge cases and special scenarios."""
    print("\n" + "="*60)
    print("EDGE CASES DEMO")
    print("="*60)
    
    # Example 1: Very small boxes
    print("\n1. Very Small Boxes:")
    box1 = create_obb(0, 0, 0.01, 0.01, 0)
    box2 = create_obb(0.005, 0.005, 0.01, 0.01, 0)
    
    iou = bbox_overlaps_obb(box1.reshape(1, -1), box2.reshape(1, -1))
    print(f"   Box1: center=(0,0), size=(0.01,0.01)")
    print(f"   Box2: center=(0.005,0.005), size=(0.01,0.01)")
    print(f"   IoU: {iou[0,0]:.6f}")
    
    # Example 2: High aspect ratio
    print("\n2. High Aspect Ratio Boxes:")
    box1 = create_obb(0, 0, 10, 0.5, 0)    # Very wide
    box2 = create_obb(0, 0, 0.5, 10, 90)   # Very tall
    
    iou = bbox_overlaps_obb(box1.reshape(1, -1), box2.reshape(1, -1))
    print(f"   Box1: center=(0,0), size=(10,0.5), angle=0°")
    print(f"   Box2: center=(0,0), size=(0.5,10), angle=90°")
    print(f"   IoU: {iou[0,0]:.6f}")
    
    # Example 3: Touching boxes
    print("\n3. Touching Boxes (Edge Contact):")
    box1 = create_obb(0, 0, 2, 2, 0)
    box2 = create_obb(2, 0, 2, 2, 0)  # Touching at right edge
    
    iou = bbox_overlaps_obb(box1.reshape(1, -1), box2.reshape(1, -1))
    print(f"   Box1: center=(0,0), size=(2,2)")
    print(f"   Box2: center=(2,0), size=(2,2)")
    print(f"   IoU: {iou[0,0]:.6f} (Should be 0 for touching)")


def main():
    """Run all demos."""
    print("ORIENTED BOUNDING BOX (OBB) IoU DEMONSTRATION")
    print("FastBBox Package - OBB Implementation")
    
    demo_basic_usage()
    demo_rotated_boxes()
    demo_batch_processing()
    demo_edge_cases()
    
    print("\n" + "="*60)
    print("DEMO COMPLETED")
    print("="*60)
    print("\nNotes:")
    print("- This implementation provides exact IoU for axis-aligned boxes")
    print("- For rotated boxes, it uses approximations based on AABB intersection")
    print("- The approximation quality depends on the rotation angles")
    print("- For production use with rotated boxes, consider more sophisticated")
    print("  polygon intersection algorithms if higher accuracy is needed")


if __name__ == "__main__":
    main()
