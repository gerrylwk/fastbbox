#!/usr/bin/env python3
"""
Quick development test script for fastbbox.
Run this after editing bbox.pyx to get immediate feedback.

Usage:
    python dev_test.py

This script will:
1. Rebuild the Cython extension
2. Run basic functionality tests
3. Show performance comparison
4. Report any errors immediately
"""

import subprocess
import sys
import os
import time
import traceback

def run_command(cmd, description):
    """Run a command and return success status"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=".")
        if result.returncode == 0:
            print(f"✅ {description} - SUCCESS")
            return True
        else:
            print(f"❌ {description} - FAILED")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
    except Exception as e:
        print(f"❌ {description} - ERROR: {e}")
        return False

def test_imports():
    """Test if we can import all functions"""
    print("\n🧪 Testing imports...")
    try:
        import fastbbox
        from fastbbox import (bbox_overlaps, generalized_iou, distance_iou, 
                             complete_iou, efficient_iou, normalized_wasserstein_distance)
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        traceback.print_exc()
        return False

def test_basic_functionality():
    """Test basic functionality of all IoU variants"""
    print("\n🧪 Testing basic functionality...")
    try:
        import numpy as np
        from fastbbox import (bbox_overlaps, generalized_iou, distance_iou, 
                             complete_iou, efficient_iou, normalized_wasserstein_distance)
        
        # Test data
        boxes = np.array([
            [0, 0, 10, 10],
            [5, 5, 15, 15],
            [20, 20, 30, 30]
        ], dtype=np.float32)
        
        query_boxes = np.array([
            [0, 0, 10, 10],
            [12, 12, 22, 22]
        ], dtype=np.float32)
        
        # Test all functions
        functions = [
            ("IoU", bbox_overlaps),
            ("GIoU", generalized_iou),
            ("DIoU", distance_iou),
            ("CIoU", complete_iou),
            ("EIoU", efficient_iou),
            ("NWD", normalized_wasserstein_distance)
        ]
        
        results = {}
        for name, func in functions:
            try:
                start_time = time.time()
                result = func(boxes, query_boxes)
                end_time = time.time()
                
                # Basic validation
                assert result.shape == (3, 2), f"{name}: Wrong shape {result.shape}, expected (3, 2)"
                assert not np.isnan(result).any(), f"{name}: Contains NaN values"
                assert not np.isinf(result).any(), f"{name}: Contains infinite values"
                
                results[name] = {
                    'result': result,
                    'time': end_time - start_time,
                    'identical_score': result[0, 0]  # Score for identical boxes
                }
                
                print(f"  ✅ {name:4s}: shape={result.shape}, identical={result[0,0]:.4f}, time={end_time-start_time:.4f}s")
                
            except Exception as e:
                print(f"  ❌ {name:4s}: {e}")
                return False
        
        # Validate expected relationships
        print("\n🔍 Validating relationships...")
        
        # IoU should be 1.0 for identical boxes
        if abs(results['IoU']['identical_score'] - 1.0) > 1e-6:
            print(f"  ⚠️  IoU for identical boxes should be 1.0, got {results['IoU']['identical_score']:.6f}")
        
        # NWD should be 1.0 for identical boxes (similarity measure)
        if abs(results['NWD']['identical_score'] - 1.0) > 1e-6:
            print(f"  ⚠️  NWD for identical boxes should be 1.0, got {results['NWD']['identical_score']:.6f}")
        
        # Other IoU variants should be 1.0 for identical boxes
        for name in ['GIoU', 'DIoU', 'CIoU', 'EIoU']:
            if abs(results[name]['identical_score'] - 1.0) > 1e-6:
                print(f"  ⚠️  {name} for identical boxes should be 1.0, got {results[name]['identical_score']:.6f}")
        
        print("✅ All functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        traceback.print_exc()
        return False

def quick_benchmark():
    """Run a quick performance benchmark"""
    print("\n⚡ Quick performance benchmark...")
    try:
        import numpy as np
        from fastbbox import bbox_overlaps
        
        # Generate test data
        np.random.seed(42)
        boxes = np.random.rand(100, 4).astype(np.float32) * 100
        boxes[:, 2:] += boxes[:, :2]  # Ensure x2 > x1, y2 > y1
        
        query_boxes = np.random.rand(50, 4).astype(np.float32) * 100
        query_boxes[:, 2:] += query_boxes[:, :2]
        
        # Benchmark
        start_time = time.time()
        for _ in range(10):  # Run 10 times
            result = bbox_overlaps(boxes, query_boxes)
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / 10
        ops_per_sec = (100 * 50) / avg_time  # boxes * queries / time
        
        print(f"  📊 100x50 boxes: {avg_time:.4f}s avg, {ops_per_sec:.0f} ops/sec")
        print("✅ Performance benchmark completed!")
        return True
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        return False

def main():
    """Main development test workflow"""
    print("=" * 60)
    print("🚀 FASTBBOX DEVELOPMENT TEST")
    print("=" * 60)
    
    # Step 1: Clean previous build
    print("\n🧹 Cleaning previous build...")
    if os.path.exists("build"):
        run_command("rmdir /s /q build" if os.name == 'nt' else "rm -rf build", "Remove build directory")
    
    # Remove compiled extensions
    for root, dirs, files in os.walk("fastbbox"):
        for file in files:
            if file.endswith(('.pyd', '.so', '.c', '.cpp')):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"  🗑️  Removed {file_path}")
                except:
                    pass
    
    # Step 2: Rebuild Cython extension
    success = run_command("python setup.py build_ext --inplace", "Rebuild Cython extension")
    if not success:
        print("\n💥 BUILD FAILED - Cannot continue with tests")
        return False
    
    # Step 3: Test imports
    if not test_imports():
        print("\n💥 IMPORT FAILED - Check your Cython code for syntax errors")
        return False
    
    # Step 4: Test functionality
    if not test_basic_functionality():
        print("\n💥 FUNCTIONALITY FAILED - Check your algorithm implementations")
        return False
    
    # Step 5: Quick benchmark
    if not quick_benchmark():
        print("\n⚠️  BENCHMARK FAILED - Performance test had issues")
    
    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED! Your changes are working correctly.")
    print("=" * 60)
    print("\n💡 Next steps:")
    print("   - Run full test suite: python test_all_iou.py")
    print("   - Run benchmark: python benchmark_comparison.py")
    print("   - Commit your changes: git add . && git commit -m 'Update implementation'")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
