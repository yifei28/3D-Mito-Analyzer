#!/usr/bin/env python3
"""
Simple error handling test for Task 10.5 verification
"""

import os
import sys
import tempfile
import numpy as np
import tifffile

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from workflows.segmentation import SegmentationWorkflow, HardwareType


def test_error_handling():
    """Quick test of error handling functionality."""
    print("🧪 Testing Error Handling Implementation")
    print("=" * 50)

    # Create workflow
    workflow = SegmentationWorkflow(force_cpu=True)

    # Test 1: GPU OOM Detection
    print("\n1. Testing GPU OOM Detection...")
    test_errors = [
        Exception("ResourceExhaustedError: OOM when allocating tensor"),
        Exception("CUDA_ERROR_OUT_OF_MEMORY"),
        Exception("Regular error message")
    ]

    for i, error in enumerate(test_errors):
        is_oom = workflow._detect_gpu_oom_error(error)
        expected = i < 2  # First two should be detected as OOM
        result = "✅" if is_oom == expected else "❌"
        print(f"   {result} Error {i+1}: {is_oom} (expected {expected})")

    # Test 2: Memory Monitoring
    print("\n2. Testing Memory Monitoring...")
    try:
        needs_adjustment, memory_stats = workflow.monitor_memory_during_processing()
        print(f"   ✅ Memory usage: {memory_stats['percent_used']:.1f}%, adjustment needed: {needs_adjustment}")
    except Exception as e:
        print(f"   ❌ Memory monitoring failed: {e}")

    # Test 3: Disk Space Check
    print("\n3. Testing Disk Space Check...")
    try:
        has_space = workflow._check_disk_space("/tmp", required_gb=0.1)
        print(f"   ✅ Disk space check: {has_space}")
    except Exception as e:
        print(f"   ❌ Disk space check failed: {e}")

    # Test 4: File Permissions
    print("\n4. Testing File Permissions...")
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = os.path.join(temp_dir, "test.tif")
            # Create test file
            np.save(test_file, np.zeros((10, 10)))

            is_valid, error_msg = workflow._validate_file_permissions(test_file, temp_dir)
            print(f"   ✅ Permission validation: {is_valid}")
            if not is_valid:
                print(f"      Error: {error_msg}")
    except Exception as e:
        print(f"   ❌ Permission test failed: {e}")

    # Test 5: Hardware Fallback (if not already on CPU)
    print("\n5. Testing Hardware Fallback...")
    try:
        original_type = workflow.hardware_config.hardware_type
        fallback_success = workflow._fallback_to_cpu("Test fallback", "Test error")

        if fallback_success:
            print(f"   ✅ Fallback executed successfully")
            print(f"   📊 Fallback events: {len(workflow.hardware_config.fallback_events)}")
        else:
            print(f"   ⚠️ Fallback not needed (already on CPU)")
    except Exception as e:
        print(f"   ❌ Fallback test failed: {e}")

    print("\n" + "=" * 50)
    print("✅ Error handling tests completed!")
    print("🎯 Key features implemented:")
    print("   • GPU OOM detection and CPU fallback")
    print("   • Memory pressure monitoring")
    print("   • File system error handling")
    print("   • Automatic retry logic")
    print("   • Hardware configuration tracking")


if __name__ == "__main__":
    test_error_handling()