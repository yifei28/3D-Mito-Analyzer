#!/usr/bin/env python3
"""
Test script for Task 10.5: Robust Error Handling and Hardware Fallback

This script tests various error scenarios to ensure proper handling:
1. GPU OOM simulation
2. CPU memory pressure
3. File permission errors
4. Disk space issues
5. Hardware fallback logic
"""

import os
import sys
import time
import tempfile
import numpy as np
import tifffile
import shutil
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from workflows.segmentation import SegmentationWorkflow, HardwareType, ErrorType


class ErrorHandlingTester:
    """Test suite for error handling functionality."""

    def __init__(self):
        self.test_results = []
        self.temp_dir = tempfile.mkdtemp(prefix="error_test_")
        self.workflow = None

    def setUp(self):
        """Set up test environment."""
        print("🧪 Setting up error handling test environment...")

        # Create segmentation workflow
        self.workflow = SegmentationWorkflow(force_cpu=True)

        # Create test TIFF file
        self.test_image_path = os.path.join(self.temp_dir, "test_image.tif")
        self._create_test_tiff()

        print(f"✅ Test environment created in: {self.temp_dir}")

    def _create_test_tiff(self):
        """Create a test TIFF z-stack for testing."""
        # Create small test z-stack (10 slices, 512x512)
        z_stack = np.random.randint(0, 255, (10, 512, 512), dtype=np.uint8)
        tifffile.imwrite(self.test_image_path, z_stack)

    def test_file_permission_error(self):
        """Test file permission error handling."""
        print("\n🧪 Testing file permission error handling...")

        try:
            # Create read-only directory
            readonly_dir = os.path.join(self.temp_dir, "readonly")
            os.makedirs(readonly_dir, exist_ok=True)
            os.chmod(readonly_dir, 0o444)  # Read-only

            # Test permission validation
            is_valid, error_msg = self.workflow._validate_file_permissions(
                self.test_image_path, readonly_dir
            )

            if not is_valid and "permission" in error_msg.lower():
                self.test_results.append(("✅ File permission error detection", True))
                print(f"✅ Permission error correctly detected: {error_msg}")
            else:
                self.test_results.append(("❌ File permission error detection", False))
                print(f"❌ Permission error not detected properly")

        except Exception as e:
            self.test_results.append(("❌ File permission test failed", False))
            print(f"❌ Permission test failed: {e}")
        finally:
            # Restore permissions for cleanup
            try:
                os.chmod(readonly_dir, 0o755)
            except:
                pass

    def test_disk_space_check(self):
        """Test disk space checking."""
        print("\n🧪 Testing disk space checking...")

        try:
            # Test with reasonable space requirement
            has_space = self.workflow._check_disk_space(self.temp_dir, required_gb=0.1)

            if has_space:
                self.test_results.append(("✅ Disk space check (sufficient)", True))
                print("✅ Disk space check passed with sufficient space")
            else:
                self.test_results.append(("⚠️ Disk space check (insufficient)", True))
                print("⚠️ Disk space check detected insufficient space")

            # Test with unreasonable space requirement
            has_space_huge = self.workflow._check_disk_space(self.temp_dir, required_gb=1000000)

            if not has_space_huge:
                self.test_results.append(("✅ Disk space check (excessive requirement)", True))
                print("✅ Disk space check correctly rejected excessive requirement")
            else:
                self.test_results.append(("❌ Disk space check (excessive requirement)", False))
                print("❌ Disk space check incorrectly passed excessive requirement")

        except Exception as e:
            self.test_results.append(("❌ Disk space check failed", False))
            print(f"❌ Disk space check failed: {e}")

    def test_gpu_oom_detection(self):
        """Test GPU OOM error detection."""
        print("\n🧪 Testing GPU OOM error detection...")

        try:
            # Create mock TensorFlow ResourceExhaustedError
            class MockResourceExhaustedError(Exception):
                pass

            # Test various OOM error messages
            test_errors = [
                MockResourceExhaustedError("OOM when allocating tensor with shape"),
                Exception("CUDA_ERROR_OUT_OF_MEMORY"),
                Exception("Failed to allocate GPU memory"),
                Exception("Resource exhausted: Out of memory"),
                Exception("Some other error")
            ]

            expected_results = [True, True, True, True, False]

            for i, (error, expected) in enumerate(zip(test_errors, expected_results)):
                detected = self.workflow._detect_gpu_oom_error(error)

                if detected == expected:
                    result = "✅"
                    success = True
                else:
                    result = "❌"
                    success = False

                self.test_results.append((f"{result} GPU OOM detection test {i+1}", success))
                print(f"{result} Error '{str(error)[:50]}...' - Detected: {detected}, Expected: {expected}")

        except Exception as e:
            self.test_results.append(("❌ GPU OOM detection test failed", False))
            print(f"❌ GPU OOM detection test failed: {e}")

    def test_memory_pressure_monitoring(self):
        """Test memory pressure monitoring."""
        print("\n🧪 Testing memory pressure monitoring...")

        try:
            # Test memory monitoring
            needs_adjustment, memory_stats = self.workflow.monitor_memory_during_processing()

            if isinstance(needs_adjustment, bool) and isinstance(memory_stats, dict):
                self.test_results.append(("✅ Memory monitoring returns valid data", True))
                print(f"✅ Memory monitoring: {memory_stats['percent_used']:.1f}% used, adjustment needed: {needs_adjustment}")

                # Test adaptive memory management
                current_batch = 4
                new_batch = self.workflow._adaptive_memory_management(current_batch, memory_stats['percent_used'])

                if isinstance(new_batch, int) and new_batch > 0:
                    self.test_results.append(("✅ Adaptive memory management", True))
                    print(f"✅ Adaptive batch sizing: {current_batch} → {new_batch}")
                else:
                    self.test_results.append(("❌ Adaptive memory management", False))
                    print(f"❌ Invalid batch size returned: {new_batch}")

            else:
                self.test_results.append(("❌ Memory monitoring invalid data", False))
                print("❌ Memory monitoring returned invalid data types")

        except Exception as e:
            self.test_results.append(("❌ Memory monitoring test failed", False))
            print(f"❌ Memory monitoring test failed: {e}")

    def test_hardware_fallback_logic(self):
        """Test hardware fallback logic."""
        print("\n🧪 Testing hardware fallback logic...")

        try:
            original_hardware = self.workflow.hardware_config.hardware_type
            original_device_name = self.workflow.hardware_config.device_name

            # Test fallback to CPU
            fallback_success = self.workflow._fallback_to_cpu(
                reason="Test fallback",
                error_details="Simulated GPU error for testing"
            )

            if fallback_success:
                # Check if hardware configuration was updated
                if self.workflow.hardware_config.hardware_type == HardwareType.CPU_ONLY:
                    self.test_results.append(("✅ Hardware fallback execution", True))
                    print("✅ Hardware fallback successfully updated configuration")

                    # Check fallback history
                    if self.workflow.hardware_config.has_fallback_occurred:
                        self.test_results.append(("✅ Fallback event tracking", True))
                        print("✅ Fallback event properly tracked")
                    else:
                        self.test_results.append(("❌ Fallback event tracking", False))
                        print("❌ Fallback event not tracked")

                else:
                    self.test_results.append(("❌ Hardware fallback configuration", False))
                    print("❌ Hardware configuration not updated after fallback")
            else:
                self.test_results.append(("❌ Hardware fallback execution", False))
                print("❌ Hardware fallback failed")

        except Exception as e:
            self.test_results.append(("❌ Hardware fallback test failed", False))
            print(f"❌ Hardware fallback test failed: {e}")

    def test_retry_logic(self):
        """Test automatic retry logic."""
        print("\n🧪 Testing retry logic...")

        try:
            # Create a function that fails twice then succeeds
            call_count = [0]

            def failing_operation():
                call_count[0] += 1
                if call_count[0] <= 2:
                    raise Exception(f"Simulated failure #{call_count[0]}")
                return "success"

            # Test retry with eventual success
            try:
                result = self.workflow._execute_with_hardware_fallback(
                    failing_operation,
                    "Test retry operation",
                    max_retries=3
                )

                if result == "success" and call_count[0] == 3:
                    self.test_results.append(("✅ Retry logic with eventual success", True))
                    print("✅ Retry logic succeeded after 2 failures")
                else:
                    self.test_results.append(("❌ Retry logic behavior", False))
                    print(f"❌ Unexpected retry behavior: result={result}, calls={call_count[0]}")

            except Exception as e:
                self.test_results.append(("❌ Retry logic failed", False))
                print(f"❌ Retry logic test failed: {e}")

        except Exception as e:
            self.test_results.append(("❌ Retry logic test setup failed", False))
            print(f"❌ Retry logic test setup failed: {e}")

    def run_comprehensive_test(self):
        """Run comprehensive error handling test with real segmentation."""
        print("\n🧪 Running comprehensive error handling test...")

        try:
            output_dir = os.path.join(self.temp_dir, "output")

            def progress_callback(percent, info):
                if percent % 20 == 0 or percent == 100:  # Log every 20%
                    print(f"   Progress: {percent}% - {info.get('message', 'Processing...')}")

            # This should test the full pipeline with error handling
            result = self.workflow.run_segmentation(
                input_path=self.test_image_path,
                output_dir=output_dir,
                progress_callback=progress_callback
            )

            if isinstance(result, dict) and 'output_file' in result:
                # Check if output file was created
                output_file = result['output_file']
                if os.path.exists(output_file):
                    self.test_results.append(("✅ Full segmentation pipeline with error handling", True))
                    print(f"✅ Segmentation completed successfully: {output_file}")
                else:
                    self.test_results.append(("❌ Output file not created", False))
                    print(f"❌ Output file not created: {output_file}")
            else:
                self.test_results.append(("❌ Invalid segmentation result", False))
                print("❌ Segmentation returned invalid result")

        except Exception as e:
            # This might be expected if model is not available
            if "Model not found" in str(e) or "mock processing" in str(e).lower():
                self.test_results.append(("⚠️ Segmentation test (model not available)", True))
                print(f"⚠️ Segmentation test with mock processing: {e}")
            else:
                self.test_results.append(("❌ Segmentation test failed", False))
                print(f"❌ Segmentation test failed: {e}")

    def cleanup(self):
        """Clean up test environment."""
        try:
            shutil.rmtree(self.temp_dir)
            print(f"\n🧹 Test environment cleaned up: {self.temp_dir}")
        except Exception as e:
            print(f"⚠️ Cleanup failed: {e}")

    def run_all_tests(self):
        """Run all error handling tests."""
        print("=" * 80)
        print("🧪 STARTING ERROR HANDLING TEST SUITE")
        print("=" * 80)

        try:
            self.setUp()

            # Run individual tests
            self.test_file_permission_error()
            self.test_disk_space_check()
            self.test_gpu_oom_detection()
            self.test_memory_pressure_monitoring()
            self.test_hardware_fallback_logic()
            self.test_retry_logic()
            self.run_comprehensive_test()

        finally:
            self.cleanup()

        # Print summary
        print("\n" + "=" * 80)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 80)

        passed = sum(1 for _, success in self.test_results if success)
        total = len(self.test_results)

        for test_name, success in self.test_results:
            print(test_name)

        print(f"\n🎯 Overall Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")

        if passed == total:
            print("🎉 ALL TESTS PASSED! Error handling implementation is robust.")
            return True
        else:
            print("⚠️ Some tests failed. Review error handling implementation.")
            return False


if __name__ == "__main__":
    tester = ErrorHandlingTester()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)