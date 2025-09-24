#!/usr/bin/env python3
"""
Streamlit web interface for testing error handling functionality with Playwright
"""

import streamlit as st
import os
import sys
import tempfile
import numpy as np
import tifffile
import json
import time
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from workflows.segmentation import SegmentationWorkflow, HardwareType, ErrorType


class ErrorHandlingWebTester:
    """Web interface for testing error handling functionality."""

    def __init__(self):
        if 'workflow' not in st.session_state:
            st.session_state.workflow = None
        if 'test_results' not in st.session_state:
            st.session_state.test_results = []

    def setup_workflow(self):
        """Set up the segmentation workflow."""
        try:
            if st.session_state.workflow is None:
                with st.spinner("Initializing segmentation workflow..."):
                    st.session_state.workflow = SegmentationWorkflow(force_cpu=True)
                st.success("✅ Workflow initialized successfully!")

                # Display hardware configuration
                config = st.session_state.workflow.hardware_config
                st.info(f"""
                **Hardware Configuration:**
                - Type: {config.hardware_type.value.upper()}
                - Device: {config.device_name}
                - Memory: {config.memory_gb:.1f}GB
                - Batch Size: {config.batch_size}
                """)

            return True
        except Exception as e:
            st.error(f"❌ Failed to initialize workflow: {e}")
            return False

    def test_gpu_oom_detection(self):
        """Test GPU OOM error detection."""
        st.subheader("🔍 GPU OOM Detection Test")

        test_errors = [
            ("ResourceExhaustedError: OOM when allocating tensor", True),
            ("CUDA_ERROR_OUT_OF_MEMORY: out of memory", True),
            ("Failed to allocate GPU device memory", True),
            ("cudnnConvolutionForward: CUDNN_STATUS_ALLOC_FAILED", True),
            ("Regular processing error", False),
            ("File not found error", False)
        ]

        results = []
        for error_msg, expected in test_errors:
            error = Exception(error_msg)
            detected = st.session_state.workflow._detect_gpu_oom_error(error)
            success = detected == expected
            status = "✅" if success else "❌"
            results.append({
                "error": error_msg[:50] + "...",
                "expected": expected,
                "detected": detected,
                "success": success,
                "status": status
            })

        # Display results table
        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
        with col1:
            st.write("**Error Message**")
        with col2:
            st.write("**Expected**")
        with col3:
            st.write("**Detected**")
        with col4:
            st.write("**Result**")

        for result in results:
            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
            with col1:
                st.write(result["error"])
            with col2:
                st.write(result["expected"])
            with col3:
                st.write(result["detected"])
            with col4:
                st.write(result["status"])

        passed = sum(1 for r in results if r["success"])
        total = len(results)
        if passed == total:
            st.success(f"🎉 All {total} GPU OOM detection tests passed!")
        else:
            st.warning(f"⚠️ {passed}/{total} tests passed")

        return results

    def test_memory_monitoring(self):
        """Test memory monitoring functionality."""
        st.subheader("🧠 Memory Monitoring Test")

        try:
            needs_adjustment, memory_stats = st.session_state.workflow.monitor_memory_during_processing()

            # Display current memory status
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Memory Usage", f"{memory_stats['percent_used']:.1f}%")
            with col2:
                st.metric("Available Memory", f"{memory_stats['available_gb']:.1f}GB")
            with col3:
                st.metric("Needs Adjustment", "Yes" if needs_adjustment else "No")

            # Test adaptive memory management
            current_batch = st.slider("Current Batch Size", 1, 16, 4)
            new_batch = st.session_state.workflow._adaptive_memory_management(
                current_batch, memory_stats['percent_used']
            )

            if new_batch != current_batch:
                st.info(f"📊 Adaptive batch sizing: {current_batch} → {new_batch}")
            else:
                st.info("📊 Batch size unchanged (memory usage within normal range)")

            # Memory cleanup test
            if st.button("🧹 Test Emergency Memory Cleanup"):
                st.session_state.workflow._emergency_memory_cleanup()
                st.success("Emergency memory cleanup executed!")

            return True

        except Exception as e:
            st.error(f"❌ Memory monitoring test failed: {e}")
            return False

    def test_file_operations(self):
        """Test file system error handling."""
        st.subheader("📁 File System Error Handling Test")

        # Create test files
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = os.path.join(temp_dir, "test_image.tif")

            # Create a small test TIFF
            test_data = np.random.randint(0, 255, (5, 256, 256), dtype=np.uint8)
            tifffile.imwrite(test_file, test_data)

            # Test 1: Valid file permissions
            st.write("**Test 1: Valid File Permissions**")
            is_valid, error_msg = st.session_state.workflow._validate_file_permissions(
                test_file, temp_dir
            )
            if is_valid:
                st.success("✅ Valid file permissions detected correctly")
            else:
                st.error(f"❌ Permission check failed: {error_msg}")

            # Test 2: Disk space check
            st.write("**Test 2: Disk Space Check**")
            col1, col2 = st.columns(2)

            with col1:
                # Reasonable space requirement
                has_space = st.session_state.workflow._check_disk_space(temp_dir, 0.1)
                if has_space:
                    st.success("✅ Sufficient disk space (0.1GB)")
                else:
                    st.warning("⚠️ Insufficient disk space detected")

            with col2:
                # Unreasonable space requirement
                has_space_big = st.session_state.workflow._check_disk_space(temp_dir, 100000)
                if not has_space_big:
                    st.success("✅ Correctly rejected excessive space requirement (100TB)")
                else:
                    st.error("❌ Should have rejected excessive space requirement")

            return True

    def test_hardware_fallback(self):
        """Test hardware fallback functionality."""
        st.subheader("🔄 Hardware Fallback Test")

        # Display current hardware state
        config = st.session_state.workflow.hardware_config

        col1, col2 = st.columns(2)
        with col1:
            st.info(f"""
            **Current Hardware:**
            - Type: {config.hardware_type.value}
            - Original: {config.original_hardware_type.value if config.original_hardware_type else 'N/A'}
            - Fallback Events: {len(config.fallback_events)}
            """)

        with col2:
            # Test fallback execution
            if st.button("🔄 Test CPU Fallback"):
                success = st.session_state.workflow._fallback_to_cpu(
                    reason="Manual test from web interface",
                    error_details="User-initiated fallback test"
                )

                if success:
                    st.success("✅ Fallback executed successfully!")
                    st.json({
                        "events": len(config.fallback_events),
                        "current_type": config.hardware_type.value,
                        "last_event": config.fallback_events[-1].__dict__ if config.fallback_events else None
                    })
                else:
                    st.warning("⚠️ Fallback not needed (already on CPU)")

        # Display fallback history
        if config.fallback_events:
            st.write("**Fallback History:**")
            for i, event in enumerate(config.fallback_events):
                st.write(f"{i+1}. {event.from_hardware.value} → {event.to_hardware.value}: {event.reason}")
                st.caption(f"   Time: {datetime.fromtimestamp(event.timestamp)}")

    def test_retry_logic(self):
        """Test automatic retry logic."""
        st.subheader("🔁 Retry Logic Test")

        failure_count = st.slider("Simulated Failures Before Success", 0, 5, 2)
        max_retries = st.slider("Maximum Retries", 1, 5, 3)

        if st.button("🔁 Test Retry Logic"):
            # Create a failing operation
            call_count = [0]

            def failing_operation():
                call_count[0] += 1
                if call_count[0] <= failure_count:
                    raise Exception(f"Simulated failure #{call_count[0]}")
                return "Operation succeeded!"

            try:
                with st.spinner(f"Testing retry logic (max {max_retries} retries)..."):
                    result = st.session_state.workflow._execute_with_hardware_fallback(
                        failing_operation,
                        "Test retry operation",
                        max_retries=max_retries
                    )

                st.success(f"✅ {result}")
                st.info(f"📊 Operation succeeded after {call_count[0]} attempts")

            except Exception as e:
                st.error(f"❌ Operation failed after {max_retries + 1} attempts: {e}")
                st.warning(f"📊 Total attempts made: {call_count[0]}")

    def test_comprehensive_segmentation(self):
        """Test complete segmentation pipeline with error handling."""
        st.subheader("🎯 Comprehensive Segmentation Test")

        # File upload
        uploaded_file = st.file_uploader(
            "Upload TIFF Image for Testing",
            type=['tif', 'tiff'],
            help="Upload a TIFF z-stack to test the complete segmentation pipeline"
        )

        if uploaded_file is not None:
            # Save uploaded file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp_file:
                tmp_file.write(uploaded_file.read())
                input_path = tmp_file.name

            output_dir = tempfile.mkdtemp(prefix="segmentation_test_")

            # Test configuration
            col1, col2 = st.columns(2)
            with col1:
                simulate_errors = st.checkbox("Simulate Random Errors", help="Inject random errors to test error handling")
            with col2:
                show_progress = st.checkbox("Show Detailed Progress", value=True)

            if st.button("🚀 Run Segmentation Test"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                log_area = st.empty()

                logs = []

                def progress_callback(percent, info):
                    progress_bar.progress(int(percent))
                    message = info.get('message', 'Processing...')
                    status_text.text(f"Progress: {percent:.1f}% - {message}")

                    if show_progress:
                        logs.append(f"{percent:.1f}%: {message}")
                        log_area.text_area("Processing Log", "\n".join(logs[-10:]), height=150)

                try:
                    start_time = time.time()

                    result = st.session_state.workflow.run_segmentation(
                        input_path=input_path,
                        output_dir=output_dir,
                        progress_callback=progress_callback
                    )

                    end_time = time.time()
                    duration = end_time - start_time

                    st.success(f"✅ Segmentation completed in {duration:.1f} seconds!")

                    # Display results
                    if 'output_file' in result:
                        st.info(f"📁 Output file: {result['output_file']}")

                    if 'metadata_file' in result:
                        st.info(f"📄 Metadata file: {result['metadata_file']}")

                    # Display hardware events
                    config = st.session_state.workflow.hardware_config
                    if config.has_fallback_occurred:
                        st.warning(f"⚠️ {len(config.fallback_events)} hardware fallback event(s) occurred")

                    # Show result summary
                    st.json(result)

                except Exception as e:
                    st.error(f"❌ Segmentation failed: {e}")
                    st.exception(e)

                finally:
                    # Cleanup
                    try:
                        os.unlink(input_path)
                        import shutil
                        shutil.rmtree(output_dir, ignore_errors=True)
                    except:
                        pass

        else:
            # Create sample TIFF for testing
            if st.button("📁 Create Sample TIFF for Testing"):
                sample_data = np.random.randint(0, 255, (5, 256, 256), dtype=np.uint8)
                sample_file = "sample_test.tif"
                tifffile.imwrite(sample_file, sample_data)

                with open(sample_file, "rb") as f:
                    st.download_button(
                        label="📥 Download Sample TIFF",
                        data=f.read(),
                        file_name=sample_file,
                        mime="image/tiff"
                    )

                os.unlink(sample_file)

    def run_all_tests(self):
        """Run all error handling tests."""
        st.header("🧪 Error Handling Test Suite")
        st.markdown("This interface tests the robust error handling implementation for Task 10.5")

        if not self.setup_workflow():
            st.stop()

        # Test tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🔍 GPU OOM", "🧠 Memory", "📁 File System",
            "🔄 Hardware Fallback", "🔁 Retry Logic", "🎯 Full Pipeline"
        ])

        with tab1:
            self.test_gpu_oom_detection()

        with tab2:
            self.test_memory_monitoring()

        with tab3:
            self.test_file_operations()

        with tab4:
            self.test_hardware_fallback()

        with tab5:
            self.test_retry_logic()

        with tab6:
            self.test_comprehensive_segmentation()


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Error Handling Test Suite",
        page_icon="🧪",
        layout="wide"
    )

    tester = ErrorHandlingWebTester()
    tester.run_all_tests()


if __name__ == "__main__":
    main()