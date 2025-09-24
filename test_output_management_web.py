#!/usr/bin/env python3
"""
Streamlit web interface for testing output management and metadata generation functionality with Playwright

This test interface validates Task 10.6: Create Output Management and Hardware-Aware Metadata
"""

import streamlit as st
import os
import sys
import tempfile
import numpy as np
import tifffile
import json
import time
import shutil
from datetime import datetime
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from workflows.segmentation import SegmentationWorkflow


class OutputManagementWebTester:
    """Web interface for testing output management and metadata functionality."""

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

    def test_file_operations(self):
        """Test file naming, atomic operations, and permissions."""
        st.subheader("📁 File Operations Test")

        # Test file naming with various inputs
        st.write("**Test 1: Enhanced File Naming**")

        test_filenames = [
            "normal_file.tif",
            "file with spaces.tif",
            "file@#$%special&chars.tif",
            "very_long_filename_that_exceeds_normal_limits_and_should_be_truncated_properly_to_ensure_filesystem_compatibility.tif"
        ]

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Original Filenames:**")
            for filename in test_filenames:
                st.text(filename[:50] + "..." if len(filename) > 50 else filename)

        with col2:
            st.write("**Enhanced Filenames:**")
            for filename in test_filenames:
                # Test filename generation
                with tempfile.TemporaryDirectory() as temp_dir:
                    test_path = os.path.join(temp_dir, filename)
                    enhanced_path = st.session_state.workflow._generate_output_filename(
                        test_path, {'test': True}
                    )
                    enhanced_name = os.path.basename(enhanced_path)
                    st.text(enhanced_name)

        st.write("**Test 2: Directory Validation**")

        # Test output directory validation
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = os.path.join(temp_dir, "test_output.tif")

            try:
                st.session_state.workflow._validate_output_directory(test_file)
                st.success("✅ Directory validation passed")
            except Exception as e:
                st.error(f"❌ Directory validation failed: {e}")

        st.write("**Test 3: File Cleanup**")

        # Test cleanup functionality
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create some temporary files
            temp_files = [
                os.path.join(temp_dir, "test_output.tif"),
                os.path.join(temp_dir, "test_output.tmp"),
                os.path.join(temp_dir, "test_output_temp_123.tif")
            ]

            for temp_file in temp_files:
                with open(temp_file, 'w') as f:
                    f.write("test content")

            # Test cleanup
            st.session_state.workflow._cleanup_partial_files(temp_files[0])

            remaining_files = os.listdir(temp_dir)
            if len(remaining_files) == 0:
                st.success("✅ File cleanup working correctly")
            else:
                st.warning(f"⚠️ {len(remaining_files)} files remain after cleanup")

    def test_metadata_generation(self):
        """Test comprehensive metadata generation."""
        st.subheader("📊 Metadata Generation Test")

        # Create test data for metadata generation
        test_segmentation = np.random.randint(0, 5, (10, 256, 256), dtype=np.uint8)

        test_metadata = {
            'input_path': '/test/input.tif',
            'original_shape': (10, 512, 512),
            'batch_size': 4,
            'model_path': '/test/model.hdf5',
            'processing_mode': 'test_mode'
        }

        start_time = time.time() - 30  # Simulate 30 second processing

        try:
            # Test comprehensive metadata generation
            comprehensive_metadata = st.session_state.workflow._generate_comprehensive_metadata(
                test_metadata, test_segmentation, start_time
            )

            # Display metadata structure
            st.write("**Generated Metadata Structure:**")

            # Create tabs for different metadata sections
            meta_tabs = st.tabs(["Processing", "Hardware", "Performance", "TensorFlow", "Input/Output"])

            with meta_tabs[0]:
                st.json(comprehensive_metadata.get('processing', {}))

            with meta_tabs[1]:
                st.json(comprehensive_metadata.get('hardware', {}))

            with meta_tabs[2]:
                st.json(comprehensive_metadata.get('performance', {}))

            with meta_tabs[3]:
                st.json(comprehensive_metadata.get('tensorflow', {}))

            with meta_tabs[4]:
                input_output = {
                    'input': comprehensive_metadata.get('input', {}),
                    'output': comprehensive_metadata.get('output', {})
                }
                st.json(input_output)

            # Validate metadata content
            st.write("**Metadata Validation:**")

            validation_results = []

            # Check required fields
            required_fields = ['metadata_version', 'generation_timestamp', 'processing', 'hardware', 'performance']
            for field in required_fields:
                if field in comprehensive_metadata:
                    validation_results.append(f"✅ {field}: Present")
                else:
                    validation_results.append(f"❌ {field}: Missing")

            # Check performance metrics
            if 'performance' in comprehensive_metadata:
                perf = comprehensive_metadata['performance']
                if 'processing_duration_seconds' in perf and perf['processing_duration_seconds'] > 0:
                    validation_results.append("✅ Processing duration: Valid")
                else:
                    validation_results.append("❌ Processing duration: Invalid")

                if 'slices_per_second' in perf:
                    validation_results.append("✅ Performance metrics: Present")
                else:
                    validation_results.append("❌ Performance metrics: Missing")

            for result in validation_results:
                if "✅" in result:
                    st.success(result)
                else:
                    st.error(result)

        except Exception as e:
            st.error(f"❌ Metadata generation test failed: {e}")

    def test_hardware_integration(self):
        """Test hardware-aware metadata collection."""
        st.subheader("🔧 Hardware Integration Test")

        try:
            # Test hardware metadata collection
            hardware_metadata = st.session_state.workflow._collect_hardware_metadata()

            st.write("**Current Hardware Configuration:**")

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Hardware Type", hardware_metadata.get('current_type', 'Unknown'))
                st.metric("Device Name", hardware_metadata.get('device_name', 'Unknown'))
                st.metric("Memory GB", f"{hardware_metadata.get('memory_gb', 0):.1f}")
                st.metric("CPU Count", hardware_metadata.get('cpu_count', 'Unknown'))

            with col2:
                if 'system_memory' in hardware_metadata:
                    mem = hardware_metadata['system_memory']
                    if 'error' not in mem:
                        st.metric("Total Memory", f"{mem.get('total_gb', 0):.1f} GB")
                        st.metric("Available Memory", f"{mem.get('available_gb', 0):.1f} GB")
                        st.metric("Memory Usage", f"{mem.get('percent_used', 0):.1f}%")
                    else:
                        st.error(f"Memory info error: {mem['error']}")

            # Test fallback event tracking
            st.write("**Fallback Event Tracking:**")

            if hardware_metadata.get('fallback_occurred', False):
                st.info(f"Fallback events recorded: {len(hardware_metadata.get('fallback_events', []))}")
                for i, event in enumerate(hardware_metadata.get('fallback_events', [])):
                    st.write(f"Event {i+1}: {event.get('from_hardware')} → {event.get('to_hardware')}")
                    st.caption(f"Reason: {event.get('reason')}")
            else:
                st.success("No fallback events recorded")

            # Test TensorFlow configuration collection
            st.write("**TensorFlow Configuration:**")

            tf_metadata = st.session_state.workflow._collect_tensorflow_metadata()

            if tf_metadata.get('status') == 'unavailable':
                st.warning("TensorFlow not available")
            elif tf_metadata.get('status') == 'error':
                st.error(f"TensorFlow error: {tf_metadata.get('error')}")
            else:
                tf_col1, tf_col2 = st.columns(2)

                with tf_col1:
                    st.metric("TensorFlow Version", tf_metadata.get('version', 'Unknown'))
                    st.metric("GPU Available", "Yes" if tf_metadata.get('gpu_available') else "No")

                with tf_col2:
                    devices = tf_metadata.get('physical_devices', {})
                    st.metric("CPU Devices", devices.get('cpu', 0))
                    st.metric("GPU Devices", devices.get('gpu', 0))

        except Exception as e:
            st.error(f"❌ Hardware integration test failed: {e}")

    def test_atomic_operations(self):
        """Test atomic file operations and integrity verification."""
        st.subheader("⚡ Atomic Operations Test")

        # Create test data
        test_data = np.random.randint(0, 255, (5, 128, 128), dtype=np.uint8)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = os.path.join(temp_dir, "test_atomic_output.tif")

            # Test metadata for atomic operations
            test_metadata = {
                'test_mode': True,
                'atomic_test': True
            }

            comprehensive_metadata = {
                'metadata_version': '2.0',
                'test_atomic_operations': True,
                'generation_timestamp': datetime.now().isoformat()
            }

            try:
                # Test atomic save operation
                with st.spinner("Testing atomic file operations..."):
                    saved_files = st.session_state.workflow._atomic_save_operation(
                        test_data, output_file, test_metadata, comprehensive_metadata
                    )

                st.write("**Atomic Operation Results:**")

                # Verify files were created
                if os.path.exists(saved_files['output_file']):
                    st.success(f"✅ Output file created: {os.path.basename(saved_files['output_file'])}")

                    # Check file size
                    file_size = os.path.getsize(saved_files['output_file']) / (1024 * 1024)
                    st.metric("Output File Size", f"{file_size:.2f} MB")
                else:
                    st.error("❌ Output file not created")

                if os.path.exists(saved_files['metadata_file']):
                    st.success(f"✅ Metadata file created: {os.path.basename(saved_files['metadata_file'])}")

                    # Validate metadata file content
                    with open(saved_files['metadata_file'], 'r') as f:
                        saved_metadata = json.load(f)

                    if saved_metadata.get('metadata_version') == '2.0':
                        st.success("✅ Metadata file content valid")
                    else:
                        st.error("❌ Metadata file content invalid")
                else:
                    st.error("❌ Metadata file not created")

                # Test checksum
                if 'checksum' in saved_files and saved_files['checksum'] != 'checksum_failed':
                    st.success(f"✅ Checksum generated: {saved_files['checksum'][:16]}...")
                else:
                    st.error("❌ Checksum generation failed")

                # Test file integrity verification
                integrity_valid = st.session_state.workflow._verify_file_integrity(
                    saved_files['output_file'], test_data
                )

                if integrity_valid:
                    st.success("✅ File integrity verification passed")
                else:
                    st.error("❌ File integrity verification failed")

            except Exception as e:
                st.error(f"❌ Atomic operations test failed: {e}")

    def test_performance_impact(self):
        """Test performance impact of metadata generation."""
        st.subheader("🏃 Performance Impact Test")

        # Create test data of different sizes
        test_sizes = [
            ("Small (5x128x128)", (5, 128, 128)),
            ("Medium (10x256x256)", (10, 256, 256)),
            ("Large (20x512x512)", (20, 512, 512))
        ]

        performance_results = []

        for size_name, shape in test_sizes:
            st.write(f"**Testing {size_name}:**")

            test_data = np.random.randint(0, 255, shape, dtype=np.uint8)

            # Test without metadata generation
            start_time = time.time()
            with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp_file:
                tifffile.imwrite(tmp_file.name, test_data)
                basic_save_time = time.time() - start_time
            os.unlink(tmp_file.name)

            # Test with full metadata generation
            with tempfile.TemporaryDirectory() as temp_dir:
                output_file = os.path.join(temp_dir, f"test_{size_name.lower().replace(' ', '_')}.tif")

                test_metadata = {
                    'input_path': '/test/input.tif',
                    'original_shape': shape,
                    'batch_size': 4,
                    'processing_mode': 'performance_test'
                }

                start_time = time.time()
                try:
                    save_result = st.session_state.workflow._save_segmentation_results(
                        test_data, output_file, test_metadata
                    )
                    enhanced_save_time = time.time() - start_time

                    overhead = enhanced_save_time - basic_save_time
                    overhead_percent = (overhead / basic_save_time) * 100 if basic_save_time > 0 else 0

                    performance_results.append({
                        'size': size_name,
                        'shape': shape,
                        'basic_time': basic_save_time,
                        'enhanced_time': enhanced_save_time,
                        'overhead': overhead,
                        'overhead_percent': overhead_percent
                    })

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Basic Save", f"{basic_save_time:.3f}s")
                    with col2:
                        st.metric("Enhanced Save", f"{enhanced_save_time:.3f}s")
                    with col3:
                        st.metric("Overhead", f"{overhead_percent:.1f}%")

                except Exception as e:
                    st.error(f"Performance test failed for {size_name}: {e}")

        # Summary
        if performance_results:
            st.write("**Performance Summary:**")
            avg_overhead = sum(r['overhead_percent'] for r in performance_results) / len(performance_results)

            if avg_overhead < 10:
                st.success(f"✅ Low performance impact: {avg_overhead:.1f}% average overhead")
            elif avg_overhead < 25:
                st.warning(f"⚠️ Moderate performance impact: {avg_overhead:.1f}% average overhead")
            else:
                st.error(f"❌ High performance impact: {avg_overhead:.1f}% average overhead")

    def test_full_pipeline(self):
        """Test complete output management pipeline with real segmentation."""
        st.subheader("🎯 Full Pipeline Test")

        # File upload
        uploaded_file = st.file_uploader(
            "Upload TIFF Image for Full Pipeline Testing",
            type=['tif', 'tiff'],
            help="Upload a TIFF z-stack to test the complete output management pipeline"
        )

        if uploaded_file is not None:
            # Save uploaded file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp_file:
                tmp_file.write(uploaded_file.read())
                input_path = tmp_file.name

            output_dir = tempfile.mkdtemp(prefix="output_management_test_")

            # Test configuration
            col1, col2 = st.columns(2)
            with col1:
                test_metadata_generation = st.checkbox("Test Metadata Generation", value=True)
            with col2:
                test_atomic_operations = st.checkbox("Test Atomic Operations", value=True)

            if st.button("🚀 Run Full Pipeline Test"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                results_area = st.empty()

                def progress_callback(percent, info):
                    progress_bar.progress(int(percent))
                    message = info.get('message', 'Processing...')
                    status_text.text(f"Progress: {percent:.1f}% - {message}")

                try:
                    start_time = time.time()

                    # Run segmentation with enhanced output management
                    result = st.session_state.workflow.run_segmentation(
                        input_path=input_path,
                        output_dir=output_dir,
                        progress_callback=progress_callback
                    )

                    end_time = time.time()
                    duration = end_time - start_time

                    # Display results
                    with results_area.container():
                        st.success(f"✅ Full pipeline completed in {duration:.1f} seconds!")

                        # File results
                        st.write("**Output Files:**")
                        col1, col2 = st.columns(2)

                        with col1:
                            if 'output_file' in result:
                                output_file = result['output_file']
                                if os.path.exists(output_file):
                                    file_size = os.path.getsize(output_file) / (1024 * 1024)
                                    st.success(f"📁 Segmentation: {os.path.basename(output_file)} ({file_size:.2f}MB)")
                                else:
                                    st.error("❌ Segmentation file not found")

                        with col2:
                            if 'metadata_file' in result:
                                metadata_file = result['metadata_file']
                                if os.path.exists(metadata_file):
                                    st.success(f"📄 Metadata: {os.path.basename(metadata_file)}")

                                    # Show metadata preview
                                    with open(metadata_file, 'r') as f:
                                        metadata_content = json.load(f)

                                    with st.expander("View Metadata Preview"):
                                        st.json({
                                            'metadata_version': metadata_content.get('metadata_version'),
                                            'processing_duration': metadata_content.get('processing', {}).get('duration_seconds'),
                                            'hardware_type': metadata_content.get('hardware', {}).get('current_type'),
                                            'performance_metrics': metadata_content.get('performance', {})
                                        })
                                else:
                                    st.error("❌ Metadata file not found")

                        # Performance metrics
                        st.write("**Performance Metrics:**")
                        perf_col1, perf_col2, perf_col3 = st.columns(3)

                        with perf_col1:
                            st.metric("Duration", f"{duration:.1f}s")
                        with perf_col2:
                            if 'file_size_mb' in result:
                                st.metric("Output Size", f"{result['file_size_mb']:.2f}MB")
                        with perf_col3:
                            if 'checksum' in result:
                                st.metric("Integrity Check", "✅ Passed" if result['checksum'] != 'checksum_failed' else "❌ Failed")

                        # Hardware summary
                        if 'metadata_summary' in result:
                            summary = result['metadata_summary']
                            st.write("**Hardware Summary:**")
                            st.info(f"""
                            - Hardware: {summary.get('hardware_type', 'Unknown')}
                            - Fallback Events: {'Yes' if summary.get('fallback_occurred') else 'No'}
                            - Performance: {summary.get('performance_metrics', {}).get('slices_per_second', 'N/A')} slices/sec
                            """)

                except Exception as e:
                    st.error(f"❌ Full pipeline test failed: {e}")
                    st.exception(e)

                finally:
                    # Cleanup
                    try:
                        os.unlink(input_path)
                        shutil.rmtree(output_dir, ignore_errors=True)
                    except:
                        pass

        else:
            # Create sample TIFF for testing
            if st.button("📁 Create Sample TIFF for Testing"):
                sample_data = np.random.randint(0, 255, (8, 256, 256), dtype=np.uint8)
                sample_file = "sample_output_test.tif"
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
        """Run all output management tests."""
        st.header("🧪 Output Management & Metadata Test Suite")
        st.markdown("This interface tests the comprehensive output management implementation for Task 10.6")

        if not self.setup_workflow():
            st.stop()

        # Test tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📁 File Operations", "📊 Metadata Generation", "🔧 Hardware Integration",
            "⚡ Performance Testing", "🎯 Full Pipeline"
        ])

        with tab1:
            self.test_file_operations()

        with tab2:
            self.test_metadata_generation()

        with tab3:
            self.test_hardware_integration()

        with tab4:
            self.test_performance_impact()

        with tab5:
            self.test_full_pipeline()


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Output Management Test Suite",
        page_icon="📊",
        layout="wide"
    )

    tester = OutputManagementWebTester()
    tester.run_all_tests()


if __name__ == "__main__":
    main()