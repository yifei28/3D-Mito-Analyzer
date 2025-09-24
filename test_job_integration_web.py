#!/usr/bin/env python3
"""
Streamlit web interface for testing job integration functionality with Playwright

This test interface validates Task 11.1: Add Segmentation Job Type Support
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
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from workflows.job_manager import (
    get_job_manager, JobType, get_supported_job_types,
    get_job_type_schema, submit_segmentation_job, get_segmentation_output_path
)


class JobIntegrationWebTester:
    """Web interface for testing job integration functionality."""

    def __init__(self):
        if 'job_manager' not in st.session_state:
            st.session_state.job_manager = None
        if 'test_results' not in st.session_state:
            st.session_state.test_results = []
        if 'created_jobs' not in st.session_state:
            st.session_state.created_jobs = []

    def setup_job_manager(self):
        """Set up the job manager."""
        try:
            if st.session_state.job_manager is None:
                with st.spinner("Initializing job manager..."):
                    st.session_state.job_manager = get_job_manager()
                st.success("✅ Job manager initialized successfully!")

                # Display job manager info
                queue_info = st.session_state.job_manager.get_job_queue_info()
                st.info(f"""
                **Job Manager Configuration:**
                - Max Workers: {queue_info['max_workers']}
                - Active Workers: {queue_info['active_workers']}
                - Queue Size: {queue_info['queue_size']}
                - Total Jobs: {queue_info['total_jobs']}
                """)

            return True
        except Exception as e:
            st.error(f"❌ Failed to initialize job manager: {e}")
            return False

    def test_job_type_management(self):
        """Test job type enumeration and management."""
        st.subheader("📝 Job Type Management")

        # Test supported job types
        st.write("**Supported Job Types:**")

        try:
            supported_types = get_supported_job_types()

            col1, col2 = st.columns(2)

            with col1:
                st.write("**Available Types:**")
                for job_type in supported_types:
                    st.write(f"• {job_type}")

            with col2:
                st.write("**Type Validation:**")
                for job_type in supported_types:
                    is_valid = JobType.is_valid(job_type)
                    status = "✅" if is_valid else "❌"
                    st.write(f"{status} {job_type}: {is_valid}")

                # Test invalid type
                is_invalid_valid = JobType.is_valid("invalid_type")
                status = "✅" if not is_invalid_valid else "❌"
                st.write(f"{status} invalid_type: {is_invalid_valid}")

            if 'segmentation' in supported_types:
                st.success("✅ Segmentation job type properly supported")
            else:
                st.error("❌ Segmentation job type missing")

        except Exception as e:
            st.error(f"❌ Job type management test failed: {e}")

        # Test job type schemas
        st.write("**Job Type Schemas:**")

        try:
            for job_type in supported_types:
                with st.expander(f"Schema for '{job_type}'"):
                    schema = get_job_type_schema(job_type)
                    st.json(schema)

        except Exception as e:
            st.error(f"❌ Schema retrieval failed: {e}")

    def test_parameter_validation(self):
        """Test segmentation job parameter validation."""
        st.subheader("✅ Segmentation Job Validation")

        # Create test TIFF file
        if st.button("📁 Create Test TIFF File"):
            with tempfile.NamedTemporaryFile(mode='wb', suffix='.tif', delete=False) as tmp_file:
                test_data = np.random.randint(0, 255, (8, 256, 256), dtype=np.uint8)
                tifffile.imwrite(tmp_file.name, test_data)
                st.session_state.test_file_path = tmp_file.name
                st.success(f"✅ Test TIFF created: {tmp_file.name}")

        if 'test_file_path' in st.session_state:
            test_file = st.session_state.test_file_path

            st.write("**Parameter Validation Tests:**")

            # Test 1: Valid parameters
            st.write("**Test 1: Valid Parameters**")

            valid_params = {
                'input_path': test_file,
                'output_dir': '/tmp/test_segmentation',
                'force_cpu': True,
                'batch_size': 4,
                'timeout_minutes': 30.0
            }

            try:
                manager = st.session_state.job_manager
                validated = manager._validate_segmentation_parameters(valid_params)

                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Input Parameters:**")
                    st.json(valid_params)
                with col2:
                    st.write("**Validated Parameters:**")
                    st.json(validated)

                st.success("✅ Valid parameters correctly validated")

            except Exception as e:
                st.error(f"❌ Valid parameter validation failed: {e}")

            # Test 2: Invalid parameters
            st.write("**Test 2: Invalid Parameters**")

            invalid_test_cases = [
                {
                    'name': 'Missing input_path',
                    'params': {'output_dir': '/tmp/test'}
                },
                {
                    'name': 'Nonexistent file',
                    'params': {'input_path': '/nonexistent/file.tif'}
                },
                {
                    'name': 'Invalid batch size',
                    'params': {'input_path': test_file, 'batch_size': 100}
                },
                {
                    'name': 'Invalid memory limit',
                    'params': {'input_path': test_file, 'memory_limit_gb': -1}
                }
            ]

            for test_case in invalid_test_cases:
                try:
                    manager._validate_segmentation_parameters(test_case['params'])
                    st.error(f"❌ {test_case['name']}: Should have failed validation")
                except ValueError as e:
                    st.success(f"✅ {test_case['name']}: Correctly rejected - {str(e)[:50]}...")
                except Exception as e:
                    st.warning(f"⚠️ {test_case['name']}: Unexpected error - {str(e)[:50]}...")

    def test_job_creation(self):
        """Test segmentation job creation and management."""
        st.subheader("🔧 Job Creation Testing")

        if 'test_file_path' not in st.session_state:
            st.warning("⚠️ Please create a test TIFF file first in the Parameter Validation tab")
            return

        test_file = st.session_state.test_file_path

        # Job creation form
        st.write("**Create Segmentation Job:**")

        col1, col2 = st.columns(2)

        with col1:
            force_cpu = st.checkbox("Force CPU Processing", value=True)
            batch_size = st.slider("Batch Size", 1, 8, 2)
            timeout_minutes = st.slider("Timeout (minutes)", 1, 60, 10)

        with col2:
            output_dir = st.text_input("Output Directory", value="/tmp/test_segmentation")
            auto_chain = st.checkbox("Auto-chain Analysis", value=False)

        if st.button("🚀 Create Segmentation Job"):
            try:
                # Create job using convenience function
                job_id = submit_segmentation_job(
                    input_path=test_file,
                    output_dir=output_dir,
                    force_cpu=force_cpu,
                    batch_size=batch_size,
                    timeout_minutes=timeout_minutes,
                    auto_chain_analysis=auto_chain
                )

                st.success(f"✅ Job created successfully: {job_id}")
                st.session_state.created_jobs.append(job_id)

                # Display initial job status
                job_status = st.session_state.job_manager.get_job_status(job_id)
                if job_status:
                    st.json(job_status)

            except Exception as e:
                st.error(f"❌ Job creation failed: {e}")

        # Display created jobs
        if st.session_state.created_jobs:
            st.write("**Created Jobs:**")

            for job_id in st.session_state.created_jobs:
                with st.expander(f"Job {job_id[:8]}..."):
                    job_status = st.session_state.job_manager.get_job_status(job_id)
                    if job_status:
                        col1, col2 = st.columns(2)

                        with col1:
                            st.write("**Status:**")
                            status = job_status['status']
                            if status == 'completed':
                                st.success(f"Status: {status}")
                            elif status == 'failed':
                                st.error(f"Status: {status}")
                            elif status == 'running':
                                st.info(f"Status: {status}")
                            else:
                                st.write(f"Status: {status}")

                            st.write(f"Progress: {job_status.get('progress', 0):.1f}%")

                            if job_status.get('status_message'):
                                st.write(f"Message: {job_status['status_message']}")

                        with col2:
                            st.write("**Details:**")
                            st.write(f"Type: {job_status['job_type']}")
                            st.write(f"Created: {job_status['created_at']}")

                            if job_status.get('started_at'):
                                st.write(f"Started: {job_status['started_at']}")

                            if job_status.get('error_message'):
                                st.error(f"Error: {job_status['error_message']}")

                        # Test output path retrieval for completed jobs
                        if status == 'completed':
                            output_path = get_segmentation_output_path(job_id)
                            if output_path:
                                st.success(f"✅ Output path: {output_path}")
                            else:
                                st.warning("⚠️ No output path available")

                        # Job actions
                        if status in ['queued', 'running']:
                            if st.button(f"Cancel Job {job_id[:8]}", key=f"cancel_{job_id}"):
                                if st.session_state.job_manager.cancel_job(job_id):
                                    st.success("Job cancellation requested")
                                    st.rerun()
                                else:
                                    st.error("Failed to cancel job")

    def test_integration_with_existing_system(self):
        """Test integration with existing job system."""
        st.subheader("🔄 Integration with Existing System")

        try:
            # Test job queue info
            queue_info = st.session_state.job_manager.get_job_queue_info()

            col1, col2 = st.columns(2)

            with col1:
                st.write("**Queue Information:**")
                st.metric("Queue Size", queue_info['queue_size'])
                st.metric("Active Workers", queue_info['active_workers'])
                st.metric("Max Workers", queue_info['max_workers'])
                st.metric("Total Jobs", queue_info['total_jobs'])

            with col2:
                st.write("**Job Status Counts:**")
                status_counts = queue_info.get('status_counts', {})
                for status, count in status_counts.items():
                    st.metric(status.title(), count)

            # Test list all jobs
            st.write("**All Jobs in System:**")
            all_jobs = st.session_state.job_manager.list_jobs()

            if all_jobs:
                job_data = []
                for job_id, job_info in all_jobs.items():
                    job_data.append({
                        'ID': job_id[:8] + "...",
                        'Type': job_info['job_type'],
                        'Status': job_info['status'],
                        'Progress': f"{job_info['progress']:.1f}%",
                        'Created': job_info['created_at'].strftime('%H:%M:%S') if job_info['created_at'] else 'N/A'
                    })

                st.dataframe(job_data)
            else:
                st.info("No jobs in the system")

            # Test creating different job types
            st.write("**Test Other Job Types:**")

            if st.button("🧪 Create Test Job"):
                try:
                    test_job_id = st.session_state.job_manager.submit_job(
                        'test',
                        {'duration': 3, 'steps': 5}
                    )
                    st.success(f"✅ Test job created: {test_job_id}")
                except Exception as e:
                    st.error(f"❌ Test job creation failed: {e}")

        except Exception as e:
            st.error(f"❌ Integration test failed: {e}")

    def run_all_tests(self):
        """Run all job integration tests."""
        st.header("🧪 Job Integration Test Suite")
        st.markdown("This interface tests the job system integration for Task 11.1")

        if not self.setup_job_manager():
            st.stop()

        # Test tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📝 Job Type Management", "✅ Parameter Validation",
            "🔧 Job Creation", "🔄 System Integration"
        ])

        with tab1:
            self.test_job_type_management()

        with tab2:
            self.test_parameter_validation()

        with tab3:
            self.test_job_creation()

        with tab4:
            self.test_integration_with_existing_system()

        # Cleanup section
        st.sidebar.header("🧹 Cleanup")
        if st.sidebar.button("Clean Test Files"):
            if 'test_file_path' in st.session_state:
                try:
                    os.unlink(st.session_state.test_file_path)
                    del st.session_state.test_file_path
                    st.sidebar.success("Test files cleaned")
                except:
                    st.sidebar.warning("Could not clean test files")

        if st.sidebar.button("Clear All Jobs"):
            try:
                # Cancel all active jobs
                active_jobs = st.session_state.job_manager.list_active_jobs()
                for job_id in active_jobs:
                    st.session_state.job_manager.cancel_job(job_id)

                st.session_state.created_jobs = []
                st.sidebar.success("All jobs cleared")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Failed to clear jobs: {e}")


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Job Integration Test Suite",
        page_icon="🔧",
        layout="wide"
    )

    tester = JobIntegrationWebTester()
    tester.run_all_tests()


if __name__ == "__main__":
    main()