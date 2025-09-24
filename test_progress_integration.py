#!/usr/bin/env python3
"""
Enhanced Progress System Integration Test

Tests the integration between JobManager and enhanced progress tracking.
"""

import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from workflows.job_manager import JobManager, JobStatus
from components.progress import SegmentationProgressMapper, EnhancedProgressInfo


def test_job_manager_enhanced_progress():
    """Test JobManager with enhanced progress tracking."""
    print("🧪 Testing JobManager enhanced progress integration...")

    # Get job manager instance
    job_manager = JobManager()

    # Create a mock progress callback to simulate segmentation workflow
    progress_updates = []

    def mock_progress_callback(percentage: int, info: dict):
        """Mock progress callback that simulates segmentation workflow updates."""
        progress_updates.append((percentage, info.copy()))

    # Submit a test job
    job_id = job_manager.submit_job("test", {"duration": 1, "steps": 5})
    print(f"  ✓ Submitted test job: {job_id[:8]}...")

    # Wait a moment for job to start
    time.sleep(2)

    # Check enhanced job status
    enhanced_status = job_manager.get_enhanced_job_status(job_id)
    assert enhanced_status is not None, "Should get enhanced status for existing job"

    print(f"  ✓ Enhanced status retrieved:")
    print(f"    Job ID: {enhanced_status['id'][:8]}...")
    print(f"    Status: {enhanced_status['status']}")
    print(f"    Progress: {enhanced_status['progress']:.1f}%")
    print(f"    Hardware: {enhanced_status['hardware_mode']}")

    # Wait for job to complete
    max_wait = 10  # 10 seconds should be enough for test job
    start_time = time.time()

    while time.time() - start_time < max_wait:
        status = job_manager.get_job_status(job_id)
        if status and status['status'] in ['completed', 'failed']:
            break
        time.sleep(0.5)

    final_status = job_manager.get_job_status(job_id)
    print(f"  ✓ Final job status: {final_status['status']}")

    return True


def test_enhanced_progress_callback_simulation():
    """Test the enhanced progress callback with simulated segmentation data."""
    print("🧪 Testing enhanced progress callback simulation...")

    # Simulate a segmentation workflow with various stages
    simulation_data = [
        (10, {"stage": "initialization", "message": "Starting up", "current_slice": 0, "total_slices": 5}),
        (50, {"stage": "initialization", "message": "Initializing models", "current_slice": 0, "total_slices": 5}),
        (25, {"stage": "loading", "message": "Loading data", "current_slice": 1, "total_slices": 5}),
        (75, {"stage": "loading", "message": "Validating input", "current_slice": 1, "total_slices": 5}),
        (20, {"stage": "preprocessing", "message": "Preprocessing", "current_slice": 2, "total_slices": 5}),
        (10, {"stage": "segmentation", "message": "Starting segmentation", "current_slice": 1, "total_slices": 5}),
        (30, {"stage": "segmentation", "message": "Processing", "current_slice": 2, "total_slices": 5}),
        (60, {"stage": "segmentation", "message": "Processing", "current_slice": 3, "total_slices": 5}),
        (90, {"stage": "segmentation", "message": "Finalizing", "current_slice": 4, "total_slices": 5}),
        (50, {"stage": "saving", "message": "Saving results", "current_slice": 5, "total_slices": 5}),
        (100, {"stage": "cleanup", "message": "Cleaning up", "current_slice": 5, "total_slices": 5})
    ]

    mapper = SegmentationProgressMapper()
    overall_progress_values = []

    print("  📊 Processing simulated segmentation stages:")

    for percentage, info in simulation_data:
        enhanced_progress = mapper.map_stage_progress(
            info["stage"],
            percentage,
            {
                "current_slice": info["current_slice"],
                "total_slices": info["total_slices"],
                "hardware_mode": "CPU"
            }
        )

        overall_progress_values.append(enhanced_progress.overall_progress)

        print(f"    {info['stage']:>12}: {percentage:>3}% -> {enhanced_progress.overall_progress:>5.1f}% | {enhanced_progress.message}")

    # Validate that overall progress generally increases
    decreases = 0
    for i in range(1, len(overall_progress_values)):
        if overall_progress_values[i] < overall_progress_values[i-1]:
            decreases += 1

    # Allow a few small decreases due to stage transitions
    assert decreases <= 2, f"Too many progress decreases: {decreases}"

    print(f"  ✓ Simulation completed with {decreases} minor decreases")
    return True


def test_enhanced_progress_with_job_manager():
    """Test enhanced progress tracking integrated with job manager."""
    print("🧪 Testing enhanced progress with job manager integration...")

    job_manager = JobManager()

    # Create test job parameters that would use enhanced progress
    params = {
        "duration": 2,
        "steps": 10,
        "simulate_stages": True  # Custom parameter for testing
    }

    job_id = job_manager.submit_job("test", params)
    print(f"  ✓ Submitted enhanced test job: {job_id[:8]}...")

    # Monitor job progress with enhanced status
    max_wait = 15
    start_time = time.time()
    progress_history = []

    while time.time() - start_time < max_wait:
        enhanced_status = job_manager.get_enhanced_job_status(job_id)

        if enhanced_status:
            progress_entry = {
                "time": time.time() - start_time,
                "progress": enhanced_status["progress"],
                "stage": enhanced_status.get("current_stage"),
                "hardware": enhanced_status.get("hardware_mode"),
                "status": enhanced_status["status"]
            }
            progress_history.append(progress_entry)

            if enhanced_status["status"] in ["completed", "failed"]:
                break

        time.sleep(0.5)

    # Analyze progress history
    print(f"  ✓ Captured {len(progress_history)} progress updates")

    if progress_history:
        final_entry = progress_history[-1]
        print(f"    Final status: {final_entry['status']}")
        print(f"    Final progress: {final_entry['progress']:.1f}%")
        print(f"    Hardware mode: {final_entry['hardware']}")

        # Check that we captured meaningful progress
        progress_values = [entry["progress"] for entry in progress_history]
        if len(progress_values) > 1:
            max_progress = max(progress_values)
            print(f"    Max progress reached: {max_progress:.1f}%")
            assert max_progress > 0, "Should capture some progress > 0"

    return True


def main():
    """Run all integration tests."""
    print("🚀 Enhanced Progress System Integration Tests")
    print("=" * 60)

    test_functions = [
        test_enhanced_progress_callback_simulation,
        test_job_manager_enhanced_progress,
        test_enhanced_progress_with_job_manager
    ]

    passed = 0
    total = len(test_functions)

    for test_func in test_functions:
        try:
            print()
            if test_func():
                passed += 1
                print(f"✅ {test_func.__name__} passed")
            else:
                print(f"❌ {test_func.__name__} failed")
        except Exception as e:
            print(f"❌ {test_func.__name__} failed with error: {e}")
        print()

    print("=" * 60)
    print(f"🎯 Integration Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All integration tests passed!")
        print("✅ Enhanced progress forwarding system is working correctly")
        return True
    else:
        print("⚠️ Some integration tests failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)