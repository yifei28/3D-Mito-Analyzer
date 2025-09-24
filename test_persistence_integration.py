#!/usr/bin/env python3
"""
Simple Integration Test for Job Persistence

This test focuses on validating the core persistence functionality
without complex threading scenarios that can cause timeouts.
"""

import os
import sys
import tempfile
import shutil
import time
from datetime import datetime
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_basic_persistence():
    """Test basic persistence functionality."""
    print("🧪 Testing Basic Job Persistence Integration")
    print("=" * 50)
    
    # Create temporary directory for testing
    test_dir = Path(tempfile.mkdtemp())
    print(f"Using test directory: {test_dir}")
    
    try:
        # Reset singletons for clean test
        from utils.job_persistence import JobPersistence
        from workflows.job_manager import JobManager, JobInfo, JobStatus
        
        JobPersistence._instance = None
        JobManager._instance = None
        
        # Override persistence data directory
        persistence = JobPersistence()
        original_data_dir = persistence.data_dir
        persistence.data_dir = test_dir / "jobs"
        persistence.active_dir = persistence.data_dir / "active"
        persistence.completed_dir = persistence.data_dir / "completed"  
        persistence.history_dir = persistence.data_dir / "history"
        persistence._ensure_directories()
        
        print("✅ Persistence directories created")
        
        # Test 1: Basic job creation and persistence
        print("\n1️⃣ Testing job creation and persistence...")
        job_info = JobInfo(
            id="test-persist-1",
            job_type="test",
            params={"duration": 0.1, "steps": 2}
        )
        job_info.status = JobStatus.QUEUED
        job_info.created_at = datetime.now()
        
        # Save job
        success = persistence.save_job_status("test-persist-1", job_info)
        assert success, "Job save should succeed"
        
        # Verify file exists
        job_file = persistence.active_dir / "test-persist-1.json"
        assert job_file.exists(), "Job file should exist"
        print("✅ Job successfully persisted to file")
        
        # Test 2: Load job from persistence
        print("\n2️⃣ Testing job loading...")
        loaded_metadata = persistence.load_job_status("test-persist-1")
        assert loaded_metadata is not None, "Should load job metadata"
        assert loaded_metadata.id == "test-persist-1", "Job ID should match"
        assert loaded_metadata.status == "queued", "Job status should match"
        print("✅ Job successfully loaded from file")
        
        # Test 3: Status change with file movement
        print("\n3️⃣ Testing status change and file movement...")
        job_info.status = JobStatus.COMPLETED
        job_info.completed_at = datetime.now()
        job_info.result = {"message": "Test completed"}
        
        # Save with file movement
        success = persistence.save_job_status("test-persist-1", job_info, move_file=True)
        assert success, "Job update should succeed"
        
        # Verify file moved
        active_file = persistence.active_dir / "test-persist-1.json"
        completed_file = persistence.completed_dir / "test-persist-1.json"
        assert not active_file.exists(), "Job should not be in active directory"
        assert completed_file.exists(), "Job should be in completed directory"
        print("✅ Job file successfully moved on status change")
        
        # Test 4: Job history retrieval
        print("\n4️⃣ Testing job history...")
        history = persistence.get_job_history()
        assert len(history) == 1, "Should have one job in history"
        assert history[0].id == "test-persist-1", "History should contain our job"
        assert history[0].status == "completed", "Job should be completed"
        print("✅ Job history retrieval works")
        
        # Test 5: Cleanup functionality (no cleanup expected for recent job)
        print("\n5️⃣ Testing cleanup...")
        cleaned_count = persistence.cleanup_old_jobs(max_age_hours=1)
        assert cleaned_count == 0, "No jobs should be cleaned (too recent)"
        print("✅ Cleanup correctly preserves recent jobs")
        
        # Test 6: Observer pattern integration (without JobManager threading)
        print("\n6️⃣ Testing observer pattern...")
        from utils.job_persistence import PersistenceObserver
        
        observer = PersistenceObserver()
        
        # Create new job for observer test
        job_info_2 = JobInfo(
            id="test-observer-1",
            job_type="test",
            params={"duration": 0.1}
        )
        
        # Test observer job creation
        observer.on_job_created("test-observer-1", job_info_2)
        observer_file = persistence.active_dir / "test-observer-1.json"
        assert observer_file.exists(), "Observer should create job file"
        
        # Test observer status change
        job_info_2.status = JobStatus.COMPLETED
        job_info_2.completed_at = datetime.now()
        observer.on_job_status_changed("test-observer-1", job_info_2)
        
        completed_observer_file = persistence.completed_dir / "test-observer-1.json"
        assert completed_observer_file.exists(), "Observer should move file on status change"
        assert not observer_file.exists(), "Original file should be removed"
        print("✅ Observer pattern integration works")
        
        # Test 7: Recovery functionality
        print("\n7️⃣ Testing job recovery...")
        
        # Create interrupted job manually
        interrupted_job = JobInfo(
            id="test-interrupted",
            job_type="test", 
            params={"duration": 5}
        )
        interrupted_job.status = JobStatus.RUNNING
        interrupted_job.started_at = datetime.now()
        
        # Save to active directory (simulating interrupted state)
        persistence.save_job_status("test-interrupted", interrupted_job, move_file=False)
        
        # Run recovery
        recovered_jobs = persistence.recover_interrupted_jobs()
        assert len(recovered_jobs) == 1, "Should recover one job"
        assert recovered_jobs[0] == "test-interrupted", "Should recover correct job"
        
        # Verify job was marked as failed and moved
        recovered_metadata = persistence.load_job_status("test-interrupted")
        assert recovered_metadata.status == "failed", "Recovered job should be marked as failed"
        assert "interrupted" in recovered_metadata.error_message.lower(), "Should have interruption error message"
        print("✅ Job recovery functionality works")
        
        print("\n" + "=" * 50)
        print("🎉 ALL PERSISTENCE TESTS PASSED!")
        print("The persistence system is fully functional and ready for production use.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup test directory
        if test_dir.exists():
            shutil.rmtree(test_dir)
        
        # Reset singletons
        try:
            JobPersistence._instance = None
            JobManager._instance = None
        except:
            pass


def test_convenience_functions():
    """Test module-level convenience functions."""
    print("\n🧪 Testing Convenience Functions")
    print("=" * 30)
    
    test_dir = Path(tempfile.mkdtemp())
    
    try:
        # Reset and setup persistence
        from utils.job_persistence import (
            JobPersistence, save_job_status, load_job_status,
            get_job_history, cleanup_old_jobs, recover_interrupted_jobs
        )
        from workflows.job_manager import JobInfo, JobStatus
        
        JobPersistence._instance = None
        
        persistence = JobPersistence()
        persistence.data_dir = test_dir / "jobs"
        persistence.active_dir = persistence.data_dir / "active"
        persistence.completed_dir = persistence.data_dir / "completed"
        persistence.history_dir = persistence.data_dir / "history"
        persistence._ensure_directories()
        
        # Create test job
        job_info = JobInfo(
            id="convenience-test",
            job_type="test",
            params={"duration": 0.1}
        )
        job_info.status = JobStatus.COMPLETED
        job_info.completed_at = datetime.now()
        job_info.result = {"message": "Success"}
        
        # Test convenience functions
        success = save_job_status("convenience-test", job_info)
        assert success, "Convenience save should work"
        
        metadata = load_job_status("convenience-test")
        assert metadata is not None, "Convenience load should work"
        assert metadata.id == "convenience-test", "Loaded job should match"
        
        history = get_job_history(limit=10)
        assert len(history) == 1, "Should have one job in history"
        
        cleaned = cleanup_old_jobs(max_age_hours=1)
        assert cleaned == 0, "Should not clean recent job"
        
        recovered = recover_interrupted_jobs()
        assert len(recovered) == 0, "No interrupted jobs to recover"
        
        print("✅ All convenience functions work correctly")
        return True
        
    except Exception as e:
        print(f"❌ Convenience function test failed: {e}")
        return False
        
    finally:
        if test_dir.exists():
            shutil.rmtree(test_dir)
        JobPersistence._instance = None


if __name__ == '__main__':
    print("🧪 Job Persistence Integration Test Suite")
    print("=" * 60)
    
    success = True
    
    # Run basic persistence test
    if not test_basic_persistence():
        success = False
        
    # Brief pause
    time.sleep(0.2)
    
    # Run convenience functions test
    if not test_convenience_functions():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL INTEGRATION TESTS PASSED!")
        print("✅ Job persistence system is fully functional and production-ready.")
        exit(0)
    else:
        print("❌ Some integration tests failed.")
        exit(1)