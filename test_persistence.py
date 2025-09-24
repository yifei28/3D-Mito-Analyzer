#!/usr/bin/env python3
"""
Test Suite for Job Persistence System

Comprehensive tests for the persistence layer including file operations,
observer integration, job recovery, and cross-session functionality.
"""

import os
import sys
import time
import tempfile
import shutil
import unittest
import threading
from datetime import datetime, timedelta
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.job_persistence import (
    JobPersistence, JobMetadata, PersistenceObserver,
    save_job_status, load_job_status, get_job_history,
    cleanup_old_jobs, recover_interrupted_jobs
)
from workflows.job_manager import JobManager, JobStatus, JobInfo


class TestJobPersistence(unittest.TestCase):
    """Test suite for JobPersistence core functionality."""
    
    def setUp(self):
        """Set up test fixtures with temporary directory."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.original_data_dir = None
        
        # Reset singleton instances
        JobPersistence._instance = None
        JobManager._instance = None
        
        # Create persistence instance with test directory
        self.persistence = JobPersistence()
        
        # Override data directory for testing
        self.original_data_dir = self.persistence.data_dir
        self.persistence.data_dir = self.test_dir / "jobs"
        self.persistence.active_dir = self.persistence.data_dir / "active"
        self.persistence.completed_dir = self.persistence.data_dir / "completed"
        self.persistence.history_dir = self.persistence.data_dir / "history"
        self.persistence._ensure_directories()
    
    def tearDown(self):
        """Clean up after each test."""
        # Restore original data directory if needed
        if self.original_data_dir:
            self.persistence.data_dir = self.original_data_dir
        
        # Clean up test directory
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        
        # Reset singletons
        JobPersistence._instance = None
        JobManager._instance = None
    
    def test_singleton_pattern(self):
        """Test that JobPersistence follows singleton pattern."""
        persistence1 = JobPersistence()
        persistence2 = JobPersistence()
        self.assertIs(persistence1, persistence2)
    
    def test_job_metadata_serialization(self):
        """Test JobMetadata to/from dict conversion."""
        metadata = JobMetadata(
            id="test-job-1",
            job_type="test",
            status="completed",
            progress=100.0,
            status_message="Test completed",
            error_message="",
            error_trace="",
            created_at=datetime.now().isoformat(),
            started_at=datetime.now().isoformat(),
            completed_at=datetime.now().isoformat(),
            total_time=5.5,
            input_files=["/test/input.tiff"],
            output_files=["/test/output.json"],
            parameters={"duration": 0.1, "steps": 2},
            result={"message": "Test completed"}
        )
        
        # Test serialization
        data_dict = metadata.to_dict()
        self.assertIsInstance(data_dict, dict)
        self.assertEqual(data_dict["id"], "test-job-1")
        self.assertEqual(data_dict["total_time"], 5.5)
        
        # Test deserialization
        restored_metadata = JobMetadata.from_dict(data_dict)
        self.assertEqual(restored_metadata.id, metadata.id)
        self.assertEqual(restored_metadata.total_time, metadata.total_time)
        self.assertEqual(restored_metadata.parameters, metadata.parameters)
    
    def test_atomic_write_operations(self):
        """Test atomic file write operations."""
        test_file = self.test_dir / "test_atomic.json"
        test_data = {"test": "data", "number": 42}
        
        # Test successful write
        self.persistence._atomic_write(test_file, test_data)
        self.assertTrue(test_file.exists())
        
        # Verify content
        import json
        with open(test_file) as f:
            loaded_data = json.load(f)
        self.assertEqual(loaded_data, test_data)
    
    def test_job_file_path_logic(self):
        """Test job file path determination based on status."""
        job_id = "test-job-1"
        
        # Test active status
        active_path = self.persistence._get_job_file_path(job_id, "running")
        self.assertEqual(active_path.parent.name, "active")
        
        # Test completed status
        completed_path = self.persistence._get_job_file_path(job_id, "completed")
        self.assertEqual(completed_path.parent.name, "completed")
        
        # Test unknown status defaults to active
        unknown_path = self.persistence._get_job_file_path(job_id, "unknown")
        self.assertEqual(unknown_path.parent.name, "active")
    
    def test_save_and_load_job_status(self):
        """Test saving and loading job status."""
        # Create mock JobInfo
        job_info = JobInfo(
            id="test-job-1",
            job_type="test",
            params={"duration": 0.1, "steps": 2}
        )
        job_info.status = JobStatus.COMPLETED
        job_info.progress = 100.0
        job_info.status_message = "Test completed"
        job_info.started_at = datetime.now()
        job_info.completed_at = datetime.now()
        job_info.result = {"message": "Success"}
        
        # Test save
        success = self.persistence.save_job_status("test-job-1", job_info)
        self.assertTrue(success)
        
        # Verify file was created in correct directory
        job_file = self.persistence.completed_dir / "test-job-1.json"
        self.assertTrue(job_file.exists())
        
        # Test load
        loaded_metadata = self.persistence.load_job_status("test-job-1")
        self.assertIsNotNone(loaded_metadata)
        self.assertEqual(loaded_metadata.id, "test-job-1")
        self.assertEqual(loaded_metadata.status, "completed")
        self.assertEqual(loaded_metadata.progress, 100.0)
    
    def test_job_file_movement_on_status_change(self):
        """Test that job files move between directories on status change."""
        # Create job in active directory
        job_info = JobInfo(
            id="test-job-move",
            job_type="test",
            params={"duration": 0.1}
        )
        job_info.status = JobStatus.RUNNING
        
        # Save as running job
        self.persistence.save_job_status("test-job-move", job_info)
        active_file = self.persistence.active_dir / "test-job-move.json"
        self.assertTrue(active_file.exists())
        
        # Change status to completed
        job_info.status = JobStatus.COMPLETED
        job_info.completed_at = datetime.now()
        
        # Save with move
        self.persistence.save_job_status("test-job-move", job_info, move_file=True)
        
        # Verify file moved
        completed_file = self.persistence.completed_dir / "test-job-move.json"
        self.assertTrue(completed_file.exists())
        self.assertFalse(active_file.exists())
    
    def test_corrupted_file_handling(self):
        """Test handling of corrupted job files."""
        # Create corrupted file
        corrupted_file = self.persistence.active_dir / "corrupted-job.json"
        corrupted_file.write_text("invalid json content {")
        
        # Attempt to load
        metadata = self.persistence.load_job_status("corrupted-job")
        self.assertIsNone(metadata)
        
        # Verify file was quarantined
        quarantine_dir = self.persistence.data_dir / "quarantine"
        if quarantine_dir.exists():
            quarantined_files = list(quarantine_dir.glob("corrupted_*"))
            self.assertGreater(len(quarantined_files), 0)
    
    def test_job_history_retrieval(self):
        """Test job history retrieval with filtering and sorting."""
        # Create multiple test jobs with different statuses
        test_jobs = [
            ("job-1", "completed", datetime.now() - timedelta(hours=1)),
            ("job-2", "failed", datetime.now() - timedelta(hours=2)),
            ("job-3", "completed", datetime.now() - timedelta(hours=3)),
            ("job-4", "cancelled", datetime.now() - timedelta(hours=4))
        ]
        
        for job_id, status, created_at in test_jobs:
            job_info = JobInfo(id=job_id, job_type="test", params={})
            job_info.status = JobStatus(status)
            job_info.created_at = created_at
            job_info.completed_at = created_at + timedelta(minutes=5)
            
            self.persistence.save_job_status(job_id, job_info)
        
        # Test get all history
        all_jobs = self.persistence.get_job_history()
        self.assertEqual(len(all_jobs), 4)
        
        # Verify sorting (newest first)
        self.assertEqual(all_jobs[0].id, "job-1")
        self.assertEqual(all_jobs[-1].id, "job-4")
        
        # Test status filter
        completed_jobs = self.persistence.get_job_history(status_filter="completed")
        self.assertEqual(len(completed_jobs), 2)
        self.assertTrue(all(job.status == "completed" for job in completed_jobs))
        
        # Test limit
        limited_jobs = self.persistence.get_job_history(limit=2)
        self.assertEqual(len(limited_jobs), 2)
    
    def test_cleanup_old_jobs(self):
        """Test cleanup of old job files."""
        # Create old job files
        old_time = datetime.now() - timedelta(days=8)
        recent_time = datetime.now() - timedelta(hours=1)
        
        old_job = JobInfo(id="old-job", job_type="test", params={})
        old_job.status = JobStatus.COMPLETED
        old_job.created_at = old_time
        old_job.completed_at = old_time
        
        recent_job = JobInfo(id="recent-job", job_type="test", params={})
        recent_job.status = JobStatus.COMPLETED
        recent_job.created_at = recent_time
        recent_job.completed_at = recent_time
        
        # Save both jobs
        self.persistence.save_job_status("old-job", old_job)
        self.persistence.save_job_status("recent-job", recent_job)
        
        # Manually set file timestamps to simulate age
        old_file = self.persistence.completed_dir / "old-job.json"
        old_timestamp = old_time.timestamp()
        os.utime(old_file, (old_timestamp, old_timestamp))
        
        # Run cleanup (7 days)
        cleaned_count = self.persistence.cleanup_old_jobs(max_age_hours=168)
        
        # Verify cleanup results
        self.assertEqual(cleaned_count, 1)
        self.assertTrue((self.persistence.completed_dir / "recent-job.json").exists())
        # Old file should be moved to history or deleted
        self.assertFalse((self.persistence.completed_dir / "old-job.json").exists())
    
    def test_interrupted_job_recovery(self):
        """Test recovery of interrupted jobs."""
        # Create interrupted running job
        interrupted_job = JobInfo(id="interrupted-job", job_type="test", params={})
        interrupted_job.status = JobStatus.RUNNING
        interrupted_job.started_at = datetime.now() - timedelta(minutes=5)
        
        # Save to active directory (simulating interrupted state)
        self.persistence.save_job_status("interrupted-job", interrupted_job, move_file=False)
        
        # Verify file is in active directory
        active_file = self.persistence.active_dir / "interrupted-job.json"
        self.assertTrue(active_file.exists())
        
        # Run recovery
        recovered_jobs = self.persistence.recover_interrupted_jobs()
        
        # Verify recovery results
        self.assertEqual(len(recovered_jobs), 1)
        self.assertEqual(recovered_jobs[0], "interrupted-job")
        
        # Verify job was marked as failed and moved to completed
        completed_file = self.persistence.completed_dir / "interrupted-job.json"
        self.assertTrue(completed_file.exists())
        self.assertFalse(active_file.exists())
        
        # Verify job status
        recovered_metadata = self.persistence.load_job_status("interrupted-job")
        self.assertEqual(recovered_metadata.status, "failed")
        self.assertIn("interrupted", recovered_metadata.error_message.lower())


class TestPersistenceObserver(unittest.TestCase):
    """Test suite for PersistenceObserver integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = Path(tempfile.mkdtemp())
        
        # Reset singleton instances
        JobPersistence._instance = None
        JobManager._instance = None
        
        # Override persistence data directory
        persistence = JobPersistence()
        persistence.data_dir = self.test_dir / "jobs"
        persistence.active_dir = persistence.data_dir / "active"
        persistence.completed_dir = persistence.data_dir / "completed"
        persistence.history_dir = persistence.data_dir / "history"
        persistence._ensure_directories()
        
        self.observer = PersistenceObserver()
    
    def tearDown(self):
        """Clean up after each test."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        
        JobPersistence._instance = None
        JobManager._instance = None
    
    def test_observer_job_creation(self):
        """Test observer handles job creation."""
        job_info = JobInfo(
            id="observer-test-1",
            job_type="test",
            params={"duration": 0.1}
        )
        
        # Trigger observer
        self.observer.on_job_created("observer-test-1", job_info)
        
        # Verify job was persisted
        job_file = self.test_dir / "jobs" / "active" / "observer-test-1.json"
        self.assertTrue(job_file.exists())
    
    def test_observer_status_change(self):
        """Test observer handles status changes with file movement."""
        job_info = JobInfo(
            id="observer-test-2",
            job_type="test",
            params={"duration": 0.1}
        )
        
        # Create job
        self.observer.on_job_created("observer-test-2", job_info)
        active_file = self.test_dir / "jobs" / "active" / "observer-test-2.json"
        self.assertTrue(active_file.exists())
        
        # Change status to completed
        job_info.status = JobStatus.COMPLETED
        job_info.completed_at = datetime.now()
        
        # Trigger status change
        self.observer.on_job_status_changed("observer-test-2", job_info)
        
        # Verify file moved to completed
        completed_file = self.test_dir / "jobs" / "completed" / "observer-test-2.json"
        self.assertTrue(completed_file.exists())
        self.assertFalse(active_file.exists())
    
    def test_observer_progress_updates(self):
        """Test observer handles progress updates selectively."""
        job_info = JobInfo(
            id="observer-test-3",
            job_type="test",
            params={"duration": 0.1}
        )
        job_info.progress = 0.0
        
        # Create job first
        self.observer.on_job_created("observer-test-3", job_info)
        
        # Test progress updates at different milestones
        milestone_progresses = [10, 25, 50, 75, 100]
        for progress in milestone_progresses:
            job_info.progress = progress
            self.observer.on_job_progress_updated("observer-test-3", job_info)
        
        # Verify job file exists (progress updates should save to file)
        job_file = self.test_dir / "jobs" / "active" / "observer-test-3.json"
        self.assertTrue(job_file.exists())
        
        # Load and verify final progress
        persistence = JobPersistence()
        metadata = persistence.load_job_status("observer-test-3")
        self.assertEqual(metadata.progress, 100.0)


class TestJobManagerPersistenceIntegration(unittest.TestCase):
    """Test integration between JobManager and persistence system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = Path(tempfile.mkdtemp())
        
        # Reset singleton instances
        JobPersistence._instance = None
        JobManager._instance = None
        
        # Override persistence data directory before creating JobManager
        persistence = JobPersistence()
        persistence.data_dir = self.test_dir / "jobs"
        persistence.active_dir = persistence.data_dir / "active"
        persistence.completed_dir = persistence.data_dir / "completed"
        persistence.history_dir = persistence.data_dir / "history"
        persistence._ensure_directories()
    
    def tearDown(self):
        """Clean up after each test."""
        try:
            if hasattr(self, 'job_manager'):
                self.job_manager.shutdown(wait=True)
                time.sleep(0.1)
        except:
            pass
        
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        
        JobPersistence._instance = None
        JobManager._instance = None
    
    def test_job_manager_with_persistence(self):
        """Test JobManager automatically persists jobs."""
        self.job_manager = JobManager()
        
        # Submit a job
        job_id = self.job_manager.submit_job("test", {"duration": 0.1, "steps": 2})
        
        # Verify job was persisted on creation
        job_file = self.test_dir / "jobs" / "active" / f"{job_id}.json"
        self.assertTrue(job_file.exists())
        
        # Wait for job to complete
        max_wait = 3.0
        start_time = time.time()
        while time.time() - start_time < max_wait:
            status = self.job_manager.get_job_status(job_id)
            if status['status'] in ['completed', 'failed']:
                break
            time.sleep(0.1)
        
        # Verify job was moved to completed directory
        completed_file = self.test_dir / "jobs" / "completed" / f"{job_id}.json"
        self.assertTrue(completed_file.exists())
        self.assertFalse(job_file.exists())
        
        # Verify job can be loaded from persistence
        persistence = JobPersistence()
        metadata = persistence.load_job_status(job_id)
        self.assertIsNotNone(metadata)
        self.assertEqual(metadata.status, "completed")
    
    def test_job_cancellation_persistence(self):
        """Test job cancellation is properly persisted."""
        self.job_manager = JobManager()
        
        # Submit a long-running job
        job_id = self.job_manager.submit_job("test", {"duration": 10, "steps": 100})
        time.sleep(0.1)  # Let it get queued
        
        # Cancel the job
        result = self.job_manager.cancel_job(job_id)
        self.assertTrue(result)
        
        time.sleep(0.2)  # Give time for persistence
        
        # Verify cancellation was persisted
        persistence = JobPersistence()
        metadata = persistence.load_job_status(job_id)
        self.assertIsNotNone(metadata)
        self.assertEqual(metadata.status, "cancelled")


class TestConvenienceFunctions(unittest.TestCase):
    """Test module-level convenience functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = Path(tempfile.mkdtemp())
        
        # Reset singleton instances
        JobPersistence._instance = None
        
        # Override persistence data directory
        persistence = JobPersistence()
        persistence.data_dir = self.test_dir / "jobs"
        persistence.active_dir = persistence.data_dir / "active"
        persistence.completed_dir = persistence.data_dir / "completed"
        persistence.history_dir = persistence.data_dir / "history"
        persistence._ensure_directories()
    
    def tearDown(self):
        """Clean up after each test."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        JobPersistence._instance = None
    
    def test_convenience_functions(self):
        """Test module-level convenience functions."""
        # Create test job
        job_info = JobInfo(
            id="convenience-test",
            job_type="test",
            params={"duration": 0.1}
        )
        job_info.status = JobStatus.COMPLETED
        job_info.completed_at = datetime.now()
        
        # Test save_job_status function
        success = save_job_status("convenience-test", job_info)
        self.assertTrue(success)
        
        # Test load_job_status function
        metadata = load_job_status("convenience-test")
        self.assertIsNotNone(metadata)
        self.assertEqual(metadata.id, "convenience-test")
        
        # Test get_job_history function
        history = get_job_history(limit=10)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].id, "convenience-test")
        
        # Test cleanup_old_jobs function (should not clean recent job)
        cleaned = cleanup_old_jobs(max_age_hours=1)
        self.assertEqual(cleaned, 0)
        
        # Test recover_interrupted_jobs function
        recovered = recover_interrupted_jobs()
        self.assertEqual(len(recovered), 0)  # No interrupted jobs


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)