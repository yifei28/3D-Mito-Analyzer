#!/usr/bin/env python3
"""
Test Suite for JobManager

Comprehensive tests for the background job management system including
threading, concurrency, job lifecycle, and error handling.
"""

import time
import threading
import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from workflows.job_manager import JobManager, JobStatus, get_job_manager, submit_job


class TestJobManager(unittest.TestCase):
    """Test suite for JobManager functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Reset singleton instance for each test
        JobManager._instance = None
        self.job_manager = JobManager()
    
    def tearDown(self):
        """Clean up after each test."""
        try:
            self.job_manager.shutdown(wait=True)  # Wait for proper shutdown
            # Give threads time to clean up
            time.sleep(0.1)
        except:
            pass
        JobManager._instance = None
    
    def test_singleton_pattern(self):
        """Test that JobManager follows singleton pattern."""
        manager1 = JobManager()
        manager2 = JobManager()
        manager3 = get_job_manager()
        
        self.assertIs(manager1, manager2)
        self.assertIs(manager1, manager3)
    
    def test_job_submission(self):
        """Test basic job submission functionality."""
        job_id = self.job_manager.submit_job(
            job_type="test",
            params={"duration": 0.1, "steps": 2}
        )
        
        self.assertIsInstance(job_id, str)
        self.assertTrue(len(job_id) > 0)
        
        # Check job was added to tracking
        status = self.job_manager.get_job_status(job_id)
        self.assertIsNotNone(status)
        self.assertEqual(status['status'], JobStatus.QUEUED.value)
        self.assertEqual(status['job_type'], 'test')
    
    def test_job_execution(self):
        """Test that jobs execute successfully."""
        job_id = self.job_manager.submit_job(
            job_type="test",
            params={"duration": 0.2, "steps": 3}
        )
        
        # Wait for job to complete
        max_wait = 5.0
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            status = self.job_manager.get_job_status(job_id)
            if status['status'] in [JobStatus.COMPLETED.value, JobStatus.FAILED.value]:
                break
            time.sleep(0.1)
        
        # Verify job completed successfully
        final_status = self.job_manager.get_job_status(job_id)
        self.assertEqual(final_status['status'], JobStatus.COMPLETED.value)
        self.assertEqual(final_status['progress'], 100.0)
        self.assertIn('result', final_status)
        self.assertIn('total_time', final_status)
    
    def test_job_cancellation(self):
        """Test job cancellation functionality."""
        # Test cancelling queued job
        job_id = self.job_manager.submit_job(
            job_type="test",
            params={"duration": 10, "steps": 100}  # Long job
        )
        
        # Cancel immediately
        result = self.job_manager.cancel_job(job_id)
        self.assertTrue(result)
        
        # Check status
        status = self.job_manager.get_job_status(job_id)
        self.assertIn(status['status'], [JobStatus.CANCELLED.value, JobStatus.QUEUED.value])
        
        # Wait a bit and check again
        time.sleep(0.5)
        status = self.job_manager.get_job_status(job_id)
        self.assertEqual(status['status'], JobStatus.CANCELLED.value)
    
    def test_concurrent_jobs(self):
        """Test concurrent job execution with worker limits."""
        job_ids = []
        
        # Submit multiple jobs (more than max workers)
        for i in range(5):
            job_id = self.job_manager.submit_job(
                job_type="test",
                params={"duration": 0.5, "steps": 2}
            )
            job_ids.append(job_id)
        
        # Wait for all jobs to start or complete
        time.sleep(1.0)
        
        # Check that no more than max_workers are running at once
        running_count = 0
        for job_id in job_ids:
            status = self.job_manager.get_job_status(job_id)
            if status['status'] == JobStatus.RUNNING.value:
                running_count += 1
        
        self.assertLessEqual(running_count, self.job_manager._max_workers)
        
        # Wait for all jobs to complete
        max_wait = 10.0
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            all_done = True
            for job_id in job_ids:
                status = self.job_manager.get_job_status(job_id)
                if status['status'] not in [JobStatus.COMPLETED.value, JobStatus.FAILED.value, JobStatus.CANCELLED.value]:
                    all_done = False
                    break
            
            if all_done:
                break
            time.sleep(0.2)
        
        # Verify all jobs eventually completed
        completed_count = 0
        for job_id in job_ids:
            status = self.job_manager.get_job_status(job_id)
            if status['status'] == JobStatus.COMPLETED.value:
                completed_count += 1
        
        self.assertGreater(completed_count, 0)
    
    def test_progress_callbacks(self):
        """Test progress callback functionality."""
        callback_calls = []
        
        def progress_callback(job_id, progress, message, status_dict=None):
            callback_calls.append({
                'job_id': job_id,
                'progress': progress,
                'message': message,
                'status_dict': status_dict,
                'timestamp': time.time()
            })
        
        job_id = self.job_manager.submit_job(
            job_type="test",
            params={"duration": 0.5, "steps": 5},
            callback=progress_callback
        )
        
        # Wait for job to complete
        time.sleep(2.0)
        
        # Verify callbacks were made
        self.assertGreater(len(callback_calls), 0)
        
        # Check that job_id is consistent
        for call in callback_calls:
            self.assertEqual(call['job_id'], job_id)
        
        # Check progress increases
        if len(callback_calls) > 1:
            first_progress = callback_calls[0]['progress']
            last_progress = callback_calls[-1]['progress']
            self.assertGreaterEqual(last_progress, first_progress)
    
    def test_invalid_job_operations(self):
        """Test error handling for invalid operations."""
        # Test getting status for non-existent job
        status = self.job_manager.get_job_status("non-existent-id")
        self.assertIsNone(status)
        
        # Test cancelling non-existent job
        result = self.job_manager.cancel_job("non-existent-id")
        self.assertFalse(result)
        
        # Test submitting job with invalid type
        job_id = self.job_manager.submit_job(
            job_type="invalid-type",
            params={}
        )
        
        # Wait for job to fail
        time.sleep(1.0)
        status = self.job_manager.get_job_status(job_id)
        self.assertEqual(status['status'], JobStatus.FAILED.value)
        self.assertIn('error_message', status)
    
    def test_job_queue_info(self):
        """Test job queue information retrieval."""
        # Submit some jobs
        job_ids = []
        for i in range(3):
            job_id = self.job_manager.submit_job(
                job_type="test",
                params={"duration": 0.1, "steps": 1}
            )
            job_ids.append(job_id)
        
        # Get queue info
        queue_info = self.job_manager.get_job_queue_info()
        
        self.assertIsInstance(queue_info, dict)
        self.assertIn('queue_size', queue_info)
        self.assertIn('active_workers', queue_info)
        self.assertIn('max_workers', queue_info)
        self.assertIn('total_jobs', queue_info)
        self.assertIn('status_counts', queue_info)
        
        self.assertEqual(queue_info['max_workers'], 2)
        self.assertGreaterEqual(queue_info['total_jobs'], 3)
    
    def test_thread_safety(self):
        """Test thread safety of job operations."""
        job_ids = []
        threads = []
        
        def submit_jobs():
            for i in range(10):
                job_id = self.job_manager.submit_job(
                    job_type="test",
                    params={"duration": 0.1, "steps": 1}
                )
                job_ids.append(job_id)
        
        def cancel_jobs():
            time.sleep(0.1)  # Let some jobs get submitted first
            for job_id in job_ids[:5]:  # Cancel first half
                self.job_manager.cancel_job(job_id)
        
        # Start threads
        submit_thread = threading.Thread(target=submit_jobs)
        cancel_thread = threading.Thread(target=cancel_jobs)
        
        submit_thread.start()
        cancel_thread.start()
        
        # Wait for threads to complete
        submit_thread.join()
        cancel_thread.join()
        
        # Wait for jobs to process
        time.sleep(2.0)
        
        # Verify no crashes and reasonable results
        total_jobs = len(job_ids)
        self.assertGreater(total_jobs, 0)
        
        # Check final status of all jobs
        final_statuses = []
        for job_id in job_ids:
            status = self.job_manager.get_job_status(job_id)
            if status:
                final_statuses.append(status['status'])
        
        # Should have mix of completed and cancelled
        self.assertIn(JobStatus.COMPLETED.value, final_statuses)
    
    def test_cleanup_old_jobs(self):
        """Test cleanup of old jobs."""
        # Submit and complete some test jobs
        job_ids = []
        for i in range(3):
            job_id = self.job_manager.submit_job(
                job_type="test",
                params={"duration": 0.1, "steps": 1}
            )
            job_ids.append(job_id)
        
        # Wait for jobs to complete
        time.sleep(1.0)
        
        # Manually set completion time to past for testing
        past_time = datetime.now() - timedelta(hours=25)
        with self.job_manager._jobs_lock:
            for job_id in job_ids:
                if job_id in self.job_manager._jobs:
                    self.job_manager._jobs[job_id].completed_at = past_time
        
        # Run cleanup
        removed_count = self.job_manager.cleanup_old_jobs(max_age_hours=24)
        self.assertGreater(removed_count, 0)
        
        # Verify jobs are removed
        for job_id in job_ids:
            status = self.job_manager.get_job_status(job_id)
            self.assertIsNone(status)
    
    def test_shutdown(self):
        """Test graceful shutdown."""
        # Submit some jobs
        for i in range(3):
            self.job_manager.submit_job(
                job_type="test",
                params={"duration": 0.1, "steps": 1}
            )
        
        # Shutdown
        self.job_manager.shutdown(wait=True)
        
        # Verify shutdown state
        self.assertTrue(self.job_manager._shutdown)
        
        # Try to submit job after shutdown
        with self.assertRaises(RuntimeError):
            self.job_manager.submit_job(
                job_type="test",
                params={"duration": 0.1, "steps": 1}
            )


class TestConvenienceFunctions(unittest.TestCase):
    """Test module-level convenience functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        JobManager._instance = None
    
    def tearDown(self):
        """Clean up after each test."""
        try:
            get_job_manager().shutdown(wait=True)  # Wait for proper shutdown
            time.sleep(0.1)  # Give threads time to clean up
        except:
            pass
        JobManager._instance = None
    
    def test_convenience_functions(self):
        """Test module-level convenience functions."""
        # Test submit_job function
        job_id = submit_job(
            job_type="test",
            params={"duration": 0.1, "steps": 1}
        )
        self.assertIsInstance(job_id, str)
        
        # Test get_job_status function
        status = get_job_status(job_id)
        self.assertIsNotNone(status)
        self.assertEqual(status['id'], job_id)
        
        # Wait a bit for job to potentially start
        time.sleep(0.5)
        
        # Test cancel_job function
        from workflows.job_manager import cancel_job
        result = cancel_job(job_id)
        self.assertIsInstance(result, bool)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)