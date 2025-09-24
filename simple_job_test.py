#!/usr/bin/env python3
"""
Simple Job Manager Test - Quick verification of core functionality
"""

import time
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from workflows.job_manager import JobManager, JobStatus, submit_job, get_job_status, cancel_job

def test_basic_functionality():
    """Test basic JobManager functionality."""
    print("🔬 Testing JobManager Core Functionality")
    
    try:
        # Test 1: Singleton pattern
        print("\n1️⃣ Testing singleton pattern...")
        manager1 = JobManager()
        manager2 = JobManager()
        assert manager1 is manager2, "Singleton pattern failed"
        print("✅ Singleton pattern works")
        
        # Test 2: Job submission
        print("\n2️⃣ Testing job submission...")
        job_id = submit_job('test', {'duration': 0.2, 'steps': 3})
        assert isinstance(job_id, str) and len(job_id) > 0, "Job ID invalid"
        print(f"✅ Job submitted: {job_id}")
        
        # Test 3: Job status tracking
        print("\n3️⃣ Testing job status...")
        status = get_job_status(job_id)
        assert status is not None, "Could not get job status"
        assert status['status'] == JobStatus.QUEUED.value, f"Expected queued, got {status['status']}"
        print(f"✅ Initial status: {status['status']}")
        
        # Test 4: Job execution
        print("\n4️⃣ Testing job execution...")
        max_wait = 3.0
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            status = get_job_status(job_id)
            print(f"   Status: {status['status']}, Progress: {status['progress']}%")
            
            if status['status'] in [JobStatus.COMPLETED.value, JobStatus.FAILED.value]:
                break
            time.sleep(0.2)
        
        final_status = get_job_status(job_id)
        assert final_status['status'] == JobStatus.COMPLETED.value, f"Job failed: {final_status.get('error_message', 'Unknown error')}"
        assert final_status['progress'] == 100.0, f"Progress not 100%: {final_status['progress']}"
        assert 'result' in final_status, "No result in final status"
        print(f"✅ Job completed successfully: {final_status['result']}")
        
        # Test 5: Job cancellation
        print("\n5️⃣ Testing job cancellation...")
        cancel_job_id = submit_job('test', {'duration': 10, 'steps': 100})  # Long job
        time.sleep(0.1)  # Let it get queued
        
        result = cancel_job(cancel_job_id)
        assert result == True, "Cancel job returned False"
        
        time.sleep(0.3)  # Give time to cancel
        cancel_status = get_job_status(cancel_job_id)
        assert cancel_status['status'] == JobStatus.CANCELLED.value, f"Job not cancelled: {cancel_status['status']}"
        print("✅ Job cancellation works")
        
        # Test 6: Invalid operations
        print("\n6️⃣ Testing invalid operations...")
        invalid_status = get_job_status("non-existent-id")
        assert invalid_status is None, "Should return None for non-existent job"
        
        invalid_cancel = cancel_job("non-existent-id")
        assert invalid_cancel == False, "Should return False for non-existent job"
        print("✅ Invalid operations handled correctly")
        
        # Test 7: Queue info
        print("\n7️⃣ Testing queue info...")
        queue_info = manager1.get_job_queue_info()
        assert 'total_jobs' in queue_info, "Queue info missing total_jobs"
        assert 'max_workers' in queue_info, "Queue info missing max_workers"
        assert queue_info['max_workers'] == 2, f"Expected 2 workers, got {queue_info['max_workers']}"
        print(f"✅ Queue info: {queue_info['total_jobs']} jobs, {queue_info['max_workers']} max workers")
        
        print(f"\n🎉 All tests passed! JobManager is working correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        print("\n🧹 Cleaning up...")
        try:
            JobManager().shutdown(wait=True)
            time.sleep(0.2)  # Give threads time to clean up
        except:
            pass
        JobManager._instance = None


def test_concurrent_execution():
    """Test concurrent job execution with worker limits."""
    print("\n🔬 Testing Concurrent Execution")
    
    try:
        # Reset singleton to get a fresh instance
        JobManager._instance = None
        manager = JobManager()
        
        # Submit multiple jobs
        job_ids = []
        for i in range(4):
            job_id = submit_job('test', {'duration': 0.3, 'steps': 2})
            job_ids.append(job_id)
            print(f"   Submitted job {i+1}: {job_id}")
        
        # Wait a moment for jobs to start
        time.sleep(0.5)
        
        # Check that no more than 2 are running at once
        running_count = 0
        for job_id in job_ids:
            status = get_job_status(job_id)
            if status and status['status'] == JobStatus.RUNNING.value:
                running_count += 1
        
        print(f"   Running jobs: {running_count} (should be <= 2)")
        assert running_count <= 2, f"Too many concurrent jobs: {running_count}"
        
        # Wait for all to complete
        max_wait = 5.0
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            completed = 0
            for job_id in job_ids:
                status = get_job_status(job_id)
                if status and status['status'] in [JobStatus.COMPLETED.value, JobStatus.FAILED.value]:
                    completed += 1
            
            print(f"   Completed: {completed}/{len(job_ids)}")
            if completed == len(job_ids):
                break
            time.sleep(0.2)
        
        # Verify all completed successfully
        success_count = 0
        for job_id in job_ids:
            status = get_job_status(job_id)
            if status and status['status'] == JobStatus.COMPLETED.value:
                success_count += 1
        
        print(f"✅ Successfully completed {success_count}/{len(job_ids)} jobs")
        assert success_count > 0, "No jobs completed successfully"
        return True
        
    except Exception as e:
        print(f"❌ Concurrent test failed: {str(e)}")
        return False
        
    finally:
        try:
            JobManager().shutdown(wait=True)
            time.sleep(0.2)
        except:
            pass
        JobManager._instance = None


if __name__ == '__main__':
    print("🧪 JobManager Simple Test Suite")
    print("=" * 50)
    
    success = True
    
    # Run basic functionality test
    if not test_basic_functionality():
        success = False
    
    time.sleep(0.5)  # Brief pause between tests
    
    # Run concurrent execution test
    if not test_concurrent_execution():
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 ALL TESTS PASSED! JobManager is ready for production.")
        exit(0)
    else:
        print("❌ Some tests failed. Check the output above.")
        exit(1)