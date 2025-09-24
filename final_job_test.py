#!/usr/bin/env python3
"""
Final JobManager Test - Comprehensive verification
"""

import time
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from workflows.job_manager import JobManager, JobStatus, submit_job, get_job_status, cancel_job

def run_comprehensive_test():
    """Run a comprehensive test of all JobManager features."""
    print("🧪 JobManager Comprehensive Test")
    print("=" * 50)
    
    test_results = []
    
    try:
        # Reset for clean test
        JobManager._instance = None
        
        # Test 1: Basic job execution
        print("\n1️⃣ Testing basic job execution...")
        job_id = submit_job('test', {'duration': 0.1, 'steps': 2})
        
        # Wait for completion
        start_time = time.time()
        while time.time() - start_time < 2.0:
            status = get_job_status(job_id)
            if status['status'] == JobStatus.COMPLETED.value:
                break
            time.sleep(0.1)
        
        final_status = get_job_status(job_id)
        success = final_status['status'] == JobStatus.COMPLETED.value
        test_results.append(("Basic execution", success))
        print(f"   Result: {'✅ PASS' if success else '❌ FAIL'}")
        
        # Test 2: Job cancellation
        print("\n2️⃣ Testing job cancellation...")
        cancel_job_id = submit_job('test', {'duration': 5, 'steps': 50})
        time.sleep(0.05)  # Let it start
        cancel_result = cancel_job(cancel_job_id)
        time.sleep(0.2)  # Give time to cancel
        
        cancel_status = get_job_status(cancel_job_id)
        success = cancel_status['status'] == JobStatus.CANCELLED.value
        test_results.append(("Job cancellation", success))
        print(f"   Result: {'✅ PASS' if success else '❌ FAIL'}")
        
        # Test 3: Multiple concurrent jobs
        print("\n3️⃣ Testing concurrent execution...")
        concurrent_jobs = []
        for i in range(3):
            job_id = submit_job('test', {'duration': 0.2, 'steps': 2})
            concurrent_jobs.append(job_id)
        
        # Check worker limit
        time.sleep(0.1)  # Let jobs start
        running_count = 0
        for job_id in concurrent_jobs:
            status = get_job_status(job_id)
            if status and status['status'] == JobStatus.RUNNING.value:
                running_count += 1
        
        # Wait for all to complete
        start_time = time.time()
        while time.time() - start_time < 3.0:
            all_done = True
            for job_id in concurrent_jobs:
                status = get_job_status(job_id)
                if status['status'] not in [JobStatus.COMPLETED.value, JobStatus.FAILED.value]:
                    all_done = False
                    break
            if all_done:
                break
            time.sleep(0.1)
        
        # Check results
        completed_count = sum(
            1 for job_id in concurrent_jobs 
            if get_job_status(job_id)['status'] == JobStatus.COMPLETED.value
        )
        
        success = (running_count <= 2) and (completed_count >= 2)
        test_results.append(("Concurrent execution", success))
        print(f"   Workers: {running_count} <= 2, Completed: {completed_count}/3")
        print(f"   Result: {'✅ PASS' if success else '❌ FAIL'}")
        
        # Test 4: Error handling
        print("\n4️⃣ Testing error handling...")
        invalid_status = get_job_status("nonexistent")
        invalid_cancel = cancel_job("nonexistent")
        
        success = (invalid_status is None) and (invalid_cancel == False)
        test_results.append(("Error handling", success))
        print(f"   Result: {'✅ PASS' if success else '❌ FAIL'}")
        
        # Test 5: Queue information
        print("\n5️⃣ Testing queue information...")
        manager = JobManager()
        queue_info = manager.get_job_queue_info()
        
        required_keys = ['total_jobs', 'active_workers', 'max_workers', 'status_counts']
        success = all(key in queue_info for key in required_keys)
        test_results.append(("Queue information", success))
        print(f"   Queue info keys: {list(queue_info.keys())}")
        print(f"   Result: {'✅ PASS' if success else '❌ FAIL'}")
        
        # Test 6: Progress callbacks
        print("\n6️⃣ Testing progress callbacks...")
        callback_calls = []
        
        def test_callback(job_id, progress, message, status_dict=None):
            callback_calls.append({'progress': progress, 'message': message})
        
        callback_job_id = submit_job('test', {'duration': 0.2, 'steps': 3}, callback=test_callback)
        
        # Wait for completion
        start_time = time.time()
        while time.time() - start_time < 2.0:
            status = get_job_status(callback_job_id)
            if status['status'] in [JobStatus.COMPLETED.value, JobStatus.FAILED.value]:
                break
            time.sleep(0.1)
        
        success = len(callback_calls) > 0
        test_results.append(("Progress callbacks", success))
        print(f"   Callback calls: {len(callback_calls)}")
        print(f"   Result: {'✅ PASS' if success else '❌ FAIL'}")
        
    except Exception as e:
        print(f"❌ Test execution failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        print("\n🧹 Cleaning up...")
        try:
            JobManager().shutdown(wait=True)
        except:
            pass
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, success in test_results if success)
    total = len(test_results)
    
    for test_name, success in test_results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {test_name:.<30} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! JobManager is fully functional and ready for production use.")
        return True
    else:
        print(f"\n⚠️ {total - passed} test(s) failed. Please review the implementation.")
        return False


if __name__ == '__main__':
    success = run_comprehensive_test()
    exit(0 if success else 1)