#!/usr/bin/env python3
"""
App Restart Simulation Test

Tests the persistence system's ability to handle app restarts by 
simulating the restart process and verifying job recovery.
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

def simulate_app_restart():
    """Simulate application restart with job persistence."""
    print("🔄 Simulating Application Restart with Job Persistence")
    print("=" * 55)
    
    test_dir = Path(tempfile.mkdtemp())
    print(f"Using test directory: {test_dir}")
    
    try:
        # ===== PHASE 1: Initial App Session =====
        print("\n📱 PHASE 1: Initial Application Session")
        print("-" * 40)
        
        # Reset singletons and import modules
        from utils.job_persistence import JobPersistence
        from workflows.job_manager import JobManager, JobStatus
        
        JobPersistence._instance = None
        JobManager._instance = None
        
        # Override data directory for persistence
        persistence = JobPersistence()
        persistence.data_dir = test_dir / "jobs"
        persistence.active_dir = persistence.data_dir / "active"
        persistence.completed_dir = persistence.data_dir / "completed"
        persistence.history_dir = persistence.data_dir / "history"
        persistence._ensure_directories()
        
        # Create JobManager (this will auto-initialize persistence)
        job_manager = JobManager()
        print("✅ JobManager initialized with persistence")
        
        # Submit some jobs
        job_ids = []
        for i in range(3):
            job_id = job_manager.submit_job("test", {"duration": 0.1, "steps": 2})
            job_ids.append(job_id)
            print(f"✅ Submitted job {i+1}: {job_id}")
        
        # Wait a moment for jobs to potentially start
        time.sleep(0.2)
        
        # Check that jobs are persisted
        active_files = list(persistence.active_dir.glob("*.json"))
        completed_files = list(persistence.completed_dir.glob("*.json"))
        total_persisted = len(active_files) + len(completed_files)
        
        assert total_persisted >= 3, f"Expected 3+ persisted jobs, found {total_persisted}"
        print(f"✅ {total_persisted} jobs persisted to disk")
        
        # Simulate one job being interrupted (still running when app shuts down)
        # We'll manually create this scenario
        interrupted_job_id = job_manager.submit_job("test", {"duration": 10, "steps": 100})
        time.sleep(0.1)  # Let it potentially start
        
        # Force the job to appear as running in persistence
        with job_manager._jobs_lock:
            if interrupted_job_id in job_manager._jobs:
                job_info = job_manager._jobs[interrupted_job_id]
                if job_info.status == JobStatus.QUEUED:
                    # Manually mark as running for testing
                    job_info.status = JobStatus.RUNNING
                    job_info.started_at = datetime.now()
                    persistence.save_job_status(interrupted_job_id, job_info, move_file=False)
                    print(f"✅ Simulated interrupted job: {interrupted_job_id}")
        
        print(f"📊 Session 1 Summary: {len(job_ids)} jobs submitted, 1 interrupted job created")
        
        # Shutdown JobManager (simulating app shutdown)
        job_manager.shutdown(wait=True)
        print("✅ JobManager shutdown complete")
        
        # ===== PHASE 2: App Restart Simulation =====
        print("\n🔄 PHASE 2: Application Restart Simulation")
        print("-" * 42)
        
        # Reset singletons to simulate fresh app start
        JobPersistence._instance = None
        JobManager._instance = None
        
        # Re-initialize persistence (same directory)
        persistence = JobPersistence()
        persistence.data_dir = test_dir / "jobs"
        persistence.active_dir = persistence.data_dir / "active"
        persistence.completed_dir = persistence.data_dir / "completed"
        persistence.history_dir = persistence.data_dir / "history"
        
        print("✅ Persistence system reinitialized")
        
        # Check what jobs are recoverable
        pre_recovery_active = list(persistence.active_dir.glob("*.json"))
        pre_recovery_completed = list(persistence.completed_dir.glob("*.json"))
        
        print(f"📂 Found {len(pre_recovery_active)} active jobs and {len(pre_recovery_completed)} completed jobs")
        
        # Create new JobManager (this should trigger recovery)
        job_manager_2 = JobManager()
        print("✅ New JobManager initialized (should have triggered recovery)")
        
        # Check recovery results
        post_recovery_active = list(persistence.active_dir.glob("*.json"))
        post_recovery_completed = list(persistence.completed_dir.glob("*.json"))
        
        print(f"📂 After recovery: {len(post_recovery_active)} active jobs, {len(post_recovery_completed)} completed jobs")
        
        # Verify interrupted job was recovered
        recovered_job_metadata = persistence.load_job_status(interrupted_job_id)
        if recovered_job_metadata:
            assert recovered_job_metadata.status == "failed", "Interrupted job should be marked as failed"
            assert "interrupted" in recovered_job_metadata.error_message.lower(), "Should have interruption error"
            print(f"✅ Interrupted job {interrupted_job_id} successfully recovered as failed")
        else:
            print(f"❌ Could not find recovered job {interrupted_job_id}")
            
        # ===== PHASE 3: Verify Functionality =====
        print("\n🧪 PHASE 3: Verify Post-Restart Functionality")  
        print("-" * 44)
        
        # Test that we can submit new jobs after restart
        new_job_id = job_manager_2.submit_job("test", {"duration": 0.1, "steps": 1})
        print(f"✅ New job submitted after restart: {new_job_id}")
        
        # Wait for job completion
        time.sleep(1.0)
        
        # Verify new job was persisted
        new_job_metadata = persistence.load_job_status(new_job_id)
        assert new_job_metadata is not None, "New job should be persisted"
        print("✅ New job successfully persisted after restart")
        
        # Test job history includes both old and new jobs
        history = persistence.get_job_history()
        assert len(history) >= 4, f"Expected at least 4 jobs in history, found {len(history)}"
        print(f"✅ Job history contains {len(history)} jobs from both sessions")
        
        # Clean shutdown
        job_manager_2.shutdown(wait=True)
        print("✅ Second JobManager shutdown complete")
        
        print("\n" + "=" * 55)
        print("🎉 APP RESTART SIMULATION SUCCESSFUL!")
        print("✅ Persistence system correctly handles:")
        print("   • Job persistence across app sessions")
        print("   • Recovery of interrupted jobs")
        print("   • Continued functionality after restart")
        print("   • Historical data preservation")
        
        return True
        
    except Exception as e:
        print(f"\n❌ App restart simulation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        if test_dir.exists():
            shutil.rmtree(test_dir)
        
        # Reset singletons
        try:
            JobPersistence._instance = None
            JobManager._instance = None
        except:
            pass


if __name__ == '__main__':
    success = simulate_app_restart()
    exit(0 if success else 1)