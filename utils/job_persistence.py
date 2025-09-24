"""
Job Persistence Module for Mitochondria Analyzer

Provides persistent storage for job status tracking, recovery, and historical analysis.
Uses JSON-based file storage with atomic writes and observer pattern integration.
"""

import json
import os
import threading
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
import shutil
import tempfile


class PersistenceError(Exception):
    """Custom exception for persistence-related errors."""
    pass


@dataclass
class JobMetadata:
    """Enhanced job metadata for persistence."""
    id: str
    job_type: str
    status: str
    progress: float
    status_message: str
    error_message: str
    error_trace: str
    created_at: str
    started_at: Optional[str]
    completed_at: Optional[str]
    total_time: float
    input_files: List[str]
    output_files: List[str]
    parameters: Dict[str, Any]
    result: Any
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'JobMetadata':
        """Create from dictionary loaded from JSON."""
        # Handle missing fields for backward compatibility
        defaults = {
            'error_trace': '',
            'total_time': 0.0,
            'input_files': [],
            'output_files': [],
            'parameters': {},
            'result': None
        }
        
        for key, default_value in defaults.items():
            if key not in data:
                data[key] = default_value
        
        return cls(**data)


class JobPersistence:
    """
    Thread-safe job persistence manager with atomic operations.
    
    Provides storage, retrieval, cleanup, and recovery functionality for job data.
    Uses observer pattern for automatic persistence on status changes.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern implementation with thread safety."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the persistence manager (only once due to singleton pattern)."""
        if hasattr(self, '_initialized'):
            return
            
        self._initialized = True
        self.logger = self._setup_logging()
        
        # File system paths
        self.data_dir = Path("data/jobs")
        self.active_dir = self.data_dir / "active"
        self.completed_dir = self.data_dir / "completed"
        self.history_dir = self.data_dir / "history"
        
        # Thread safety
        self._file_lock = threading.Lock()
        self._cleanup_lock = threading.Lock()
        
        # Ensure directories exist
        self._ensure_directories()
        
        self.logger.info("JobPersistence initialized with data directory: %s", self.data_dir.absolute())
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging for persistence operations."""
        logger = logging.getLogger(f"{__name__}.JobPersistence")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _ensure_directories(self):
        """Ensure all required directories exist."""
        directories = [self.data_dir, self.active_dir, self.completed_dir, self.history_dir]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            self.logger.debug("Ensured directory exists: %s", directory)
    
    def _get_job_file_path(self, job_id: str, status: str) -> Path:
        """Get the appropriate file path based on job status."""
        filename = f"{job_id}.json"
        
        if status in ["queued", "running"]:
            return self.active_dir / filename
        elif status in ["completed", "failed", "cancelled"]:
            return self.completed_dir / filename
        else:
            # Default to active for unknown status
            return self.active_dir / filename
    
    def _atomic_write(self, filepath: Path, data: Dict[str, Any]):
        """
        Write JSON data atomically using temporary file.
        
        This prevents corruption from concurrent access or interrupted writes.
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Create temporary file in same directory for atomic move
        temp_fd, temp_path = tempfile.mkstemp(
            suffix='.tmp',
            prefix=f'{filepath.stem}_',
            dir=filepath.parent
        )
        
        try:
            with os.fdopen(temp_fd, 'w') as temp_file:
                json.dump(data, temp_file, indent=2, default=str, ensure_ascii=False)
            
            # Atomic move (POSIX systems)
            shutil.move(temp_path, filepath)
            self.logger.debug("Atomically wrote job data to: %s", filepath)
            
        except Exception as e:
            # Cleanup temp file if something went wrong
            try:
                os.unlink(temp_path)
            except:
                pass
            raise PersistenceError(f"Failed to write job data atomically: {e}")
    
    def save_job_status(self, job_id: str, job_info: Any, move_file: bool = True) -> bool:
        """
        Save job status to persistent storage with atomic write.
        
        Args:
            job_id: Unique job identifier
            job_info: JobInfo object from JobManager
            move_file: Whether to move file based on status change
            
        Returns:
            bool: Success status
        """
        try:
            with self._file_lock:
                # Convert JobInfo to JobMetadata for persistence
                metadata = self._job_info_to_metadata(job_info)
                
                # Determine target file path
                current_path = self._get_job_file_path(job_id, job_info.status.value)
                
                # If moving file, check for existing file in different location
                if move_file:
                    old_path = self._find_existing_job_file(job_id)
                    if old_path and old_path != current_path:
                        self.logger.debug("Moving job file from %s to %s", old_path, current_path)
                        # Remove old file after successful write
                        try:
                            old_path.unlink()
                        except FileNotFoundError:
                            pass
                
                # Atomic write to target location
                self._atomic_write(current_path, metadata.to_dict())
                
                self.logger.info("Saved job %s status: %s", job_id, job_info.status.value)
                return True
                
        except Exception as e:
            self.logger.error("Failed to save job %s: %s", job_id, e)
            return False
    
    def load_job_status(self, job_id: str) -> Optional[JobMetadata]:
        """
        Load job status from persistent storage with error handling.
        
        Args:
            job_id: Unique job identifier
            
        Returns:
            JobMetadata or None if not found/corrupted
        """
        try:
            job_file = self._find_existing_job_file(job_id)
            if not job_file:
                return None
            
            with open(job_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            metadata = JobMetadata.from_dict(data)
            self.logger.debug("Loaded job %s from: %s", job_id, job_file)
            return metadata
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            self.logger.error("Corrupted job file for %s: %s", job_id, e)
            # Move corrupted file for debugging
            self._quarantine_corrupted_file(job_file)
            return None
        except FileNotFoundError:
            return None
        except Exception as e:
            self.logger.error("Failed to load job %s: %s", job_id, e)
            return None
    
    def cleanup_old_jobs(self, max_age_hours: int = 168) -> int:  # Default 7 days
        """
        Clean up old job files based on age.
        
        Args:
            max_age_hours: Maximum age in hours before cleanup
            
        Returns:
            int: Number of files cleaned up
        """
        try:
            with self._cleanup_lock:
                cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
                cleaned_count = 0
                
                # Clean completed and history directories
                for directory in [self.completed_dir, self.history_dir]:
                    for job_file in directory.glob("*.json"):
                        try:
                            file_mtime = datetime.fromtimestamp(job_file.stat().st_mtime)
                            if file_mtime < cutoff_time:
                                # Move to history if in completed, delete if in history
                                if directory == self.completed_dir:
                                    self._move_to_history(job_file)
                                else:
                                    job_file.unlink()
                                cleaned_count += 1
                                self.logger.debug("Cleaned up old job file: %s", job_file)
                        except Exception as e:
                            self.logger.warning("Failed to cleanup %s: %s", job_file, e)
                
                self.logger.info("Cleaned up %d old job files", cleaned_count)
                return cleaned_count
                
        except Exception as e:
            self.logger.error("Cleanup operation failed: %s", e)
            return 0
    
    def get_job_history(self, limit: Optional[int] = None, 
                       status_filter: Optional[str] = None) -> List[JobMetadata]:
        """
        Get job history sorted by creation time (newest first).
        
        Args:
            limit: Maximum number of jobs to return
            status_filter: Filter by specific status
            
        Returns:
            List[JobMetadata]: Sorted job history
        """
        try:
            jobs = []
            
            # Scan all directories for job files
            for directory in [self.active_dir, self.completed_dir, self.history_dir]:
                for job_file in directory.glob("*.json"):
                    try:
                        metadata = self._load_job_metadata_from_file(job_file)
                        if metadata:
                            if status_filter is None or metadata.status == status_filter:
                                jobs.append(metadata)
                    except Exception as e:
                        self.logger.warning("Failed to load job from %s: %s", job_file, e)
            
            # Sort by creation time (newest first)
            jobs.sort(key=lambda job: job.created_at, reverse=True)
            
            # Apply limit if specified
            if limit:
                jobs = jobs[:limit]
            
            self.logger.debug("Retrieved %d jobs from history", len(jobs))
            return jobs
            
        except Exception as e:
            self.logger.error("Failed to get job history: %s", e)
            return []
    
    def recover_interrupted_jobs(self) -> List[str]:
        """
        Recover jobs that were interrupted by unexpected shutdown.
        
        Returns:
            List[str]: Job IDs that were recovered
        """
        try:
            recovered_jobs = []
            
            for job_file in self.active_dir.glob("*.json"):
                try:
                    metadata = self._load_job_metadata_from_file(job_file)
                    if metadata and metadata.status == "running":
                        # Mark as failed due to unexpected shutdown
                        metadata.status = "failed"
                        metadata.error_message = "Job interrupted by system shutdown"
                        metadata.completed_at = datetime.now().isoformat()
                        
                        # Save updated status and move to completed
                        completed_path = self.completed_dir / job_file.name
                        self._atomic_write(completed_path, metadata.to_dict())
                        job_file.unlink()
                        
                        recovered_jobs.append(metadata.id)
                        self.logger.info("Recovered interrupted job: %s", metadata.id)
                
                except Exception as e:
                    self.logger.error("Failed to recover job from %s: %s", job_file, e)
            
            if recovered_jobs:
                self.logger.info("Recovered %d interrupted jobs", len(recovered_jobs))
            
            return recovered_jobs
            
        except Exception as e:
            self.logger.error("Job recovery failed: %s", e)
            return []
    
    def _job_info_to_metadata(self, job_info: Any) -> JobMetadata:
        """Convert JobInfo object to JobMetadata for persistence."""
        # Calculate total time
        total_time = 0.0
        if job_info.started_at and job_info.completed_at:
            total_time = (job_info.completed_at - job_info.started_at).total_seconds()
        
        # Extract input/output files from parameters if available
        input_files = []
        output_files = []
        if hasattr(job_info, 'params') and job_info.params:
            # Look for common file parameter names
            for key in ['input_file', 'image_path', 'file_path']:
                if key in job_info.params:
                    input_files.append(str(job_info.params[key]))
            for key in ['output_file', 'output_path', 'result_path']:
                if key in job_info.params:
                    output_files.append(str(job_info.params[key]))
        
        # Get error trace if available
        error_trace = ""
        if hasattr(job_info, 'error_trace'):
            error_trace = job_info.error_trace
        elif job_info.error_message:
            error_trace = job_info.error_message
        
        return JobMetadata(
            id=job_info.id,
            job_type=job_info.job_type,
            status=job_info.status.value,
            progress=job_info.progress,
            status_message=job_info.status_message,
            error_message=job_info.error_message,
            error_trace=error_trace,
            created_at=job_info.created_at.isoformat(),
            started_at=job_info.started_at.isoformat() if job_info.started_at else None,
            completed_at=job_info.completed_at.isoformat() if job_info.completed_at else None,
            total_time=total_time,
            input_files=input_files,
            output_files=output_files,
            parameters=job_info.params or {},
            result=job_info.result
        )
    
    def _find_existing_job_file(self, job_id: str) -> Optional[Path]:
        """Find existing job file across all directories."""
        filename = f"{job_id}.json"
        
        # Check in order of likelihood
        for directory in [self.active_dir, self.completed_dir, self.history_dir]:
            job_file = directory / filename
            if job_file.exists():
                return job_file
        
        return None
    
    def _load_job_metadata_from_file(self, job_file: Path) -> Optional[JobMetadata]:
        """Load job metadata from a specific file."""
        try:
            with open(job_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return JobMetadata.from_dict(data)
        except Exception:
            return None
    
    def _move_to_history(self, job_file: Path):
        """Move completed job file to history directory."""
        history_file = self.history_dir / job_file.name
        shutil.move(str(job_file), str(history_file))
    
    def _quarantine_corrupted_file(self, job_file: Optional[Path]):
        """Move corrupted file to quarantine for debugging."""
        if not job_file:
            return
        
        try:
            quarantine_dir = self.data_dir / "quarantine"
            quarantine_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            quarantine_file = quarantine_dir / f"corrupted_{timestamp}_{job_file.name}"
            shutil.move(str(job_file), str(quarantine_file))
            
            self.logger.warning("Moved corrupted file to quarantine: %s", quarantine_file)
        except Exception as e:
            self.logger.error("Failed to quarantine corrupted file: %s", e)


# Observer interface for JobManager integration
class JobObserver:
    """Observer interface for job status changes."""
    
    def on_job_created(self, job_id: str, job_info: Any):
        """Called when a new job is created."""
        pass
    
    def on_job_status_changed(self, job_id: str, job_info: Any):
        """Called when job status changes."""
        pass
    
    def on_job_progress_updated(self, job_id: str, job_info: Any):
        """Called when job progress is updated."""
        pass


class PersistenceObserver(JobObserver):
    """Persistence observer that automatically saves job changes."""
    
    def __init__(self):
        self.persistence = JobPersistence()
    
    def on_job_created(self, job_id: str, job_info: Any):
        """Save new job to persistence."""
        self.persistence.save_job_status(job_id, job_info, move_file=False)
    
    def on_job_status_changed(self, job_id: str, job_info: Any):
        """Save job status change with file movement."""
        self.persistence.save_job_status(job_id, job_info, move_file=True)
    
    def on_job_progress_updated(self, job_id: str, job_info: Any):
        """Save progress updates (less frequent to avoid I/O overhead)."""
        # Only save progress at significant milestones to reduce I/O
        progress = job_info.progress
        if progress in [0, 25, 50, 75, 100] or progress % 10 == 0:
            self.persistence.save_job_status(job_id, job_info, move_file=False)


# Convenience functions for external use
def get_job_persistence() -> JobPersistence:
    """Get the singleton JobPersistence instance."""
    return JobPersistence()


def save_job_status(job_id: str, job_info: Any) -> bool:
    """Convenience function to save job status."""
    return get_job_persistence().save_job_status(job_id, job_info)


def load_job_status(job_id: str) -> Optional[JobMetadata]:
    """Convenience function to load job status."""
    return get_job_persistence().load_job_status(job_id)


def get_job_history(limit: Optional[int] = None, status_filter: Optional[str] = None) -> List[JobMetadata]:
    """Convenience function to get job history."""
    return get_job_persistence().get_job_history(limit, status_filter)


def cleanup_old_jobs(max_age_hours: int = 168) -> int:
    """Convenience function to cleanup old jobs."""
    return get_job_persistence().cleanup_old_jobs(max_age_hours)


def recover_interrupted_jobs() -> List[str]:
    """Convenience function to recover interrupted jobs."""
    return get_job_persistence().recover_interrupted_jobs()