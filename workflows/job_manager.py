"""
Background Job Manager for Mitochondria Analyzer

Thread-based job management system for long-running tasks with GPU constraints.
Provides thread-safe job submission, status tracking, and progress callbacks.
"""

import os
import threading
import time
import uuid
import logging
import gc
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, Optional, Callable, List
from concurrent.futures import ThreadPoolExecutor
import traceback


class JobStatus(Enum):
    """Job status enumeration for state machine."""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobType(Enum):
    """Supported job types enumeration."""
    TEST = "test"
    SEGMENTATION = "segmentation"
    ANALYSIS = "analysis"  # For future analysis jobs

    @classmethod
    def is_valid(cls, job_type: str) -> bool:
        """Check if a job type string is valid."""
        return job_type in cls._value2member_map_


@dataclass
class JobInfo:
    """Data class for job information and tracking."""
    id: str
    job_type: str
    params: Dict[str, Any]
    callback: Optional[Callable] = None
    status: JobStatus = JobStatus.QUEUED
    progress: float = 0.0
    status_message: str = ""
    error_message: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Any = None
    cancelled_event: threading.Event = field(default_factory=threading.Event)

    # Enhanced progress tracking fields
    current_stage: Optional[str] = None
    stage_progress: float = 0.0
    stage_metadata: Dict[str, Any] = field(default_factory=dict)
    hardware_mode: str = "Unknown"
    estimated_completion: Optional[datetime] = None

    # GPU queue management fields
    gpu_requested: bool = False
    gpu_allocated: bool = False
    gpu_queue_position: int = 0
    gpu_timeout_at: Optional[datetime] = None


class GPUResourceManager:
    """
    GPU Resource Queue Management System.

    Manages GPU resource allocation with concurrent job limits, queuing, and timeout handling.
    Ensures maximum 1 GPU job runs at a time with automatic TensorFlow cleanup.
    """

    def __init__(self, max_gpu_jobs: int = 1, default_timeout_seconds: int = 300):
        """
        Initialize GPU resource manager.

        Args:
            max_gpu_jobs: Maximum concurrent GPU jobs (default: 1)
            default_timeout_seconds: Default timeout for GPU jobs (default: 5 minutes)
        """
        self.max_gpu_jobs = max_gpu_jobs
        self.default_timeout_seconds = default_timeout_seconds

        # Logger (initialize first)
        self.logger = logging.getLogger(f"{__name__}.GPUResourceManager")

        # GPU job tracking
        self._active_gpu_jobs: Dict[str, datetime] = {}  # job_id -> start_time
        self._gpu_queue: deque = deque()  # jobs waiting for GPU
        self._gpu_lock = threading.Lock()

        # GPU availability detection
        self._gpu_available = self._detect_gpu_availability()

        # Timeout monitoring
        self._timeout_thread = threading.Thread(
            target=self._monitor_timeouts,
            name="GPUTimeoutMonitor",
            daemon=True
        )
        self._timeout_thread.start()

    def _detect_gpu_availability(self) -> bool:
        """Detect if GPU is available on the system."""
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            available = len(gpus) > 0
            self.logger.info(f"GPU detection: {'Available' if available else 'Not available'}")
            return available
        except ImportError:
            self.logger.info("TensorFlow not available - GPU disabled")
            return False
        except Exception as e:
            self.logger.warning(f"GPU detection failed: {e}")
            return False

    def request_gpu(self, job_id: str, timeout_seconds: Optional[int] = None) -> tuple[bool, int]:
        """
        Request GPU resource for a job.

        Args:
            job_id: Job ID requesting GPU
            timeout_seconds: Timeout in seconds (uses default if None)

        Returns:
            tuple: (allocated: bool, queue_position: int)
                - allocated: True if GPU allocated immediately, False if queued
                - queue_position: 0 if allocated, >0 if queued (position in queue)
        """
        if not self._gpu_available:
            self.logger.debug(f"GPU not available for job {job_id}")
            return False, 0

        timeout = timeout_seconds or self.default_timeout_seconds
        timeout_at = datetime.now() + timedelta(seconds=timeout)

        with self._gpu_lock:
            # Check if GPU can be allocated immediately
            if len(self._active_gpu_jobs) < self.max_gpu_jobs:
                self._active_gpu_jobs[job_id] = datetime.now()
                self.logger.info(f"GPU allocated immediately to job {job_id}")
                return True, 0
            else:
                # Add to queue
                queue_entry = {
                    'job_id': job_id,
                    'timeout_at': timeout_at,
                    'requested_at': datetime.now()
                }
                self._gpu_queue.append(queue_entry)
                queue_position = len(self._gpu_queue)
                self.logger.info(f"Job {job_id} queued for GPU at position {queue_position}")
                return False, queue_position

    def release_gpu(self, job_id: str) -> None:
        """
        Release GPU resource and process next job in queue.

        Args:
            job_id: Job ID releasing GPU
        """
        with self._gpu_lock:
            if job_id in self._active_gpu_jobs:
                # Release the GPU
                start_time = self._active_gpu_jobs.pop(job_id)
                duration = datetime.now() - start_time
                self.logger.info(f"GPU released by job {job_id} after {duration}")

                # Clean up TensorFlow resources
                self._cleanup_tensorflow_session(job_id)

                # Process next job in queue
                self._process_gpu_queue()
            else:
                self.logger.warning(f"Attempted to release GPU for job {job_id} that doesn't have it")

    def get_queue_position(self, job_id: str) -> int:
        """
        Get current queue position for a job.

        Args:
            job_id: Job ID to check

        Returns:
            int: 0 if job has GPU, >0 if queued, -1 if not found
        """
        with self._gpu_lock:
            # Check if job has GPU
            if job_id in self._active_gpu_jobs:
                return 0

            # Check queue position
            for i, entry in enumerate(self._gpu_queue):
                if entry['job_id'] == job_id:
                    return i + 1

            return -1  # Not found

    def get_gpu_status(self) -> Dict[str, Any]:
        """Get current GPU resource status."""
        with self._gpu_lock:
            return {
                'gpu_available': self._gpu_available,
                'max_gpu_jobs': self.max_gpu_jobs,
                'active_gpu_jobs': len(self._active_gpu_jobs),
                'queue_length': len(self._gpu_queue),
                'active_job_ids': list(self._active_gpu_jobs.keys()),
                'queued_job_ids': [entry['job_id'] for entry in self._gpu_queue]
            }

    def cancel_gpu_request(self, job_id: str) -> bool:
        """
        Cancel GPU request for a job.

        Args:
            job_id: Job ID to cancel

        Returns:
            bool: True if cancelled successfully
        """
        with self._gpu_lock:
            # Check if job is in queue
            for i, entry in enumerate(self._gpu_queue):
                if entry['job_id'] == job_id:
                    del self._gpu_queue[i]
                    self.logger.info(f"GPU request cancelled for job {job_id}")
                    return True

            # Check if job has GPU (shouldn't happen during normal cancellation)
            if job_id in self._active_gpu_jobs:
                self.release_gpu(job_id)
                return True

            return False

    def _process_gpu_queue(self) -> None:
        """Process the next job in the GPU queue (called with lock held)."""
        while self._gpu_queue and len(self._active_gpu_jobs) < self.max_gpu_jobs:
            entry = self._gpu_queue.popleft()
            job_id = entry['job_id']

            # Check if job has timed out
            if datetime.now() > entry['timeout_at']:
                self.logger.warning(f"GPU request timed out for job {job_id}")
                continue

            # Allocate GPU to this job
            self._active_gpu_jobs[job_id] = datetime.now()
            self.logger.info(f"GPU allocated from queue to job {job_id}")

            # Notify job manager that GPU is now available for this job
            # This will be handled through the queue position monitoring
            break

    def _monitor_timeouts(self) -> None:
        """Background thread to monitor job timeouts."""
        while True:
            try:
                current_time = datetime.now()

                with self._gpu_lock:
                    # Check queue timeouts
                    expired_jobs = []
                    for entry in list(self._gpu_queue):
                        if current_time > entry['timeout_at']:
                            expired_jobs.append(entry['job_id'])

                    # Remove expired jobs from queue
                    for job_id in expired_jobs:
                        self._gpu_queue = deque([
                            entry for entry in self._gpu_queue
                            if entry['job_id'] != job_id
                        ])
                        self.logger.warning(f"GPU request timeout for queued job {job_id}")

                # Sleep for 5 seconds before next check
                time.sleep(5)

            except Exception as e:
                self.logger.error(f"GPU timeout monitoring error: {e}")
                time.sleep(10)  # Longer sleep on error

    def _cleanup_tensorflow_session(self, job_id: str) -> None:
        """
        Comprehensive TensorFlow session cleanup.

        Args:
            job_id: Job ID for logging purposes
        """
        try:
            import tensorflow as tf

            # Clear Keras session
            tf.keras.backend.clear_session()

            # Reset default graph (for TF 1.x compatibility)
            if hasattr(tf.compat.v1, 'reset_default_graph'):
                tf.compat.v1.reset_default_graph()

            # GPU memory cleanup
            try:
                gpus = tf.config.list_physical_devices('GPU')
                for gpu in gpus:
                    # Reset memory growth (this may not always work)
                    try:
                        tf.config.experimental.reset_memory_growth(gpu)
                    except Exception:
                        pass  # Some operations may not be supported
            except Exception:
                pass

            # Force garbage collection
            gc.collect()

            self.logger.debug(f"TensorFlow cleanup completed for job {job_id}")

        except ImportError:
            self.logger.debug(f"TensorFlow not available for cleanup (job {job_id})")
        except Exception as e:
            self.logger.warning(f"TensorFlow cleanup failed for job {job_id}: {e}")


class JobManager:
    """
    Thread-based job management system with singleton pattern.
    
    Manages a queue of background jobs with limited worker threads for GPU constraints.
    Provides thread-safe operations for job submission, status tracking, and cancellation.
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
        """Initialize the job manager (only once due to singleton pattern)."""
        if hasattr(self, '_initialized') and not self._shutdown:
            return
            
        self._initialized = True
        self.logger = self._setup_logging()
        
        # Job tracking and queue
        self._jobs: Dict[str, JobInfo] = {}
        self._job_queue: deque = deque()
        self._jobs_lock = threading.Lock()
        
        # Worker thread pool (max 2 for GPU constraint)
        self._max_workers = 2
        self._executor = ThreadPoolExecutor(max_workers=self._max_workers, thread_name_prefix="JobWorker")
        self._active_jobs = set()
        self._active_jobs_lock = threading.Lock()
        
        # Observer pattern for persistence
        self._observers = []
        self._observers_lock = threading.Lock()
        
        # Initialize persistence observer
        try:
            from utils.job_persistence import PersistenceObserver, recover_interrupted_jobs
            persistence_observer = PersistenceObserver()
            self._observers.append(persistence_observer)
            self.logger.info("Persistence observer registered")
            
            # Recovery interrupted jobs from previous session
            recovered_jobs = recover_interrupted_jobs()
            if recovered_jobs:
                self.logger.info("Recovered %d interrupted jobs from previous session", len(recovered_jobs))
        except ImportError as e:
            self.logger.warning("Could not initialize persistence: %s", e)

        # Initialize GPU resource manager
        self._gpu_manager = GPUResourceManager(max_gpu_jobs=1, default_timeout_seconds=300)
        self.logger.info("GPU resource manager initialized")

        # Shutdown flag
        self._shutdown = False
        self._shutdown_event = threading.Event()
        
        # Start the job dispatcher
        self._dispatcher_thread = threading.Thread(
            target=self._job_dispatcher,
            name="JobDispatcher",
            daemon=True
        )
        self._dispatcher_thread.start()
        
        self.logger.info("JobManager initialized with %d max workers", self._max_workers)
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration for the job manager."""
        logger = logging.getLogger(f"{__name__}.JobManager")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def submit_job(self, job_type: str, params: Dict[str, Any],
                   callback: Optional[Callable] = None) -> str:
        """
        Submit a new job to the queue.

        Args:
            job_type: Type of job to execute
            params: Job parameters dictionary
            callback: Optional callback function for progress updates

        Returns:
            Unique job ID string

        Raises:
            ValueError: If job_type is invalid or params are invalid
            RuntimeError: If JobManager is shutting down
        """
        if self._shutdown:
            raise RuntimeError("JobManager is shutting down")

        # Validate job type
        if not JobType.is_valid(job_type):
            valid_types = [jt.value for jt in JobType]
            raise ValueError(f"Invalid job_type '{job_type}'. Supported types: {valid_types}")

        # Validate parameters based on job type
        validated_params = self._validate_job_parameters(job_type, params)

        job_id = str(uuid.uuid4())
        job_info = JobInfo(
            id=job_id,
            job_type=job_type,
            params=validated_params,
            callback=callback
        )

        with self._jobs_lock:
            self._jobs[job_id] = job_info
            self._job_queue.append(job_id)

        # Notify observers about job creation
        self._notify_observers('created', job_id, job_info)

        self.logger.info("Job submitted: %s (type: %s)", job_id, job_type)
        return job_id

    def _validate_job_parameters(self, job_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate job parameters based on job type.

        Args:
            job_type: Type of job
            params: Job parameters to validate

        Returns:
            Validated and normalized parameters

        Raises:
            ValueError: If parameters are invalid
        """
        if job_type == JobType.SEGMENTATION.value:
            return self._validate_segmentation_parameters(params)
        elif job_type == JobType.TEST.value:
            return self._validate_test_parameters(params)
        elif job_type == JobType.ANALYSIS.value:
            return self._validate_analysis_parameters(params)
        else:
            # Return params as-is for unknown types (backward compatibility)
            return params.copy()

    def _validate_segmentation_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate segmentation job parameters.

        Args:
            params: Raw parameters

        Returns:
            Validated parameters with defaults applied

        Raises:
            ValueError: If required parameters are missing or invalid
        """
        # Required parameters
        input_path = params.get('input_path')
        if not input_path:
            raise ValueError("Segmentation job requires 'input_path' parameter")

        if not os.path.exists(input_path):
            raise ValueError(f"Input file does not exist: {input_path}")

        if not input_path.lower().endswith(('.tif', '.tiff')):
            raise ValueError("Input file must be a TIFF image (.tif or .tiff)")

        # Validate file is readable
        try:
            with open(input_path, 'rb') as f:
                # Try to read first few bytes to ensure file is accessible
                f.read(10)
        except (OSError, IOError) as e:
            raise ValueError(f"Input file is not readable: {e}")

        # Output directory validation
        output_dir = params.get('output_dir', 'data/segmented')
        try:
            os.makedirs(output_dir, exist_ok=True)
        except (OSError, IOError) as e:
            raise ValueError(f"Cannot create output directory '{output_dir}': {e}")

        if not os.access(output_dir, os.W_OK):
            raise ValueError(f"Output directory is not writable: {output_dir}")

        # Optional parameters with validation
        validated_params = {
            'input_path': input_path,
            'output_dir': output_dir,
            'force_cpu': bool(params.get('force_cpu', False)),
            'auto_chain_analysis': bool(params.get('auto_chain_analysis', False))
        }

        # Batch size validation
        if 'batch_size' in params:
            batch_size = params['batch_size']
            if not isinstance(batch_size, int) or batch_size < 1 or batch_size > 32:
                raise ValueError("batch_size must be an integer between 1 and 32")
            validated_params['batch_size'] = batch_size

        # Memory limit validation
        if 'memory_limit_gb' in params:
            memory_limit = params['memory_limit_gb']
            if not isinstance(memory_limit, (int, float)) or memory_limit <= 0 or memory_limit > 128:
                raise ValueError("memory_limit_gb must be a positive number <= 128")
            validated_params['memory_limit_gb'] = float(memory_limit)

        # Timeout validation
        if 'timeout_minutes' in params:
            timeout = params['timeout_minutes']
            if not isinstance(timeout, (int, float)) or timeout <= 0 or timeout > 120:
                raise ValueError("timeout_minutes must be a positive number <= 120")
            validated_params['timeout_minutes'] = float(timeout)
        else:
            validated_params['timeout_minutes'] = 45.0  # Default 45 minute timeout

        return validated_params

    def _validate_test_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate test job parameters."""
        validated_params = {
            'duration': params.get('duration', 5),
            'steps': params.get('steps', 10)
        }

        # Validate duration
        if not isinstance(validated_params['duration'], (int, float)) or validated_params['duration'] <= 0:
            raise ValueError("duration must be a positive number")

        # Validate steps
        if not isinstance(validated_params['steps'], int) or validated_params['steps'] <= 0:
            raise ValueError("steps must be a positive integer")

        return validated_params

    def _validate_analysis_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate analysis job parameters (placeholder for future implementation)."""
        # TODO: Implement analysis job parameter validation
        return params.copy()
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get current status of a job.
        
        Args:
            job_id: Job ID to query
            
        Returns:
            Dictionary with job status information or None if job not found
        """
        with self._jobs_lock:
            job_info = self._jobs.get(job_id)
            if job_info is None:
                return None
            
            status_dict = {
                'id': job_info.id,
                'job_type': job_info.job_type,
                'status': job_info.status.value,
                'progress': job_info.progress,
                'status_message': job_info.status_message,
                'created_at': job_info.created_at.isoformat(),
                'result': job_info.result
            }
            
            if job_info.error_message:
                status_dict['error_message'] = job_info.error_message
            
            if job_info.started_at:
                status_dict['started_at'] = job_info.started_at.isoformat()
                if job_info.status == JobStatus.RUNNING:
                    status_dict['elapsed_time'] = (datetime.now() - job_info.started_at).total_seconds()
            
            if job_info.completed_at:
                status_dict['completed_at'] = job_info.completed_at.isoformat()
                if job_info.started_at:
                    status_dict['total_time'] = (job_info.completed_at - job_info.started_at).total_seconds()
                else:
                    status_dict['total_time'] = 0.0
            
            return status_dict

    def get_enhanced_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get enhanced status of a job including stage-aware progress information.

        Args:
            job_id: Job ID to query

        Returns:
            Dictionary with enhanced job status information or None if job not found
        """
        with self._jobs_lock:
            job_info = self._jobs.get(job_id)
            if job_info is None:
                return None

            # Get basic status
            status_dict = {
                'id': job_info.id,
                'job_type': job_info.job_type,
                'status': job_info.status.value,
                'progress': job_info.progress,
                'status_message': job_info.status_message,
                'created_at': job_info.created_at.isoformat(),
                'result': job_info.result,

                # Enhanced progress information
                'current_stage': job_info.current_stage,
                'stage_progress': job_info.stage_progress,
                'stage_metadata': job_info.stage_metadata,
                'hardware_mode': job_info.hardware_mode,
                'params': job_info.params
            }

            if job_info.error_message:
                status_dict['error_message'] = job_info.error_message

            if job_info.started_at:
                status_dict['started_at'] = job_info.started_at.isoformat()
                if job_info.status == JobStatus.RUNNING:
                    status_dict['elapsed_time'] = (datetime.now() - job_info.started_at).total_seconds()

            if job_info.completed_at:
                status_dict['completed_at'] = job_info.completed_at.isoformat()
                if job_info.started_at:
                    status_dict['total_time'] = (job_info.completed_at - job_info.started_at).total_seconds()
                else:
                    status_dict['total_time'] = 0.0

            if job_info.estimated_completion:
                status_dict['estimated_completion'] = job_info.estimated_completion.isoformat()
                status_dict['eta_seconds'] = (job_info.estimated_completion - datetime.now()).total_seconds()

            return status_dict

    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a job by ID.
        
        Args:
            job_id: Job ID to cancel
            
        Returns:
            True if job was cancelled, False if job not found or already completed
        """
        with self._jobs_lock:
            job_info = self._jobs.get(job_id)
            if job_info is None:
                return False
            
            if job_info.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                return False
            
            # Set cancellation event
            job_info.cancelled_event.set()
            
            if job_info.status == JobStatus.QUEUED:
                # Remove from queue if not started
                try:
                    queue_list = list(self._job_queue)
                    if job_id in queue_list:
                        queue_list.remove(job_id)
                        self._job_queue = deque(queue_list)
                except ValueError:
                    pass  # Job might have been picked up by dispatcher
                
                # Update using observer-aware method
                self._update_job_status(job_id, JobStatus.CANCELLED, "Cancelled before execution")
                
            elif job_info.status == JobStatus.RUNNING:
                # Job is running, it will check cancelled_event
                # Don't change status yet, just update message
                with self._jobs_lock:
                    job_info.status_message = "Cancellation requested"
        
        self.logger.info("Job cancellation requested: %s", job_id)
        return True
    
    def _job_dispatcher(self):
        """Main dispatcher loop that assigns jobs to worker threads."""
        self.logger.info("Job dispatcher started")
        
        while not self._shutdown_event.is_set():
            try:
                # Check if we have available worker capacity
                with self._active_jobs_lock:
                    if len(self._active_jobs) >= self._max_workers:
                        time.sleep(0.1)
                        continue
                
                # Get next job from queue
                job_id = None
                with self._jobs_lock:
                    if self._job_queue:
                        job_id = self._job_queue.popleft()
                
                if job_id is None:
                    time.sleep(0.1)
                    continue
                
                # Check if job is still valid and not cancelled
                with self._jobs_lock:
                    job_info = self._jobs.get(job_id)
                    if job_info is None or job_info.status != JobStatus.QUEUED:
                        continue
                
                # Submit to thread pool
                with self._active_jobs_lock:
                    self._active_jobs.add(job_id)
                
                future = self._executor.submit(self._execute_job, job_id)
                future.add_done_callback(lambda f, jid=job_id: self._job_completed_callback(jid))
                
            except Exception as e:
                self.logger.error("Error in job dispatcher: %s", str(e))
                time.sleep(1)
        
        self.logger.info("Job dispatcher stopped")
    
    def _execute_job(self, job_id: str):
        """
        Execute a single job in a worker thread.
        
        Args:
            job_id: Job ID to execute
        """
        with self._jobs_lock:
            job_info = self._jobs.get(job_id)
            if job_info is None:
                return
        
        # Update status using observer-aware method
        self._update_job_status(job_id, JobStatus.RUNNING, "Starting execution")
        
        self.logger.info("Starting job execution: %s", job_id)
        
        try:
            # Execute the job based on job_type
            result = self._run_job_implementation(job_info)
            
            # Check for cancellation one more time
            if job_info.cancelled_event.is_set():
                self._update_job_status(job_id, JobStatus.CANCELLED, "Cancelled during execution")
                self.logger.info("Job cancelled during execution: %s", job_id)
                return
            
            # Job completed successfully
            with self._jobs_lock:
                job_info.result = result
                job_info.progress = 100.0
            self._update_job_status(job_id, JobStatus.COMPLETED, "Completed successfully", result=result)
            
            self.logger.info("Job completed successfully: %s", job_id)
            
        except InterruptedError as e:
            # Job was cancelled
            self.logger.info("Job cancelled during execution: %s", job_id)
            # Status was already set by check_cancelled()
            
        except Exception as e:
            # Job failed
            error_msg = str(e)
            self.logger.error("Job failed: %s - %s", job_id, error_msg)
            
            self._update_job_status(job_id, JobStatus.FAILED, f"Failed: {error_msg}", error_message=error_msg)
    
    def _run_job_implementation(self, job_info: JobInfo) -> Any:
        """
        Run the actual job implementation based on job type.
        
        Args:
            job_info: Job information object
            
        Returns:
            Job result
            
        Raises:
            NotImplementedError: If job_type is not supported
            Exception: Any exception from job execution
        """
        job_type = job_info.job_type
        params = job_info.params
        
        # Progress update helper
        def update_progress(progress: float, message: str = ""):
            with self._jobs_lock:
                job_info.progress = progress
                job_info.status_message = message
            
            # Notify observers about progress update
            self._notify_observers('progress_updated', job_info.id, job_info)
            
            if job_info.callback:
                try:
                    job_info.callback(job_info.id, progress, message)
                except Exception as e:
                    self.logger.warning("Progress callback failed for job %s: %s", job_info.id, str(e))
        
        # Check for cancellation periodically
        def check_cancelled():
            if job_info.cancelled_event.is_set():
                self._update_job_status(job_info.id, JobStatus.CANCELLED, "Cancelled during execution")
                raise InterruptedError("Job was cancelled")
        
        # Job type implementations
        if job_type == "test":
            # Test job implementation
            update_progress(10, "Starting test job")
            check_cancelled()
            
            duration = params.get('duration', 5)
            steps = params.get('steps', 10)
            
            for i in range(steps):
                time.sleep(duration / steps)
                check_cancelled()
                progress = (i + 1) / steps * 90 + 10
                update_progress(progress, f"Test step {i+1}/{steps}")
            
            return {"message": "Test job completed", "duration": duration}
        
        elif job_type == "segmentation":
            # Enhanced segmentation job implementation (Task 11.2)
            return self._execute_segmentation_job(job_info, update_progress, check_cancelled)
        
        else:
            raise NotImplementedError(f"Job type '{job_type}' is not implemented")

    def _execute_segmentation_job(self, job_info: JobInfo,
                                update_progress: Callable,
                                check_cancelled: Callable) -> Dict[str, Any]:
        """
        Execute segmentation job with enhanced parameter support and resource management.

        This method implements Task 11.2: Comprehensive segmentation job executor that:
        - Handles all validated parameters from Task 11.1
        - Implements GPU resource management and timeout handling
        - Provides enhanced error handling and logging
        - Includes TensorFlow resource cleanup
        - Collects comprehensive result metadata

        Args:
            job_info: Job information with validated parameters
            update_progress: Progress update callback function
            check_cancelled: Cancellation check callback function

        Returns:
            Comprehensive job results with metadata

        Raises:
            ValueError: If required parameters are missing or invalid
            TimeoutError: If job exceeds specified timeout
            RuntimeError: If segmentation processing fails
        """
        from .segmentation import create_segmentation_workflow

        params = job_info.params
        start_time = time.time()

        # Extract and validate all parameters
        input_path = params.get('input_path')
        output_dir = params.get('output_dir', 'data/segmented')
        force_cpu = params.get('force_cpu', False)
        batch_size = params.get('batch_size')
        memory_limit_gb = params.get('memory_limit_gb')
        timeout_minutes = params.get('timeout_minutes', 45.0)
        auto_chain_analysis = params.get('auto_chain_analysis', False)

        # Validation
        if not input_path:
            raise ValueError("input_path parameter required for segmentation job")

        # Setup timeout monitoring
        timeout_seconds = timeout_minutes * 60

        def timeout_check():
            if time.time() - start_time > timeout_seconds:
                raise TimeoutError(f"Job exceeded timeout of {timeout_minutes} minutes")

        self.logger.info(f"🚀 Starting enhanced segmentation job: {job_info.id}")
        self.logger.info(f"Parameters: input={input_path}, output={output_dir}, force_cpu={force_cpu}")
        if batch_size:
            self.logger.info(f"Batch size: {batch_size}")
        if memory_limit_gb:
            self.logger.info(f"Memory limit: {memory_limit_gb}GB")
        self.logger.info(f"Timeout: {timeout_minutes} minutes")

        update_progress(5, "Initializing segmentation workflow")

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Enhanced GPU resource management with queuing (Task 11.4)
        gpu_allocated = False
        queue_position = 0

        try:
            # Check GPU availability and allocate/queue if needed
            if not force_cpu:
                gpu_allocated, queue_position = self._gpu_manager.request_gpu(
                    job_info.id,
                    timeout_seconds=int(timeout_seconds)
                )

                # Update job info with GPU status
                with self._jobs_lock:
                    job_info.gpu_requested = True
                    job_info.gpu_allocated = gpu_allocated
                    job_info.gpu_queue_position = queue_position
                    if queue_position > 0:
                        job_info.hardware_mode = "Queued"
                        timeout_at = datetime.now() + timedelta(seconds=timeout_seconds)
                        job_info.gpu_timeout_at = timeout_at

                if gpu_allocated:
                    job_info.hardware_mode = "GPU"
                    self.logger.info(f"GPU allocated immediately for job {job_info.id}")
                    update_progress(8, "GPU allocated - starting processing")
                elif queue_position > 0:
                    self.logger.info(f"Job {job_info.id} queued for GPU at position {queue_position}")
                    update_progress(5, f"Queued for GPU (position {queue_position})")

                    # Wait for GPU allocation
                    self._wait_for_gpu_allocation(job_info, update_progress, check_cancelled, timeout_check)
                    gpu_allocated = job_info.gpu_allocated
                else:
                    self.logger.warning(f"GPU not available for job {job_info.id}, falling back to CPU")
                    force_cpu = True
                    job_info.hardware_mode = "CPU"

            update_progress(10, "Creating segmentation workflow")

            # Create segmentation workflow with enhanced parameters
            workflow = create_segmentation_workflow(
                model_path=None,  # Auto-detect
                force_cpu=force_cpu,
                memory_limit_gb=memory_limit_gb
            )

            # Configure batch size if provided
            if batch_size and hasattr(workflow, 'batch_size'):
                workflow.batch_size = batch_size
                self.logger.info(f"Workflow batch size set to: {batch_size}")

            update_progress(15, "Workflow configuration complete")

            # Enhanced progress callback with stage-aware tracking
            def enhanced_progress_callback(percentage: int, info: dict):
                try:
                    check_cancelled()
                    timeout_check()

                    # Import SegmentationProgressMapper here to avoid circular imports
                    from components.progress import SegmentationProgressMapper

                    # Initialize progress mapper if not exists
                    if not hasattr(enhanced_progress_callback, 'progress_mapper'):
                        enhanced_progress_callback.progress_mapper = SegmentationProgressMapper()

                    mapper = enhanced_progress_callback.progress_mapper

                    # Extract stage information
                    stage = info.get('stage', 'segmentation')
                    slice_info = {
                        'current_slice': info.get('current_slice', 0),
                        'total_slices': info.get('total_slices', 1),
                        'hardware_mode': 'GPU' if gpu_allocated else 'CPU',
                        'batch_size': batch_size,
                        'memory_limit_gb': memory_limit_gb
                    }

                    # Map progress using enhanced mapper
                    enhanced_progress = mapper.map_stage_progress(stage, percentage, slice_info)

                    # Update job with enhanced progress information
                    with self._jobs_lock:
                        if job_info.id in self._jobs:
                            job_info.current_stage = enhanced_progress.stage
                            job_info.stage_progress = enhanced_progress.stage_progress
                            job_info.stage_metadata = enhanced_progress.metadata
                            job_info.hardware_mode = enhanced_progress.hardware_mode
                            if enhanced_progress.eta_seconds:
                                job_info.estimated_completion = datetime.now() + timedelta(
                                    seconds=enhanced_progress.eta_seconds
                                )

                    # Use enhanced overall progress and message
                    scaled_percentage = enhanced_progress.overall_progress
                    update_progress(scaled_percentage, enhanced_progress.message)

                    # Notify observers with enhanced progress data
                    self._notify_observers('enhanced_progress_updated', job_info.id, enhanced_progress)

                except Exception as e:
                    # Fallback to basic progress tracking if enhanced tracking fails
                    self.logger.warning(f"Enhanced progress tracking failed: {e}, falling back to basic tracking")
                    stage = info.get('stage', 'processing')
                    message = info.get('message', f'{stage.title()} in progress')
                    scaled_percentage = 15 + (percentage * 0.8)  # 15-95% range for actual processing
                    update_progress(scaled_percentage, message)

            update_progress(15, "Starting segmentation processing")

            # Run actual segmentation with enhanced monitoring
            try:
                result = workflow.run_segmentation(
                    input_path=input_path,
                    output_dir=output_dir,
                    progress_callback=enhanced_progress_callback
                )

                end_time = time.time()
                processing_time = end_time - start_time

                update_progress(95, "Processing complete, finalizing results")

                # Collect comprehensive result metadata
                job_result = {
                    # Core results
                    "input_path": input_path,
                    "output_path": result.get('output_file', 'Unknown'),
                    "processing_time": processing_time,

                    # Hardware and performance
                    "hardware_mode": result.get('hardware_mode', 'Unknown'),
                    "batch_size_used": result.get('batch_size_used', batch_size or 'default'),
                    "memory_limit_applied": memory_limit_gb,
                    "gpu_allocated": gpu_allocated,

                    # Processing details
                    "total_slices": result.get('total_slices', 0),
                    "successful_slices": result.get('successful_slices', 0),
                    "failed_slices": result.get('failed_slices', 0),

                    # Resource usage
                    "peak_memory_usage": result.get('peak_memory_mb', 0),
                    "gpu_utilization": result.get('gpu_utilization', 0),

                    # Job configuration
                    "timeout_limit_minutes": timeout_minutes,
                    "force_cpu": force_cpu,

                    # Job chaining setup
                    "auto_chain_analysis": auto_chain_analysis,
                    "chain_ready": result.get('output_file') is not None,

                    # Job completion metadata
                    "job_id": job_info.id,
                    "completed_at": time.time()
                }

                update_progress(100, "Segmentation job completed successfully")

                self.logger.info(f"✅ Segmentation job completed: {job_info.id}")
                self.logger.info(f"Processing time: {processing_time:.2f}s")
                self.logger.info(f"Output: {job_result.get('output_path')}")

                return job_result

            except Exception as e:
                self.logger.error(f"Segmentation processing failed: {e}")
                raise RuntimeError(f"Segmentation failed: {str(e)}")

        finally:
            # Enhanced resource cleanup (Task 11.4)
            try:
                # GPU resource cleanup with comprehensive TensorFlow cleanup
                if job_info.gpu_requested:
                    if job_info.gpu_allocated:
                        # Release GPU and perform TensorFlow cleanup
                        self._gpu_manager.release_gpu(job_info.id)
                        self.logger.info(f"GPU resource released for job {job_info.id}")
                    else:
                        # Cancel GPU request if still queued
                        cancelled = self._gpu_manager.cancel_gpu_request(job_info.id)
                        if cancelled:
                            self.logger.info(f"GPU request cancelled for job {job_info.id}")

                    # Update job info
                    with self._jobs_lock:
                        job_info.gpu_allocated = False
                        job_info.gpu_queue_position = 0

            except Exception as cleanup_error:
                self.logger.error(f"Resource cleanup error: {cleanup_error}")

    def _wait_for_gpu_allocation(self, job_info: JobInfo, update_progress: Callable,
                               check_cancelled: Callable, timeout_check: Callable) -> None:
        """
        Wait for GPU allocation from queue with progress updates.

        Args:
            job_info: Job information object
            update_progress: Progress update callback
            check_cancelled: Cancellation check callback
            timeout_check: Timeout check callback

        Raises:
            InterruptedError: If job is cancelled
            TimeoutError: If GPU allocation times out
        """
        start_wait = time.time()
        last_position = job_info.gpu_queue_position

        while not job_info.gpu_allocated:
            check_cancelled()
            timeout_check()

            # Get current queue position
            current_position = self._gpu_manager.get_queue_position(job_info.id)

            if current_position == 0:
                # GPU allocated!
                with self._jobs_lock:
                    job_info.gpu_allocated = True
                    job_info.gpu_queue_position = 0
                    job_info.hardware_mode = "GPU"

                update_progress(8, "GPU allocated - starting processing")
                self.logger.info(f"GPU allocated to job {job_info.id} after queue wait")
                break

            elif current_position == -1:
                # Job not found in queue (possibly timed out)
                raise TimeoutError("GPU request timed out or was cancelled")

            elif current_position != last_position:
                # Position changed, update progress
                with self._jobs_lock:
                    job_info.gpu_queue_position = current_position

                wait_time = time.time() - start_wait
                update_progress(5, f"GPU queue position {current_position} (waiting {wait_time:.0f}s)")
                last_position = current_position

            # Sleep briefly before checking again
            time.sleep(1)

    def get_gpu_status(self) -> Dict[str, Any]:
        """
        Get comprehensive GPU resource status.

        Returns:
            Dictionary with GPU status information
        """
        return self._gpu_manager.get_gpu_status()

    def get_enhanced_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get enhanced job status including GPU queue information.

        Args:
            job_id: Job ID to get status for

        Returns:
            Enhanced job status dictionary or None if job not found
        """
        with self._jobs_lock:
            job_info = self._jobs.get(job_id)
            if job_info is None:
                return None

            status = {
                "id": job_info.id,
                "job_type": job_info.job_type,
                "status": job_info.status.value,
                "progress": job_info.progress,
                "status_message": job_info.status_message,
                "error_message": job_info.error_message,
                "created_at": job_info.created_at.isoformat(),
                "started_at": job_info.started_at.isoformat() if job_info.started_at else None,
                "completed_at": job_info.completed_at.isoformat() if job_info.completed_at else None,
                "result": job_info.result,

                # Enhanced progress tracking fields
                "current_stage": job_info.current_stage,
                "stage_progress": job_info.stage_progress,
                "stage_metadata": job_info.stage_metadata,
                "hardware_mode": job_info.hardware_mode,
                "estimated_completion": job_info.estimated_completion.isoformat() if job_info.estimated_completion else None,

                # GPU queue management fields
                "gpu_requested": job_info.gpu_requested,
                "gpu_allocated": job_info.gpu_allocated,
                "gpu_queue_position": job_info.gpu_queue_position,
                "gpu_timeout_at": job_info.gpu_timeout_at.isoformat() if job_info.gpu_timeout_at else None,
            }

            return status

    def _job_completed_callback(self, job_id: str):
        """
        Callback when a job finishes execution.
        
        Args:
            job_id: Job ID that completed
        """
        with self._active_jobs_lock:
            self._active_jobs.discard(job_id)
        
        # Trigger final callback if provided
        with self._jobs_lock:
            job_info = self._jobs.get(job_id)
            if job_info and job_info.callback:
                try:
                    status_dict = self.get_job_status(job_id)
                    if status_dict:
                        job_info.callback(job_id, job_info.progress, "Job completed", status_dict)
                except Exception as e:
                    self.logger.warning("Final callback failed for job %s: %s", job_id, str(e))
    
    def list_jobs(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all jobs with their information.
        
        Returns:
            Dictionary mapping job IDs to job information dictionaries
        """
        jobs_dict = {}
        with self._jobs_lock:
            for job_id, job_info in self._jobs.items():
                jobs_dict[job_id] = {
                    'job_id': job_id,
                    'job_type': job_info.job_type,
                    'status': job_info.status.value,
                    'progress': job_info.progress,
                    'status_message': job_info.status_message,
                    'error_message': job_info.error_message,
                    'created_at': job_info.created_at,
                    'started_at': job_info.started_at,
                    'completed_at': job_info.completed_at,
                    'params': job_info.params
                }
        return jobs_dict

    def list_active_jobs(self) -> List[str]:
        """
        Get list of active job IDs (queued or running).
        
        Returns:
            List of active job IDs
        """
        active_jobs = []
        with self._jobs_lock:
            for job_id, job_info in self._jobs.items():
                if job_info.status in [JobStatus.QUEUED, JobStatus.RUNNING]:
                    active_jobs.append(job_id)
        return active_jobs
    
    def get_job_queue_info(self) -> Dict[str, Any]:
        """
        Get information about the job queue and workers.
        
        Returns:
            Dictionary with queue and worker information
        """
        with self._jobs_lock:
            queue_size = len(self._job_queue)
            total_jobs = len(self._jobs)
            
            status_counts = {}
            for job_info in self._jobs.values():
                status = job_info.status.value
                status_counts[status] = status_counts.get(status, 0) + 1
        
        with self._active_jobs_lock:
            active_workers = len(self._active_jobs)
        
        return {
            'queue_size': queue_size,
            'active_workers': active_workers,
            'max_workers': self._max_workers,
            'total_jobs': total_jobs,
            'status_counts': status_counts
        }
    
    def cleanup_old_jobs(self, max_age_hours: int = 24):
        """
        Clean up old completed/failed/cancelled jobs.
        
        Args:
            max_age_hours: Maximum age in hours for jobs to keep
        """
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        removed_count = 0
        
        with self._jobs_lock:
            jobs_to_remove = []
            
            for job_id, job_info in self._jobs.items():
                if (job_info.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED] and
                    job_info.completed_at and job_info.completed_at < cutoff_time):
                    jobs_to_remove.append(job_id)
            
            for job_id in jobs_to_remove:
                del self._jobs[job_id]
                removed_count += 1
        
        if removed_count > 0:
            self.logger.info("Cleaned up %d old jobs", removed_count)
        
        return removed_count

    def submit_segmentation_job(self, input_path: str, output_dir: str = None,
                              force_cpu: bool = False, **kwargs) -> str:
        """
        Convenience method to submit a segmentation job.

        Args:
            input_path: Path to input TIFF file
            output_dir: Output directory (default: 'data/segmented')
            force_cpu: Force CPU-only processing
            **kwargs: Additional parameters (batch_size, memory_limit_gb, etc.)

        Returns:
            Job ID string

        Raises:
            ValueError: If parameters are invalid
        """
        params = {
            'input_path': input_path,
            'force_cpu': force_cpu,
            **kwargs
        }

        if output_dir is not None:
            params['output_dir'] = output_dir

        return self.submit_job(JobType.SEGMENTATION.value, params)

    def get_segmentation_output_path(self, job_id: str) -> Optional[str]:
        """
        Get the output path for a completed segmentation job.

        Args:
            job_id: Job ID to query

        Returns:
            Output file path if job completed successfully, None otherwise
        """
        job_status = self.get_job_status(job_id)
        if not job_status:
            return None

        if job_status['status'] != JobStatus.COMPLETED.value:
            return None

        result = job_status.get('result')
        if not result:
            return None

        return result.get('output_path')

    def list_segmentation_jobs(self,
                              status_filter: Optional[List[str]] = None,
                              limit: int = 100,
                              offset: int = 0,
                              start_date: Optional[datetime] = None,
                              end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        List segmentation jobs with filtering options (following Celery GroupResult patterns).

        Args:
            status_filter: Filter by job status (e.g., ['completed', 'running', 'failed'])
            limit: Maximum number of jobs to return (pagination)
            offset: Number of jobs to skip (pagination)
            start_date: Filter jobs created after this date
            end_date: Filter jobs created before this date

        Returns:
            Dictionary containing:
            - jobs: List of job information dictionaries
            - total_count: Total number of matching jobs
            - filtered_count: Number of jobs after filtering
            - has_more: Whether there are more jobs beyond the current page
            - summary: Status count summary (successful, failed, running, etc.)
        """
        self.logger.info(f"Listing segmentation jobs: status_filter={status_filter}, "
                        f"limit={limit}, offset={offset}")

        with self._jobs_lock:
            # Get all segmentation jobs
            segmentation_jobs = []
            for job_id, job_info in self._jobs.items():
                if job_info.job_type == JobType.SEGMENTATION.value:
                    segmentation_jobs.append((job_id, job_info))

            total_count = len(segmentation_jobs)

            # Apply filters
            filtered_jobs = []
            for job_id, job_info in segmentation_jobs:
                # Status filter
                if status_filter and job_info.status.value not in status_filter:
                    continue

                # Date range filter
                if start_date and job_info.created_at < start_date:
                    continue
                if end_date and job_info.created_at > end_date:
                    continue

                filtered_jobs.append((job_id, job_info))

            filtered_count = len(filtered_jobs)

            # Sort by creation time (newest first)
            filtered_jobs.sort(key=lambda x: x[1].created_at, reverse=True)

            # Apply pagination
            paginated_jobs = filtered_jobs[offset:offset + limit]
            has_more = (offset + limit) < filtered_count

            # Build job list with comprehensive information
            job_list = []
            for job_id, job_info in paginated_jobs:
                # Get enhanced status for additional GPU info
                enhanced_status = self.get_enhanced_job_status(job_id)

                job_dict = {
                    'job_id': job_id,
                    'status': job_info.status.value,
                    'created_at': job_info.created_at.isoformat(),
                    'started_at': job_info.started_at.isoformat() if job_info.started_at else None,
                    'completed_at': job_info.completed_at.isoformat() if job_info.completed_at else None,
                    'progress': job_info.progress,
                    'parameters': job_info.params,
                    'error_message': job_info.error_message,
                    'result': job_info.result
                }

                # Add enhanced information if available
                if enhanced_status:
                    job_dict.update({
                        'hardware_mode': enhanced_status.get('hardware_mode'),
                        'gpu_requested': enhanced_status.get('gpu_requested', False),
                        'gpu_allocated': enhanced_status.get('gpu_allocated', False),
                        'gpu_queue_position': enhanced_status.get('gpu_queue_position', 0),
                        'execution_time_seconds': enhanced_status.get('execution_time_seconds'),
                        'memory_usage_mb': enhanced_status.get('memory_usage_mb')
                    })

                job_list.append(job_dict)

            # Calculate status summary (Celery GroupResult style)
            status_summary = {
                'successful': 0,
                'failed': 0,
                'running': 0,
                'waiting': 0,  # queued
                'cancelled': 0,
                'total': filtered_count
            }

            for _, job_info in filtered_jobs:
                if job_info.status == JobStatus.COMPLETED:
                    status_summary['successful'] += 1
                elif job_info.status == JobStatus.FAILED:
                    status_summary['failed'] += 1
                elif job_info.status == JobStatus.RUNNING:
                    status_summary['running'] += 1
                elif job_info.status == JobStatus.QUEUED:
                    status_summary['waiting'] += 1
                elif job_info.status == JobStatus.CANCELLED:
                    status_summary['cancelled'] += 1

            result = {
                'jobs': job_list,
                'total_count': total_count,
                'filtered_count': filtered_count,
                'returned_count': len(job_list),
                'has_more': has_more,
                'pagination': {
                    'limit': limit,
                    'offset': offset,
                    'next_offset': offset + limit if has_more else None
                },
                'summary': status_summary
            }

            self.logger.info(f"Returned {len(job_list)} segmentation jobs out of {filtered_count} filtered "
                           f"({total_count} total)")
            return result

    def get_segmentation_job_results(self, job_id: str) -> Dict[str, Any]:
        """
        Get comprehensive results for a segmentation job (following AsyncResult patterns).

        Args:
            job_id: Job ID to get results for

        Returns:
            Dictionary containing:
            - job_info: Basic job information
            - status: Current job status
            - result: Job execution result
            - output_files: List of generated output files
            - metadata: Processing metadata and statistics
            - error_info: Error details if job failed
            - performance_metrics: Timing and resource usage information

        Raises:
            ValueError: If job_id is invalid or job is not a segmentation job
        """
        self.logger.info(f"Getting segmentation job results for: {job_id}")

        # Get basic job status
        job_status = self.get_job_status(job_id)
        if not job_status:
            raise ValueError(f"Job not found: {job_id}")

        # Verify it's a segmentation job
        with self._jobs_lock:
            job_info = self._jobs.get(job_id)
            if not job_info or job_info.job_type != JobType.SEGMENTATION.value:
                raise ValueError(f"Job {job_id} is not a segmentation job")

        # Get enhanced status for additional information
        enhanced_status = self.get_enhanced_job_status(job_id)

        # Build comprehensive result following AsyncResult patterns
        result = {
            'job_id': job_id,
            'job_info': {
                'status': job_status['status'],
                'created_at': job_info.created_at.isoformat(),
                'started_at': job_info.started_at.isoformat() if job_info.started_at else None,
                'completed_at': job_info.completed_at.isoformat() if job_info.completed_at else None,
                'progress': job_status['progress'],
                'parameters': job_info.params
            },
            'status': {
                'current': job_status['status'],
                'is_ready': job_status['status'] in [JobStatus.COMPLETED.value, JobStatus.FAILED.value, JobStatus.CANCELLED.value],
                'is_successful': job_status['status'] == JobStatus.COMPLETED.value,
                'is_failed': job_status['status'] == JobStatus.FAILED.value,
                'is_running': job_status['status'] == JobStatus.RUNNING.value,
                'is_waiting': job_status['status'] == JobStatus.QUEUED.value
            },
            'result': job_status.get('result'),
            'output_files': [],
            'metadata': {},
            'error_info': None,
            'performance_metrics': {}
        }

        # Add enhanced information if available
        if enhanced_status:
            result['job_info'].update({
                'hardware_mode': enhanced_status.get('hardware_mode'),
                'gpu_requested': enhanced_status.get('gpu_requested', False),
                'gpu_allocated': enhanced_status.get('gpu_allocated', False),
                'gpu_queue_position': enhanced_status.get('gpu_queue_position', 0)
            })

            result['performance_metrics'] = {
                'execution_time_seconds': enhanced_status.get('execution_time_seconds'),
                'memory_usage_mb': enhanced_status.get('memory_usage_mb'),
                'hardware_mode': enhanced_status.get('hardware_mode'),
                'processing_speed': enhanced_status.get('processing_speed')
            }

        # Add output files information if job completed successfully
        if job_status['status'] == JobStatus.COMPLETED.value and job_status.get('result'):
            job_result = job_status['result']
            output_path = job_result.get('output_path')

            if output_path:
                # Check if output file exists
                from pathlib import Path
                output_file = Path(output_path)

                if output_file.exists():
                    result['output_files'].append({
                        'type': 'segmented_image',
                        'path': str(output_file),
                        'filename': output_file.name,
                        'size_bytes': output_file.stat().st_size,
                        'exists': True
                    })

                    # Look for metadata JSON file
                    metadata_path = output_file.with_suffix('.json')
                    if metadata_path.exists():
                        result['output_files'].append({
                            'type': 'metadata',
                            'path': str(metadata_path),
                            'filename': metadata_path.name,
                            'size_bytes': metadata_path.stat().st_size,
                            'exists': True
                        })

                        # Load metadata content
                        try:
                            import json
                            with open(metadata_path, 'r') as f:
                                result['metadata'] = json.load(f)
                        except Exception as e:
                            self.logger.warning(f"Could not load metadata from {metadata_path}: {e}")
                else:
                    result['output_files'].append({
                        'type': 'segmented_image',
                        'path': str(output_file),
                        'filename': output_file.name,
                        'size_bytes': None,
                        'exists': False,
                        'error': 'Output file not found'
                    })

        # Add error information if job failed
        if job_status['status'] == JobStatus.FAILED.value:
            result['error_info'] = {
                'error_message': job_info.error_message,
                'error_type': 'ProcessingError',
                'timestamp': job_info.completed_at.isoformat() if job_info.completed_at else None,
                'traceback': job_status.get('traceback'),
                'suggestions': self._get_error_suggestions(job_info.error_message)
            }

        self.logger.info(f"Retrieved comprehensive results for segmentation job {job_id}: "
                        f"status={result['status']['current']}, "
                        f"files={len(result['output_files'])}")

        return result

    def _get_error_suggestions(self, error_message: str) -> List[str]:
        """
        Generate helpful suggestions based on error message.

        Args:
            error_message: The error message to analyze

        Returns:
            List of suggested solutions
        """
        if not error_message:
            return []

        suggestions = []
        error_lower = error_message.lower()

        if 'memory' in error_lower or 'oom' in error_lower:
            suggestions.extend([
                "Try reducing the batch size in job parameters",
                "Consider using force_cpu=True to reduce memory usage",
                "Check if system has sufficient RAM available"
            ])

        if 'gpu' in error_lower or 'cuda' in error_lower:
            suggestions.extend([
                "GPU may be unavailable - try force_cpu=True",
                "Check GPU memory usage and availability",
                "Verify CUDA drivers are properly installed"
            ])

        if 'file' in error_lower or 'path' in error_lower:
            suggestions.extend([
                "Verify input file exists and is readable",
                "Check file format is valid TIFF",
                "Ensure output directory is writable"
            ])

        if 'timeout' in error_lower:
            suggestions.extend([
                "Increase timeout_minutes parameter",
                "Try processing smaller images or image sections",
                "Check system performance and load"
            ])

        return suggestions

    def get_supported_job_types(self) -> List[str]:
        """
        Get list of supported job types.

        Returns:
            List of supported job type strings
        """
        return [job_type.value for job_type in JobType]

    def get_job_type_schema(self, job_type: str) -> Dict[str, Any]:
        """
        Get parameter schema for a specific job type.

        Args:
            job_type: Job type to get schema for

        Returns:
            Dictionary describing expected parameters

        Raises:
            ValueError: If job_type is not supported
        """
        if not JobType.is_valid(job_type):
            raise ValueError(f"Unsupported job type: {job_type}")

        if job_type == JobType.SEGMENTATION.value:
            return {
                'required': ['input_path'],
                'optional': {
                    'output_dir': {'type': 'str', 'default': 'data/segmented'},
                    'force_cpu': {'type': 'bool', 'default': False},
                    'batch_size': {'type': 'int', 'range': [1, 32]},
                    'memory_limit_gb': {'type': 'float', 'range': [0.1, 128.0]},
                    'timeout_minutes': {'type': 'float', 'range': [1.0, 120.0], 'default': 45.0},
                    'auto_chain_analysis': {'type': 'bool', 'default': False}
                },
                'description': 'Mitochondrial segmentation using MoDL U-RNet+ model'
            }
        elif job_type == JobType.TEST.value:
            return {
                'required': [],
                'optional': {
                    'duration': {'type': 'float', 'default': 5.0},
                    'steps': {'type': 'int', 'default': 10}
                },
                'description': 'Test job for validation and debugging'
            }
        elif job_type == JobType.ANALYSIS.value:
            return {
                'required': [],
                'optional': {},
                'description': 'Analysis job (not yet implemented)'
            }

        return {}

    def shutdown(self, wait: bool = True):
        """
        Shutdown the job manager gracefully.
        
        Args:
            wait: Whether to wait for current jobs to complete
        """
        if self._shutdown:
            return
        
        self.logger.info("JobManager shutdown initiated")
        self._shutdown = True
        self._shutdown_event.set()
        
        # Cancel all queued jobs
        # Cancel all pending and running jobs using observer-aware method
        job_ids_to_cancel = []
        with self._jobs_lock:
            for job_id, job_info in self._jobs.items():
                if job_info.status == JobStatus.QUEUED:
                    job_ids_to_cancel.append(job_id)
                elif job_info.status == JobStatus.RUNNING:
                    # Signal running jobs to cancel
                    job_info.cancelled_event.set()
        
        # Cancel queued jobs outside the lock to avoid deadlock with observers
        for job_id in job_ids_to_cancel:
            self._update_job_status(job_id, JobStatus.CANCELLED, "Cancelled due to shutdown")
        
        # Shutdown thread pool
        self._executor.shutdown(wait=wait)
        
        # Wait for dispatcher thread with timeout
        if self._dispatcher_thread and self._dispatcher_thread.is_alive():
            self._dispatcher_thread.join(timeout=1.0)
            if self._dispatcher_thread.is_alive():
                self.logger.warning("Dispatcher thread did not shut down cleanly")
        
        self.logger.info("JobManager shutdown complete")
    
    def _notify_observers(self, event_type: str, job_id: str, job_info: JobInfo):
        """Notify all registered observers about job events."""
        if not self._observers:
            return
            
        with self._observers_lock:
            for observer in self._observers:
                try:
                    if event_type == 'created':
                        observer.on_job_created(job_id, job_info)
                    elif event_type == 'status_changed':
                        observer.on_job_status_changed(job_id, job_info)
                    elif event_type == 'progress_updated':
                        observer.on_job_progress_updated(job_id, job_info)
                except Exception as e:
                    self.logger.error("Observer notification failed for %s: %s", event_type, e)
    
    def _update_job_status(self, job_id: str, status: JobStatus, message: str = "", 
                          error_message: str = "", result: Any = None):
        """Update job status and notify observers."""
        with self._jobs_lock:
            if job_id not in self._jobs:
                return False
                
            job_info = self._jobs[job_id]
            old_status = job_info.status
            job_info.status = status
            job_info.status_message = message
            
            if error_message:
                job_info.error_message = error_message
            if result is not None:
                job_info.result = result
                
            # Update timestamps
            if status == JobStatus.RUNNING and not job_info.started_at:
                job_info.started_at = datetime.now()
            elif status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                job_info.completed_at = datetime.now()
            
            # Notify observers if status changed
            if old_status != status:
                self._notify_observers('status_changed', job_id, job_info)
            
            return True


# Module-level convenience functions
_job_manager_instance = None

def get_job_manager() -> JobManager:
    """Get the singleton JobManager instance."""
    global _job_manager_instance
    if _job_manager_instance is None:
        _job_manager_instance = JobManager()
    return _job_manager_instance

def submit_job(job_type: str, params: Dict[str, Any], callback: Optional[Callable] = None) -> str:
    """Convenience function to submit a job."""
    return get_job_manager().submit_job(job_type, params, callback)

def get_job_status(job_id: str) -> Optional[Dict[str, Any]]:
    """Convenience function to get job status."""
    return get_job_manager().get_job_status(job_id)

def cancel_job(job_id: str) -> bool:
    """Convenience function to cancel a job."""
    return get_job_manager().cancel_job(job_id)

def submit_segmentation_job(input_path: str, output_dir: str = None,
                          force_cpu: bool = False, **kwargs) -> str:
    """Convenience function to submit a segmentation job."""
    return get_job_manager().submit_segmentation_job(input_path, output_dir, force_cpu, **kwargs)

def get_segmentation_output_path(job_id: str) -> Optional[str]:
    """Convenience function to get segmentation output path."""
    return get_job_manager().get_segmentation_output_path(job_id)

def list_segmentation_jobs(status_filter: Optional[List[str]] = None,
                         limit: int = 100,
                         offset: int = 0,
                         start_date: Optional[datetime] = None,
                         end_date: Optional[datetime] = None) -> Dict[str, Any]:
    """Convenience function to list segmentation jobs with filtering options."""
    return get_job_manager().list_segmentation_jobs(status_filter, limit, offset, start_date, end_date)

def get_segmentation_job_results(job_id: str) -> Dict[str, Any]:
    """Convenience function to get comprehensive results for a segmentation job."""
    return get_job_manager().get_segmentation_job_results(job_id)

def get_supported_job_types() -> List[str]:
    """Convenience function to get supported job types."""
    return get_job_manager().get_supported_job_types()

def get_job_type_schema(job_type: str) -> Dict[str, Any]:
    """Convenience function to get job type schema."""
    return get_job_manager().get_job_type_schema(job_type)