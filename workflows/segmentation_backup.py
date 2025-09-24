"""
Segmentation Workflow Wrapper for MoDL Pipeline Integration
Provides Streamlit-friendly interface with background processing and progress tracking
"""

import os
import subprocess
import threading
import time
import logging
import json
import shutil
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import streamlit as st
import numpy as np
from PIL import Image
import tifffile
from collections import defaultdict


class SegmentationWorkflow:
    """
    Streamlit-friendly wrapper for MoDL segmentation pipeline with enhanced error handling,
    background job management, and progress tracking for web interface integration.
    """
    
    def __init__(self):
        """Initialize the segmentation workflow wrapper."""
        self.logger = self._setup_logging()
        self.modl_path = Path("MoDL")
        self.jobs = {}  # Active job tracking
        self.job_lock = threading.Lock()
        self._validation_cache = {}  # Cache validation results to prevent repeated logging
        self._cache_lock = threading.Lock()
        
        # Ensure required directories exist
        self._setup_directories()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration for the segmentation workflow."""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
        
    def _setup_directories(self):
        """Setup required directories for MoDL processing."""
        required_dirs = [
            "data/raw",
            "data/segmented",
            "data/temp",
            "MoDL/temp_jobs"
        ]
        
        for dir_path in required_dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            
    def validate_raw_image(self, image_path: str) -> Tuple[bool, Optional[str]]:
        """
        Validate raw TIFF image for segmentation processing.
        
        Args:
            image_path: Path to the raw TIFF file
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check cache first to avoid repeated validation logging
        cache_key = f"{image_path}_{os.path.getmtime(image_path) if os.path.exists(image_path) else 0}"
        
        with self._cache_lock:
            if cache_key in self._validation_cache:
                return self._validation_cache[cache_key]
        
        try:
            # Check file existence and readability
            if not os.path.exists(image_path):
                result = (False, f"File does not exist: {image_path}")
                with self._cache_lock:
                    self._validation_cache[cache_key] = result
                return result
                
            if not os.access(image_path, os.R_OK):
                result = (False, f"File is not readable: {image_path}")
                with self._cache_lock:
                    self._validation_cache[cache_key] = result
                return result
                
            # Check file extension
            file_ext = Path(image_path).suffix.lower()
            if file_ext not in ['.tif', '.tiff']:
                result = (False, f"Invalid file type: {file_ext}. Expected .tif or .tiff")
                with self._cache_lock:
                    self._validation_cache[cache_key] = result
                return result
                
            # Try to load and validate image properties
            try:
                with tifffile.TiffFile(image_path) as tif:
                    if len(tif.pages) < 1:
                        result = (False, "TIFF file contains no images")
                        with self._cache_lock:
                            self._validation_cache[cache_key] = result
                        return result
                        
                    # Check image dimensions
                    page = tif.pages[0]
                    height, width = page.shape[:2]
                    
                    # Validate minimum dimensions for MoDL processing
                    if width < 512 or height < 512:
                        result = (False, f"Image too small: {width}x{height}. Minimum 512x512 required.")
                        with self._cache_lock:
                            self._validation_cache[cache_key] = result
                        return result
                        
                    # Check if image is too large (memory constraints)
                    if width > 4096 or height > 4096:
                        result = (False, f"Image too large: {width}x{height}. Maximum 4096x4096 supported.")
                        with self._cache_lock:
                            self._validation_cache[cache_key] = result
                        return result
                        
                    # Check bit depth
                    if page.dtype not in [np.uint8, np.uint16]:
                        result = (False, f"Unsupported bit depth: {page.dtype}. Expected uint8 or uint16.")
                        with self._cache_lock:
                            self._validation_cache[cache_key] = result
                        return result
                        
            except Exception as e:
                result = (False, f"Invalid TIFF file format: {str(e)}")
                with self._cache_lock:
                    self._validation_cache[cache_key] = result
                return result
                
            # Check file size for memory estimation
            file_size_mb = os.path.getsize(image_path) / (1024 * 1024)
            if file_size_mb > 500:  # 500MB limit
                result = (False, f"File size ({file_size_mb:.1f} MB) exceeds processing limit (500 MB)")
                with self._cache_lock:
                    self._validation_cache[cache_key] = result
                return result
                
            # Only log on first successful validation, not on cache hits
            self.logger.info(f"Image validation passed for {Path(image_path).name}")
            result = (True, None)
            with self._cache_lock:
                self._validation_cache[cache_key] = result
            return result
            
        except Exception as e:
            result = (False, f"Validation error: {str(e)}")
            with self._cache_lock:
                self._validation_cache[cache_key] = result
            return result
    
    def estimate_processing_time(self, image_path: str) -> Dict[str, Any]:
        """
        Estimate processing time and resource requirements.
        
        Args:
            image_path: Path to the raw TIFF file
            
        Returns:
            Dictionary with time and resource estimates
        """
        try:
            with tifffile.TiffFile(image_path) as tif:
                page = tif.pages[0]
                height, width = page.shape[:2]
                num_slices = len(tif.pages)
                
            # Rough time estimation based on image size
            pixels = height * width * num_slices
            
            # Base processing time estimates (in seconds per million pixels)
            base_time_per_mpx = 30  # Conservative estimate
            estimated_time = (pixels / 1_000_000) * base_time_per_mpx
            
            # Factor in patch processing (MoDL uses 512x512 patches)
            patches_per_slice = max(1, (height // 512) * (width // 512))
            total_patches = patches_per_slice * num_slices
            
            estimates = {
                'image_dimensions': f"{width}x{height}",
                'num_slices': num_slices,
                'total_patches': total_patches,
                'estimated_time_minutes': max(1, estimated_time / 60),
                'estimated_memory_gb': max(0.5, (pixels * 4) / (1024**3)),  # 4 bytes per pixel estimate
                'processing_complexity': 'High' if total_patches > 100 else 'Medium' if total_patches > 25 else 'Low'
            }
            
            return estimates
            
        except Exception as e:
            self.logger.error(f"Time estimation error: {e}")
            return {'error': str(e)}
    
    def create_job(self, image_path: str, params: Dict[str, Any]) -> str:
        """
        Create a new segmentation job.
        
        Args:
            image_path: Path to the raw TIFF file
            params: Segmentation parameters
            
        Returns:
            Unique job ID
        """
        job_id = str(uuid.uuid4())
        
        job_info = {
            'job_id': job_id,
            'image_path': image_path,
            'image_name': Path(image_path).stem,
            'params': params.copy(),
            'status': 'created',
            'progress': 0,
            'created_time': time.time(),
            'start_time': None,
            'completion_time': None,
            'error_message': None,
            'warnings': [],
            'output_path': None,
            'temp_dir': Path(f"MoDL/temp_jobs/{job_id}"),
            'log_file': Path(f"MoDL/temp_jobs/{job_id}/processing.log")
        }
        
        # Create temporary directory for this job
        job_info['temp_dir'].mkdir(parents=True, exist_ok=True)
        
        with self.job_lock:
            self.jobs[job_id] = job_info
            
        self.logger.info(f"Created segmentation job {job_id} for {Path(image_path).name}")
        return job_id
    
    def submit_job(self, job_id: str) -> bool:
        """
        Submit a job for background processing.
        
        Args:
            job_id: Job identifier
            
        Returns:
            True if job was submitted successfully
        """
        with self.job_lock:
            if job_id not in self.jobs:
                return False
                
            job_info = self.jobs[job_id]
            if job_info['status'] != 'created':
                return False
                
            job_info['status'] = 'queued'
            job_info['start_time'] = time.time()
            
        # Start background processing thread
        thread = threading.Thread(
            target=self._process_segmentation_job, 
            args=(job_id,),
            daemon=True
        )
        thread.start()
        
        self.logger.info(f"Submitted segmentation job {job_id}")
        return True
    
    def _process_segmentation_job(self, job_id: str):
        """
        Background thread function for processing segmentation jobs.
        
        Args:
            job_id: Job identifier
        """
        try:
            with self.job_lock:
                job_info = self.jobs[job_id]
                job_info['status'] = 'processing'
                job_info['progress'] = 5
                
            self.logger.info(f"Starting processing for job {job_id}")
            
            # Stage 1: Setup and prepare input (10%)
            self._update_job_progress(job_id, 10, "Setting up processing environment...")
            success = self._setup_job_environment(job_id)
            if not success:
                self._fail_job(job_id, "Failed to setup processing environment")
                return
                
            # Stage 2: Preprocess image (20%)
            self._update_job_progress(job_id, 20, "Preprocessing raw image...")
            success = self._preprocess_image(job_id)
            if not success:
                self._fail_job(job_id, "Failed to preprocess image")
                return
                
            # Stage 3: Run MoDL segmentation (30% - 80%)
            self._update_job_progress(job_id, 30, "Running MoDL segmentation...")
            success = self._run_modl_segmentation(job_id)
            if not success:
                self._fail_job(job_id, "MoDL segmentation failed")
                return
                
            # Stage 4: Postprocess results (90%)
            self._update_job_progress(job_id, 90, "Postprocessing segmentation results...")
            success = self._postprocess_results(job_id)
            if not success:
                self._fail_job(job_id, "Failed to postprocess results")
                return
                
            # Stage 5: Finalize and cleanup (100%)
            self._update_job_progress(job_id, 100, "Finalizing results...")
            self._complete_job(job_id)
            
        except Exception as e:
            self.logger.error(f"Job {job_id} failed with exception: {e}")
            self._fail_job(job_id, f"Unexpected error: {str(e)}")
    
    def _update_job_progress(self, job_id: str, progress: int, status_message: str):
        """Update job progress and status message."""
        with self.job_lock:
            if job_id in self.jobs:
                self.jobs[job_id]['progress'] = progress
                self.jobs[job_id]['status_message'] = status_message
                
                # Log to file
                log_file = self.jobs[job_id]['log_file']
                with open(log_file, 'a') as f:
                    f.write(f"{time.strftime('%H:%M:%S')} - {progress}% - {status_message}\n")
    
    def _setup_job_environment(self, job_id: str) -> bool:
        """Setup processing environment for a specific job."""
        try:
            job_info = self.jobs[job_id]
            temp_dir = job_info['temp_dir']
            
            # Create required subdirectories
            subdirs = ['input', 'patches', 'results', 'output']
            for subdir in subdirs:
                (temp_dir / subdir).mkdir(exist_ok=True)
                
            # Copy input image to temp directory
            input_path = job_info['image_path']
            temp_input = temp_dir / 'input' / Path(input_path).name
            shutil.copy2(input_path, temp_input)
            
            job_info['temp_input_path'] = str(temp_input)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Environment setup failed for job {job_id}: {e}")
            return False
    
    def _preprocess_image(self, job_id: str) -> bool:
        """Preprocess raw image for MoDL processing."""
        try:
            job_info = self.jobs[job_id]
            
            # For now, we'll implement basic preprocessing
            # In the full implementation, this would handle:
            # - Image normalization
            # - Format conversion if needed
            # - Quality checks
            
            self.logger.info(f"Preprocessing completed for job {job_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Preprocessing failed for job {job_id}: {e}")
            return False
    
    def _run_modl_segmentation(self, job_id: str) -> bool:
        """Run the MoDL segmentation pipeline."""
        try:
            job_info = self.jobs[job_id]
            temp_dir = job_info['temp_dir']
            
            # Update progress during segmentation
            for i in range(30, 81, 10):
                time.sleep(2)  # Simulate processing time
                self._update_job_progress(job_id, i, f"Processing patches... ({i-30+10}/50)")
                
            # For demo purposes, create a mock segmentation result
            # In the real implementation, this would call MoDL subprocess
            output_path = temp_dir / 'output' / f"{job_info['image_name']}_segmented.tif"
            
            # Create a simple mock segmented image
            input_image_path = job_info['temp_input_path']
            with tifffile.TiffFile(input_image_path) as tif:
                original = tif.asarray()
                if len(original.shape) == 3:  # Multi-slice
                    segmented = (original > np.percentile(original, 50)).astype(np.uint8) * 255
                else:  # Single slice
                    segmented = (original > np.percentile(original, 50)).astype(np.uint8) * 255
                    
            tifffile.imwrite(output_path, segmented)
            job_info['segmentation_output'] = str(output_path)
            
            return True
            
        except Exception as e:
            self.logger.error(f"MoDL segmentation failed for job {job_id}: {e}")
            return False
    
    def _postprocess_results(self, job_id: str) -> bool:
        """Postprocess segmentation results."""
        try:
            job_info = self.jobs[job_id]
            
            # Move final result to segmented data directory
            temp_output = job_info['segmentation_output']
            final_output = Path("data/segmented") / Path(temp_output).name
            shutil.move(temp_output, final_output)
            
            job_info['final_output_path'] = str(final_output)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Postprocessing failed for job {job_id}: {e}")
            return False
    
    def _complete_job(self, job_id: str):
        """Mark job as completed."""
        with self.job_lock:
            if job_id in self.jobs:
                self.jobs[job_id]['status'] = 'completed'
                self.jobs[job_id]['completion_time'] = time.time()
                self.jobs[job_id]['progress'] = 100
                
                # Cleanup temp directory (keep logs)
                self._cleanup_job_temp_files(job_id)
                
                self.logger.info(f"Job {job_id} completed successfully")
    
    def _fail_job(self, job_id: str, error_message: str):
        """Mark job as failed."""
        with self.job_lock:
            if job_id in self.jobs:
                self.jobs[job_id]['status'] = 'failed'
                self.jobs[job_id]['error_message'] = error_message
                self.jobs[job_id]['completion_time'] = time.time()
                
                self.logger.error(f"Job {job_id} failed: {error_message}")
    
    def _cleanup_job_temp_files(self, job_id: str):
        """Cleanup temporary files for completed job."""
        try:
            job_info = self.jobs[job_id]
            temp_dir = job_info['temp_dir']
            
            # Remove all files except logs
            for item in temp_dir.iterdir():
                if item.name != 'processing.log':
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)
                        
        except Exception as e:
            self.logger.warning(f"Cleanup warning for job {job_id}: {e}")
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a job."""
        with self.job_lock:
            if job_id not in self.jobs:
                return None
                
            job_info = self.jobs[job_id].copy()
            
            # Add calculated fields
            if job_info['start_time']:
                if job_info['completion_time']:
                    job_info['total_time'] = job_info['completion_time'] - job_info['start_time']
                else:
                    job_info['elapsed_time'] = time.time() - job_info['start_time']
                    
            return job_info
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job."""
        with self.job_lock:
            if job_id not in self.jobs:
                return False
                
            job_info = self.jobs[job_id]
            if job_info['status'] in ['completed', 'failed', 'cancelled']:
                return False
                
            job_info['status'] = 'cancelled'
            job_info['completion_time'] = time.time()
            
            # Cleanup
            self._cleanup_job_temp_files(job_id)
            
            self.logger.info(f"Job {job_id} cancelled")
            return True
    
    def list_active_jobs(self) -> List[Dict[str, Any]]:
        """Get list of all active jobs."""
        with self.job_lock:
            return [
                {
                    'job_id': job_id,
                    'image_name': info['image_name'],
                    'status': info['status'],
                    'progress': info['progress'],
                    'created_time': info['created_time']
                }
                for job_id, info in self.jobs.items()
                if info['status'] not in ['completed', 'failed', 'cancelled']
            ]
    
    def cleanup_old_jobs(self, max_age_hours: int = 24):
        """Cleanup old completed/failed jobs."""
        current_time = time.time()
        cutoff_time = current_time - (max_age_hours * 3600)
        
        jobs_to_remove = []
        with self.job_lock:
            for job_id, job_info in self.jobs.items():
                if (job_info.get('completion_time', current_time) < cutoff_time and 
                    job_info['status'] in ['completed', 'failed', 'cancelled']):
                    jobs_to_remove.append(job_id)
        
        for job_id in jobs_to_remove:
            self._cleanup_job_temp_files(job_id)
            with self.job_lock:
                del self.jobs[job_id]
                
        if jobs_to_remove:
            self.logger.info(f"Cleaned up {len(jobs_to_remove)} old jobs")


# Global instance for use throughout the application
segmentation_workflow = SegmentationWorkflow()