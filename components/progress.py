"""
Enhanced Progress Bar and Job Queue UI Components

Streamlit components for displaying job progress, queue status, and job management.
Integrates with JobManager and persistence system for real-time updates.
Includes enhanced stage-aware progress tracking for complex workflows.
"""

import streamlit as st
import time
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import os
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

# Import job management systems
try:
    from workflows.job_manager import get_job_manager, JobStatus
    from utils.job_persistence import get_job_persistence, get_job_history
except ImportError as e:
    st.error(f"Failed to import job management systems: {e}")
    st.stop()


# Enhanced Progress Tracking Classes
class SegmentationStage(Enum):
    """Enumeration of segmentation workflow stages."""
    INITIALIZATION = "initialization"
    LOADING = "loading"
    PLANNING = "planning"
    SETUP = "setup"
    PREPROCESSING = "preprocessing"
    SEGMENTATION = "segmentation"
    RECONSTRUCTION = "reconstruction"
    ASSEMBLY = "assembly"
    SAVING = "saving"
    CLEANUP = "cleanup"


@dataclass
class EnhancedProgressInfo:
    """Enhanced progress information with stage-aware tracking."""
    overall_progress: float  # 0-100 overall job progress
    stage: str  # Current stage name
    stage_progress: float  # 0-100 progress within current stage
    current_item: int  # Current slice/batch being processed
    total_items: int  # Total slices/batches
    message: str  # Detailed progress message
    hardware_mode: str  # CPU/GPU processing mode
    memory_usage_mb: Optional[float] = None  # Current memory usage
    eta_seconds: Optional[float] = None  # Estimated time remaining
    stage_start_time: Optional[datetime] = None  # When current stage started
    metadata: Dict[str, Any] = field(default_factory=dict)  # Stage-specific metadata


class SegmentationProgressMapper:
    """Maps segmentation workflow progress to enhanced job progress."""

    # Stage progress ranges (overall percentage)
    STAGE_RANGES = {
        SegmentationStage.INITIALIZATION: (0, 5),
        SegmentationStage.LOADING: (5, 10),
        SegmentationStage.PLANNING: (10, 12),
        SegmentationStage.SETUP: (12, 15),
        SegmentationStage.PREPROCESSING: (15, 25),
        SegmentationStage.SEGMENTATION: (25, 75),  # Main processing stage
        SegmentationStage.RECONSTRUCTION: (75, 80),
        SegmentationStage.ASSEMBLY: (80, 85),
        SegmentationStage.SAVING: (85, 95),
        SegmentationStage.CLEANUP: (95, 100)
    }

    def __init__(self):
        """Initialize the progress mapper."""
        self.current_stage = None
        self.stage_start_time = None

    def map_stage_progress(self, stage: str, stage_progress: float,
                          slice_info: Optional[Dict] = None) -> EnhancedProgressInfo:
        """
        Map workflow stage progress to enhanced job progress.

        Args:
            stage: Current stage name
            stage_progress: Progress within the stage (0-100)
            slice_info: Optional slice processing information

        Returns:
            EnhancedProgressInfo object with mapped progress
        """
        # Try to map string stage to enum
        stage_enum = None
        for enum_stage in SegmentationStage:
            if enum_stage.value == stage or stage.lower().startswith(enum_stage.value):
                stage_enum = enum_stage
                break

        # Default to segmentation stage if unknown
        if stage_enum is None:
            stage_enum = SegmentationStage.SEGMENTATION

        # Track stage transitions
        if self.current_stage != stage_enum:
            self.current_stage = stage_enum
            self.stage_start_time = datetime.now()

        # Get stage range
        start_pct, end_pct = self.STAGE_RANGES[stage_enum]

        # Calculate overall progress
        stage_range = end_pct - start_pct
        overall_progress = start_pct + (stage_progress / 100.0) * stage_range

        # Extract slice information
        current_item = 0
        total_items = 1
        hardware_mode = "Unknown"

        if slice_info:
            current_item = slice_info.get('current_slice', 0)
            total_items = slice_info.get('total_slices', 1)
            hardware_mode = slice_info.get('hardware_mode', 'Unknown')

        # Create enhanced message
        if stage_enum == SegmentationStage.SEGMENTATION and total_items > 1:
            message = f"Processing slice {current_item}/{total_items} on {hardware_mode}"
        elif stage_enum == SegmentationStage.PREPROCESSING:
            message = f"Preprocessing data for {hardware_mode} processing"
        elif stage_enum == SegmentationStage.LOADING:
            message = "Loading and validating input data"
        elif stage_enum == SegmentationStage.SAVING:
            message = "Saving segmentation results"
        else:
            message = f"{stage_enum.value.title()} in progress"

        # Calculate ETA if we have timing information
        eta_seconds = None
        if self.stage_start_time and stage_progress > 0:
            elapsed = (datetime.now() - self.stage_start_time).total_seconds()
            if stage_progress < 100:
                eta_seconds = elapsed * ((100 - stage_progress) / stage_progress)

        return EnhancedProgressInfo(
            overall_progress=min(100.0, max(0.0, overall_progress)),
            stage=stage_enum.value,
            stage_progress=max(0.0, min(100.0, stage_progress)),
            current_item=current_item,
            total_items=total_items,
            message=message,
            hardware_mode=hardware_mode,
            eta_seconds=eta_seconds,
            stage_start_time=self.stage_start_time,
            metadata=slice_info or {}
        )

    def get_stage_summary(self) -> Dict[str, Any]:
        """Get a summary of all stages and their ranges."""
        return {
            "stages": [
                {
                    "name": stage.value,
                    "start_percent": range_info[0],
                    "end_percent": range_info[1],
                    "duration_percent": range_info[1] - range_info[0]
                }
                for stage, range_info in self.STAGE_RANGES.items()
            ],
            "total_stages": len(self.STAGE_RANGES),
            "current_stage": self.current_stage.value if self.current_stage else None
        }

# Status configuration
STATUS_CONFIG = {
    "queued": {
        "icon": "🔵",
        "color": "#fd7e14",  # warning-orange from app.py
        "label": "Queued",
        "css_class": "status-pending"
    },
    "running": {
        "icon": "🟡", 
        "color": "#26a69a",  # accent-teal from app.py
        "label": "Running",
        "css_class": "status-active"
    },
    "completed": {
        "icon": "🟢",
        "color": "#28a745",  # success-green from app.py
        "label": "Completed", 
        "css_class": "status-active"
    },
    "failed": {
        "icon": "🔴",
        "color": "#dc3545",  # danger-red from app.py
        "label": "Failed",
        "css_class": "status-error"
    },
    "cancelled": {
        "icon": "⚪",
        "color": "#6c757d",  # medium-gray equivalent
        "label": "Cancelled",
        "css_class": "status-error"
    }
}

def get_job_status_icon(status: str) -> str:
    """Get the icon for a job status."""
    return STATUS_CONFIG.get(status.lower(), {"icon": "❓"})["icon"]

def get_job_status_color(status: str) -> str:
    """Get the color for a job status."""
    return STATUS_CONFIG.get(status.lower(), {"color": "#6c757d"})["color"]

def get_job_status_label(status: str) -> str:
    """Get the human-readable label for a job status."""
    return STATUS_CONFIG.get(status.lower(), {"label": status.title()})["label"]

def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"

def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"

def estimate_time_remaining(job_status: Dict[str, Any]) -> str:
    """Estimate remaining time based on progress and elapsed time."""
    try:
        progress = job_status.get('progress', 0)
        if progress <= 0:
            return "Calculating..."
        
        # Get elapsed time
        created_at = job_status.get('created_at')
        if created_at:
            if isinstance(created_at, str):
                created_time = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            else:
                created_time = created_at
            
            elapsed = (datetime.now() - created_time.replace(tzinfo=None)).total_seconds()
            
            # Estimate total time based on progress
            if progress > 5:  # Avoid division by very small numbers
                estimated_total = elapsed * (100 / progress)
                remaining = max(0, estimated_total - elapsed)
                return format_duration(remaining)
        
        return "Unknown"
    except Exception:
        return "Unknown"

def get_historical_avg_time(job_type: str) -> Optional[float]:
    """Get average completion time for job type from historical data."""
    try:
        history = get_job_history(status_filter="completed")
        type_jobs = [job for job in history if job.job_type == job_type and job.total_time > 0]
        
        if type_jobs:
            avg_time = sum(job.total_time for job in type_jobs) / len(type_jobs)
            return avg_time
        return None
    except Exception:
        return None

def job_status_badge(status: str) -> str:
    """Generate HTML for a status badge."""
    config = STATUS_CONFIG.get(status.lower(), STATUS_CONFIG["failed"])
    return f"""
    <span style="
        background-color: {config['color']}15;
        color: {config['color']};
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8em;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 4px;
    ">
        {config['icon']} {config['label']}
    </span>
    """

def job_progress_bar(job_id: str, job_status: Dict[str, Any], show_estimates: bool = True) -> None:
    """Display a progress bar for a specific job."""
    progress = job_status.get('progress', 0)
    status = job_status.get('status', 'unknown')
    message = job_status.get('status_message', '')

    # Progress bar
    if status == 'running':
        st.progress(progress / 100.0, text=f"{progress:.1f}% - {message}")
    elif status == 'completed':
        st.progress(1.0, text="✅ Completed")
    elif status == 'failed':
        st.progress(0.0, text="❌ Failed")
    elif status == 'cancelled':
        st.progress(0.0, text="🛑 Cancelled")
    else:
        st.progress(0.0, text="⏳ Queued")

    # Time estimates
    if show_estimates and status == 'running' and progress > 0:
        col1, col2 = st.columns(2)
        with col1:
            time_remaining = estimate_time_remaining(job_status)
            st.caption(f"⏱️ Est. {time_remaining} remaining")

        with col2:
            # Show historical average if available
            job_type = job_status.get('job_type', '')
            avg_time = get_historical_avg_time(job_type)
            if avg_time:
                st.caption(f"📊 Avg: {format_duration(avg_time)}")


def enhanced_job_progress_bar(job_id: str, job_status: Dict[str, Any],
                             enhanced_progress: Optional[EnhancedProgressInfo] = None,
                             show_estimates: bool = True) -> None:
    """Display an enhanced progress bar with stage-aware information."""
    progress = job_status.get('progress', 0)
    status = job_status.get('status', 'unknown')
    message = job_status.get('status_message', '')

    # Use enhanced progress info if available
    if enhanced_progress and status == 'running':
        # Main progress bar with enhanced message
        progress_text = f"{enhanced_progress.overall_progress:.1f}% - {enhanced_progress.message}"
        st.progress(enhanced_progress.overall_progress / 100.0, text=progress_text)

        # Stage information bar
        stage_info_col1, stage_info_col2, stage_info_col3 = st.columns([2, 1, 1])

        with stage_info_col1:
            # Stage progress indicator
            stage_text = f"🔄 {enhanced_progress.stage.title()}"
            if enhanced_progress.total_items > 1:
                stage_text += f" ({enhanced_progress.current_item}/{enhanced_progress.total_items})"
            st.caption(stage_text)

        with stage_info_col2:
            # Hardware mode indicator with GPU queue support
            hardware_mode = enhanced_progress.hardware_mode.lower()
            if hardware_mode == "gpu":
                hardware_icon = "🚀"
                hardware_text = "GPU"
            elif hardware_mode == "queued":
                hardware_icon = "⏳"
                hardware_text = "Queued"
            else:
                hardware_icon = "💻"
                hardware_text = "CPU"
            st.caption(f"{hardware_icon} {hardware_text}")

        with stage_info_col3:
            # Stage progress percentage
            st.caption(f"Stage: {enhanced_progress.stage_progress:.0f}%")

        # Enhanced time estimates
        if show_estimates:
            est_col1, est_col2, est_col3 = st.columns(3)

            with est_col1:
                if enhanced_progress.eta_seconds:
                    st.caption(f"⏱️ ETA: {format_duration(enhanced_progress.eta_seconds)}")
                else:
                    time_remaining = estimate_time_remaining(job_status)
                    st.caption(f"⏱️ Est: {time_remaining}")

            with est_col2:
                # Show historical average if available
                job_type = job_status.get('job_type', '')
                avg_time = get_historical_avg_time(job_type)
                if avg_time:
                    st.caption(f"📊 Avg: {format_duration(avg_time)}")

            with est_col3:
                # Memory usage if available
                if enhanced_progress.memory_usage_mb:
                    memory_gb = enhanced_progress.memory_usage_mb / 1024
                    st.caption(f"💾 {memory_gb:.1f}GB")

    else:
        # Fallback to regular progress bar
        if status == 'running':
            st.progress(progress / 100.0, text=f"{progress:.1f}% - {message}")
        elif status == 'completed':
            st.progress(1.0, text="✅ Completed")
        elif status == 'failed':
            st.progress(0.0, text="❌ Failed")
        elif status == 'cancelled':
            st.progress(0.0, text="🛑 Cancelled")
        else:
            st.progress(0.0, text="⏳ Queued")

        # Basic time estimates
        if show_estimates and status == 'running' and progress > 0:
            col1, col2 = st.columns(2)
            with col1:
                time_remaining = estimate_time_remaining(job_status)
                st.caption(f"⏱️ Est. {time_remaining} remaining")

            with col2:
                job_type = job_status.get('job_type', '')
                avg_time = get_historical_avg_time(job_type)
                if avg_time:
                    st.caption(f"📊 Avg: {format_duration(avg_time)}")


def display_stage_progress_chart(progress_mapper: SegmentationProgressMapper) -> None:
    """Display a visual chart of all segmentation stages and current progress."""
    stage_summary = progress_mapper.get_stage_summary()

    if not stage_summary["stages"]:
        return

    # Create a visual representation of stages
    st.subheader("📊 Segmentation Stages Overview")

    stages = stage_summary["stages"]
    current_stage = stage_summary["current_stage"]

    # Create columns for each stage
    cols = st.columns(len(stages))

    for i, (col, stage_info) in enumerate(zip(cols, stages)):
        with col:
            stage_name = stage_info["name"]
            start_pct = stage_info["start_percent"]
            end_pct = stage_info["end_percent"]
            duration_pct = stage_info["duration_percent"]

            # Highlight current stage
            if stage_name == current_stage:
                st.markdown(f"**🔄 {stage_name.title()}**")
                st.progress(0.5, text=f"{start_pct}-{end_pct}%")
            else:
                st.markdown(f"{stage_name.title()}")
                st.caption(f"{start_pct}-{end_pct}% ({duration_pct}%)")

    # Show current stage details
    if current_stage:
        st.info(f"Currently in **{current_stage.title()}** stage")

def job_control_buttons(job_id: str, job_status: Dict[str, Any]) -> None:
    """Display control buttons for a job."""
    status = job_status.get('status', 'unknown')
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        # Cancel button
        if status in ['queued', 'running']:
            if st.button("🛑 Cancel", key=f"cancel_{job_id}", help="Cancel this job"):
                try:
                    job_manager = get_job_manager()
                    result = job_manager.cancel_job(job_id)
                    if result:
                        st.success("Job cancellation requested")
                        st.rerun()
                    else:
                        st.error("Failed to cancel job")
                except Exception as e:
                    st.error(f"Error cancelling job: {e}")
    
    with col2:
        # Retry button for failed jobs
        if status in ['failed', 'cancelled']:
            if st.button("🔄 Retry", key=f"retry_{job_id}", help="Retry this job"):
                try:
                    # Get original job parameters
                    params = job_status.get('params', {})
                    job_type = job_status.get('job_type', 'test')
                    
                    job_manager = get_job_manager()
                    new_job_id = job_manager.submit_job(job_type, params)
                    st.success(f"Job resubmitted as {new_job_id}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error retrying job: {e}")
    
    with col3:
        # Job details
        if st.button("ℹ️ Details", key=f"details_{job_id}", help="Show job details"):
            st.session_state[f"show_details_{job_id}"] = not st.session_state.get(f"show_details_{job_id}", False)

def display_job_details(job_id: str, job_status: Dict[str, Any]) -> None:
    """Display detailed information about a job."""
    if not st.session_state.get(f"show_details_{job_id}", False):
        return
    
    with st.expander(f"📋 Job Details: {job_id}", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Job Information**")
            st.write(f"**Type:** {job_status.get('job_type', 'N/A')}")
            st.write(f"**Status:** {get_job_status_label(job_status.get('status', 'unknown'))}")
            st.write(f"**Progress:** {job_status.get('progress', 0):.1f}%")
            
            created_at = job_status.get('created_at')
            if created_at:
                if isinstance(created_at, str):
                    try:
                        created_time = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                        st.write(f"**Created:** {created_time.strftime('%Y-%m-%d %H:%M:%S')}")
                    except:
                        st.write(f"**Created:** {created_at}")
                else:
                    st.write(f"**Created:** {created_at.strftime('%Y-%m-%d %H:%M:%S')}")
        
        with col2:
            st.write("**Status Information**")
            message = job_status.get('status_message', 'No message')
            st.write(f"**Message:** {message}")
            
            error_message = job_status.get('error_message', '')
            if error_message:
                st.write(f"**Error:** {error_message}")
            
            # Parameters
            params = job_status.get('params', {})
            if params:
                st.write("**Parameters:**")
                for key, value in params.items():
                    st.write(f"  • {key}: {value}")

def job_queue_dashboard() -> None:
    """Main job queue dashboard component."""
    st.subheader("🚀 Job Queue Dashboard")
    
    try:
        # Get job manager and current jobs
        job_manager = get_job_manager()
        
        # Get queue info
        queue_info = job_manager.get_job_queue_info()
        
        # Display queue statistics with GPU information
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("Total Jobs", queue_info.get('total_jobs', 0))

        with col2:
            st.metric("Active Workers", f"{queue_info.get('active_workers', 0)}/{queue_info.get('max_workers', 2)}")

        with col3:
            st.metric("Queue Size", queue_info.get('queue_size', 0))

        with col4:
            # Count running jobs
            status_counts = queue_info.get('status_counts', {})
            running_jobs = status_counts.get('running', 0)
            st.metric("Running Jobs", running_jobs)

        with col5:
            # GPU status information
            try:
                gpu_status = job_manager.get_gpu_status()
                gpu_available = gpu_status.get('gpu_available', False)
                active_gpu_jobs = gpu_status.get('active_gpu_jobs', 0)
                gpu_queue_length = gpu_status.get('queue_length', 0)

                if gpu_available:
                    gpu_label = f"🚀 GPU: {active_gpu_jobs}/1"
                    if gpu_queue_length > 0:
                        gpu_label += f" (+{gpu_queue_length})"
                else:
                    gpu_label = "💻 CPU Only"

                st.metric("GPU Status", gpu_label)
            except Exception:
                st.metric("GPU Status", "Unknown")
        
        # Get all jobs with status
        jobs_data = []
        
        # Get current jobs from job manager
        with job_manager._jobs_lock:
            for job_id, job_info in job_manager._jobs.items():
                job_data = {
                    'id': job_id,
                    'type': job_info.job_type,
                    'status': job_info.status.value,
                    'progress': job_info.progress,
                    'message': job_info.status_message,
                    'created_at': job_info.created_at,
                    'params': job_info.params,
                    'error_message': job_info.error_message
                }
                jobs_data.append(job_data)
        
        # Add recent completed jobs from persistence
        try:
            recent_jobs = get_job_history(limit=10)
            for job_metadata in recent_jobs:
                # Avoid duplicates
                if not any(job['id'] == job_metadata.id for job in jobs_data):
                    job_data = {
                        'id': job_metadata.id,
                        'type': job_metadata.job_type,
                        'status': job_metadata.status,
                        'progress': job_metadata.progress,
                        'message': job_metadata.status_message,
                        'created_at': job_metadata.created_at,
                        'params': job_metadata.parameters,
                        'error_message': job_metadata.error_message
                    }
                    jobs_data.append(job_data)
        except Exception as e:
            st.warning(f"Could not load job history: {e}")
        
        if not jobs_data:
            st.info("No jobs found. Submit a job to see it appear here!")
            return
        
        # Sort jobs by creation time (newest first)
        jobs_data.sort(key=lambda x: x['created_at'] if x['created_at'] else datetime.min, reverse=True)
        
        # Display jobs table
        st.write("### Current Jobs")
        
        for job_data in jobs_data[:20]:  # Show up to 20 most recent jobs
            job_id = job_data['id']
            
            with st.container():
                # Job header row
                col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 2])
                
                with col1:
                    st.write(f"**{job_id[:8]}...** ({job_data['type']})")
                
                with col2:
                    st.markdown(job_status_badge(job_data['status']), unsafe_allow_html=True)
                
                with col3:
                    if job_data['status'] == 'running':
                        st.write(f"{job_data['progress']:.1f}%")
                    else:
                        st.write("—")
                
                with col4:
                    created_at = job_data['created_at']
                    if isinstance(created_at, str):
                        try:
                            created_time = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                            st.write(created_time.strftime('%H:%M:%S'))
                        except:
                            st.write("—")
                    elif hasattr(created_at, 'strftime'):
                        st.write(created_at.strftime('%H:%M:%S'))
                    else:
                        st.write("—")
                
                with col5:
                    # Control buttons
                    job_control_buttons(job_id, job_data)
                
                # Enhanced progress bar for running jobs
                if job_data['status'] == 'running':
                    try:
                        # Try to get enhanced progress information
                        job_manager = get_job_manager()
                        enhanced_status = job_manager.get_enhanced_job_status(job_id)

                        if enhanced_status and enhanced_status.get('current_stage'):
                            # Create EnhancedProgressInfo from job status
                            enhanced_progress = EnhancedProgressInfo(
                                overall_progress=enhanced_status.get('progress', 0),
                                stage=enhanced_status.get('current_stage', 'unknown'),
                                stage_progress=enhanced_status.get('stage_progress', 0),
                                current_item=enhanced_status.get('stage_metadata', {}).get('current_slice', 0),
                                total_items=enhanced_status.get('stage_metadata', {}).get('total_slices', 1),
                                message=enhanced_status.get('status_message', ''),
                                hardware_mode=enhanced_status.get('hardware_mode', 'Unknown'),
                                eta_seconds=enhanced_status.get('eta_seconds'),
                                metadata=enhanced_status.get('stage_metadata', {})
                            )
                            enhanced_job_progress_bar(job_id, job_data, enhanced_progress)
                        else:
                            # Fallback to regular progress bar
                            job_progress_bar(job_id, job_data)
                    except Exception as e:
                        # If enhanced progress fails, use regular progress bar
                        job_progress_bar(job_id, job_data)
                
                # Job details if expanded
                display_job_details(job_id, job_data)
                
                st.divider()
    
    except Exception as e:
        st.error(f"Error loading job queue: {e}")
        st.exception(e)

def auto_refresh_container(refresh_interval: int = 2) -> None:
    """Auto-refresh container for real-time updates."""
    # Initialize refresh state
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = time.time()
    
    if 'auto_refresh_enabled' not in st.session_state:
        st.session_state.auto_refresh_enabled = True
    
    # Auto-refresh controls
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("🔄 Refresh Now"):
            st.rerun()
    
    with col2:
        auto_refresh = st.checkbox("Auto-refresh", value=st.session_state.auto_refresh_enabled)
        st.session_state.auto_refresh_enabled = auto_refresh
    
    with col3:
        if st.session_state.auto_refresh_enabled:
            current_time = time.time()
            time_since_refresh = current_time - st.session_state.last_refresh
            
            if time_since_refresh >= refresh_interval:
                try:
                    # Check if there are active jobs
                    job_manager = get_job_manager()
                    queue_info = job_manager.get_job_queue_info()
                    
                    running_jobs = queue_info.get('status_counts', {}).get('running', 0)
                    queued_jobs = queue_info.get('queue_size', 0)
                    
                    # Only auto-refresh if there are active jobs
                    if running_jobs > 0 or queued_jobs > 0:
                        st.session_state.last_refresh = current_time
                        st.rerun()
                    
                except Exception:
                    pass  # Silently handle refresh errors
            
            # Show next refresh countdown
            next_refresh = max(0, refresh_interval - time_since_refresh)
            st.caption(f"Next refresh: {next_refresh:.0f}s")

def job_history_viewer() -> None:
    """Display historical job data with filtering."""
    st.subheader("📊 Job History")
    
    try:
        # Get job history
        history = get_job_history(limit=100)
        
        if not history:
            st.info("No job history available.")
            return
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Status filter
            status_options = ['All'] + list(set(job.status for job in history))
            selected_status = st.selectbox("Filter by Status", status_options)
        
        with col2:
            # Job type filter
            type_options = ['All'] + list(set(job.job_type for job in history))
            selected_type = st.selectbox("Filter by Type", type_options)
        
        with col3:
            # Time range
            time_range = st.selectbox("Time Range", ["Last 24h", "Last 7 days", "Last 30 days", "All time"])
        
        # Apply filters
        filtered_history = history
        
        if selected_status != 'All':
            filtered_history = [job for job in filtered_history if job.status == selected_status]
        
        if selected_type != 'All':
            filtered_history = [job for job in filtered_history if job.job_type == selected_type]
        
        # Time range filter
        if time_range != "All time":
            now = datetime.now()
            if time_range == "Last 24h":
                cutoff = now - timedelta(hours=24)
            elif time_range == "Last 7 days":
                cutoff = now - timedelta(days=7)
            elif time_range == "Last 30 days":
                cutoff = now - timedelta(days=30)
            
            filtered_history = [
                job for job in filtered_history 
                if datetime.fromisoformat(job.created_at.replace('Z', '+00:00')).replace(tzinfo=None) > cutoff
            ]
        
        # Display statistics
        if filtered_history:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Jobs", len(filtered_history))
            
            with col2:
                completed = len([job for job in filtered_history if job.status == 'completed'])
                st.metric("Completed", completed)
            
            with col3:
                failed = len([job for job in filtered_history if job.status == 'failed'])
                st.metric("Failed", failed)
            
            with col4:
                if completed > 0:
                    avg_time = sum(job.total_time for job in filtered_history if job.status == 'completed' and job.total_time > 0) / completed
                    st.metric("Avg Time", format_duration(avg_time))
        
        # Display job table
        if filtered_history:
            # Convert to DataFrame for better display
            df_data = []
            for job in filtered_history[:50]:  # Limit to 50 for performance
                df_data.append({
                    'Job ID': job.id[:12] + '...',
                    'Type': job.job_type,
                    'Status': job.status,
                    'Progress': f"{job.progress:.1f}%",
                    'Duration': format_duration(job.total_time) if job.total_time > 0 else '—',
                    'Created': datetime.fromisoformat(job.created_at.replace('Z', '+00:00')).strftime('%m/%d %H:%M'),
                    'Message': job.status_message[:50] + '...' if len(job.status_message) > 50 else job.status_message
                })
            
            df = pd.DataFrame(df_data)
            st.dataframe(df, width='stretch', hide_index=True)
        
        else:
            st.info("No jobs match the selected filters.")
    
    except Exception as e:
        st.error(f"Error loading job history: {e}")

# Test job submission for demo purposes
def test_job_controls() -> None:
    """Controls for submitting test jobs."""
    st.subheader("🧪 Test Job Controls")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Submit Quick Test Job"):
            try:
                job_manager = get_job_manager()
                job_id = job_manager.submit_job("test", {"duration": 5, "steps": 10})
                st.success(f"Test job submitted: {job_id[:8]}...")
                st.rerun()
            except Exception as e:
                st.error(f"Error submitting job: {e}")
    
    with col2:
        if st.button("Submit Long Test Job"):
            try:
                job_manager = get_job_manager()
                job_id = job_manager.submit_job("test", {"duration": 30, "steps": 100})
                st.success(f"Long job submitted: {job_id[:8]}...")
                st.rerun()
            except Exception as e:
                st.error(f"Error submitting job: {e}")
    
    with col3:
        if st.button("Submit Multiple Jobs"):
            try:
                job_manager = get_job_manager()
                for i in range(3):
                    job_id = job_manager.submit_job("test", {"duration": 10 + i*5, "steps": 20 + i*10})
                st.success("3 test jobs submitted!")
                st.rerun()
            except Exception as e:
                st.error(f"Error submitting jobs: {e}")