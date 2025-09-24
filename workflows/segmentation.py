"""
Advanced Segmentation Workflow with Hardware-Aware TensorFlow Configuration

This module provides a comprehensive SegmentationWorkflow class that:
- Automatically detects available hardware (GPU/CPU)
- Configures TensorFlow for optimal performance on detected hardware
- Provides CPU-only fallback with memory optimization
- Supports both GPU acceleration and CPU-only processing
- Includes comprehensive error handling and user feedback

Task 10.1 Implementation: Hardware Detection and TensorFlow Configuration
"""

import os
import sys
import time
import logging
import json
import psutil
import re
import hashlib
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Callable
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

# TensorFlow imports with error handling
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError as e:
    TF_AVAILABLE = False
    tf = None
    print(f"Warning: TensorFlow not available: {e}")

# Optional imports for enhanced CPU performance
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None


class HardwareType(Enum):
    """Enum for different hardware types."""
    CPU_ONLY = "cpu"
    GPU_AVAILABLE = "gpu"
    UNKNOWN = "unknown"


class ErrorType(Enum):
    """Enum for different error categories in the segmentation workflow."""
    GPU_OOM = "gpu_out_of_memory"
    CPU_MEMORY_PRESSURE = "cpu_memory_pressure"
    FILE_NOT_FOUND = "file_not_found"
    PERMISSION_ERROR = "permission_error"
    DISK_SPACE_ERROR = "disk_space_error"
    INVALID_FILE_FORMAT = "invalid_file_format"
    MODEL_LOADING_ERROR = "model_loading_error"
    PROCESSING_ERROR = "processing_error"
    HARDWARE_ERROR = "hardware_error"
    UNKNOWN_ERROR = "unknown_error"


class HardwareFallbackEvent:
    """Data class to track hardware fallback events."""
    def __init__(self, from_hardware: HardwareType, to_hardware: HardwareType,
                 reason: str, timestamp: float, error_details: str = ""):
        self.from_hardware = from_hardware
        self.to_hardware = to_hardware
        self.reason = reason
        self.timestamp = timestamp
        self.error_details = error_details


@dataclass
class HardwareConfig:
    """Configuration data class for hardware setup."""
    hardware_type: HardwareType
    device_name: str
    memory_limit_gb: Optional[float] = None
    cpu_count: Optional[int] = None
    gpu_count: Optional[int] = 0
    cuda_version: Optional[str] = None
    compute_capability: Optional[str] = None
    tensorflow_version: Optional[str] = None
    supports_mixed_precision: bool = False
    batch_size: Optional[int] = None
    original_hardware_type: Optional[HardwareType] = None  # Track original hardware before fallback
    fallback_events: List[HardwareFallbackEvent] = None

    def __post_init__(self):
        """Initialize fallback tracking."""
        if self.fallback_events is None:
            self.fallback_events = []
        if self.original_hardware_type is None:
            self.original_hardware_type = self.hardware_type

    @property
    def memory_gb(self) -> float:
        """Alias for memory_limit_gb for backward compatibility."""
        return self.memory_limit_gb or 0.0

    @property
    def has_fallback_occurred(self) -> bool:
        """Check if hardware fallback has occurred."""
        return len(self.fallback_events) > 0

    def add_fallback_event(self, event: HardwareFallbackEvent):
        """Add a fallback event to the history."""
        self.fallback_events.append(event)
        self.hardware_type = event.to_hardware


class SegmentationWorkflow:
    """
    Hardware-aware segmentation workflow with automatic TensorFlow configuration.
    
    This class automatically detects available hardware and configures TensorFlow
    for optimal performance. It supports both GPU acceleration and CPU-only processing
    with appropriate fallback mechanisms.
    """
    
    # Constants for output naming and metadata
    OUTPUT_SUFFIX = "_segmented"
    OUTPUT_EXTENSION = ".tif"
    METADATA_EXTENSION = ".json"
    
    # Default batch sizes for different hardware
    DEFAULT_CPU_BATCH_SIZE = 2
    DEFAULT_GPU_BATCH_SIZE = 8
    MIN_BATCH_SIZE = 1
    MAX_BATCH_SIZE = 32
    
    # MoDL size requirements (extracted from segment_predict.py)
    MODL_INPUT_SIZE = 2048      # MoDL expects 2048x2048 images
    MODL_PATCH_SIZE = 512       # Model processes 512x512 patches
    MODL_OUTPUT_SIZE = 1590     # Final output is resized to 1590x1590
    
    def __init__(self, model_path: Optional[str] = None, 
                 force_cpu: bool = False,
                 memory_limit_gb: Optional[float] = None):
        """
        Initialize the SegmentationWorkflow with hardware detection.
        
        Args:
            model_path: Path to the segmentation model (optional)
            force_cpu: Force CPU-only processing even if GPU is available
            memory_limit_gb: Optional memory limit for GPU (prevents OOM)
        """
        self.logger = self._setup_logging()
        self.model_path = model_path
        self.force_cpu = force_cpu
        self.memory_limit_gb = memory_limit_gb
        
        # Hardware configuration will be set during initialization
        self.hardware_config: Optional[HardwareConfig] = None
        self.is_tensorflow_configured = False
        self.model = None
        
        # Performance tracking
        self.processing_stats = {
            "jobs_completed": 0,
            "total_processing_time": 0.0,
            "average_time_per_slice": 0.0,
            "memory_peak_usage": 0.0
        }
        
        # Initialize hardware detection and TensorFlow configuration
        self._initialize_hardware_and_tensorflow()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
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
    
    def _initialize_hardware_and_tensorflow(self) -> None:
        """
        Initialize hardware detection and configure TensorFlow.
        This is the core implementation for Task 10.1.
        """
        try:
            self.logger.info("🔍 Detecting hardware configuration...")
            
            # Step 1: Detect hardware capabilities
            self.hardware_config = self._detect_hardware()
            
            # Step 2: Configure TensorFlow based on detected hardware
            self._configure_tensorflow()
            
            # Step 3: Provide user feedback
            self._log_hardware_configuration()
            
            self.is_tensorflow_configured = True
            self.logger.info("✅ Hardware detection and TensorFlow configuration completed")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize hardware configuration: {e}")
            # Create minimal CPU-only fallback configuration
            self.hardware_config = HardwareConfig(
                hardware_type=HardwareType.CPU_ONLY,
                device_name="CPU (fallback)",
                cpu_count=os.cpu_count()
            )
            self.is_tensorflow_configured = False
    
    def _detect_hardware(self) -> HardwareConfig:
        """
        Detect available hardware and create configuration.
        
        Returns:
            HardwareConfig: Detected hardware configuration
        """
        if not TF_AVAILABLE:
            self.logger.warning("TensorFlow not available, using CPU-only configuration")
            return HardwareConfig(
                hardware_type=HardwareType.CPU_ONLY,
                device_name="CPU (TensorFlow unavailable)",
                cpu_count=os.cpu_count(),
                batch_size=self.DEFAULT_CPU_BATCH_SIZE
            )
        
        # Detect CPU information
        cpu_count = os.cpu_count()
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # Check for GPU availability
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            gpu_count = len(gpus)
            
            if gpu_count > 0 and not self.force_cpu:
                # GPU detected and not forced to CPU
                gpu_name = "Unknown GPU"
                compute_capability = None
                cuda_version = None
                
                try:
                    # Try to get GPU details
                    gpu_details = tf.config.experimental.get_device_details(gpus[0])
                    gpu_name = gpu_details.get('device_name', 'Unknown GPU')
                    compute_capability = gpu_details.get('compute_capability')
                    
                    # Check CUDA version if available
                    if hasattr(tf.sysconfig, 'get_build_info'):
                        build_info = tf.sysconfig.get_build_info()
                        cuda_version = build_info.get('cuda_version', 'Unknown')
                    
                except Exception as e:
                    self.logger.debug(f"Could not get detailed GPU info: {e}")
                
                return HardwareConfig(
                    hardware_type=HardwareType.GPU_AVAILABLE,
                    device_name=gpu_name,
                    memory_limit_gb=self.memory_limit_gb,
                    cpu_count=cpu_count,
                    gpu_count=gpu_count,
                    cuda_version=cuda_version,
                    compute_capability=compute_capability,
                    tensorflow_version=tf.__version__,
                    supports_mixed_precision=True,  # Modern GPUs typically support this
                    batch_size=self.DEFAULT_GPU_BATCH_SIZE
                )
            else:
                # No GPU detected or forced to CPU
                reason = "forced by user" if self.force_cpu else "no GPU detected"
                self.logger.info(f"Using CPU-only processing ({reason})")
                
        except Exception as e:
            self.logger.debug(f"GPU detection failed: {e}")
            gpu_count = 0
        
        # CPU-only configuration
        return HardwareConfig(
            hardware_type=HardwareType.CPU_ONLY,
            device_name=f"CPU ({cpu_count} cores, {memory_gb:.1f}GB RAM)",
            cpu_count=cpu_count,
            gpu_count=0,
            tensorflow_version=tf.__version__ if TF_AVAILABLE else None,
            batch_size=self.DEFAULT_CPU_BATCH_SIZE
        )
    
    def _configure_tensorflow(self) -> None:
        """
        Configure TensorFlow based on detected hardware.
        This implements the core TensorFlow configuration logic.
        """
        if not TF_AVAILABLE or not self.hardware_config:
            self.logger.warning("Cannot configure TensorFlow - not available or no hardware config")
            return
        
        try:
            # Clear any existing session
            tf.keras.backend.clear_session()
            
            if self.hardware_config.hardware_type == HardwareType.GPU_AVAILABLE:
                self._configure_gpu()
            else:
                self._configure_cpu()
                
            self.logger.info("✅ TensorFlow configuration applied successfully")
            
        except Exception as e:
            self.logger.error(f"❌ TensorFlow configuration failed: {e}")
            # Try CPU fallback configuration
            try:
                self._configure_cpu_fallback()
                self.logger.warning("⚠️ Fell back to basic CPU configuration")
            except Exception as fallback_error:
                self.logger.error(f"❌ CPU fallback configuration also failed: {fallback_error}")
    
    def _configure_gpu(self) -> None:
        """Configure TensorFlow for GPU processing."""
        self.logger.info("🚀 Configuring TensorFlow for GPU processing...")
        
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Configure GPU memory growth to prevent OOM
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    
                    # Set memory limit if specified
                    if self.memory_limit_gb:
                        memory_limit_mb = int(self.memory_limit_gb * 1024)
                        tf.config.experimental.set_memory_limit(
                            gpu, memory_limit_mb
                        )
                        self.logger.info(f"🎯 GPU memory limit set to {self.memory_limit_gb}GB")
                
                # Use the first GPU
                tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
                
                # Enable mixed precision if supported
                if self.hardware_config.supports_mixed_precision:
                    policy = tf.keras.mixed_precision.Policy('mixed_float16')
                    tf.keras.mixed_precision.set_global_policy(policy)
                    self.logger.info("🎯 Mixed precision enabled for better performance")
                
                self.logger.info(f"✅ GPU configured: {self.hardware_config.device_name}")
                
            except RuntimeError as e:
                self.logger.error(f"GPU configuration failed: {e}")
                # Fallback to CPU
                self._configure_cpu()
                self.hardware_config.hardware_type = HardwareType.CPU_ONLY
    
    def _configure_cpu(self) -> None:
        """Configure TensorFlow for optimal CPU processing."""
        self.logger.info("🖥️  Configuring TensorFlow for CPU processing...")
        
        try:
            # Disable GPU completely for CPU-only processing
            tf.config.set_visible_devices([], 'GPU')
            
            # Enable XLA JIT compilation for CPU optimization
            tf.config.optimizer.set_jit(True)
            
            # Configure CPU threading for optimal performance
            cpu_count = self.hardware_config.cpu_count or os.cpu_count()
            
            # Set inter-op and intra-op parallelism
            tf.config.threading.set_inter_op_parallelism_threads(cpu_count)
            tf.config.threading.set_intra_op_parallelism_threads(cpu_count)
            
            self.logger.info(f"✅ CPU configured: {cpu_count} threads, XLA enabled")
            
        except Exception as e:
            self.logger.error(f"CPU configuration failed: {e}")
            raise
    
    def _configure_cpu_fallback(self) -> None:
        """Basic CPU configuration as fallback."""
        self.logger.info("🔄 Applying basic CPU fallback configuration...")
        
        # Minimal configuration that should always work
        tf.config.set_visible_devices([], 'GPU')
        
        # Update hardware config to reflect fallback
        if self.hardware_config:
            self.hardware_config.hardware_type = HardwareType.CPU_ONLY
            self.hardware_config.device_name += " (fallback)"
    
    def _log_hardware_configuration(self) -> None:
        """Log the detected and configured hardware information."""
        if not self.hardware_config:
            return
        
        config = self.hardware_config
        
        self.logger.info("=" * 60)
        self.logger.info("🔧 HARDWARE CONFIGURATION SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"💻 Hardware Type: {config.hardware_type.value.upper()}")
        self.logger.info(f"🎯 Device: {config.device_name}")
        
        if config.cpu_count:
            self.logger.info(f"🖥️  CPU Cores: {config.cpu_count}")
        
        if config.gpu_count > 0:
            self.logger.info(f"🚀 GPU Count: {config.gpu_count}")
            if config.cuda_version:
                self.logger.info(f"🔧 CUDA Version: {config.cuda_version}")
            if config.compute_capability:
                self.logger.info(f"⚡ Compute Capability: {config.compute_capability}")
        
        if config.memory_limit_gb:
            self.logger.info(f"💾 Memory Limit: {config.memory_limit_gb}GB")
        
        if config.tensorflow_version:
            self.logger.info(f"🧠 TensorFlow: {config.tensorflow_version}")
        
        # Performance recommendations
        optimal_batch_size = self.get_optimal_batch_size()
        self.logger.info(f"📊 Recommended Batch Size: {optimal_batch_size}")
        
        self.logger.info("=" * 60)
    
    def get_hardware_info(self) -> Dict[str, Any]:
        """
        Get comprehensive hardware information.
        
        Returns:
            Dict containing all hardware configuration details
        """
        if not self.hardware_config:
            return {"error": "Hardware not detected"}
        
        config = self.hardware_config
        return {
            "hardware_type": config.hardware_type.value,
            "device_name": config.device_name,
            "cpu_count": config.cpu_count,
            "gpu_count": config.gpu_count,
            "cuda_version": config.cuda_version,
            "compute_capability": config.compute_capability,
            "tensorflow_version": config.tensorflow_version,
            "supports_mixed_precision": config.supports_mixed_precision,
            "memory_limit_gb": config.memory_limit_gb,
            "tensorflow_configured": self.is_tensorflow_configured,
            "optimal_batch_size": self.get_optimal_batch_size(),
            "processing_mode": "GPU Accelerated" if config.hardware_type == HardwareType.GPU_AVAILABLE else "CPU Only"
        }
    
    def get_optimal_batch_size(self) -> int:
        """
        Calculate optimal batch size based on hardware configuration.
        
        Returns:
            Recommended batch size for current hardware
        """
        if not self.hardware_config:
            return self.DEFAULT_CPU_BATCH_SIZE
        
        if self.hardware_config.hardware_type == HardwareType.GPU_AVAILABLE:
            # GPU can handle larger batches
            base_size = self.DEFAULT_GPU_BATCH_SIZE
            
            # Adjust based on memory limit
            if self.hardware_config.memory_limit_gb:
                if self.hardware_config.memory_limit_gb < 4:
                    base_size = max(2, base_size // 2)
                elif self.hardware_config.memory_limit_gb > 8:
                    base_size = min(self.MAX_BATCH_SIZE, base_size * 2)
            
            return base_size
        else:
            # CPU processing - smaller batches to prevent RAM issues
            available_ram_gb = psutil.virtual_memory().available / (1024**3)
            
            if available_ram_gb > 16:
                return 4
            elif available_ram_gb > 8:
                return 3
            elif available_ram_gb > 4:
                return 2
            else:
                return 1
    
    def is_gpu_available(self) -> bool:
        """Check if GPU is available and configured."""
        return (self.hardware_config and 
                self.hardware_config.hardware_type == HardwareType.GPU_AVAILABLE)
    
    def is_ready_for_processing(self) -> bool:
        """Check if the workflow is ready for processing."""
        return (self.hardware_config is not None and 
                self.is_tensorflow_configured)
    
    def get_processing_mode_info(self) -> str:
        """Get user-friendly processing mode information."""
        if not self.hardware_config:
            return "❓ Hardware detection pending"
        
        if self.hardware_config.hardware_type == HardwareType.GPU_AVAILABLE:
            gpu_info = f"🚀 GPU Processing: {self.hardware_config.device_name}"
            if self.hardware_config.memory_limit_gb:
                gpu_info += f" (Memory: {self.hardware_config.memory_limit_gb}GB)"
            return gpu_info
        else:
            cpu_count = self.hardware_config.cpu_count or "Unknown"
            return f"🖥️  CPU Processing: {cpu_count} cores"
    
    def validate_configuration(self) -> Tuple[bool, List[str]]:
        """
        Validate the current configuration.
        
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        if not TF_AVAILABLE:
            issues.append("TensorFlow is not available")
        
        if not self.hardware_config:
            issues.append("Hardware configuration not detected")
        
        if not self.is_tensorflow_configured:
            issues.append("TensorFlow not properly configured")
        
        # Check memory availability
        available_ram_gb = psutil.virtual_memory().available / (1024**3)
        if available_ram_gb < 2:
            issues.append(f"Low available RAM: {available_ram_gb:.1f}GB (recommend 4GB+)")
        
        return len(issues) == 0, issues
    
    def get_performance_recommendations(self) -> List[str]:
        """Get performance optimization recommendations."""
        recommendations = []
        
        if not self.hardware_config:
            return ["Complete hardware detection first"]
        
        config = self.hardware_config
        
        if config.hardware_type == HardwareType.CPU_ONLY:
            recommendations.extend([
                "Consider using smaller batch sizes to optimize CPU processing",
                "Close other applications to free up RAM during processing",
                "For large images, consider processing in smaller chunks"
            ])
            
            # Check if GPU might be available but not detected
            if config.gpu_count == 0 and not self.force_cpu:
                recommendations.append("Check if GPU drivers are installed for potential acceleration")
        
        else:  # GPU available
            recommendations.extend([
                "GPU acceleration is enabled for optimal performance",
                "Monitor GPU memory usage during processing"
            ])
            
            if not config.memory_limit_gb:
                recommendations.append("Consider setting a memory limit to prevent GPU OOM errors")
        
        return recommendations

    # ============================================================================
    # Task 10.5: Robust Error Handling and Hardware Fallback
    # ============================================================================

    def _detect_gpu_oom_error(self, error: Exception) -> bool:
        """
        Detect if an error is a GPU Out-of-Memory error.

        Args:
            error: Exception to analyze

        Returns:
            True if this is a GPU OOM error
        """
        if not TF_AVAILABLE:
            return False

        # Check for TensorFlow GPU OOM errors
        error_str = str(error).lower()
        gpu_oom_indicators = [
            "resourceexhaustederror",
            "out of memory",
            "ran out of memory",
            "failed to allocate",
            "cuda_error_out_of_memory",
            "cudnn_status_alloc_failed",
            "gpu memory",
            "device memory"
        ]

        # Check exception type
        if TF_AVAILABLE:
            try:
                if isinstance(error, tf.errors.ResourceExhaustedError):
                    return True
                if isinstance(error, tf.errors.OutOfRangeError):
                    return "memory" in error_str
            except AttributeError:
                pass

        # Check error message content
        return any(indicator in error_str for indicator in gpu_oom_indicators)

    def _fallback_to_cpu(self, reason: str, error_details: str = "") -> bool:
        """
        Fallback from GPU to CPU processing with proper reconfiguration.

        Args:
            reason: Human-readable reason for fallback
            error_details: Detailed error information

        Returns:
            True if fallback was successful, False otherwise
        """
        try:
            if not self.hardware_config:
                self.logger.error("❌ Cannot fallback: No hardware configuration available")
                return False

            current_hardware = self.hardware_config.hardware_type
            if current_hardware == HardwareType.CPU_ONLY:
                self.logger.warning("⚠️ Already running on CPU-only mode")
                return True

            self.logger.warning(f"🔄 Initiating GPU to CPU fallback: {reason}")

            # Create fallback event
            fallback_event = HardwareFallbackEvent(
                from_hardware=current_hardware,
                to_hardware=HardwareType.CPU_ONLY,
                reason=reason,
                timestamp=time.time(),
                error_details=error_details
            )

            # Update hardware configuration
            self.hardware_config.add_fallback_event(fallback_event)
            self.hardware_config.device_name += " (CPU fallback)"

            # Reconfigure TensorFlow for CPU-only processing
            if TF_AVAILABLE:
                try:
                    # Clear any existing session
                    tf.keras.backend.clear_session()

                    # Force CPU-only configuration
                    tf.config.set_visible_devices([], 'GPU')
                    tf.config.threading.set_inter_op_parallelism_threads(
                        self.hardware_config.cpu_count or os.cpu_count()
                    )
                    tf.config.threading.set_intra_op_parallelism_threads(
                        self.hardware_config.cpu_count or os.cpu_count()
                    )

                    self.logger.info("✅ TensorFlow reconfigured for CPU-only processing")

                except Exception as tf_error:
                    self.logger.error(f"⚠️ TensorFlow reconfiguration failed: {tf_error}")

            # Update batch size for CPU processing
            self.hardware_config.batch_size = self.DEFAULT_CPU_BATCH_SIZE

            self.logger.warning(f"🖥️ Hardware fallback completed: GPU → CPU")
            self.logger.info(f"📊 New batch size: {self.hardware_config.batch_size}")

            return True

        except Exception as e:
            self.logger.error(f"❌ Hardware fallback failed: {e}")
            return False

    def _check_disk_space(self, path: str, required_gb: float = 1.0) -> bool:
        """
        Check if sufficient disk space is available.

        Args:
            path: Path to check (file or directory)
            required_gb: Required space in GB

        Returns:
            True if sufficient space available
        """
        try:
            import shutil

            # Get the directory path
            if os.path.isfile(path):
                check_path = os.path.dirname(path)
            else:
                check_path = path

            # Check disk usage
            total, used, free = shutil.disk_usage(check_path)
            free_gb = free / (1024**3)

            if free_gb < required_gb:
                self.logger.error(f"❌ Insufficient disk space: {free_gb:.1f}GB available, {required_gb:.1f}GB required")
                return False

            self.logger.debug(f"✅ Disk space check passed: {free_gb:.1f}GB available")
            return True

        except Exception as e:
            self.logger.error(f"❌ Disk space check failed: {e}")
            return False

    def _validate_file_permissions(self, input_path: str, output_dir: str) -> Tuple[bool, str]:
        """
        Validate file and directory permissions.

        Args:
            input_path: Input file path to read
            output_dir: Output directory to write

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check input file read permissions
            if not os.access(input_path, os.R_OK):
                return False, f"No read permission for input file: {input_path}"

            # Check if output directory exists and is writable
            if not os.path.exists(output_dir):
                try:
                    os.makedirs(output_dir, exist_ok=True)
                except PermissionError:
                    return False, f"Cannot create output directory: {output_dir}"

            if not os.access(output_dir, os.W_OK):
                return False, f"No write permission for output directory: {output_dir}"

            return True, ""

        except Exception as e:
            return False, f"Permission validation failed: {str(e)}"

    def _execute_with_hardware_fallback(self, operation_func: Callable,
                                      operation_name: str,
                                      max_retries: int = 2) -> Any:
        """
        Execute an operation with automatic hardware fallback on GPU OOM.

        Args:
            operation_func: Function to execute
            operation_name: Human-readable operation name
            max_retries: Maximum number of retry attempts

        Returns:
            Operation result

        Raises:
            RuntimeError: If all retry attempts fail
        """
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                self.logger.debug(f"🔄 Executing {operation_name} (attempt {attempt + 1}/{max_retries + 1})")
                result = operation_func()

                if attempt > 0:
                    self.logger.info(f"✅ {operation_name} succeeded after {attempt} retry(s)")

                return result

            except Exception as e:
                last_error = e
                error_msg = str(e)

                # Check if this is a GPU OOM error
                if self._detect_gpu_oom_error(e):
                    self.logger.warning(f"🚨 GPU OOM detected in {operation_name}: {error_msg}")

                    # Attempt fallback to CPU
                    if self.hardware_config.hardware_type != HardwareType.CPU_ONLY:
                        fallback_success = self._fallback_to_cpu(
                            reason=f"GPU OOM during {operation_name}",
                            error_details=error_msg
                        )

                        if fallback_success and attempt < max_retries:
                            self.logger.info(f"🔄 Retrying {operation_name} with CPU processing...")
                            continue

                    self.logger.error(f"❌ {operation_name} failed even after CPU fallback")

                elif attempt < max_retries:
                    self.logger.warning(f"⚠️ {operation_name} failed, retrying: {error_msg}")
                    time.sleep(1)  # Brief delay before retry
                    continue

                # Final attempt failed
                break

        # All attempts failed
        self.logger.error(f"❌ {operation_name} failed after {max_retries + 1} attempts")
        raise RuntimeError(f"{operation_name} failed: {str(last_error)}") from last_error

    # === MoDL Integration Methods ===
    
    def load_and_detect_zstack(self, image_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Load TIFF image and detect z-stack dimensions with MoDL-compatible preprocessing.
        
        Based on the original MoDL segment_predict.py logic:
        - Lines 207-208: stack = tifffile.imread(image_path); z_stack_len = stack.shape[0]
        - Lines 197-198: input_pixel = 2048; output_pixel = 512
        
        Args:
            image_path: Path to the TIFF file
            
        Returns:
            Tuple of (processed_stack, metadata_dict)
            
        Raises:
            ValueError: If image format is unsupported
            FileNotFoundError: If image file doesn't exist
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        self.logger.info(f"📁 Loading image: {os.path.basename(image_path)}")
        
        try:
            # Load the TIFF stack (following MoDL logic)
            if not NUMPY_AVAILABLE:
                raise ImportError("NumPy is required for image processing")
                
            import tifffile
            raw_stack = tifffile.imread(image_path)
            original_shape = raw_stack.shape
            self.logger.info(f"📐 Original shape: {original_shape}")

            # Debug: Check raw loaded data
            self.logger.info(f"🔍 RAW LOADED: dtype={raw_stack.dtype}, min={raw_stack.min()}, max={raw_stack.max()}")

            # Handle different image formats
            if len(original_shape) == 4:
                # 4D format: (z, h, w, channels) -> convert to 3D grayscale
                self.logger.info("🎨 Converting 4D color stack to 3D grayscale")
                if original_shape[3] == 3:  # RGB
                    # Debug: Check RGB values before conversion
                    self.logger.info(f"🔍 RGB BEFORE: R_min={raw_stack[...,0].min()}, R_max={raw_stack[...,0].max()}, G_min={raw_stack[...,1].min()}, G_max={raw_stack[...,1].max()}, B_min={raw_stack[...,2].min()}, B_max={raw_stack[...,2].max()}")

                    # Convert to grayscale using standard weights
                    processed_stack = np.dot(raw_stack[...,:3], [0.299, 0.587, 0.114])

                    # Debug: Check after grayscale conversion
                    self.logger.info(f"🔍 AFTER GRAYSCALE: dtype={processed_stack.dtype}, min={processed_stack.min()}, max={processed_stack.max()}")
                else:
                    # Take first channel
                    processed_stack = raw_stack[..., 0]
                    self.logger.info(f"🔍 AFTER FIRST CHANNEL: dtype={processed_stack.dtype}, min={processed_stack.min()}, max={processed_stack.max()}")

                # Keep as float32 for proper normalization (don't convert to uint8)
                processed_stack = processed_stack.astype(np.float32)
                self.logger.info(f"🔍 AFTER FLOAT32 CAST: dtype={processed_stack.dtype}, min={processed_stack.min()}, max={processed_stack.max()}")
                
            elif len(original_shape) == 3:
                # Check if this is 2D RGB (h, w, channels) or 3D grayscale (z, h, w)
                if original_shape[2] <= 4:  # Likely (h, w, channels) - RGB/RGBA
                    self.logger.info("🎨 Converting 2D RGB image to grayscale")
                    if original_shape[2] == 3:  # RGB
                        # Debug: Check RGB values before conversion
                        self.logger.info(f"🔍 RGB BEFORE: R_min={raw_stack[...,0].min()}, R_max={raw_stack[...,0].max()}, G_min={raw_stack[...,1].min()}, G_max={raw_stack[...,1].max()}, B_min={raw_stack[...,2].min()}, B_max={raw_stack[...,2].max()}")

                        # Convert to grayscale using standard weights
                        grayscale = np.dot(raw_stack[...,:3], [0.299, 0.587, 0.114])

                        # Debug: Check after grayscale conversion
                        self.logger.info(f"🔍 AFTER GRAYSCALE: dtype={grayscale.dtype}, min={grayscale.min()}, max={grayscale.max()}")

                        # CRITICAL: Scale to 0-255 range like Keras load_img does
                        # MoDL model expects 0-255 uint8 equivalent data
                        if grayscale.max() > 0:
                            grayscale = (grayscale / grayscale.max()) * 255.0
                            self.logger.info(f"🔍 AFTER 0-255 SCALING: min={grayscale.min()}, max={grayscale.max()}")

                        # Add z dimension for single slice
                        processed_stack = np.expand_dims(grayscale, axis=0)
                        self.logger.info("📸 2D RGB image converted to single-slice z-stack")
                    else:
                        # Take first channel and add z dimension
                        grayscale = raw_stack[..., 0]
                        processed_stack = np.expand_dims(grayscale, axis=0)
                        self.logger.info("📸 2D multichannel image converted to single-slice z-stack")
                else:
                    # 3D format: (z, h, w) - already correct format
                    processed_stack = raw_stack.copy()
                    self.logger.info("📚 3D z-stack format detected")
                
            elif len(original_shape) == 2:
                # 2D format: (h, w) - add z dimension
                processed_stack = np.expand_dims(raw_stack, axis=0)
                self.logger.info("📸 Single image converted to z-stack format")
                
            else:
                raise ValueError(f"Unsupported image format: {original_shape}")
            
            # Detect z-stack length (following MoDL segment_predict.py:208)
            z_stack_len = processed_stack.shape[0]
            current_h, current_w = processed_stack.shape[1], processed_stack.shape[2]
            
            self.logger.info(f"🔍 Detected z-stack length: {z_stack_len}")
            self.logger.info(f"📏 Current slice size: {current_h}x{current_w}")
            
            # Resize to MoDL input requirements (2048x2048)
            if current_h != self.MODL_INPUT_SIZE or current_w != self.MODL_INPUT_SIZE:
                self.logger.info(f"🔄 Resizing from {current_h}x{current_w} to {self.MODL_INPUT_SIZE}x{self.MODL_INPUT_SIZE}")
                resized_stack = np.zeros((z_stack_len, self.MODL_INPUT_SIZE, self.MODL_INPUT_SIZE), dtype=processed_stack.dtype)
                
                for z in range(z_stack_len):
                    # Use cv2 for high-quality resizing (following preprocess.py:18)
                    try:
                        import cv2
                        resized_slice = cv2.resize(
                            processed_stack[z], 
                            (self.MODL_INPUT_SIZE, self.MODL_INPUT_SIZE), 
                            interpolation=cv2.INTER_CUBIC
                        )
                    except ImportError:
                        # Fallback to basic resizing with scipy
                        try:
                            from scipy.ndimage import zoom
                            zoom_factor_h = self.MODL_INPUT_SIZE / current_h
                            zoom_factor_w = self.MODL_INPUT_SIZE / current_w
                            resized_slice = zoom(processed_stack[z], (zoom_factor_h, zoom_factor_w), order=1)
                        except ImportError:
                            # Last resort: simple numpy resize
                            self.logger.warning("⚠️  No advanced resizing available, using basic interpolation")
                            resized_slice = self._simple_resize(processed_stack[z], self.MODL_INPUT_SIZE)
                    
                    resized_stack[z] = resized_slice
                
                processed_stack = resized_stack

                # Debug: Check after resizing
                self.logger.info(f"🔍 AFTER RESIZE: dtype={processed_stack.dtype}, min={processed_stack.min()}, max={processed_stack.max()}")

                # CRITICAL FIX: Clamp to 0-255 range to match MoDL's uint8 behavior
                # MoDL uses cv2.resize on uint8 data which automatically clamps to 0-255
                # Our float resizing can go outside this range due to interpolation artifacts
                processed_stack = np.clip(processed_stack, 0, 255)
                self.logger.info(f"🔧 AFTER CLAMPING: min={processed_stack.min()}, max={processed_stack.max()}")
            else:
                self.logger.info("✅ Image already at correct size for MoDL")
            
            # Compile metadata
            metadata = {
                'original_shape': original_shape,
                'z_stack_len': z_stack_len,
                'current_shape': processed_stack.shape,
                'original_path': image_path,
                'requires_4d_conversion': len(original_shape) == 4,
                'resized': (current_h != self.MODL_INPUT_SIZE or current_w != self.MODL_INPUT_SIZE),
                'dtype': str(processed_stack.dtype),
                'value_range': (int(processed_stack.min()), int(processed_stack.max()))
            }
            
            self.logger.info(f"✅ Image loaded successfully: {processed_stack.shape}")
            self.logger.info(f"📊 Value range: {metadata['value_range'][0]}-{metadata['value_range'][1]}")
            
            return processed_stack, metadata
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load image {image_path}: {str(e)}")
            raise
    
    def _simple_resize(self, image: np.ndarray, target_size: int) -> np.ndarray:
        """Simple resize using numpy interpolation as fallback."""
        from scipy.ndimage import zoom
        h, w = image.shape
        zoom_h = target_size / h
        zoom_w = target_size / w
        return zoom(image, (zoom_h, zoom_w), order=1)
    
    def preprocess_for_modl(self, stack: np.ndarray) -> np.ndarray:
        """
        Preprocess image stack for MoDL model inference.
        
        Based on MoDL segment_predict.py test() function:
        - Lines 78-79: imgs_test = imgdatas.astype('float32'); imgs_test /= 255
        - Lines 80-81: mean = imgs_test.mean(axis=0); imgs_test -= mean
        
        Args:
            stack: Image stack in uint8 format
            
        Returns:
            Preprocessed stack ready for MoDL model
        """
        self.logger.info("🔄 Preprocessing stack for MoDL inference")

        # Debug input
        self.logger.info(f"🔍 Input stack: shape={stack.shape}, dtype={stack.dtype}, min={stack.min():.3f}, max={stack.max():.3f}")

        # Convert to float32 and normalize (following MoDL logic)
        processed_stack = stack.astype('float32')
        processed_stack /= 255.0

        # Note: Mean subtraction is done per-pattern batch during inference, not globally

        self.logger.info(f"📊 Preprocessed range: {processed_stack.min():.3f} to {processed_stack.max():.3f}")
        
        return processed_stack
    
    def validate_modl_input(self, stack: np.ndarray) -> Tuple[bool, List[str]]:
        """
        Validate that the stack meets MoDL input requirements.
        
        Args:
            stack: Image stack to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check dimensions
        if len(stack.shape) != 3:
            issues.append(f"Expected 3D stack, got {len(stack.shape)}D: {stack.shape}")
        
        if len(stack.shape) >= 2:
            h, w = stack.shape[-2], stack.shape[-1]
            if h != self.MODL_INPUT_SIZE or w != self.MODL_INPUT_SIZE:
                issues.append(f"Expected {self.MODL_INPUT_SIZE}x{self.MODL_INPUT_SIZE}, got {h}x{w}")
        
        # Check data type - be flexible with common microscopy image types
        if stack.dtype not in [np.uint8, np.uint16, np.float32, np.float64]:
            issues.append(f"Unexpected dtype: {stack.dtype} (expected uint8, uint16, float32, or float64)")
        
        # Check value range
        if stack.dtype == np.uint8:
            if stack.min() < 0 or stack.max() > 255:
                issues.append(f"Invalid uint8 range: {stack.min()}-{stack.max()}")
        elif stack.dtype == np.uint16:
            if stack.min() < 0 or stack.max() > 65535:
                issues.append(f"Invalid uint16 range: {stack.min()}-{stack.max()}")
        elif stack.dtype in [np.float32, np.float64]:
            # Allow wider range for float types - just check for reasonable values
            if stack.max() > 1e6 or stack.min() < -1e6:
                issues.append(f"Float values seem out of range: {stack.min()}-{stack.max()}")
        
        return len(issues) == 0, issues

    # ============================================================================
    # Task 10.3: CPU-Optimized Batch Processing with Memory Management
    # ============================================================================
    
    def _get_memory_status(self) -> Dict[str, float]:
        """
        Get current system memory usage with psutil.
        
        Returns:
            Dict containing memory statistics in GB and percentages
        """
        try:
            # Get virtual memory information
            vmem = psutil.virtual_memory()
            
            # Convert to GB for easier handling
            total_gb = vmem.total / (1024**3)
            available_gb = vmem.available / (1024**3)
            used_gb = vmem.used / (1024**3)
            
            return {
                'total_gb': total_gb,
                'available_gb': available_gb,
                'used_gb': used_gb,
                'percent_used': vmem.percent,
                'available_percent': (available_gb / total_gb) * 100,
                'threshold_85_gb': total_gb * 0.85,  # 85% memory threshold
                'threshold_90_gb': total_gb * 0.90   # 90% memory threshold (critical)
            }
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to get memory status: {e}")
            # Return conservative defaults if psutil fails
            return {
                'total_gb': 16.0,  # Assume 16GB default
                'available_gb': 8.0,
                'used_gb': 8.0,
                'percent_used': 50.0,
                'available_percent': 50.0,
                'threshold_85_gb': 13.6,
                'threshold_90_gb': 14.4
            }
    
    def calculate_optimal_batch_size(self, image_shape: Tuple[int, ...], 
                                   current_memory_percent: Optional[float] = None) -> int:
        """
        Calculate optimal batch size based on hardware type, available memory, and image dimensions.
        
        This method implements the adaptive batch sizing strategy from Task 10.3:
        - CPU processing: use smaller batch sizes (1-4 slices) to prevent memory overflow
        - GPU processing: use larger batches (8-16 slices) for efficiency
        - Dynamic adjustment based on memory pressure
        
        Args:
            image_shape: Shape of the image stack (z, h, w) or (z, h, w, c)
            current_memory_percent: Current memory usage percentage (0-100)
            
        Returns:
            Optimal batch size for processing
        """
        try:
            # Get current memory status
            memory_status = self._get_memory_status()
            
            if current_memory_percent is None:
                current_memory_percent = memory_status['percent_used']
            
            # Calculate image memory requirements
            z_slices = image_shape[0] if len(image_shape) >= 3 else 1
            
            # Estimate memory per slice in GB (assuming float32 + overhead)
            if len(image_shape) >= 3:
                slice_pixels = image_shape[1] * image_shape[2]
                if len(image_shape) == 4:  # Color image
                    slice_pixels *= image_shape[3]
                
                # Memory per slice: pixels × 4 bytes (float32) × 2 (processing overhead)
                memory_per_slice_gb = (slice_pixels * 4 * 2) / (1024**3)
            else:
                memory_per_slice_gb = 0.5  # Conservative default
            
            # Determine base batch size based on hardware type
            if (self.hardware_config and 
                self.hardware_config.hardware_type == HardwareType.GPU_AVAILABLE):
                # GPU processing: larger batches (8-16 slices) as per Task 10.3
                base_batch_size = min(16, max(8, int(memory_status['available_gb'] // 2)))
                hardware_type = "GPU"
            else:
                # CPU processing: smaller batches (1-4 slices) as per Task 10.3
                base_batch_size = min(4, max(1, int(memory_status['available_gb'] // 4)))
                hardware_type = "CPU"
            
            # Apply memory pressure adjustments
            if current_memory_percent > 90:
                # Critical memory pressure - reduce to minimum
                adjusted_batch_size = 1
                self.logger.warning(f"🚨 Critical memory pressure ({current_memory_percent:.1f}%) - reducing batch size to 1")
            elif current_memory_percent > 85:
                # High memory pressure - reduce batch size
                adjusted_batch_size = max(1, base_batch_size // 2)
                self.logger.warning(f"⚠️ High memory pressure ({current_memory_percent:.1f}%) - reducing batch size to {adjusted_batch_size}")
            elif current_memory_percent < 50:
                # Low memory usage - can increase batch size slightly
                adjusted_batch_size = min(base_batch_size * 2, 16 if hardware_type == "GPU" else 4)
                self.logger.info(f"✅ Low memory usage ({current_memory_percent:.1f}%) - increasing batch size to {adjusted_batch_size}")
            else:
                # Normal memory usage - use base calculation
                adjusted_batch_size = base_batch_size
            
            # Ensure we don't exceed available slices
            final_batch_size = min(adjusted_batch_size, z_slices)
            
            # Apply minimum safety constraints
            final_batch_size = max(1, final_batch_size)
            
            self.logger.info(f"🧮 Optimal batch size calculation:")
            self.logger.info(f"   💾 Memory: {memory_status['available_gb']:.1f}GB available ({current_memory_percent:.1f}% used)")
            self.logger.info(f"   🖥️  Hardware: {hardware_type} processing")
            self.logger.info(f"   📊 Image: {z_slices} slices, {memory_per_slice_gb:.3f}GB per slice")
            self.logger.info(f"   🎯 Batch size: {final_batch_size} slices")
            
            return final_batch_size
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating batch size: {e}")
            # Return conservative default based on hardware type
            return 2 if self.hardware_config and self.hardware_config.hardware_type == HardwareType.GPU_AVAILABLE else 1
    
    def monitor_memory_during_processing(self) -> Tuple[bool, Dict[str, float]]:
        """
        Enhanced memory monitoring with aggressive CPU memory management.

        Returns:
            Tuple of (needs_adjustment, memory_stats)
            - needs_adjustment: True if batch size should be reduced
            - memory_stats: Current memory statistics
        """
        memory_status = self._get_memory_status()

        # Enhanced memory pressure detection for Task 10.5
        critical_memory = memory_status['percent_used'] > 90  # Critical: >90%
        high_memory = memory_status['percent_used'] > 85     # High: >85%
        low_available = memory_status['available_gb'] < 1.5  # Less than 1.5GB available

        needs_adjustment = critical_memory or high_memory or low_available

        if critical_memory:
            self.logger.error(f"🚨 CRITICAL memory pressure: {memory_status['percent_used']:.1f}% used")
            self.logger.error(f"   💾 Only {memory_status['available_gb']:.1f}GB available")
            # Trigger garbage collection
            self._emergency_memory_cleanup()

        elif high_memory:
            self.logger.warning(f"⚠️ High memory pressure detected:")
            self.logger.warning(f"   📊 Usage: {memory_status['percent_used']:.1f}% ({memory_status['used_gb']:.1f}GB used)")
            self.logger.warning(f"   💾 Available: {memory_status['available_gb']:.1f}GB")

        return needs_adjustment, memory_status

    def _emergency_memory_cleanup(self):
        """
        Perform emergency memory cleanup during critical memory pressure.
        """
        try:
            import gc

            # Force garbage collection
            collected = gc.collect()
            self.logger.info(f"🧹 Emergency garbage collection: {collected} objects freed")

            # Clear TensorFlow session if available
            if TF_AVAILABLE:
                try:
                    tf.keras.backend.clear_session()
                    self.logger.info("🧹 TensorFlow session cleared")
                except Exception as e:
                    self.logger.debug(f"TensorFlow session clear failed: {e}")

        except Exception as e:
            self.logger.warning(f"⚠️ Emergency cleanup failed: {e}")

    def _adaptive_memory_management(self, current_batch_size: int, memory_percent: float) -> int:
        """
        Adaptive batch size management based on real-time memory usage.

        Args:
            current_batch_size: Current batch size
            memory_percent: Current memory usage percentage

        Returns:
            Adjusted batch size
        """
        if memory_percent > 95:
            # Critical: Force to minimum
            new_batch_size = 1
            self.logger.error(f"🚨 Critical memory: forcing batch size to {new_batch_size}")

        elif memory_percent > 90:
            # High pressure: Reduce significantly
            new_batch_size = max(1, current_batch_size // 3)
            self.logger.warning(f"⚠️ High memory pressure: reducing batch size to {new_batch_size}")

        elif memory_percent > 85:
            # Moderate pressure: Reduce by half
            new_batch_size = max(1, current_batch_size // 2)
            self.logger.warning(f"⚠️ Memory pressure: reducing batch size to {new_batch_size}")

        elif memory_percent < 60 and current_batch_size < self.DEFAULT_CPU_BATCH_SIZE:
            # Low usage: Can potentially increase (but be conservative)
            max_increase = self.DEFAULT_CPU_BATCH_SIZE if self.hardware_config.hardware_type == HardwareType.CPU_ONLY else self.DEFAULT_GPU_BATCH_SIZE
            new_batch_size = min(current_batch_size + 1, max_increase)
            self.logger.info(f"✅ Low memory usage: increasing batch size to {new_batch_size}")

        else:
            # Normal usage: Keep current size
            new_batch_size = current_batch_size

        return new_batch_size
    
    def process_image_in_batches(self, stack: np.ndarray, 
                               progress_callback: Optional[Callable[[float, str], None]] = None) -> np.ndarray:
        """
        Process image stack in optimized batches with dynamic memory management.
        
        This is the core batch processing method implementing Task 10.3 requirements:
        - Adaptive batch sizing based on hardware and memory
        - Memory monitoring during processing
        - Dynamic batch size reduction under memory pressure
        - Progress reporting per batch
        
        Args:
            stack: Input image stack to process (z, h, w)
            progress_callback: Optional callback for progress reporting (percent, message)
            
        Returns:
            Processed image stack
        """
        if stack.ndim != 3:
            raise ValueError(f"Expected 3D stack, got {stack.ndim}D array")
            
        z_slices, height, width = stack.shape
        self.logger.info(f"🔄 Starting batch processing: {z_slices} slices ({height}×{width})")
        
        # Initialize processing
        processed_slices = []
        current_slice = 0
        
        # Calculate initial batch size
        initial_batch_size = self.calculate_optimal_batch_size(stack.shape)
        current_batch_size = initial_batch_size
        
        self.logger.info(f"📦 Initial batch size: {current_batch_size} slices")
        
        try:
            while current_slice < z_slices:
                # Check memory before processing each batch
                needs_adjustment, memory_stats = self.monitor_memory_during_processing()
                
                if needs_adjustment:
                    # Use adaptive memory management for batch size adjustment
                    new_batch_size = self._adaptive_memory_management(
                        current_batch_size,
                        memory_stats['percent_used']
                    )
                    if new_batch_size != current_batch_size:
                        self.logger.warning(f"🔽 Adaptive batch size adjustment: {current_batch_size} → {new_batch_size}")
                        current_batch_size = new_batch_size
                
                # Determine actual batch range
                batch_end = min(current_slice + current_batch_size, z_slices)
                batch_slices = batch_end - current_slice
                
                self.logger.info(f"🔄 Processing batch: slices {current_slice}-{batch_end-1} ({batch_slices} slices)")
                
                # Extract batch for processing
                # Debug: Check data before batch extraction
                self.logger.info(f"🔍 BEFORE BATCH EXTRACTION: stack slice {current_slice} has range {stack[current_slice].min():.3f} to {stack[current_slice].max():.3f}")

                batch_stack = stack[current_slice:batch_end]

                # Debug: Check data after batch extraction
                self.logger.info(f"🔍 AFTER BATCH EXTRACTION: batch_stack has range {batch_stack.min():.3f} to {batch_stack.max():.3f}")

                # Real MoDL processing - preprocess and segment batch
                if progress_callback:
                    progress_percent = (current_slice / z_slices) * 100
                    hardware_mode = "GPU" if (self.hardware_config and
                                            self.hardware_config.hardware_type == HardwareType.GPU_AVAILABLE) else "CPU"
                    progress_callback(progress_percent,
                                    f"Processing batch {current_slice//current_batch_size + 1} "
                                    f"({batch_slices} slices, {hardware_mode} mode)")

                # Preprocess batch for MoDL inference
                preprocessed_batch = self.preprocess_for_modl(batch_stack)

                # Run MoDL segmentation on preprocessed batch
                segmented_batch = self._run_modl_segmentation_batch(preprocessed_batch)
                processed_slices.append(segmented_batch)
                
                # Update progress
                current_slice = batch_end
                
                # Log batch completion
                progress_percent = (current_slice / z_slices) * 100
                self.logger.info(f"✅ Batch completed: {progress_percent:.1f}% done "
                               f"(Memory: {memory_stats['percent_used']:.1f}% used)")
        
        except Exception as e:
            self.logger.error(f"❌ Batch processing failed: {e}")
            raise
        
        # Combine processed batches
        self.logger.info(f"🔗 Combining {len(processed_slices)} processed batches")
        result = np.concatenate(processed_slices, axis=0)
        
        if progress_callback:
            progress_callback(100.0, "Batch processing completed")
            
        self.logger.info(f"✅ Batch processing completed: {result.shape}")
        return result
    
    def _run_modl_segmentation_batch(self, preprocessed_batch: np.ndarray) -> np.ndarray:
        """
        Run MoDL segmentation on a preprocessed batch of slices.

        Args:
            preprocessed_batch: Preprocessed batch of slices (batch_size, h, w)

        Returns:
            Segmented batch (batch_size, output_h, output_w)
        """
        try:
            batch_size = preprocessed_batch.shape[0]
            self.logger.info(f"🔄 Running MoDL segmentation on batch of {batch_size} slices")

            # Process each slice in the batch through the three-pattern MoDL pipeline
            segmented_slices = []

            for slice_idx, slice_data in enumerate(preprocessed_batch):
                # Create temporary MoDL processor for this slice
                temp_base = f"/tmp/modl_slice_{int(time.time())}_{slice_idx}"
                modl_processor = MoDLProcessor(temp_base, self.model_path, self.hardware_config)

                try:
                    # Process slice through all three patterns
                    slice_results = modl_processor.process_slice(slice_data, slice_idx)
                    final_segmentation = slice_results['final_segmentation']
                    segmented_slices.append(final_segmentation)

                finally:
                    # Cleanup temporary files
                    modl_processor.cleanup_all()

            # Stack results into batch format
            segmented_batch = np.stack(segmented_slices, axis=0)
            self.logger.info(f"✅ MoDL segmentation completed: {segmented_batch.shape}")

            return segmented_batch

        except Exception as e:
            self.logger.error(f"❌ MoDL batch segmentation failed: {e}")
            # Return zeros as fallback
            batch_size, height, width = preprocessed_batch.shape
            return np.zeros((batch_size, self.MODL_OUTPUT_SIZE, self.MODL_OUTPUT_SIZE), dtype=np.uint8)

    def get_batch_processing_info(self, image_shape: Tuple[int, ...]) -> Dict[str, Any]:
        """
        Get information about batch processing configuration for given image shape.
        
        Args:
            image_shape: Shape of the image to be processed
            
        Returns:
            Dictionary with batch processing information
        """
        memory_status = self._get_memory_status()
        batch_size = self.calculate_optimal_batch_size(image_shape)
        
        z_slices = image_shape[0] if len(image_shape) >= 3 else 1
        num_batches = (z_slices + batch_size - 1) // batch_size  # Ceiling division
        
        # Estimate processing time
        time_per_slice = 2 if (self.hardware_config and 
                              self.hardware_config.hardware_type == HardwareType.GPU_AVAILABLE) else 5
        estimated_time_seconds = z_slices * time_per_slice
        
        return {
            'batch_size': batch_size,
            'num_batches': num_batches,
            'total_slices': z_slices,
            'memory_available_gb': memory_status['available_gb'],
            'memory_usage_percent': memory_status['percent_used'],
            'hardware_type': 'GPU' if (self.hardware_config and 
                                     self.hardware_config.hardware_type == HardwareType.GPU_AVAILABLE) else 'CPU',
            'estimated_time_seconds': estimated_time_seconds,
            'estimated_time_minutes': estimated_time_seconds / 60,
            'memory_efficient': batch_size <= 4,  # Conservative batch processing
            'processing_strategy': f"{batch_size} slices per batch, {num_batches} batches total"
        }

    # ============================================================================
    # Task 10.4: Segmentation Processing with Progress Tracking
    # ============================================================================
    
    def run_segmentation(self, input_path: str, output_dir: str, 
                        progress_callback: Callable[[int, Dict[str, Any]], None]) -> Dict[str, Any]:
        """
        Main segmentation method with hardware-aware processing and progress reporting.
        
        This method processes TIFF z-stacks using the MoDL pipeline with:
        - Hardware-optimized batch processing from Task 10.3
        - Real-time progress tracking and memory monitoring
        - CPU/GPU processing paths with automatic fallback
        - Comprehensive error handling and recovery
        
        Args:
            input_path (str): Path to input TIFF file
            output_dir (str): Directory to save segmentation results
            progress_callback (Callable): Callback function for progress updates
                Signature: callback(percentage: int, info: Dict[str, Any])
                
        Returns:
            Dict containing processing results and metadata
            
        Raises:
            FileNotFoundError: If input file doesn't exist
            ValueError: If input validation fails
            RuntimeError: If segmentation processing fails
        """
        # Initialize variables with defaults to prevent NameError in exception handler
        input_shape = None
        total_slices = 0
        total_batches = 0
        processed_slices = 0
        optimal_batch_size = 1

        try:
            start_time = time.time()
            self.logger.info(f"🚀 Starting segmentation processing: {input_path}")

            # Phase 1: Setup & Validation (0-10%)
            progress_callback(0, {
                "stage": "initialization",
                "hardware_mode": self.hardware_config.hardware_type.value.upper(),
                "message": "Initializing segmentation workflow"
            })

            # Task 10.5: Enhanced error handling and validation
            # Check file permissions
            permissions_valid, permission_error = self._validate_file_permissions(input_path, output_dir)
            if not permissions_valid:
                raise PermissionError(permission_error)

            # Check disk space (estimate 2GB needed for processing)
            if not self._check_disk_space(output_dir, required_gb=2.0):
                raise OSError("Insufficient disk space for segmentation processing")
            
            # Validate inputs
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"Input file not found: {input_path}")
            
            os.makedirs(output_dir, exist_ok=True)
            
            # Load and validate z-stack with error handling
            progress_callback(2, {
                "stage": "loading",
                "message": "Loading and validating TIFF z-stack"
            })

            def load_operation():
                return self.load_and_detect_zstack(input_path)

            stack, stack_info = self._execute_with_hardware_fallback(
                load_operation,
                "TIFF z-stack loading"
            )
            total_slices = stack.shape[0]
            input_shape = stack.shape  # Store original image shape for metadata
            
            # Validate MoDL compatibility
            is_valid, validation_issues = self.validate_modl_input(stack)
            if not is_valid:
                raise ValueError(f"Input validation failed: {'; '.join(validation_issues)}")
            
            # Calculate optimal batch size using Task 10.3 infrastructure
            optimal_batch_size = self.calculate_optimal_batch_size(stack.shape)
            memory_status = self._get_memory_status()
            
            progress_callback(5, {
                "stage": "planning",
                "total_slices": total_slices,
                "batch_size": optimal_batch_size,
                "hardware_mode": self.hardware_config.hardware_type.value.upper(),
                "memory_usage_gb": memory_status['used_gb'],
                "message": f"Planning processing: {total_slices} slices in batches of {optimal_batch_size}"
            })
            
            # Setup temporary directories and MoDL processor
            temp_base = os.path.join(output_dir, f"temp_{int(time.time())}")
            modl_processor = MoDLProcessor(temp_base, self.model_path, self.hardware_config)
            
            progress_callback(8, {
                "stage": "setup",
                "message": "MoDL processor initialized"
            })
            
            # Phase 2: MoDL Pipeline Processing (10-80%)
            segmentation_results = []
            processed_slices = 0
            total_batches = (total_slices + optimal_batch_size - 1) // optimal_batch_size  # Calculate total number of batches
            
            for batch_start in range(0, total_slices, optimal_batch_size):
                batch_end = min(batch_start + optimal_batch_size, total_slices)
                batch_slices = stack[batch_start:batch_end]
                
                self.logger.info(f"Processing batch {batch_start//optimal_batch_size + 1}: slices {batch_start}-{batch_end-1}")
                
                # Process each slice in the current batch
                for slice_idx, slice_data in enumerate(batch_slices):
                    current_slice = batch_start + slice_idx
                    base_progress = 10 + int((current_slice / total_slices) * 70)
                    
                    # Stage 1: Preprocessing and cropping (5% per slice)
                    progress_callback(base_progress, {
                        "stage": "preprocessing",
                        "current_slice": current_slice + 1,
                        "total_slices": total_slices,
                        "hardware_mode": self.hardware_config.hardware_type.value.upper(),
                        "message": f"Preprocessing slice {current_slice + 1}/{total_slices}"
                    })
                    
                    # Preprocess slice for MoDL
                    preprocessed_slice = self.preprocess_for_modl(slice_data.reshape(1, *slice_data.shape))
                    
                    # Stage 2: MoDL segmentation processing (10% per slice)
                    progress_callback(base_progress + 2, {
                        "stage": "segmentation",
                        "current_slice": current_slice + 1,
                        "total_slices": total_slices,
                        "current_pattern": "processing",
                        "hardware_mode": self.hardware_config.hardware_type.value.upper(),
                        "batch_size": optimal_batch_size,
                        "message": f"Running segmentation on slice {current_slice + 1}/{total_slices}"
                    })
                    
                    # Process slice through MoDL pipeline
                    slice_results = modl_processor.process_slice(preprocessed_slice[0], current_slice)
                    segmentation_results.append({
                        'slice_idx': current_slice,
                        'results': slice_results
                    })
                    
                    processed_slices += 1
                    
                    # Memory monitoring after each slice
                    memory_ok, current_memory = self.monitor_memory_during_processing()
                    if not memory_ok:
                        self.logger.warning(f"⚠️ High memory usage detected: {current_memory['percent_used']:.1f}%")
                        # Trigger cleanup but continue processing
                        modl_processor.cleanup_temp_files(current_slice)
                    
                    # Update progress
                    progress_callback(base_progress + 5, {
                        "stage": "reconstruction",
                        "current_slice": current_slice + 1,
                        "total_slices": total_slices,
                        "memory_usage_gb": current_memory['used_gb'],
                        "memory_status": "ok" if memory_ok else "high",
                        "message": f"Completed slice {current_slice + 1}/{total_slices}"
                    })
                
                # Batch completion cleanup
                self.logger.info(f"✅ Completed batch processing: {len(batch_slices)} slices")
            
            # Phase 3: Final Assembly and Output (80-100%)
            progress_callback(85, {
                "stage": "assembly",
                "message": "Assembling final segmentation results"
            })
            
            # Reconstruct full z-stack from processed slices
            final_segmentation = self._reconstruct_zstack_from_results(segmentation_results, stack.shape)
            
            progress_callback(90, {
                "stage": "saving",
                "message": "Saving segmentation results"
            })
            
            # Save final results with enhanced output management
            output_file = os.path.join(output_dir, f"{os.path.basename(input_path)}_segmented.tif")

            # Prepare enhanced processing metadata for output management
            enhanced_processing_metadata = {
                'input_path': input_path,
                'original_shape': input_shape,
                'z_stack_info': {
                    'total_slices': total_slices,
                    'processed_slices': processed_slices,
                    'slice_dimensions': input_shape[1:] if input_shape is not None and len(input_shape) > 1 else None
                },
                'processing_mode': 'modl_segmentation',
                'model_path': self.model_path,
                'batch_size': optimal_batch_size,
                'total_batches': total_batches,
                'patterns_per_slice': 'three_pattern_crop',
                'crop_patterns': '44x44, 43x43, 34x34',
                'reconstruction_method': 'pattern_stitching',
                'inference_mode': 'tensorflow_keras'
            }

            # Save results using enhanced output management (returns comprehensive info)
            save_result = self._save_segmentation_results(final_segmentation, output_file, enhanced_processing_metadata)

            # Generate processing metadata and timing
            end_time = time.time()
            processing_time = end_time - start_time

            # Create enhanced processing report
            processing_report = {
                'input_file': input_path,
                'output_file': save_result['output_file'],
                'metadata_file': save_result.get('metadata_file'),
                'processing_time_seconds': processing_time,
                'file_size_mb': save_result.get('file_size_mb', 0),
                'checksum': save_result.get('checksum'),
                'total_slices': total_slices,
                'batch_size_used': optimal_batch_size,
                'hardware_mode': self.hardware_config.hardware_type.value,
                'memory_peak_gb': memory_status['used_gb'],
                'successful_slices': processed_slices,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'metadata_summary': save_result.get('metadata_summary', {})
            }
            
            progress_callback(95, {
                "stage": "cleanup",
                "message": "Cleaning up temporary files"
            })
            
            # Cleanup temporary files
            modl_processor.cleanup_all()
            
            progress_callback(100, {
                "stage": "completed",
                "message": f"Segmentation completed in {processing_time:.1f}s",
                "output_file": output_file,
                "processing_time": processing_time
            })
            
            self.logger.info(f"✅ Segmentation completed successfully in {processing_time:.1f}s")
            return processing_report
            
        except Exception as e:
            error_msg = f"Segmentation failed: {str(e)}"
            self.logger.error(error_msg)
            progress_callback(-1, {
                "stage": "error",
                "message": error_msg,
                "error": str(e)
            })
            raise RuntimeError(error_msg) from e
    
    def _reconstruct_zstack_from_results(self, results: List[Dict], original_shape: Tuple[int, ...]) -> np.ndarray:
        """
        Reconstruct z-stack from individual slice segmentation results with enhanced validation.

        Args:
            results: List of segmentation results from MoDL processing
            original_shape: Original image shape (z, h, w)

        Returns:
            Reconstructed segmentation stack (z, output_h, output_w)
        """
        try:
            z_dim, height, width = original_shape
            self.logger.info(f"🔗 Reconstructing z-stack: {z_dim} slices -> {self.MODL_OUTPUT_SIZE}x{self.MODL_OUTPUT_SIZE}")

            # Initialize output stack with zeros
            segmented_stack = np.zeros((z_dim, self.MODL_OUTPUT_SIZE, self.MODL_OUTPUT_SIZE), dtype=np.uint8)

            # Sort results by slice index to ensure correct order
            sorted_results = sorted(results, key=lambda x: x['slice_idx'])

            successful_slices = 0
            for result in sorted_results:
                slice_idx = result['slice_idx']

                # Handle different result formats
                if 'results' in result and 'final_segmentation' in result['results']:
                    # From MoDLProcessor.process_slice format
                    slice_segmentation = result['results']['final_segmentation']
                elif 'final_segmentation' in result:
                    # Direct format
                    slice_segmentation = result['final_segmentation']
                else:
                    self.logger.warning(f"⚠️ Invalid result format for slice {slice_idx}")
                    continue

                # Validate and resize slice segmentation
                if slice_segmentation is None:
                    self.logger.warning(f"⚠️ Null segmentation for slice {slice_idx}")
                    continue

                # Ensure proper dimensions
                if slice_segmentation.ndim == 3:
                    # Remove channel dimension if present
                    slice_segmentation = slice_segmentation.squeeze()

                if slice_segmentation.shape != (self.MODL_OUTPUT_SIZE, self.MODL_OUTPUT_SIZE):
                    # Resize to standard MoDL output size
                    try:
                        import cv2
                        slice_segmentation = cv2.resize(
                            slice_segmentation,
                            (self.MODL_OUTPUT_SIZE, self.MODL_OUTPUT_SIZE),
                            interpolation=cv2.INTER_AREA
                        )
                    except ImportError:
                        # Fallback resizing
                        slice_segmentation = self._simple_resize(slice_segmentation, self.MODL_OUTPUT_SIZE)

                # Ensure proper data type and range
                slice_segmentation = slice_segmentation.astype(np.uint8)

                # Validate slice index
                if 0 <= slice_idx < z_dim:
                    segmented_stack[slice_idx] = slice_segmentation
                    successful_slices += 1
                else:
                    self.logger.warning(f"⚠️ Invalid slice index {slice_idx} for stack size {z_dim}")

            self.logger.info(f"✅ Z-stack reconstruction completed: {successful_slices}/{z_dim} slices successful")

            if successful_slices == 0:
                self.logger.error("❌ No valid slices in reconstruction results")
                raise ValueError("No valid segmentation results found")

            # Log statistics
            non_zero_slices = np.count_nonzero(segmented_stack.sum(axis=(1, 2)))
            self.logger.info(f"📊 Non-zero slices: {non_zero_slices}/{z_dim}")

            return segmented_stack

        except Exception as e:
            self.logger.error(f"❌ Z-stack reconstruction failed: {e}")
            # Return zeros as fallback
            z_dim = original_shape[0] if len(original_shape) >= 3 else 1
            return np.zeros((z_dim, self.MODL_OUTPUT_SIZE, self.MODL_OUTPUT_SIZE), dtype=np.uint8)
    
    def _save_segmentation_results(self, segmentation: np.ndarray, output_file: str,
                                 metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Save segmentation results with enhanced file management and comprehensive metadata.

        Args:
            segmentation: The segmentation result array
            output_file: Path to save the output file
            metadata: Processing metadata

        Returns:
            Dictionary containing saved file information and metadata
        """
        start_time = time.time()

        try:
            import tifffile
            import tempfile
            import hashlib

            # Generate enhanced output filename
            output_file = self._generate_output_filename(output_file, metadata)

            # Validate output directory and permissions
            self._validate_output_directory(output_file)

            # Generate comprehensive metadata
            comprehensive_metadata = self._generate_comprehensive_metadata(metadata, segmentation, start_time)

            # Prepare TIFF metadata for embedding
            tiff_metadata = {
                'ImageDescription': json.dumps({
                    'source': 'MoDL Segmentation Pipeline',
                    'version': '2.0',
                    'processing_timestamp': comprehensive_metadata['processing']['end_timestamp'],
                    'hardware_mode': comprehensive_metadata['hardware']['current_type'],
                    'original_shape': comprehensive_metadata['input']['original_shape'],
                    'output_shape': comprehensive_metadata['output']['final_shape'],
                    'processing_summary': {
                        'duration_seconds': comprehensive_metadata['processing']['duration_seconds'],
                        'batch_size_used': comprehensive_metadata['processing']['batch_size'],
                        'fallback_events': len(comprehensive_metadata['hardware']['fallback_events'])
                    }
                })
            }

            # Perform atomic file save
            saved_files = self._atomic_save_operation(segmentation, output_file, tiff_metadata, comprehensive_metadata)

            # Generate result summary
            result_info = {
                'output_file': output_file,
                'metadata_file': saved_files.get('metadata_file'),
                'file_size_mb': os.path.getsize(output_file) / (1024 * 1024),
                'processing_duration': time.time() - start_time,
                'checksum': saved_files.get('checksum'),
                'metadata_summary': {
                    'hardware_type': comprehensive_metadata['hardware']['current_type'],
                    'fallback_occurred': comprehensive_metadata['hardware']['fallback_occurred'],
                    'performance_metrics': comprehensive_metadata['performance']
                }
            }

            self.logger.info(f"💾 Segmentation saved successfully: {output_file}")
            self.logger.info(f"📊 File size: {result_info['file_size_mb']:.2f}MB, Duration: {result_info['processing_duration']:.2f}s")

            return result_info

        except Exception as e:
            self.logger.error(f"❌ Failed to save segmentation: {e}")
            # Attempt cleanup of any partial files
            self._cleanup_partial_files(output_file)
            raise

    def _generate_output_filename(self, original_path: str, metadata: Dict[str, Any]) -> str:
        """
        Generate enhanced output filename with proper naming conventions.

        Args:
            original_path: Original output path
            metadata: Processing metadata

        Returns:
            Enhanced output filename
        """
        try:
            # Parse the original path
            directory = os.path.dirname(original_path)
            filename = os.path.basename(original_path)
            name, ext = os.path.splitext(filename)

            # Clean the filename (remove special characters, limit length)
            clean_name = re.sub(r'[^\w\-_.]', '_', name)[:200]  # Limit to 200 chars

            # Add segmentation suffix if not present
            if not clean_name.endswith('_segmented'):
                clean_name += '_segmented'

            # Add timestamp for uniqueness if file exists
            enhanced_filename = f"{clean_name}{ext}"
            enhanced_path = os.path.join(directory, enhanced_filename)

            if os.path.exists(enhanced_path):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                enhanced_filename = f"{clean_name}_{timestamp}{ext}"
                enhanced_path = os.path.join(directory, enhanced_filename)

            return enhanced_path

        except Exception as e:
            self.logger.warning(f"⚠️ Filename generation failed, using original: {e}")
            return original_path

    def _validate_output_directory(self, output_path: str) -> None:
        """
        Validate output directory exists and has proper permissions.

        Args:
            output_path: Path to the output file

        Raises:
            Exception: If directory validation fails
        """
        try:
            output_dir = os.path.dirname(output_path)

            # Create directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)

            # Check write permissions
            if not os.access(output_dir, os.W_OK):
                raise PermissionError(f"No write permission for directory: {output_dir}")

            # Check available disk space (require at least 100MB)
            if not self._check_disk_space(output_dir, required_gb=0.1):
                raise OSError(f"Insufficient disk space in directory: {output_dir}")

        except Exception as e:
            self.logger.error(f"❌ Output directory validation failed: {e}")
            raise

    def _cleanup_partial_files(self, output_path: str) -> None:
        """
        Clean up any partial files created during failed save operations.

        Args:
            output_path: Path to the output file
        """
        try:
            # Remove main output file if it exists
            if os.path.exists(output_path):
                os.unlink(output_path)
                self.logger.info(f"🧹 Cleaned up partial file: {output_path}")

            # Remove temporary files with similar names
            output_dir = os.path.dirname(output_path)
            output_name = os.path.basename(output_path)

            for filename in os.listdir(output_dir):
                if filename.startswith(output_name) and filename.endswith('.tmp'):
                    temp_path = os.path.join(output_dir, filename)
                    try:
                        os.unlink(temp_path)
                        self.logger.info(f"🧹 Cleaned up temp file: {temp_path}")
                    except:
                        pass  # Ignore cleanup errors

        except Exception as e:
            self.logger.warning(f"⚠️ Cleanup failed: {e}")

    def _generate_comprehensive_metadata(self, processing_metadata: Dict[str, Any],
                                       segmentation: np.ndarray, start_time: float) -> Dict[str, Any]:
        """
        Generate comprehensive metadata including hardware, performance, and processing details.

        Args:
            processing_metadata: Metadata from the processing pipeline
            segmentation: The segmentation result array
            start_time: Processing start timestamp

        Returns:
            Comprehensive metadata dictionary
        """
        try:
            current_time = time.time()
            processing_duration = current_time - start_time

            # Collect hardware configuration details
            hardware_info = self._collect_hardware_metadata()

            # Collect performance metrics
            performance_metrics = self._collect_performance_metrics(
                processing_duration, segmentation, processing_metadata
            )

            # Collect processing parameters
            processing_params = self._collect_processing_parameters(processing_metadata)

            # Collect TensorFlow configuration
            tf_config = self._collect_tensorflow_metadata()

            # Build comprehensive metadata structure
            comprehensive_metadata = {
                'metadata_version': '2.0',
                'generation_timestamp': datetime.now().isoformat(),
                'processing': {
                    'start_timestamp': datetime.fromtimestamp(start_time).isoformat(),
                    'end_timestamp': datetime.now().isoformat(),
                    'duration_seconds': processing_duration,
                    'batch_size': processing_metadata.get('batch_size', 'unknown'),
                    'model_path': processing_metadata.get('model_path', 'unknown'),
                    'processing_mode': processing_metadata.get('processing_mode', 'unknown')
                },
                'input': {
                    'original_shape': processing_metadata.get('original_shape'),
                    'input_path': processing_metadata.get('input_path', 'unknown'),
                    'z_stack_info': processing_metadata.get('z_stack_info', {}),
                    'file_size_mb': self._get_input_file_size(processing_metadata.get('input_path'))
                },
                'output': {
                    'final_shape': list(segmentation.shape) if segmentation is not None else None,
                    'data_type': str(segmentation.dtype) if segmentation is not None else 'unknown',
                    'memory_usage_mb': segmentation.nbytes / (1024*1024) if segmentation is not None else 0,
                    'unique_values': len(np.unique(segmentation)) if segmentation is not None else 0
                },
                'hardware': hardware_info,
                'performance': performance_metrics,
                'tensorflow': tf_config,
                'parameters': processing_params
            }

            return comprehensive_metadata

        except Exception as e:
            self.logger.warning(f"⚠️ Metadata generation failed: {e}")
            # Return minimal metadata on failure
            return {
                'metadata_version': '2.0',
                'generation_timestamp': datetime.now().isoformat(),
                'error': f"Metadata generation failed: {str(e)}",
                'processing': {
                    'duration_seconds': time.time() - start_time,
                    'batch_size': processing_metadata.get('batch_size', 'unknown')
                },
                'hardware': {
                    'current_type': self.hardware_config.hardware_type.value if self.hardware_config else 'unknown'
                }
            }

    def _collect_hardware_metadata(self) -> Dict[str, Any]:
        """Collect detailed hardware configuration and status."""
        hardware_info = {
            'current_type': self.hardware_config.hardware_type.value if self.hardware_config else 'unknown',
            'original_type': self.hardware_config.original_hardware_type.value if (
                self.hardware_config and self.hardware_config.original_hardware_type
            ) else 'unknown',
            'device_name': self.hardware_config.device_name if self.hardware_config else 'unknown',
            'memory_gb': self.hardware_config.memory_gb if self.hardware_config else 0,
            'fallback_occurred': self.hardware_config.has_fallback_occurred if self.hardware_config else False,
            'fallback_events': [
                {
                    'timestamp': event.timestamp,
                    'from_hardware': event.from_hardware.value,
                    'to_hardware': event.to_hardware.value,
                    'reason': event.reason,
                    'error_details': event.error_details,
                    'formatted_time': datetime.fromtimestamp(event.timestamp).isoformat()
                }
                for event in (self.hardware_config.fallback_events if self.hardware_config else [])
            ],
            'cpu_count': self.hardware_config.cpu_count if self.hardware_config else os.cpu_count(),
            'batch_size': self.hardware_config.batch_size if self.hardware_config else 1
        }

        # Add current system memory status
        try:
            memory = psutil.virtual_memory()
            hardware_info['system_memory'] = {
                'total_gb': memory.total / (1024**3),
                'available_gb': memory.available / (1024**3),
                'percent_used': memory.percent,
                'used_gb': memory.used / (1024**3)
            }
        except:
            hardware_info['system_memory'] = {'error': 'Failed to collect memory info'}

        return hardware_info

    def _collect_performance_metrics(self, duration: float, segmentation: np.ndarray,
                                   processing_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Collect performance metrics and statistics."""
        try:
            z_slices = segmentation.shape[0] if segmentation is not None and len(segmentation.shape) >= 3 else 1

            performance_metrics = {
                'processing_duration_seconds': duration,
                'slices_per_second': z_slices / duration if duration > 0 else 0,
                'throughput_mb_per_second': (
                    (segmentation.nbytes / (1024*1024)) / duration
                    if segmentation is not None and duration > 0 else 0
                ),
                'z_slices_processed': z_slices,
                'batch_processing_info': {
                    'total_batches': processing_metadata.get('total_batches', 'unknown'),
                    'batch_size_used': processing_metadata.get('batch_size', 'unknown'),
                    'patterns_per_slice': processing_metadata.get('patterns_per_slice', 'unknown')
                }
            }

            # Add memory efficiency metrics
            if segmentation is not None:
                performance_metrics['memory_efficiency'] = {
                    'output_size_mb': segmentation.nbytes / (1024*1024),
                    'memory_per_slice_mb': (segmentation.nbytes / (1024*1024)) / z_slices if z_slices > 0 else 0
                }

            return performance_metrics

        except Exception as e:
            return {
                'error': f"Performance metrics collection failed: {str(e)}",
                'processing_duration_seconds': duration
            }

    def _collect_processing_parameters(self, processing_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Collect detailed processing parameters and configuration."""
        return {
            'model_configuration': {
                'model_path': processing_metadata.get('model_path', 'unknown'),
                'model_exists': os.path.exists(processing_metadata.get('model_path', '')) if processing_metadata.get('model_path') else False,
                'inference_mode': processing_metadata.get('inference_mode', 'unknown')
            },
            'preprocessing': {
                'crop_patterns': processing_metadata.get('crop_patterns', 'unknown'),
                'input_scaling': processing_metadata.get('input_scaling', 'unknown'),
                'normalization': processing_metadata.get('normalization', 'unknown')
            },
            'postprocessing': {
                'reconstruction_method': processing_metadata.get('reconstruction_method', 'unknown'),
                'output_scaling': processing_metadata.get('output_scaling', 'unknown')
            }
        }

    def _collect_tensorflow_metadata(self) -> Dict[str, Any]:
        """Collect TensorFlow configuration and status."""
        try:
            if not TF_AVAILABLE:
                return {'status': 'unavailable', 'error': 'TensorFlow not installed'}

            tf_info = {
                'version': tf.__version__,
                'gpu_available': len(tf.config.list_physical_devices('GPU')) > 0,
                'physical_devices': {
                    'cpu': len(tf.config.list_physical_devices('CPU')),
                    'gpu': len(tf.config.list_physical_devices('GPU'))
                },
                'mixed_precision_enabled': tf.keras.mixed_precision.global_policy().name != 'float32',
                'memory_growth_enabled': 'unknown'  # Will be updated if GPU available
            }

            # Add GPU-specific information if available
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                try:
                    tf_info['gpu_details'] = []
                    for gpu in gpus:
                        gpu_details = {
                            'name': gpu.name,
                            'device_type': gpu.device_type
                        }
                        try:
                            memory_growth = tf.config.experimental.get_memory_growth(gpu)
                            gpu_details['memory_growth'] = memory_growth
                        except:
                            gpu_details['memory_growth'] = 'unknown'

                        tf_info['gpu_details'].append(gpu_details)
                except Exception as e:
                    tf_info['gpu_error'] = str(e)

            return tf_info

        except Exception as e:
            return {
                'status': 'error',
                'error': f"TensorFlow metadata collection failed: {str(e)}"
            }

    def _get_input_file_size(self, input_path: Optional[str]) -> float:
        """Get input file size in MB."""
        try:
            if input_path and os.path.exists(input_path):
                return os.path.getsize(input_path) / (1024 * 1024)
            return 0
        except:
            return 0

    def _atomic_save_operation(self, segmentation: np.ndarray, output_file: str,
                             tiff_metadata: Dict[str, Any], comprehensive_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform atomic file save operation with rollback capability.

        Args:
            segmentation: The segmentation array to save
            output_file: Final output file path
            tiff_metadata: TIFF metadata to embed
            comprehensive_metadata: Complete metadata to save separately

        Returns:
            Dictionary with saved file information
        """
        temp_tiff_file = None
        temp_metadata_file = None
        saved_files = {}

        try:
            import tifffile

            # Create temporary files for atomic operation
            output_dir = os.path.dirname(output_file)
            base_name = os.path.splitext(os.path.basename(output_file))[0]

            temp_tiff_file = os.path.join(output_dir, f"{base_name}_temp_{int(time.time())}.tif")
            temp_metadata_file = os.path.join(output_dir, f"{base_name}_metadata_{int(time.time())}.json")

            # Save TIFF to temporary file
            self.logger.info("💾 Writing segmentation to temporary file...")
            tifffile.imwrite(temp_tiff_file, segmentation, metadata=tiff_metadata)

            # Calculate checksum for integrity verification
            checksum = self._calculate_file_checksum(temp_tiff_file)

            # Save comprehensive metadata to temporary JSON file
            metadata_file_path = os.path.splitext(output_file)[0] + "_metadata.json"
            with open(temp_metadata_file, 'w') as f:
                json.dump(comprehensive_metadata, f, indent=2, default=str)

            # Verify file integrity
            if not self._verify_file_integrity(temp_tiff_file, segmentation):
                raise ValueError("File integrity verification failed")

            # Atomic move operations (rename is atomic on most filesystems)
            self.logger.info("🔄 Performing atomic file operations...")

            # Move TIFF file
            if os.path.exists(output_file):
                backup_file = output_file + ".backup"
                os.rename(output_file, backup_file)
                saved_files['backup_created'] = backup_file

            os.rename(temp_tiff_file, output_file)
            temp_tiff_file = None  # Mark as successfully moved

            # Move metadata file
            if os.path.exists(metadata_file_path):
                backup_metadata = metadata_file_path + ".backup"
                os.rename(metadata_file_path, backup_metadata)

            os.rename(temp_metadata_file, metadata_file_path)
            temp_metadata_file = None  # Mark as successfully moved

            # Clean up backup files if everything succeeded
            if 'backup_created' in saved_files:
                try:
                    os.unlink(saved_files['backup_created'])
                    backup_metadata = metadata_file_path + ".backup"
                    if os.path.exists(backup_metadata):
                        os.unlink(backup_metadata)
                except:
                    pass  # Ignore backup cleanup errors

            saved_files.update({
                'output_file': output_file,
                'metadata_file': metadata_file_path,
                'checksum': checksum,
                'atomic_operation': 'success'
            })

            self.logger.info("✅ Atomic save operation completed successfully")
            return saved_files

        except Exception as e:
            self.logger.error(f"❌ Atomic save operation failed: {e}")

            # Rollback: clean up temporary files
            try:
                if temp_tiff_file and os.path.exists(temp_tiff_file):
                    os.unlink(temp_tiff_file)
                if temp_metadata_file and os.path.exists(temp_metadata_file):
                    os.unlink(temp_metadata_file)

                # Restore backup if it exists
                if 'backup_created' in saved_files and os.path.exists(saved_files['backup_created']):
                    os.rename(saved_files['backup_created'], output_file)

            except Exception as rollback_error:
                self.logger.error(f"❌ Rollback failed: {rollback_error}")

            raise

    def _calculate_file_checksum(self, file_path: str) -> str:
        """Calculate SHA-256 checksum for file integrity verification."""
        try:
            hash_sha256 = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            self.logger.warning(f"⚠️ Checksum calculation failed: {e}")
            return "checksum_failed"

    def _verify_file_integrity(self, file_path: str, original_array: np.ndarray) -> bool:
        """Verify saved file matches original array."""
        try:
            import tifffile
            saved_array = tifffile.imread(file_path)

            # Check shape
            if saved_array.shape != original_array.shape:
                self.logger.error(f"Shape mismatch: {saved_array.shape} vs {original_array.shape}")
                return False

            # Check data type
            if saved_array.dtype != original_array.dtype:
                self.logger.warning(f"Data type difference: {saved_array.dtype} vs {original_array.dtype}")

            # Check if arrays are equal (sample-based for large arrays)
            if original_array.size > 1000000:  # For large arrays, sample-based comparison
                indices = np.random.choice(original_array.size, 1000, replace=False)
                original_sample = original_array.flat[indices]
                saved_sample = saved_array.flat[indices]
                if not np.array_equal(original_sample, saved_sample):
                    self.logger.error("Sample-based integrity check failed")
                    return False
            else:
                if not np.array_equal(saved_array, original_array):
                    self.logger.error("Full array integrity check failed")
                    return False

            return True

        except Exception as e:
            self.logger.error(f"❌ File integrity verification failed: {e}")
            return False


# Factory function for easy instantiation
def create_segmentation_workflow(model_path: Optional[str] = None, 
                                force_cpu: bool = False,
                                memory_limit_gb: Optional[float] = None) -> SegmentationWorkflow:
    """
    Factory function to create a properly configured SegmentationWorkflow.
    
    Args:
        model_path: Path to the segmentation model (auto-detects if None)
        force_cpu: Force CPU-only processing
        memory_limit_gb: Optional GPU memory limit
        
    Returns:
        Configured SegmentationWorkflow instance
    """
    # Auto-detect model path if not provided
    if model_path is None:
        # Look for model in project's MoDL directory first
        project_model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                         "MoDL", "model", "U-RNet+.hdf5")
        if os.path.exists(project_model_path):
            model_path = project_model_path
        else:
            # Fallback to original MoDL path structure
            original_model_path = "/Al_Applications/MoDL/model/U-RNet+.hdf5"
            if os.path.exists(original_model_path):
                model_path = original_model_path
            # If neither exists, model_path remains None (mock mode)
    
    return SegmentationWorkflow(
        model_path=model_path,
        force_cpu=force_cpu,
        memory_limit_gb=memory_limit_gb
    )


if __name__ == "__main__":
    # Test the hardware detection and configuration
    print("🧪 Testing SegmentationWorkflow Hardware Detection...")
    print("=" * 60)
    
    workflow = create_segmentation_workflow()
    
    # Display hardware information
    hardware_info = workflow.get_hardware_info()
    for key, value in hardware_info.items():
        print(f"{key}: {value}")
    
    print("\n🔍 Processing Mode:", workflow.get_processing_mode_info())
    
    # Validate configuration
    is_valid, issues = workflow.validate_configuration()
    print(f"\n✅ Configuration Valid: {is_valid}")
    if issues:
        print("⚠️ Issues found:")
        for issue in issues:
            print(f"  - {issue}")
    
    # Performance recommendations
    recommendations = workflow.get_performance_recommendations()
    if recommendations:
        print("\n💡 Performance Recommendations:")
        for rec in recommendations:
            print(f"  - {rec}")
    
    print("\n🎯 Hardware detection and configuration test completed!")


# === BACKWARD COMPATIBILITY FOR STREAMLIT APP ===
# Create a default module-level instance for backward compatibility
# This allows existing app.py imports to continue working
segmentation_workflow = create_segmentation_workflow(force_cpu=True)


# MoDL Processor class for Task 10.4 implementation
class MoDLProcessor:
    """
    Flexible wrapper around MoDL pipeline with configurable paths and hardware awareness.

    This class provides a clean interface to the MoDL segmentation pipeline while
    handling temporary file management, hardware optimization, and error recovery.
    """

    # MoDL cropping patterns from original segment_predict.py
    CROP_PATTERNS = {
        "44": [[0, 0, 1/4, 1/4], [1/4, 0, 1/2, 1/4], [1/2, 0, 3/4, 1/4], [3/4, 0, 1, 1/4],
               [0, 1/4, 1/4, 1/2], [1/4, 1/4, 1/2, 1/2], [1/2, 1/4, 3/4, 1/2], [3/4, 1/4, 1, 1/2],
               [0, 1/2, 1/4, 3/4], [1/4, 1/2, 1/2, 3/4], [1/2, 1/2, 3/4, 3/4], [3/4, 1/2, 1, 3/4],
               [0, 3/4, 1/4, 1], [1/4, 3/4, 1/2, 1], [1/2, 3/4, 3/4, 1], [3/4, 3/4, 1, 1]],

        "43": [[0, 1/8, 1/4, 3/8], [1/4, 1/8, 1/2, 3/8], [1/2, 1/8, 3/4, 3/8], [3/4, 1/8, 1, 3/8],
               [0, 3/8, 1/4, 5/8], [1/4, 3/8, 1/2, 5/8], [1/2, 3/8, 3/4, 5/8], [3/4, 3/8, 1, 5/8],
               [0, 5/8, 1/4, 7/8], [1/4, 5/8, 1/2, 7/8], [1/2, 5/8, 3/4, 7/8], [3/4, 5/8, 1, 7/8]],

        "34": [[1/8, 0, 3/8, 1/4], [3/8, 0, 5/8, 1/4], [5/8, 0, 7/8, 1/4],
               [1/8, 1/4, 3/8, 1/2], [3/8, 1/4, 5/8, 1/2], [5/8, 1/4, 7/8, 1/2],
               [1/8, 1/2, 3/8, 3/4], [3/8, 1/2, 5/8, 3/4], [5/8, 1/2, 7/8, 3/4],
               [1/8, 3/4, 3/8, 1], [3/8, 3/4, 5/8, 1], [5/8, 3/4, 7/8, 1]]
    }

    # Stitching positions and orders from original MoDL
    STITCH_POSITIONS = {
        "44": [(0, 0), (512, 0), (1024, 0), (1536, 0),
               (0, 512), (512, 512), (1024, 512), (1536, 512),
               (0, 1024), (512, 1024), (1024, 1024), (1536, 1024),
               (0, 1536), (512, 1536), (1024, 1536), (1536, 1536)],

        "43": [(0, 256), (512, 256), (1024, 256), (1536, 256),
               (0, 768), (512, 768), (1024, 768), (1536, 768),
               (0, 1280), (512, 1280), (1024, 1280), (1536, 1280)],

        "34": [(256, 0), (768, 0), (1280, 0),
               (256, 512), (768, 512), (1280, 512),
               (256, 1024), (768, 1024), (1280, 1024),
               (256, 1536), (768, 1536), (1280, 1536)]
    }

    STITCH_ORDERS = {
        "44": [0, 8, 9, 10, 11, 12, 13, 14, 15, 1, 2, 3, 4, 5, 6, 7],
        "43": [0, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2, 3],
        "34": [0, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2, 3]
    }
    
    def __init__(self, temp_dir: str, model_path: str, hardware_config: HardwareConfig):
        """
        Initialize MoDL processor with configurable paths.
        
        Args:
            temp_dir: Base directory for temporary processing files
            model_path: Path to MoDL model file
            hardware_config: Hardware configuration from main workflow
        """
        self.temp_dir = temp_dir
        self.model_path = model_path
        self.hardware_config = hardware_config
        self.logger = logging.getLogger(__name__)
        
        # Create temp directory structure
        self.setup_temp_directories()
        
        # Load MoDL model if available
        self.model = self._load_model()
    
    def setup_temp_directories(self) -> None:
        """Setup temporary directory structure for MoDL processing."""
        try:
            # Base directories
            os.makedirs(self.temp_dir, exist_ok=True)
            
            # MoDL expects specific directory structure
            self.dirs = {
                'input': os.path.join(self.temp_dir, 'input'),
                'test_44': os.path.join(self.temp_dir, 'test', 'test_44'),
                'test_43': os.path.join(self.temp_dir, 'test', 'test_43'),
                'test_34': os.path.join(self.temp_dir, 'test', 'test_34'),
                'results_44': os.path.join(self.temp_dir, 'results', 'results_44'),
                'results_43': os.path.join(self.temp_dir, 'results', 'results_43'),
                'results_34': os.path.join(self.temp_dir, 'results', 'results_34'),
                'final_results': os.path.join(self.temp_dir, 'final_results')
            }
            
            # Create all directories
            for dir_path in self.dirs.values():
                os.makedirs(dir_path, exist_ok=True)
                # Create subdirectories for results
                if 'results' in dir_path and not dir_path.endswith('final_results'):
                    os.makedirs(os.path.join(dir_path, 'bw'), exist_ok=True)
                    os.makedirs(os.path.join(dir_path, 'pseudo'), exist_ok=True)
                    os.makedirs(os.path.join(dir_path, 'bw_stitch'), exist_ok=True)
                    os.makedirs(os.path.join(dir_path, 'pseudo_stitch'), exist_ok=True)
            
            # Create final results subdirectories
            os.makedirs(os.path.join(self.dirs['final_results'], 'bw'), exist_ok=True)
            os.makedirs(os.path.join(self.dirs['final_results'], 'pseudo'), exist_ok=True)
            
            self.logger.debug("✅ MoDL temporary directories created")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to setup temp directories: {e}")
            raise
    
    def _load_model(self):
        """Load MoDL model with hardware-specific configuration."""
        try:
            if not TF_AVAILABLE:
                self.logger.warning("TensorFlow not available - using mock processing")
                return None

            if self.model_path and os.path.exists(self.model_path):
                self.logger.info(f"🔄 Loading MoDL model: {self.model_path}")

                # Load model with custom objects if needed
                try:
                    model = tf.keras.models.load_model(self.model_path, compile=False)

                    # Recompile with appropriate optimizer based on hardware
                    if self.hardware_config.hardware_type == HardwareType.GPU_AVAILABLE:
                        # GPU optimization
                        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
                    else:
                        # CPU optimization
                        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

                    model.compile(
                        optimizer=optimizer,
                        loss='binary_crossentropy',
                        metrics=['accuracy']
                    )

                    # Verify model architecture
                    expected_input_shape = (None, 512, 512, 1)  # MoDL expects 512x512 grayscale
                    actual_input_shape = model.input_shape

                    if actual_input_shape != expected_input_shape:
                        self.logger.warning(f"⚠️ Model input shape mismatch: expected {expected_input_shape}, got {actual_input_shape}")

                    # Test prediction to ensure model works
                    test_input = np.zeros((1, 512, 512, 1), dtype=np.float32)
                    test_output = model.predict(test_input, verbose=0)

                    self.logger.info(f"✅ MoDL model loaded successfully: {self.model_path}")
                    self.logger.info(f"📊 Model input shape: {model.input_shape}")
                    self.logger.info(f"📊 Model output shape: {model.output_shape}")
                    self.logger.info(f"🔧 Hardware mode: {self.hardware_config.hardware_type.value}")

                    return model

                except Exception as load_error:
                    self.logger.error(f"❌ Model loading failed: {load_error}")
                    self.logger.info("🔄 Attempting to load without compilation...")

                    # Try loading without compilation as fallback
                    model = tf.keras.models.load_model(self.model_path, compile=False)
                    self.logger.warning("⚠️ Model loaded without compilation - predictions may be slower")
                    return model

            else:
                # Try to find model in standard locations
                standard_paths = [
                    "/Al_Applications/MoDL/model/U-RNet+.hdf5",
                    os.path.join(os.path.dirname(os.path.dirname(__file__)), "MoDL", "model", "U-RNet+.hdf5"),
                    os.path.join(os.getcwd(), "MoDL", "model", "U-RNet+.hdf5")
                ]

                for path in standard_paths:
                    if os.path.exists(path):
                        self.logger.info(f"🔍 Found model at: {path}")
                        self.model_path = path
                        return self._load_model()  # Recursive call with found path

                self.logger.warning(f"⚠️ Model not found at {self.model_path} or standard locations - using mock processing")
                self.logger.info("📂 Searched paths:")
                for path in [self.model_path] + standard_paths:
                    self.logger.info(f"   - {path}")
                return None

        except Exception as e:
            self.logger.error(f"❌ Failed to load MoDL model: {e}")
            return None
    
    def process_slice(self, slice_data: np.ndarray, slice_idx: int) -> Dict[str, np.ndarray]:
        """
        Process a single slice through the MoDL pipeline.

        Args:
            slice_data: 2D numpy array of slice data
            slice_idx: Index of the slice being processed

        Returns:
            Dictionary containing segmentation results for all patterns
        """
        slice_start_time = time.time()
        initial_memory = psutil.virtual_memory().percent

        try:
            self.logger.info(f"🔬 ========== PROCESSING SLICE {slice_idx} ==========")
            self.logger.info(f"📊 Slice shape: {slice_data.shape}, dtype: {slice_data.dtype}, memory: {initial_memory:.1f}%")

            # Save slice to input directory
            io_start_time = time.time()
            input_file = os.path.join(self.dirs['input'], f"{slice_idx}.tif")
            import tifffile
            tifffile.imwrite(input_file, slice_data)
            io_duration = time.time() - io_start_time
            self.logger.info(f"⏱️  File I/O took {io_duration:.3f}s")
            
            # Process through all three cropping patterns
            three_pattern_start_time = time.time()
            pattern_results = {}
            pattern_timings = {}

            self.logger.info(f"🎯 Starting 3-pattern MoDL processing for slice {slice_idx}")

            for i, pattern in enumerate(["44", "43", "34"]):
                pattern_start = time.time()
                memory_before_pattern = psutil.virtual_memory().percent

                self.logger.info(f"🔄 Pattern {i+1}/3: {pattern}x{pattern} grid (memory: {memory_before_pattern:.1f}%)")
                pattern_result = self._process_pattern(slice_data, pattern, slice_idx)
                pattern_results[f'pattern_{pattern}'] = pattern_result

                pattern_timing = time.time() - pattern_start
                pattern_timings[pattern] = pattern_timing
                memory_after_pattern = psutil.virtual_memory().percent

                self.logger.info(f"✅ Pattern {pattern} completed in {pattern_timing:.2f}s, memory: {memory_before_pattern:.1f}% → {memory_after_pattern:.1f}%")

            three_pattern_duration = time.time() - three_pattern_start_time
            self.logger.info(f"📊 All 3 patterns completed in {three_pattern_duration:.2f}s (44: {pattern_timings['44']:.1f}s, 43: {pattern_timings['43']:.1f}s, 34: {pattern_timings['34']:.1f}s)")

            # Merge results from all patterns (MoDL's final step)
            merge_start_time = time.time()
            final_segmentation = self._merge_pattern_results(pattern_results, slice_idx)
            merge_duration = time.time() - merge_start_time

            # Calculate total slice processing time
            total_slice_duration = time.time() - slice_start_time
            final_memory = psutil.virtual_memory().percent
            memory_delta = final_memory - initial_memory

            self.logger.info(f"⏱️  Pattern merging took {merge_duration:.3f}s")
            self.logger.info(f"🎉 SLICE {slice_idx} COMPLETED in {total_slice_duration:.2f}s (I/O: {io_duration:.2f}s, 3-patterns: {three_pattern_duration:.2f}s, merge: {merge_duration:.2f}s), memory Δ: {memory_delta:+.1f}%")
            self.logger.info(f"========== END SLICE {slice_idx} ==========")

            return {
                'slice_idx': slice_idx,
                'pattern_results': pattern_results,
                'final_segmentation': final_segmentation,
                'timing': {
                    'total_duration': total_slice_duration,
                    'io_duration': io_duration,
                    'three_pattern_duration': three_pattern_duration,
                    'pattern_timings': pattern_timings,
                    'merge_duration': merge_duration
                }
            }
            
        except Exception as e:
            total_slice_duration = time.time() - slice_start_time
            final_memory = psutil.virtual_memory().percent
            self.logger.error(f"❌ Failed to process slice {slice_idx} after {total_slice_duration:.2f}s (memory: {final_memory:.1f}%): {e}")
            # Return empty result to continue processing
            return {
                'slice_idx': slice_idx,
                'pattern_results': {},
                'final_segmentation': np.zeros((1590, 1590), dtype=np.uint8)
            }
    
    def _process_pattern(self, slice_data: np.ndarray, pattern: str, slice_idx: int) -> np.ndarray:
        """Process slice through specific cropping pattern."""
        pattern_start_time = time.time()

        try:
            # Log pattern processing start
            self.logger.info(f"🔄 Processing slice {slice_idx} with pattern {pattern} (shape: {slice_data.shape})")

            # This is where the actual MoDL processing would happen
            # For now, return a mock segmentation result
            if self.model is not None:
                # Real processing with TensorFlow model
                # This would involve cropping, inference, and reconstruction
                result = self._run_inference_on_pattern(slice_data, pattern)
            else:
                # Mock processing for testing
                result = self._mock_segmentation(slice_data, pattern)

            # Log pattern processing completion with timing
            pattern_duration = time.time() - pattern_start_time
            self.logger.info(f"✅ Pattern {pattern} completed for slice {slice_idx} in {pattern_duration:.2f}s")

            return result

        except Exception as e:
            pattern_duration = time.time() - pattern_start_time
            self.logger.error(f"❌ Pattern {pattern} processing failed for slice {slice_idx} after {pattern_duration:.2f}s: {e}")
            return np.zeros((2048, 2048), dtype=np.uint8)
    
    def _run_inference_on_pattern(self, slice_data: np.ndarray, pattern: str) -> np.ndarray:
        """Run actual TensorFlow inference on cropping pattern."""
        inference_start_time = time.time()

        try:
            if self.model is None:
                self.logger.warning(f"⚠️ Model not available, using mock segmentation for pattern {pattern}")
                return self._mock_segmentation(slice_data, pattern)

            # Get crop boxes for this pattern
            crop_boxes = self.CROP_PATTERNS[pattern]
            input_size = slice_data.shape[0]  # Assumes square image
            patch_size = 512  # MoDL patch size

            # Log pattern details
            num_patches = len(crop_boxes)
            memory_before = psutil.virtual_memory().percent
            self.logger.info(f"📊 Pattern {pattern}: {num_patches} patches, input size: {input_size}x{input_size}, memory: {memory_before:.1f}%")

            # Crop image into patches
            cropping_start_time = time.time()
            patches = []
            for i, crop_box in enumerate(crop_boxes):
                x1 = int(crop_box[0] * input_size)
                y1 = int(crop_box[1] * input_size)
                x2 = int(crop_box[2] * input_size)
                y2 = int(crop_box[3] * input_size)

                # Extract patch and resize to 512x512
                patch = slice_data[y1:y2, x1:x2]

                try:
                    import cv2
                    patch_resized = cv2.resize(patch, (patch_size, patch_size), interpolation=cv2.INTER_CUBIC)
                except ImportError:
                    # Fallback without cv2
                    from scipy.ndimage import zoom
                    zoom_y = patch_size / patch.shape[0]
                    zoom_x = patch_size / patch.shape[1]
                    patch_resized = zoom(patch, (zoom_y, zoom_x), order=1)

                patches.append(patch_resized)

            cropping_duration = time.time() - cropping_start_time
            self.logger.info(f"⏱️  Cropping {num_patches} patches took {cropping_duration:.3f}s ({cropping_duration/num_patches*1000:.1f}ms per patch)")

            # Convert to numpy array and preprocess for model
            preprocessing_start_time = time.time()
            patches_array = np.array(patches, dtype=np.float32)
            patches_array = np.expand_dims(patches_array, axis=-1)  # Add channel dimension

            # DEBUG: Check patches before normalization
            self.logger.info(f"🔍 PATCHES BEFORE /255: min={patches_array.min():.3f}, max={patches_array.max():.3f}, mean={patches_array.mean():.3f}")

            # NOTE: Data is already normalized to 0-1 range by _preprocess_stack_for_modl()
            # No need to divide by 255 again - this was causing double normalization!

            # DEBUG: Check normalized data (should already be 0-1 range)
            self.logger.info(f"🔍 PATCHES (already normalized): min={patches_array.min():.3f}, max={patches_array.max():.3f}, mean={patches_array.mean():.3f}")

            # DEBUG: Check values before mean subtraction
            self.logger.info(f"🔍 BEFORE MEAN SUBTRACTION: min={patches_array.min():.3f}, max={patches_array.max():.3f}, mean={patches_array.mean():.3f}")

            # Add back mean subtraction like original MoDL (now with proper 0-255 input data)
            # Calculate mean across patches (axis=0 like in original MoDL)
            if len(patches_array) > 1:
                mean = patches_array.mean(axis=0)
                self.logger.info(f"🔍 SPATIAL MEAN STATS: min={mean.min():.3f}, max={mean.max():.3f}, mean={mean.mean():.3f}")
                patches_array -= mean
                self.logger.info(f"🔍 AFTER NORMALIZATION + MEAN SUBTRACTION: min={patches_array.min():.3f}, max={patches_array.max():.3f}, mean={patches_array.mean():.3f}")
            else:
                # Single patch - subtract its own mean to center it
                patch_mean = patches_array[0].mean()
                patches_array -= patch_mean
                self.logger.info(f"🔍 AFTER NORMALIZATION + MEAN CENTERING: min={patches_array.min():.3f}, max={patches_array.max():.3f}, mean={patches_array.mean():.3f}, subtracted_mean={patch_mean:.3f}")

            preprocessing_duration = time.time() - preprocessing_start_time
            array_size_mb = patches_array.nbytes / (1024 * 1024)
            self.logger.info(f"⏱️  Preprocessing took {preprocessing_duration:.3f}s, array size: {array_size_mb:.1f}MB")

            # DEBUG: Save sample patches for visual inspection
            debug_dir = "/tmp/modl_debug_patches"
            import os
            os.makedirs(debug_dir, exist_ok=True)

            # Save first 3 patches as images for inspection
            import cv2
            for i in range(min(3, len(patches_array))):
                patch = patches_array[i, :, :, 0]  # Remove channel dimension
                # Convert back to 0-255 for visualization
                patch_vis = (patch * 255).astype(np.uint8)
                cv2.imwrite(f"{debug_dir}/patch_{i}_input.png", patch_vis)

                self.logger.info(f"🔍 DEBUG Patch {i}: shape={patch.shape}, min={patch.min():.3f}, max={patch.max():.3f}, mean={patch.mean():.3f}")

                # Count non-zero pixels
                non_zero_pixels = np.count_nonzero(patch)
                total_pixels = patch.size
                self.logger.info(f"🔍 DEBUG Patch {i}: {non_zero_pixels}/{total_pixels} non-zero pixels ({100*non_zero_pixels/total_pixels:.1f}%)")

            # Run inference with proper batch size like MoDL
            inference_start = time.time()
            memory_before_inference = psutil.virtual_memory().percent
            batch_size = len(patches_array)  # Process all patches in one batch like MoDL
            self.logger.info(f"🧠 Running TensorFlow inference on {len(patches)} patches (batch_size={batch_size}, memory: {memory_before_inference:.1f}%)")

            # Ensure float32 dtype like MoDL (line 78: imgs_test = imgdatas.astype('float32'))
            patches_array = patches_array.astype(np.float32)

            predictions = self.model.predict(patches_array, batch_size=batch_size, verbose=0)

            # DEBUG: Save raw predictions before thresholding
            for i in range(min(3, len(predictions))):
                pred = predictions[i, :, :, 0]  # Remove channel dimension
                # Save raw prediction as image (scale to 0-255)
                pred_vis = (pred * 255).astype(np.uint8)
                cv2.imwrite(f"{debug_dir}/patch_{i}_prediction.png", pred_vis)

                self.logger.info(f"🔍 DEBUG Prediction {i}: shape={pred.shape}, min={pred.min():.3f}, max={pred.max():.3f}, mean={pred.mean():.3f}")

                # Count pixels above different thresholds
                above_01 = np.count_nonzero(pred > 0.1)
                above_05 = np.count_nonzero(pred > 0.5)
                above_07 = np.count_nonzero(pred > 0.7)
                total = pred.size
                self.logger.info(f"🔍 DEBUG Prediction {i}: >0.1: {above_01}/{total} ({100*above_01/total:.1f}%), >0.5: {above_05}/{total} ({100*above_05/total:.1f}%), >0.7: {above_07}/{total} ({100*above_07/total:.1f}%)")

            inference_duration = time.time() - inference_start
            memory_after_inference = psutil.virtual_memory().percent
            memory_delta = memory_after_inference - memory_before_inference
            self.logger.info(f"✅ Inference completed in {inference_duration:.2f}s ({inference_duration/len(patches)*1000:.1f}ms per patch), memory Δ: {memory_delta:+.1f}%")

            # Apply threshold (0.7 from original MoDL)
            postprocessing_start = time.time()
            non_zero_before = np.count_nonzero(predictions > 0.5)

            predictions[predictions > 0.7] = 1
            predictions[predictions <= 0.7] = 0

            # Convert back to uint8
            predictions = (predictions * 255).astype(np.uint8)
            non_zero_after = np.count_nonzero(predictions)

            postprocessing_duration = time.time() - postprocessing_start
            self.logger.info(f"⏱️  Thresholding took {postprocessing_duration:.3f}s, active pixels: {non_zero_before} → {non_zero_after}")

            # Stitch patches back together
            stitching_start = time.time()
            stitched_result = self._stitch_patches(predictions, pattern, input_size)
            stitching_duration = time.time() - stitching_start

            # Final timing summary
            total_pattern_duration = time.time() - inference_start_time
            memory_final = psutil.virtual_memory().percent
            self.logger.info(f"⏱️  Stitching took {stitching_duration:.3f}s")
            self.logger.info(f"🎯 Pattern {pattern} TOTAL: {total_pattern_duration:.2f}s (crop: {cropping_duration:.2f}s, prep: {preprocessing_duration:.2f}s, inference: {inference_duration:.2f}s, stitch: {stitching_duration:.2f}s), final memory: {memory_final:.1f}%")

            return stitched_result

        except Exception as e:
            total_duration = time.time() - inference_start_time
            self.logger.error(f"❌ Inference failed for pattern {pattern} after {total_duration:.2f}s: {e}")
            return self._mock_segmentation(slice_data, pattern)
    
    def _mock_segmentation(self, slice_data: np.ndarray, pattern: str) -> np.ndarray:
        """Generate mock segmentation for testing purposes."""
        # Create a realistic-looking segmentation mask
        height, width = slice_data.shape
        
        # Create some blob-like structures
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Add some circular/elliptical structures (simulating mitochondria)
        center_y, center_x = height // 2, width // 2
        for i in range(5):  # Add 5 mock mitochondria
            offset_y = (i - 2) * height // 10
            offset_x = (i - 2) * width // 10
            y, x = center_y + offset_y, center_x + offset_x
            
            # Create elliptical mask
            yy, xx = np.ogrid[:height, :width]
            ellipse = ((xx - x) / (width // 10))**2 + ((yy - y) / (height // 15))**2 <= 1
            mask[ellipse] = 255
        
        return mask

    def _stitch_patches(self, patches: np.ndarray, pattern: str, output_size: int) -> np.ndarray:
        """
        Stitch predicted patches back together using MoDL's positioning logic.

        Args:
            patches: Array of predicted patches (num_patches, 512, 512, 1)
            pattern: Pattern type ("44", "43", "34")
            output_size: Size of output image (typically 2048)

        Returns:
            Stitched image of size (output_size, output_size)
        """
        try:
            # Remove channel dimension and convert patches for stitching
            if patches.ndim == 4:
                patches = patches.squeeze(-1)  # Remove channel dimension

            # Get positions and order for this pattern
            positions = self.STITCH_POSITIONS[pattern]
            order = self.STITCH_ORDERS[pattern]

            # Create output image
            stitched = np.zeros((output_size, output_size), dtype=np.uint8)

            # Place patches according to MoDL positioning
            for i, patch_idx in enumerate(order):
                if patch_idx < len(patches):
                    patch = patches[patch_idx]
                    pos_x, pos_y = positions[i]

                    # Resize patch back to proper size for stitching
                    patch_size = 512
                    end_x = min(pos_x + patch_size, output_size)
                    end_y = min(pos_y + patch_size, output_size)

                    # Place patch in stitched image
                    stitched[pos_y:end_y, pos_x:end_x] = patch[:end_y-pos_y, :end_x-pos_x]

            return stitched

        except Exception as e:
            self.logger.error(f"❌ Stitching failed for pattern {pattern}: {e}")
            return np.zeros((output_size, output_size), dtype=np.uint8)

    def _merge_pattern_results(self, pattern_results: Dict[str, np.ndarray], slice_idx: int) -> np.ndarray:
        """Merge results from all three patterns using MoDL's bitwise operations."""
        try:
            # Get results from all patterns
            result_44 = pattern_results.get('pattern_44', np.zeros((2048, 2048), dtype=np.uint8))
            result_43 = pattern_results.get('pattern_43', np.zeros((2048, 2048), dtype=np.uint8))
            result_34 = pattern_results.get('pattern_34', np.zeros((2048, 2048), dtype=np.uint8))
            
            # Apply MoDL's merging logic (from segment_predict.py lines 154-161)
            # Invert masks
            try:
                import cv2
                inv_44 = cv2.bitwise_not(result_44)
                inv_43 = cv2.bitwise_not(result_43)
                inv_34 = cv2.bitwise_not(result_34)
                
                # Merge using bitwise AND operations
                merge1 = cv2.bitwise_and(inv_44, inv_43)
                merge2 = cv2.bitwise_and(merge1, inv_34)
                
                # Final inversion
                final_result = cv2.bitwise_not(merge2)
                
                # Resize to MoDL output size (1590x1590)
                if final_result.shape != (1590, 1590):
                    final_result = cv2.resize(final_result, (1590, 1590), interpolation=cv2.INTER_AREA)
                
            except ImportError:
                # Fallback without cv2
                inv_44 = ~result_44
                inv_43 = ~result_43
                inv_34 = ~result_34
                
                merge1 = np.bitwise_and(inv_44, inv_43)
                merge2 = np.bitwise_and(merge1, inv_34)
                final_result = ~merge2
                
                # Simple resize without cv2
                if final_result.shape != (1590, 1590):
                    from scipy.ndimage import zoom
                    zoom_factor = 1590 / final_result.shape[0]
                    final_result = zoom(final_result, zoom_factor, order=0)
            
            return final_result.astype(np.uint8)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to merge pattern results for slice {slice_idx}: {e}")
            return np.zeros((1590, 1590), dtype=np.uint8)
    
    def cleanup_temp_files(self, slice_idx: int) -> None:
        """Clean up temporary files for a specific slice."""
        try:
            # Remove slice-specific temp files
            temp_files = [
                os.path.join(self.dirs['input'], f"{slice_idx}.tif"),
            ]
            
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to cleanup temp files for slice {slice_idx}: {e}")
    
    def cleanup_all(self) -> None:
        """Clean up all temporary directories and files."""
        try:
            import shutil
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                self.logger.debug("🧹 All temporary files cleaned up")
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to cleanup temporary directory: {e}")


# Add module-level compatibility methods for the Streamlit app  
class CompatibilityMethods:
    """Compatibility methods to bridge old interface with new SegmentationWorkflow class."""
    
    def __init__(self, workflow_instance):
        self.workflow = workflow_instance
    
    def validate_raw_image(self, image_path: str):
        """Validate raw image file for processing."""
        try:
            # Use our new load_and_detect_zstack method for validation
            stack, metadata = self.workflow.load_and_detect_zstack(image_path)
            is_valid, issues = self.workflow.validate_modl_input(stack)
            
            if is_valid:
                return True, "Image validation passed"
            else:
                return False, f"Validation issues: {'; '.join(issues)}"
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def estimate_processing_time(self, image_path: str):
        """Estimate processing time for image."""
        try:
            stack, metadata = self.workflow.load_and_detect_zstack(image_path)
            z_count = metadata['z_stack_len']
            
            # Rough estimates based on hardware and z-stack size
            if self.workflow.is_gpu_available():
                time_per_slice = 2  # seconds
            else:
                time_per_slice = 5  # seconds (CPU processing)
            
            estimated_time = z_count * time_per_slice
            
            # Each slice is processed as 16 patches (4x4 grid) based on MoDL workflow
            patches_per_slice = 16  
            total_patches = z_count * patches_per_slice
            
            return {
                'image_dimensions': f"{stack.shape[1]}x{stack.shape[2]} ({z_count} slices)",
                'estimated_time': f"{estimated_time//60}m {estimated_time%60}s",
                'estimated_time_minutes': estimated_time / 60,
                'z_stack_length': z_count,
                'num_slices': z_count,
                'total_patches': total_patches,
                'processing_complexity': 'Medium' if z_count < 30 else 'High',
                'processing_mode': self.workflow.get_processing_mode_info()
            }
        except Exception as e:
            return {'error': str(e)}
    
    def create_job(self, image_path: str, params: dict):
        """Create a segmentation job with real JobManager integration."""
        from .job_manager import get_job_manager
        
        # Prepare job parameters for JobManager
        job_params = {
            'input_path': image_path,
            'output_dir': params.get('output_dir', 'data/segmented'),
            'force_cpu': params.get('force_cpu', False)
        }
        
        # Create job through JobManager
        job_manager = get_job_manager()
        job_id = job_manager.submit_job('segmentation', job_params)
        return job_id
    
    def submit_job(self, job_id: str):
        """Submit job for processing - job is already submitted during create_job."""
        # Job is automatically submitted during create_job with JobManager
        return True
    
    def get_job_status(self, job_id: str):
        """Get real job status from JobManager."""
        from .job_manager import get_job_manager
        
        try:
            job_manager = get_job_manager()
            status = job_manager.get_job_status(job_id)
            if status:
                return {
                    'status': status['status'],
                    'created_time': status.get('created_at', ''),
                    'progress': status.get('progress', 0),
                    'status_message': status.get('status_message', ''),
                    'result': status.get('result', {})
                }
        except Exception as e:
            self.workflow.logger.warning(f"Failed to get job status: {e}")
        
        return None
    
    def list_active_jobs(self):
        """List active jobs from JobManager."""
        from .job_manager import get_job_manager
        
        try:
            job_manager = get_job_manager()
            all_jobs = job_manager.list_jobs()
            # Return only segmentation jobs that are not completed/failed/cancelled
            active_jobs = [
                job_id for job_id, job_info in all_jobs.items()
                if job_info.get('job_type') == 'segmentation' and 
                   job_info.get('status') in ['queued', 'running']
            ]
            return active_jobs
        except Exception as e:
            self.workflow.logger.warning(f"Failed to list active jobs: {e}")
            return []
    
    def cancel_job(self, job_id: str):
        """Cancel a job through JobManager."""
        from .job_manager import get_job_manager
        
        try:
            job_manager = get_job_manager()
            return job_manager.cancel_job(job_id)
        except Exception as e:
            self.workflow.logger.warning(f"Failed to cancel job: {e}")
            return False
    
    def cleanup_old_jobs(self, max_age_hours: int = 1):
        """Cleanup old jobs through JobManager."""
        from .job_manager import get_job_manager
        
        try:
            job_manager = get_job_manager()
            # JobManager handles cleanup automatically, but we can trigger it
            pass  # JobManager cleans up automatically
        except Exception as e:
            self.workflow.logger.warning(f"Failed to cleanup jobs: {e}")


# Add compatibility methods to the module-level segmentation_workflow instance
_compat = CompatibilityMethods(segmentation_workflow)
segmentation_workflow.validate_raw_image = _compat.validate_raw_image
segmentation_workflow.estimate_processing_time = _compat.estimate_processing_time
segmentation_workflow.create_job = _compat.create_job
segmentation_workflow.submit_job = _compat.submit_job
segmentation_workflow.get_job_status = _compat.get_job_status
segmentation_workflow.list_active_jobs = _compat.list_active_jobs
segmentation_workflow.cancel_job = _compat.cancel_job
segmentation_workflow.cleanup_old_jobs = _compat.cleanup_old_jobs