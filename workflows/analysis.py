"""
Analysis Workflow Wrapper for MitoNetworkAnalyzer
Provides Streamlit-friendly interface with error handling and structured output
"""

import logging
import time
import psutil
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import streamlit as st
import numpy as np
from collections import Counter

# Import the existing analyzer
from Analyzer import MitoNetworkAnalyzer


class AnalysisWorkflow:
    """
    Streamlit-friendly wrapper for MitoNetworkAnalyzer with enhanced error handling,
    progress tracking, and structured output for web interface integration.
    """
    
    def __init__(self):
        """Initialize the analysis workflow wrapper."""
        self.logger = self._setup_logging()
        self.results_cache = {}
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration for the workflow."""
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
        
    def validate_parameters(self, file_path: str, xRes: float, yRes: float, 
                          zRes: float, zDepth: int) -> Tuple[bool, Optional[str]]:
        """
        Validate input parameters for analysis.
        
        Args:
            file_path: Path to the TIFF file
            xRes: X resolution in micrometers per pixel
            yRes: Y resolution in micrometers per pixel  
            zRes: Z resolution in micrometers per slice
            zDepth: Number of Z slices
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check file existence and readability
            if not os.path.exists(file_path):
                return False, f"File does not exist: {file_path}"
                
            if not os.access(file_path, os.R_OK):
                return False, f"File is not readable: {file_path}"
                
            # Check file extension
            file_ext = Path(file_path).suffix.lower()
            if file_ext not in ['.tif', '.tiff']:
                return False, f"Invalid file type: {file_ext}. Expected .tif or .tiff"
                
            # Validate resolution parameters
            if not all(isinstance(param, (int, float)) and param > 0 
                      for param in [xRes, yRes, zRes]):
                return False, "Resolution parameters (xRes, yRes, zRes) must be positive numbers"
                
            # Validate zDepth
            if not isinstance(zDepth, int) or zDepth <= 0:
                return False, "zDepth must be a positive integer"
                
            # Check reasonable parameter ranges
            if any(res > 10.0 for res in [xRes, yRes, zRes]):
                return False, "Resolution values seem unusually large (>10 μm). Please verify units."
                
            if zDepth > 1000:
                return False, f"zDepth ({zDepth}) seems unusually large. Please verify."
                
            # Check file size for memory estimation
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            if file_size_mb > 2000:  # 2GB
                return False, f"File size ({file_size_mb:.1f} MB) may cause memory issues"
                
            self.logger.info(f"Parameter validation passed for {Path(file_path).name}")
            return True, None
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def estimate_memory_usage(self, file_path: str, zDepth: int) -> Dict[str, float]:
        """
        Estimate memory usage for the analysis.
        
        Args:
            file_path: Path to the TIFF file
            zDepth: Number of Z slices
            
        Returns:
            Dictionary with memory estimates in GB
        """
        try:
            file_size_gb = os.path.getsize(file_path) / (1024**3)
            
            # Rough estimates based on typical processing
            estimates = {
                'file_size_gb': file_size_gb,
                'estimated_peak_gb': file_size_gb * 3.5,  # Original + labeled + processing
                'available_memory_gb': psutil.virtual_memory().available / (1024**3),
                'memory_usage_percent': (file_size_gb * 3.5) / (psutil.virtual_memory().total / (1024**3)) * 100
            }
            
            return estimates
            
        except Exception as e:
            self.logger.error(f"Memory estimation error: {e}")
            return {'error': str(e)}
    
    def run_analysis(self, file_path: str, xRes: float, yRes: float, 
                    zRes: float, zDepth: int, show_progress: bool = True) -> Dict[str, Any]:
        """
        Run the complete mitochondrial network analysis with error handling and progress tracking.
        
        Args:
            file_path: Path to the segmented TIFF file
            xRes: X resolution in micrometers per pixel
            yRes: Y resolution in micrometers per pixel
            zRes: Z resolution in micrometers per slice
            zDepth: Number of Z slices in the stack
            show_progress: Whether to show Streamlit progress bar
            
        Returns:
            Dictionary with analysis results or error information
        """
        start_time = time.time()
        
        # Create result structure
        result = {
            'success': False,
            'file_path': file_path,
            'parameters': {
                'xRes': xRes,
                'yRes': yRes, 
                'zRes': zRes,
                'zDepth': zDepth
            },
            'network_count': 0,
            'total_volume': 0.0,
            'volume_distribution': {},
            'largest_network_id': None,
            'processing_time': 0.0,
            'memory_usage': {},
            'labeled_image': None,
            'error_message': None,
            'warnings': []
        }
        
        try:
            # Step 1: Parameter validation
            if show_progress:
                progress_bar = st.progress(0)
                status_text = st.empty()
                status_text.text("Validating parameters...")
                
            is_valid, error_msg = self.validate_parameters(file_path, xRes, yRes, zRes, zDepth)
            if not is_valid:
                result['error_message'] = error_msg
                self.logger.error(f"Parameter validation failed: {error_msg}")
                return result
                
            if show_progress:
                progress_bar.progress(10)
                status_text.text("Estimating memory usage...")
                
            # Step 2: Memory estimation
            memory_info = self.estimate_memory_usage(file_path, zDepth)
            result['memory_usage'] = memory_info
            
            if 'error' not in memory_info:
                if memory_info['memory_usage_percent'] > 80:
                    result['warnings'].append(
                        f"High memory usage expected: {memory_info['memory_usage_percent']:.1f}%"
                    )
                    
            if show_progress:
                progress_bar.progress(20)
                status_text.text("Initializing MitoNetworkAnalyzer...")
            
            # Step 3: Initialize analyzer with custom print suppression
            self.logger.info(f"Starting analysis of {Path(file_path).name}")
            
            # Suppress print statements from MitoNetworkAnalyzer
            import sys
            from io import StringIO
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            
            try:
                analyzer = MitoNetworkAnalyzer(file_path, xRes, yRes, zRes, zDepth)
            finally:
                captured_output = sys.stdout.getvalue()
                sys.stdout = old_stdout
                
            if show_progress:
                progress_bar.progress(60)
                status_text.text("Extracting analysis results...")
            
            # Step 4: Extract results
            result['network_count'] = analyzer.network_count
            result['total_volume'] = analyzer.total_mito_volume
            result['volume_distribution'] = analyzer.volumes.copy()
            result['labeled_image'] = analyzer.labeled.copy() if analyzer.labeled is not None else None
            
            # Find largest network
            if analyzer.volumes:
                result['largest_network_id'] = max(analyzer.volumes, key=analyzer.volumes.get)
            
            if show_progress:
                progress_bar.progress(80)
                status_text.text("Computing additional metrics...")
                
            # Step 5: Additional analysis
            try:
                # Compute volume statistics
                if analyzer.volumes:
                    volumes_array = np.array(list(analyzer.volumes.values()))
                    result['volume_statistics'] = {
                        'mean_volume': float(np.mean(volumes_array)),
                        'median_volume': float(np.median(volumes_array)),
                        'std_volume': float(np.std(volumes_array)),
                        'min_volume': float(np.min(volumes_array)),
                        'max_volume': float(np.max(volumes_array)),
                        'total_volume': float(np.sum(volumes_array))
                    }
                    
                # Z-spread analysis
                z_spread = analyzer.fast_label_z_spread()
                multi_slice_networks = sum(1 for z_count in z_spread.values() if z_count > 1)
                result['z_spread_analysis'] = {
                    'networks_spanning_multiple_slices': multi_slice_networks,
                    'single_slice_networks': len(z_spread) - multi_slice_networks,
                    'average_z_span': float(np.mean(list(z_spread.values()))) if z_spread else 0.0
                }
                
            except Exception as e:
                result['warnings'].append(f"Additional analysis warning: {str(e)}")
                self.logger.warning(f"Additional analysis failed: {e}")
            
            if show_progress:
                progress_bar.progress(100)
                status_text.text("Analysis complete!")
                
            result['success'] = True
            result['processing_time'] = time.time() - start_time
            
            self.logger.info(
                f"Analysis completed successfully: {result['network_count']} networks found, "
                f"total volume: {result['total_volume']:.2f} μm³, "
                f"processing time: {result['processing_time']:.1f}s"
            )
            
            # Cache results for potential reuse
            cache_key = f"{file_path}_{xRes}_{yRes}_{zRes}_{zDepth}"
            self.results_cache[cache_key] = result.copy()
            
        except MemoryError:
            error_msg = "Insufficient memory to process this image. Try closing other applications or use a smaller image."
            result['error_message'] = error_msg
            self.logger.error(error_msg)
            
        except FileNotFoundError as e:
            error_msg = f"File not found: {str(e)}"
            result['error_message'] = error_msg
            self.logger.error(error_msg)
            
        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"
            result['error_message'] = error_msg
            self.logger.error(f"Unexpected error during analysis: {e}", exc_info=True)
            
        finally:
            result['processing_time'] = time.time() - start_time
            
            # Clean up progress indicators
            if show_progress and 'progress_bar' in locals():
                progress_bar.empty()
            if show_progress and 'status_text' in locals():
                status_text.empty()
        
        return result
    
    def get_cached_result(self, file_path: str, xRes: float, yRes: float, 
                         zRes: float, zDepth: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached analysis result if available.
        
        Args:
            file_path: Path to the TIFF file
            xRes, yRes, zRes: Resolution parameters
            zDepth: Number of Z slices
            
        Returns:
            Cached result dictionary or None if not found
        """
        cache_key = f"{file_path}_{xRes}_{yRes}_{zRes}_{zDepth}"
        return self.results_cache.get(cache_key)
    
    def clear_cache(self):
        """Clear the results cache."""
        self.results_cache.clear()
        self.logger.info("Results cache cleared")
        
    def format_results_summary(self, result: Dict[str, Any]) -> str:
        """
        Format analysis results into a human-readable summary.
        
        Args:
            result: Analysis result dictionary
            
        Returns:
            Formatted summary string
        """
        if not result['success']:
            return f"❌ Analysis failed: {result['error_message']}"
        
        summary_lines = [
            f"✅ Analysis completed successfully in {result['processing_time']:.1f}s",
            f"📊 Networks found: {result['network_count']}",
            f"🔬 Total volume: {result['total_volume']:.2f} μm³"
        ]
        
        if 'volume_statistics' in result:
            stats = result['volume_statistics']
            summary_lines.extend([
                f"📈 Average network volume: {stats['mean_volume']:.2f} μm³",
                f"📉 Largest network: {stats['max_volume']:.2f} μm³"
            ])
            
        if 'z_spread_analysis' in result:
            z_info = result['z_spread_analysis']
            summary_lines.append(
                f"🗂️ Multi-slice networks: {z_info['networks_spanning_multiple_slices']}/{result['network_count']}"
            )
        
        if result['warnings']:
            summary_lines.append(f"⚠️ Warnings: {len(result['warnings'])}")
            
        return "\n".join(summary_lines)


# Global instance for use throughout the application
analysis_workflow = AnalysisWorkflow()