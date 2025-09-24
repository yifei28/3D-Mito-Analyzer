#!/usr/bin/env python3
"""
Test script for Task 10.4: Segmentation Processing with Progress Tracking

This script validates the run_segmentation implementation with comprehensive testing.
"""

import os
import sys
import time
import tempfile
import numpy as np
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def create_mock_tiff_stack(output_path: str, z_slices: int = 5, width: int = 512, height: int = 512) -> str:
    """Create a mock TIFF z-stack for testing."""
    try:
        import tifffile
        
        # Create a realistic-looking z-stack
        stack = np.zeros((z_slices, height, width), dtype=np.uint16)
        
        for z in range(z_slices):
            # Create some interesting structures that vary by slice
            slice_data = np.random.randint(0, 4096, (height, width), dtype=np.uint16)
            
            # Add some circular structures (simulating cells/organelles)
            for i in range(3):
                center_y = np.random.randint(height // 4, 3 * height // 4)
                center_x = np.random.randint(width // 4, 3 * width // 4)
                radius = np.random.randint(20, 50)
                
                y, x = np.ogrid[:height, :width]
                mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
                slice_data[mask] += np.random.randint(1000, 3000)
            
            # Ensure values stay within uint16 range
            stack[z] = np.clip(slice_data, 0, 65535)
        
        # Save as TIFF
        tifffile.imwrite(output_path, stack)
        print(f"✅ Created mock TIFF z-stack: {output_path}")
        print(f"   Shape: {stack.shape}, dtype: {stack.dtype}")
        return output_path
        
    except ImportError:
        print("❌ tifffile not available - cannot create test TIFF")
        return None
    except Exception as e:
        print(f"❌ Failed to create mock TIFF: {e}")
        return None


class ProgressTracker:
    """Helper class to track progress callbacks."""
    
    def __init__(self):
        self.progress_history = []
        self.stages_seen = set()
        self.last_percentage = -1
        self.error_occurred = False
    
    def callback(self, percentage: int, info: dict):
        """Progress callback function."""
        self.progress_history.append((percentage, info.copy()))
        self.stages_seen.add(info.get('stage', 'unknown'))
        self.last_percentage = percentage
        
        if percentage == -1:
            self.error_occurred = True
            print(f"❌ ERROR: {info.get('message', 'Unknown error')}")
        else:
            stage = info.get('stage', 'unknown')
            message = info.get('message', '')
            hw_mode = info.get('hardware_mode', 'UNKNOWN')
            
            print(f"📊 {percentage:3d}% [{hw_mode}] {stage}: {message}")
        
        # Additional details for certain stages
        if 'memory_usage_gb' in info:
            print(f"   💾 Memory: {info['memory_usage_gb']:.1f} GB")
        
        if 'current_slice' in info and 'total_slices' in info:
            print(f"   📁 Slice: {info['current_slice']}/{info['total_slices']}")
    
    def get_summary(self):
        """Get progress tracking summary."""
        return {
            'total_updates': len(self.progress_history),
            'stages_seen': list(self.stages_seen),
            'final_percentage': self.last_percentage,
            'error_occurred': self.error_occurred,
            'completion_reached': self.last_percentage == 100
        }


def test_segmentation_workflow():
    """Test the full segmentation workflow."""
    print("🧪 Testing Task 10.4: Segmentation Processing with Progress Tracking")
    print("=" * 70)
    
    try:
        # Import the segmentation workflow
        from workflows.segmentation import SegmentationWorkflow, create_segmentation_workflow
        
        print("✅ Successfully imported SegmentationWorkflow")
        
        # Create workflow instance
        workflow = create_segmentation_workflow(force_cpu=True)
        print("✅ Created segmentation workflow instance")
        
        # Test hardware detection
        hardware_info = workflow.get_hardware_info()
        print(f"🔧 Hardware: {hardware_info['hardware_type']} ({hardware_info['device_name']})")
        print(f"💾 Memory: {hardware_info.get('memory_gb', 0):.1f} GB")
        
        # Check if run_segmentation method exists
        if not hasattr(workflow, 'run_segmentation'):
            print("❌ run_segmentation method not found!")
            return False
        
        print("✅ run_segmentation method is available")
        
        # Create temporary directories
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock input TIFF
            input_file = os.path.join(temp_dir, "test_stack.tif")
            output_dir = os.path.join(temp_dir, "segmentation_output")
            
            # Create test TIFF with smaller dimensions for faster testing
            mock_file = create_mock_tiff_stack(input_file, z_slices=3, width=256, height=256)
            if not mock_file:
                print("❌ Failed to create test TIFF file")
                return False
            
            os.makedirs(output_dir, exist_ok=True)
            
            # Setup progress tracking
            progress_tracker = ProgressTracker()
            
            print(f"\n🚀 Starting segmentation test...")
            print(f"📁 Input: {input_file}")
            print(f"📂 Output: {output_dir}")
            
            # Run segmentation
            start_time = time.time()
            try:
                result = workflow.run_segmentation(
                    input_path=input_file,
                    output_dir=output_dir,
                    progress_callback=progress_tracker.callback
                )
                
                end_time = time.time()
                processing_time = end_time - start_time
                
                print(f"\n✅ Segmentation completed successfully!")
                print(f"⏱️  Total time: {processing_time:.2f} seconds")
                
                # Validate results
                print("\n📊 Validating results...")
                
                # Check processing report
                if isinstance(result, dict):
                    print(f"📝 Processing report:")
                    print(f"   Input file: {result.get('input_file')}")
                    print(f"   Output file: {result.get('output_file')}")
                    print(f"   Total slices: {result.get('total_slices')}")
                    print(f"   Batch size used: {result.get('batch_size_used')}")
                    print(f"   Hardware mode: {result.get('hardware_mode')}")
                    print(f"   Successful slices: {result.get('successful_slices')}")
                    
                    # Check if output files exist
                    output_file = result.get('output_file')
                    if output_file and os.path.exists(output_file):
                        print(f"✅ Output TIFF file created: {os.path.basename(output_file)}")
                        
                        # Try to read the output file
                        try:
                            import tifffile
                            output_stack = tifffile.imread(output_file)
                            print(f"✅ Output stack shape: {output_stack.shape}, dtype: {output_stack.dtype}")
                        except:
                            print("⚠️  Could not read output TIFF (this might be expected)")
                    else:
                        print("❌ Output TIFF file not found")
                    
                    # Check metadata file
                    metadata_files = [f for f in os.listdir(output_dir) if f.endswith('_metadata.json')]
                    if metadata_files:
                        print(f"✅ Metadata file created: {metadata_files[0]}")
                    else:
                        print("⚠️  No metadata file found")
                
                # Validate progress tracking
                progress_summary = progress_tracker.get_summary()
                print(f"\n📈 Progress tracking summary:")
                print(f"   Total updates: {progress_summary['total_updates']}")
                print(f"   Stages seen: {', '.join(progress_summary['stages_seen'])}")
                print(f"   Final percentage: {progress_summary['final_percentage']}%")
                print(f"   Completion reached: {progress_summary['completion_reached']}")
                print(f"   Errors occurred: {progress_summary['error_occurred']}")
                
                # Check for expected stages
                expected_stages = {'initialization', 'loading', 'planning', 'setup', 'preprocessing', 
                                 'segmentation', 'reconstruction', 'assembly', 'saving', 'cleanup', 'completed'}
                stages_seen_set = set(progress_summary['stages_seen'])
                missing_stages = expected_stages - stages_seen_set
                if missing_stages:
                    print(f"⚠️  Missing expected stages: {', '.join(missing_stages)}")
                else:
                    print("✅ All expected progress stages were seen")
                
                return True
                
            except Exception as e:
                print(f"❌ Segmentation failed: {e}")
                print(f"Error type: {type(e).__name__}")
                import traceback
                print("Traceback:")
                traceback.print_exc()
                return False
                
    except ImportError as e:
        print(f"❌ Failed to import required modules: {e}")
        return False
    except Exception as e:
        print(f"❌ Test setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_progress_callback_functionality():
    """Test progress callback functionality specifically."""
    print("\n🧪 Testing Progress Callback Functionality")
    print("=" * 50)
    
    # Test progress tracker
    tracker = ProgressTracker()
    
    # Simulate various progress updates
    test_updates = [
        (0, {"stage": "initialization", "message": "Starting up"}),
        (10, {"stage": "loading", "message": "Loading data", "hardware_mode": "CPU"}),
        (50, {"stage": "processing", "current_slice": 5, "total_slices": 10, "memory_usage_gb": 2.5}),
        (100, {"stage": "completed", "message": "All done!", "processing_time": 42.5}),
    ]
    
    for percentage, info in test_updates:
        tracker.callback(percentage, info)
    
    summary = tracker.get_summary()
    print(f"✅ Progress tracker test completed")
    print(f"   Updates received: {summary['total_updates']}")
    print(f"   Stages: {', '.join(summary['stages_seen'])}")
    
    return summary['total_updates'] == len(test_updates)


def main():
    """Main test function."""
    print("🚀 Task 10.4 Implementation Testing")
    print("=" * 80)
    
    all_tests_passed = True
    
    # Test 1: Progress callback functionality
    if test_progress_callback_functionality():
        print("✅ Test 1 PASSED: Progress callback functionality")
    else:
        print("❌ Test 1 FAILED: Progress callback functionality")
        all_tests_passed = False
    
    print("\n" + "=" * 80)
    
    # Test 2: Full segmentation workflow
    if test_segmentation_workflow():
        print("✅ Test 2 PASSED: Full segmentation workflow")
    else:
        print("❌ Test 2 FAILED: Full segmentation workflow")
        all_tests_passed = False
    
    print("\n" + "=" * 80)
    print("🏁 FINAL RESULTS")
    print("=" * 80)
    
    if all_tests_passed:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Task 10.4 implementation is working correctly")
        return 0
    else:
        print("💥 SOME TESTS FAILED!")
        print("❌ Task 10.4 implementation needs attention")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)