#!/usr/bin/env python3
"""
Direct Segmentation Test

Tests the segmentation workflow directly without job manager overhead
to verify the input_shape fix works.
"""

import os
import tempfile
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from workflows.segmentation import SegmentationWorkflow, HardwareConfig, HardwareType


def test_segmentation_direct():
    """Test segmentation workflow directly."""
    print("🧪 Testing segmentation workflow directly...")

    # Use existing test image
    test_image = "/Users/yifei/mito-analyzer/MoDL/test/test_44/0015.tif"
    if not os.path.exists(test_image):
        print(f"❌ Test image not found: {test_image}")
        return False

    print(f"  ✓ Using test image: {test_image}")

    # Create output directory
    output_dir = tempfile.mkdtemp()
    print(f"  ✓ Output directory: {output_dir}")

    try:
        # Create segmentation workflow
        hardware_config = HardwareConfig(HardwareType.CPU_ONLY, "CPU")
        model_path = "/Users/yifei/mito-analyzer/MoDL/model/U-RNet+.hdf5"

        workflow = SegmentationWorkflow(model_path, hardware_config)
        print("  ✓ Segmentation workflow created")

        # Simple progress callback
        def progress_callback(percent, info):
            stage = info.get('stage', 'unknown')
            message = info.get('message', '')
            print(f"    Progress: {percent}% - {stage}: {message}")

        # Run segmentation
        print("  🔄 Running segmentation...")
        result = workflow.run_segmentation(
            input_path=test_image,
            output_dir=output_dir,
            progress_callback=progress_callback
        )

        print(f"  📊 Result structure: {result}")

        if result:
            success = result.get('success', False)
            print(f"  📊 Success flag: {success}")

            # Check if output file exists directly in result
            output_file = result.get('output_file')
            if output_file and os.path.exists(output_file):
                print("  ✅ Segmentation completed successfully!")
                print(f"    Output file: {output_file}")
                print(f"    Processing time: {result.get('processing_time_seconds', 0):.2f}s")
                file_size = os.path.getsize(output_file)
                print(f"    Output file size: {file_size} bytes")
                return True
            elif success:
                print("  ✅ Segmentation reported success!")
                print(f"    Processing time: {result.get('processing_time_seconds', 0):.2f}s")
                return True
            else:
                error = result.get('error', 'No specific error')
                print(f"  ❌ Segmentation failed: {error}")
                return False
        else:
            print("  ❌ Segmentation failed: No result returned")
            return False

    except Exception as e:
        print(f"  ❌ Segmentation failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Cleanup
        try:
            import shutil
            shutil.rmtree(output_dir)
            print(f"  ✓ Cleaned up output directory")
        except:
            pass


def main():
    """Run direct segmentation test."""
    print("🚀 DIRECT SEGMENTATION TEST")
    print("=" * 50)
    print("Testing segmentation workflow with input_shape fix")
    print()

    success = test_segmentation_direct()

    print("\n" + "=" * 50)
    print("🎯 DIRECT SEGMENTATION TEST RESULTS")
    print("=" * 50)

    if success:
        print("🎉 SUCCESS: Segmentation workflow works correctly!")
        print("✅ The input_shape bug has been fixed")
        print("📋 Status: Segmentation functionality is OPERATIONAL!")
    else:
        print("⚠️ Segmentation test failed")
        print("❌ Further debugging may be needed")

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)