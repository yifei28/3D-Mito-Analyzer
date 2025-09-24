#!/usr/bin/env python3
"""
Test the segmentation pipeline with the official MoDL test image
to verify that removing the double normalization fixes the all-black output.
"""

import sys
import os
import numpy as np

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from workflows.segmentation import SegmentationWorkflow

def test_official_image():
    """Test segmentation on official MoDL test image."""

    # Initialize segmentation with official test image
    image_path = "/Users/yifei/mito-analyzer/data/raw/0.tif"
    print(f"🧪 Testing segmentation on official MoDL image: {image_path}")

    try:
        # Create segmentation instance
        seg = SegmentationWorkflow()
        print("✅ Successfully created segmentation workflow")

        # Setup output directory
        output_dir = "/tmp/segmentation_test"
        os.makedirs(output_dir, exist_ok=True)

        # Run segmentation
        print("\n🔬 Running segmentation...")

        def progress_callback(slice_idx, info):
            print(f"Processing slice {slice_idx}, info: {info}")

        result_info = seg.run_segmentation(image_path, output_dir, progress_callback)

        # Check results
        if result_info and "output_path" in result_info:
            output_path = result_info["output_path"]
            print(f"✅ Segmentation completed!")
            print(f"📁 Output saved to: {output_path}")

            if os.path.exists(output_path):
                # Load and analyze the result
                import tifffile
                result = tifffile.imread(output_path)
                print(f"📊 Result shape: {result.shape}")
                print(f"📊 Result range: [{result.min():.6f}, {result.max():.6f}]")
                print(f"📊 Non-zero pixels: {np.count_nonzero(result)}/{result.size}")
                print(f"📊 Percentage non-zero: {100.0 * np.count_nonzero(result) / result.size:.2f}%")

                if np.count_nonzero(result) > 0:
                    print("🎉 SUCCESS: Segmentation detected mitochondria!")
                    return True
                else:
                    print("❌ FAILURE: All pixels are zero (still black output)")
                    return False
            else:
                print(f"❌ FAILURE: Output file not found at {output_path}")
                return False
        else:
            print("❌ FAILURE: Segmentation failed to return output path")
            return False

    except Exception as e:
        print(f"❌ ERROR during segmentation: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_official_image()
    sys.exit(0 if success else 1)