import numpy as np
import tifffile
import os

def combine_tif_to_zstack(input_dir, output_path):
    # List and sort input files (ensure correct order)
    tif_files = [f for f in os.listdir(input_dir) if f.endswith(('.tif', '.tiff'))]
    tif_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))  # Sort by slice number
    print(tif_files)

    # Read all slices into a list
    slices = []
    for filename in tif_files:
        print(f"Reading file: {filename}")
        filepath = os.path.join(input_dir, filename)
        img = tifffile.imread(filepath)
        slices.append(img)

    # Convert list to 3D numpy array (Z-stack)
    z_stack = np.stack(slices, axis=0)  # Shape: (depth, height, width)

    # Save as a Z-stack TIFF
    tifffile.imwrite(output_path, z_stack, imagej=True)  # Use `imagej=True` for compatibility with ImageJ


if __name__ == "__main__":
    input_dir = "/Al_Applications/MoDL/processed_images/"  # Directory containing individual TIFF slices
    output_path = "/Al_Applications/MoDL/final_images/Site5_63x_4xMean_Zstack_1-2.tif"  # Output path for the Z-stack TIFF
    combine_tif_to_zstack(input_dir, output_path)
    print("Z-stack TIFF created successfully.")