import numpy as np
import cv2
import os
import tifffile
from glob import glob
import segment_predict

def modify_tif_dim(img_path, in_dim, out_dim):
    if (in_dim == out_dim):
        print("No modification needed.")
        return
    imgs = glob(os.path.join(img_path, "*.tif"))
    print(f"Found {len(imgs)} images to resize.")

    for img in imgs:
        img_data = cv2.imread(img, cv2.IMREAD_UNCHANGED)
        # Resize the image to the target dimensions
        resized_img = cv2.resize(img_data, (out_dim, out_dim), interpolation=cv2.INTER_CUBIC)
        # Save the resized image
        cv2.imwrite(img, resized_img)

def read_zstack(img_path):
    stack = tifffile.imread(img_path)
    data_folder_path = f"/Al_Applications/MoDL/temp/"
    segment_predict.clear_files(data_folder_path)

    for z in range(stack.shape[0]):
        img = stack[z]
        # Perform any additional processing on img here
        # For example, save the image or display it
        tifffile.imwrite(os.path.join(data_folder_path, f"{z}.tif"), img)

    modify_tif_dim(data_folder_path, stack[0].shape[0], target_dim)

if __name__ == "__main__":
    target_dim = 1590
    img_path = "/Al_Applications/MoDL/images/Site1.tif"
    read_zstack(img_path)
    print("Image processing completed.")