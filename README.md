# 3D-Mito-Analyzer

In this branch, image processing method is introduced to acquire training dataset for 3D-UNet. The main purpose of the python scripts in this branch is to extract 2D images from a z-stack images, and their dimension can be specified with arguments in api.
 
The 2D UNet is used for mitochondria segmentation is from this source https://github.com/OBPNPW2024/MoDL.git. Replace the original segment_predict.py with this new version, in which multiprocessing is used to accelerate the process. User can use image_preprocess.py to extract the images and post_process.py to stitch them together back to a z-stack.

From past experience, it takes about 40 minutes to process 32 images.
