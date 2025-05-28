import os
import numpy as np
import tifffile
from aicspylibczi import CziFile


def read_czi(filepath: str) -> np.ndarray:
    # Read a .czi file and return image as a numpy array
    czi = CziFile(filepath)
    image, _ = czi.read_image()
    image = np.squeeze(image)
    return image

def read_tif(filepath: str) -> np.ndarray:
    # Read a .tif file and return image as a numpy array
    image = tifffile.imread(filepath)
    return image

def read_image(filepath: str) -> np.ndarray:
    # Automatically choose reader based on file extension
    ext = os.path.splitext(filepath)[-1].lower()
    if ext in ['.tif', '.tiff']:
        return read_tif(filepath)
    elif ext == '.czi':
        return read_czi(filepath)
    else:
        raise ValueError(f"Unsupported file extension: {ext}.")

