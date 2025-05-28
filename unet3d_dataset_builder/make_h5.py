import os
import h5py
import numpy as np
from typing import List, Tuple

def save_patches_to_h5(patches: List[Tuple[np.ndarray, np.ndarray]],
                       prefix: str,
                       save_dir: str,
                       specific_name: str = None):
    """
    Save image-label patches to .h5 files.

    Args:
        patches: List of (image, label) tuples, shape (Z, H, W)
        prefix: File prefix (ignored if specific_name is used)
        save_dir: Destination directory
        specific_name: If set, save only one sample using this exact filename
    """
    os.makedirs(save_dir, exist_ok=True)

    if specific_name:
        # Save only the first patch with a specific filename
        img, lbl = patches[0]
        fpath = os.path.join(save_dir, specific_name)
        with h5py.File(fpath, 'w') as f:
            f.create_dataset('raw', data=img.astype('uint8'))
            f.create_dataset('label', data=lbl.astype('uint8'))
        return

    # Normal batch save mode
    for i, (img, lbl) in enumerate(patches, start=1):
        fname = f"{prefix}_{i}.h5"
        fpath = os.path.join(save_dir, fname)
        with h5py.File(fpath, 'w') as f:
            f.create_dataset('raw', data=img.astype('uint8'))
            f.create_dataset('label', data=lbl.astype('uint8'))
