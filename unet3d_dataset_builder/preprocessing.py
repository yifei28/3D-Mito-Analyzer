import numpy as np

import numpy as np

def extract_single_channel_gray(volume: np.ndarray, tile_idx: int = 0, channel_idx: int = 0) -> np.ndarray:
    """
    Extract a single channel from a multi-channel CZI volume and return it as a grayscale 3D volume.
    Args:
        volume (np.ndarray): Input CZI volume of shape (H, C, Z, Y, X)
        tile_idx (int): Which tile to use (default 0)
        channel_idx (int): Which channel to extract (default 0)
    Returns:
        np.ndarray: Grayscale volume of shape (Z, Y, X)
    """
    assert volume.ndim == 5, f"Expected 5D input (H, C, Z, Y, X), got shape {volume.shape}"
    assert 0 <= tile_idx < volume.shape[0], f"tile_idx {tile_idx} out of range"
    assert 0 <= channel_idx < volume.shape[1], f"channel_idx {channel_idx} out of range"
    # Extract and return the selected channel
    gray_volume = volume[tile_idx, channel_idx]  # shape (Z, Y, X)
    return gray_volume

def process_label_volume(label_volume: np.ndarray, threshold: int = 127) -> np.ndarray:
    """
    Convert a 4D label volume (Z, H, W, C) to 3D (Z, H, W) by selecting the first channel,
    binarizing it, and validating the result only contains 0 and 1.

    Args:
        label_volume (np.ndarray): Input label volume of shape (Z, H, W, C)
        threshold (int): Threshold for binarization (default 127)

    Returns:
        np.ndarray: Binary label volume of shape (Z, H, W)

    Raises:
        ValueError: If the result contains values other than 0 or 1
    """
    assert label_volume.ndim == 4 and label_volume.shape[-1] >= 1, \
        f"Expected shape (Z, H, W, C≥1), got {label_volume.shape}"

    # Extract first channel (Z, H, W)
    label = label_volume[..., 0]

    # Binarize the label: values > threshold → 1, else → 0
    binary_label = (label > threshold).astype(np.uint8)

    # Validate the output contains only 0 and 1
    unique_vals = np.unique(binary_label)
    if not np.all(np.isin(unique_vals, [0, 1])):
        raise ValueError(f"Binarized label contains invalid values: {unique_vals}. Only 0 and 1 allowed.")

    return binary_label


def analyze_label_channels(label_volume: np.ndarray, slice_idx: int = 0):
    """
    Analyze all 3 channels of a given slice in a 4D label volume (Z, H, W, C).

    Args:
        label_volume (np.ndarray): Label volume of shape (Z, H, W, 3)
        slice_idx (int): Index of the slice to analyze (default 0)
    """
    assert label_volume.ndim == 4 and label_volume.shape[-1] == 3, f"Expected shape (Z, H, W, 3), got {label_volume.shape}"
    assert 0 <= slice_idx < label_volume.shape[0], f"slice_idx out of range: {slice_idx}"

    print(f"Analyzing slice {slice_idx}:")

    for ch in range(3):
        slice_channel = label_volume[slice_idx, :, :, ch]
        unique_vals = np.unique(slice_channel)
        print(f"  Channel {ch}: unique values = {unique_vals}")


from typing import List, Tuple

def extract_paired_patches(image: np.ndarray, label: np.ndarray,
                           x: int, stride: int, min_positive_ratio: float = 0.0
                          ) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Extract paired (image, label) patches from 3D volume using sliding window,
    and filter by minimum positive voxel ratio in label.

    Args:
        image (np.ndarray): 3D image of shape (Z, H, W)
        label (np.ndarray): 3D label of shape (Z, H, W)
        x (int): Patch height and width (output will be (Z, x, x))
        stride (int): Sliding window stride in height and width
        min_positive_ratio (float): Minimum ratio of positive voxels (0-1) to include patch

    Returns:
        List[Tuple[np.ndarray, np.ndarray]]: Paired image and label patches
    """
    assert image.shape == label.shape, "Image and label must have the same shape"
    assert image.ndim == 3, "Image and label must be 3D (Z, H, W)"
    assert 0.0 <= min_positive_ratio <= 1.0, "min_positive_ratio must be between 0 and 1"

    Z, H, W = image.shape
    patches = []

    for i in range(0, H - x + 1, stride):
        for j in range(0, W - x + 1, stride):
            patch_img = image[:, i:i+x, j:j+x]
            patch_lbl = label[:, i:i+x, j:j+x]

            # Apply positive ratio filter
            total_voxels = patch_lbl.size
            positive_voxels = np.sum(patch_lbl == 1)
            ratio = positive_voxels / total_voxels

            if ratio >= min_positive_ratio:
                patches.append((patch_img, patch_lbl))

    return patches

def is_positive_patch(label_patch: np.ndarray, threshold: float = 0.01) -> bool:
    """
    Check whether the proportion of positive voxels (==1) in the label patch
    exceeds the given threshold.

    Args:
        label_patch (np.ndarray): 3D label of shape (Z, x, x) with values 0 or 1
        threshold (float): Ratio threshold, e.g., 0.01 means 1% positive voxels

    Returns:
        bool: True if positive voxel ratio >= threshold, False otherwise
    """
    assert label_patch.ndim == 3, f"Expected 3D label patch, got shape {label_patch.shape}"

    total_voxels = np.prod(label_patch.shape)   # 32 * x * x
    positive_voxels = np.sum(label_patch == 1)
    ratio = positive_voxels / total_voxels

    return ratio >= threshold