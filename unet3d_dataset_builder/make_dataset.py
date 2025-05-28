import os
import h5py
from typing import Tuple
from io_utils import read_image
from preprocessing import extract_single_channel_gray
from preprocessing import process_label_volume
from preprocessing import extract_paired_patches
from make_h5 import save_patches_to_h5
from split import split_dataset_by_ratio
from split import remove_label_from_h5_files

def preprocess_and_split(
    label_path: str,
    image_path: str,
    patch_size: int,
    stride: int,
    min_positive_ratio: float,
    output_prefix: str,
    output_root: str,
    split: Tuple[float, float, float] = (0.8, 0.1, 0.1)
):
    assert sum(split) == 1.0, "Split ratios must sum to 1.0"

    # Step 1: Load data
    label = read_image(label_path)
    image = read_image(image_path)

    # Step 2: Process label and image
    label_bin = process_label_volume(label)
    image_gray = extract_single_channel_gray(image)

    # Step 3: Extract patches
    patches = extract_paired_patches(image_gray, label_bin, patch_size, stride, min_positive_ratio)
    print(f"[INFO] Total valid patches: {len(patches)}")

    # Step 4: Save all patches to All_Sample with custom names
    all_patch_dir = os.path.join(output_root, "All_Sample")
    os.makedirs(all_patch_dir, exist_ok=True)

    existing_files = [f for f in os.listdir(all_patch_dir) if f.endswith(".h5")]
    start_idx = len(existing_files) + 1

    new_files = []
    for i, (img, lbl) in enumerate(patches, start=start_idx):
        fname = f"{output_prefix}_{i}.h5"
        save_patches_to_h5([(img, lbl)], prefix="", save_dir=all_patch_dir, specific_name=fname)
        new_files.append(fname)

    # Step 5: Split only new files
    train_dir = os.path.join(output_root, "train")
    val_dir = os.path.join(output_root, "val")
    test_with_truth_dir = os.path.join(output_root, "test_with_truth")

    split_dataset_by_ratio(
        input_dir=all_patch_dir,
        train_ratio=split[0],
        val_ratio=split[1],
        test_ratio=split[2],
        output_train_dir=train_dir,
        output_val_dir=val_dir,
        output_test_dir=test_with_truth_dir,
        file_list=new_files
    )

    # Step 6: Remove label from test set
    test_dir = os.path.join(output_root, "test")
    remove_label_from_h5_files(test_with_truth_dir, test_dir)

    print(f"[DONE] {output_prefix} complete. Output saved under {output_root}")