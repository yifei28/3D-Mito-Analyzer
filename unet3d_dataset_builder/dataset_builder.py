# dataset_builder.py
import os
from io_utils import read_image
from preprocessing import process_label_volume, extract_single_channel_gray, extract_paired_patches
from make_h5 import save_patches_to_h5
from split import split_dataset_by_ratio, remove_label_from_h5_files

def preprocess_image_and_label(image_path, label_path):
    image = read_image(image_path)
    label = read_image(label_path)
    image_gray = extract_single_channel_gray(image)
    label_bin = process_label_volume(label)
    return image_gray, label_bin

def extract_valid_patches(image, label, patch_size, stride, min_ratio):
    patches = extract_paired_patches(image, label, patch_size, stride, min_ratio)
    print(f"[INFO] Total valid patches: {len(patches)}")
    return patches

def save_all_patches(patches, output_prefix, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    existing = [f for f in os.listdir(output_dir) if f.endswith(".h5")]
    start_idx = len(existing) + 1

    new_files = []
    for i, (img, lbl) in enumerate(patches, start=start_idx):
        fname = f"{output_prefix}_{i}.h5"
        save_patches_to_h5([(img, lbl)], prefix="", save_dir=output_dir, specific_name=fname)
        new_files.append(fname)
    return new_files

def split_and_clean(output_root, all_file_names, split_ratio):
    all_patch_dir = os.path.join(output_root, "All_Sample")
    train_dir = os.path.join(output_root, "train")
    val_dir = os.path.join(output_root, "val")
    test_with_truth_dir = os.path.join(output_root, "test_with_truth")
    test_dir = os.path.join(output_root, "test")

    split_dataset_by_ratio(
        input_dir=all_patch_dir,
        train_ratio=split_ratio[0],
        val_ratio=split_ratio[1],
        test_ratio=split_ratio[2],
        output_train_dir=train_dir,
        output_val_dir=val_dir,
        output_test_dir=test_with_truth_dir,
        file_list=all_file_names
    )

    remove_label_from_h5_files(test_with_truth_dir, test_dir)
