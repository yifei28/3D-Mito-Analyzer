import os
import argparse
import yaml
from pathlib import Path
from dataset_builder import preprocess_image_and_label, extract_valid_patches, save_all_patches, split_and_clean


if __name__ == '__main__':
    # Load config from YAML
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yml', help='Path to config file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    patch_size = cfg["patch_size"]
    stride = cfg["stride"]
    min_positive_ratio = cfg["min_positive_ratio"]
    output_root = cfg["output_root"]
    split_ratio = tuple(cfg["split_ratio"])
    data_root = Path(cfg["data_root"])
    dish_sites = cfg["dish_sites"]

    for dish, sites in dish_sites.items():
        for site in sites:
            label_path = data_root / f"{dish} Label" / f"Site{site}.tif"
            image_path = data_root / f"{dish} Image" / f"Site{site}.czi"
            prefix = f"{dish.replace(' ', '')}_Site{site}"

            if not label_path.exists():
                print(f"[WARNING] Label not found: {label_path}")
                continue
            if not image_path.exists():
                print(f"[WARNING] Image not found: {image_path}")
                continue

            image, label = preprocess_image_and_label(str(image_path), str(label_path))
            patches = extract_valid_patches(image, label, patch_size, stride, min_positive_ratio)
            patch_dir = os.path.join(output_root, "All_Sample")
            new_files = save_all_patches(patches, prefix, patch_dir)
            split_and_clean(output_root, new_files, split_ratio)

    def count(dir):
        return len([f for f in os.listdir(dir) if f.endswith(".h5")])

    print(f"all: {count(os.path.join(output_root, 'All_Sample'))}")
    print(f"Train: {count(os.path.join(output_root, 'train'))}")
    print(f"Val: {count(os.path.join(output_root, 'val'))}")
    print(f"Test: {count(os.path.join(output_root, 'test_with_truth'))}")
