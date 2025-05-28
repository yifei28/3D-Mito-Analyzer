import os
import argparse
import random
from visiualization import visualize_h5_sample_with_slider

def count_h5(dir_path):
    return len([f for f in os.listdir(dir_path) if f.endswith(".h5")])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize .h5 samples from a dataset folder.")
    parser.add_argument('--root', type=str, required=True,
                        help='Path to dataset root folder (e.g., 128_64_005)')
    args = parser.parse_args()
    output_root = args.root

    try:
        all_sample_dir = os.path.join(output_root, "All_Sample")
        all_files = [f for f in os.listdir(all_sample_dir) if f.endswith(".h5")]
        random_all_file = random.choice(all_files)

        print(f"all: {count_h5(all_sample_dir)}")
        print(f"Train: {count_h5(os.path.join(output_root, 'train'))}")
        print(f"Val: {count_h5(os.path.join(output_root, 'val'))}")
        print(f"Test: {count_h5(os.path.join(output_root, 'test_with_truth'))}")
        print(f"Total: {count_h5(os.path.join(output_root, 'train')) + count_h5(os.path.join(output_root, 'val')) + count_h5(os.path.join(output_root, 'test_with_truth'))}")

        print(f"[INFO] Visualizing ALL_SAMPLE file: {random_all_file}")
        visualize_h5_sample_with_slider(os.path.join(all_sample_dir, random_all_file))
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        print("[HINT] Make sure the --root path is correct and contains All_Sample/")
