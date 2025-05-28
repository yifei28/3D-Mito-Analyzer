from visiualization import visualize_h5_sample_with_slider
from visiualization import visualize_test_h5_with_slider
import matplotlib
import os
import random

if __name__ == '__main__':
    matplotlib.use('TkAgg')
    output_root = "128_64_005"
    all_samples = len([f for f in os.listdir(os.path.join(output_root, "All_Sample")) if f.endswith(".h5")])
    train_count = len([f for f in os.listdir(os.path.join(output_root, "train")) if f.endswith(".h5")])
    val_count = len([f for f in os.listdir(os.path.join(output_root, "val")) if f.endswith(".h5")])
    test_count = len([f for f in os.listdir(os.path.join(output_root, "test_with_truth")) if f.endswith(".h5")])

    print(f"all: {all_samples}")
    print(f"Train: {train_count}")
    print(f"Val: {val_count}")
    print(f"Test: {test_count}")
    print(f"Total: {train_count + val_count + test_count}")

    all_sample_dir = os.path.join(output_root, "All_Sample")
    all_files = [f for f in os.listdir(all_sample_dir) if f.endswith(".h5")]
    random_all_file = random.choice(all_files)
    print(f"[INFO] Visualizing ALL_SAMPLE file: {random_all_file}")
    visualize_h5_sample_with_slider(os.path.join(all_sample_dir, random_all_file))

    # test_dir = "64_32_010/test"
    # test_files = [f for f in os.listdir(test_dir) if f.endswith(".h5")]
    # random_test_file = random.choice(test_files)
    # print(f"[INFO] Visualizing TEST file: {random_test_file}")
    # visualize_test_h5_with_slider(os.path.join(test_dir, random_test_file))