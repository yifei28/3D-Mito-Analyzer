import os
import h5py
import shutil
import random
from typing import Tuple

def split_dataset_by_ratio(input_dir, train_ratio, val_ratio, test_ratio,
                           output_train_dir, output_val_dir, output_test_dir,
                           file_list=None):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    files = file_list if file_list is not None else [
        f for f in os.listdir(input_dir) if f.endswith('.h5')]

    random.shuffle(files)

    total = len(files)
    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)
    n_test = total - n_train - n_val

    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]

    os.makedirs(output_train_dir, exist_ok=True)
    os.makedirs(output_val_dir, exist_ok=True)
    os.makedirs(output_test_dir, exist_ok=True)

    def copy_files(file_list, dest_dir):
        for f in file_list:
            src = os.path.join(input_dir, f)
            dst = os.path.join(dest_dir, f)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)

    copy_files(train_files, output_train_dir)
    copy_files(val_files, output_val_dir)
    copy_files(test_files, output_test_dir)

    print(f"[INFO] Total: {total}")
    print(f"[INFO] Train: {len(train_files)} → {output_train_dir}")
    print(f"[INFO] Val:   {len(val_files)} → {output_val_dir}")
    print(f"[INFO] Test:  {len(test_files)} → {output_test_dir}")


def remove_label_from_h5_files(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for fname in os.listdir(input_dir):
        if not fname.endswith(".h5"):
            continue
        src = os.path.join(input_dir, fname)
        dst = os.path.join(output_dir, fname)
        with h5py.File(src, 'r') as fsrc:
            data = fsrc['raw'][:]
        with h5py.File(dst, 'w') as fdst:
            fdst.create_dataset('raw', data=data)

if __name__ == "__main__":
    split_dataset_by_ratio("All_Sample", 0.8, 0.1, 0.1, "train", "val", "test_with_truth")
    remove_label_from_h5_files("test_with_truth", "test")