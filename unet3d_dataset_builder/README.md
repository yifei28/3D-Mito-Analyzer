# unet3d_dataset_builder

A lightweight and modular pipeline for preparing 3D image segmentation datasets, especially suited for 3D U-Net training. Supports `.czi` and `.tif` input formats and exports labeled `.h5` patches organized into train/val/test splits.

## 📂 Project Structure

```
unet3d_dataset_builder/
├── config.yml                  # Dataset generation parameters
├── main.py                    # Entry script for dataset creation
├── dataset_builder.py         # Core pipeline logic (preprocessing, patching, saving, splitting)
├── preprocessing.py           # Image & label processing functions
├── io_utils.py                # File reading (.czi, .tif)
├── make_h5.py                 # Save image-label patches to .h5
├── split.py                   # Train/val/test splitting
├── visual.py                  # Visualization utilities
├── 128_64_005/                # Example output dataset
└── ...
```

## 🧠 Key Features

- Supports **.czi** and **.tif** input image formats
- Extracts **(Z, H, W)** patches from 3D volumes
- Filters patches based on **positive label ratio**
- Saves data in **HDF5 (.h5)** format
- Automatically splits into **train / val / test / test_without_label**
- Easy to configure via `config.yml`

## 🗂️ Input Data Structure

Your raw images and labels should be organized in the following folder structure:

```
Data/
├── Dish 1 Image/
│   ├── Site1.czi
│   ├── Site2.czi
│   └── Site3.czi
├── Dish 1 Label/
│   ├── Site1.tif
│   ├── Site2.tif
│   └── Site3.tif
├── Dish 2 Image/
│   ├── Site2.czi
│   ├── Site3.czi
│   ├── Site4.czi
│   └── Site5.czi
├── Dish 2 Label/
│   ├── Site2.tif
│   ├── Site3.tif
│   ├── Site4.tif
│   └── Site5.tif
├── Dish 3 Image/
│   ├── Site2.czi
│   ├── Site3.czi
│   ├── Site4.czi
│   └── Site5.czi
└── Dish 3 Label/
    ├── Site2.tif
    ├── Site3.tif
    ├── Site4.tif
    └── Site5.tif
```

### 📌 Notes:
- Each `.czi` file contains multi-channel 3D microscopy image data.
- Each `.tif` file is the corresponding semantic label (binary or multi-channel).
- The pairing is determined by folder + filename convention:  
  e.g., `Dish 2 Image/Site3.czi` ⬌ `Dish 2 Label/Site3.tif`
- The script automatically parses this structure using `dish_sites` defined in `config.yml`.

Example:

```yaml
dish_sites:
  Dish 1: [1, 2, 3]
  Dish 2: [2, 3, 4, 5]
  Dish 3: [2, 3, 4, 5]
```

## 📛 Dataset Naming Convention

## 🗂️ Output Dataset Naming Convention

Each generated dataset folder (e.g., `128_64_005`) follows the naming pattern:

```
<patch_size>_<stride>_<min_positive_ratio_in_thousandths>
```

### Example:
- `128_64_005` means:
  - Patch size = 128
  - Stride = 64
  - Minimum positive label ratio = 0.005 (i.e., 0.5%)

These folder names are used as `output_root` in `config.yml` and are where the `.h5` patches and splits are saved:

```
128_64_005/
├── All_Sample/        # All extracted patches
├── train/             # Training patches with labels
├── val/               # Validation patches with labels
├── test_with_truth/   # Test patches with labels
└── test/              # Test patches without labels (for inference)
```

## ⚙️ Configuration (`config.yml`)

Example:

```yaml
patch_size: 128
stride: 64
min_positive_ratio: 0.02
output_root: "128_64_005"
split_ratio: [0.8, 0.1, 0.1]
data_root: "../Data"

dish_sites:
  Dish 1: [1, 2, 3]
  Dish 2: [2, 3, 4, 5]
  Dish 3: [2, 3, 4, 5]
```

## 🚀 How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run dataset generation

```bash
python main.py --config config.yml
```


## 🖼️ Visualize Samples

```bash
python visual.py --root 128_64_005
```

This will:
- Count the number of `.h5` files in each split
- Randomly pick a sample from `All_Sample/` and open an interactive slice viewer

If the path is invalid or missing, a helpful error will be shown.


```bash
python visual.py
```

Randomly picks a `.h5` file and opens a slider viewer for slices.

## 📌 Notes

- `.czi` images are expected to have shape `(H, C, Z, Y, X)`
- Labels should be 4D: `(Z, H, W, C)`; only the first channel is binarized
- Label thresholding is set to `127` by default

## 📄 License

MIT License. Free to use and modify for research and educational purposes.

## 🙋‍♂️ Contact

If you encounter bugs or want to suggest improvements, feel free to open an issue or PR.

## ✅ To Do (Optional)

- [ ] Add data augmentation
- [ ] Add multiprocessing support
- [ ] Add CLI argument validation
