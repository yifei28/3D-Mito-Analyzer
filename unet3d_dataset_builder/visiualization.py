import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def visualize_volume_with_label_slider(image: np.ndarray, label: np.ndarray):
    """
    Visualize image and label slices side by side with a slider to scroll through Z slices.

    Args:
        image (np.ndarray): Image volume of shape (Z, H, W)
        label (np.ndarray): Label volume of shape (Z, H, W)
    """
    assert image.shape == label.shape, f"Image and label must have the same shape, got {image.shape} and {label.shape}"
    assert image.ndim == 3, f"Expected 3D volumes, got shape {image.shape}"

    z_slices = image.shape[0]

    # Create figure and initial display
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    plt.subplots_adjust(bottom=0.2)  # Leave space for slider

    img_ax, label_ax = axes
    img_disp = img_ax.imshow(image[0], cmap='gray')
    label_disp = label_ax.imshow(label[0], cmap='gray')

    img_ax.set_title("Image Slice")
    label_ax.set_title("Label Slice")
    for ax in axes:
        ax.axis('off')

    # Add slider below the images
    ax_slider = plt.axes([0.25, 0.05, 0.5, 0.03])
    slice_slider = Slider(ax_slider, 'Slice', 0, z_slices - 1, valinit=0, valstep=1)

    # Update function
    def update(val):
        z = int(slice_slider.val)
        img_disp.set_data(image[z])
        label_disp.set_data(label[z])
        fig.canvas.draw_idle()

    slice_slider.on_changed(update)
    plt.show()


def visualize_h5_sample_with_slider(filepath):
    """
    Visualize 3D image and label slices from a .h5 sample file using a slider.

    Args:
        filepath (str): Path to the .h5 file containing 'raw' and 'label' datasets.
    """
    with h5py.File(filepath, 'r') as f:
        raw = f['raw'][:]
        label = f['label'][:]

    assert raw.shape == label.shape, "Raw and label shapes must match"
    assert raw.ndim == 3, "Expect 3D raw and label volumes (Z, H, W)"

    num_slices = raw.shape[0]

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(f"Visualizing {filepath.split('/')[-1]}", fontsize=12)

    # Initial slices
    raw_img = ax1.imshow(raw[0], cmap='gray')
    ax1.set_title("Raw Image Slice")
    ax1.axis('off')

    label_img = ax2.imshow(label[0], cmap='hot', vmin=0, vmax=1)
    ax2.set_title(f"Label Slice\nPos voxels: {np.sum(label[0] == 1)}")
    ax2.axis('off')

    # Add slider
    # Add slider
    ax_slider = plt.axes([0.25, 0.03, 0.5, 0.02], facecolor='lightgray')
    slider = Slider(ax_slider, 'Slice', 0, num_slices - 1, valinit=0, valstep=1)

    def update(val):
        idx = int(slider.val)
        raw_img.set_data(raw[idx])
        label_img.set_data(label[idx])
        ax2.set_title(f"Label Slice\nPos voxels: {np.sum(label[idx] == 1)}")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.tight_layout()
    plt.show()

def visualize_test_h5_with_slider(h5_path: str):
    """
    Visualize a test .h5 file containing only a 'raw' dataset using a slice slider.

    Args:
        h5_path (str): Path to the test .h5 file with dataset 'raw' (Z, H, W)
    """
    with h5py.File(h5_path, 'r') as f:
        raw = f['raw'][:]

    assert raw.ndim == 3, f"'raw' must be 3D (Z, H, W), but got shape {raw.shape}"

    z_slices = raw.shape[0]

    # Set up figure
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    plt.subplots_adjust(bottom=0.2)

    img_disp = ax.imshow(raw[0], cmap='gray')
    ax.set_title("Test Image Slice")
    ax.axis('off')

    # Slider
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slice_slider = Slider(ax_slider, 'Slice', 0, z_slices - 1, valinit=0, valstep=1)

    def update(val):
        z = int(slice_slider.val)
        img_disp.set_data(raw[z])
        fig.canvas.draw_idle()

    slice_slider.on_changed(update)
    plt.show()