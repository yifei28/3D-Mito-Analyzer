from tifffile import imread, imwrite
from skimage import measure
from skimage.color import rgb2gray, label2rgb
from skimage.morphology import binary_erosion
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

class MitoNetworkAnalyzer:
    def __init__(self, imagePath : str, xRes : float, yRes : float, zRes : float, zDepth : int) -> None:
        self.image = imagePath
        self.zDepth = zDepth
        self.voxel_volume = xRes * yRes * zRes
        self.labeled = None
        self.network_count = 0
        self.volumes = {}
        self.total_mito_volume = 0

        self.label_image(imagePath)
        self.count_networks()
        self.analyze_volume()

    def label_image(self, image_path : str) -> None:
        # Load image
        image = imread(image_path)

        if self.is_rgb_array(image):
            print("Image is RGB, converting to grayscale.")
            image = rgb2gray(image)

        # Convert to binary if needed
        if not self.is_binary_array(image):
            image = (image > 0).astype(np.uint8)

        # Label in 3D
        image = binary_erosion(image)
        self.labeled = measure.label(image, connectivity=1)
        print("Labeled Image shape:", self.labeled.shape)
    
    def count_networks(self) -> None:
        if self.labeled is None:
            raise ValueError("Labeled image not found")
        
        unique_labels = np.unique(self.labeled)
        self.network_count = len(unique_labels) - 1  # Exclude background label (0)
        print("Number of mitochondrial networks:", self.network_count)

    
    def visualize_labeled_image(self) -> None:
        if self.labeled is None:
            raise ValueError("Labeled image not found.")

        num_slices = self.zDepth
        ncols = 6
        nrows = (num_slices + ncols - 1) // ncols

        fig, axs = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
        axs = axs.ravel()

        for i in range(num_slices):
            ax = axs[i]
            slice_2d = self.labeled[i, :, :]
            overlay = label2rgb(slice_2d, bg_label=0)
            ax.imshow(overlay)
            ax.set_title(f"Z = {i}")
            ax.axis('off')

        # Hide any unused subplots
        for j in range(num_slices, len(axs)):
            axs[j].axis('off')

        plt.tight_layout()
        plt.suptitle("Labeled Mitochondrial Networks (Z Slices)", fontsize=16)
        plt.subplots_adjust(top=0.92)
        plt.show()

    def is_binary_array(self, arr : np.ndarray) -> bool:
        return np.isin(arr, [0, 1]).all()

    def is_rgb_array(self, arr : np.ndarray) -> bool:
        if len(arr.shape) == 4 and arr.shape[3] == 3:
            return True
        elif (len(arr.shape) == 3 and arr.shape[2] == 3):
            return True
        else:
            return False
        
    def is_same_network(self, sliceIndex : int, firstCoord : tuple, secondCoord : tuple) -> bool:
        """
            Checks if two coordinates in a given slice belong to the same network.
            Parameters:
            sliceIndex (int): Index of the slice to check.
            firstCoord (tuple): Coordinates of the first point (x, y).
            secondCoord (tuple): Coordinates of the second point (x, y).
        """
        if sliceIndex < self.labeled_image.shape[0]:
            slice = self.labeled_image[sliceIndex]
            print(f"Slice {sliceIndex} shape: {slice.shape}")

            label_a = slice[firstCoord[1], firstCoord[0]]
            label_b = slice[secondCoord[1], secondCoord[0]]

            print(f"Label at {firstCoord}: {label_a}, Label at {secondCoord}: {label_b}")
            return label_a == label_b

        
    def analyze_volume(self) -> None:
        if self.labeled is None:
            raise ValueError("Call binarize_and_label() first.")

        voxel_counts = Counter(self.labeled.ravel())
        voxel_counts.pop(0, None)  # Remove background

        self.volumes = {
            label: count * self.voxel_volume
            for label, count in voxel_counts.items()
        }

        self.total_mito_volume = sum(self.volumes.values())

    def plot_volume_distribution(self):
        if not self.volumes:
            raise ValueError("Volumes dictionary is empty.")
        volume_array = np.array(list(self.volumes.values()))
        plt.hist(volume_array, bins=50)
        plt.title("Mitochondrial Network Volume Distribution")
        plt.xlabel("Volume (μm³)")
        plt.ylabel("Count")
        plt.show()

    def visualize_largest_network(self):
        if not self.volumes:
            raise ValueError("Volumes dictionary is empty.")
        
        largest_label = max(self.volumes, key=self.volumes.get)
        mask = (self.labeled == largest_label)

        num_slices = mask.shape[0]
        ncols = 6
        nrows = (num_slices + ncols - 1) // ncols

        fig, axs = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
        axs = axs.ravel()

        for i in range(num_slices):
            axs[i].imshow(mask[i], cmap='gray')
            axs[i].set_title(f"Z = {i}")
            axs[i].axis('off')

        # Hide unused axes
        for j in range(num_slices, len(axs)):
            axs[j].axis('off')

        plt.tight_layout()
        plt.suptitle(f"Largest Network (Label: {largest_label}) - Z slices", fontsize=16)
        plt.subplots_adjust(top=0.9)
        plt.show()


    def export_binary_tif(self, output_path: str) -> None:
        """
        Exports the binarized version of the input image to a .tif file.

        Parameters:
        output_path (str): Path to save the binary .tif file.
        """
        image = imread(self.image)

        if self.is_rgb_array(image):
            print("Image is RGB, converting to grayscale.")
            image = rgb2gray(image)

        if not self.is_binary_array(image):
            binary = (image > 0).astype(np.uint8)
        else:
            binary = image.astype(np.uint8)

        print("Saving binary image to:", output_path)
        imwrite(output_path, binary)


if __name__ == "__main__":
    zDepth = 32 # Replace with your z depth
    image_path = "/Al_Applications/3D-Mito-Analyzer/volume-analyzer/Images/3d/Site2.tif"  # Replace with your image path
    xRes = 67.61 / 1590  # Replace with your x resolution
    yRes = 67.61 / 1590  # Replace with your y resolution
    zRes = 0.16  # Replace with your z resolution

    analyzer = MitoNetworkAnalyzer(image_path, xRes, yRes, zRes, zDepth)
    # analyzer.visualize_labeled_image()
    # analyzer.plot_volume_distribution()
    # print("Total Mitochondrial Volume:", analyzer.total_mito_volume)
    # print("Volumes of individual networks:", analyzer.volumes)
    print(max(analyzer.volumes.values()))
    analyzer.visualize_labeled_image()
    analyzer.visualize_largest_network()
    analyzer.plot_volume_distribution()
    # analyzer.export_binary_tif("/Al_Applications/3D-Mito-Analyzer/volume-analyzer/Images/3d/binary_output.tif")
