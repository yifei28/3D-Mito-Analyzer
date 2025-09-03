from tifffile import imread, imwrite
from skimage import measure
from skimage.color import rgb2gray, label2rgb
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

from skimage import morphology, segmentation, measure, filters
from scipy import ndimage as ndi
from scipy.ndimage import zoom


class MitoNetworkAnalyzer:
    def __init__(self, imagePath : str, xRes : float, yRes : float, zRes : float, zDepth : int) -> None:
        self.image = imagePath
        self.zDepth = zDepth
        self.spacing = (xRes, yRes, zRes)  # Spacing in micrometers
        self.voxel_volume = xRes * yRes * zRes
        self.rescale_factor = zRes / xRes  # Assuming isotropic scaling for simplicity
        self.original = None
        self.labeled = None
        self.labeled_rescaled = None
        self.network_count = 0
        self.volumes = {}
        self.total_mito_volume = 0

        self.label_image(imagePath)
        self.count_networks()
        self.analyze_volume()

    def label_image(self, image_path : str) -> None:
        # Load image
        self.original = imread(image_path)

        if self.is_rgb_array(self.original):
            print("Image is RGB, converting to grayscale.")
            self.original = rgb2gray(self.original)

        # Convert to binary if needed
        if not self.is_binary_array(self.original):
            self.original = (self.original > 0).astype(np.uint8)

        # --- Morphological Opening by Reconstruction ---
        # Erode to remove thin bridges (radius=1 voxel)
        marker = morphology.erosion(self.original, morphology.ball(1))
        # Reconstruct to regrow main bodies but keep bridges broken
        cleaned = morphology.reconstruction(marker, self.original)

        # Label connected components in 3D with 26-connectivity
        self.labeled = measure.label(cleaned, connectivity=2)

        # (Optional) Rescaling step if needed
        scale_factors = (4, 1, 1)
        self.labeled_rescaled = ndi.zoom(self.labeled, scale_factors, order=0)

        print("Labeled Image shape:", self.labeled_rescaled.shape)
    
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

        fig, axs = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3 * nrows))
        axs = axs.ravel()

        # Step 1: Create a consistent color map for all labels
        max_label = self.labeled.max()
        cmap = plt.get_cmap('gist_ncar', max_label + 1)
        rgb_lookup = (cmap(np.arange(max_label + 1))[:, :3] * 255).astype(np.uint8)  # RGB colors

        # Step 2: Go through each slice and color with consistent lookup
        for i in range(num_slices):
            ax = axs[i]
            slice_2d = self.labeled[i, :, :]

            # Create RGB overlay from label IDs using consistent colormap
            rgb_image = np.zeros(slice_2d.shape + (3,), dtype=np.uint8)
            nonzero_mask = slice_2d > 0
            rgb_image[nonzero_mask] = rgb_lookup[slice_2d[nonzero_mask]]

            ax.imshow(rgb_image)
            ax.set_title(f"Z = {i}")
            ax.axis('off')

        # Hide unused subplots
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
        if sliceIndex < self.labeled.shape[0]:
            slice = self.labeled[sliceIndex]
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


    def save_labeled_image(self, output_path: str) -> None:
        """
        Saves the labeled 3D image as a multi-page .tif file.

        Parameters:
            output_path (str): File path to save the labeled image.
        """
        if self.labeled is None:
            raise ValueError("Labeled image not found. Run label_image() first.")

        labeled_uint16 = self.labeled.astype(np.uint16)  # Make sure it's compatible with TIFF
        imwrite(output_path, labeled_uint16)
        print(f"Labeled image saved to: {output_path}")

    def fast_label_z_spread(self) -> dict:
        z_dim = self.labeled.shape[0]
        label_z_presence = {}

        # Loop through Z slices and track labels present in each slice
        for z in range(z_dim):
            labels_in_slice = np.unique(self.labeled[z])
            for label in labels_in_slice:
                if label == 0:
                    continue  # skip background
                if label not in label_z_presence:
                    label_z_presence[label] = set()
                label_z_presence[label].add(z)

        # Count how many Z-slices each label spans
        spread = {label: len(z_indices) for label, z_indices in label_z_presence.items()}
        return spread
    
    def visualize_with_napari(self):
        try:
            import napari
        except ImportError:
            print("napari is not installed. Please run: pip install napari[all]")
            return

        if self.labeled is None or self.original is None:
            print("No image or label to visualize.")
            return

        viewer = napari.Viewer()
        volume = self.labeled_rescaled.copy()
        scale = 2.0
        volume_up = zoom(volume, (scale, scale, scale), order=0)
        viewer.add_labels(volume_up, name='Labeled Network', opacity=0.5)

        napari.run()

    def visualize_smooth_colored_networks(self,
                                        upsample_scale=(1, 2, 2),
                                        raw_sigma=(0, 1, 1),
                                        mask_sigma=(0, 1, 1),
                                        mesh_smooth_iters=20):
        """
        Upsamples, smooths, and visualizes each labeled network as its own colored, smoothed 3D surface.
        
        Parameters:
            upsample_scale (tuple of floats): scales for (Z, Y, X) when upsampling labels & raw.
            raw_sigma (tuple of floats): Gaussian sigma for smoothing the raw volume.
            mask_sigma (tuple of floats): Gaussian sigma for smoothing each label mask.
            mesh_smooth_iters (int): number of Laplacian smoothing iterations on each mesh.
        """
        import napari
        import numpy as np
        from scipy.ndimage import zoom, gaussian_filter
        from skimage import measure
        import open3d as o3d

        # 1) Upsample labels (nearest-neighbor) and raw (linear interpolation)
        labels_up = zoom(self.labeled_rescaled, upsample_scale, order=0)
        raw_up    = zoom(self.original,          upsample_scale, order=1)

        # 2) Gaussian-smooth the raw for context
        raw_smooth = gaussian_filter(raw_up, sigma=raw_sigma)

        # 3) Launch Napari and add the smoothed raw image
        viewer = napari.Viewer()
        img_layer = viewer.add_image(
            raw_smooth,
            name='Smoothed raw'
        )
        # enable linear interpolation for slice view if available
        try:
            img_layer.interpolation2d = 'linear'
        except Exception:
            pass

        # 4) Loop over each network ID and build a smoothed surface mesh
        for label_id in np.unique(labels_up)[1:]:
            mask = (labels_up == label_id).astype(np.uint8)
            if mask.sum() == 0:
                continue

            # 4a) Blur the binary mask & threshold
            mask_blur   = gaussian_filter(mask.astype(float), sigma=mask_sigma)
            mask_smooth = mask_blur > 0.5

            # 4b) Extract isosurface (verts, faces)
            verts, faces, normals, _ = measure.marching_cubes(mask_smooth, level=0.5)

            # 4c) Smooth the mesh using Open3D Laplacian smoothing
            mesh = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(verts),
                o3d.utility.Vector3iVector(faces)
            )
            mesh = mesh.filter_smooth_simple(number_of_iterations=mesh_smooth_iters)
            verts_s = np.asarray(mesh.vertices)
            faces_s = np.asarray(mesh.triangles)

            # 4d) Create a uniform RGBA color for this network
            color = np.concatenate([np.random.rand(3), [1.0]])
            vertex_colors = np.repeat(color[np.newaxis, :], verts_s.shape[0], axis=0)

            # 4e) Add the surface mesh with uniform vertex_colors
            viewer.add_surface(
                (verts_s, faces_s),
                vertex_colors=vertex_colors,
                name=f'Network {label_id}',
                shading='smooth'
            )

        napari.run()


    
if __name__ == "__main__":
    zDepth = 45 # Replace with your z depth
    image_path = "images/Site1_63x_2xMean_Zstack_2-3.tif"  # Replace with your image path
    xRes = 67.61 / 1590  # Replace with your x resolution
    yRes = 67.61 / 1590  # Replace with your y resolution
    zRes = 0.16  # Replace with your z resolution

    analyzer = MitoNetworkAnalyzer(image_path, xRes, yRes, zRes, zDepth)
    # analyzer.visualize_labeled_image()
    # print("Total Mitochondrial Volume:", analyzer.total_mito_volume)
    # print("Volumes of individual networks:", analyzer.volumes)
    # print(max(analyzer.volumes.values()))
    # analyzer.visualize_labeled_image()
    # analyzer.visualize_largest_network()
    # analyzer.save_original_image("/Al_Applications/3D-Mito-Analyzer/volume-analyzer/Images/3d/original_output.tif")
    # analyzer.plot_volume_distribution()
    # analyzer.export_binary_tif("/Al_Applications/3D-Mito-Analyzer/volume-analyzer/Images/3d/binary_output.tif")
    # spread = analyzer.fast_label_z_spread()
    # multi_slice = [label for label, z_count in spread.items() if z_count > 1]
    # print(f"Labels spanning >1 Z-slice: {len(multi_slice)} out of {len(spread)}")
    # analyzer.plot_volume_distribution()
    # analyzer.visualize_with_napari()
    analyzer.visualize_smooth_colored_networks()