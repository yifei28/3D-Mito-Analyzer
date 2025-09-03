import SimpleITK as sitk
import numpy as np
import pyvista as pv

# 1) Read original image
img = sitk.ReadImage('images/Site1_63x_2xMean_Zstack_2-3.tif')
orig_size    = np.array(img.GetSize(),    dtype=float)   # (X, Y, Z)
orig_spacing = np.array(img.GetSpacing(), dtype=float)   # (x, y, z)

# 2) Decide on target isotropic spacing
target_spacing = orig_spacing.min()
new_spacing    = [target_spacing]*3

# 3) Compute new grid size
new_size = (orig_size * (orig_spacing / target_spacing)).astype(int).tolist()

# 4) Resample intensity volume
resampler = sitk.ResampleImageFilter()
resampler.SetOutputSpacing(new_spacing)
resampler.SetSize(new_size)
resampler.SetInterpolator(sitk.sitkBSpline)
img_up = resampler.Execute(img)

# 5) Optional Gaussian blur
img_up = sitk.SmoothingRecursiveGaussian(img_up, sigma=1.0)

# 6) Convert to NumPy (Z, Y, X)
vol = sitk.GetArrayFromImage(img_up)

# 7) Build a VTK grid
#    Try UniformGrid, else fall back to ImageData
GridClass = getattr(pv, 'UniformGrid', pv.ImageData)
grid = GridClass()

# VTK wants dims = (nx+1, ny+1, nz+1) and spacing & origin in (x,y,z) order
grid.dimensions = (np.array(img_up.GetSize()) + 1).tolist()
grid.spacing    = img_up.GetSpacing()
grid.origin     = img_up.GetOrigin()

# Flatten in Fortran order so VTK reads it correctly
grid.point_data['intensity'] = vol.flatten(order='F')

# 8) Render with volume shading & trilinear interpolation
p = pv.Plotter()
p.add_volume(
    grid,
    scalars='intensity',
    opacity='sigmoid_6',
    shade=True,
    interpolation='linear'
)
p.show(cpos='iso')