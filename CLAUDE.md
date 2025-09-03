# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a mitochondrial image analysis tool that combines:
1. **MoDL** - Deep learning-based mitochondrial segmentation and function prediction
2. **MitoNetworkAnalyzer** - 3D mitochondrial network analysis and visualization 
3. **Streamlit integration** (planned) - Interactive web interface

## Core Architecture

### MitoNetworkAnalyzer (Analyzer.py)
Main analysis class that processes 3D mitochondrial TIFF images:
- **Input**: TIFF image path + spatial resolution parameters (xRes, yRes, zRes, zDepth)
- **Processing**: Morphological operations, 3D connected component labeling, volume analysis
- **Output**: Labeled networks, volume distributions, visualization options
- **Key methods**: `label_image()`, `count_networks()`, `analyze_volume()`, `visualize_with_napari()`

### MoDL Integration (MoDL/ directory)
Deep learning pipeline for mitochondrial segmentation:
- **MoDL_seg/**: U-RNet+ based segmentation model with automatic z-stack detection
- **MoDL_pre/**: Function prediction based on morphological features
- **Requirements**: TensorFlow 2.15.0, simplified dependencies in MoDL/requirements.txt

### Visualization Options
Multiple visualization approaches implemented:
- Matplotlib slice-by-slice viewing
- Napari 3D volume rendering 
- PyVista/SimpleITK volume visualization (SimpleITK_PyVista.py)
- Open3D mesh smoothing for surface rendering

## Key Dependencies

Main packages:
- `tifffile` - TIFF image I/O
- `scikit-image` - Image processing and morphological operations
- `scipy` - Zoom/scaling operations
- `napari` - 3D visualization (optional)
- `open3d` - Mesh operations (optional)

MoDL dependencies (separate venv recommended):
- `tensorflow==2.15.0` 
- `opencv-python`, `pillow`, `tifffile`
- Note: May need to add `scikit-image`, `numpy`, `matplotlib` if imports fail

## Development Commands

**Main analyzer usage:**
```python
# Basic analysis
analyzer = MitoNetworkAnalyzer(image_path, xRes, yRes, zRes, zDepth)
print(f"Networks found: {analyzer.network_count}")
print(f"Total volume: {analyzer.total_mito_volume}")

# Visualization options
analyzer.visualize_labeled_image()        # 2D slice view
analyzer.visualize_with_napari()          # 3D interactive
analyzer.visualize_largest_network()      # Focus on largest
```

**MoDL workflow:**
```bash
# Segmentation (now with automatic z-stack detection)
cd MoDL/MoDL_seg
python data_load.py      # Prepare training data
python train.py          # Train segmentation model
python segment_predict.py # Generate predictions (auto-detects z-stack length)

# Function prediction  
cd ../MoDL_pre
python "morphology analysis.py"  # Extract features
python train.py                  # Train function model
python function_prediction.py    # Predict function
```

## File Structure Context

- `images/` - Contains sample TIFF files for testing
- Main analysis entry point is in `Analyzer.py` at line 323
- MoDL expects specific directory structure (testraw/, final_results/, etc.)
- `MoDL/MoDL_seg/segment_predict.py` now automatically detects z-stack dimensions
- Both projects use separate virtual environments with different dependency versions

## Streamlit Integration Notes

When implementing Streamlit interface:
- Load analyzer with user-uploaded TIFF files
- Allow parameter adjustment (resolution, depth)  
- Provide multiple visualization options
- Consider performance for large 3D volumes
- May need to integrate MoDL segmentation pipeline

## Docker Deployment

Project is intended for Docker deployment:
- Consider multi-stage build for MoDL dependencies and main analyzer
- Streamlit app will be the main interface in containerized deployment
- Handle GPU support for TensorFlow inference (optional nvidia-docker)
- Volume mounts needed for input/output TIFF files
- Environment variables for model paths and configuration

## Task Master AI Instructions
**Import Task Master's development workflow commands and guidelines, treat as if import is in the main CLAUDE.md file.**
@./.taskmaster/CLAUDE.md
