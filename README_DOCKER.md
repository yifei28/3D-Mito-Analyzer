# Docker Deployment Guide

## Quick Start

1. **Prepare your data directories:**
```bash
mkdir -p sample_data/raw sample_data/segmented sample_data/results
# Copy your TIFF files to sample_data/raw/ and sample_data/segmented/
```

2. **Run with Docker Compose:**
```bash
docker-compose up --build
```

3. **Access the app:** http://localhost:8501

## Manual Docker Run

```bash
# Build the image
docker build -t mito-analyzer .

# Run with volume mounts
docker run -p 8501:8501 \
  -v /path/to/your/raw_images:/app/data/raw \
  -v /path/to/your/segmented_images:/app/data/segmented \
  -v /path/to/output:/app/data/results \
  mito-analyzer
```

## Directory Structure

The app expects the following mounted directories:

- `/app/data/raw` - Raw TIFF z-stack images for segmentation
- `/app/data/segmented` - Pre-segmented TIFF images for analysis  
- `/app/data/results` - Output directory for processed results
- `/app/data/temp` - Temporary processing files

## Usage

1. **Segmentation Tab**: Select raw TIFF files from the mounted raw directory
2. **Analysis Tab**: Select segmented TIFF files from the mounted segmented directory  
3. **File Manager**: Browse and manage all mounted files

## Example Data Structure

```
your_project/
├── raw_images/
│   ├── sample1.tif
│   └── sample2.tif
├── segmented_images/
│   ├── segmented1.tif
│   └── segmented2.tif
└── results/
    └── (outputs will appear here)
```