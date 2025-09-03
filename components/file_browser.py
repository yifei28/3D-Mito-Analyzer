import streamlit as st
import os
from pathlib import Path
import tifffile

def browse_directory(directory_path: str, file_extensions: list = ['.tif', '.tiff']):
    """Browse files in a mounted directory"""
    if not os.path.exists(directory_path):
        st.warning(f"Directory not found: {directory_path}")
        st.info("Make sure to mount the directory when running Docker:\n`docker run -v /host/path:/app/data/raw ...`")
        return []
    
    files = []
    for ext in file_extensions:
        files.extend(list(Path(directory_path).glob(f"*{ext}")))
        files.extend(list(Path(directory_path).glob(f"*{ext.upper()}")))
    
    return sorted(files)

def display_file_browser(directory_path: str, title: str, file_type: str = "TIFF", key_prefix: str = None):
    """Display file browser component"""
    st.subheader(f"📂 {title}")
    
    files = browse_directory(directory_path)
    
    if not files:
        st.info(f"No {file_type} files found in {directory_path}")
        st.caption("Place your files in the mounted directory and refresh")
        return []
    
    # File selection
    st.write(f"Found {len(files)} {file_type} files:")
    
    selected_files = []
    
    # Show files with checkboxes
    if key_prefix is None:
        key_prefix = directory_path.replace("/", "_").replace("\\", "_")
    
    for i, file_path in enumerate(files):
        file_size = file_path.stat().st_size / (1024 * 1024)  # MB
        
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            selected = st.checkbox(
                f"{file_path.name}",
                key=f"{key_prefix}_{file_type}_{i}_{file_path.name}"
            )
        
        with col2:
            st.write(f"{file_size:.1f} MB")
            
        with col3:
            # Show z-stack info for TIFF files
            try:
                with tifffile.TiffFile(file_path) as tif:
                    z_depth = len(tif.pages)
                    st.write(f"Z: {z_depth}")
            except:
                st.write("—")
        
        if selected:
            selected_files.append(file_path)
    
    if selected_files:
        st.success(f"✅ {len(selected_files)} files selected")
    
    return selected_files

def display_file_info(file_path: Path):
    """Display detailed file information"""
    try:
        with tifffile.TiffFile(file_path) as tif:
            st.write(f"**File**: {file_path.name}")
            st.write(f"**Size**: {file_path.stat().st_size / (1024*1024):.1f} MB")
            st.write(f"**Z-depth**: {len(tif.pages)} slices")
            if tif.pages:
                shape = tif.pages[0].shape
                st.write(f"**Dimensions**: {shape[1]} × {shape[0]} pixels")
    except Exception as e:
        st.error(f"Error reading file info: {e}")