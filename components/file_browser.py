import streamlit as st
import os
from pathlib import Path
import tifffile
import glob
from typing import List, Optional
from datetime import datetime

def get_subdirectories(directory_path: str) -> List[Path]:
    """Get all subdirectories in the given path"""
    try:
        path = Path(directory_path)
        if not path.exists():
            return []
        return sorted([p for p in path.iterdir() if p.is_dir()])
    except PermissionError:
        return []

def browse_directory(directory_path: str, file_extensions: list = ['.tif', '.tiff'], recursive: bool = False):
    """Browse files in a mounted directory with optional recursive search"""
    if not os.path.exists(directory_path):
        st.warning(f"Directory not found: {directory_path}")
        st.info("Make sure to mount the directory when running Docker:\n`docker run -v /host/path:/app/data/raw ...`")
        return []
    
    files = []
    path = Path(directory_path)
    
    try:
        for ext in file_extensions:
            if recursive:
                files.extend(list(path.rglob(f"*{ext}")))
                files.extend(list(path.rglob(f"*{ext.upper()}")))
            else:
                files.extend(list(path.glob(f"*{ext}")))
                files.extend(list(path.glob(f"*{ext.upper()}")))
    except PermissionError:
        st.error(f"Permission denied accessing {directory_path}")
        return []
    
    return sorted(files)

def search_files(directory_path: str, search_pattern: str, file_extensions: list = ['.tif', '.tiff']) -> List[Path]:
    """Search for files using glob patterns"""
    if not search_pattern.strip():
        return browse_directory(directory_path, file_extensions)
    
    files = []
    path = Path(directory_path)
    
    try:
        # Add file extensions to search pattern if not present
        if not any(ext in search_pattern.lower() for ext in ['.tif', '.tiff']):
            for ext in file_extensions:
                pattern = f"*{search_pattern}*{ext}"
                files.extend(list(path.rglob(pattern)))
                files.extend(list(path.rglob(f"*{search_pattern}*{ext.upper()}")))
        else:
            files.extend(list(path.rglob(f"*{search_pattern}*")))
    except Exception as e:
        st.error(f"Search error: {e}")
        return []
    
    return sorted(files)

def display_breadcrumb_navigation(current_path: str, base_path: str, key_prefix: str = "nav") -> str:
    """Display breadcrumb navigation for directory traversal"""
    current = Path(current_path).resolve()
    base = Path(base_path).resolve()
    
    # Ensure current path is within base path for security
    try:
        current.relative_to(base)
    except ValueError:
        current = base
    
    # Build breadcrumb parts
    parts = []
    temp_path = current
    
    while temp_path != base.parent and temp_path != temp_path.parent:
        parts.insert(0, temp_path)
        temp_path = temp_path.parent
        if temp_path == base:
            parts.insert(0, base)
            break
    
    if not parts:
        parts = [base]
    
    # Display breadcrumb navigation
    st.html("""
    <div style="
        background: white; 
        padding: 1rem 1.5rem; 
        border-radius: 8px; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border: 1px solid #e9ecef;
        margin-bottom: 1rem;
    ">
        <div style="color: #495057; font-weight: 600; margin-bottom: 0.5rem;">
            📁 Navigate Directory
        </div>
    </div>
    """)
    
    # Create navigation buttons
    cols = st.columns(len(parts) + 1)
    
    selected_path = str(current)
    
    for i, part in enumerate(parts):
        with cols[i]:
            part_name = part.name if part.name else "Root"
            if st.button(f"📂 {part_name}", key=f"{key_prefix}_breadcrumb_{i}_{part_name}"):
                selected_path = str(part)
    
    # Add subdirectory navigation
    subdirs = get_subdirectories(current)
    if subdirs:
        with cols[-1]:
            subdir_names = ["📁 " + d.name for d in subdirs]
            selected_subdir = st.selectbox(
                "Subdirectories", 
                [""] + subdir_names,
                key=f"{key_prefix}_subdir_select"
            )
            if selected_subdir:
                subdir_name = selected_subdir.replace("📁 ", "")
                selected_path = str(current / subdir_name)
    
    return selected_path

def display_file_browser(directory_path: str, title: str, file_type: str = "TIFF", key_prefix: str = None, 
                        enable_navigation: bool = True, enable_search: bool = True):
    """Enhanced file browser component with navigation and search"""
    
    # Initialize session state for current directory
    if key_prefix is None:
        key_prefix = directory_path.replace("/", "_").replace("\\", "_")
    
    current_dir_key = f"{key_prefix}_current_dir"
    if current_dir_key not in st.session_state:
        st.session_state[current_dir_key] = directory_path
    
    # Custom styled header
    st.html(f"""
    <div class="file-browser">
        <h3 style="color: var(--primary-blue); margin-bottom: 1rem;">
            <span class="science-icon">📂</span>{title}
        </h3>
    </div>
    """)
    
    # Navigation breadcrumbs
    if enable_navigation:
        new_path = display_breadcrumb_navigation(
            st.session_state[current_dir_key], 
            directory_path, 
            key_prefix
        )
        if new_path != st.session_state[current_dir_key]:
            st.session_state[current_dir_key] = new_path
            st.rerun()
    
    current_directory = st.session_state[current_dir_key]
    
    # Search functionality
    search_pattern = ""
    if enable_search:
        col1, col2 = st.columns([3, 1])
        with col1:
            search_pattern = st.text_input(
                "🔍 Search files", 
                placeholder="Enter filename pattern (e.g., 'mito', 'cell_*', '*.tif')",
                key=f"{key_prefix}_search"
            )
        with col2:
            recursive_search = st.checkbox(
                "Recursive", 
                value=False,
                key=f"{key_prefix}_recursive",
                help="Search in subdirectories"
            )
    
    # Get files based on search or browse
    if search_pattern:
        files = search_files(current_directory, search_pattern)
    else:
        files = browse_directory(current_directory, recursive=enable_search and st.session_state.get(f"{key_prefix}_recursive", False))
    
    if not files:
        if search_pattern:
            st.info(f"No {file_type} files found matching '{search_pattern}' in {current_directory}")
        else:
            st.info(f"No {file_type} files found in {current_directory}")
        st.caption("Place your files in the mounted directory and refresh")
        return []
    
    # File count and bulk selection
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.html(f"""
        <div style="color: var(--accent-teal); font-weight: 600; padding: 0.5rem 0;">
            Found {len(files)} {file_type} files
        </div>
        """)
    
    with col2:
        select_all = st.button(f"✅ Select All", key=f"{key_prefix}_select_all")
    
    with col3:
        clear_all = st.button(f"❌ Clear All", key=f"{key_prefix}_clear_all")
    
    selected_files = []
    
    # File list with enhanced styling
    st.html("""
    <div style="
        background: white; 
        border-radius: 8px; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border: 1px solid #e9ecef;
        padding: 1rem;
        margin: 1rem 0;
    ">
    """)
    
    for i, file_path in enumerate(files):
        try:
            file_stats = file_path.stat()
            file_size = file_stats.st_size / (1024 * 1024)  # MB
            modified_time = datetime.fromtimestamp(file_stats.st_mtime).strftime("%Y-%m-%d %H:%M")
            
            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
            
            checkbox_key = f"{key_prefix}_{file_type}_{i}_{file_path.name}"
            
            # Handle bulk operations
            if select_all:
                st.session_state[checkbox_key] = True
            elif clear_all:
                st.session_state[checkbox_key] = False
            
            with col1:
                selected = st.checkbox(
                    f"📄 {file_path.name}",
                    key=checkbox_key,
                    value=st.session_state.get(checkbox_key, False)
                )
            
            with col2:
                st.html(f"""
                <div style="color: var(--dark-gray); font-size: 0.9rem; padding: 0.5rem 0;">
                    {file_size:.1f} MB
                </div>
                """)
                
            with col3:
                # Show z-stack info for TIFF files
                try:
                    with tifffile.TiffFile(file_path) as tif:
                        z_depth = len(tif.pages)
                        st.html(f"""
                        <div style="color: var(--accent-teal); font-weight: 600; padding: 0.5rem 0;">
                            Z: {z_depth}
                        </div>
                        """)
                except:
                    st.html("""
                    <div style="color: var(--medium-gray); padding: 0.5rem 0;">
                        —
                    </div>
                    """)
            
            with col4:
                st.html(f"""
                <div style="color: var(--dark-gray); font-size: 0.8rem; padding: 0.5rem 0;">
                    {modified_time}
                </div>
                """)
            
            if selected:
                selected_files.append(file_path)
                
        except Exception as e:
            st.error(f"Error reading {file_path.name}: {e}")
    
    st.html("</div>")
    
    # Selection summary
    if selected_files:
        st.html(f"""
        <div style="
            background: linear-gradient(45deg, #e8f5e8 0%, #f1f8e9 100%);
            border-left: 4px solid var(--success-green);
            border-radius: 8px;
            padding: 1rem 1.5rem;
            margin: 1rem 0;
        ">
            <strong style="color: var(--success-green);">
                ✅ {len(selected_files)} files selected for processing
            </strong>
        </div>
        """)
    
    return selected_files

def display_file_info(file_path: Path):
    """Display detailed file information with enhanced styling"""
    try:
        file_stats = file_path.stat()
        file_size = file_stats.st_size / (1024 * 1024)  # MB
        modified_time = datetime.fromtimestamp(file_stats.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        
        with tifffile.TiffFile(file_path) as tif:
            z_depth = len(tif.pages)
            
            # Get image dimensions
            if tif.pages:
                shape = tif.pages[0].shape
                width, height = shape[1], shape[0]
                
                # Estimate memory usage for full volume
                estimated_memory = (width * height * z_depth * 2) / (1024**3)  # Assuming 16-bit, in GB
            else:
                width = height = 0
                estimated_memory = 0
        
        # Display styled file information
        st.html(f"""
        <div style="
            background: white; 
            border-radius: 8px; 
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border: 1px solid #e9ecef;
            padding: 1.5rem;
            margin: 1rem 0;
        ">
            <h4 style="color: var(--primary-blue); margin-bottom: 1rem;">
                📄 File Information
            </h4>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                <div style="grid-column: 1 / -1;">
                    <strong style="color: var(--dark-gray);">Filename:</strong><br>
                    <span style="color: var(--accent-teal); font-family: monospace; word-break: break-all; font-size: 0.9rem;">{file_path.name}</span>
                </div>
                
                <div>
                    <strong style="color: var(--dark-gray);">File Size:</strong><br>
                    <span style="color: var(--accent-teal);">{file_size:.1f} MB</span>
                </div>
                
                <div>
                    <strong style="color: var(--dark-gray);">Z-Stack Depth:</strong><br>
                    <span style="color: var(--accent-teal);">{z_depth} slices</span>
                </div>
                
                <div>
                    <strong style="color: var(--dark-gray);">Dimensions:</strong><br>
                    <span style="color: var(--accent-teal);">{width} × {height} pixels</span>
                </div>
                
                <div>
                    <strong style="color: var(--dark-gray);">Modified:</strong><br>
                    <span style="color: var(--dark-gray); font-size: 0.85rem; font-family: monospace;">{modified_time}</span>
                </div>
                
                <div>
                    <strong style="color: var(--dark-gray);">Est. Memory:</strong><br>
                    <span style="color: var(--warning-orange);">{estimated_memory:.2f} GB</span>
                </div>
            </div>
        </div>
        """)
        
    except Exception as e:
        st.html(f"""
        <div style="
            background: linear-gradient(45deg, #fff3cd 0%, #fcf8e3 100%);
            border-left: 4px solid var(--warning-orange);
            border-radius: 8px;
            padding: 1rem 1.5rem;
            margin: 1rem 0;
        ">
            <strong style="color: var(--warning-orange);">⚠️ Error reading file information:</strong><br>
            <span style="color: var(--dark-gray);">{str(e)}</span>
        </div>
        """)