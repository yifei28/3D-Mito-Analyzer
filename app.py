import streamlit as st
import os
from pathlib import Path
from components.file_browser import display_file_browser, display_file_info

# Configure page
st.set_page_config(
    page_title="Mitochondria Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
def init_session_state():
    if 'jobs' not in st.session_state:
        st.session_state.jobs = {}
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = {}
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = {}

# Create data directories
def setup_directories():
    directories = [
        "data/raw",
        "data/segmented", 
        "data/results",
        "data/temp"
    ]
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

def main():
    # Initialize app
    init_session_state()
    setup_directories()
    
    # App header
    st.title("🔬 Mitochondria Analyzer")
    st.markdown("Advanced mitochondrial segmentation and network analysis")
    
    # Sidebar for global settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Global parameters
        st.subheader("Analysis Parameters")
        x_res = st.number_input("X Resolution (μm/pixel)", value=0.0425, format="%.4f")
        y_res = st.number_input("Y Resolution (μm/pixel)", value=0.0425, format="%.4f") 
        z_res = st.number_input("Z Resolution (μm)", value=0.16, format="%.2f")
        
        # Store in session state
        st.session_state.resolution_params = {
            'x_res': x_res,
            'y_res': y_res,
            'z_res': z_res
        }
        
        st.divider()
        
        # File management
        st.subheader("📁 File Management")
        if st.button("🗑️ Cleanup Old Files"):
            st.success("Cleanup completed!")
    
    # Main navigation tabs
    tab1, tab2, tab3 = st.tabs(["🔬 Segmentation", "📊 Analysis", "📁 File Manager"])
    
    with tab1:
        st.header("Mitochondrial Segmentation Pipeline")
        st.info("Select raw TIFF z-stack images from mounted directory for deep learning-based segmentation (⏱️ ~30 min per image)")
        
        # File browser section
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_raw_files = display_file_browser(
                "data/raw", 
                "Raw Images Directory", 
                "TIFF",
                "raw"
            )
        
        with col2:
            st.subheader("Segmentation Queue")
            if selected_raw_files and st.button("🚀 Start Segmentation", type="primary"):
                st.success(f"Segmentation jobs queued for {len(selected_raw_files)} files!")
                st.info("Check status below...")
                # TODO: Add files to job queue
            
            # Job status (placeholder)
            st.subheader("Active Jobs")
            if not st.session_state.jobs:
                st.write("No active jobs")
            else:
                for job_id, job_info in st.session_state.jobs.items():
                    st.write(f"Job {job_id}: {job_info['status']}")
            
            # Show details of first selected file
            if selected_raw_files:
                st.subheader("File Details")
                display_file_info(selected_raw_files[0])
    
    with tab2:
        st.header("Mitochondrial Network Analysis")
        st.info("Select segmented TIFF images from mounted directory for fast network analysis (⏱️ <1 min)")
        
        # File browser section for segmented images
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_segmented_files = display_file_browser(
                "data/segmented", 
                "Segmented Images Directory", 
                "TIFF",
                "segmented"
            )
        
        with col2:
            st.subheader("Analysis Controls")
            if selected_segmented_files and st.button("🔍 Run Analysis", type="primary"):
                st.success(f"Analysis started for {len(selected_segmented_files)} files!")
                st.info("Processing...")
                # TODO: Run analysis workflow
            
            # Analysis results (placeholder)
            st.subheader("Results")
            st.write("No analysis results yet")
            
            # Show details of first selected file
            if selected_segmented_files:
                st.subheader("File Details")
                display_file_info(selected_segmented_files[0])
    
    with tab3:
        st.header("File Management Dashboard")
        
        # File overview
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Raw Images", "0", "0")
            
        with col2:
            st.metric("Segmented Images", "0", "0")
            
        with col3:
            st.metric("Analysis Results", "0", "0")
        
        st.divider()
        
        # File browser (placeholder)
        st.subheader("📂 Browse Files")
        st.info("File browser will be implemented here")
        
        # Download section
        st.subheader("⬇️ Downloads")
        st.info("Download links for completed jobs will appear here")

if __name__ == "__main__":
    main()