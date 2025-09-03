import streamlit as st
import os
import time
from pathlib import Path
from components.file_browser import display_file_browser, display_file_info
from components.visualizations import display_complete_analysis_results
from workflows.analysis import analysis_workflow
from workflows.segmentation import segmentation_workflow

# Configure page
st.set_page_config(
    page_title="Mitochondria Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional scientific interface
st.html("""
<style>
    /* Scientific theme color palette */
    :root {
        --primary-blue: #1f4e79;
        --secondary-blue: #2e5f8f;
        --accent-teal: #26a69a;
        --light-gray: #f8f9fa;
        --medium-gray: #e9ecef;
        --dark-gray: #495057;
        --success-green: #28a745;
        --warning-orange: #fd7e14;
        --danger-red: #dc3545;
        --text-dark: #212529;
        --shadow-light: 0 2px 4px rgba(0,0,0,0.1);
        --shadow-medium: 0 4px 8px rgba(0,0,0,0.15);
        --border-radius: 8px;
        --transition: all 0.3s ease;
    }

    /* Main app styling */
    .stApp {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, var(--primary-blue) 0%, var(--secondary-blue) 100%);
        color: white;
        padding: 2rem 0;
        margin: -1rem -1rem 2rem -1rem;
        text-align: center;
        box-shadow: var(--shadow-medium);
    }

    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }

    .main-header p {
        font-size: 1.2rem;
        opacity: 0.9;
        font-weight: 300;
    }

    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #ffffff 0%, #f8f9fa 100%);
        border-right: 1px solid var(--medium-gray);
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: white;
        padding: 1rem;
        border-radius: var(--border-radius);
        box-shadow: var(--shadow-light);
        margin-bottom: 1.5rem;
    }

    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        padding: 0 1.5rem;
        border-radius: var(--border-radius);
        border: 1px solid var(--medium-gray);
        background: white;
        color: var(--dark-gray);
        font-weight: 500;
        transition: var(--transition);
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(45deg, var(--primary-blue) 0%, var(--secondary-blue) 100%);
        color: white !important;
        border-color: var(--primary-blue);
        box-shadow: var(--shadow-medium);
        transform: translateY(-2px);
    }

    .stTabs [data-baseweb="tab"]:hover {
        border-color: var(--accent-teal);
        transform: translateY(-1px);
        box-shadow: var(--shadow-light);
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(45deg, var(--accent-teal) 0%, #4db6ac 100%);
        color: white;
        border: none;
        border-radius: var(--border-radius);
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: var(--transition);
        box-shadow: var(--shadow-light);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-medium);
        background: linear-gradient(45deg, #229688 0%, var(--accent-teal) 100%);
    }

    .stButton > button[data-testid="primary"] {
        background: linear-gradient(45deg, var(--primary-blue) 0%, var(--secondary-blue) 100%);
    }

    .stButton > button[data-testid="primary"]:hover {
        background: linear-gradient(45deg, #1a4269 0%, var(--primary-blue) 100%);
    }

    /* Cards and containers */
    .analysis-card {
        background: white;
        border-radius: var(--border-radius);
        padding: 1.5rem;
        box-shadow: var(--shadow-light);
        border: 1px solid var(--medium-gray);
        margin-bottom: 1.5rem;
        transition: var(--transition);
    }

    .analysis-card:hover {
        box-shadow: var(--shadow-medium);
        transform: translateY(-2px);
    }

    /* Info boxes */
    .stInfo {
        background: linear-gradient(45deg, #e3f2fd 0%, #f3e5f5 100%);
        border-left: 4px solid var(--accent-teal);
        border-radius: var(--border-radius);
        padding: 1rem 1.5rem;
        margin: 1rem 0;
    }

    .stSuccess {
        background: linear-gradient(45deg, #e8f5e8 0%, #f1f8e9 100%);
        border-left: 4px solid var(--success-green);
    }

    .stWarning {
        background: linear-gradient(45deg, #fff3cd 0%, #fcf8e3 100%);
        border-left: 4px solid var(--warning-orange);
    }

    /* Metrics styling */
    .metric-container {
        background: white;
        border-radius: var(--border-radius);
        padding: 1.5rem;
        text-align: center;
        box-shadow: var(--shadow-light);
        border: 1px solid var(--medium-gray);
        transition: var(--transition);
    }

    .metric-container:hover {
        box-shadow: var(--shadow-medium);
        transform: translateY(-2px);
    }

    /* File browser enhancements */
    .file-browser {
        background: white;
        border-radius: var(--border-radius);
        padding: 1.5rem;
        box-shadow: var(--shadow-light);
        border: 1px solid var(--medium-gray);
    }

    /* Progress indicators */
    .stProgress > div > div {
        background: linear-gradient(90deg, var(--accent-teal) 0%, var(--primary-blue) 100%);
        border-radius: var(--border-radius);
    }

    /* Input styling */
    .stNumberInput > div > div > input {
        border-radius: var(--border-radius);
        border: 1px solid var(--medium-gray);
        transition: var(--transition);
    }

    .stNumberInput > div > div > input:focus {
        border-color: var(--accent-teal);
        box-shadow: 0 0 0 2px rgba(38, 166, 154, 0.2);
    }

    /* Divider styling */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent 0%, var(--accent-teal) 50%, transparent 100%);
        margin: 2rem 0;
    }

    /* Animation for loading states */
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }

    .loading {
        animation: pulse 1.5s infinite;
    }

    /* Responsive design */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2rem;
        }
        
        .main-header p {
            font-size: 1rem;
        }
        
        .stTabs [data-baseweb="tab"] {
            padding: 0 1rem;
            font-size: 0.9rem;
        }
    }

    /* Status indicators */
    .status-active {
        color: var(--success-green);
        font-weight: 600;
    }

    .status-pending {
        color: var(--warning-orange);
        font-weight: 600;
    }

    .status-error {
        color: var(--danger-red);
        font-weight: 600;
    }

    /* Scientific icons */
    .science-icon {
        font-size: 1.5rem;
        margin-right: 0.5rem;
        vertical-align: middle;
    }
</style>
""")

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
    
    # App header with custom styling
    st.html("""
    <div class="main-header">
        <h1><span class="science-icon">🔬</span>Mitochondria Analyzer</h1>
        <p>Advanced mitochondrial segmentation and network analysis</p>
    </div>
    """)
    
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
        st.info("Select raw TIFF z-stack images for deep learning-based MoDL segmentation")
        
        # Initialize segmentation jobs in session state
        if 'segmentation_jobs' not in st.session_state:
            st.session_state.segmentation_jobs = {}
            
        # File browser and job submission section
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_raw_files = display_file_browser(
                "data/raw", 
                "Raw Images Directory", 
                "TIFF",
                "raw"
            )
            
            # Display file validation and estimates
            if selected_raw_files:
                st.subheader("📋 Pre-processing Validation")
                
                for file_path in selected_raw_files[:3]:  # Show up to 3 files
                    with st.expander(f"📄 {file_path.name}"):
                        # Validate file
                        is_valid, error_msg = segmentation_workflow.validate_raw_image(str(file_path))
                        
                        if is_valid:
                            st.success("✅ File validation passed")
                            
                            # Show processing estimates
                            estimates = segmentation_workflow.estimate_processing_time(str(file_path))
                            if 'error' not in estimates:
                                col_est1, col_est2, col_est3 = st.columns(3)
                                with col_est1:
                                    st.metric("Dimensions", estimates['image_dimensions'])
                                with col_est2:
                                    st.metric("Est. Time", f"{estimates['estimated_time_minutes']:.1f} min")
                                with col_est3:
                                    st.metric("Complexity", estimates['processing_complexity'])
                                
                                st.info(f"📊 {estimates['num_slices']} slices, {estimates['total_patches']} patches to process")
                            else:
                                st.warning(f"Estimation failed: {estimates['error']}")
                        else:
                            st.error(f"❌ Validation failed: {error_msg}")
                
                if len(selected_raw_files) > 3:
                    st.info(f"... and {len(selected_raw_files) - 3} more files selected")
        
        with col2:
            st.subheader("🚀 Segmentation Control")
            
            # Segmentation parameters
            with st.expander("⚙️ Advanced Parameters", expanded=False):
                patch_size = st.selectbox("Patch Size", [512, 256, 1024], index=0, 
                                        help="Size of image patches for processing")
                model_type = st.selectbox("Model Type", ["Standard", "High-Precision"], index=0,
                                        help="Segmentation model variant")
                gpu_enabled = st.checkbox("Use GPU (if available)", value=True,
                                        help="Enable GPU acceleration for faster processing")
            
            # Job submission
            if selected_raw_files:
                if st.button("🚀 Start Segmentation Jobs", type="primary", 
                           help=f"Process {len(selected_raw_files)} image(s)"):
                    
                    # Create and submit jobs
                    submitted_jobs = []
                    params = {
                        'patch_size': patch_size,
                        'model_type': model_type,
                        'gpu_enabled': gpu_enabled
                    }
                    
                    progress_container = st.empty()
                    
                    for i, file_path in enumerate(selected_raw_files):
                        progress_container.write(f"Creating job {i+1}/{len(selected_raw_files)}: {file_path.name}")
                        
                        # Validate before submitting
                        is_valid, error_msg = segmentation_workflow.validate_raw_image(str(file_path))
                        
                        if is_valid:
                            # Create and submit job
                            job_id = segmentation_workflow.create_job(str(file_path), params)
                            success = segmentation_workflow.submit_job(job_id)
                            
                            if success:
                                submitted_jobs.append(job_id)
                                st.session_state.segmentation_jobs[job_id] = {
                                    'file_name': file_path.name,
                                    'created_time': segmentation_workflow.get_job_status(job_id)['created_time']
                                }
                            else:
                                st.error(f"Failed to submit job for {file_path.name}")
                        else:
                            st.error(f"Skipping {file_path.name}: {error_msg}")
                    
                    progress_container.empty()
                    
                    if submitted_jobs:
                        st.success(f"✅ Successfully submitted {len(submitted_jobs)} segmentation job(s)!")
                        st.info("Monitor progress in the job status section below.")
                        st.rerun()
                    else:
                        st.error("No valid jobs could be submitted.")
            else:
                st.info("Select raw TIFF files to enable segmentation")
        
        # Job monitoring section
        st.divider()
        st.subheader("📊 Job Status Dashboard")
        
        # Get active jobs
        active_jobs = segmentation_workflow.list_active_jobs()
        all_session_jobs = list(st.session_state.segmentation_jobs.keys())
        
        if active_jobs or all_session_jobs:
            # Job overview metrics
            total_jobs = len(all_session_jobs)
            active_count = len(active_jobs)
            completed_count = 0
            failed_count = 0
            
            for job_id in all_session_jobs:
                status_info = segmentation_workflow.get_job_status(job_id)
                if status_info:
                    if status_info['status'] == 'completed':
                        completed_count += 1
                    elif status_info['status'] == 'failed':
                        failed_count += 1
            
            col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)
            with col_metric1:
                st.metric("Total Jobs", total_jobs)
            with col_metric2:
                st.metric("Active", active_count)  
            with col_metric3:
                st.metric("Completed", completed_count)
            with col_metric4:
                st.metric("Failed", failed_count)
            
            # Detailed job status
            st.subheader("Detailed Job Status")
            
            # Auto-refresh toggle
            col_refresh1, col_refresh2 = st.columns([1, 3])
            with col_refresh1:
                auto_refresh = st.checkbox("🔄 Auto-refresh", value=True, key="seg_auto_refresh")
            with col_refresh2:
                if st.button("🔄 Refresh Now", key="manual_refresh_jobs"):
                    st.rerun()
            
            # Auto-refresh logic using session state and time tracking
            if auto_refresh:
                if 'last_refresh_time' not in st.session_state:
                    st.session_state.last_refresh_time = time.time()
                
                current_time = time.time()
                time_since_refresh = current_time - st.session_state.last_refresh_time
                
                # Check if any jobs are still active (not completed/failed/cancelled)
                has_active_jobs = any(
                    segmentation_workflow.get_job_status(job_id) and 
                    segmentation_workflow.get_job_status(job_id)['status'] in ['queued', 'processing']
                    for job_id in all_session_jobs
                )
                
                # Auto-refresh every 3 seconds if there are active jobs
                if has_active_jobs and time_since_refresh >= 3:
                    st.session_state.last_refresh_time = current_time
                    st.rerun()
            
            # Display each job
            for job_id in all_session_jobs:
                job_status = segmentation_workflow.get_job_status(job_id)
                if job_status:
                    file_name = st.session_state.segmentation_jobs[job_id]['file_name']
                    
                    with st.expander(f"📄 {file_name} - {job_status['status'].title()}", 
                                   expanded=(job_status['status'] == 'processing')):
                        
                        # Progress bar
                        progress = job_status.get('progress', 0)
                        st.progress(progress / 100)
                        
                        # Status information
                        col_info1, col_info2 = st.columns(2)
                        
                        with col_info1:
                            st.write(f"**Status:** {job_status['status'].title()}")
                            st.write(f"**Progress:** {progress}%")
                            if 'status_message' in job_status:
                                st.write(f"**Current:** {job_status['status_message']}")
                        
                        with col_info2:
                            if 'elapsed_time' in job_status:
                                st.write(f"**Elapsed:** {job_status['elapsed_time']:.1f}s")
                            elif 'total_time' in job_status:
                                st.write(f"**Total Time:** {job_status['total_time']:.1f}s")
                            
                            if job_status['status'] == 'completed' and 'final_output_path' in job_status:
                                st.success(f"✅ Output: {Path(job_status['final_output_path']).name}")
                            elif job_status['status'] == 'failed':
                                st.error(f"❌ Error: {job_status.get('error_message', 'Unknown error')}")
                        
                        # Action buttons
                        col_btn1, col_btn2, col_btn3 = st.columns(3)
                        
                        with col_btn1:
                            if job_status['status'] in ['queued', 'processing']:
                                if st.button(f"🛑 Cancel", key=f"cancel_{job_id}"):
                                    if segmentation_workflow.cancel_job(job_id):
                                        st.warning(f"Job {job_id} cancelled")
                                        st.rerun()
                                    else:
                                        st.error("Failed to cancel job")
                        
                        with col_btn2:
                            if job_status['status'] == 'completed':
                                if st.button(f"🔍 View Results", key=f"view_{job_id}"):
                                    st.info("Results viewing will be integrated with Analysis tab")
                        
                        with col_btn3:
                            if job_status['status'] in ['completed', 'failed', 'cancelled']:
                                if st.button(f"🗑️ Remove", key=f"remove_{job_id}"):
                                    if job_id in st.session_state.segmentation_jobs:
                                        del st.session_state.segmentation_jobs[job_id]
                                        st.rerun()
            
            # Cleanup old jobs button
            st.divider()
            col_cleanup1, col_cleanup2 = st.columns(2)
            
            with col_cleanup1:
                if st.button("🧹 Cleanup Completed Jobs"):
                    # Remove completed jobs from session state
                    completed_jobs = [
                        job_id for job_id in st.session_state.segmentation_jobs.keys()
                        if segmentation_workflow.get_job_status(job_id) and 
                           segmentation_workflow.get_job_status(job_id)['status'] in ['completed', 'failed', 'cancelled']
                    ]
                    
                    for job_id in completed_jobs:
                        del st.session_state.segmentation_jobs[job_id]
                    
                    # Cleanup old jobs in workflow
                    segmentation_workflow.cleanup_old_jobs(max_age_hours=1)
                    
                    if completed_jobs:
                        st.success(f"Cleaned up {len(completed_jobs)} jobs")
                        st.rerun()
                    else:
                        st.info("No completed jobs to clean up")
            
            with col_cleanup2:
                if st.button("⚠️ Cancel All Jobs"):
                    cancelled_count = 0
                    for job_id in list(st.session_state.segmentation_jobs.keys()):
                        if segmentation_workflow.cancel_job(job_id):
                            cancelled_count += 1
                    
                    if cancelled_count > 0:
                        st.warning(f"Cancelled {cancelled_count} jobs")
                        st.rerun()
                    else:
                        st.info("No active jobs to cancel")
        
        else:
            st.info("No segmentation jobs yet. Select raw images and click 'Start Segmentation Jobs' to begin.")
            
            # Show some helpful information
            st.subheader("💡 Getting Started")
            st.markdown("""
            **Steps to segment mitochondrial images:**
            1. **Upload raw images** to the `data/raw/` directory (via Docker volume mount)
            2. **Select files** using the file browser above
            3. **Review validation** results and processing estimates
            4. **Configure parameters** if needed (Advanced Parameters)
            5. **Start segmentation** and monitor progress below
            6. **View results** in the Analysis tab once completed
            
            **Supported formats:** TIFF z-stack images (.tif, .tiff)
            **Processing time:** ~30 minutes per image (varies by size and complexity)
            **Requirements:** Images should be 512x512 pixels minimum
            """)
        
        # Show details of first selected file for reference
        if selected_raw_files:
            st.divider()
            st.subheader("📄 Selected File Details")
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
            
            if selected_segmented_files:
                # Get resolution parameters from session state
                resolution_params = st.session_state.get('resolution_params', {
                    'x_res': 0.0425, 'y_res': 0.0425, 'z_res': 0.16
                })
                
                # Z-depth input
                z_depth = st.number_input(
                    "Z-stack depth (number of slices)", 
                    min_value=1, 
                    max_value=1000, 
                    value=45,
                    help="Number of Z slices in the TIFF stack"
                )
                
                # Analysis button
                if st.button("🔍 Run Analysis", type="primary"):
                    st.info(f"Starting analysis for {len(selected_segmented_files)} file(s)...")
                    
                    # Initialize session state for results
                    if 'analysis_results' not in st.session_state:
                        st.session_state.analysis_results = {}
                    
                    # Process each selected file
                    for file_path in selected_segmented_files:
                        st.write(f"**Processing:** {file_path.name}")
                        
                        # Run analysis using the workflow
                        result = analysis_workflow.run_analysis(
                            file_path=str(file_path),
                            xRes=resolution_params['x_res'],
                            yRes=resolution_params['y_res'], 
                            zRes=resolution_params['z_res'],
                            zDepth=z_depth,
                            show_progress=True
                        )
                        
                        # Store result in session state
                        st.session_state.analysis_results[str(file_path)] = result
                        
                        # Display immediate summary
                        if result['success']:
                            st.success(analysis_workflow.format_results_summary(result))
                        else:
                            st.error(f"Analysis failed: {result['error_message']}")
            
            # Analysis results display
            st.subheader("📊 Analysis Results")
            
            if 'analysis_results' in st.session_state and st.session_state.analysis_results:
                # Results overview
                total_results = len(st.session_state.analysis_results)
                successful_results = sum(1 for r in st.session_state.analysis_results.values() if r['success'])
                
                col_overview1, col_overview2 = st.columns(2)
                with col_overview1:
                    st.metric("Total Analyses", total_results)
                with col_overview2:
                    st.metric("Successful", f"{successful_results}/{total_results}")
                
                # File selection for detailed visualization
                if len(st.session_state.analysis_results) > 1:
                    selected_file = st.selectbox(
                        "Select file for detailed visualization:",
                        options=list(st.session_state.analysis_results.keys()),
                        format_func=lambda x: Path(x).name,
                        key="results_file_selector"
                    )
                else:
                    selected_file = list(st.session_state.analysis_results.keys())[0]
                
                # Display comprehensive results using visualization components
                if selected_file in st.session_state.analysis_results:
                    result = st.session_state.analysis_results[selected_file]
                    st.markdown(f"### Results for: `{Path(selected_file).name}`")
                    
                    # Use the new comprehensive visualization component
                    display_complete_analysis_results(result)
                
                # Quick overview for all files
                if len(st.session_state.analysis_results) > 1:
                    st.subheader("🔍 Quick Overview - All Files")
                    
                    overview_data = []
                    for file_path, result in st.session_state.analysis_results.items():
                        file_name = Path(file_path).name
                        overview_data.append({
                            "File": file_name,
                            "Status": "✅ Success" if result['success'] else "❌ Failed",
                            "Networks": str(result.get('network_count', 0)) if result['success'] else "N/A",
                            "Total Volume (μm³)": f"{result.get('total_volume', 0):.2f}" if result['success'] else "N/A",
                            "Time (s)": f"{result.get('processing_time', 0):.1f}" if result['success'] else "N/A"
                        })
                    
                    # Display as a table
                    import pandas as pd
                    df = pd.DataFrame(overview_data)
                    st.dataframe(df, width='stretch')
                
                # Clear results button
                if st.button("🗑️ Clear All Results"):
                    st.session_state.analysis_results = {}
                    st.rerun()
                    
            else:
                st.info("No analysis results yet. Select files and run analysis to see results here.")
            
            # Show details of first selected file
            if selected_segmented_files:
                st.subheader("File Details")
                display_file_info(selected_segmented_files[0])
    
    with tab3:
        st.header("File Management Dashboard")
        
        # File overview with custom styling
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.html("""
            <div class="metric-container">
                <h3 style="color: var(--primary-blue); margin-bottom: 0.5rem;">📁 Raw Images</h3>
                <h2 style="color: var(--dark-gray); margin: 0;">0</h2>
                <p style="color: var(--accent-teal); margin: 0.5rem 0 0 0; font-size: 0.9rem;">Ready for processing</p>
            </div>
            """)
            
        with col2:
            st.html("""
            <div class="metric-container">
                <h3 style="color: var(--primary-blue); margin-bottom: 0.5rem;">🧬 Segmented Images</h3>
                <h2 style="color: var(--dark-gray); margin: 0;">0</h2>
                <p style="color: var(--accent-teal); margin: 0.5rem 0 0 0; font-size: 0.9rem;">Ready for analysis</p>
            </div>
            """)
            
        with col3:
            st.html("""
            <div class="metric-container">
                <h3 style="color: var(--primary-blue); margin-bottom: 0.5rem;">📊 Analysis Results</h3>
                <h2 style="color: var(--dark-gray); margin: 0;">0</h2>
                <p style="color: var(--accent-teal); margin: 0.5rem 0 0 0; font-size: 0.9rem;">Completed</p>
            </div>
            """)
        
        st.divider()
        
        # File browser (placeholder)
        st.subheader("📂 Browse Files")
        st.info("File browser will be implemented here")
        
        # Download section
        st.subheader("⬇️ Downloads")
        st.info("Download links for completed jobs will appear here")

if __name__ == "__main__":
    main()