# Mitochondria Analyzer - Task Master Implementation Plan

## Project Overview
Building a Streamlit-based mitochondrial image analysis tool with two main workflows:
1. **Segmentation Pipeline** - Deep learning segmentation using MoDL (~30 min per image)
2. **Analysis Pipeline** - Fast network analysis using MitoNetworkAnalyzer (<1 min)

## Project Goals
- Separate segmentation and analysis workflows for efficiency
- Docker deployment with volume mounting (no file uploads)
- Background processing for long-running segmentation jobs
- Real-time analysis results with visualization
- Professional UI with progress tracking

## Task Structure

Each task follows this format:
```json
{
  "id": 4,
  "title": "Task Title",
  "description": "Brief one-sentence description",
  "status": "pending|done|deferred", 
  "dependencies": [1, 2],
  "priority": "high|medium|low",
  "details": "Detailed implementation instructions and guidance",
  "testStrategy": "Approach for verifying completion",
  "subtasks": [...]
}
```

## Implementation Tasks

### Task 4: Create Analysis Workflow Wrapper
**Title**: Create analysis workflow wrapper for MitoNetworkAnalyzer
**Description**: Build a wrapper around the existing MitoNetworkAnalyzer class for Streamlit integration
**Status**: pending
**Dependencies**: []
**Priority**: high
**Details**: 
Create `workflows/analysis.py` that wraps the existing `MitoNetworkAnalyzer` class from `Analyzer.py`. Accept file paths and resolution parameters (xRes, yRes, zRes) from the UI and return structured results dictionary. Include error handling for invalid files, memory issues, and processing failures. The wrapper should validate input parameters, handle TIFF file loading, and format results for UI consumption.
**Test Strategy**: 
Test with sample segmented TIFF files from `data/segmented/`. Verify correct network counting, volume calculations, and error handling with invalid files. Confirm results match direct MitoNetworkAnalyzer usage.

### Task 5: Build Visualization Components
**Title**: Build visualization components for analysis results
**Description**: Create reusable Streamlit components for displaying mitochondrial analysis results
**Status**: pending
**Dependencies**: [4]
**Priority**: high
**Details**:
Create `components/visualization.py` with functions for displaying key metrics (network count, total volume, largest network size), matplotlib plots for volume distribution, and slice-by-slice visualization. Components should handle empty results gracefully and provide interactive features where appropriate. Include functions: display_summary_metrics(), plot_volume_distribution(), show_network_slices(), and display_analysis_table().
**Test Strategy**:
Test with various analysis result formats including empty results, single network, and multiple networks. Verify plots render correctly and interactive features work in Streamlit.

### Task 6: Implement Results Display in Analysis Tab
**Title**: Implement results display in analysis tab
**Description**: Connect file selection to analysis workflow and display results in the analysis tab
**Status**: pending
**Dependencies**: [4, 5]
**Priority**: high
**Details**:
Update `app.py` analysis tab to connect file selection to analysis workflow. When users select segmented files and click "Run Analysis", execute the analysis workflow and display results using visualization components. Add export options for results (CSV download) and show processing status with error messages. Include progress indicators during analysis and clear result display sections.
**Test Strategy**:
End-to-end test of selecting files, running analysis, and viewing results. Verify export functionality works and error messages are user-friendly.

### Task 7: Create Background Job Manager
**Title**: Create background job manager with threading
**Description**: Build a thread-based job management system for long-running tasks
**Status**: pending
**Dependencies**: []
**Priority**: medium
**Details**:
Create `workflows/job_manager.py` with a JobManager class that handles thread-based job execution, job queue management, unique job ID generation, and thread-safe operations. Include methods: submit_job(), get_job_status(), cancel_job(), cleanup_old_jobs(), and list_active_jobs(). Use threading.Thread for job execution and threading.Lock for thread safety. Store job status in memory with optional persistence.
**Test Strategy**:
Test concurrent job submission, status tracking, and job cancellation. Verify thread safety with multiple simultaneous operations.

### Task 8: Add Job Status Tracking and Persistence
**Title**: Add job status tracking and persistence
**Description**: Implement persistent job status storage and recovery
**Status**: pending
**Dependencies**: [7]
**Priority**: medium
**Details**:
Create `utils/job_persistence.py` to save job status to disk using JSON files in `data/jobs/`. Implement job history tracking, automatic cleanup of completed jobs older than 7 days, and job recovery on app restart. Include functions: save_job_status(), load_job_status(), cleanup_old_jobs(), and get_job_history().
**Test Strategy**:
Test job persistence across app restarts, verify cleanup functionality, and ensure job history is maintained correctly.

### Task 9: Implement Progress Bars and Job Queue UI
**Title**: Implement progress bars and job queue UI
**Description**: Create UI components for displaying job progress and queue status
**Status**: pending
**Dependencies**: [7, 8]
**Priority**: medium
**Details**:
Create `components/progress.py` with Streamlit components for real-time progress bars, job queue dashboard showing all active/completed jobs, cancel/restart job controls, and estimated time remaining calculations. Use st.progress(), st.empty() for real-time updates, and st.rerun() for periodic refreshes.
**Test Strategy**:
Test progress updates during job execution, verify cancel functionality works, and ensure UI updates in real-time without blocking.

### Task 10: Create Segmentation Workflow Wrapper
**Title**: Create segmentation workflow wrapper for MoDL
**Description**: Adapt the existing MoDL segmentation code for the job system
**Status**: pending
**Dependencies**: []
**Priority**: medium
**Details**:
Create `workflows/segmentation.py` that adapts `MoDL/MoDL_seg/segment_predict.py` for the job system. Add progress callbacks for UI updates, handle file I/O with mounted volumes, and preserve automatic z-stack detection. The wrapper should accept raw TIFF file paths and save segmented results to `data/segmented/`. Include error handling for GPU memory issues, invalid input files, and processing failures.
**Test Strategy**:
Test with sample raw TIFF files, verify progress reporting works, and ensure output files are correctly saved to the segmented directory.

### Task 11: Integrate MoDL with Background Job System
**Title**: Integrate MoDL with background job system
**Description**: Connect MoDL segmentation to the job manager for background processing
**Status**: pending
**Dependencies**: [7, 8, 10]
**Priority**: medium
**Details**:
Update `workflows/job_manager.py` to support segmentation job type, integrate progress reporting from MoDL wrapper, handle long-running processes (up to 30 minutes), and implement error recovery with cleanup. Add segmentation-specific methods and ensure proper resource cleanup after job completion.
**Test Strategy**:
Test complete segmentation job lifecycle including queuing, progress updates, completion, and error handling scenarios.

### Task 12: Add Model Caching and GPU Optimization
**Title**: Add model caching and GPU optimization
**Description**: Implement TensorFlow model caching and GPU memory management
**Status**: pending
**Dependencies**: [10]
**Priority**: low
**Details**:
Create `workflows/model_cache.py` to cache loaded TensorFlow models in memory, manage GPU memory allocation efficiently, implement model warming on startup, and ensure proper memory cleanup after jobs. Include GPU detection and fallback to CPU processing when GPU is unavailable.
**Test Strategy**:
Test model loading performance, verify GPU memory is properly managed, and ensure graceful fallback to CPU processing.

### Task 13: Implement Export Functionality
**Title**: Implement export functionality for results
**Description**: Create comprehensive export options for all analysis results
**Status**: pending
**Dependencies**: [5, 6]
**Priority**: low
**Details**:
Create `components/export.py` with functions for CSV export of analysis results, ZIP download of segmented images, PDF report generation with plots and summary, and high-resolution plot exports. All exports should be accessible via download buttons in the UI.
**Test Strategy**:
Test all export formats with various result types, verify download functionality works in different browsers, and ensure exported files contain correct data.

### Task 14: Create File Management Utilities
**Title**: Create file management and cleanup utilities
**Description**: Build utilities for managing files and disk space
**Status**: pending
**Dependencies**: []
**Priority**: low
**Details**:
Create `utils/file_manager.py` with functions for disk space monitoring, automatic cleanup of old files based on age and size, file organization utilities, and backup/restore functionality for important results. Include configurable cleanup policies and disk usage alerts.
**Test Strategy**:
Test cleanup policies work correctly, verify disk space monitoring is accurate, and ensure important files are not accidentally deleted.

### Task 15: Add Comprehensive Error Handling
**Title**: Add comprehensive error handling and user feedback
**Description**: Implement robust error handling throughout the application
**Status**: pending
**Dependencies**: []
**Priority**: medium
**Details**:
Add user-friendly error messages throughout the application, implement input validation for all user inputs, create graceful failure recovery mechanisms, and add comprehensive activity logging. Ensure all error states provide actionable feedback to users.
**Test Strategy**:
Test error scenarios including invalid files, out of memory conditions, network issues, and invalid parameters. Verify error messages are helpful and recovery mechanisms work.

### Task 16: Test Complete Workflows
**Title**: Test complete workflows with sample data
**Description**: Perform comprehensive end-to-end testing of all workflows
**Status**: pending
**Dependencies**: [6, 11, 13]
**Priority**: medium
**Details**:
Create comprehensive test suite covering both analysis and segmentation workflows, generate sample data for testing, perform performance benchmarking, and update documentation with test results. Include stress testing with large files and concurrent operations.
**Test Strategy**:
Execute full workflow tests from file selection to result export, measure performance metrics, and document any issues found during testing.

### Task 17: Optimize Docker Setup
**Title**: Optimize Docker setup and documentation
**Description**: Finalize Docker configuration and create comprehensive documentation
**Status**: pending
**Dependencies**: [16]
**Priority**: low
**Details**:
Implement multi-stage Docker builds for optimized image size, add GPU support configuration for NVIDIA Docker, create production deployment guide, and develop comprehensive user manual with examples. Include environment variable configuration and volume mounting best practices.
**Test Strategy**:
Test Docker builds on different platforms, verify GPU support works when available, and validate all documentation steps with fresh installations.

## Success Criteria
1. Users can analyze segmented images in <1 minute
2. Segmentation jobs run in background without UI blocking  
3. Clear progress indication for all operations
4. Professional, intuitive interface
5. Robust error handling and recovery
6. Easy Docker deployment with volume mounting
7. Export functionality for all results

## Technical Constraints
- Use existing `MitoNetworkAnalyzer` and MoDL code
- Docker deployment with volume mounting only
- Streamlit for UI (no custom web framework)
- Background processing must be thread-safe
- Handle large TIFF files (up to 2GB)
- Support both CPU and GPU processing

## Current Status
✅ Task 1: Basic Streamlit app structure with navigation (completed)
✅ Task 2: Docker deployment setup with volume mounting (completed)
✅ Task 3: Directory browsing instead of file uploads (completed)
🔄 Ready to start Task 4: Create Analysis Workflow Wrapper