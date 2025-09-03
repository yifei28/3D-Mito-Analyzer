"""
Visualization Components for MitoNetworkAnalyzer
Provides reusable Streamlit components for displaying mitochondrial analysis results
"""

import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Apply scientific plotting style
mplstyle.use('default')


@st.cache_data
def create_volume_distribution_plot(volume_data: Dict[int, float], 
                                   title: str = "Volume Distribution") -> plt.Figure:
    """
    Create a histogram of network volume distribution.
    
    Args:
        volume_data: Dictionary mapping network labels to volumes
        title: Plot title
        
    Returns:
        Matplotlib figure object
    """
    if not volume_data:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', 
                fontsize=14, transform=ax.transAxes)
        ax.set_title(title)
        return fig
    
    volumes = list(volume_data.values())
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create histogram with scientific styling
    n, bins, patches = ax.hist(volumes, bins=min(20, len(volumes)), 
                              alpha=0.7, color='#2E86C1', edgecolor='black', 
                              linewidth=0.5)
    
    # Styling
    ax.set_xlabel('Volume (μm³)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Networks', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    
    # Add statistics annotations
    mean_vol = np.mean(volumes)
    median_vol = np.median(volumes)
    
    textstr = f'Mean: {mean_vol:.2f} μm³\nMedian: {median_vol:.2f} μm³\nNetworks: {len(volumes)}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.98, 0.98, textstr, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    return fig


@st.cache_data
def create_slice_viewer_plot(labeled_image: np.ndarray, slice_idx: int, 
                            title: str = "Network Slice View") -> plt.Figure:
    """
    Create a 2D slice view of the labeled 3D image.
    
    Args:
        labeled_image: 3D labeled array (Z, Y, X)
        slice_idx: Z-slice index to display
        title: Plot title
        
    Returns:
        Matplotlib figure object
    """
    if labeled_image is None or labeled_image.size == 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, 'No labeled image available', ha='center', va='center', 
                fontsize=14, transform=ax.transAxes)
        ax.set_title(title)
        return fig
    
    # Ensure slice_idx is within bounds
    slice_idx = max(0, min(slice_idx, labeled_image.shape[0] - 1))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Get the slice
    slice_2d = labeled_image[slice_idx, :, :]
    
    # Create colored overlay
    if slice_2d.max() > 0:
        # Use a colormap for labeled regions
        im = ax.imshow(slice_2d, cmap='tab20', vmin=0, vmax=slice_2d.max())
        
        # Add colorbar if there are multiple labels
        if slice_2d.max() > 1:
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('Network Label', fontsize=12, fontweight='bold')
    else:
        # Show empty slice
        ax.imshow(np.zeros_like(slice_2d), cmap='gray')
        ax.text(0.5, 0.5, 'No networks in this slice', ha='center', va='center', 
                fontsize=12, transform=ax.transAxes, bbox=dict(boxstyle='round', 
                facecolor='white', alpha=0.8))
    
    ax.set_title(f"{title} (Z={slice_idx})", fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('X (pixels)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y (pixels)', fontsize=12, fontweight='bold')
    ax.axis('equal')
    
    plt.tight_layout()
    return fig


def display_analysis_summary(result: Dict[str, Any]) -> None:
    """
    Display high-level analysis summary with metric cards.
    
    Args:
        result: Analysis result dictionary from AnalysisWorkflow
    """
    if not result.get('success', False):
        st.error(f"❌ Analysis failed: {result.get('error_message', 'Unknown error')}")
        return
    
    st.success("✅ Analysis completed successfully!")
    
    # Create columns for metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="🔬 Networks Found", 
            value=result.get('network_count', 0)
        )
    
    with col2:
        st.metric(
            label="📊 Total Volume", 
            value=f"{result.get('total_volume', 0):.2f} μm³"
        )
    
    with col3:
        if 'volume_statistics' in result:
            st.metric(
                label="📈 Average Volume", 
                value=f"{result['volume_statistics']['mean_volume']:.2f} μm³"
            )
        else:
            st.metric(label="📈 Average Volume", value="N/A")
    
    with col4:
        st.metric(
            label="⏱️ Processing Time", 
            value=f"{result.get('processing_time', 0):.1f}s"
        )
    
    # Additional metrics if available
    if 'z_spread_analysis' in result:
        z_info = result['z_spread_analysis']
        st.info(
            f"🗂️ **Multi-slice networks:** {z_info['networks_spanning_multiple_slices']} "
            f"out of {result['network_count']} total networks "
            f"(Average Z-span: {z_info['average_z_span']:.1f} slices)"
        )
    
    # Display warnings if any
    if result.get('warnings'):
        with st.expander("⚠️ Warnings", expanded=False):
            for warning in result['warnings']:
                st.warning(warning)


def display_volume_distribution(result: Dict[str, Any]) -> None:
    """
    Display volume distribution visualization.
    
    Args:
        result: Analysis result dictionary
    """
    if not result.get('success', False) or not result.get('volume_distribution'):
        st.warning("No volume data available for visualization")
        return
    
    st.subheader("📊 Network Volume Distribution")
    
    # Choice between matplotlib and plotly
    viz_type = st.radio(
        "Visualization type:", 
        ["Histogram", "Box Plot", "Interactive Plot"], 
        horizontal=True,
        key="volume_viz_type"
    )
    
    volume_data = result['volume_distribution']
    
    if viz_type == "Histogram":
        # Matplotlib histogram
        fig = create_volume_distribution_plot(volume_data)
        st.pyplot(fig)
        
    elif viz_type == "Box Plot":
        # Simple box plot
        volumes = list(volume_data.values())
        fig, ax = plt.subplots(figsize=(8, 6))
        bp = ax.boxplot(volumes, patch_artist=True)
        bp['boxes'][0].set_facecolor('#85C1E9')
        ax.set_ylabel('Volume (μm³)', fontsize=12, fontweight='bold')
        ax.set_title('Network Volume Distribution', fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        
    elif viz_type == "Interactive Plot":
        # Plotly interactive histogram
        volumes = list(volume_data.values())
        labels = list(volume_data.keys())
        
        fig = px.histogram(
            x=volumes, 
            nbins=min(20, len(volumes)),
            title="Interactive Volume Distribution",
            labels={'x': 'Volume (μm³)', 'y': 'Count'},
            color_discrete_sequence=['#2E86C1']
        )
        fig.update_layout(
            xaxis_title="Volume (μm³)",
            yaxis_title="Number of Networks",
            showlegend=False
        )
        st.plotly_chart(fig, width='stretch')


def display_slice_viewer(result: Dict[str, Any]) -> None:
    """
    Display interactive slice viewer for labeled image.
    
    Args:
        result: Analysis result dictionary
    """
    if not result.get('success', False) or result.get('labeled_image') is None:
        st.warning("No labeled image available for slice viewing")
        return
    
    st.subheader("🔍 Interactive Slice Viewer")
    
    labeled_image = result['labeled_image']
    z_depth = labeled_image.shape[0]
    
    # Slice selection slider
    slice_idx = st.slider(
        "Select Z-slice:", 
        min_value=0, 
        max_value=z_depth - 1, 
        value=z_depth // 2,
        key="slice_viewer_slider"
    )
    
    # Display current slice info
    st.info(f"Showing slice {slice_idx} of {z_depth - 1} (Z-depth: {z_depth})")
    
    # Create and display the plot
    fig = create_slice_viewer_plot(labeled_image, slice_idx)
    st.pyplot(fig)
    
    # Show slice statistics
    slice_2d = labeled_image[slice_idx, :, :]
    unique_labels = np.unique(slice_2d)
    networks_in_slice = len(unique_labels) - 1  # Exclude background (0)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Networks in slice", networks_in_slice)
    with col2:
        st.metric("Max label", int(slice_2d.max()))
    with col3:
        nonzero_pixels = np.count_nonzero(slice_2d)
        st.metric("Network pixels", nonzero_pixels)


def display_statistics_table(result: Dict[str, Any]) -> None:
    """
    Display detailed numerical statistics in a table format.
    
    Args:
        result: Analysis result dictionary
    """
    if not result.get('success', False):
        st.warning("No statistics available - analysis failed")
        return
    
    st.subheader("📋 Detailed Statistics")
    
    # Create statistics dataframe
    stats_data = []
    
    # Basic metrics - ensure all values are strings to avoid Arrow serialization issues
    stats_data.append(["Networks Found", str(result.get('network_count', 0)), "count"])
    stats_data.append(["Total Volume", f"{result.get('total_volume', 0):.3f}", "μm³"])
    stats_data.append(["Processing Time", f"{result.get('processing_time', 0):.2f}", "seconds"])
    
    # Volume statistics if available
    if 'volume_statistics' in result:
        vol_stats = result['volume_statistics']
        stats_data.extend([
            ["Mean Volume", f"{vol_stats['mean_volume']:.3f}", "μm³"],
            ["Median Volume", f"{vol_stats['median_volume']:.3f}", "μm³"],
            ["Std Volume", f"{vol_stats['std_volume']:.3f}", "μm³"],
            ["Min Volume", f"{vol_stats['min_volume']:.3f}", "μm³"],
            ["Max Volume", f"{vol_stats['max_volume']:.3f}", "μm³"]
        ])
    
    # Z-spread analysis if available
    if 'z_spread_analysis' in result:
        z_stats = result['z_spread_analysis']
        stats_data.extend([
            ["Multi-slice Networks", str(z_stats['networks_spanning_multiple_slices']), "count"],
            ["Single-slice Networks", str(z_stats['single_slice_networks']), "count"],
            ["Average Z-span", f"{z_stats['average_z_span']:.1f}", "slices"]
        ])
    
    # Memory usage if available
    if 'memory_usage' in result and isinstance(result['memory_usage'], dict):
        mem_info = result['memory_usage']
        if 'estimated_peak_gb' in mem_info:
            stats_data.append(["Peak Memory Est.", f"{mem_info['estimated_peak_gb']:.2f}", "GB"])
    
    # Create and display table
    df = pd.DataFrame(stats_data, columns=["Metric", "Value", "Unit"])
    
    # Style the dataframe
    styled_df = df.style.set_properties(**{
        'background-color': '#f0f2f6',
        'color': 'black',
        'border-color': 'white'
    })
    
    st.dataframe(styled_df, width='stretch')


def display_network_details(result: Dict[str, Any]) -> None:
    """
    Display detailed information about individual networks.
    
    Args:
        result: Analysis result dictionary
    """
    if not result.get('success', False) or not result.get('volume_distribution'):
        st.warning("No network data available for detailed view")
        return
    
    st.subheader("🔬 Individual Network Analysis")
    
    volume_data = result['volume_distribution']
    
    # Sort networks by volume for better presentation
    sorted_networks = sorted(volume_data.items(), key=lambda x: x[1], reverse=True)
    
    # Show top networks
    num_to_show = st.selectbox(
        "Number of networks to display:", 
        options=[5, 10, 20, len(sorted_networks)],
        index=0 if len(sorted_networks) > 5 else len([5, 10, 20, len(sorted_networks)]) - 1,
        key="network_details_count"
    )
    
    # Create detailed table
    network_data = []
    for i, (label, volume) in enumerate(sorted_networks[:num_to_show]):
        rank = i + 1
        percentage = (volume / result['total_volume']) * 100
        network_data.append({
            "Rank": rank,
            "Network ID": int(label),
            "Volume (μm³)": f"{volume:.3f}",
            "% of Total": f"{percentage:.1f}%"
        })
    
    df = pd.DataFrame(network_data)
    
    # Display with color coding
    def highlight_top3(row):
        if row.name < 3:  # Top 3 networks
            return ['background-color: #ffd700'] * len(row)  # Gold
        elif row.name < 10:  # Top 10 networks
            return ['background-color: #e6f3ff'] * len(row)  # Light blue
        else:
            return [''] * len(row)
    
    styled_df = df.style.apply(highlight_top3, axis=1)
    st.dataframe(styled_df, width='stretch')
    
    # Show largest network info
    if sorted_networks:
        largest_id, largest_vol = sorted_networks[0]
        st.info(
            f"🏆 **Largest Network:** ID {largest_id} with volume {largest_vol:.3f} μm³ "
            f"({(largest_vol/result['total_volume']*100):.1f}% of total volume)"
        )


def display_complete_analysis_results(result: Dict[str, Any]) -> None:
    """
    Display complete analysis results using all visualization components.
    
    Args:
        result: Analysis result dictionary from AnalysisWorkflow
    """
    # Summary metrics at the top
    display_analysis_summary(result)
    
    if not result.get('success', False):
        return  # Stop here if analysis failed
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Volume Distribution", 
        "🔍 Slice Viewer", 
        "📋 Statistics", 
        "🔬 Network Details",
        "📁 Raw Data"
    ])
    
    with tab1:
        display_volume_distribution(result)
    
    with tab2:
        display_slice_viewer(result)
    
    with tab3:
        display_statistics_table(result)
    
    with tab4:
        display_network_details(result)
    
    with tab5:
        st.subheader("📁 Raw Analysis Data")
        with st.expander("View complete result data", expanded=False):
            # Display raw result data (excluding large arrays)
            display_result = result.copy()
            if 'labeled_image' in display_result:
                labeled_shape = display_result['labeled_image'].shape
                display_result['labeled_image'] = f"<3D array with shape {labeled_shape}>"
            
            st.json(display_result)