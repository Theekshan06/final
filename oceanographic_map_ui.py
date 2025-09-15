#!/usr/bin/env python3
"""
Oceanographic Data Visualization Interface
- Interactive world map with data points
- Comprehensive filtering controls
- Two-panel layout: controls (20%) + map (80%)
"""

import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Set page configuration
st.set_page_config(
    page_title="Global Oceanographic Data Visualization",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for the interface
st.markdown("""
<style>
    /* Remove default streamlit padding */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
        padding-left: 1rem;
        padding-right: 1rem;
        max-width: 100%;
    }

    /* Control panel styling */
    .control-panel {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #e9ecef;
        height: 100vh;
        overflow-y: auto;
    }

    /* Filter section headers */
    .filter-header {
        font-weight: 600;
        font-size: 1rem;
        color: #495057;
        margin-bottom: 0.5rem;
        margin-top: 1rem;
    }

    /* Time filter buttons */
    .time-filter-buttons {
        display: flex;
        gap: 5px;
        margin-bottom: 1rem;
        flex-wrap: wrap;
    }

    .time-btn {
        background: #6c757d;
        color: white;
        border: none;
        padding: 0.4rem 0.8rem;
        border-radius: 4px;
        font-size: 0.85rem;
        cursor: pointer;
    }

    .time-btn.active {
        background: #007bff;
    }

    /* Search button styling */
    .search-button {
        background: #007bff;
        color: white;
        border: none;
        padding: 0.7rem 1rem;
        border-radius: 4px;
        width: 100%;
        font-size: 1rem;
        cursor: pointer;
        margin-top: 0.5rem;
    }

    /* Action buttons */
    .action-buttons {
        display: flex;
        gap: 10px;
        margin-top: 1rem;
    }

    .btn-reset {
        background: #6c757d;
        color: white;
        border: none;
        padding: 0.6rem 1rem;
        border-radius: 4px;
        cursor: pointer;
        flex: 1;
    }

    .btn-primary {
        background: #007bff;
        color: white;
        border: none;
        padding: 0.6rem 1rem;
        border-radius: 4px;
        cursor: pointer;
        flex: 1;
    }

    /* Checkbox styling */
    .parameter-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.3rem;
    }

    .parameter-count {
        background: #e9ecef;
        padding: 0.2rem 0.5rem;
        border-radius: 10px;
        font-size: 0.8rem;
        color: #495057;
    }

    /* Map controls */
    .map-controls {
        position: absolute;
        top: 10px;
        right: 10px;
        z-index: 1000;
        display: flex;
        flex-direction: column;
        gap: 5px;
    }

    .map-control-btn {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        background: white;
        border: 2px solid #ccc;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }

    /* Data count display */
    .data-count {
        background: #e3f2fd;
        padding: 0.8rem;
        border-radius: 4px;
        text-align: center;
        font-weight: 600;
        color: #1976d2;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'selected_time_filter' not in st.session_state:
    st.session_state.selected_time_filter = 'ALL'
if 'pressure_range' not in st.session_state:
    st.session_state.pressure_range = [0, 6]
if 'location_search' not in st.session_state:
    st.session_state.location_search = ''
if 'selected_parameters' not in st.session_state:
    st.session_state.selected_parameters = ['BBP470', 'BBP532']
if 'selected_years' not in st.session_state:
    st.session_state.selected_years = ['2024', '2023']

# Generate sample oceanographic data points
@st.cache_data
def generate_sample_data():
    """Generate sample oceanographic data points"""
    np.random.seed(42)

    # Generate points concentrated in ocean areas
    ocean_regions = [
        # Atlantic Ocean
        {'lat_range': (-60, 70), 'lon_range': (-80, 20), 'density': 800},
        # Pacific Ocean
        {'lat_range': (-60, 65), 'lon_range': (120, -70), 'density': 1200},
        # Indian Ocean
        {'lat_range': (-60, 30), 'lon_range': (20, 150), 'density': 600},
        # Southern Ocean
        {'lat_range': (-70, -40), 'lon_range': (-180, 180), 'density': 400},
        # Arctic Ocean
        {'lat_range': (70, 85), 'lon_range': (-180, 180), 'density': 200}
    ]

    all_points = []

    for region in ocean_regions:
        num_points = region['density']
        lats = np.random.uniform(region['lat_range'][0], region['lat_range'][1], num_points)

        # Handle longitude wrapping
        if region['lon_range'][1] < region['lon_range'][0]:  # Pacific case
            lons1 = np.random.uniform(region['lon_range'][0], 180, num_points//2)
            lons2 = np.random.uniform(-180, region['lon_range'][1], num_points - num_points//2)
            lons = np.concatenate([lons1, lons2])
        else:
            lons = np.random.uniform(region['lon_range'][0], region['lon_range'][1], num_points)

        # Generate other attributes
        pressures = np.random.exponential(1.5, num_points) * 1000  # Depth in meters
        years = np.random.choice([2021, 2022, 2023, 2024, 2025], num_points, p=[0.1, 0.15, 0.25, 0.35, 0.15])
        parameters = np.random.choice(['BBP470', 'BBP532', 'BBP700', 'CDOM'], num_points, p=[0.3, 0.25, 0.25, 0.2])

        for i in range(num_points):
            all_points.append({
                'lat': lats[i],
                'lon': lons[i],
                'pressure': pressures[i],
                'year': years[i],
                'parameter': parameters[i],
                'days_ago': np.random.randint(1, 3650)  # Random days for filtering
            })

    return pd.DataFrame(all_points)

# Load data
data = generate_sample_data()

def filter_data(df, time_filter, pressure_range, parameters, years):
    """Filter data based on current selections"""
    filtered_df = df.copy()

    # Time filtering
    if time_filter != 'ALL':
        days_map = {'3 days': 3, '10 days': 10, '1 year': 365, '10 years': 3650}
        max_days = days_map.get(time_filter, 3650)
        filtered_df = filtered_df[filtered_df['days_ago'] <= max_days]

    # Pressure filtering (convert km to meters)
    min_pressure = pressure_range[0] * 1000
    max_pressure = pressure_range[1] * 1000
    filtered_df = filtered_df[
        (filtered_df['pressure'] >= min_pressure) &
        (filtered_df['pressure'] <= max_pressure)
    ]

    # Parameter filtering
    if parameters:
        filtered_df = filtered_df[filtered_df['parameter'].isin(parameters)]

    # Year filtering
    if years:
        year_ints = [int(y) for y in years]
        filtered_df = filtered_df[filtered_df['year'].isin(year_ints)]

    return filtered_df

def create_world_map(filtered_data):
    """Create the main interactive world map"""
    # Create base map
    m = folium.Map(
        location=[20, 0],  # Center on equator
        zoom_start=2,
        tiles=None,
        width='100%',
        height='100%'
    )

    # Add different tile layers
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Satellite',
        overlay=False,
        control=True
    ).add_to(m)

    folium.TileLayer(
        tiles='OpenStreetMap',
        name='Streets',
        overlay=False,
        control=True
    ).add_to(m)

    # Add data points as yellow circles
    for idx, row in filtered_data.iterrows():
        if idx > 2000:  # Limit points for performance
            break

        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=3,
            popup=f"""
            <div style="font-family: Arial, sans-serif;">
                <b>Oceanographic Data Point</b><br>
                <b>Location:</b> {row['lat']:.2f}°, {row['lon']:.2f}°<br>
                <b>Pressure:</b> {row['pressure']:.0f} m<br>
                <b>Parameter:</b> {row['parameter']}<br>
                <b>Year:</b> {row['year']}<br>
                <b>Days ago:</b> {row['days_ago']}
            </div>
            """,
            tooltip=f"{row['parameter']} - {row['year']}",
            color='#ffd700',
            weight=1,
            fillColor='#ffd700',
            fillOpacity=0.8
        ).add_to(m)

    # Add layer control
    folium.LayerControl().add_to(m)

    # Add scale
    folium.plugins.MeasureControl().add_to(m)

    return m

# Main layout
def main():
    # Page header
    st.markdown("""
    <div style="background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
                padding: 1rem; margin-bottom: 1rem; border-radius: 8px;">
        <h1 style="color: white; margin: 0; font-size: 1.8rem;">
            Global Oceanographic Data Visualization
        </h1>
        <p style="color: #e3f2fd; margin: 0; font-size: 1rem;">
            Interactive mapping and analysis of global ocean observations
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Create two-column layout
    col1, col2 = st.columns([2, 8])  # 20% and 80% split

    with col1:
        st.markdown('<div class="control-panel">', unsafe_allow_html=True)
        create_control_panel()
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        create_map_panel()

def create_control_panel():
    """Create the left control and filter panel"""

    # Time/Date Filter
    st.markdown('<div class="filter-header">Time Range</div>', unsafe_allow_html=True)

    time_options = ['3 days', '10 days', '1 year', '10 years', 'ALL']
    selected_time = st.radio(
        "",
        time_options,
        index=time_options.index(st.session_state.selected_time_filter),
        key="time_filter",
        horizontal=True
    )
    st.session_state.selected_time_filter = selected_time

    # Pressure Range Slider
    st.markdown('<div class="filter-header">Minimum deepest pressure (km)</div>', unsafe_allow_html=True)
    pressure_range = st.slider(
        "",
        min_value=0.0,
        max_value=6.0,
        value=st.session_state.pressure_range,
        step=0.1,
        key="pressure_slider"
    )
    st.session_state.pressure_range = pressure_range

    # Location Search
    st.markdown('<div class="filter-header">Location</div>', unsafe_allow_html=True)
    location_search = st.text_input(
        "",
        placeholder="Everywhere",
        value=st.session_state.location_search,
        key="location_input"
    )
    st.session_state.location_search = location_search

    # Search button
    if st.button("🔍 Search", key="search_btn", use_container_width=True):
        st.rerun()

    # Parameters Filter
    st.markdown('<div class="filter-header">Parameters</div>', unsafe_allow_html=True)

    parameter_options = ['BBP470', 'BBP532', 'BBP700', 'CDOM']
    parameter_counts = {
        'BBP470': len(data[data['parameter'] == 'BBP470']),
        'BBP532': len(data[data['parameter'] == 'BBP532']),
        'BBP700': len(data[data['parameter'] == 'BBP700']),
        'CDOM': len(data[data['parameter'] == 'CDOM'])
    }

    selected_parameters = []
    for param in parameter_options:
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.checkbox(param, value=param in st.session_state.selected_parameters, key=f"param_{param}"):
                selected_parameters.append(param)
        with col2:
            st.markdown(f'<span class="parameter-count">{parameter_counts[param]}</span>', unsafe_allow_html=True)

    st.session_state.selected_parameters = selected_parameters

    # Deployment Year Filter
    st.markdown('<div class="filter-header">Deployment year</div>', unsafe_allow_html=True)

    year_options = ['2025', '2024', '2023', '2022', '2021']
    year_counts = {
        '2025': len(data[data['year'] == 2025]),
        '2024': len(data[data['year'] == 2024]),
        '2023': len(data[data['year'] == 2023]),
        '2022': len(data[data['year'] == 2022]),
        '2021': len(data[data['year'] == 2021])
    }

    selected_years = []
    for year in year_options:
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.checkbox(year, value=year in st.session_state.selected_years, key=f"year_{year}"):
                selected_years.append(year)
        with col2:
            st.markdown(f'<span class="parameter-count">{year_counts[year]}</span>', unsafe_allow_html=True)

    st.session_state.selected_years = selected_years

    # Filter data and show count
    filtered_data = filter_data(
        data,
        st.session_state.selected_time_filter,
        st.session_state.pressure_range,
        st.session_state.selected_parameters,
        st.session_state.selected_years
    )

    # Data count display
    st.markdown(f"""
    <div class="data-count">
        {len(filtered_data):,} cycles selected
    </div>
    """, unsafe_allow_html=True)

    # Action buttons
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Reset", key="reset_btn", use_container_width=True):
            # Reset all filters
            st.session_state.selected_time_filter = 'ALL'
            st.session_state.pressure_range = [0, 6]
            st.session_state.location_search = ''
            st.session_state.selected_parameters = ['BBP470', 'BBP532']
            st.session_state.selected_years = ['2024', '2023']
            st.rerun()

    with col2:
        if st.button("View", key="view_btn", use_container_width=True):
            st.session_state.refresh_map = True

    with col3:
        if st.button("Export", key="export_btn", use_container_width=True):
            # Create CSV download
            csv = filtered_data.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"oceanographic_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

def create_map_panel():
    """Create the main map panel"""

    # Filter data based on current selections
    filtered_data = filter_data(
        data,
        st.session_state.selected_time_filter,
        st.session_state.pressure_range,
        st.session_state.selected_parameters,
        st.session_state.selected_years
    )

    # Create and display map
    world_map = create_world_map(filtered_data)

    # Display map with custom height
    map_data = st_folium(
        world_map,
        width='100%',
        height=700,
        returned_objects=["last_object_clicked", "bounds"]
    )

    # Map statistics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Data Points", f"{len(filtered_data):,}")

    with col2:
        unique_params = filtered_data['parameter'].nunique()
        st.metric("Parameters", unique_params)

    with col3:
        unique_years = filtered_data['year'].nunique()
        st.metric("Years Covered", unique_years)

    with col4:
        avg_depth = filtered_data['pressure'].mean() / 1000
        st.metric("Avg Depth", f"{avg_depth:.1f} km")

if __name__ == "__main__":
    main()