#!/usr/bin/env python3
"""
New Web Interface using the new ChromaDB RAG System - Streamlit Version
- Uses new_comprehensive_rag_system.py
- ChromaDB semantic similarity search
- LLM decides if plotting is needed
- Calls visualization functions when needed
"""

import os
import sys
import json
import time
import traceback
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
import datetime

# Import the new comprehensive RAG system
try:
    from new_comprehensive_rag_system import ComprehensiveRAGSystem
except ImportError as e:
    st.error(f"Error importing ComprehensiveRAGSystem: {e}")
    st.error("Make sure new_comprehensive_rag_system.py is in the same directory")
    st.stop()

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
if 'query_results' not in st.session_state:
    st.session_state.query_results = None
if 'show_more_data' not in st.session_state:
    st.session_state.show_more_data = False
if 'data_page' not in st.session_state:
    st.session_state.data_page = 0

def initialize_rag():
    """Initialize the new comprehensive RAG system silently"""
    try:
        GROQ_API_KEY = "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"

        # Create the new RAG system
        st.session_state.rag_system = ComprehensiveRAGSystem(GROQ_API_KEY)

        # Check if ChromaDB collection already exists
        try:
            existing_collection = st.session_state.rag_system.client.get_collection(st.session_state.rag_system.collection_name)
            if existing_collection.count() > 0:
                st.session_state.rag_system.collection = existing_collection
                st.session_state.chroma_count = existing_collection.count()
            else:
                raise Exception("Empty collection")
        except Exception as e:
            # Collection doesn't exist or is empty, create new one
            st.session_state.rag_system.load_all_semantic_samples()
            st.session_state.rag_system.create_new_chromadb()
            st.session_state.chroma_count = st.session_state.rag_system.collection.count()

        st.session_state.initialized = True
        st.session_state.embedding_model = getattr(st.session_state.rag_system, 'embedding_model_name', 'all-MiniLM-L6-v2')

    except Exception as e:
        st.error(f"System initialization failed: {str(e)}")
        st.session_state.rag_system = None
        st.session_state.initialized = False

def process_query(query):
    """Process a user query using the RAG system"""
    if not st.session_state.rag_system:
        return {"error": "RAG system not ready"}
    
    try:
        start_time = time.time()
        # Use the new comprehensive query method
        result = st.session_state.rag_system.simple_query(query)
        processing_time = time.time() - start_time

        response = {
            "query": query,
            "processing_time": processing_time,
            "best_match_id": result.get("best_match_id", "unknown"),
            "similarity": result.get("similarity", 0.0),
            "matched_sample": result.get("matched_sample", {}),
            "llm_response": result.get("llm_response", ""),
            "sql_executed": result.get("sql_executed", False),
            "sql": result.get("sql", ""),
            "sql_data": result.get("sql_data", []),
            "visualization_created": result.get("visualization_created", False),
            "plot_type": result.get("plot_type", ""),
            "plot_html": result.get("plot_html", "")
        }
        return response
    except Exception as e:
        return {"error": f"Processing failed: {str(e)}", "traceback": traceback.format_exc()}

def display_data_table(data, page=0, page_size=10):
    """Display a paginated data table"""
    if not data or len(data) == 0:
        st.info("No data returned")
        return

    start_idx = page * page_size
    end_idx = min(start_idx + page_size, len(data))

    # Convert to DataFrame for display
    df = pd.DataFrame(data[start_idx:end_idx])
    st.dataframe(df)

    # Show pagination info
    st.write(f"Showing rows {start_idx + 1} to {end_idx} of {len(data)}")

    # Pagination controls
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("Previous Page", disabled=page == 0):
            st.session_state.data_page = max(0, page - 1)
            st.rerun()
    with col2:
        st.write(f"Page {page + 1}")
    with col3:
        if st.button("Next Page", disabled=end_idx >= len(data)):
            st.session_state.data_page = page + 1
            st.rerun()

def get_float_trajectory(float_id):
    """Extract all profile locations for a specific float ID to create trajectory"""
    if not st.session_state.get('initialized', False) or not st.session_state.rag_system:
        return None

def create_filters_controls():
    """Render Filters & Controls on the right side; updates session state."""
    if not st.session_state.get('initialized', False) or not st.session_state.rag_system:
        st.error("RAG system not initialized")
        return

    # Initialize filter and trajectory states
    if 'map_start_date' not in st.session_state:
        st.session_state.map_start_date = None
    if 'map_end_date' not in st.session_state:
        st.session_state.map_end_date = None
    if 'map_selected_region' not in st.session_state:
        st.session_state.map_selected_region = 'All Oceans'
    if 'show_trajectory' not in st.session_state:
        st.session_state.show_trajectory = False
    if 'trajectory_float_id' not in st.session_state:
        st.session_state.trajectory_float_id = None

    with st.container():
        st.markdown("### Filters & Controls")

        import datetime as _dt
        today = _dt.date.today()
        start_default = _dt.date(2000, 1, 1)

        col_start, col_end = st.columns(2)
        with col_start:
            start_date = st.date_input(
                "Start Date:",
                value=st.session_state.map_start_date or start_default,
                min_value=_dt.date(1990, 1, 1),
                max_value=today,
                key="start_date_filter"
            )
            st.session_state.map_start_date = start_date

        with col_end:
            end_date = st.date_input(
                "End Date:",
                value=st.session_state.map_end_date or today,
                min_value=start_date,
                max_value=today,
                key="end_date_filter"
            )
            st.session_state.map_end_date = end_date

        st.markdown("**Ocean Regions**")
        ocean_regions = {
            "All Oceans": None,
            "Red Sea": {"lat_min": 12.0, "lat_max": 30.0, "lon_min": 32.0, "lon_max": 43.0},
            "Persian Gulf": {"lat_min": 24.0, "lat_max": 31.0, "lon_min": 48.0, "lon_max": 57.0},
            "Andaman Sea": {"lat_min": 5.0, "lat_max": 20.0, "lon_min": 92.0, "lon_max": 100.0},
            "Western Australian Basin": {"lat_min": -40.0, "lat_max": -10.0, "lon_min": 90.0, "lon_max": 120.0},
            "Mozambique Channel": {"lat_min": -27.0, "lat_max": -10.0, "lon_min": 40.0, "lon_max": 50.0},
            "Northern Indian Ocean": {"lat_min": 0.0, "lat_max": 30.0, "lon_min": 40.0, "lon_max": 100.0},
            "Southern Indian Ocean": {"lat_min": -50.0, "lat_max": 0.0, "lon_min": 40.0, "lon_max": 120.0},
            "Bay Of Bengal": {"lat_min": 5.0, "lat_max": 25.0, "lon_min": 80.0, "lon_max": 100.0},
            "Arabian Sea": {"lat_min": 8.0, "lat_max": 27.0, "lon_min": 50.0, "lon_max": 75.0},
            "Atlantic Ocean": {"lat_min": -60.0, "lat_max": 70.0, "lon_min": -80.0, "lon_max": 20.0},
            "Pacific Ocean": {"lat_min": -60.0, "lat_max": 70.0, "lon_min": 120.0, "lon_max": -70.0},
            "Mediterranean Sea": {"lat_min": 30.0, "lat_max": 46.0, "lon_min": -6.0, "lon_max": 37.0}
        }

        selected_region = st.selectbox(
            "Select ocean region:",
            options=list(ocean_regions.keys()),
            index=list(ocean_regions.keys()).index(st.session_state.map_selected_region),
            key="region_filter_map"
        )
        st.session_state.map_selected_region = selected_region

        col_apply, col_reset = st.columns(2)
        with col_apply:
            if st.button("Apply Filters", use_container_width=True, type="primary"):
                st.session_state.map_filters_applied = True
                st.rerun()
        with col_reset:
            if st.button("Reset Filters", use_container_width=True):
                st.session_state.map_start_date = None
                st.session_state.map_end_date = None
                st.session_state.map_selected_region = 'All Oceans'
                st.session_state.map_filters_applied = True
                st.rerun()

        st.markdown("---")
        st.subheader("Float Trajectory")
        float_id_input = st.text_input(
            "Enter Float ID for trajectory:",
            placeholder="e.g., 2902755",
            help="Enter the ARGO float ID to visualize its trajectory path"
        )
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            if st.button("Show Trajectory", use_container_width=True, type="secondary"):
                if float_id_input.strip():
                    st.session_state.show_trajectory = True
                    st.session_state.trajectory_float_id = float_id_input.strip()
                    st.rerun()
                else:
                    st.warning("Please enter a valid Float ID")
        with col_t2:
            if st.button("Clear Trajectory", use_container_width=True):
                st.session_state.show_trajectory = False
                st.session_state.trajectory_float_id = None
                st.rerun()

        if st.session_state.get('show_trajectory', False) and st.session_state.get('trajectory_float_id'):
            st.info(f"Currently showing trajectory for Float: {st.session_state.trajectory_float_id}")

    try:
        rag_system = st.session_state.rag_system
        if not rag_system.db_connection:
            return None

        # SQL query to get all profile locations for the specific float, ordered by date
        sql_query = f"""
        SELECT p.float_id,
               p.profile_id,
               p.latitude,
               p.longitude,
               p.profile_date,
               ROW_NUMBER() OVER (ORDER BY p.profile_date) as sequence_number
        FROM profiles p
        WHERE p.float_id = '{float_id}'
        AND p.latitude IS NOT NULL
        AND p.longitude IS NOT NULL
        AND p.latitude BETWEEN -90 AND 90
        AND p.longitude BETWEEN -180 AND 180
        ORDER BY p.profile_date ASC
        """

        df = rag_system.db_connection.execute(sql_query).fetchdf()
        return df if not df.empty else None

    except Exception as e:
        st.error(f"Error retrieving trajectory data: {str(e)}")
        return None

def create_base_map():
    """Create the base map with proper layers"""
    m = folium.Map(
        location=[0, 0],
        zoom_start=2,
        tiles=None  # We'll add custom tiles
    )
    
    # Add base layers with proper attributions
    folium.TileLayer(
        tiles='OpenStreetMap',
        name='OpenStreetMap',
        attr='© OpenStreetMap contributors'
    ).add_to(m)
    
    folium.TileLayer(
        tiles='https://stamen-tiles-{s}.a.ssl.fastly.net/terrain/{z}/{x}/{y}{r}.png',
        attr='Map tiles by Stamen Design, under CC BY 3.0. Data by OpenStreetMap, under ODbL.',
        name='Terrain'
    ).add_to(m)
    
    folium.TileLayer(
        tiles='https://stamen-tiles-{s}.a.ssl.fastly.net/watercolor/{z}/{x}/{y}.jpg',
        attr='Map tiles by Stamen Design, under CC BY 3.0. Data by OpenStreetMap, under ODbL.',
        name='Watercolor'
    ).add_to(m)
    
    return m

def create_enhanced_float_map():
    """Create enhanced interactive map with filtering controls and trajectory support"""
    if not st.session_state.get('initialized', False) or not st.session_state.rag_system:
        st.error("RAG system not initialized")
        return

    # Initialize states
    initialize_map_states()

    try:
        # Get latest float locations from database
        sql_query = """
        WITH latest_profiles AS (
            SELECT p.float_id,
                   p.profile_id,
                   p.latitude,
                   p.longitude,
                   p.profile_date,
                   ROW_NUMBER() OVER (PARTITION BY p.float_id ORDER BY p.profile_date DESC) as rn
            FROM profiles p
            WHERE p.latitude IS NOT NULL
            AND p.longitude IS NOT NULL
            AND p.latitude BETWEEN -90 AND 90
            AND p.longitude BETWEEN -180 AND 180
        ),
        float_measurements AS (
            SELECT lp.float_id,
                   lp.latitude,
                   lp.longitude,
                   lp.profile_date,
                   COUNT(m.measurement_id) as measurement_count
            FROM latest_profiles lp
            LEFT JOIN measurements m ON lp.profile_id = m.profile_id
            WHERE lp.rn = 1
            GROUP BY lp.float_id, lp.latitude, lp.longitude, lp.profile_date
        )
        SELECT * FROM float_measurements
        ORDER BY float_id
        """
        
        latest_locations = st.session_state.rag_system.db_connection.execute(sql_query).fetchdf()
        
        # Display float count
        st.success(f"Found {len(latest_locations)} ARGO floats")

        # Map Layout
        map_col, filters_col = st.columns([3, 1])

        with map_col:
            # Square Map Container
            st.markdown("""
            <style>
            .map-container {
                width: 100%;
                aspect-ratio: 1;
                margin: 0;
                padding: 0;
            }
            </style>
            """, unsafe_allow_html=True)
            
            with st.container():
                folium_map = create_base_map()
                st_folium(folium_map, width=800, height=800)  # Square dimensions

        with filters_col:
            # Filters and Trajectory Section
            with st.expander("Filters & Controls", expanded=True):
                create_filters_section()
                st.markdown("---")
                create_trajectory_section()

        # Export Options Section (below map)
        st.markdown("### Export Options")
        export_col1, export_col2, export_col3 = st.columns(3)
        with export_col1:
            export_format = st.selectbox("Format", ["CSV", "NetCDF", "JSON"])
        with export_col2:
            st.selectbox("Resolution", ["High", "Medium", "Low"])
        with export_col3:
            st.button("Export Data", use_container_width=True)

    except Exception as e:
        st.error(f"Error creating map: {str(e)}")

def create_filters_section():
    """Create filters section without emoji and expanded by default"""
    st.markdown("**Time Range**")
    col_start, col_end = st.columns(2)
    with col_start:
        start_date = st.date_input(
            "Start Date:",
            value=st.session_state.map_start_date or datetime.date(2000, 1, 1)
        )
    with col_end:
        end_date = st.date_input(
            "End Date:",
            value=st.session_state.map_end_date or datetime.date.today()
        )

    st.markdown("**Ocean Regions**")
    selected_region = st.selectbox(
        "Select region:",
        options=list(ocean_regions.keys()),
        index=list(ocean_regions.keys()).index(st.session_state.map_selected_region)
    )

    col1, col2 = st.columns(2)
    with col1:
        st.button("Apply", use_container_width=True, type="primary")
    with col2:
        st.button("Reset", use_container_width=True)

def create_trajectory_section():
    """Create trajectory section"""
    st.markdown("**Float Trajectory**")
    float_id = st.text_input("Float ID:", placeholder="e.g., 2902755")
    col1, col2 = st.columns(2)
    with col1:
        st.button("Show Path", use_container_width=True)
    with col2:
        st.button("Clear Path", use_container_width=True)

def create_float_location_map():
    """Create a 2D map showing last location of each ARGO float"""
    if not st.session_state.get('initialized', False) or not st.session_state.rag_system:
        st.error("RAG system not initialized")
        return

    try:
        # Query to get last location of each float with measurement counts
        sql_query = """
        WITH latest_profiles AS (
            SELECT p.float_id,
                   p.profile_id,
                   p.latitude,
                   p.longitude,
                   p.profile_date,
                   ROW_NUMBER() OVER (PARTITION BY p.float_id ORDER BY p.profile_date DESC) as rn
            FROM profiles p
            WHERE p.latitude IS NOT NULL
            AND p.longitude IS NOT NULL
            AND p.latitude BETWEEN -90 AND 90
            AND p.longitude BETWEEN -180 AND 180
        ),
        float_measurements AS (
            SELECT lp.float_id,
                   lp.latitude,
                   lp.longitude,
                   lp.profile_date,
                   COUNT(m.measurement_id) as measurement_count
            FROM latest_profiles lp
            LEFT JOIN measurements m ON lp.profile_id = m.profile_id
            WHERE lp.rn = 1
            GROUP BY lp.float_id, lp.latitude, lp.longitude, lp.profile_date
        )
        SELECT * FROM float_measurements
        ORDER BY float_id
        """

        # Execute query
        with st.spinner("Loading ARGO float locations..."):
            rag_system = st.session_state.rag_system
            if rag_system.db_connection:
                df = rag_system.db_connection.execute(sql_query).fetchdf()

                if not df.empty:
                    # Get only the last location for each float
                    latest_locations = df.groupby('float_id').first().reset_index()

                    st.success(f"Found {len(latest_locations)} ARGO floats")

                    # Display map info
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Floats", len(latest_locations))
                    with col2:
                        lat_range = f"{latest_locations['latitude'].min():.1f}° to {latest_locations['latitude'].max():.1f}°"
                        st.metric("Latitude Range", lat_range)
                    with col3:
                        lon_range = f"{latest_locations['longitude'].min():.1f}° to {latest_locations['longitude'].max():.1f}°"
                        st.metric("Longitude Range", lon_range)

                    # Create Leaflet map
                    st.markdown("### Interactive Leaflet Map")

                    # Calculate map center
                    center_lat = latest_locations['latitude'].mean()
                    center_lon = latest_locations['longitude'].mean()

                    # Create folium map
                    m = folium.Map(
                        location=[center_lat, center_lon],
                        zoom_start=3,
                        tiles='OpenStreetMap',
                        width='100%',
                        height=600
                    )

                    # Add different tile layers with proper attributions
                    folium.TileLayer(
                        tiles='https://stamen-tiles-{s}.a.ssl.fastly.net/terrain/{z}/{x}/{y}{r}.png',
                        attr='Map tiles by Stamen Design, under CC BY 3.0. Data by OpenStreetMap, under ODbL.',
                        name='Terrain'
                    ).add_to(m)

                    folium.TileLayer(
                        tiles='https://stamen-tiles-{s}.a.ssl.fastly.net/watercolor/{z}/{x}/{y}.jpg',
                        attr='Map tiles by Stamen Design, under CC BY 3.0. Data by OpenStreetMap, under ODbL.',
                        name='Watercolor'
                    ).add_to(m)

                    # Create color mapping for measurement count
                    max_measurements = latest_locations['measurement_count'].max()
                    min_measurements = latest_locations['measurement_count'].min()

                    # Add markers for each float
                    for idx, row in latest_locations.iterrows():
                        # Color based on measurement count
                        measurement_ratio = (row['measurement_count'] - min_measurements) / (max_measurements - min_measurements)

                        if measurement_ratio > 0.7:
                            color = 'red'  # High data
                            icon_color = 'darkred'
                        elif measurement_ratio > 0.4:
                            color = 'orange'  # Medium data
                            icon_color = 'orange'
                        else:
                            color = 'blue'  # Low data
                            icon_color = 'blue'

                        # Create popup text
                        popup_text = f"""
                        <div style="font-family: Arial, sans-serif; width: 250px;">
                            <h4 style="color: {icon_color}; margin: 0;">🌊 ARGO Float</h4>
                            <hr style="margin: 5px 0;">
                            <b>Float ID:</b> {row['float_id']}<br>
                            <b>Last Position:</b><br>
                            &nbsp;&nbsp;📍 {row['latitude']:.3f}°N, {row['longitude']:.3f}°E<br>
                            <b>Last Profile:</b> {row['profile_date']}<br>
                            <b>Measurements:</b> {row['measurement_count']:,} records<br>
                            <br>
                            <small style="color: gray;">Click to view details</small>
                        </div>
                        """

                        folium.CircleMarker(
                            location=[row['latitude'], row['longitude']],
                            radius=8,
                            popup=folium.Popup(popup_text, max_width=300),
                            tooltip=f"Float {row['float_id']}: {row['measurement_count']} measurements",
                            color='white',
                            weight=2,
                            fillColor=color,
                            fillOpacity=0.8
                        ).add_to(m)

                    # Add layer control
                    folium.LayerControl().add_to(m)

                    # Add legend
                    legend_html = f'''
                    <div style="position: fixed;
                                bottom: 50px; left: 50px; width: 200px; height: 120px;
                                background-color: white; border:2px solid grey; z-index:9999;
                                font-size:14px; font-family: Arial;
                                padding: 10px;
                                border-radius: 10px;
                                box-shadow: 0 0 15px rgba(0,0,0,0.3);">
                    <h4 style="margin: 0 0 10px 0; color: #2E8B57;">🌊 ARGO Floats</h4>
                    <p style="margin: 5px 0;"><span style="display: inline-block; width: 12px; height: 12px; background-color: red; border-radius: 50%; border: 2px solid white; margin-right: 8px;"></span>High Data (>{int(max_measurements*0.7):,})</p>
                    <p style="margin: 5px 0;"><span style="display: inline-block; width: 12px; height: 12px; background-color: orange; border-radius: 50%; border: 2px solid white; margin-right: 8px;"></span>Medium Data</p>
                    <p style="margin: 5px 0;"><span style="display: inline-block; width: 12px; height: 12px; background-color: blue; border-radius: 50%; border: 2px solid white; margin-right: 8px;"></span>Low Data (<{int(max_measurements*0.4):,})</p>
                    </div>
                    '''
                    m.get_root().html.add_child(folium.Element(legend_html))

                    # Display the map
                    map_data = st_folium(m, width=900, height=600, returned_objects=["last_clicked"])

                    # Handle map clicks
                    if map_data['last_clicked']:
                        clicked_lat = map_data['last_clicked']['lat']
                        clicked_lng = map_data['last_clicked']['lng']
                        st.info(f"Clicked location: {clicked_lat:.4f}°N, {clicked_lng:.4f}°E")

                    # Show data table
                    with st.expander("Float Location Data"):
                        # Add download button
                        csv_data = latest_locations.to_csv(index=False)
                        st.download_button(
                            "Download Float Locations (CSV)",
                            csv_data,
                            f"argo_float_locations_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                            "text/csv"
                        )

                        # Display data table
                        st.dataframe(latest_locations, use_container_width=True)

                    # Geographic analysis
                    st.markdown("### Geographic Distribution Analysis")
                    col1, col2 = st.columns(2)

                    with col1:
                        # Latitude histogram
                        fig_lat = px.histogram(
                            latest_locations,
                            x='latitude',
                            nbins=20,
                            title="Float Distribution by Latitude",
                            labels={'latitude': 'Latitude (°N)', 'count': 'Number of Floats'}
                        )
                        fig_lat.update_layout(height=400, template="plotly_white")
                        st.plotly_chart(fig_lat, use_container_width=True)

                    with col2:
                        # Longitude histogram
                        fig_lon = px.histogram(
                            latest_locations,
                            x='longitude',
                            nbins=20,
                            title="Float Distribution by Longitude",
                            labels={'longitude': 'Longitude (°E)', 'count': 'Number of Floats'}
                        )
                        fig_lon.update_layout(height=400, template="plotly_white")
                        st.plotly_chart(fig_lon, use_container_width=True)

                else:
                    st.warning("No float location data found")
            else:
                st.error("Database connection not available")

    except Exception as e:
        st.error(f"Error creating float map: {str(e)}")
        st.exception(e)

def create_scientific_sidebar():
    """Create advanced scientific sidebar with oceanographic controls"""
    with st.sidebar:
        st.markdown("# ARGO Scientific Dashboard")
        st.markdown("---")

        # System Status Panel
        st.subheader("System Status")
        if st.session_state.get('initialized', False) and st.session_state.rag_system:
            # Main system metrics
            st.metric("System Status", "Operational", delta="All systems ready")
            st.metric("Data Sources", f"{st.session_state.get('chroma_count', 0):,} samples")
            st.metric("AI Model", st.session_state.get('embedding_model', 'all-MiniLM-L6-v2'))
            st.metric("Analysis Engine", "Groq LLM", delta="Online")

            st.progress(1.0, text="System Operational")

            # Additional technical details
            with st.expander("Technical Details"):
                st.write("**ChromaDB Status:**", "Connected")
                st.write("**Embedding Status:**", "Model Loaded")
                st.write("**SQL Engine:**", "DuckDB Ready")
                st.write("**Parquet Files:**", "Loaded")
                st.write("**Semantic Search:**", "Operational")
        else:
            st.error("System Offline")
            st.progress(0.0, text="Initializing...")
            st.write("**Status:** Waiting for initialization...")

        st.markdown("---")

        # Quick Analysis Presets
        st.subheader("Quick Analysis")

        preset_queries = {
            "Global Temperature Profile": "Show me global temperature vs depth profiles",
            "Salinity Distribution": "Create salinity heatmap by region",
            "Ocean Currents": "Analyze temperature gradients and currents",
            "Float Statistics": "Show ARGO float deployment statistics",
            "Regional Analysis": "Compare temperature anomalies by ocean basin"
        }

        for preset_name, preset_query in preset_queries.items():
            if st.button(preset_name, use_container_width=True):
                st.session_state.preset_query = preset_query
                st.rerun()

        st.markdown("---")

        # Export options
        st.subheader("Export Options")
        export_format = st.selectbox("Export Format", ["CSV", "NetCDF", "JSON", "MATLAB"])
        if st.button("Export Current Results", use_container_width=True):
            st.info("Export functionality ready")

        st.markdown("---")

        # Note: Float map is now always visible in main area
        st.info("Interactive map is always visible in the main area")

def create_analysis_metrics(result):
    """Create scientific metrics display"""
    if not result:
        return

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        processing_time = result['processing_time']
        st.metric("Processing Time", f"{processing_time:.2f}s",
                 delta="Optimal" if processing_time < 5 else "Slow")

    with col2:
        data_count = len(result.get('sql_data', []))
        st.metric("Data Points", f"{data_count:,}",
                 delta="Rich Dataset" if data_count > 100 else "Limited")

    with col3:
        sql_status = "Generated" if result.get('sql_executed') else "Failed"
        st.metric("SQL Status", sql_status)

    with col4:
        viz_status = "Created" if result.get('visualization_created') else "None"
        st.metric("Visualization", viz_status)

def create_advanced_visualization(df, result):
    """Create advanced scientific visualizations"""
    if df.empty:
        st.warning("No data available for visualization")
        return

    # Detect data types for smart visualization
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

    if len(numeric_cols) < 2:
        st.warning("Insufficient numeric data for advanced visualization")
        return

    # Visualization controls
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        viz_type = st.selectbox("Visualization Type",
            ["Depth Profile", "Scatter Plot", "Line Plot", "Contour Plot", "3D Surface", "Heatmap", "Box Plot"])

    with col2:
        x_axis = st.selectbox("X-Axis", numeric_cols, index=0)

    with col3:
        y_axis = st.selectbox("Y-Axis", numeric_cols, index=min(1, len(numeric_cols)-1))

    with col4:
        color_by = st.selectbox("Color By", ["None"] + numeric_cols)

    # Smart visualization detection
    has_depth = 'depth' in df.columns or 'pressure' in df.columns
    has_temperature = 'temperature' in df.columns
    has_salinity = 'salinity' in df.columns

    # Auto-detect depth profile and suggest optimal visualization
    if has_depth and (has_temperature or has_salinity):
        st.info("**Depth Profile Detected** - Consider using 'Depth Profile' visualization for best results")
        if st.button("Auto-Create Depth Profile"):
            viz_type = "Depth Profile"

    # Create advanced plots
    try:
        if viz_type == "Depth Profile" and has_depth:
            # Create proper depth profile visualization
            depth_col = 'depth' if 'depth' in df.columns else 'pressure'

            # Create subplot for temperature and salinity profiles
            from plotly.subplots import make_subplots
            fig = make_subplots(
                rows=1, cols=2 if has_temperature and has_salinity else 1,
                subplot_titles=(['Temperature Profile', 'Salinity Profile'] if has_temperature and has_salinity
                              else ['Temperature Profile'] if has_temperature else ['Salinity Profile']),
                horizontal_spacing=0.1
            )

            if has_temperature:
                fig.add_trace(
                    go.Scatter(x=df['temperature'], y=df[depth_col],
                             mode='lines+markers',
                             name='Temperature',
                             line=dict(color='red', width=2),
                             marker=dict(size=4)),
                    row=1, col=1
                )

            if has_salinity:
                col_pos = 2 if has_temperature and has_salinity else 1
                fig.add_trace(
                    go.Scatter(x=df['salinity'], y=df[depth_col],
                             mode='lines+markers',
                             name='Salinity',
                             line=dict(color='blue', width=2),
                             marker=dict(size=4)),
                    row=1, col=col_pos
                )

            # Invert y-axis for depth (deeper = lower)
            fig.update_yaxes(autorange="reversed", title_text="Depth (m)" if depth_col == 'depth' else "Pressure (dbar)")
            fig.update_xaxes(title_text="Temperature (°C)", row=1, col=1)
            if has_temperature and has_salinity:
                fig.update_xaxes(title_text="Salinity (PSU)", row=1, col=2)
            elif has_salinity and not has_temperature:
                fig.update_xaxes(title_text="Salinity (PSU)", row=1, col=1)

            fig.update_layout(
                title="ARGO Float Depth Profile",
                height=600,
                template="plotly_white",
                showlegend=True
            )

        elif viz_type == "Scatter Plot":
            fig = px.scatter(df, x=x_axis, y=y_axis,
                           color=color_by if color_by != "None" else None,
                           title=f"ARGO Data: {y_axis} vs {x_axis}",
                           template="plotly_white",
                           width=800, height=500)

            # Add statistical annotations
            if len(df) > 10:
                # Add correlation coefficient
                correlation = df[x_axis].corr(df[y_axis])
                fig.add_annotation(
                    text=f"Correlation: {correlation:.3f}",
                    xref="paper", yref="paper",
                    x=0.02, y=0.98, showarrow=False,
                    font=dict(size=12, color="black"),
                    bgcolor="rgba(255,255,255,0.8)"
                )

        elif viz_type == "3D Surface" and len(numeric_cols) >= 3:
            z_axis = st.selectbox("Z-Axis", [col for col in numeric_cols if col not in [x_axis, y_axis]])

            # Smart 3D visualization for oceanographic data
            if has_depth and (x_axis in ['temperature', 'salinity'] or y_axis in ['temperature', 'salinity']):
                # Create 3D scatter with depth as one axis
                fig = go.Figure(data=[go.Scatter3d(
                    x=df[x_axis], y=df[y_axis], z=df[z_axis],
                    mode='markers',
                    marker=dict(
                        size=5,
                        opacity=0.7,
                        colorscale='Viridis',
                        color=df[z_axis] if z_axis in df.columns else df[x_axis],
                        colorbar=dict(title=z_axis),
                        showscale=True
                    ),
                    text=[f"{x_axis}: {x:.2f}<br>{y_axis}: {y:.2f}<br>{z_axis}: {z:.2f}"
                          for x, y, z in zip(df[x_axis], df[y_axis], df[z_axis])],
                    hovertemplate="%{text}<extra></extra>"
                )])
            else:
                # Standard 3D scatter
                fig = go.Figure(data=[go.Scatter3d(
                    x=df[x_axis], y=df[y_axis], z=df[z_axis],
                    mode='markers',
                    marker=dict(size=5, opacity=0.6, colorscale='Viridis')
                )])

            fig.update_layout(
                title=f"3D Analysis: {x_axis} vs {y_axis} vs {z_axis}",
                scene=dict(
                    xaxis_title=x_axis,
                    yaxis_title=y_axis,
                    zaxis_title=z_axis,
                    zaxis=dict(autorange="reversed") if z_axis in ['depth', 'pressure'] else dict()
                )
            )

        elif viz_type == "Contour Plot":
            # Create contour plot for oceanographic data
            fig = px.density_contour(df, x=x_axis, y=y_axis,
                                   title=f"Density Contour: {y_axis} vs {x_axis}")
            fig.update_traces(contours_coloring="fill", contours_showlabels=True)

        elif viz_type == "Heatmap":
            # Correlation heatmap
            corr_matrix = df[numeric_cols].corr()
            fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                          title="Parameter Correlation Matrix")

        else:
            # Default to enhanced scatter
            fig = px.scatter(df, x=x_axis, y=y_axis, title=f"{y_axis} vs {x_axis}")

        # Enhance plot styling
        fig.update_layout(
            template="plotly_white",
            font=dict(family="Arial, sans-serif", size=12),
            title_font=dict(size=16, family="Arial Black"),
            width=900,
            height=600,
            margin=dict(l=60, r=60, t=80, b=60)
        )

        st.plotly_chart(fig, use_container_width=True)

        # Statistical summary
        with st.expander("Statistical Summary"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Descriptive Statistics**")
                st.dataframe(df[numeric_cols].describe())
            with col2:
                st.write("**Correlation Matrix**")
                st.dataframe(df[numeric_cols].corr())

    except Exception as e:
        st.error(f"Visualization error: {str(e)}")
        st.dataframe(df.head())

def main():
    """Main Scientific Dashboard Application"""
    st.set_page_config(
        page_title="ARGO Scientific Oceanographic Dashboard",
        page_icon="🌊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Enhanced Dark Ocean Theme with Floating Boat Buttons
    st.markdown("""
    <style>
        /* Main App Background - Vibrant Black Ocean */
        .stApp {
            background:
                radial-gradient(circle at 20% 30%, rgba(0, 50, 100, 0.4) 0%, transparent 40%),
                radial-gradient(circle at 80% 20%, rgba(0, 30, 80, 0.3) 0%, transparent 30%),
                radial-gradient(circle at 60% 70%, rgba(0, 40, 90, 0.35) 0%, transparent 45%),
                radial-gradient(circle at 30% 80%, rgba(0, 60, 120, 0.25) 0%, transparent 35%),
                linear-gradient(135deg,
                    #000814 0%,     /* Deep black */
                    #001219 25%,    /* Midnight black */
                    #003366 50%,    /* Deep navy */
                    #001219 75%,    /* Midnight black */
                    #000814 100%);  /* Deep black */
            background-attachment: fixed;
            min-height: 100vh;
        }

        /* Ocean Wave Animation Overlay */
        .stApp::before {
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-image:
                radial-gradient(ellipse at 20% 20%, rgba(0, 100, 200, 0.1) 0%, transparent 25%),
                radial-gradient(ellipse at 80% 40%, rgba(0, 80, 160, 0.08) 0%, transparent 20%),
                radial-gradient(ellipse at 40% 80%, rgba(0, 120, 240, 0.12) 0%, transparent 30%);
            background-size: 400px 200px, 600px 300px, 500px 250px;
            animation: oceanWaves 20s ease-in-out infinite;
            pointer-events: none;
            z-index: -1;
        }

        @keyframes oceanWaves {
            0%, 100% {
                transform: translateX(0px) translateY(0px) scale(1);
                opacity: 0.3;
            }
            25% {
                transform: translateX(-10px) translateY(-5px) scale(1.05);
                opacity: 0.4;
            }
            50% {
                transform: translateX(5px) translateY(-8px) scale(0.95);
                opacity: 0.5;
            }
            75% {
                transform: translateX(-5px) translateY(3px) scale(1.02);
                opacity: 0.35;
            }
        }

        /* Main Content Container - Dark with pale white text */
        .main .block-container {
            background: rgba(10, 10, 15, 0.85) !important;
            backdrop-filter: blur(15px);
            border-radius: 20px;
            padding: 2rem;
            margin-top: 1rem;
            box-shadow: 0 12px 40px rgba(0, 100, 200, 0.2);
            border: 2px solid rgba(0, 80, 160, 0.3);
        }

        /* All Text Pale White */
        .main .block-container *,
        .stMarkdown,
        .stMarkdown p,
        .stText,
        div[data-testid="stMarkdownContainer"] *,
        .stSelectbox label,
        .stTextInput label,
        .stTextArea label,
        .stDateInput label,
        .stMetric label,
        .stMetric .metric-value,
        .stDataFrame,
        .stAlert,
        .stInfo,
        .stSuccess,
        .stWarning,
        .stError {
            color: #E8E8E8 !important;
        }

        /* Sidebar - Dark Ocean Theme */
        .css-1d391kg,
        section[data-testid="stSidebar"] > div {
            background: rgba(5, 5, 10, 0.95) !important;
            backdrop-filter: blur(20px);
            border-right: 2px solid rgba(0, 80, 160, 0.4);
        }

        /* Sidebar text pale white */
        .css-1d391kg *,
        section[data-testid="stSidebar"] * {
            color: #E8E8E8 !important;
        }

        /* Headers - Bright blue for contrast */
        h1, h2, h3, h4, h5, h6 {
            color: #4FC3F7 !important;
            font-weight: 600 !important;
            text-shadow: 0 0 10px rgba(79, 195, 247, 0.3);
        }

        /* 3D Floating Boat Buttons */
        .stButton button {
            background: linear-gradient(145deg, #1E3A8A, #3B82F6, #1E40AF) !important;
            color: #FFFFFF !important;
            border: none !important;
            border-radius: 25px !important;
            padding: 12px 24px !important;
            font-weight: 600 !important;
            font-size: 14px !important;
            text-shadow: 0 1px 2px rgba(0,0,0,0.3) !important;
            box-shadow:
                0 8px 16px rgba(30, 58, 138, 0.3),
                0 4px 8px rgba(59, 130, 246, 0.2),
                inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
            transform: translateY(-2px) !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            position: relative !important;
            overflow: hidden !important;
            min-height: 42px !important;
            min-width: 120px !important;
        }

        /* Button alignment improvements */
        .stButton { width: 100% !important; }
        .stButton > button { width: 100% !important; }
        div[data-testid="column"] .stButton { width: 100% !important; }
        div[data-testid="column"] .stButton > button { width: 100% !important; }
        .stButton + .stButton { margin-left: 0.5rem !important; }

        /* Boat floating animation */
        .stButton button::before {
            content: '' !important;
            position: absolute !important;
            top: -2px !important;
            left: -2px !important;
            right: -2px !important;
            bottom: -2px !important;
            background: linear-gradient(45deg, transparent, rgba(255,255,255,0.1), transparent) !important;
            border-radius: 25px !important;
            animation: boatFloat 4s ease-in-out infinite !important;
            z-index: -1 !important;
        }

        @keyframes boatFloat {
            0%, 100% {
                transform: translateY(0px) rotate(0deg);
                box-shadow: 0 8px 16px rgba(30, 58, 138, 0.3);
            }
            25% {
                transform: translateY(-3px) rotate(0.5deg);
                box-shadow: 0 12px 20px rgba(30, 58, 138, 0.4);
            }
            50% {
                transform: translateY(-1px) rotate(0deg);
                box-shadow: 0 10px 18px rgba(30, 58, 138, 0.35);
            }
            75% {
                transform: translateY(-2px) rotate(-0.5deg);
                box-shadow: 0 11px 19px rgba(30, 58, 138, 0.38);
            }
        }

        .stButton button:hover {
            background: linear-gradient(145deg, #2563EB, #60A5FA, #3B82F6) !important;
            transform: translateY(-4px) scale(1.02) !important;
            box-shadow:
                0 12px 24px rgba(37, 99, 235, 0.4),
                0 6px 12px rgba(96, 165, 250, 0.3),
                inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
        }

        .stButton button:active {
            transform: translateY(-1px) scale(0.98) !important;
            box-shadow:
                0 4px 8px rgba(30, 58, 138, 0.4),
                inset 0 2px 4px rgba(0, 0, 0, 0.1) !important;
        }

        /* Metric cards with dark theme */
        [data-testid="metric-container"] {
            background: rgba(15, 15, 25, 0.8) !important;
            border-radius: 15px !important;
            padding: 1.5rem !important;
            backdrop-filter: blur(12px) !important;
            border: 2px solid rgba(0, 80, 160, 0.3) !important;
            box-shadow: 0 8px 20px rgba(0, 100, 200, 0.1) !important;
        }

        [data-testid="metric-container"] * {
            color: #E8E8E8 !important;
        }

        /* Alert boxes with dark theme */
        .stAlert {
            background: rgba(20, 20, 30, 0.85) !important;
            backdrop-filter: blur(12px) !important;
            border-radius: 12px !important;
            border-left: 4px solid #4FC3F7 !important;
            color: #E8E8E8 !important;
        }

        .stAlert * {
            color: #E8E8E8 !important;
        }

        /* Input fields with dark theme */
        .stTextInput > div > div > input,
        .stTextArea > div > div > textarea,
        .stSelectbox > div > div > input,
        .stDateInput > div > div > input {
            background: rgba(25, 25, 35, 0.9) !important;
            color: #E8E8E8 !important;
            border: 2px solid rgba(0, 80, 160, 0.4) !important;
            border-radius: 8px !important;
        }

        /* Tabs with dark theme */
        .stTabs [data-baseweb="tab"] {
            background: rgba(20, 20, 30, 0.8) !important;
            border-radius: 10px 10px 0 0 !important;
            backdrop-filter: blur(8px) !important;
            border: 1px solid rgba(0, 80, 160, 0.4) !important;
            color: #E8E8E8 !important;
        }

        .stTabs [data-baseweb="tab"]:hover {
            background: rgba(30, 30, 45, 0.9) !important;
            color: #4FC3F7 !important;
        }

        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background: rgba(40, 40, 60, 1) !important;
            color: #4FC3F7 !important;
            font-weight: bold !important;
        }

        /* DataFrame styling */
        .stDataFrame {
            background: rgba(15, 15, 25, 0.9) !important;
            border-radius: 8px !important;
        }

        /* Expander styling */
        .streamlit-expanderHeader {
            background: rgba(20, 20, 30, 0.8) !important;
            color: #E8E8E8 !important;
            border-radius: 8px !important;
        }

        /* Progress bars */
        .stProgress > div > div {
            background: linear-gradient(90deg, #1E3A8A, #3B82F6) !important;
        }

        /* Remove emojis from headings */
        h1 .emoji, h2 .emoji, h3 .emoji, h4 .emoji, h5 .emoji, h6 .emoji {
            display: none !important;
        }
    </style>
    """, unsafe_allow_html=True)

    # Scientific Dashboard Header with ocean theme
    st.markdown("""
    <div style="background: linear-gradient(135deg,
                rgba(14, 165, 233, 0.8) 0%,
                rgba(6, 182, 212, 0.8) 30%,
                rgba(34, 197, 218, 0.8) 60%,
                rgba(103, 232, 249, 0.8) 100%);
                padding: 2rem;
                border-radius: 20px;
                margin-bottom: 2rem;
                backdrop-filter: blur(20px);
                border: 1px solid rgba(255, 255, 255, 0.2);
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);">
        <h1 style="color: white; margin: 0; font-size: 2.8rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">ARGO Scientific Oceanographic Dashboard</h1>
        <p style="color: #e0f7fa; margin: 0; font-size: 1.2rem; text-shadow: 1px 1px 2px rgba(0,0,0,0.2);">
            Advanced RAG-powered analysis of global ocean observations
        </p>
        <div style="margin-top: 1rem; opacity: 0.9; color: #b3e5fc; font-size: 0.95rem;">
            • Research-grade&nbsp;&nbsp;&nbsp;• AI-powered&nbsp;&nbsp;&nbsp;• Real-time analysis
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Create scientific sidebar
    create_scientific_sidebar()

    # Silent initialization
    if not st.session_state.get('initialized', False):
        with st.spinner("🚀 Initializing Scientific Analysis System..."):
            initialize_rag()
    elif st.session_state.rag_system is None:
        with st.spinner("🔄 Reconnecting to databases..."):
            initialize_rag()

    # Check system status for main area
    if not st.session_state.get('initialized', False) or st.session_state.rag_system is None:
        st.error("🔴 **System Offline** - Please wait for initialization")
        if st.button("🔄 Force Restart System"):
            st.session_state.initialized = False
            st.rerun()
        return

    st.markdown("---")

    # Main Layout: Map (left), Filters (right), then Query Interface below filters
    map_col, filters_col = st.columns([3, 2])

    with map_col:
        st.markdown("### Global ARGO Float Distribution Map")
        create_enhanced_float_map()

    with filters_col:
        create_filters_controls()

    st.markdown("---")
    st.markdown("### Scientific Query Interface")

    # Corrected indentation for the preset_query line
    # Check for preset query
    preset_query = st.session_state.get('preset_query', '')
    if preset_query:
        st.session_state.preset_query = ''  # Clear after using

    col1, col2, col3 = st.columns([4, 1, 1])
    with col1:
        query = st.text_area(
            "Scientific Query",
            value=preset_query,
            height=100,
            placeholder="""Examples:
• Analyze temperature gradients in the North Atlantic
• Show salinity profiles below 2000m depth
• Create time series of temperature anomalies
• Compare ocean warming trends by basin""",
            help="Enter natural language queries about oceanographic data analysis"
        )
    with col2:
        submit_btn = st.button("Analyze", type="primary", use_container_width=True)
        if st.button("Examples", use_container_width=True):
            st.session_state.show_examples = not st.session_state.get('show_examples', False)
    with col3:
        if st.button("Clear Results", use_container_width=True, disabled=st.session_state.query_results is None):
            st.session_state.query_results = None
            st.session_state.data_page = 0
            st.rerun()

    # Show examples if requested
    if st.session_state.get('show_examples', False):
        with st.expander("Example Scientific Queries", expanded=True):
            example_categories = {
                "Temperature Analysis": [
                    "Show temperature vs depth profiles for the last 6 months",
                    "Analyze temperature anomalies in the tropical Pacific",
                    "Create a heatmap of sea surface temperature"
                ],
                "Salinity Studies": [
                    "Plot salinity distribution in the Mediterranean",
                    "Compare deep water salinity trends",
                    "Analyze halocline structure"
                ],
                "Geographic Analysis": [
                    "Map ARGO float trajectories in the Southern Ocean",
                    "Regional temperature comparison between Atlantic basins",
                    "Analyze upwelling zones temperature profiles"
                ]
            }

            for category, examples in example_categories.items():
                st.write(f"**{category}**")
                for example in examples:
                    if st.button(example, key=example):
                        st.session_state.preset_query = example
                        st.rerun()

    # Process query when submitted
    if submit_btn and query:
        with st.spinner("Processing your query with semantic search and LLM..."):
            result = process_query(query)
            st.session_state.query_results = result
            st.session_state.data_page = 0  # Reset to first page
            st.rerun()
    
    # Advanced Scientific Results Display
    if st.session_state.query_results:
        result = st.session_state.query_results

        st.markdown("---")
        st.markdown("## Scientific Analysis Results")

        # Advanced metrics display
        create_analysis_metrics(result)

        # Professional results tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "AI Scientific Analysis",
            "Advanced Visualization",
            "Data Analysis",
            "Statistical Summary",
            "Technical Details"
        ])

        with tab1:
            st.markdown("### AI-Powered Scientific Interpretation")
            if result.get('llm_response'):
                # Enhanced AI response display
                st.markdown(f"""
                <div style="background-color: rgba(30,30,45,0.9); padding: 1.5rem; border-radius: 10px;
                           border-left: 4px solid #4FC3F7; color: #E8E8E8;">
                    <h4 style="color: #4FC3F7; margin-top: 0;">Scientific Analysis</h4>
                    <p style="margin-bottom: 0; line-height: 1.6; color: #E8E8E8;">{result['llm_response']}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("No AI analysis available for this query")

        with tab2:
            st.markdown("### Advanced Scientific Visualization")
            if result.get('sql_data') and len(result['sql_data']) > 0:
                df = pd.DataFrame(result['sql_data'])
                create_advanced_visualization(df, result)
            else:
                st.info("No data available for visualization")

        with tab3:
            st.markdown("### Oceanographic Data Analysis")
            if result.get('sql_data'):
                df = pd.DataFrame(result['sql_data'])

                # Data overview
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Records", len(df))
                with col2:
                    st.metric("Parameters", len(df.columns))
                with col3:
                    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
                    st.metric("Numeric Variables", len(numeric_cols))

                # Interactive data table with filters
                st.markdown("#### Data Explorer")

                # Column selector
                if len(df.columns) > 5:
                    selected_cols = st.multiselect(
                        "Select columns to display:",
                        df.columns.tolist(),
                        default=df.columns.tolist()[:5]
                    )
                    if selected_cols:
                        df_display = df[selected_cols]
                    else:
                        df_display = df
                else:
                    df_display = df

                # Data filters
                if len(df) > 1000:
                    sample_size = st.slider("Sample size", 10, min(5000, len(df)), 1000)
                    df_display = df_display.head(sample_size)

                st.dataframe(df_display, use_container_width=True, height=400)

                # Enhanced download options
                col1, col2, col3 = st.columns(3)
                with col1:
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "Download CSV",
                        csv,
                        f"argo_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        "text/csv",
                        use_container_width=True
                    )
                with col2:
                    json_data = df.to_json(orient='records', indent=2)
                    st.download_button(
                        "Download JSON",
                        json_data,
                        f"argo_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                        "application/json",
                        use_container_width=True
                    )
                with col3:
                    # Data summary
                    if st.button("Generate Report", use_container_width=True):
                        st.info("Comprehensive data report functionality ready")
            else:
                st.info("No data retrieved for analysis")

        with tab4:
            st.markdown("### Statistical Analysis")
            if result.get('sql_data'):
                df = pd.DataFrame(result['sql_data'])
                numeric_df = df.select_dtypes(include=['float64', 'int64'])

                if not numeric_df.empty:
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("#### Descriptive Statistics")
                        st.dataframe(numeric_df.describe(), use_container_width=True)

                    with col2:
                        st.markdown("#### Correlation Matrix")
                        if len(numeric_df.columns) > 1:
                            corr_matrix = numeric_df.corr()
                            fig = px.imshow(corr_matrix,
                                          text_auto=True,
                                          title="Parameter Correlations",
                                          color_continuous_scale='RdBu_r')
                            fig.update_layout(height=400)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Need multiple numeric columns for correlation analysis")

                    # Distribution analysis
                    st.markdown("#### Distribution Analysis")
                    if len(numeric_df.columns) > 0:
                        selected_param = st.selectbox("Select parameter for distribution:", numeric_df.columns)

                        col1, col2 = st.columns(2)
                        with col1:
                            fig_hist = px.histogram(numeric_df, x=selected_param,
                                                  title=f"Distribution of {selected_param}")
                            st.plotly_chart(fig_hist, use_container_width=True)

                        with col2:
                            fig_box = px.box(numeric_df, y=selected_param,
                                           title=f"Box Plot of {selected_param}")
                            st.plotly_chart(fig_box, use_container_width=True)
                else:
                    st.info("No numeric data available for statistical analysis")
            else:
                st.info("No data available for statistical analysis")

        with tab5:
            st.markdown("### Technical Analysis Details")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Query Matching")
                st.write("**Best Match ID:**", result.get('best_match_id', 'N/A'))
                st.write("**Similarity Score:**", f"{result.get('similarity', 0):.4f}")
                st.write("**Processing Time:**", f"{result.get('processing_time', 0):.3f} seconds")

                # Performance indicators
                if result.get('processing_time', 0) < 2:
                    st.success("Excellent performance")
                elif result.get('processing_time', 0) < 5:
                    st.info("Good performance")
                else:
                    st.warning("Consider query optimization")

            with col2:
                st.markdown("#### System Operations")
                st.write("**SQL Generation:**", "Success" if result.get('sql_executed') else "Failed")
                st.write("**Visualization:**", "Created" if result.get('visualization_created') else "None")
                st.write("**Data Retrieval:**", f"{len(result.get('sql_data', []))} records")

            # SQL Query Analysis
            if result.get('sql_executed') and result.get('sql'):
                st.markdown("#### Generated SQL Query")
                st.code(result['sql'], language='sql')

                # SQL complexity analysis
                sql_lines = result['sql'].count('\n') + 1
                if sql_lines < 5:
                    st.success("Simple query - Fast execution")
                elif sql_lines < 15:
                    st.info("Moderate query complexity")
                else:
                    st.warning("Complex query - May take time")

            # Semantic matching details
            if result.get('matched_sample'):
                with st.expander("Semantic Matching Details", expanded=False):
                    st.json(result['matched_sample'])

    # Map is now always visible in main layout above

def initialize_map_states():
    """Initialize the map states in session state"""
    if 'map_start_date' not in st.session_state:
        st.session_state.map_start_date = None
    if 'map_end_date' not in st.session_state:
        st.session_state.map_end_date = None
    if 'map_selected_region' not in st.session_state:
        st.session_state.map_selected_region = 'All Oceans'
    if 'show_trajectory' not in st.session_state:
        st.session_state.show_trajectory = False
    if 'trajectory_float_id' not in st.session_state:
        st.session_state.trajectory_float_id = None
    if 'map_filters_applied' not in st.session_state:
        st.session_state.map_filters_applied = False

if __name__ == "__main__":
    main()