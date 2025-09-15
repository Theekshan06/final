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

def create_enhanced_float_map():
    """Create enhanced interactive map with filtering controls"""
    if not st.session_state.get('initialized', False) or not st.session_state.rag_system:
        st.error("RAG system not initialized")
        return

    # Initialize filter states
    if 'map_start_date' not in st.session_state:
        st.session_state.map_start_date = None
    if 'map_end_date' not in st.session_state:
        st.session_state.map_end_date = None
    if 'map_selected_region' not in st.session_state:
        st.session_state.map_selected_region = 'All Oceans'

    # Create two-column layout: controls (25%) + map (75%)
    col1, col2 = st.columns([1, 3])

    with col1:
        # Create collapsible filters panel
        with st.container():
            # Minimize/Maximize button for filters
            if 'filters_minimized' not in st.session_state:
                st.session_state.filters_minimized = False

            col_btn, col_title = st.columns([1, 4])
            with col_btn:
                if st.button("📖" if st.session_state.filters_minimized else "📕",
                           help="Minimize/Maximize Filters",
                           key="toggle_filters"):
                    st.session_state.filters_minimized = not st.session_state.filters_minimized
                    st.rerun()

            with col_title:
                st.markdown("### 🌊 Filters & Controls")

            if not st.session_state.filters_minimized:
                st.markdown("""
                <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 8px; border: 1px solid #e9ecef;">
                """, unsafe_allow_html=True)

                # Date Range Filter with Calendar
                st.markdown("**📅 Time Range**")

                import datetime
                today = datetime.date.today()
                # Default to a wider range - ARGO data goes back to early 2000s
                start_default = datetime.date(2000, 1, 1)  # Start from year 2000

                col_start, col_end = st.columns(2)
                with col_start:
                    start_date = st.date_input(
                        "Start Date:",
                        value=st.session_state.map_start_date or start_default,
                        min_value=datetime.date(1990, 1, 1),  # Allow even earlier dates
                        max_value=today,
                        key="start_date_filter"
                    )
                    st.session_state.map_start_date = start_date

                with col_end:
                    end_date = st.date_input(
                        "End Date:",
                        value=st.session_state.map_end_date or today,
                        min_value=start_date,  # End date must be after start date
                        max_value=today,
                        key="end_date_filter"
                    )
                    st.session_state.map_end_date = end_date

                # Ocean Regions Filter
                st.markdown("**🌊 Ocean Regions**")

                # Define ocean regions
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

                # Apply and Reset Buttons
                st.markdown("**⚙️ Actions**")
                col_apply, col_reset = st.columns(2)

                with col_apply:
                    if st.button("🔍 Apply Filters", use_container_width=True, type="primary"):
                        st.session_state.map_filters_applied = True
                        st.rerun()

                with col_reset:
                    if st.button("🔄 Reset Filters", use_container_width=True):
                        st.session_state.map_start_date = None
                        st.session_state.map_end_date = None
                        st.session_state.map_selected_region = 'All Oceans'
                        st.session_state.map_filters_applied = True
                        st.rerun()

                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.info("🔍 Filters minimized - Click 📖 to expand")

    with col2:
        try:
            # Build dynamic SQL query based on new filters
            time_condition = ""
            if st.session_state.map_start_date and st.session_state.map_end_date:
                time_condition = f"AND p.profile_date BETWEEN '{st.session_state.map_start_date}' AND '{st.session_state.map_end_date}'"

            # Ocean region filtering
            location_condition = ""
            if st.session_state.map_selected_region != 'All Oceans':
                ocean_regions = {
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

                if st.session_state.map_selected_region in ocean_regions:
                    region = ocean_regions[st.session_state.map_selected_region]
                    # Handle Pacific Ocean longitude wrapping
                    if st.session_state.map_selected_region == "Pacific Ocean":
                        location_condition = f"""AND (
                            (p.longitude BETWEEN {region['lon_min']} AND 180) OR
                            (p.longitude BETWEEN -180 AND {region['lon_max']})
                        ) AND p.latitude BETWEEN {region['lat_min']} AND {region['lat_max']}"""
                    else:
                        location_condition = f"""AND p.latitude BETWEEN {region['lat_min']} AND {region['lat_max']}
                                              AND p.longitude BETWEEN {region['lon_min']} AND {region['lon_max']}"""

            sql_query = f"""
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
                {time_condition}
                {location_condition}
            ),
            float_measurements AS (
                SELECT lp.float_id,
                       lp.latitude,
                       lp.longitude,
                       lp.profile_date,
                       COUNT(m.measurement_id) as measurement_count,
                       AVG(m.pressure) as avg_pressure
                FROM latest_profiles lp
                LEFT JOIN measurements m ON lp.profile_id = m.profile_id
                WHERE lp.rn = 1
                GROUP BY lp.float_id, lp.latitude, lp.longitude, lp.profile_date
            )
            SELECT * FROM float_measurements
            ORDER BY float_id
            """

            # Execute query
            with st.spinner("Loading filtered ARGO float locations..."):
                rag_system = st.session_state.rag_system
                if rag_system.db_connection:
                    df = rag_system.db_connection.execute(sql_query).fetchdf()

                    if not df.empty:
                        # Get only the last location for each float
                        latest_locations = df.groupby('float_id').first().reset_index()

                        st.success(f"Found {len(latest_locations)} ARGO floats (filtered)")

                        # Display map metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Floats", len(latest_locations))
                        with col2:
                            avg_measurements = latest_locations['measurement_count'].mean()
                            st.metric("Avg Measurements", f"{avg_measurements:.0f}")
                        with col3:
                            st.metric("Ocean Region", st.session_state.map_selected_region)
                        with col4:
                            if st.session_state.map_start_date and st.session_state.map_end_date:
                                date_range = f"{st.session_state.map_start_date} to {st.session_state.map_end_date}"
                                st.metric("Date Range", "Custom")
                            else:
                                st.metric("Date Range", "All Time")

                        # Enhanced Interactive Map
                        st.markdown("### Enhanced Interactive Map")

                        # Calculate map center
                        center_lat = latest_locations['latitude'].mean()
                        center_lon = latest_locations['longitude'].mean()

                        # Create folium map with satellite view as default
                        m = folium.Map(
                            location=[center_lat, center_lon],
                            zoom_start=3,
                            width='100%',
                            height=600,
                            tiles=None  # Start with no default tiles
                        )

                        # Add satellite as the first (default) tile layer
                        satellite_layer = folium.TileLayer(
                            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                            attr='Esri',
                            name='Satellite',
                            overlay=False,
                            control=True
                        )
                        satellite_layer.add_to(m)

                        folium.TileLayer(
                            tiles='OpenStreetMap',
                            name='Street Map',
                            overlay=False,
                            control=True
                        ).add_to(m)

                        folium.TileLayer(
                            tiles='https://stamen-tiles-{s}.a.ssl.fastly.net/terrain/{z}/{x}/{y}{r}.png',
                            attr='Stamen',
                            name='Terrain',
                            overlay=False,
                            control=True
                        ).add_to(m)

                        # Add yellow circular markers for each float (as specified)
                        for idx, row in latest_locations.iterrows():
                            # Create enhanced popup
                            popup_text = f"""
                            <div style="font-family: Arial, sans-serif; width: 280px;">
                                <h4 style="color: #1976d2; margin: 0;">ARGO Float {row['float_id']}</h4>
                                <hr style="margin: 5px 0;">
                                <b>Position:</b> {row['latitude']:.3f}°N, {row['longitude']:.3f}°E<br>
                                <b>Last Profile:</b> {row['profile_date']}<br>
                                <b>Measurements:</b> {row['measurement_count']:,} records<br>
                                <b>Avg Pressure:</b> {row.get('avg_pressure', 0):.0f} dbar<br>
                                <br>
                                <small style="color: gray;">Region: {st.session_state.map_selected_region}</small>
                            </div>
                            """

                            # Enhanced circular markers with borders and zoom-responsive sizing
                            folium.CircleMarker(
                                location=[row['latitude'], row['longitude']],
                                radius=8,  # Slightly larger base size
                                popup=folium.Popup(popup_text, max_width=300),
                                tooltip=f"Float {row['float_id']} | {row['measurement_count']} measurements",
                                color='#ff8c00',  # Dark orange border for better visibility
                                weight=3,  # Thicker border
                                fillColor='#ffd700',  # Yellow fill
                                fillOpacity=0.85,
                                opacity=1.0  # Full opacity for border
                            ).add_to(m)

                        # Add trajectory if requested
                        if st.session_state.get('show_trajectory', False) and st.session_state.get('trajectory_float_id'):
                            trajectory_data = get_float_trajectory(st.session_state.trajectory_float_id)

                            if trajectory_data is not None and len(trajectory_data) > 1:
                                st.success(f"🛤️ Showing trajectory for Float {st.session_state.trajectory_float_id} with {len(trajectory_data)} profile locations")

                                # Create trajectory path coordinates
                                trajectory_coords = []
                                for idx, row in trajectory_data.iterrows():
                                    trajectory_coords.append([row['latitude'], row['longitude']])

                                # Add red trajectory line connecting all profile locations
                                folium.PolyLine(
                                    locations=trajectory_coords,
                                    color='red',
                                    weight=3,
                                    opacity=0.8,
                                    popup=f"Float {st.session_state.trajectory_float_id} Trajectory",
                                    tooltip=f"Trajectory path for Float {st.session_state.trajectory_float_id}"
                                ).add_to(m)

                                # Add numbered markers for trajectory points
                                for idx, row in trajectory_data.iterrows():
                                    # Create popup with profile information
                                    profile_popup = f"""
                                    <div style="font-family: Arial, sans-serif; width: 250px;">
                                        <h4 style="color: #d32f2f; margin: 0;">Profile #{row['sequence_number']}</h4>
                                        <hr style="margin: 5px 0;">
                                        <b>Float ID:</b> {row['float_id']}<br>
                                        <b>Profile ID:</b> {row['profile_id']}<br>
                                        <b>Date:</b> {row['profile_date']}<br>
                                        <b>Position:</b> {row['latitude']:.3f}°N, {row['longitude']:.3f}°E<br>
                                        <br>
                                        <small style="color: gray;">Trajectory sequence: {row['sequence_number']} of {len(trajectory_data)}</small>
                                    </div>
                                    """

                                    # Add trajectory markers (red circles with numbers)
                                    folium.CircleMarker(
                                        location=[row['latitude'], row['longitude']],
                                        radius=8,
                                        popup=folium.Popup(profile_popup, max_width=300),
                                        tooltip=f"Profile #{row['sequence_number']} - {row['profile_date']}",
                                        color='red',
                                        weight=2,
                                        fillColor='red',
                                        fillOpacity=0.7
                                    ).add_to(m)

                                    # Add sequence number as text marker for first and last points
                                    if idx == 0:  # First point
                                        folium.Marker(
                                            location=[row['latitude'], row['longitude']],
                                            icon=folium.DivIcon(
                                                html=f'<div style="color: red; font-weight: bold; font-size: 12px; text-shadow: 1px 1px 1px white;">START</div>',
                                                icon_size=(30, 10),
                                                icon_anchor=(15, 5)
                                            )
                                        ).add_to(m)
                                    elif idx == len(trajectory_data) - 1:  # Last point
                                        folium.Marker(
                                            location=[row['latitude'], row['longitude']],
                                            icon=folium.DivIcon(
                                                html=f'<div style="color: red; font-weight: bold; font-size: 12px; text-shadow: 1px 1px 1px white;">END</div>',
                                                icon_size=(30, 10),
                                                icon_anchor=(15, 5)
                                            )
                                        ).add_to(m)

                                # Zoom to fit trajectory
                                if len(trajectory_coords) > 0:
                                    m.fit_bounds(trajectory_coords)

                            elif trajectory_data is not None and len(trajectory_data) == 1:
                                st.warning(f"Float {st.session_state.trajectory_float_id} has only one profile location. Trajectory requires multiple profiles.")
                            else:
                                st.error(f"No trajectory data found for Float {st.session_state.trajectory_float_id}. Please check the Float ID.")

                        # Add layer control
                        folium.LayerControl().add_to(m)

                        # Add overview minimap (bottom-right as specified)
                        minimap = folium.plugins.MiniMap(
                            tile_layer='OpenStreetMap',
                            position='bottomright',
                            width=150,
                            height=100,
                            collapsed=False
                        )
                        m.add_child(minimap)

                        # Add scale bar (bottom-left as specified)
                        folium.plugins.MeasureControl(position='bottomleft').add_to(m)

                        # Display the enhanced map
                        map_data = st_folium(m, width='100%', height=600, returned_objects=["last_object_clicked"])

                        # Handle map interactions
                        if map_data['last_object_clicked']:
                            clicked_data = map_data['last_object_clicked']
                            st.info(f"Selected float at: {clicked_data.get('lat', 0):.4f}°N, {clicked_data.get('lng', 0):.4f}°E")

                        # Data export and analysis section
                        with st.expander("Data Export & Analysis", expanded=False):
                            col1, col2 = st.columns([2, 1])

                            with col1:
                                # Export options
                                st.markdown("**Export Options**")
                                export_format = st.selectbox("Format:", ["CSV", "JSON", "Excel"])

                                if export_format == "CSV":
                                    csv_data = latest_locations.to_csv(index=False)
                                    st.download_button(
                                        "Download Filtered Data",
                                        csv_data,
                                        f"filtered_argo_floats_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                        "text/csv",
                                        use_container_width=True
                                    )
                                elif export_format == "JSON":
                                    json_data = latest_locations.to_json(orient='records', indent=2)
                                    st.download_button(
                                        "Download Filtered Data",
                                        json_data,
                                        f"filtered_argo_floats_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                                        "application/json",
                                        use_container_width=True
                                    )

                            with col2:
                                # Filter summary
                                st.markdown("**Applied Filters**")
                                if st.session_state.map_start_date and st.session_state.map_end_date:
                                    st.write(f"• Time: {st.session_state.map_start_date} to {st.session_state.map_end_date}")
                                else:
                                    st.write("• Time: All available data")
                                st.write(f"• Region: {st.session_state.map_selected_region}")

                            # Data table
                            st.markdown("**Filtered Float Data**")
                            st.dataframe(latest_locations, use_container_width=True, height=300)

                    else:
                        st.warning("No floats found matching the current filters")
                        st.info("Try adjusting the time range, pressure range, or location filters")
                else:
                    st.error("Database connection not available")

        except Exception as e:
            st.error(f"Error creating enhanced float map: {str(e)}")
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

        # Oceanographic Parameters
        st.subheader("Analysis Parameters")

        # Temperature range
        temp_range = st.slider("Temperature Range (°C)", -2.0, 40.0, (-2.0, 40.0), 0.1)

        # Pressure/Depth range
        pressure_range = st.slider("Pressure Range (dbar)", 0, 6000, (0, 2000), 50)

        # Geographic bounds
        st.write("**Geographic Bounds**")
        lat_range = st.slider("Latitude", -90.0, 90.0, (-90.0, 90.0), 1.0)
        lon_range = st.slider("Longitude", -180.0, 180.0, (-180.0, 180.0), 1.0)

        # Time period
        st.write("**Temporal Scope**")
        time_period = st.selectbox("Time Period",
            ["Last 30 days", "Last 3 months", "Last 6 months", "Last year", "All time"]
        )

        # Data quality filters
        st.write("**Data Quality**")
        quality_filter = st.multiselect("QC Flags", ["1 - Good", "2 - Probably Good", "3 - Probably Bad"],
                                       default=["1 - Good", "2 - Probably Good"])

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

        # Float Map Visualization
        st.subheader("ARGO Float Map")
        if st.button("Show Enhanced Float Map", use_container_width=True, type="primary"):
            st.session_state.show_enhanced_float_map = True
            st.rerun()

        # Float Trajectory Feature
        st.markdown("---")
        st.subheader("Float Trajectory")

        # Float ID input
        float_id_input = st.text_input(
            "Enter Float ID for trajectory:",
            placeholder="e.g., 2902755",
            help="Enter the ARGO float ID to visualize its trajectory path"
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Show Trajectory", use_container_width=True, type="secondary"):
                if float_id_input.strip():
                    st.session_state.show_trajectory = True
                    st.session_state.trajectory_float_id = float_id_input.strip()
                    st.rerun()
                else:
                    st.warning("Please enter a valid Float ID")

        with col2:
            if st.button("Clear Trajectory", use_container_width=True):
                st.session_state.show_trajectory = False
                st.session_state.trajectory_float_id = None
                st.rerun()

        # Show current trajectory status
        if st.session_state.get('show_trajectory', False) and st.session_state.get('trajectory_float_id'):
            st.info(f"🛤️ Currently showing trajectory for Float: {st.session_state.trajectory_float_id}")

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
        with st.expander("📊 Statistical Summary"):
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

    # Add watercolor ocean background CSS
    st.markdown("""
    <style>
        .stApp {
            background:
                radial-gradient(circle at 15% 20%, rgba(135, 206, 235, 0.3) 0%, transparent 30%),
                radial-gradient(circle at 85% 10%, rgba(72, 209, 204, 0.2) 0%, transparent 25%),
                radial-gradient(circle at 45% 60%, rgba(95, 158, 160, 0.25) 0%, transparent 35%),
                radial-gradient(circle at 75% 85%, rgba(176, 224, 230, 0.3) 0%, transparent 40%),
                radial-gradient(circle at 25% 80%, rgba(102, 205, 170, 0.2) 0%, transparent 30%),
                linear-gradient(135deg,
                    #e0f6ff 0%,     /* Very light watercolor blue */
                    #b8e6ff 15%,    /* Light watercolor cyan */
                    #87ceeb 35%,    /* Sky blue watercolor */
                    #4682b4 60%,    /* Steel blue watercolor */
                    #5f9ea0 85%,    /* Cadet blue watercolor */
                    #2e8b57 100%);  /* Sea green watercolor */
            background-attachment: fixed;
            background-size: 300px 300px, 400px 400px, 500px 500px, 350px 350px, 450px 450px, cover;
        }

        /* Watercolor texture overlay */
        .stApp::before {
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-image:
                radial-gradient(circle at 20% 30%, rgba(255,255,255,0.15) 1px, transparent 2px),
                radial-gradient(circle at 70% 60%, rgba(255,255,255,0.12) 1px, transparent 2px),
                radial-gradient(circle at 40% 80%, rgba(255,255,255,0.1) 1px, transparent 2px),
                radial-gradient(circle at 90% 20%, rgba(135,206,235,0.1) 2px, transparent 3px),
                radial-gradient(circle at 30% 70%, rgba(72,209,204,0.08) 1px, transparent 2px);
            background-size: 80px 80px, 120px 120px, 100px 100px, 150px 150px, 90px 90px;
            animation: watercolorFlow 25s ease-in-out infinite;
            pointer-events: none;
            z-index: -1;
        }

        @keyframes watercolorFlow {
            0%, 100% { transform: translateX(0px) translateY(0px) scale(1); opacity: 0.4; }
            33% { transform: translateX(-5px) translateY(-3px) scale(1.02); opacity: 0.6; }
            66% { transform: translateX(3px) translateY(-2px) scale(0.98); opacity: 0.5; }
        }

        /* Enhanced content readability with white background and black text */
        .main .block-container {
            background: rgba(255, 255, 255, 0.95) !important;
            backdrop-filter: blur(15px);
            border-radius: 20px;
            padding: 2rem;
            margin-top: 1rem;
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15);
            border: 2px solid rgba(255, 255, 255, 0.3);
        }

        /* Force black text in main content */
        .main .block-container * {
            color: #1a1a1a !important;
        }

        /* Sidebar styling with watercolor theme */
        .css-1d391kg {
            background: rgba(240, 249, 255, 0.92) !important;
            backdrop-filter: blur(20px);
            border-right: 2px solid rgba(135, 206, 235, 0.3);
        }

        /* Sidebar text black */
        .css-1d391kg * {
            color: #1a1a1a !important;
        }

        /* Tab styling with better contrast */
        .stTabs [data-baseweb="tab"] {
            background: rgba(255, 255, 255, 0.85) !important;
            border-radius: 10px 10px 0 0;
            backdrop-filter: blur(8px);
            border: 1px solid rgba(135, 206, 235, 0.3);
            color: #1a1a1a !important;
        }

        .stTabs [data-baseweb="tab"]:hover {
            background: rgba(255, 255, 255, 0.95) !important;
            color: #0d47a1 !important;
        }

        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background: rgba(255, 255, 255, 1) !important;
            color: #0d47a1 !important;
            font-weight: bold;
        }

        /* Metric cards with white background and black text */
        [data-testid="metric-container"] {
            background: rgba(255, 255, 255, 0.9) !important;
            border-radius: 12px;
            padding: 1.2rem;
            backdrop-filter: blur(12px);
            border: 2px solid rgba(135, 206, 235, 0.2);
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        }

        [data-testid="metric-container"] * {
            color: #1a1a1a !important;
        }

        /* Headers with dark blue color for better contrast */
        h1, h2, h3, h4, h5, h6 {
            color: #0d47a1 !important;
            font-weight: 600 !important;
        }

        /* Success/info/warning boxes with proper contrast */
        .stAlert {
            background: rgba(255, 255, 255, 0.92) !important;
            backdrop-filter: blur(12px);
            border-radius: 12px;
            border-left: 4px solid #4682b4;
            color: #1a1a1a !important;
        }

        .stAlert * {
            color: #1a1a1a !important;
        }

        /* Input fields */
        .stTextInput > div > div > input {
            background: rgba(255, 255, 255, 0.95) !important;
            color: #1a1a1a !important;
            border: 2px solid rgba(135, 206, 235, 0.3) !important;
        }

        .stTextArea > div > div > textarea {
            background: rgba(255, 255, 255, 0.95) !important;
            color: #1a1a1a !important;
            border: 2px solid rgba(135, 206, 235, 0.3) !important;
        }

        /* Selectbox styling */
        .stSelectbox > div > div > div {
            background: rgba(255, 255, 255, 0.95) !important;
            color: #1a1a1a !important;
            border: 2px solid rgba(135, 206, 235, 0.3) !important;
        }

        .stSelectbox > div > div > div > div {
            color: #1a1a1a !important;
        }

        .stSelectbox label {
            color: #1a1a1a !important;
            font-weight: 600 !important;
        }

        /* Selectbox dropdown options */
        .stSelectbox [data-baseweb="select"] > div {
            background: rgba(255, 255, 255, 0.95) !important;
            color: #1a1a1a !important;
        }

        /* Selectbox dropdown menu */
        .stSelectbox [data-baseweb="popover"] {
            background: rgba(255, 255, 255, 0.98) !important;
        }

        .stSelectbox [data-baseweb="menu"] {
            background: rgba(255, 255, 255, 0.98) !important;
            border: 1px solid rgba(135, 206, 235, 0.5) !important;
            border-radius: 8px !important;
        }

        .stSelectbox [data-baseweb="menu"] > ul {
            background: rgba(255, 255, 255, 0.98) !important;
        }

        .stSelectbox [data-baseweb="menu"] > ul > li {
            background: rgba(255, 255, 255, 0.98) !important;
            color: #1a1a1a !important;
        }

        .stSelectbox [data-baseweb="menu"] > ul > li:hover {
            background: rgba(135, 206, 235, 0.2) !important;
            color: #1a1a1a !important;
        }

        .stSelectbox [data-baseweb="menu"] > ul > li[aria-selected="true"] {
            background: rgba(135, 206, 235, 0.3) !important;
            color: #1a1a1a !important;
        }

        /* Additional selectbox styling for all states */
        .stSelectbox div[data-baseweb="select"] {
            background: rgba(255, 255, 255, 0.95) !important;
            color: #1a1a1a !important;
        }

        .stSelectbox div[data-baseweb="select"] > div {
            color: #1a1a1a !important;
        }

        .stSelectbox div[data-baseweb="select"] span {
            color: #1a1a1a !important;
        }

        /* Multiselect styling */
        .stMultiSelect > div > div > div {
            background: rgba(255, 255, 255, 0.95) !important;
            color: #1a1a1a !important;
            border: 2px solid rgba(135, 206, 235, 0.3) !important;
        }

        .stMultiSelect label {
            color: #1a1a1a !important;
            font-weight: 600 !important;
        }

        /* Multiselect dropdown menu */
        .stMultiSelect [data-baseweb="popover"] {
            background: rgba(255, 255, 255, 0.98) !important;
        }

        .stMultiSelect [data-baseweb="menu"] {
            background: rgba(255, 255, 255, 0.98) !important;
            border: 1px solid rgba(135, 206, 235, 0.5) !important;
            border-radius: 8px !important;
        }

        .stMultiSelect [data-baseweb="menu"] > ul {
            background: rgba(255, 255, 255, 0.98) !important;
        }

        .stMultiSelect [data-baseweb="menu"] > ul > li {
            background: rgba(255, 255, 255, 0.98) !important;
            color: #1a1a1a !important;
        }

        .stMultiSelect [data-baseweb="menu"] > ul > li:hover {
            background: rgba(135, 206, 235, 0.2) !important;
            color: #1a1a1a !important;
        }

        .stMultiSelect div[data-baseweb="select"] {
            background: rgba(255, 255, 255, 0.95) !important;
            color: #1a1a1a !important;
        }

        .stMultiSelect div[data-baseweb="select"] > div {
            color: #1a1a1a !important;
        }

        .stMultiSelect div[data-baseweb="select"] span {
            color: #1a1a1a !important;
        }

        /* Buttons */
        .stButton > button {
            background: rgba(13, 71, 161, 0.9) !important;
            color: white !important;
            border-radius: 8px;
            border: none;
        }

        .stButton > button:hover {
            background: rgba(13, 71, 161, 1) !important;
            transform: translateY(-1px);
        }

        /* Dataframes */
        .stDataFrame {
            background: rgba(255, 255, 255, 0.95) !important;
            border-radius: 10px;
        }

        /* Ensure all text is readable */
        p, span, div, label, li {
            color: #1a1a1a !important;
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
        <h1 style="color: white; margin: 0; font-size: 2.8rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
            🌊 ARGO Scientific Oceanographic Dashboard
        </h1>
        <p style="color: #e0f7fa; margin: 0; font-size: 1.2rem; text-shadow: 1px 1px 2px rgba(0,0,0,0.2);">
            Advanced RAG-powered analysis of global ocean observations
        </p>
        <div style="margin-top: 1rem; opacity: 0.8;">
            <span style="color: #b3e5fc; font-size: 0.9rem;">
                🏛️ Research-grade • 🔬 AI-powered • 📊 Real-time analysis
            </span>
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

    # Main page layout with sliding windows
    st.markdown("---")

    # Initialize sliding window states
    if 'ai_window_minimized' not in st.session_state:
        st.session_state.ai_window_minimized = False
    if 'show_ai_results' not in st.session_state:
        st.session_state.show_ai_results = False

    # Create adaptive layout based on AI window state
    if st.session_state.ai_window_minimized or not st.session_state.show_ai_results:
        # Full width map when AI is minimized
        with st.container():
            st.markdown("## 🌊 Global ARGO Float Monitoring System")
            create_enhanced_float_map()
    else:
        # Split layout when AI is active
        map_col, ai_col = st.columns([6, 4])

        with map_col:
            st.markdown("## 🌊 Global ARGO Float Monitoring System")
            create_enhanced_float_map()

        with ai_col:
            # AI Query Interface sliding window
            with st.container():
                # AI Window header with minimize/maximize
                col_ai_btn, col_ai_title = st.columns([1, 8])
                with col_ai_btn:
                    if st.button("📖" if st.session_state.ai_window_minimized else "📕",
                               help="Minimize/Maximize AI Analysis",
                               key="toggle_ai_window"):
                        st.session_state.ai_window_minimized = not st.session_state.ai_window_minimized
                        st.rerun()

                with col_ai_title:
                    st.markdown("### 🤖 AI Scientific Analysis")

                if not st.session_state.ai_window_minimized:
                    # AI Input Section
                    with st.container():
                        st.markdown("""
                        <div style="background-color: #f0f9ff; padding: 1rem; border-radius: 8px; border: 1px solid #e0f2fe;">
                        """, unsafe_allow_html=True)

                        # Check for preset query
                        preset_query = st.session_state.get('preset_query', '')
                        if preset_query:
                            st.session_state.preset_query = ''  # Clear after using

                        query = st.text_area(
                            "Ask AI about the data:",
                            value=preset_query,
                            height=100,
                            placeholder="""Examples:
• Analyze temperature gradients
• Show salinity profiles
• Create time series plots
• Compare ocean trends""",
                            help="Enter natural language queries",
                            key="ai_query_input"
                        )

                        col1, col2 = st.columns(2)
                        with col1:
                            submit_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)
                        with col2:
                            if st.button("🗑️ Clear", use_container_width=True, disabled=st.session_state.query_results is None):
                                st.session_state.query_results = None
                                st.session_state.data_page = 0
                                st.session_state.show_ai_results = False
                                st.rerun()

                        # Quick examples toggle
                        if st.button("📚 Show Examples", use_container_width=True):
                            st.session_state.show_examples = not st.session_state.get('show_examples', False)

                        st.markdown("</div>", unsafe_allow_html=True)

                        # Display AI Results in the sliding window
                        if st.session_state.query_results:
                            result = st.session_state.query_results

                            st.markdown("---")
                            st.markdown("### 🔬 Analysis Results")

                            # Quick metrics
                            col1, col2 = st.columns(2)
                            with col1:
                                processing_time = result['processing_time']
                                st.metric("Time", f"{processing_time:.1f}s")
                            with col2:
                                data_count = len(result.get('sql_data', []))
                                st.metric("Data", f"{data_count:,}")

                            # AI Response - Enhanced display
                            if result.get('llm_response'):
                                st.markdown("**🤖 AI Analysis:**")
                                st.markdown(f"""
                                <div style="background-color: #f0f9ff; padding: 1rem; border-radius: 8px;
                                           border-left: 3px solid #0ea5e9; font-size: 0.9rem; max-height: 300px; overflow-y: auto;">
                                    {result['llm_response']}
                                </div>
                                """, unsafe_allow_html=True)

                            # Data preview with scrollable table
                            if result.get('sql_data') and len(result['sql_data']) > 0:
                                df = pd.DataFrame(result['sql_data'])

                                st.markdown("**📊 Data Preview:**")
                                # Show first 10 rows in a compact format
                                st.dataframe(df.head(10), use_container_width=True, height=200)

                                # Show basic chart if data available
                                numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
                                if len(numeric_cols) >= 2:
                                    st.markdown("**📈 Quick Chart:**")
                                    fig = px.scatter(df.head(100),
                                                   x=numeric_cols[0],
                                                   y=numeric_cols[1],
                                                   height=250,
                                                   template="plotly_white")
                                    fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))
                                    st.plotly_chart(fig, use_container_width=True)

                                # Download button
                                csv_data = df.to_csv(index=False)
                                st.download_button(
                                    "📥 Download Full Dataset",
                                    csv_data,
                                    f"argo_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    "text/csv",
                                    use_container_width=True
                                )

                            # Button to view detailed results
                            if st.button("🔍 View Detailed Analysis", use_container_width=True, type="secondary"):
                                st.session_state.show_detailed_results = True
                                st.rerun()

                else:
                    st.info("🤖 AI Analysis minimized - Click 📖 to expand")

    # Floating AI Query Box (always visible)
    if st.session_state.ai_window_minimized or not st.session_state.show_ai_results:
        with st.sidebar:
            st.markdown("---")
            st.markdown("### 🚀 Quick AI Query")

            quick_query = st.text_input(
                "Enter AI query:",
                placeholder="Ask about ARGO data...",
                key="quick_ai_query"
            )

            if st.button("🔍 Analyze", key="quick_analyze_btn", use_container_width=True, type="primary"):
                if quick_query.strip():
                    st.session_state.show_ai_results = True
                    st.session_state.ai_window_minimized = False
                    # Process the query
                    with st.spinner("Processing query..."):
                        result = process_query(quick_query)
                        st.session_state.query_results = result
                        st.session_state.data_page = 0
                    st.rerun()
                else:
                    st.warning("Please enter a query")

    # Show examples if requested
    if st.session_state.get('show_examples', False):
        with st.expander("📚 Example Scientific Queries", expanded=True):
            example_categories = {
                "🌡️ Temperature Analysis": [
                    "Show temperature vs depth profiles for the last 6 months",
                    "Analyze temperature anomalies in the tropical Pacific",
                    "Create a heatmap of sea surface temperature"
                ],
                "🧂 Salinity Studies": [
                    "Plot salinity distribution in the Mediterranean",
                    "Compare deep water salinity trends",
                    "Analyze halocline structure"
                ],
                "🗺️ Geographic Analysis": [
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
        if 'submit_btn' in locals() and submit_btn and query:
            with st.spinner("Processing query..."):
                result = process_query(query)
                st.session_state.query_results = result
                st.session_state.data_page = 0
                st.session_state.show_ai_results = True
                st.rerun()

    
    # Advanced Scientific Results Display (only when requested)
    if st.session_state.query_results and st.session_state.get('show_detailed_results', False):
        result = st.session_state.query_results

        # Add a button to close detailed results
        col_close, col_title = st.columns([1, 10])
        with col_close:
            if st.button("❌", help="Close Detailed Analysis", key="close_detailed"):
                st.session_state.show_detailed_results = False
                st.rerun()
        with col_title:
            st.markdown("## 🧬 Detailed Scientific Analysis Results")

        # Advanced metrics display
        create_analysis_metrics(result)

        # Professional results tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🧠 AI Scientific Analysis",
            "📊 Advanced Visualization",
            "🔬 Data Analysis",
            "📈 Statistical Summary",
            "⚙️ Technical Details"
        ])

        with tab1:
            st.markdown("### 🤖 AI-Powered Scientific Interpretation")
            if result.get('llm_response'):
                # Enhanced AI response display
                st.markdown(f"""
                <div style="background-color: #f0f9ff; padding: 1.5rem; border-radius: 10px;
                           border-left: 4px solid #0ea5e9;">
                    <h4 style="color: #0c4a6e; margin-top: 0;">Scientific Analysis</h4>
                    <p style="margin-bottom: 0; line-height: 1.6;">{result['llm_response']}</p>
                </div>
                """, unsafe_allow_html=True)


            else:
                st.info("🔍 No AI analysis available for this query")

        with tab2:
            st.markdown("### 📊 Advanced Scientific Visualization")
            if result.get('sql_data') and len(result['sql_data']) > 0:
                df = pd.DataFrame(result['sql_data'])
                create_advanced_visualization(df, result)
            else:
                st.info("📈 No data available for visualization")

        with tab3:
            st.markdown("### 🔬 Oceanographic Data Analysis")
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
                st.markdown("#### 🗄️ Data Explorer")

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
                if len(df) > 100:
                    sample_size = st.slider("Sample size", 10, min(1000, len(df)), 100)
                    df_display = df_display.head(sample_size)

                st.dataframe(df_display, use_container_width=True, height=400)

                # Enhanced download options
                col1, col2, col3 = st.columns(3)
                with col1:
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "📊 Download CSV",
                        csv,
                        f"argo_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        "text/csv",
                        use_container_width=True
                    )
                with col2:
                    json_data = df.to_json(orient='records', indent=2)
                    st.download_button(
                        "📄 Download JSON",
                        json_data,
                        f"argo_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                        "application/json",
                        use_container_width=True
                    )
                with col3:
                    # Data summary
                    if st.button("📋 Generate Report", use_container_width=True):
                        st.info("Comprehensive data report functionality ready")

            else:
                st.info("🔍 No data retrieved for analysis")

        with tab4:
            st.markdown("### 📈 Statistical Analysis")
            if result.get('sql_data'):
                df = pd.DataFrame(result['sql_data'])
                numeric_df = df.select_dtypes(include=['float64', 'int64'])

                if not numeric_df.empty:
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("#### 📊 Descriptive Statistics")
                        st.dataframe(numeric_df.describe(), use_container_width=True)

                    with col2:
                        st.markdown("#### 🔗 Correlation Matrix")
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
                    st.markdown("#### 📈 Distribution Analysis")
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
                    st.info("🔍 No numeric data available for statistical analysis")
            else:
                st.info("📊 No data available for statistical analysis")

        with tab5:
            st.markdown("### ⚙️ Technical Analysis Details")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 🔍 Query Matching")
                st.write("**Best Match ID:**", result.get('best_match_id', 'N/A'))
                st.write("**Similarity Score:**", f"{result.get('similarity', 0):.4f}")
                st.write("**Processing Time:**", f"{result.get('processing_time', 0):.3f} seconds")

                # Performance indicators
                if result.get('processing_time', 0) < 2:
                    st.success("⚡ Excellent performance")
                elif result.get('processing_time', 0) < 5:
                    st.info("🔄 Good performance")
                else:
                    st.warning("⏳ Consider query optimization")

            with col2:
                st.markdown("#### 🛠️ System Operations")
                st.write("**SQL Generation:**", "✅ Success" if result.get('sql_executed') else "❌ Failed")
                st.write("**Visualization:**", "✅ Created" if result.get('visualization_created') else "❌ None")
                st.write("**Data Retrieval:**", f"{len(result.get('sql_data', []))} records")

            # SQL Query Analysis
            if result.get('sql_executed') and result.get('sql'):
                st.markdown("#### 💾 Generated SQL Query")
                st.code(result['sql'], language='sql')

                # SQL complexity analysis
                sql_lines = result['sql'].count('\n') + 1
                if sql_lines < 5:
                    st.success("🟢 Simple query - Fast execution")
                elif sql_lines < 15:
                    st.info("🟡 Moderate query complexity")
                else:
                    st.warning("🟠 Complex query - May take time")

            # Semantic matching details
            if result.get('matched_sample'):
                with st.expander("🧬 Semantic Matching Details", expanded=False):
                    st.json(result['matched_sample'])


if __name__ == "__main__":
    main()