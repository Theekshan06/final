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
        if st.button("Show Float Locations", use_container_width=True, type="primary"):
            st.session_state.show_float_map = True
            st.rerun()

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

    st.markdown("---")

    # Advanced Query Interface
    st.markdown("### 🔬 Scientific Query Interface")

    # Check for preset query
    preset_query = st.session_state.get('preset_query', '')
    if preset_query:
        st.session_state.preset_query = ''  # Clear after using

    col1, col2, col3 = st.columns([6, 1, 1])
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
        submit_btn = st.button("🧪 Analyze", type="primary", use_container_width=True)
        if st.button("📊 Examples", use_container_width=True):
            st.session_state.show_examples = not st.session_state.get('show_examples', False)
    with col3:
        if st.button("🗑️ Clear Results", use_container_width=True, disabled=st.session_state.query_results is None):
            st.session_state.query_results = None
            st.session_state.data_page = 0
            st.rerun()

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
        st.markdown("## 🧬 Scientific Analysis Results")

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

    # Display ARGO Float Map if requested
    if st.session_state.get('show_float_map', False):
        st.markdown("---")
        st.markdown("## 🗺️ ARGO Float Global Distribution Map")

        col1, col2 = st.columns([6, 1])
        with col2:
            if st.button("❌ Close Map"):
                st.session_state.show_float_map = False
                st.rerun()

        create_float_location_map()

if __name__ == "__main__":
    main()