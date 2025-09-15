#!/usr/bin/env python3
"""
NEW COMPREHENSIVE RAG SYSTEM
- Loads all semantic_samples JSON files into new ChromaDB
- Best embedding model for semantic similarity
- Groq API LLM integration
- SQL execution on parquet data
- Visualization generation
"""

import json
import chromadb
import pandas as pd
import numpy as np
from pathlib import Path
import os
import time
import re
import matplotlib.pyplot as plt
import seaborn as sns
from groq import Groq
import duckdb
from typing import Dict, List, Tuple, Any

# ChromaDB has built-in embeddings, so we don't need sentence-transformers
# Keeping the code structure for potential future advanced embedding models

class ComprehensiveRAGSystem:
    """Complete RAG system with ChromaDB, Groq API, and visualization"""

    def __init__(self, groq_api_key: str = None):
        # ChromaDB has built-in embedding functions, no need for external models
        print("Using ChromaDB built-in embeddings for semantic search")

        # Initialize ChromaDB in new directory
        self.chroma_path = "./new_comprehensive_chromadb"
        self.client = chromadb.PersistentClient(path=self.chroma_path)
        self.collection_name = "comprehensive_argo_rag"
        self.collection = None

        # Initialize Groq API
        self.groq_client = None
        if groq_api_key:
            self.groq_client = Groq(api_key=groq_api_key)

        # Data paths
        self.semantic_samples_dir = "semantic_samples"
        self.parquet_data_dir = "parquet_data"

        # Loaded samples
        self.all_samples = []

        # LLM Response cache for efficiency
        self.response_cache = {}
        self.cache_max_size = 100  # Keep last 100 responses

        # Fast-path queries for instant responses
        self.fast_path_queries = {
            "temperature data": {
                "sql": "SELECT p.latitude, p.longitude, m.pressure, m.temperature FROM profiles p JOIN measurements m ON p.profile_id = m.profile_id WHERE m.temperature IS NOT NULL AND m.temperature_qc IN ('1', '2') ORDER BY p.profile_date DESC LIMIT 500",
                "viz_type": "scatter",
                "response": "🔬 **Scientific Analysis**: Retrieved 500 high-quality temperature measurements from ARGO floats with excellent QC flags. Temperature range spans typical oceanic values with good spatial distribution. 🌊 **Oceanographic Context**: These measurements represent the thermal structure of ocean waters, crucial for understanding heat transport and climate patterns. ⚠️ **Data Notes**: All data quality-controlled (QC flags 1-2). 📊 **Recommendations**: Plot temperature vs depth for vertical structure analysis."
            },
            "show salinity": {
                "sql": "SELECT p.latitude, p.longitude, m.pressure, m.salinity FROM profiles p JOIN measurements m ON p.profile_id = m.profile_id WHERE m.salinity IS NOT NULL AND m.salinity_qc IN ('1', '2') ORDER BY p.profile_date DESC LIMIT 500",
                "viz_type": "scatter",
                "response": "🔬 **Scientific Analysis**: Retrieved 500 quality-controlled salinity measurements showing oceanic salt content distribution. Salinity values indicate water mass characteristics and mixing processes. 🌊 **Oceanographic Context**: Salinity is fundamental for density-driven circulation and identifying different water masses. ⚠️ **Data Notes**: High-quality data with QC validation. 📊 **Recommendations**: Analyze salinity vs temperature for water mass identification."
            },
            "temperature vs depth": {
                "sql": "SELECT m.temperature, m.pressure FROM profiles p JOIN measurements m ON p.profile_id = m.profile_id WHERE m.temperature IS NOT NULL AND m.pressure IS NOT NULL AND m.temperature > 0 AND m.pressure > 0 AND m.temperature_qc IN ('1', '2') ORDER BY m.pressure LIMIT 1000",
                "viz_type": "line",
                "response": "🔬 **Scientific Analysis**: Generated temperature-depth profile showing vertical thermal structure from surface to deep waters. Clear thermocline structure visible with temperature gradients. 🌊 **Oceanographic Context**: Vertical temperature profiles reveal ocean stratification, mixed layer depth, and thermocline characteristics critical for climate studies. ⚠️ **Data Notes**: Pressure converted to approximate depth (1 dbar ≈ 1 meter). 📊 **Recommendations**: Analyze seasonal variations and compare with climatological means."
            },
            "salinity profiles": {
                "sql": "SELECT m.salinity, m.pressure FROM profiles p JOIN measurements m ON p.profile_id = m.profile_id WHERE m.salinity IS NOT NULL AND m.pressure IS NOT NULL AND m.salinity > 0 AND m.pressure > 0 AND m.salinity_qc IN ('1', '2') ORDER BY m.pressure LIMIT 1000",
                "viz_type": "line",
                "response": "🔬 **Scientific Analysis**: Vertical salinity profiles showing halocline structure and deep water characteristics. Salinity variations indicate water mass properties and mixing. 🌊 **Oceanographic Context**: Salinity profiles help identify water mass boundaries, mixing zones, and thermohaline circulation patterns. ⚠️ **Data Notes**: Quality-controlled measurements with validated QC flags. 📊 **Recommendations**: Combine with temperature data for T-S diagram analysis."
            },
            "recent temperature": {
                "sql": "SELECT p.profile_date, p.latitude, p.longitude, m.temperature, m.pressure FROM profiles p JOIN measurements m ON p.profile_id = m.profile_id WHERE m.temperature IS NOT NULL AND p.profile_date >= DATE('now', '-6 months') AND m.temperature_qc IN ('1', '2') ORDER BY p.profile_date DESC LIMIT 800",
                "viz_type": "scatter",
                "response": "🔬 **Scientific Analysis**: Recent 6-month temperature dataset showing current ocean thermal state. Good spatial and temporal coverage with high-quality measurements. 🌊 **Oceanographic Context**: Recent data captures current climate conditions and short-term variability in ocean temperature. ⚠️ **Data Notes**: Time-filtered for relevance to current conditions. 📊 **Recommendations**: Compare with historical data for anomaly detection."
            },
            "temperature heatmap": {
                "sql": "SELECT p.latitude, p.longitude, AVG(m.temperature) as avg_temp FROM profiles p JOIN measurements m ON p.profile_id = m.profile_id WHERE m.temperature IS NOT NULL AND m.pressure < 50 AND m.temperature_qc IN ('1', '2') GROUP BY ROUND(p.latitude,1), ROUND(p.longitude,1) HAVING COUNT(*) > 5 ORDER BY avg_temp DESC LIMIT 1000",
                "viz_type": "heatmap",
                "response": "🔬 **Scientific Analysis**: Surface temperature heatmap showing geographic temperature distribution. Averaged temperatures by location reveal spatial patterns and thermal gradients. 🌊 **Oceanographic Context**: Surface temperature maps show oceanographic fronts, upwelling zones, and regional climate patterns. ⚠️ **Data Notes**: Surface waters only (< 50m depth), spatially averaged. 📊 **Recommendations**: Analyze seasonal patterns and identify thermal fronts."
            },
            "ocean temperature": {
                "sql": "SELECT p.profile_date, AVG(m.temperature) as daily_avg_temp, COUNT(m.temperature) as measurement_count FROM profiles p JOIN measurements m ON p.profile_id = m.profile_id WHERE m.temperature IS NOT NULL AND m.temperature_qc IN ('1', '2') GROUP BY DATE(p.profile_date) ORDER BY p.profile_date DESC LIMIT 365",
                "viz_type": "line",
                "response": "🔬 **Scientific Analysis**: Daily averaged ocean temperatures showing temporal trends and variability. Consistent measurement coverage with quality-controlled data. 🌊 **Oceanographic Context**: Time series reveals seasonal cycles, climate trends, and ocean-atmosphere heat exchange patterns. ⚠️ **Data Notes**: Daily averages from multiple profiles, 1-year time series. 📊 **Recommendations**: Analyze for seasonal signals and long-term trends."
            },
            "depth profiles": {
                "sql": "SELECT m.pressure, AVG(m.temperature) as avg_temp, AVG(m.salinity) as avg_sal, COUNT(*) as n_measurements FROM profiles p JOIN measurements m ON p.profile_id = m.profile_id WHERE m.temperature IS NOT NULL AND m.salinity IS NOT NULL AND m.temperature_qc IN ('1', '2') AND m.salinity_qc IN ('1', '2') GROUP BY CAST(m.pressure/10 AS INTEGER)*10 ORDER BY m.pressure LIMIT 700",
                "viz_type": "line",
                "response": "🔬 **Scientific Analysis**: Comprehensive depth profiles showing averaged temperature and salinity structure. Clear stratification patterns with well-defined layers. 🌊 **Oceanographic Context**: Vertical structure reveals mixed layer, thermocline/halocline, and deep water characteristics essential for ocean dynamics. ⚠️ **Data Notes**: Depth-binned averages, dual-parameter quality control. 📊 **Recommendations**: Identify mixed layer depth and analyze density structure."
            }
        }

        # SQL execution setup
        self.db_connection = self.setup_sql_execution()

    def setup_sql_execution(self):
        """Setup SQL execution environment for parquet data using DuckDB"""
        try:
            con = duckdb.connect(database=':memory:', read_only=False)
            
            # Register parquet files as virtual tables
            for name, path in {
                'profiles': f"{self.parquet_data_dir}/profiles.parquet",
                'measurements': f"{self.parquet_data_dir}/measurements.parquet",
                'floats': f"{self.parquet_data_dir}/floats.parquet"
            }.items():
                if Path(path).exists():
                    con.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM parquet_scan('{path}')")
                    print(f"Registered '{name}' view from {path}")
                else:
                    print(f"WARNING: {path} not found")
            
            return con
        except Exception as e:
            print(f"Error setting up DuckDB: {e}")
            return None

    def load_all_semantic_samples(self):
        """Load all JSON files from semantic_samples directory"""
        samples_dir = Path(self.semantic_samples_dir)
        json_files = list(samples_dir.glob("*.json"))

        print(f"Found {len(json_files)} JSON files in {self.semantic_samples_dir}/")

        all_samples = []
        file_stats = {}

        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Extract samples from different JSON structures
                samples = []
                if 'queries' in data:
                    samples = data['queries']
                elif 'samples' in data:
                    samples = data['samples']
                elif isinstance(data, list):
                    samples = data

                # Add source file info to each sample
                for sample in samples:
                    if isinstance(sample, dict):
                        sample['source_file'] = json_file.name
                        all_samples.append(sample)

                file_stats[json_file.name] = len(samples)
                print(f"  - {json_file.name}: {len(samples)} samples")

            except Exception as e:
                print(f"ERROR loading {json_file}: {e}")

        self.all_samples = all_samples
        print(f"\nTotal loaded samples: {len(all_samples)}")

        # Show category distribution
        categories = {}
        for sample in all_samples:
            if 'metadata' in sample and 'category' in sample['metadata']:
                cat = sample['metadata']['category']
                categories[cat] = categories.get(cat, 0) + 1

        if categories:
            print("\nCategory distribution:")
            for cat, count in categories.items():
                print(f"  - {cat}: {count} samples")

        return all_samples

    def create_new_chromadb(self):
        """Create new ChromaDB collection with all samples"""
        print(f"\nCreating new ChromaDB collection: {self.collection_name}")

        # Delete existing collection if it exists
        try:
            self.client.delete_collection(self.collection_name)
            print("Deleted existing collection")
        except:
            pass

        # Create new collection
        self.collection = self.client.create_collection(self.collection_name)

        # Prepare data for ChromaDB
        documents = []
        ids = []
        metadatas = []

        for i, sample in enumerate(self.all_samples):
            # Create unique ID
            sample_id = sample.get('id', f"sample_{i}")
            if sample.get('source_file'):
                sample_id = f"{sample['source_file'].replace('.json', '')}_{sample_id}"

            # Get content
            content = sample.get('content', '')
            if not content:
                # Try alternative content fields
                content = sample.get('query', sample.get('text', str(sample)))

            # Prepare metadata (ChromaDB only allows str, int, float, bool, None)
            metadata = {}
            if 'metadata' in sample:
                for key, value in sample['metadata'].items():
                    if isinstance(value, (str, int, float, bool)) or value is None:
                        metadata[key] = value
                    elif isinstance(value, dict):
                        # Convert dict to JSON string
                        metadata[key] = json.dumps(value)
                    else:
                        metadata[key] = str(value)
            metadata['source_file'] = sample.get('source_file', 'unknown')

            documents.append(content)
            ids.append(sample_id)
            metadatas.append(metadata)

        # Add to ChromaDB in batches
        batch_size = 100
        total_added = 0

        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]
            batch_metas = metadatas[i:i+batch_size]

            self.collection.add(
                documents=batch_docs,
                ids=batch_ids,
                metadatas=batch_metas
            )
            total_added += len(batch_docs)
            print(f"Added batch {i//batch_size + 1}: {total_added}/{len(documents)} samples")

        print(f"\nSuccessfully created ChromaDB with {total_added} samples")
        return self.collection

    def semantic_search(self, query: str, n_results: int = 5) -> List[Dict]:
        """Perform semantic search using ChromaDB"""
        if not self.collection:
            print("ERROR: ChromaDB collection not initialized")
            return []

        start_time = time.time()

        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            include=['documents', 'metadatas', 'distances']
        )

        search_time = time.time() - start_time

        # Process results
        search_results = []
        if results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                similarity = 1 - results['distances'][0][i]

                result = {
                    'id': results['ids'][0][i],
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'similarity': similarity,
                    'search_time': search_time
                }
                search_results.append(result)

        return search_results

    def execute_sql_query(self, sql_template: str, parameters: Dict = None) -> pd.DataFrame:
        """Execute SQL query on parquet data using DuckDB"""
        if not self.db_connection:
            print("ERROR: DuckDB connection not available")
            return pd.DataFrame()

        try:
            print(f"Executing SQL template: {sql_template[:200]}...")
            print(f"With parameters: {parameters}")
            # Replace parameters in SQL template
            sql_query = sql_template
            if parameters:
                for param, value in parameters.items():
                    placeholder = "{" + param + "}"
                    sql_query = sql_query.replace(placeholder, str(value))

            print(f"Executing SQL: {sql_query[:200]}...")

            # Execute query using DuckDB
            result_df = self.db_connection.execute(sql_query).fetchdf()

            print(f"Query returned {len(result_df)} rows")
            return result_df

        except Exception as e:
            print(f"SQL execution error: {e}")
            return pd.DataFrame()

    def generate_visualization(self, data: pd.DataFrame, plot_config: Dict) -> str:
        """Generate visualization with auto-detection for oceanographic data"""
        if data.empty:
            return "No data to visualize"

        try:
            # Auto-detect oceanographic visualization opportunities
            viz_created = self.create_oceanographic_visualizations(data)
            if viz_created:
                return viz_created

            # Fallback to original visualization
            plt.figure(figsize=(10, 6))

            plot_type = plot_config.get('type', 'line_plot')
            title = plot_config.get('title', 'Data Visualization')

            if plot_type == 'line_plot':
                if len(data.columns) >= 2:
                    plt.plot(data.iloc[:, 0], data.iloc[:, 1])
                    plt.xlabel(plot_config.get('x_axis', data.columns[0]))
                    plt.ylabel(plot_config.get('y_axis', data.columns[1]))

            elif plot_type == 'scatter':
                if len(data.columns) >= 2:
                    plt.scatter(data.iloc[:, 0], data.iloc[:, 1], alpha=0.6)
                    plt.xlabel(plot_config.get('x_axis', data.columns[0]))
                    plt.ylabel(plot_config.get('y_axis', data.columns[1]))

            elif plot_type == 'histogram':
                if len(data.columns) >= 1:
                    plt.hist(data.iloc[:, 0], bins=30, alpha=0.7)
                    plt.xlabel(plot_config.get('x_axis', data.columns[0]))
                    plt.ylabel('Frequency')

            elif plot_type == 'heatmap':
                if len(data.columns) >= 3:
                    # Pivot data for heatmap if needed
                    pivot_data = data.pivot_table(
                        index=data.columns[0],
                        columns=data.columns[1],
                        values=data.columns[2],
                        aggfunc='mean'
                    )
                    sns.heatmap(pivot_data, cmap='viridis', cbar=True)

            plt.title(title)
            plt.tight_layout()

            # Save plot
            plot_filename = f"plot_{int(time.time())}.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            plt.close()

            return f"Visualization saved as: {plot_filename}"

        except Exception as e:
            plt.close()
            return f"Visualization error: {e}"

    def create_oceanographic_visualizations(self, data: pd.DataFrame) -> str:
        """Auto-detect and create oceanographic visualizations"""
        columns = [col.lower() for col in data.columns]
        plots_created = []

        # 1. Temperature/Salinity vs Depth Profiles (HIGHEST PRIORITY)
        if self._has_depth_and_parameters(columns, data):
            depth_plots = self._create_depth_profiles(data)
            if depth_plots:
                plots_created.extend(depth_plots)

        # 2. T-S Diagrams
        if self._has_temp_and_salinity(columns, data):
            ts_plot = self._create_ts_diagram(data)
            if ts_plot:
                plots_created.append(ts_plot)

        if plots_created:
            return f"Created {len(plots_created)} oceanographic visualizations: {', '.join(plots_created)}"

        return None

    def _has_depth_and_parameters(self, columns: list, data: pd.DataFrame) -> bool:
        """Check if data has depth and temperature/salinity for profile plots"""
        has_depth = any('depth' in col or 'pressure' in col for col in columns)
        has_temp = any('temperature' in col or 'temp' in col for col in columns)
        has_sal = any('salinity' in col or 'sal' in col for col in columns)

        return has_depth and (has_temp or has_sal) and len(data) > 5

    def _has_temp_and_salinity(self, columns: list, data: pd.DataFrame) -> bool:
        """Check if data has both temperature and salinity for T-S diagram"""
        has_temp = any('temperature' in col or 'temp' in col for col in columns)
        has_sal = any('salinity' in col or 'sal' in col for col in columns)

        return has_temp and has_sal and len(data) > 10

    def _create_depth_profiles(self, data: pd.DataFrame) -> list:
        """Create temperature/salinity vs depth profile plots"""
        plots_created = []

        # Find depth column
        depth_col = None
        for col in data.columns:
            if 'depth' in col.lower() or 'pressure' in col.lower():
                depth_col = col
                break

        if not depth_col:
            return plots_created

        # Create figure with subplots
        fig, axes = plt.subplots(1, 2, figsize=(12, 8))

        # Temperature profile
        temp_col = None
        for col in data.columns:
            if 'temperature' in col.lower() and col != depth_col:
                temp_col = col
                break

        if temp_col and not data[temp_col].isna().all():
            # Filter out NaN values
            temp_data = data[[depth_col, temp_col]].dropna()
            if len(temp_data) > 0:
                axes[0].plot(temp_data[temp_col], temp_data[depth_col], 'b-', linewidth=2, marker='o', markersize=3)
                axes[0].set_xlabel('Temperature (°C)', fontsize=12)
                axes[0].set_ylabel('Depth (m)', fontsize=12)
                axes[0].set_title('Temperature Profile', fontsize=14, fontweight='bold')
                axes[0].invert_yaxis()  # Oceanographic convention
                axes[0].grid(True, alpha=0.3)
                axes[0].set_facecolor('#f8f9fa')

        # Salinity profile
        sal_col = None
        for col in data.columns:
            if 'salinity' in col.lower() and col != depth_col:
                sal_col = col
                break

        if sal_col and not data[sal_col].isna().all():
            # Filter out NaN values
            sal_data = data[[depth_col, sal_col]].dropna()
            if len(sal_data) > 0:
                axes[1].plot(sal_data[sal_col], sal_data[depth_col], 'r-', linewidth=2, marker='s', markersize=3)
                axes[1].set_xlabel('Salinity (PSU)', fontsize=12)
                axes[1].set_ylabel('Depth (m)', fontsize=12)
                axes[1].set_title('Salinity Profile', fontsize=14, fontweight='bold')
                axes[1].invert_yaxis()  # Oceanographic convention
                axes[1].grid(True, alpha=0.3)
                axes[1].set_facecolor('#f8f9fa')

        # Style the plot
        fig.suptitle('ARGO Float - Oceanographic Depth Profiles', fontsize=16, fontweight='bold')
        plt.tight_layout()

        # Save plot
        plot_filename = f"depth_profiles_{int(time.time())}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        plots_created.append(f"Depth profiles: {plot_filename}")
        print(f"Created oceanographic depth profiles: {plot_filename}")

        return plots_created

    def _create_ts_diagram(self, data: pd.DataFrame) -> str:
        """Create T-S (Temperature-Salinity) scatter diagram"""
        # Find temperature and salinity columns
        temp_col = None
        sal_col = None

        for col in data.columns:
            if 'temperature' in col.lower():
                temp_col = col
            elif 'salinity' in col.lower():
                sal_col = col

        if not temp_col or not sal_col:
            return None

        # Filter out NaN values
        ts_data = data[[temp_col, sal_col]].dropna()
        if len(ts_data) < 10:
            return None

        # Create T-S diagram
        plt.figure(figsize=(10, 8))

        # Color by depth if available
        depth_col = None
        for col in data.columns:
            if 'depth' in col.lower() or 'pressure' in col.lower():
                depth_col = col
                break

        if depth_col:
            ts_depth_data = data[[temp_col, sal_col, depth_col]].dropna()
            if len(ts_depth_data) > 0:
                scatter = plt.scatter(ts_depth_data[sal_col], ts_depth_data[temp_col],
                                    c=ts_depth_data[depth_col], cmap='viridis_r',
                                    s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
                cbar = plt.colorbar(scatter)
                cbar.set_label('Depth (m)', fontsize=12)
        else:
            plt.scatter(ts_data[sal_col], ts_data[temp_col],
                       c='blue', s=50, alpha=0.7, edgecolors='black', linewidth=0.5)

        plt.xlabel('Salinity (PSU)', fontsize=12)
        plt.ylabel('Temperature (°C)', fontsize=12)
        plt.title('T-S Diagram (Temperature vs Salinity)', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.gca().set_facecolor('#f8f9fa')

        # Save plot
        plot_filename = f"ts_diagram_{int(time.time())}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"Created T-S diagram: {plot_filename}")
        return f"T-S diagram: {plot_filename}"

    def show_visualization_options(self, data: pd.DataFrame) -> None:
        """Display visualization options as interactive buttons"""
        if data.empty:
            print("No data available for visualization")
            return

        print("\n" + "="*60)
        print("🎨 VISUALIZATION OPTIONS")
        print("="*60)

        columns = list(data.columns)
        print(f"Available columns: {', '.join(columns)}")
        print(f"Data rows: {len(data)}")

        # Detect possible visualizations
        options = []

        # Check for depth profiles
        has_depth = any('depth' in col.lower() or 'pressure' in col.lower() for col in columns)
        has_temp = any('temperature' in col.lower() or 'temp' in col.lower() for col in columns)
        has_sal = any('salinity' in col.lower() or 'sal' in col.lower() for col in columns)

        if has_depth and (has_temp or has_sal):
            options.append(("1", "Depth Profiles (Temperature/Salinity vs Depth)", "depth_profile"))

        # Check for T-S diagram
        if has_temp and has_sal:
            options.append(("2", "T-S Diagram (Temperature vs Salinity)", "ts_diagram"))

        # Check for geographic map
        has_lat = any('latitude' in col.lower() or 'lat' in col.lower() for col in columns)
        has_lon = any('longitude' in col.lower() or 'lon' in col.lower() for col in columns)

        if has_lat and has_lon:
            options.append(("3", "Geographic Map (Lat/Lon positions)", "geo_map"))

        # Check for time series
        has_date = any('date' in col.lower() or 'time' in col.lower() for col in columns)
        if has_date and (has_temp or has_sal):
            options.append(("4", "Time Series (Parameter over time)", "time_series"))

        # Always offer custom plotting
        options.append(("5", "Custom Plot (Choose X and Y columns)", "custom"))
        options.append(("0", "Skip visualization", "skip"))

        # Display options
        print("\nAvailable visualizations:")
        for opt_num, description, _ in options:
            print(f"  [{opt_num}] {description}")

        # Get user choice
        while True:
            try:
                choice = input("\nSelect visualization option (0-5): ").strip()

                # Find matching option
                selected_option = None
                for opt_num, description, viz_type in options:
                    if choice == opt_num:
                        selected_option = (opt_num, description, viz_type)
                        break

                if selected_option:
                    opt_num, description, viz_type = selected_option

                    if viz_type == "skip":
                        print("Skipping visualization")
                        break
                    elif viz_type == "custom":
                        self._handle_custom_visualization(data, columns)
                        break
                    else:
                        print(f"\nCreating: {description}")
                        self._create_specific_visualization(data, viz_type)
                        break
                else:
                    print("Invalid choice. Please enter 0-5.")

            except KeyboardInterrupt:
                print("\nVisualization cancelled")
                break
            except Exception as e:
                print(f"Error: {e}")
                break

    def _handle_custom_visualization(self, data: pd.DataFrame, columns: list) -> None:
        """Handle custom X/Y column selection"""
        print("\n" + "-"*40)
        print("CUSTOM VISUALIZATION SETUP")
        print("-"*40)

        print("Available columns:")
        for i, col in enumerate(columns):
            print(f"  [{i}] {col}")

        try:
            # Get X-axis column
            x_idx = int(input(f"\nSelect X-axis column (0-{len(columns)-1}): "))
            if x_idx < 0 or x_idx >= len(columns):
                print("Invalid X column selection")
                return

            # Get Y-axis column
            y_idx = int(input(f"Select Y-axis column (0-{len(columns)-1}): "))
            if y_idx < 0 or y_idx >= len(columns):
                print("Invalid Y column selection")
                return

            x_col = columns[x_idx]
            y_col = columns[y_idx]

            print(f"\nCreating plot: {x_col} vs {y_col}")

            # Choose plot type
            plot_types = [
                ("1", "Line Plot", "line"),
                ("2", "Scatter Plot", "scatter"),
                ("3", "Bar Chart", "bar")
            ]

            print("\nPlot types:")
            for opt, desc, _ in plot_types:
                print(f"  [{opt}] {desc}")

            plot_choice = input("Select plot type (1-3): ").strip()

            plot_type = "scatter"  # default
            for opt, desc, ptype in plot_types:
                if plot_choice == opt:
                    plot_type = ptype
                    break

            # Create custom plot
            self._create_custom_plot(data, x_col, y_col, plot_type)

        except ValueError:
            print("Invalid input. Please enter numbers only.")
        except Exception as e:
            print(f"Error creating custom plot: {e}")

    def _create_specific_visualization(self, data: pd.DataFrame, viz_type: str) -> None:
        """Create specific type of visualization"""
        try:
            if viz_type == "depth_profile":
                result = self._create_depth_profiles(data)
                if result:
                    print(f"✅ Created: {result[0]}")

            elif viz_type == "ts_diagram":
                result = self._create_ts_diagram(data)
                if result:
                    print(f"✅ Created: {result}")

            elif viz_type == "geo_map":
                result = self._create_geographic_map(data)
                if result:
                    print(f"✅ Created: {result}")

            elif viz_type == "time_series":
                result = self._create_time_series(data)
                if result:
                    print(f"✅ Created: {result}")

        except Exception as e:
            print(f"❌ Error creating visualization: {e}")

    def _create_custom_plot(self, data: pd.DataFrame, x_col: str, y_col: str, plot_type: str) -> None:
        """Create custom plot with selected columns"""
        try:
            # Clean data
            plot_data = data[[x_col, y_col]].dropna()
            if len(plot_data) == 0:
                print("No valid data for plotting")
                return

            plt.figure(figsize=(10, 6))

            if plot_type == "line":
                plt.plot(plot_data[x_col], plot_data[y_col], 'b-', linewidth=2, marker='o', markersize=4)
            elif plot_type == "scatter":
                plt.scatter(plot_data[x_col], plot_data[y_col], alpha=0.6, s=50)
            elif plot_type == "bar":
                plt.bar(range(len(plot_data)), plot_data[y_col], alpha=0.7)
                plt.xlabel(f"Index ({x_col})")

            if plot_type != "bar":
                plt.xlabel(x_col)
            plt.ylabel(y_col)
            plt.title(f"{x_col} vs {y_col}")
            plt.grid(True, alpha=0.3)

            # Special handling for oceanographic data
            if 'depth' in y_col.lower() or 'pressure' in y_col.lower():
                plt.gca().invert_yaxis()  # Oceanographic convention

            plt.tight_layout()

            # Save plot
            plot_filename = f"custom_plot_{int(time.time())}.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"✅ Created custom plot: {plot_filename}")

        except Exception as e:
            print(f"❌ Error creating custom plot: {e}")

    def _create_geographic_map(self, data: pd.DataFrame) -> str:
        """Create geographic map of lat/lon positions"""
        try:
            # Find lat/lon columns
            lat_col = None
            lon_col = None

            for col in data.columns:
                if 'latitude' in col.lower() or 'lat' in col.lower():
                    lat_col = col
                elif 'longitude' in col.lower() or 'lon' in col.lower():
                    lon_col = col

            if not lat_col or not lon_col:
                return None

            geo_data = data[[lat_col, lon_col]].dropna()
            if len(geo_data) == 0:
                return None

            plt.figure(figsize=(12, 8))
            plt.scatter(geo_data[lon_col], geo_data[lat_col], alpha=0.6, s=50)
            plt.xlabel('Longitude (°E)')
            plt.ylabel('Latitude (°N)')
            plt.title('Geographic Distribution of Data Points')
            plt.grid(True, alpha=0.3)

            plot_filename = f"geo_map_{int(time.time())}.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            plt.close()

            return f"Geographic map: {plot_filename}"
        except Exception as e:
            return f"Error creating geographic map: {e}"

    def _create_time_series(self, data: pd.DataFrame) -> str:
        """Create time series plot"""
        try:
            # Find date column
            date_col = None
            for col in data.columns:
                if 'date' in col.lower() or 'time' in col.lower():
                    date_col = col
                    break

            if not date_col:
                return None

            # Find parameter column (temp or salinity)
            param_col = None
            for col in data.columns:
                if ('temperature' in col.lower() or 'salinity' in col.lower()) and col != date_col:
                    param_col = col
                    break

            if not param_col:
                return None

            ts_data = data[[date_col, param_col]].dropna()
            if len(ts_data) == 0:
                return None

            plt.figure(figsize=(12, 6))
            plt.plot(ts_data[date_col], ts_data[param_col], 'b-', linewidth=2, marker='o', markersize=3)
            plt.xlabel('Date/Time')
            plt.ylabel(param_col)
            plt.title(f'{param_col} Time Series')
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)

            plot_filename = f"time_series_{int(time.time())}.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            plt.close()

            return f"Time series: {plot_filename}"
        except Exception as e:
            return f"Error creating time series: {e}"

    def query_groq_llm(self, prompt: str, context: str = "") -> str:
        """Query Groq API for LLM response with enhanced prompting and caching"""
        if not self.groq_client:
            return "Groq API not configured. Please provide API key."

        # Check cache first
        cache_key = hash(prompt + context)
        if cache_key in self.response_cache:
            return self.response_cache[cache_key]

        try:
            # Enhanced structured prompt
            full_prompt = f"""You are Dr. ARGO, a world-class oceanographic data analyst. Analyze the provided SQL data results and answer the user's question directly using the actual data.

ACTUAL DATA RESULTS:
{context}

USER QUESTION: {prompt}

ANALYSIS INSTRUCTIONS:
1. Use ONLY the specific data provided in the SQL results above
2. Extract the exact information the user requested from the data
3. Provide concrete numbers, ranges, and statistics from the actual results
4. If asked for float IDs, list the actual IDs from the data
5. Be precise and factual based on the real data

RESPONSE REQUIREMENTS:
- Answer using the actual data provided
- Include specific numbers, ranges, or lists from the results
- Keep response under 100 words but include relevant details
- Do not make up or guess any information

Analyze the real data above and provide a direct, factual answer."""

            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": full_prompt,
                    }
                ],
                model="llama-3.1-8b-instant",  # Fast and capable model
                temperature=0.1,  # Optimized for speed and consistency
                max_tokens=600,   # Optimized response length
                top_p=0.9,        # Focus on most likely tokens
            )

            response = chat_completion.choices[0].message.content

            # Cache the response
            self._update_cache(cache_key, response)

            return response

        except Exception as e:
            return f"Groq API error: {e}"

    def _update_cache(self, cache_key: int, response: str):
        """Update response cache with size limit"""
        if len(self.response_cache) >= self.cache_max_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self.response_cache))
            del self.response_cache[oldest_key]

        self.response_cache[cache_key] = response

    def comprehensive_query(self, user_query: str, execute_sql: bool = True,
                          generate_viz: bool = True) -> Dict:
        """Complete query processing pipeline"""
        print(f"\nProcessing query: '{user_query}'")
        print("=" * 60)

        # Step 1: Semantic search
        print("1. Performing semantic search...")
        search_results = self.semantic_search(user_query, n_results=3)

        if not search_results:
            return {"error": "No search results found"}

        best_match = search_results[0]
        print(f"   Best match: {best_match['id']} (similarity: {best_match['similarity']:.3f})")

        # Step 2: Extract SQL and plot config
        sql_template = best_match['metadata'].get('sql_template', '')
        plot_config = best_match['metadata'].get('plot_config', {})
        parameters = best_match['metadata'].get('parameters', {})

        result = {
            'query': user_query,
            'best_match': best_match,
            'sql_template': sql_template,
            'plot_config': plot_config,
            'parameters': parameters
        }

        # Step 3: Execute SQL if requested and available
        sql_data = pd.DataFrame()
        if execute_sql and sql_template:
            print("2. Executing SQL query...")
            # Use default parameters for demo
            default_params = {}
            if isinstance(parameters, dict):
                for param, param_info in parameters.items():
                    if 'default' in param_info:
                        if 'integer' in param_info:
                            default_params[param] = 10000
                        elif 'float' in param_info:
                            default_params[param] = 50.0
                        elif 'date' in param_info:
                            default_params[param] = '2020-01-01'
                        else:
                            default_params[param] = 'profile_001'

            sql_data = self.execute_sql_query(sql_template, default_params)
            result['sql_data'] = sql_data

        # Step 4: Generate visualization if requested OR auto-detect oceanographic data
        if generate_viz and not sql_data.empty:
            print("3. Generating visualization...")
            viz_result = self.generate_visualization(sql_data, plot_config or {})
            result['visualization'] = viz_result

        # Step 5: Generate LLM response with enhanced data analysis
        print("4. Generating LLM response...")
        context = f"Best match: {best_match['content']}\n"

        if not sql_data.empty:
            # Enhanced data analysis context
            context += self.generate_data_summary(sql_data, user_query)

        if plot_config:
            context += f"Visualization type: {plot_config.get('type', 'unknown')}\n"

        llm_response = self.query_groq_llm(user_query, context)
        result['llm_response'] = llm_response

        return result

    def generate_data_summary(self, sql_data, user_query):
        """Generate intelligent data summary for LLM context"""
        import numpy as np

        summary = f"SQL Data Analysis ({len(sql_data)} records):\n"

        # Column analysis
        columns = sql_data.columns.tolist()
        summary += f"Columns: {', '.join(columns)}\n"

        # Specific analysis based on query type and available columns
        if 'float_id' in columns:
            unique_floats = sql_data['float_id'].nunique()
            float_list = sql_data['float_id'].unique()
            summary += f"Unique Floats: {unique_floats} (IDs: {', '.join(map(str, sorted(float_list)[:10]))}{'...' if len(float_list) > 10 else ''})\n"

        # Numeric column analysis
        numeric_cols = sql_data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in sql_data.columns and not sql_data[col].isna().all():
                data_col = sql_data[col].dropna()
                if len(data_col) > 0:
                    summary += f"{col}: Range {data_col.min():.2f} to {data_col.max():.2f}, Mean {data_col.mean():.2f}\n"

        # Geographic analysis
        if 'latitude' in columns and 'longitude' in columns:
            lat_range = f"{sql_data['latitude'].min():.2f} to {sql_data['latitude'].max():.2f}"
            lon_range = f"{sql_data['longitude'].min():.2f} to {sql_data['longitude'].max():.2f}"
            summary += f"Geographic Coverage: Lat {lat_range}, Lon {lon_range}\n"

        # Temporal analysis
        if 'profile_date' in columns:
            dates = sql_data['profile_date'].dropna()
            if len(dates) > 0:
                summary += f"Time Range: {dates.min()} to {dates.max()}\n"

        # Quality control analysis
        qc_cols = [col for col in columns if 'qc' in col.lower()]
        if qc_cols:
            for qc_col in qc_cols[:2]:  # Limit to 2 QC columns
                qc_counts = sql_data[qc_col].value_counts()
                summary += f"{qc_col}: {dict(qc_counts.head(3))}\n"

        # Measurement analysis
        if 'measurement_id' in columns:
            measurements_per_profile = sql_data.groupby('profile_id')['measurement_id'].count() if 'profile_id' in columns else None
            if measurements_per_profile is not None:
                summary += f"Measurements per Profile: Avg {measurements_per_profile.mean():.0f}, Range {measurements_per_profile.min()}-{measurements_per_profile.max()}\n"

        return summary

    def check_fast_path(self, query: str) -> Dict:
        """Check if query matches fast-path patterns for instant responses"""
        query_clean = query.lower().strip()

        # Direct matches
        if query_clean in self.fast_path_queries:
            return self.fast_path_queries[query_clean]

        # Fuzzy matches for common variations
        for fast_query, fast_data in self.fast_path_queries.items():
            if fast_query in query_clean or query_clean in fast_query:
                return fast_data

        return None

    def preprocess_query(self, user_query: str) -> str:
        """Enhance user query for better semantic matching"""
        # Oceanographic term enhancements
        enhancements = {
            'temp': 'temperature measurements',
            'depth': 'depth profile oceanographic',
            'plot': 'create visualization plot',
            'show': 'display oceanographic data',
            'salinity': 'salinity measurements ocean',
            'pressure': 'pressure depth water column',
            'profile': 'vertical profile ocean structure',
            'float': 'ARGO float oceanographic instrument',
            'data': 'oceanographic measurements data',
            'heatmap': 'heatmap spatial visualization',
            'recent': 'recent temporal analysis',
            'vs': 'versus comparison analysis'
        }

        query_lower = user_query.lower()
        enhanced_query = user_query

        for key, enhancement in enhancements.items():
            if key in query_lower and enhancement not in query_lower:
                enhanced_query += f" {enhancement}"

        return enhanced_query

    def simple_query(self, user_query: str) -> Dict:
        """Simplified query method for web interface with intelligent SQL generation"""
        result = {
            "best_match_id": "",
            "similarity": 0.0,
            "matched_sample": {},
            "llm_response": "",
            "sql_executed": False,
            "sql": "",
            "sql_data": [],
            "visualization_created": False,
            "plot_type": "",
            "plot_html": ""
        }

        try:
            # Check fast-path first for instant responses
            fast_path = self.check_fast_path(user_query)
            if fast_path:
                try:
                    sql_data = self.execute_sql_query(fast_path["sql"])
                    result.update({
                        "best_match_id": "fast_path_match",
                        "similarity": 1.0,  # Perfect match for fast-path
                        "llm_response": fast_path["response"],
                        "sql_executed": True,
                        "sql": fast_path["sql"],
                        "sql_data": sql_data.to_dict('records') if not sql_data.empty else [],
                        "visualization_created": len(sql_data) > 0 if not sql_data.empty else False,
                        "plot_type": fast_path["viz_type"]
                    })
                    return result
                except Exception as e:
                    # Fall through to normal processing if fast-path fails
                    pass

            # Step 1: Preprocess query for better matching
            enhanced_query = self.preprocess_query(user_query)

            # Step 2: Semantic search with enhanced query
            search_results = self.semantic_search(enhanced_query, n_results=3)
            if not search_results:
                result["llm_response"] = "No similar queries found in the database."
                return result

            best_match = search_results[0]
            result["best_match_id"] = best_match['id']
            result["similarity"] = best_match['similarity']

            # Try to parse JSON content, fallback to raw content
            try:
                result["matched_sample"] = json.loads(best_match['content'])
            except (json.JSONDecodeError, TypeError):
                result["matched_sample"] = {"raw_content": best_match['content'], "metadata": best_match.get('metadata', {})}

            # Step 2: Check if LLM thinks plotting is needed
            plot_decision = self.llm_decide_plotting(user_query, best_match['content'])

            # Step 3: Generate visualization-aware SQL using LLM
            sql_generated = self.generate_visualization_aware_sql(user_query, search_results, plot_decision)

            if sql_generated["success"]:
                try:
                    sql_data = self.execute_sql_query(sql_generated["sql"])
                    result["sql_executed"] = True
                    result["sql"] = sql_generated["sql"]
                    result["sql_data"] = sql_data.to_dict('records') if not sql_data.empty else []

                    # If no data, try a fallback query
                    if len(result["sql_data"]) == 0:
                        fallback_sql = self.generate_fallback_sql(user_query)
                        if fallback_sql:
                            fallback_data = self.execute_sql_query(fallback_sql)
                            if not fallback_data.empty:
                                result["sql"] = f"Original returned 0 rows. Using fallback: {fallback_sql}"
                                result["sql_data"] = fallback_data.to_dict('records')
                            else:
                                result["sql"] = f"No data found. Fallback also returned 0 rows."

                except Exception as e:
                    result["sql_executed"] = False
                    result["sql"] = f"SQL execution failed: {str(e)}"
            else:
                result["sql"] = f"SQL generation failed: {sql_generated['error']}"

            # Step 4: Create visualization if needed
            if plot_decision["needs_plotting"] and result["sql_executed"] and result["sql_data"]:
                try:
                    plot_info = self.create_visualization(
                        result["sql_data"],
                        plot_decision["plot_type"],
                        user_query
                    )
                    result["visualization_created"] = plot_info.get("success", False)
                    result["plot_type"] = plot_info["type"]
                    result["plot_html"] = plot_info.get("html", "")
                except Exception as e:
                    result["plot_html"] = f"<p>Visualization failed: {str(e)}</p>"

            # Step 5: Generate optimized LLM response
            context = self._build_smart_context(user_query, best_match, result)
            result["llm_response"] = self.query_groq_llm(user_query, context)

        except Exception as e:
            result["llm_response"] = f"Error processing query: {str(e)}"

        return result

    def _build_smart_context(self, user_query: str, best_match: Dict, result: Dict) -> str:
        """Build optimized context for LLM with only relevant information"""
        context_parts = []

        # Core query info
        context_parts.append(f"QUERY: {user_query}")
        context_parts.append(f"MATCH_CONFIDENCE: {best_match['similarity']:.3f}")

        # Data summary (key metrics only)
        if result["sql_executed"] and result["sql_data"]:
            data = result["sql_data"]
            context_parts.append(f"DATA_COUNT: {len(data)} records")

            # Smart data statistics
            if data and isinstance(data[0], dict):
                # Temperature stats if available
                temps = [row.get('temperature') for row in data if row.get('temperature')]
                if temps:
                    temps = [t for t in temps if t and t > -5 and t < 50]  # Valid range
                    if temps:
                        context_parts.append(f"TEMPERATURE_RANGE: {min(temps):.1f}°C to {max(temps):.1f}°C (avg: {sum(temps)/len(temps):.1f}°C)")

                # Pressure/depth stats if available
                pressures = [row.get('pressure') for row in data if row.get('pressure')]
                if pressures:
                    pressures = [p for p in pressures if p and p > 0 and p < 7000]  # Valid range
                    if pressures:
                        context_parts.append(f"DEPTH_RANGE: {min(pressures):.0f} to {max(pressures):.0f} dbar")

                # Geographic extent if available
                lats = [row.get('latitude') for row in data if row.get('latitude')]
                lons = [row.get('longitude') for row in data if row.get('longitude')]
                if lats and lons:
                    context_parts.append(f"GEOGRAPHIC_EXTENT: {min(lats):.1f}°N-{max(lats):.1f}°N, {min(lons):.1f}°E-{max(lons):.1f}°E")

        # Analysis type
        if result["visualization_created"]:
            context_parts.append(f"VISUALIZATION: {result.get('plot_type', 'created')}")

        # Quality indicators
        if best_match['similarity'] > 0.8:
            context_parts.append("CONFIDENCE: HIGH - Data interpretation reliable")
        elif best_match['similarity'] > 0.6:
            context_parts.append("CONFIDENCE: MEDIUM - Verify findings")
        else:
            context_parts.append("CONFIDENCE: LOW - Results may need validation")

        return "\n".join(context_parts)

    def llm_decide_plotting(self, user_query: str, matched_content: str) -> Dict:
        """Intelligent oceanographic visualization decision with domain expertise"""
        if not self.groq_client:
            return self._fallback_plot_detection(user_query)

        prompt = f"""
        OCEANOGRAPHIC VISUALIZATION INTELLIGENCE

        User Query: "{user_query}"
        Matched Context: {matched_content}

        OCEANOGRAPHIC ANALYSIS PATTERNS:
        1. VERTICAL ANALYSIS (depth_profile):
           Triggers: "profile", "depth", "vertical", "thermocline", "vs pressure"
           Data needed: pressure + parameter (temperature/salinity)
           Plot: Parameter vs Pressure (Y-axis inverted for ocean convention)

        2. WATER MASS ANALYSIS (ts_diagram):
           Triggers: "T-S", "temperature vs salinity", "water mass", "mixing"
           Data needed: temperature + salinity
           Plot: Temperature vs Salinity scatter (colored by depth)

        3. SPATIAL ANALYSIS (geographic_heatmap):
           Triggers: "map", "spatial", "geographic", "distribution", "heatmap"
           Data needed: latitude + longitude + parameter
           Plot: Geographic map with parameter overlay

        4. TEMPORAL ANALYSIS (time_series):
           Triggers: "time", "temporal", "evolution", "trends", "seasonal"
           Data needed: date + parameter
           Plot: Parameter vs Time line chart

        5. CORRELATION ANALYSIS (parameter_scatter):
           Triggers: "vs", "versus", "correlation", "relationship", "compare"
           Data needed: two parameters
           Plot: Parameter1 vs Parameter2 scatter

        6. COMPARATIVE ANALYSIS (multi_comparison):
           Triggers: "compare", "multiple", "different floats"
           Data needed: float_id + parameters
           Plot: Multiple series comparison

        Respond with JSON:
        {{
            "needs_plotting": true/false,
            "plot_type": "depth_profile|ts_diagram|geographic_heatmap|time_series|parameter_scatter|multi_comparison|none",
            "analysis_intent": "vertical_analysis|water_mass_analysis|spatial_analysis|temporal_analysis|correlation_analysis|comparative_analysis|data_query",
            "required_columns": ["column1", "column2"],
            "reason": "oceanographic analysis explanation"
        }}

        CRITICAL: Consider oceanographic domain context, not just plot keywords.
        """

        try:
            response = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.1-8b-instant",
                temperature=0.1
            )

            response_text = response.choices[0].message.content.strip()

            # Try to parse JSON response
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                # Fallback: simple keyword detection
                plot_keywords = ['plot', 'graph', 'chart', 'heatmap', 'visualization', 'show', 'display']
                needs_plot = any(keyword in user_query.lower() for keyword in plot_keywords)
                return {
                    "needs_plotting": needs_plot,
                    "plot_type": "line" if needs_plot else "none",
                    "reason": "keyword detection fallback"
                }
        except Exception as e:
            return self._fallback_plot_detection(user_query)

    def _fallback_plot_detection(self, user_query: str) -> Dict:
        """Fallback oceanographic plot detection using domain patterns"""
        query_lower = user_query.lower()

        # Oceanographic pattern detection
        if any(word in query_lower for word in ['profile', 'depth', 'vertical', 'thermocline']):
            return {
                "needs_plotting": True,
                "plot_type": "depth_profile",
                "analysis_intent": "vertical_analysis",
                "required_columns": ["pressure", "temperature"],
                "reason": "Detected depth profile request"
            }

        elif any(word in query_lower for word in ['t-s', 'temperature vs salinity', 'water mass']):
            return {
                "needs_plotting": True,
                "plot_type": "ts_diagram",
                "analysis_intent": "water_mass_analysis",
                "required_columns": ["temperature", "salinity"],
                "reason": "Detected T-S diagram request"
            }

        elif any(word in query_lower for word in ['map', 'spatial', 'geographic', 'heatmap']):
            return {
                "needs_plotting": True,
                "plot_type": "geographic_heatmap",
                "analysis_intent": "spatial_analysis",
                "required_columns": ["latitude", "longitude", "temperature"],
                "reason": "Detected spatial analysis request"
            }

        elif any(word in query_lower for word in ['time', 'temporal', 'evolution', 'trend']):
            return {
                "needs_plotting": True,
                "plot_type": "time_series",
                "analysis_intent": "temporal_analysis",
                "required_columns": ["profile_date", "temperature"],
                "reason": "Detected time series request"
            }

        elif ' vs ' in query_lower or 'versus' in query_lower:
            return {
                "needs_plotting": True,
                "plot_type": "parameter_scatter",
                "analysis_intent": "correlation_analysis",
                "required_columns": ["temperature", "salinity"],
                "reason": "Detected parameter comparison"
            }

        # Basic visualization keywords
        elif any(word in query_lower for word in ['plot', 'chart', 'graph', 'show', 'display', 'visualize']):
            return {
                "needs_plotting": True,
                "plot_type": "depth_profile",  # Default to most common oceanographic plot
                "analysis_intent": "vertical_analysis",
                "required_columns": ["pressure", "temperature"],
                "reason": "General visualization request - defaulting to depth profile"
            }

        return {
            "needs_plotting": False,
            "plot_type": "none",
            "analysis_intent": "data_query",
            "required_columns": [],
            "reason": "No visualization indicators detected"
        }

    def create_visualization(self, data: List[Dict], plot_type: str, query: str) -> Dict:
        """Create interactive visualization from data using Plotly"""
        if not data:
            return {"type": "none", "html": "<p>No data to plot</p>"}

        try:
            import plotly.graph_objects as go
            import plotly.express as px
            from plotly.offline import plot

            df = pd.DataFrame(data)
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

            if len(numeric_cols) < 1:
                return {"type": "error", "html": "<p>No numeric data found for plotting</p>"}

            fig = None

            # Temperature vs Pressure plot (oceanographic standard)
            if 'temperature' in df.columns and 'pressure' in df.columns:
                # Filter out invalid data (zeros and nulls)
                df_clean = df[(df['temperature'] > 0) & (df['pressure'] > 0)].copy()

                if len(df_clean) == 0:
                    return {"type": "error", "html": "<p>No valid temperature/pressure data found (all values are zero or null)</p>"}

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_clean['temperature'],
                    y=df_clean['pressure'],
                    mode='markers+lines',
                    name='Temperature Profile',
                    marker=dict(size=6, color='red', opacity=0.7),
                    line=dict(width=2, color='red'),
                    hovertemplate='<b>Temperature</b>: %{x:.2f}°C<br><b>Pressure</b>: %{y:.1f} dbar<extra></extra>'
                ))
                fig.update_layout(
                    title=f"Temperature vs Pressure Profile<br><sub>{query}</sub><br><span style='font-size:12px'>({len(df_clean)} data points)</span>",
                    xaxis_title="Temperature (°C)",
                    yaxis_title="Pressure (dbar) - Depth increases downward",
                    yaxis=dict(autorange="reversed"),  # Invert Y-axis (deeper = higher pressure)
                    template="plotly_white",
                    height=500
                )

            # Temperature vs Salinity scatter plot
            elif 'temperature' in df.columns and 'salinity' in df.columns:
                fig = px.scatter(
                    df,
                    x='temperature',
                    y='salinity',
                    title=f"Temperature vs Salinity Scatter Plot<br><sub>{query}</sub>",
                    labels={"temperature": "Temperature (°C)", "salinity": "Salinity (PSU)"},
                    template="plotly_white",
                    height=500
                )
                fig.update_traces(marker=dict(size=8, opacity=0.7))

            # Generic scatter plot for any two numeric columns
            elif plot_type == "scatter" and len(numeric_cols) >= 2:
                fig = px.scatter(
                    df,
                    x=numeric_cols[0],
                    y=numeric_cols[1],
                    title=f"Scatter Plot: {numeric_cols[0]} vs {numeric_cols[1]}<br><sub>{query}</sub>",
                    template="plotly_white",
                    height=500
                )

            # Line plot
            elif plot_type == "line" and len(numeric_cols) >= 1:
                if len(numeric_cols) >= 2:
                    fig = px.line(
                        df,
                        x=numeric_cols[0],
                        y=numeric_cols[1],
                        title=f"Line Plot: {numeric_cols[0]} vs {numeric_cols[1]}<br><sub>{query}</sub>",
                        template="plotly_white",
                        height=500
                    )
                else:
                    fig = px.line(
                        df,
                        y=numeric_cols[0],
                        title=f"Line Plot: {numeric_cols[0]}<br><sub>{query}</sub>",
                        template="plotly_white",
                        height=500
                    )

            # Histogram
            elif plot_type == "histogram":
                fig = px.histogram(
                    df,
                    x=numeric_cols[0],
                    title=f"Distribution: {numeric_cols[0]}<br><sub>{query}</sub>",
                    template="plotly_white",
                    height=500
                )

            # Default: line plot of first numeric column
            else:
                fig = px.line(
                    df,
                    y=numeric_cols[0],
                    title=f"Data Plot: {numeric_cols[0]}<br><sub>{query}</sub>",
                    template="plotly_white",
                    height=500
                )

            if fig:
                # Convert to HTML with embedded JavaScript
                html_plot = plot(fig, output_type='div', include_plotlyjs='cdn')
                return {"type": plot_type, "html": html_plot, "success": True}
            else:
                return {"type": "error", "html": "<p>Could not create plot with available data</p>"}

        except ImportError:
            # Fallback to matplotlib if plotly not available
            return self.create_matplotlib_fallback(data, plot_type, query)
        except Exception as e:
            return {"type": "error", "html": f"<p>Plotting failed: {str(e)}</p>"}

    def create_matplotlib_fallback(self, data: List[Dict], plot_type: str, query: str) -> Dict:
        """Fallback to matplotlib if plotly is not available"""
        try:
            import base64
            from io import BytesIO

            df = pd.DataFrame(data)
            numeric_cols = df.select_dtypes(include=[np.number]).columns

            plt.figure(figsize=(10, 6))

            if 'temperature' in df.columns and 'pressure' in df.columns:
                plt.plot(df['temperature'], df['pressure'], 'ro-', markersize=4, linewidth=2)
                plt.xlabel("Temperature (°C)")
                plt.ylabel("Pressure (dbar)")
                plt.gca().invert_yaxis()  # Invert Y-axis for oceanographic convention
            elif len(numeric_cols) >= 2:
                plt.plot(df[numeric_cols[0]], df[numeric_cols[1]], 'bo-', markersize=4)
                plt.xlabel(numeric_cols[0])
                plt.ylabel(numeric_cols[1])
            elif len(numeric_cols) >= 1:
                plt.plot(df[numeric_cols[0]], 'b-', linewidth=2)
                plt.ylabel(numeric_cols[0])
                plt.xlabel("Index")

            plt.title(f"ARGO Data Visualization\n{query}")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            # Convert to base64 for embedding in HTML
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close()

            html_img = f'<img src="data:image/png;base64,{image_base64}" style="width:100%; max-width:800px;" alt="ARGO Data Plot">'
            return {"type": plot_type, "html": html_img, "success": True}

        except Exception as e:
            return {"type": "error", "html": f"<p>Fallback plotting failed: {str(e)}</p>"}

    def generate_intelligent_sql(self, user_query: str, search_results: List[Dict]) -> Dict:
        """Generate SQL using LLM based on user query and search results"""
        if not self.groq_client:
            return {"success": False, "error": "LLM not available"}

        # Get database schema information
        schema_info = self.get_database_schema()

        # Prepare context from search results
        context_samples = []
        for result in search_results[:3]:  # Use top 3 matches
            try:
                sample = json.loads(result['content'])
                if 'metadata' in result and 'sql_template' in result['metadata']:
                    context_samples.append({
                        "id": result['id'],
                        "similarity": result['similarity'],
                        "sql_template": result['metadata']['sql_template']
                    })
            except:
                pass

        prompt = f"""
        ARGO OCEANOGRAPHIC SQL GENERATOR - ANTI-HALLUCINATION MODE

        User Query: "{user_query}"

        Database Schema:
        {schema_info}

        Similar validated examples:
        {json.dumps(context_samples, indent=2)}

        CRITICAL ANTI-HALLUCINATION RULES:
        1. ✓ COLUMN VALIDATION: Only use columns that exist in schema above
        2. ✓ DATA TYPE ACCURACY: QC flags are INTEGERS (1,2,3) NOT strings ('1','2','3')
        3. ✓ OCEANOGRAPHIC DOMAIN: Temperature CAN BE NEGATIVE (polar waters -2°C to +30°C)
        4. ✓ STRING IDENTIFIERS: float_id is STRING, use quotes: float_id = '2900826'
        5. ✓ NULL HANDLING: Use IS NOT NULL, never exclude valid negative temperatures

        STEP-BY-STEP VALIDATION:
        Step 1: Verify all column names exist in schema
        Step 2: Apply correct data types (integer QC, string float_id)
        Step 3: Use proper table joins (profiles.profile_id = measurements.profile_id)
        Step 4: Include oceanographically valid ranges (temperature IS NOT NULL only)
        Step 5: Add LIMIT for performance

        COMMON HALLUCINATION PREVENTION:
        ❌ NEVER use: temperature > 0 (excludes polar waters)
        ❌ NEVER use: temperature_qc IN ('1','2') (QC flags are integers)
        ❌ NEVER use: float_id = 2900826 (missing quotes for string)
        ❌ NEVER use: temp, sal, lat, lon (use full column names)

        ✅ ALWAYS use: temperature IS NOT NULL
        ✅ ALWAYS use: temperature_qc IN (1,2)
        ✅ ALWAYS use: float_id = '2900826'
        ✅ ALWAYS use: temperature, salinity, latitude, longitude

        Generate ONLY validated SQL query:
        """

        try:
            response = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.1-8b-instant",
                temperature=0.1
            )

            sql_query = response.choices[0].message.content.strip()

            # Clean up the response - remove markdown formatting if present
            if "```sql" in sql_query:
                sql_query = sql_query.split("```sql")[1].split("```")[0].strip()
            elif "```" in sql_query:
                sql_query = sql_query.split("```")[1].split("```")[0].strip()

            # Enhanced validation
            validation_result = self.validate_generated_sql(sql_query, user_query)
            if not validation_result["valid"]:
                return {"success": False, "error": f"SQL validation failed: {validation_result['error']}"}

            return {"success": True, "sql": sql_query}

        except Exception as e:
            return {"success": False, "error": f"LLM SQL generation failed: {str(e)}"}

    def generate_visualization_aware_sql(self, user_query: str, search_results: List[Dict], plot_decision: Dict) -> Dict:
        """Generate SQL that includes required columns for visualization"""
        if not self.groq_client:
            return {"success": False, "error": "LLM not available"}

        # Get database schema information
        schema_info = self.get_database_schema()

        # Build visualization requirements
        viz_requirements = ""
        if plot_decision.get("needs_plotting", False):
            required_cols = plot_decision.get("required_columns", [])
            analysis_intent = plot_decision.get("analysis_intent", "")
            plot_type = plot_decision.get("plot_type", "")

            viz_requirements = f"""
            VISUALIZATION REQUIREMENTS DETECTED:
            Analysis Intent: {analysis_intent}
            Plot Type: {plot_type}
            Required Columns: {', '.join(required_cols)}

            CRITICAL: SQL must include ALL required columns for visualization.
            """

        # Prepare context from search results
        context_samples = []
        for result in search_results[:3]:
            try:
                sample = json.loads(result['content'])
                if 'metadata' in result and 'sql_template' in result['metadata']:
                    context_samples.append({
                        "id": result['id'],
                        "similarity": result['similarity'],
                        "sql_template": result['metadata']['sql_template']
                    })
            except:
                pass

        prompt = f"""
        ARGO OCEANOGRAPHIC SQL GENERATOR - VISUALIZATION-AWARE MODE

        User Query: "{user_query}"

        {viz_requirements}

        Database Schema:
        {schema_info}

        Similar validated examples:
        {json.dumps(context_samples, indent=2)}

        VISUALIZATION-AWARE SQL GENERATION RULES:
        1. ✓ Include ALL required columns for visualization
        2. ✓ For depth profiles: MUST include pressure AND parameter columns
        3. ✓ For spatial analysis: MUST include latitude, longitude AND parameter
        4. ✓ For time series: MUST include profile_date AND parameter
        5. ✓ For correlations: MUST include both parameter columns
        6. ✓ Order data appropriately for visualization (pressure ASC for profiles)
        7. ✓ Use proper quality filters for clean visualizations
        8. ✓ Include sufficient data points (avoid overly restrictive filters)

        OCEANOGRAPHIC PLOT-SPECIFIC SQL PATTERNS:
        - Depth Profile: SELECT pressure, temperature, salinity ORDER BY pressure ASC
        - T-S Diagram: SELECT temperature, salinity, pressure (for coloring by depth)
        - Geographic: SELECT latitude, longitude, AVG(parameter) GROUP BY lat/lon bins
        - Time Series: SELECT profile_date, AVG(parameter) GROUP BY date ORDER BY date

        Generate SQL that provides the RIGHT DATA for the intended visualization:
        """

        try:
            response = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.1-8b-instant",
                temperature=0.1
            )

            sql_query = response.choices[0].message.content.strip()

            # Clean up the response
            if "```sql" in sql_query:
                sql_query = sql_query.split("```sql")[1].split("```")[0].strip()
            elif "```" in sql_query:
                sql_query = sql_query.split("```")[1].split("```")[0].strip()

            # Enhanced validation with visualization requirements
            validation_result = self.validate_visualization_sql(sql_query, user_query, plot_decision)
            if not validation_result["valid"]:
                return {"success": False, "error": f"Visualization SQL validation failed: {validation_result['error']}"}

            return {"success": True, "sql": sql_query}

        except Exception as e:
            # Fallback to original method
            return self.generate_intelligent_sql(user_query, search_results)

    def validate_visualization_sql(self, sql_query: str, user_query: str, plot_decision: Dict) -> dict:
        """Validate SQL includes required columns for visualization"""
        base_validation = self.validate_generated_sql(sql_query, user_query)
        if not base_validation["valid"]:
            return base_validation

        # Additional visualization-specific validation
        if not plot_decision.get("needs_plotting", False):
            return {"valid": True, "error": None}

        required_columns = plot_decision.get("required_columns", [])
        sql_lower = sql_query.lower()

        missing_columns = []
        for col in required_columns:
            if col.lower() not in sql_lower:
                missing_columns.append(col)

        if missing_columns:
            return {
                "valid": False,
                "error": f"SQL missing required columns for {plot_decision['plot_type']}: {', '.join(missing_columns)}"
            }

        # Check for appropriate ordering for specific plot types
        plot_type = plot_decision.get("plot_type", "")
        if plot_type == "depth_profile" and "order by" not in sql_lower:
            return {
                "valid": False,
                "error": "Depth profile requires ORDER BY pressure for proper visualization"
            }

        return {"valid": True, "error": None}

    def generate_fallback_sql(self, user_query: str) -> str:
        """Generate simple fallback SQL queries for common cases"""
        query_lower = user_query.lower()

        # Temperature vs salinity scatter plot
        if "temperature" in query_lower and "salinity" in query_lower and "scatter" in query_lower:
            return """
            SELECT m.temperature, m.salinity, p.profile_date, p.latitude, p.longitude
            FROM profiles p
            JOIN measurements m ON p.profile_id = m.profile_id
            WHERE m.temperature IS NOT NULL
            AND m.salinity IS NOT NULL
            AND m.temperature_qc IN ('1', '2')
            AND m.salinity_qc IN ('1', '2')
            ORDER BY p.profile_date DESC
            LIMIT 1000
            """

        # Temperature vs depth/pressure queries
        if "temperature" in query_lower and ("depth" in query_lower or "pressure" in query_lower):
            # Check if specific float_id is mentioned
            import re
            float_id_match = re.search(r'\b(\d{7})\b', user_query)
            if float_id_match:
                float_id = float_id_match.group(1)
                return f"""
                SELECT m.temperature, m.pressure
                FROM profiles p
                JOIN measurements m ON p.profile_id = m.profile_id
                WHERE p.float_id = '{float_id}'
                AND m.temperature IS NOT NULL AND m.temperature > 0
                AND m.pressure IS NOT NULL AND m.pressure > 0
                AND m.temperature_qc IN ('1', '2')
                ORDER BY m.pressure
                LIMIT 1000
                """
            else:
                return """
                SELECT m.temperature, m.pressure
                FROM profiles p
                JOIN measurements m ON p.profile_id = m.profile_id
                WHERE m.temperature IS NOT NULL AND m.temperature > 0
                AND m.pressure IS NOT NULL AND m.pressure > 0
                AND m.temperature_qc IN ('1', '2')
                ORDER BY m.pressure
                LIMIT 1000
                """

        # Float information queries
        if "float" in query_lower and any(word in query_lower for word in ["info", "about", "details"]):
            return "SELECT * FROM floats LIMIT 10"

        # Profile queries
        if "profile" in query_lower:
            return """
            SELECT p.profile_id, p.profile_date, p.latitude, p.longitude,
                   COUNT(m.measurement_id) as measurement_count
            FROM profiles p
            LEFT JOIN measurements m ON p.profile_id = m.profile_id
            GROUP BY p.profile_id
            ORDER BY p.profile_date DESC
            LIMIT 50
            """

        # General data queries
        if any(word in query_lower for word in ["data", "measurements", "show"]):
            return """
            SELECT p.profile_date, p.latitude, p.longitude,
                   m.temperature, m.salinity, m.pressure
            FROM profiles p
            JOIN measurements m ON p.profile_id = m.profile_id
            WHERE m.temperature IS NOT NULL
            AND m.temperature_qc IN ('1', '2')
            ORDER BY p.profile_date DESC
            LIMIT 200
            """

        return None

    def validate_generated_sql(self, sql_query: str, user_query: str) -> dict:
        """Validate generated SQL to prevent common hallucinations"""

        # Valid columns in our schema
        valid_columns = {
            'measurements': ['measurement_id', 'profile_id', 'pressure', 'depth', 'temperature', 'salinity',
                           'temperature_qc', 'salinity_qc', 'pressure_qc', 'dissolved_oxygen', 'ph_in_situ',
                           'chlorophyll_a', 'potential_temperature', 'potential_density'],
            'profiles': ['profile_id', 'float_id', 'cycle_number', 'profile_date', 'latitude', 'longitude',
                        'max_pressure', 'num_levels', 'data_mode', 'data_quality_flag'],
            'floats': ['float_id', 'wmo_number', 'deployment_date', 'deployment_latitude', 'deployment_longitude',
                      'current_status', 'last_latitude', 'last_longitude', 'total_profiles']
        }

        errors = []

        # 1. Check for basic SQL structure
        if not sql_query.upper().strip().startswith('SELECT'):
            errors.append("Query must be a SELECT statement")

        # 2. Check for hallucinated column names
        sql_lower = sql_query.lower()
        hallucinated_columns = ['temp', 'sal', 'lat', 'lon', 'depth_meters', 'temp_qc', 'sal_qc']
        for bad_col in hallucinated_columns:
            if bad_col in sql_lower:
                errors.append(f"Hallucinated column '{bad_col}' detected. Use full column names.")

        # 3. Check for string QC flags (common hallucination)
        if "temperature_qc in ('1'," in sql_lower or "salinity_qc in ('1'," in sql_lower:
            errors.append("QC flags are integers, not strings. Use IN (1, 2) not IN ('1', '2')")

        # 4. Check for temperature > 0 filter (oceanographic error)
        if "temperature > 0" in sql_lower:
            errors.append("Never filter temperature > 0. Ocean temperatures can be negative in polar waters.")

        # 5. Check float_id without quotes (if specific float mentioned)
        import re
        float_match = re.search(r'float_id\s*=\s*(\d+)', sql_lower)
        if float_match:
            errors.append("Float IDs are strings. Use float_id = '2900826' not float_id = 2900826")

        # 6. Check for proper joins
        if 'join' in sql_lower and 'profile_id' not in sql_lower:
            errors.append("Use profile_id for joining profiles and measurements tables")

        # 7. Check for LIMIT (performance)
        if 'limit' not in sql_lower and ('select *' in sql_lower or 'join' in sql_lower):
            errors.append("Add LIMIT clause to prevent massive result sets")

        if errors:
            return {"valid": False, "error": "; ".join(errors)}

        return {"valid": True, "error": None}

    def get_database_schema(self) -> str:
        """Get comprehensive database schema information for LLM context"""
        schema_info = """
        ARGO OCEANOGRAPHIC DATABASE SCHEMA (7.1M measurements, 42K profiles, 767 floats):

        === 1. FLOATS TABLE (767 records) - Float Hardware & Deployment Info ===
        Purpose: Autonomous oceanographic sensors that drift and profile the ocean

        COLUMNS & SCIENTIFIC USES:
        • float_id (string, PK): Unique 7-digit identifier (e.g., '2900826')
          Use: Primary identifier for tracking individual floats
        • wmo_number (int64): World Meteorological Organization identifier
          Use: International standardized float identification
        • program_name (string): Research program (e.g., 'ARGO', 'Euro-Argo')
          Use: Identify funding source, research focus, data protocols
        • platform_type (string): Float model/manufacturer (e.g., 'APEX', 'NOVA')
          Use: Understand sensor capabilities, measurement accuracy
        • deployment_date (string): When float was released into ocean
          Use: Calculate float age, operational lifespan analysis
        • deployment_latitude/longitude (float64): Initial drop location
          Use: Starting point for drift analysis, regional studies
        • deployment_depth (string): Water depth at deployment site
          Use: Understand bathymetry, deep vs shallow water deployment
        • current_status (string): Operational status ('ACTIVE', 'INACTIVE')
          Use: Filter for currently reporting floats
        • last_latitude/longitude (float64): Most recent known position
          Use: Current location, drift path analysis
        • cycle_time_days (int64): Days between dive cycles (typically 10)
          Use: Data frequency analysis, temporal resolution planning
        • total_profiles (int64): Number of completed dive cycles
          Use: Data availability assessment, float productivity

        DATA EXTRACTION EXAMPLES:
        - Active floats: WHERE current_status = 'ACTIVE'
        - Recent deployments: WHERE deployment_date >= '2020-01-01'
        - Pacific floats: WHERE deployment_longitude BETWEEN 120 AND -70
        - High-productivity floats: WHERE total_profiles > 100

        === 2. PROFILES TABLE (42,056 records) - Individual Dive Cycles ===
        Purpose: Each record represents one complete dive cycle (surface → deep → surface)

        COLUMNS & SCIENTIFIC USES:
        • profile_id (int64, PK): Unique dive cycle identifier
          Use: Primary key for linking to measurements
        • float_id (string, FK): Links to floats table
          Use: Group all dives from same float
        • cycle_number (int64): Sequential dive number (1, 2, 3...)
          Use: Track float aging, temporal analysis
        • profile_direction (string): 'A' (ascending) or 'D' (descending)
          Use: Distinguish upward vs downward measurements
        • profile_date (string): Date/time of dive cycle
          Use: Time series analysis, seasonal studies
        • latitude/longitude (float64): Location of this specific dive
          Use: Spatial analysis, current tracking, regional studies
        • max_pressure (float64): Deepest point reached (dbar)
          Use: Analyze dive depth capability, deep water access
        • num_levels (int64): Number of measurement points in profile
          Use: Data density assessment, vertical resolution
        • data_mode (string): 'R' (real-time) or 'D' (delayed-mode)
          Use: Data quality level - delayed mode has better calibration
        • data_quality_flag (int64): Overall profile quality assessment
          Use: Filter high-quality profiles for scientific analysis

        DATA EXTRACTION EXAMPLES:
        - Recent profiles: WHERE profile_date >= '2023-01-01'
        - Deep profiles: WHERE max_pressure > 2000
        - High-resolution profiles: WHERE num_levels > 100
        - Quality profiles: WHERE data_mode = 'D' AND data_quality_flag <= 2

        === 3. MEASUREMENTS TABLE (7,118,411 records) - Sensor Data at Each Depth ===
        Purpose: Individual sensor readings at specific depths during each dive

        CORE PHYSICAL PARAMETERS:
        • measurement_id (int64, PK): Unique measurement identifier
        • profile_id (int64, FK): Links to profiles table
        • pressure (float64): Water pressure in dbar (1 dbar ≈ 1 meter depth)
          Use: Depth reference, water column structure, pressure effects
        • depth (float64): Calculated depth in meters from pressure
          Use: Easier depth visualization, mixed layer analysis
        • temperature (float64): Water temperature in °C (CAN BE NEGATIVE!)
          Use: Ocean heat content, climate change, thermocline analysis
          Range: Typically -2°C to +30°C (polar to tropical waters)
        • salinity (float64): Practical salinity in PSU (no units)
          Use: Water mass identification, circulation patterns, evaporation
          Range: Typically 32-37 PSU (freshwater mixing to hypersaline)

        QUALITY CONTROL FLAGS (ALL INTEGERS):
        • temperature_qc (int64): 1=good, 2=probably good, 3=probably bad, 4=bad
        • salinity_qc (int64): Same quality scale
        • pressure_qc (int64): Pressure measurement quality
          Use: Filter data by reliability, exclude bad measurements

        BIOGEOCHEMICAL PARAMETERS (Advanced floats only):
        • dissolved_oxygen (string): O₂ concentration (mg/L or μmol/kg)
          Use: Ocean health, hypoxic zones, biological productivity
        • ph_in_situ (string): Ocean acidity (pH scale)
          Use: Ocean acidification studies, carbon cycle
        • chlorophyll_a (string): Phytoplankton indicator (mg/m³)
          Use: Primary productivity, ecosystem health, algal blooms
        • particle_backscattering (string): Water clarity (optical)
          Use: Particle concentration, water quality

        DERIVED PARAMETERS:
        • potential_temperature (string): Temperature corrected for pressure
          Use: Water mass analysis, removing pressure effects
        • potential_density (string): Density corrected for pressure
          Use: Ocean stratification, buoyancy, mixing analysis
        • mixed_layer_depth (string): Surface mixed layer thickness
          Use: Air-sea interaction, seasonal mixing

        DATA EXTRACTION EXAMPLES:
        - Surface waters: WHERE pressure < 50
        - Deep ocean: WHERE pressure > 2000
        - Temperature profiles: SELECT pressure, temperature ORDER BY pressure
        - Water mass analysis: SELECT temperature, salinity, potential_density
        - Quality data only: WHERE temperature_qc IN (1, 2) AND salinity_qc IN (1, 2)
        - Biogeochemical data: WHERE dissolved_oxygen IS NOT NULL

        === KEY RELATIONSHIPS & JOINS ===
        floats.float_id → profiles.float_id → measurements.profile_id

        Common Analysis Patterns:
        - Float trajectory: JOIN floats+profiles for lat/lon over time
        - Depth profiles: JOIN profiles+measurements for vertical structure
        - Full context: JOIN all three tables for complete oceanographic analysis

        === CRITICAL SQL RULES ===
        1. QC FLAGS ARE INTEGERS: temperature_qc IN (1, 2) NOT ('1', '2')
        2. TEMPERATURE CAN BE NEGATIVE: Never filter temperature > 0
        3. FLOAT IDs ARE STRINGS: float_id = '2900826' with quotes
        4. PRESSURE = DEPTH: Use pressure for depth-based queries
        5. QUALITY FILTERING: Always consider QC flags for scientific analysis

        === OCEANOGRAPHIC QUERY EXAMPLES ===
        -- Float temperature range (all data):
        SELECT MIN(temperature), MAX(temperature), AVG(temperature)
        FROM profiles p JOIN measurements m ON p.profile_id = m.profile_id
        WHERE p.float_id = '2900826' AND m.temperature IS NOT NULL

        -- Thermocline analysis:
        SELECT pressure, temperature FROM profiles p
        JOIN measurements m ON p.profile_id = m.profile_id
        WHERE pressure BETWEEN 0 AND 1000 ORDER BY pressure

        -- Water mass properties:
        SELECT temperature, salinity, pressure FROM measurements
        WHERE temperature_qc IN (1,2) AND salinity_qc IN (1,2)
        """
        return schema_info
   

    def interactive_mode(self):
        """Interactive mode for querying the system"""
        print("\n" + "=" * 80)
        print("COMPREHENSIVE RAG SYSTEM - INTERACTIVE MODE")
        print("=" * 80)
        print("Commands:")
        print("  - Enter any oceanographic query")
        print("  - 'stats' - Show system statistics")
        print("  - 'test [query]' - Test semantic search only")
        print("  - 'exit' - Quit")
        print()

        while True:
            user_input = input("Enter your query: ").strip()

            if user_input.lower() == 'exit':
                print("Goodbye!")
                break

            elif user_input.lower() == 'stats':
                print(f"System Statistics:")
                print(f"  - Total samples in ChromaDB: {len(self.all_samples)}")
                print(f"  - Parquet files loaded: {len(self.dataframes)}")
                print(f"  - Groq API configured: {'Yes' if self.groq_client else 'No'}")
                continue

            elif user_input.lower().startswith('test '):
                query = user_input[5:].strip()
                if query:
                    results = self.semantic_search(query, n_results=3)
                    print(f"\nTop 3 matches for '{query}':")
                    for i, result in enumerate(results, 1):
                        print(f"{i}. {result['id']} (similarity: {result['similarity']:.3f})")
                continue

            elif user_input == '':
                continue

            # Full comprehensive query
            result = self.comprehensive_query(user_input)

            print(f"\nRESULTS:")
            print(f"Best match: {result['best_match']['id']}")
            print(f"Similarity: {result['best_match']['similarity']:.3f}")

            if 'sql_data' in result and not result['sql_data'].empty:
                print(f"SQL data: {len(result['sql_data'])} rows")
                print(result['sql_data'].head())

            if 'visualization' in result:
                print(f"Visualization: {result['visualization']}")

            if 'llm_response' in result:
                print(f"\nLLM Response:")
                print(result['llm_response'])

            print("\n" + "-" * 60)

def main():
    """Initialize and run the comprehensive RAG system"""
    from dotenv import load_dotenv
    load_dotenv()
    print("Comprehensive RAG System Initialization")
    print("=" * 60)

    # Get Groq API key from environment or user
    groq_api_key = os.environ.get('GROQ_API_KEY')
    if not groq_api_key:
        print("WARNING: GROQ_API_KEY not found in environment variables")
        print("LLM responses will be disabled")
    else:
        print(f"Found GROQ_API_KEY: {groq_api_key[:5]}...")

    # Initialize system
    rag_system = ComprehensiveRAGSystem(groq_api_key)

    # Load all semantic samples
    samples = rag_system.load_all_semantic_samples()

    # Create new ChromaDB
    collection = rag_system.create_new_chromadb()

    print(f"\nSystem ready!")
    print(f"ChromaDB path: {rag_system.chroma_path}")
    print(f"Total samples: {len(samples)}")

    # Start interactive mode
    rag_system.interactive_mode()

if __name__ == "__main__":
    main()