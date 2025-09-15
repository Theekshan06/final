#!/usr/bin/env python3
"""
ARGO RAG System - Streamlit Interface
Combines the functionality of both files into a Streamlit web app
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import time
import traceback
from pathlib import Path
from typing import Dict, List
import os
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
import duckdb

class ComprehensiveRAGSystem:
    """Complete RAG system with ChromaDB, Groq API, and visualization"""

    def __init__(self, groq_api_key: str = None):
        # Try to use a local or cached model, fallback to simple TF-IDF if needed
        self.embedding_model_name = "all-MiniLM-L6-v2"
        st.info(f"Loading embedding model: {self.embedding_model_name}")
        try:
            # Try with local files only first
            self.embedding_model = SentenceTransformer(self.embedding_model_name, local_files_only=True)
            st.success("Loaded model from local cache")
        except Exception as e:
            st.warning(f"Local model failed: {e}")
            try:
                # Try without local_files_only restriction
                st.info("Attempting to download model...")
                self.embedding_model = SentenceTransformer(self.embedding_model_name, trust_remote_code=True)
                st.success("Downloaded model successfully")
            except Exception as e2:
                st.error(f"Model download failed: {e2}")
                st.warning("Using fallback simple embedding approach...")
                self.embedding_model = None
                self.use_simple_embeddings = True

        # Initialize ChromaDB in new directory - use absolute path
        self.chroma_path = script_dir / "new_comprehensive_chromadb"
        self.client = chromadb.PersistentClient(path=self.chroma_path)
        self.collection_name = "comprehensive_argo_rag"
        self.collection = None

        # Initialize Groq API
        self.groq_client = None
        if groq_api_key:
            try:
                self.groq_client = Groq(api_key=groq_api_key)
                st.success("Groq API configured successfully")
            except Exception as e:
                st.error(f"Groq API configuration failed: {e}")

        # Data paths - use absolute paths relative to script location
        script_dir = Path(__file__).parent.absolute()
        self.semantic_samples_dir = script_dir / "semantic_samples"
        self.parquet_data_dir = script_dir / "parquet_data"

        # Loaded samples
        self.all_samples = []

        # SQL execution setup
        self.db_connection = self.setup_sql_execution()

    def setup_sql_execution(self):
        """Setup SQL execution environment for parquet data using DuckDB"""
        try:
            con = duckdb.connect(database=':memory:', read_only=False)
            
            # Register parquet files as virtual tables
            for name, path in {
                'profiles': str(self.parquet_data_dir / "profiles.parquet"),
                'measurements': str(self.parquet_data_dir / "measurements.parquet"),
                'floats': str(self.parquet_data_dir / "floats.parquet")
            }.items():
                if Path(path).exists():
                    con.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM parquet_scan('{path}')")
                    st.info(f"Registered '{name}' view from {path}")
                else:
                    st.warning(f"{path} not found")
            
            return con
        except Exception as e:
            st.error(f"Error setting up DuckDB: {e}")
            return None

    def load_all_semantic_samples(self):
        """Load all JSON files from semantic_samples directory"""
        samples_dir = Path(self.semantic_samples_dir)
        json_files = list(samples_dir.glob("*.json"))

        st.info(f"Found {len(json_files)} JSON files in {self.semantic_samples_dir}/")

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
                st.info(f"  - {json_file.name}: {len(samples)} samples")

            except Exception as e:
                st.error(f"ERROR loading {json_file}: {e}")

        self.all_samples = all_samples
        st.success(f"Total loaded samples: {len(all_samples)}")

        # Show category distribution
        categories = {}
        for sample in all_samples:
            if 'metadata' in sample and 'category' in sample['metadata']:
                cat = sample['metadata']['category']
                categories[cat] = categories.get(cat, 0) + 1

        if categories:
            st.info("Category distribution:")
            for cat, count in categories.items():
                st.info(f"  - {cat}: {count} samples")

        return all_samples

    def create_new_chromadb(self):
        """Create new ChromaDB collection with all samples"""
        st.info(f"Creating new ChromaDB collection: {self.collection_name}")

        # Delete existing collection if it exists
        try:
            self.client.delete_collection(self.collection_name)
            st.info("Deleted existing collection")
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
            st.info(f"Added batch {i//batch_size + 1}: {total_added}/{len(documents)} samples")

        st.success(f"Successfully created ChromaDB with {total_added} samples")
        return self.collection

    def semantic_search(self, query: str, n_results: int = 5) -> List[Dict]:
        """Perform semantic search using ChromaDB"""
        if not self.collection:
            st.error("ERROR: ChromaDB collection not initialized")
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
            st.error("ERROR: DuckDB connection not available")
            return pd.DataFrame()

        try:
            st.info(f"Executing SQL template: {sql_template[:200]}...")
            st.info(f"With parameters: {parameters}")
            # Replace parameters in SQL template
            sql_query = sql_template
            if parameters:
                for param, value in parameters.items():
                    placeholder = "{" + param + "}"
                    sql_query = sql_query.replace(placeholder, str(value))

            st.info(f"Executing SQL: {sql_query[:200]}...")

            # Execute query using DuckDB
            result_df = self.db_connection.execute(sql_query).fetchdf()

            st.info(f"Query returned {len(result_df)} rows")
            return result_df

        except Exception as e:
            st.error(f"SQL execution error: {e}")
            return pd.DataFrame()

    def query_groq_llm(self, prompt: str, context: str = "") -> str:
        """Query Groq API for LLM response"""
        if not self.groq_client:
            return "Groq API not configured. Please provide API key."

        try:
            full_prompt = f"""
You are an expert oceanographic data analyst. Use the following context to answer the user's question.

Context: {context}

User Question: {prompt}

Please provide a comprehensive answer based on the context provided. If the context contains SQL templates or visualization instructions, incorporate them into your response.
"""

            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": full_prompt,
                    }
                ],
                model="llama-3.1-8b-instant",  # Fast and capable model
            )

            return chat_completion.choices[0].message.content

        except Exception as e:
            return f"Groq API error: {e}"

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
            # Step 1: Semantic search
            search_results = self.semantic_search(user_query, n_results=3)
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

            # Step 3: Generate SQL intelligently using LLM
            sql_generated = self.generate_intelligent_sql(user_query, search_results)

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

            # Step 5: Generate LLM response
            context = f"Query: {user_query}\n"
            context += f"Best match: {best_match['id']} (similarity: {best_match['similarity']:.3f})\n"
            if result["sql_executed"]:
                context += f"SQL executed successfully, returned {len(result['sql_data'])} rows\n"
            if result["visualization_created"]:
                context += f"Visualization created: {result['plot_type']}\n"

            result["llm_response"] = self.query_groq_llm(user_query, context)

        except Exception as e:
            result["llm_response"] = f"Error processing query: {str(e)}"

        return result

    def llm_decide_plotting(self, user_query: str, matched_content: str) -> Dict:
        """Use LLM to decide if plotting is needed and what type"""
        if not self.groq_client:
            return {"needs_plotting": False, "plot_type": "none"}

        prompt = f"""
        Analyze this user query and determine if a visualization/plot is needed:

        User Query: "{user_query}"
        Matched Sample: {matched_content}

        Respond with JSON format:
        {{
            "needs_plotting": true/false,
            "plot_type": "line|scatter|heatmap|histogram|none",
            "reason": "brief explanation"
        }}

        Plot types:
        - line: for time series, depth profiles
        - scatter: for correlations, relationships
        - heatmap: for geographic data, temperature distributions
        - histogram: for distributions, frequency analysis
        - none: for simple data queries

        Look for keywords like: plot, graph, chart, heatmap, visualization, show, display
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
            # Fallback to simple keyword detection
            plot_keywords = ['plot', 'graph', 'chart', 'heatmap', 'visualization', 'show', 'display']
            needs_plot = any(keyword in user_query.lower() for keyword in plot_keywords)
            return {
                "needs_plotting": needs_plot,
                "plot_type": "line" if needs_plot else "none",
                "reason": f"LLM failed, used keyword detection: {str(e)}"
            }

    def create_visualization(self, data: List[Dict], plot_type: str, query: str) -> Dict:
        """Create interactive visualization from data using Plotly"""
        if not data:
            return {"type": "none", "html": "<p>No data to plot</p>"}

        try:
            import plotly.graph_objects as go
            import plotly.express as px

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
                html_plot = fig.to_html(include_plotlyjs='cdn')
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
        Generate a SQL query for ARGO oceanographic data based on the user's request.

        User Query: "{user_query}"

        Database Schema:
        {schema_info}

        Similar examples from database:
        {json.dumps(context_samples, indent=2)}

        Instructions:
        1. Create a SQL query that answers the user's question
        2. Use proper table joins (profiles JOIN measurements ON profiles.profile_id = measurements.profile_id)
        3. Add appropriate WHERE clauses and LIMIT to avoid massive results
        4. For temperature vs depth queries, use temperature and pressure columns
        5. Include QC filters (temperature_qc IN ('1', '2') for good data)
        6. IMPORTANT: Filter out invalid data - exclude NULL values and zeros
        7. For temperature: WHERE temperature IS NOT NULL AND temperature > 0
        8. For pressure: WHERE pressure IS NOT NULL AND pressure > 0
        9. Return only valid SQL without explanation

        SQL Query:
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

            # Basic validation
            if not sql_query.upper().startswith('SELECT'):
                return {"success": False, "error": "Generated query is not a SELECT statement"}

            return {"success": True, "sql": sql_query}

        except Exception as e:
            return {"success": False, "error": f"LLM SQL generation failed: {str(e)}"}

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

    def get_database_schema(self) -> str:
        """Get database schema information for LLM context"""
        schema_info = """
        Tables:
        1. profiles: profile_id, float_id, profile_date, latitude, longitude
        2. measurements: measurement_id, profile_id, pressure, temperature, salinity, temperature_qc, salinity_qc
        3. floats: float_id, wmo_number, deployment_date, deployment_latitude, deployment_longitude

        Common Query Patterns:
        - Join profiles and measurements on profile_id
        - Use temperature_qc IN ('1', '2') for good quality data
        - pressure represents depth (higher pressure = deeper)
        - Always use LIMIT to prevent large result sets
        """
        return schema_info

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="ARGO RAG System - Enhanced with New ChromaDB",
        page_icon="🌊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🌊 ARGO RAG System - Enhanced with New ChromaDB")
    
    # Initialize session state
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
        st.session_state.initialized = False
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # API Key input
        groq_api_key = st.text_input(
            "Groq API Key",
            type="password",
            value=os.environ.get("GROQ_API_KEY", ""),
            help="Enter your Groq API key for LLM integration"
        )
        
        # Initialize button
        if st.button("Initialize RAG System", type="primary"):
            with st.spinner("Initializing RAG system..."):
                try:
                    st.session_state.rag_system = ComprehensiveRAGSystem(groq_api_key)
                    st.session_state.rag_system.load_all_semantic_samples()
                    st.session_state.rag_system.create_new_chromadb()
                    st.session_state.initialized = True
                    st.success("RAG System initialized successfully!")
                except Exception as e:
                    st.error(f"Initialization failed: {str(e)}")
                    st.error(traceback.format_exc())
    
    # Main content area
    if not st.session_state.initialized:
        st.info("Please configure and initialize the RAG system using the sidebar.")
        st.markdown("""
        ### About this System
        
        This ARGO RAG (Retrieval Augmented Generation) system provides:
        
        - **Semantic Search**: Find similar oceanographic queries in the database
        - **SQL Generation**: Automatically generate SQL queries for ARGO data
        - **Data Visualization**: Create interactive plots of oceanographic data
        - **LLM Integration**: Get expert analysis using Groq's LLM API
        
        ### How to use:
        
        1. Enter your Groq API key in the sidebar
        2. Click "Initialize RAG System"
        3. Enter your oceanographic query in the main panel
        4. View results including SQL queries, data, and visualizations
        """)
        return
    
    # Query input
    st.header("Query the ARGO RAG System")
    user_query = st.text_input(
        "Enter your oceanographic query:",
        placeholder="e.g., show temperature anomalies, plot temperature vs depth, create heatmap",
        help="Ask about ARGO float data, oceanographic measurements, or request visualizations"
    )
    
    if st.button("Submit Query", type="primary") and user_query:
        with st.spinner("Processing your query..."):
            try:
                start_time = time.time()
                result = st.session_state.rag_system.simple_query(user_query)
                processing_time = time.time() - start_time
                
                # Display results
                st.header("Results")
                
                # Best match info
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Best Match")
                    st.write(f"**ID**: {result.get('best_match_id', 'N/A')}")
                    st.write(f"**Similarity**: {result.get('similarity', 0):.3f}")
                
                with col2:
                    st.subheader("Processing Info")
                    st.write(f"**Time**: {processing_time:.2f} seconds")
                    st.write(f"**SQL Executed**: {result.get('sql_executed', False)}")
                    st.write(f"**Visualization Created**: {result.get('visualization_created', False)}")
                
                # SQL Query
                st.subheader("SQL Query")
                st.code(result.get('sql', ''), language='sql')
                
                # SQL Data
                st.subheader("SQL Data")
                if result.get('sql_executed'):
                    if result.get('sql_data') and len(result['sql_data']) > 0:
                        st.write(f"Returned {len(result['sql_data'])} rows")
                        st.dataframe(pd.DataFrame(result['sql_data']))
                    else:
                        st.info("No data returned from the SQL query.")
                else:
                    st.info("SQL query not executed.")
                
                # Visualization
                if result.get('visualization_created') and result.get('plot_html'):
                    st.subheader("Visualization")
                    st.components.v1.html(
                        result['plot_html'], 
                        height=600, 
                        scrolling=True
                    )
                
                # LLM Response
                st.subheader("LLM Response")
                st.write(result.get('llm_response', ''))
                
            except Exception as e:
                st.error(f"Error processing query: {str(e)}")
                st.error(traceback.format_exc())

if __name__ == "__main__":
    main()