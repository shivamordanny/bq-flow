"""
BQ Flow: Setup - Streamlit Application
A clean interface for generating rich embeddings for BigQuery datasets
"""

import streamlit as st
import pandas as pd
import yaml
import time
from datetime import datetime
import sys
import os
import json
from pathlib import Path
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")

# Add core modules to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load .env file
load_dotenv()

try:
    from core.bigquery_client import BigQueryClient
    from core.profiler import DataProfiler
    from core.embeddings import EmbeddingsGenerator
    from core.ai_assistant import AIAnalyst
    from core.logger import app_logger, profiler_logger, ai_assistant_logger
except ImportError:
    # Fallback imports if running from different context
    import core.bigquery_client as bigquery_client
    import core.profiler as profiler
    import core.embeddings as embeddings
    import core.ai_assistant as ai_assistant
    import core.logger as logger

    BigQueryClient = bigquery_client.BigQueryClient
    DataProfiler = profiler.DataProfiler
    EmbeddingsGenerator = embeddings.EmbeddingsGenerator
    AIAnalyst = ai_assistant.AIAnalyst
    app_logger = logger.app_logger
    profiler_logger = logger.profiler_logger
    ai_assistant_logger = logger.ai_assistant_logger

# Page configuration
st.set_page_config(
    page_title="BQ Flow: Data Onboarding & AI Training",
    page_icon="images/favicon.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load configuration
def load_config():
    """Load configuration from main config file and environment"""
    # Load main config
    # Use absolute path based on the script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    config_path = os.path.join(parent_dir, 'config', 'config.yaml')
    app_logger.log_operation("CONFIG_LOAD_START", {
        "config_path": config_path,
        "script_dir": script_dir,
        "parent_dir": parent_dir,
        "cwd": os.getcwd()
    })

    if os.path.exists(config_path):
        app_logger.log_operation("CONFIG_FILE_FOUND", {"path": config_path})
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                app_logger.log_operation("CONFIG_LOADED", {
                    "keys": list(config.keys())[:10],
                    "total_keys": len(config.keys())
                })
        except yaml.YAMLError as e:
            app_logger.log_error("CONFIG_YAML_ERROR", e, {"config_path": config_path})
            st.error(f"Error parsing config.yaml: {e}")
            config = {}
        except Exception as e:
            app_logger.log_error("CONFIG_LOAD_ERROR", e, {"config_path": config_path})
            st.error(f"Error loading config.yaml: {e}")
            config = {}
    else:
        app_logger.log_operation("CONFIG_FILE_NOT_FOUND", {"path": config_path}, level="WARNING")
        config = {}

    # Check if embedding_models exists in config
    if 'embedding_models' not in config:
        # Add default embedding_models configuration
        config['embedding_models'] = {
            'default': 'text-embedding-005',
            'text-embedding-005': {
                'model_name': 'text-embedding-005',
                'dimensions': 768,
                'task_type': 'RETRIEVAL_QUERY',
                'batch_size': 250,
                'cost_per_1k_tokens': 0.01,
                'max_tokens': 2048,
                'description': 'Enterprise model with enhanced performance'
            },
            'text-multimodal-embedding-001': {
                'model_name': 'text-multimodal-embedding-001',
                'dimensions': 3072,
                'batch_size': 50,
                'cost_per_1k_tokens': 0.02,
                'max_tokens': 2048,
                'description': 'Premium model with highest accuracy'
            }
        }

    # Load databases from separate file if specified
    if 'databases_file' in config:
        db_path = os.path.join(parent_dir, 'config', config['databases_file'])
        app_logger.log_operation("DATABASE_FILE_CHECK", {
            "databases_file": config['databases_file'],
            "full_path": db_path,
            "exists": os.path.exists(db_path)
        })

        if os.path.exists(db_path):
            try:
                with open(db_path, 'r') as f:
                    databases_config = yaml.safe_load(f)
                    app_logger.log_operation("DATABASE_FILE_LOADED", {
                        "path": db_path,
                        "keys": list(databases_config.keys()) if databases_config else [],
                        "has_bq_public": 'bq_public' in databases_config if databases_config else False,
                        "has_custom": 'custom' in databases_config if databases_config else False
                    })

                    # Merge BQ public and custom databases
                    bq_public = databases_config.get('bq_public', {}) if databases_config else {}
                    custom = databases_config.get('custom', {}) if databases_config else {}

                    app_logger.log_operation("DATABASE_MERGE", {
                        "bq_public_count": len(bq_public),
                        "custom_count": len(custom),
                        "total_databases": len(bq_public) + len(custom)
                    })

                    config['databases'] = {
                        **bq_public,
                        **custom
                    }
                    config['profiling_strategies'] = databases_config.get('profiling_strategies', {}) if databases_config else {}
            except yaml.YAMLError as e:
                app_logger.log_error("DATABASE_YAML_ERROR", e, {"db_path": db_path})
                st.error(f"Error parsing databases.yaml: {e}")
            except Exception as e:
                app_logger.log_error("DATABASE_LOAD_ERROR", e, {"db_path": db_path})
                st.error(f"Error loading databases.yaml: {e}")
        else:
            app_logger.log_operation("DATABASE_FILE_NOT_FOUND", {
                "databases_file": config['databases_file'],
                "path": db_path
            }, level="ERROR")
            st.error(f"⚠️ databases.yaml file not found at: {db_path}")
    else:
        app_logger.log_operation("NO_DATABASE_FILE_CONFIG", {
            "config_keys": list(config.keys())[:10]
        }, level="WARNING")

    # Ensure databases key exists even if databases file wasn't loaded
    if 'databases' not in config:
        config['databases'] = {}
        app_logger.log_operation("DATABASES_DEFAULT_INIT", {"reason": "No databases loaded"}, level="WARNING")
    else:
        app_logger.log_operation("DATABASES_SUMMARY", {
            "total_databases": len(config['databases']),
            "database_ids": list(config['databases'].keys())[:5] if config['databases'] else []
        })

    # Ensure profiling_strategies exists
    if 'profiling_strategies' not in config:
        config['profiling_strategies'] = {}
        app_logger.log_operation("PROFILING_STRATEGIES_DEFAULT_INIT", {}, level="INFO")

    # Add required environment variables
    config['bigquery'] = config.get('bigquery', {})
    config['bigquery']['project_id'] = os.getenv('PROJECT_ID')
    config['bigquery']['dataset_id'] = os.getenv('DATASET_ID')
    config['bigquery']['location'] = os.getenv('LOCATION', config.get('bigquery', {}).get('location', 'US'))

    # Validate required variables
    if not config['bigquery']['project_id']:
        st.error("❌ PROJECT_ID environment variable is not set. Please set it in your .env file.")
        st.stop()
    if not config['bigquery']['dataset_id']:
        st.error("❌ DATASET_ID environment variable is not set. Please set it in your .env file.")
        st.stop()

    # Validate configuration structure
    validation_errors = []
    if 'databases' in config and config['databases']:
        # Validate database structure
        for db_id, db_info in config['databases'].items():
            if not isinstance(db_info, dict):
                validation_errors.append(f"Database '{db_id}' is not a dictionary")
                continue

            # Check required fields
            required_fields = ['display_name', 'project_id', 'dataset_name', 'description']
            for field in required_fields:
                if field not in db_info:
                    validation_errors.append(f"Database '{db_id}' missing required field: {field}")

    if validation_errors:
        app_logger.log_operation("CONFIG_VALIDATION_ERRORS", {
            "errors": validation_errors,
            "error_count": len(validation_errors)
        }, level="ERROR")
        st.warning(f"Configuration validation found {len(validation_errors)} issues. Check logs for details.")
    else:
        app_logger.log_operation("CONFIG_VALIDATION_SUCCESS", {
            "databases_count": len(config.get('databases', {}))
        })

    return config

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.bq_client = None
    st.session_state.profiler = None
    st.session_state.embeddings_gen = None
    st.session_state.current_job = None
    st.session_state.job_history = []

# Initialize clients
@st.cache_resource
def init_clients():
    config = load_config()
    bq_config = config['bigquery']

    client = BigQueryClient(
        project_id=bq_config['project_id'],
        dataset_id=bq_config['dataset_id']
    )

    # Create tables if they don't exist
    client.create_tables_if_not_exist()

    profiler = DataProfiler()
    embeddings_gen = EmbeddingsGenerator(
        client.client,
        project_id=bq_config['project_id'],
        dataset_id=bq_config['dataset_id']
    )

    return client, profiler, embeddings_gen

# Main Application
def main():
    st.title("🚀 BQ Flow: Data Onboarding & AI Training")
    st.markdown("*Transform your BigQuery datasets into AI-ready knowledge bases*")
    st.markdown("**Generate rich embeddings for BigQuery public datasets using ML.GENERATE_EMBEDDING**")

    # Initialize clients
    if not st.session_state.initialized:
        with st.spinner("Initializing BigQuery connection..."):
            try:
                client, profiler, embeddings_gen = init_clients()
                st.session_state.bq_client = client
                st.session_state.profiler = profiler
                st.session_state.embeddings_gen = embeddings_gen
                st.session_state.initialized = True
                st.success("✅ Successfully connected to BigQuery")
            except Exception as e:
                st.error(f"Failed to initialize: {e}")
                st.stop()

    # Sidebar Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["📊 Dashboard", "⚙️ Configuration", "🚀 Workflow", "📈 Monitoring"]
    )

    if page == "📊 Dashboard":
        show_dashboard()
    elif page == "⚙️ Configuration":
        show_configuration()
    elif page == "🚀 Workflow":
        show_workflow()
    elif page == "📈 Monitoring":
        show_monitoring()

def show_dashboard():
    """Dashboard page showing overview of all databases"""
    st.header("📊 Dashboard")

    # Add sync button at the top right
    col_title, col_sync = st.columns([5, 1])
    with col_title:
        st.markdown("**Live embedding statistics across all databases**")
    with col_sync:
        sync_button = st.button("🔄 Sync Stats", help="Refresh all embedding statistics from BigQuery")

    client = st.session_state.bq_client

    # Force refresh if sync button clicked
    if sync_button:
        with st.spinner("Syncing statistics from BigQuery..."):
            # Clear any cached data
            if hasattr(client, '_cache'):
                client._cache = {}
            databases_df = client.get_registered_databases()
            # Update embedding counts for each database
            for db_id in databases_df['database_id']:
                try:
                    client._update_database_stats(db_id)
                except:
                    pass
            # Refresh after update
            databases_df = client.get_registered_databases()
            st.success("✅ Statistics synced successfully!")
            st.rerun()
    else:
        # Get registered databases
        databases_df = client.get_registered_databases()

    if databases_df.empty:
        st.info("No databases registered yet. Go to Configuration to add databases.")
        return

    # Overview metrics with better formatting
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Databases", len(databases_df))

    with col2:
        total_tables = databases_df['table_count'].sum()
        st.metric("Total Tables", int(total_tables) if pd.notna(total_tables) else 0)

    with col3:
        total_columns = databases_df['column_count'].sum()
        st.metric("Total Columns", int(total_columns) if pd.notna(total_columns) else 0)

    with col4:
        total_embeddings = databases_df['total_embeddings'].sum()
        st.metric("Total Embeddings", int(total_embeddings) if pd.notna(total_embeddings) else 0)

    # Vector Index Section - One-click creation when threshold is met
    if total_embeddings >= 5000:
        st.success(f"✅ **Vector Index Ready!** You have {int(total_embeddings)} embeddings across all databases (≥5000 required)")

        # Check if index already exists
        embeddings_gen = st.session_state.embeddings_gen

        # Check if the new method exists (handles cached instances)
        if hasattr(embeddings_gen, 'check_vector_index_status'):
            index_status = embeddings_gen.check_vector_index_status()
        else:
            # Fallback for cached instances without the new method
            st.warning("⚠️ Please restart the app to enable new vector index features")
            index_status = {'exists': False}

        if index_status.get('exists'):
            st.info(f"🎯 **Vector Index Active**: {index_status.get('index_name')} with {index_status.get('rows_indexed', 0)} rows indexed")

            # Show index details in expander
            with st.expander("Vector Index Details", expanded=False):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Index Status", index_status.get('status', 'Unknown'))
                with col2:
                    st.metric("Rows Indexed", f"{index_status.get('rows_indexed', 0):,}")
                with col3:
                    size_mb = index_status.get('size_bytes', 0) / (1024 * 1024)
                    st.metric("Index Size", f"{size_mb:.2f} MB")

                if index_status.get('creation_time'):
                    st.caption(f"Created: {index_status.get('creation_time')}")
        else:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.warning("⚡ **Vector Index Not Created** - Create it now for 10-100x faster searches!")
            with col2:
                if st.button("🚀 Create Vector Index", type="primary", use_container_width=True):
                    with st.spinner("Creating vector index... This may take a few minutes..."):
                        if hasattr(embeddings_gen, 'create_vector_index'):
                            result = embeddings_gen.create_vector_index()
                        else:
                            result = {'success': False, 'message': 'Please restart the app to enable vector index creation'}

                        if result['success']:
                            st.success(result['message'])
                            st.balloons()
                            st.rerun()
                        else:
                            st.error(result['message'])
                            if 'error' in result:
                                st.caption(f"Error details: {result['error']}")
    else:
        # Show progress towards vector index
        progress = (total_embeddings / 5000) * 100 if total_embeddings > 0 else 0
        remaining = 5000 - int(total_embeddings)

        st.info(f"📊 **Vector Index Progress**: {int(total_embeddings)}/5000 embeddings ({progress:.1f}%)")
        st.progress(progress / 100, text=f"Need {remaining} more embeddings to enable vector index")

        with st.expander("ℹ️ About Vector Index", expanded=False):
            st.markdown("""
            **Why 5000 embeddings?**
            - BigQuery requires minimum 5000 rows for IVF index creation
            - Each database you add contributes to the total
            - Once created, ALL databases benefit from faster searches

            **How to reach 5000?**
            - Add more databases from Configuration
            - Process existing databases in Workflow
            - Small datasets contribute to the cumulative total

            **Performance Impact:**
            - Without index: ML.DISTANCE (slower but functional)
            - With index: VECTOR_SEARCH (10-100x faster)
            """)

    # Database details table with enhanced display
    st.subheader("📦 Registered Databases")

    # Format the dataframe for display
    display_df = databases_df.copy()

    # Ensure columns used in arithmetic are numeric (coerce errors to NaN)
    total = pd.to_numeric(display_df['total_embeddings'], errors='coerce')
    cols = pd.to_numeric(display_df['column_count'], errors='coerce')

    # Calculate coverage, keep it numeric, then fillna and round
    coverage = (total / cols * 100)
    display_df['Coverage %'] = coverage.fillna(0).astype(float).round(1)

    # Add status indicator based on coverage
    def get_status_indicator(cov):
        if pd.isna(cov) or cov == 0:
            return "🔴 Not Started"
        elif cov < 50:
            return "🟡 In Progress"
        elif cov < 100:
            return "🟢 Partial"
        else:
            return "✅ Complete"

    display_df['Status'] = display_df['Coverage %'].apply(get_status_indicator)

    # Display with custom formatting
    st.dataframe(
        display_df[['database_id', 'display_name', 'Status', 'table_count', 'column_count',
                'total_embeddings', 'Coverage %', 'last_profiled_at', 'last_embedded_at']],
        width='stretch',
        hide_index=True,
        column_config={
            "Coverage %": st.column_config.ProgressColumn(
                "Coverage %",
                help="Percentage of columns with embeddings",
                min_value=0,
                max_value=100,
                format="%.1f%%"
            ),
            "last_profiled_at": st.column_config.DatetimeColumn(
                "Last Profiled",
                format="DD MMM HH:mm"
            ),
            "last_embedded_at": st.column_config.DatetimeColumn(
                "Last Embedded",
                format="DD MMM HH:mm"
            )
        }
    )

    # Quick stats per database with enhanced display
    st.subheader("📊 Database Details")

    col_select, col_refresh = st.columns([5, 1])
    with col_select:
        selected_db = st.selectbox("Select Database", databases_df['database_id'].tolist())
    with col_refresh:
        refresh_db = st.button("🔄", help="Refresh stats for this database")

    if selected_db:
        if refresh_db:
            with st.spinner(f"Refreshing stats for {selected_db}..."):
                try:
                    # Update stats for this specific database
                    st.session_state.embeddings_gen._update_database_stats(selected_db)
                    st.success("✅ Updated!")
                    st.rerun()
                except:
                    pass

        stats = client.get_enriched_metadata_count(selected_db)

        # Display metrics with better formatting
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("📋 Tables", stats['table_count'])
        with col2:
            st.metric("📝 Columns", stats['column_count'])
        with col3:
            st.metric("🔍 Enriched", stats['total_rows'])
        with col4:
            st.metric("🎯 Embedded", stats['embedded_count'])
        with col5:
            coverage = (stats['embedded_count'] / stats['total_rows'] * 100) if stats['total_rows'] > 0 else 0
            if coverage == 0:
                delta_color = "off"
            elif coverage < 50:
                delta_color = "normal"
            else:
                delta_color = "normal"
            st.metric("📊 Coverage", f"{coverage:.1f}%",
                     delta=f"{stats['embedded_count']}/{stats['total_rows']}" if stats['total_rows'] > 0 else "0/0",
                     delta_color=delta_color)

        # Show progress bar for coverage
        if stats['total_rows'] > 0:
            st.progress(coverage / 100, text=f"Embedding Progress: {stats['embedded_count']} of {stats['total_rows']} columns")

        # Show last update times if available
        db_info = databases_df[databases_df['database_id'] == selected_db].iloc[0]
        if pd.notna(db_info['last_profiled_at']) or pd.notna(db_info['last_embedded_at']):
            st.divider()
            col1, col2 = st.columns(2)
            with col1:
                if pd.notna(db_info['last_profiled_at']):
                    st.caption(f"Last profiled: {db_info['last_profiled_at']}")
            with col2:
                if pd.notna(db_info['last_embedded_at']):
                    st.caption(f"Last embedded: {db_info['last_embedded_at']}")

def show_configuration():
    """Configuration page for database and model settings"""
    st.header("⚙️ Configuration")

    config = load_config()
    client = st.session_state.bq_client

    tabs = st.tabs(["📁 Databases", "🤖 Models", "🔧 Settings"])

    with tabs[0]:  # Databases tab
        st.subheader("Database Configuration")

        # Add new database
        with st.expander("➕ Add New Database", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                db_id = st.text_input("Database ID", placeholder="my_database")
                project_id = st.text_input("Project ID", value="bigquery-public-data")
                dataset_name = st.text_input("Dataset Name", placeholder="dataset_name")

            with col2:
                display_name = st.text_input("Display Name", placeholder="My Database")
                sample_size = st.number_input("Sample Size", min_value=100, max_value=10000, value=1000)
                # Get available models and set default
                available_models = list(config['embedding_models'].keys())
                default_model = config['embedding_models']['default']
                default_index = available_models.index(default_model) if default_model in available_models else 0
                model = st.selectbox("Embedding Model", available_models, index=default_index)

            description = st.text_area("Description", placeholder="Database description...")

            if st.button("Register Database", type="primary"):
                if db_id and project_id and dataset_name:
                    db_config = {
                        'database_id': db_id,
                        'display_name': display_name or db_id,
                        'project_id': project_id,
                        'dataset_name': dataset_name,
                        'description': description,
                        'sample_size': sample_size,
                        'embedding_model': model,
                        'profiling_strategy': 'auto'
                    }

                    if client.register_database(db_config):
                        st.success(f"✅ Database '{db_id}' registered successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to register database")
                else:
                    st.error("Please fill in all required fields")

        # Pre-configured databases
        st.subheader("Quick Add: Pre-configured Public Databases")

        # Debug information in expander
        with st.expander("🔍 Debug Configuration Info"):
            st.write("**Configuration Paths:**")
            script_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(script_dir)
            config_path = os.path.join(parent_dir, 'config', 'config.yaml')
            st.write(f"- Config path: `{config_path}`")
            st.write(f"- Config exists: {os.path.exists(config_path)}")

            # Check config file permissions
            if os.path.exists(config_path):
                st.write(f"- Config readable: {os.access(config_path, os.R_OK)}")
                stat_info = os.stat(config_path)
                st.write(f"- Config size: {stat_info.st_size} bytes")

            if 'databases_file' in config:
                db_path = os.path.join(parent_dir, 'config', config['databases_file'])
                st.write(f"- Database file configured: `{config['databases_file']}`")
                st.write(f"- Database file path: `{db_path}`")
                st.write(f"- Database file exists: {os.path.exists(db_path)}")

                # Check database file permissions and content
                if os.path.exists(db_path):
                    st.write(f"- Database file readable: {os.access(db_path, os.R_OK)}")
                    stat_info = os.stat(db_path)
                    st.write(f"- Database file size: {stat_info.st_size} bytes")

                    # Try to read and show first few lines
                    try:
                        with open(db_path, 'r') as f:
                            first_lines = f.read(500)
                            st.write("\n**First 500 chars of databases.yaml:**")
                            st.code(first_lines, language='yaml')
                    except Exception as e:
                        st.error(f"Could not read databases.yaml: {e}")
            else:
                st.write("- Database file: Not configured in config.yaml")
                st.write("  Check if 'databases_file' key exists in config.yaml")

            st.write("\n**Configuration Content:**")
            st.write(f"- Config has 'databases' key: {'databases' in config}")
            st.write(f"- Config has 'databases_file' key: {'databases_file' in config}")

            if 'databases_file' in config:
                st.write(f"- databases_file value: `{config['databases_file']}`")

            if 'databases' in config:
                st.write(f"- Number of databases loaded: {len(config['databases'])}")
                if config['databases']:
                    st.write(f"- Sample database IDs: {list(config['databases'].keys())[:5]}")
                    # Show structure of first database
                    first_db = list(config['databases'].keys())[0]
                    st.write(f"\n**Sample database structure ({first_db}):**")
                    st.json(config['databases'][first_db])
                else:
                    st.warning("databases key exists but is empty - check YAML structure")
            else:
                st.error("No 'databases' key in config - loading failed")

            st.write(f"\n**All config keys:** {list(config.keys())}")

            # Check for common issues
            st.write("\n**Common Issues Check:**")
            issues = []
            if not os.path.exists(config_path):
                issues.append("❌ config.yaml not found")
            if 'databases_file' not in config:
                issues.append("❌ databases_file not configured in config.yaml")
            elif not os.path.exists(db_path):
                issues.append("❌ databases.yaml file not found at expected path")
            if 'databases' not in config or not config['databases']:
                issues.append("❌ No databases loaded from file")

            if issues:
                for issue in issues:
                    st.write(issue)
            else:
                st.success("✅ All configuration checks passed")

        pre_configs = config.get('databases', {})

        # Only show error if the file doesn't exist or couldn't be loaded
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        db_file_path = os.path.join(parent_dir, 'config', 'databases.yaml')
        if not os.path.exists(db_file_path):
            st.error(f"⚠️ databases.yaml file not found at: {db_file_path}")
            st.info("Please ensure the file exists in the config/ directory.")
            app_logger.log_operation("UI_DATABASE_FILE_MISSING", {"path": db_file_path}, level="ERROR")
        elif not pre_configs:
            st.warning("⚠️ No databases loaded from configuration. Check the Debug Configuration Info above for details.")
            app_logger.log_operation("UI_NO_DATABASES_LOADED", {
                "config_has_databases": 'databases' in config,
                "config_has_databases_file": 'databases_file' in config,
                "file_exists": os.path.exists(db_file_path)
            }, level="WARNING")

        registered_df = client.get_registered_databases()
        registered = registered_df['database_id'].tolist() if not registered_df.empty else []

        # Group databases by category (show all databases)
        categories = {}
        for db_id, db_info in pre_configs.items():
            category = db_info.get('category', 'Other')
            if category not in categories:
                categories[category] = []
            categories[category].append((db_id, db_info, db_id in registered))

        if categories:
            # Show database count summary
            total_databases = sum(len(dbs) for dbs in categories.values())
            registered_count = sum(1 for cat in categories.values() for _, _, is_reg in cat if is_reg)
            unregistered_count = total_databases - registered_count

            st.info(f"📚 {total_databases} databases available across {len(categories)} categories | ✅ {registered_count} added | ➕ {unregistered_count} available to add")

            # Display by category with expandable sections
            for category in sorted(categories.keys()):
                with st.expander(f"**{category}** ({len(categories[category])} datasets)", expanded=False):
                    for db_id, preset, is_registered in categories[category]:
                        col1, col2, col3 = st.columns([3, 5, 2])

                        with col1:
                            st.write(f"**{preset['display_name']}**")
                            st.caption(f"ID: `{db_id}`")

                        with col2:
                            st.write(preset['description'][:100] + "...")
                            if 'highlights' in preset:
                                st.caption("Key features: " + ", ".join(preset['highlights'][:2]))

                        with col3:
                            if is_registered:
                                st.button("✅ Added", key=f"added_{db_id}", disabled=True, use_container_width=True)
                            else:
                                if st.button(f"➕ Add", key=f"add_{db_id}", use_container_width=True):
                                    db_config = {
                                        'database_id': db_id,
                                        'display_name': preset['display_name'],
                                        'project_id': preset['project_id'],
                                        'dataset_name': preset['dataset_name'],
                                        'description': preset['description'],
                                        'sample_size': preset['sample_size'],
                                        'embedding_model': config['embedding_models']['default'],
                                        'profiling_strategy': preset.get('profiling_strategy', 'generic')
                                    }

                                    if client.register_database(db_config):
                                        st.success(f"✅ Added '{db_id}' successfully!")
                                        st.rerun()
                                    else:
                                        st.error(f"Failed to add '{db_id}'")
        else:
            st.warning("No pre-configured databases available. Check the Debug Configuration Info above for details.")

    with tabs[1]:  # Models tab
        st.subheader("Embedding Model Configuration")

        models = config['embedding_models']
        default_model = models['default']

        # Model comparison table
        model_data = []
        for name, cfg in models.items():
            if name != 'default':
                model_data.append({
                    'Model': name,
                    'Dimensions': cfg['dimensions'],
                    'Batch Size': cfg['batch_size'],
                    'Cost per 1K': f"${cfg['cost_per_1k_tokens']}",
                    'Description': cfg['description']
                })

        st.dataframe(pd.DataFrame(model_data), width='stretch', hide_index=True)

        # Model selection
        selected_model = st.selectbox(
            "Select Default Model",
            [k for k in models.keys() if k != 'default'],
            index=0
        )

        if st.button("Set as Default"):
            st.session_state.embeddings_gen.set_model(selected_model)
            st.success(f"✅ Default model set to: {selected_model}")

        # Test embedding generation
        st.subheader("Test Embedding Generation")
        test_text = st.text_input("Test Text", value="Sample column with customer purchase data")

        if st.button("Generate Test Embedding"):
            with st.spinner("Generating embedding..."):
                if st.session_state.embeddings_gen.test_embedding_generation(test_text):
                    st.success("✅ Embedding generation successful!")
                else:
                    st.error("❌ Embedding generation failed. Check model configuration.")

    with tabs[2]:  # Settings tab
        st.subheader("Processing Settings")

        col1, col2 = st.columns(2)

        with col1:
            st.number_input("Batch Size", min_value=10, max_value=500, value=100, key="batch_size")
            st.number_input("Sample Size", min_value=100, max_value=10000, value=1000, key="sample_size")
            st.number_input("Parallel Workers", min_value=1, max_value=10, value=3, key="workers")

        with col2:
            st.number_input("Retry Attempts", min_value=1, max_value=5, value=3, key="retries")
            st.number_input("Timeout (seconds)", min_value=60, max_value=600, value=300, key="timeout")
            st.checkbox("Enable Cost Tracking", value=True, key="cost_tracking")

def show_workflow():
    """Workflow page for running the embedding generation pipeline"""
    st.header("🚀 Data Onboarding Workflow")
    st.info("Follow these 4 steps to prepare your data for AI-powered natural language queries:")

    client = st.session_state.bq_client
    profiler = st.session_state.profiler
    embeddings_gen = st.session_state.embeddings_gen

    # Get registered databases
    databases_df = client.get_registered_databases()

    if databases_df.empty:
        st.warning("No databases registered. Please go to Configuration to add databases.")
        return

    # Database selection
    selected_db = st.selectbox(
        "Select Database to Process",
        databases_df['database_id'].tolist()
    )

    if selected_db:
        db_info = databases_df[databases_df['database_id'] == selected_db].iloc[0]

        # Display database info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"**Project**: {db_info['project_id']}")
        with col2:
            st.info(f"**Dataset**: {db_info['dataset_name']}")
        with col3:
            total_embeddings = db_info['total_embeddings'] if pd.notna(db_info['total_embeddings']) else 0
            column_count = db_info['column_count'] if pd.notna(db_info['column_count']) else 0
            current_coverage = (total_embeddings / column_count * 100) if column_count > 0 else 0
            st.info(f"**Current Coverage**: {current_coverage:.1f}%")

        # Workflow steps
        st.subheader("Workflow Steps")

        # Step 1: Database Discovery
        with st.expander("Step 1: Database Discovery", expanded=True):
            if st.button("🔍 Discover Tables and Columns", key="discover"):
                st.caption("*Scans the database schema to find all available data.*")
                with st.spinner("Discovering database schema..."):
                    job_id = client.create_job(selected_db, 'discovery')

                    # Discover tables
                    tables = client.discover_tables(db_info['project_id'], db_info['dataset_name'])
                    st.success(f"Found {len(tables)} tables")

                    # Discover columns for each table
                    total_columns = 0
                    for table in tables:  # Process ALL tables, no limits
                        columns = client.discover_columns(
                            db_info['project_id'],
                            db_info['dataset_name'],
                            table['table_name']
                        )
                        total_columns += len(columns)

                    st.success(f"Discovered {total_columns} columns across {len(tables)} tables")

                    # Update database registry
                    query = f"""
                    UPDATE `{client.full_dataset_id}.database_registry`
                    SET table_count = {len(tables)},
                        column_count = {total_columns}
                    WHERE database_id = '{selected_db}'
                    """
                    client.client.query(query).result()

                    client.complete_job(job_id, success=True)

        # Step 2: Data Profiling (Separate from AI Selection)
        with st.expander("Step 2: Data Profiling", expanded=True):
            st.info("Profile all columns in the database to understand their structure and content")

            sample_size = st.number_input("Sample Size", min_value=100, max_value=5000, value=1000, key="profile_sample")

            # Check if profiling results exist
            profiles_dir = Path("profiles")
            profiles_dir.mkdir(exist_ok=True)
            profile_path = profiles_dir / f"{selected_db}_profile.json"

            if profile_path.exists():
                st.success(f"✅ Existing profile found: {profile_path}")
                with open(profile_path, 'r') as f:
                    existing_profile = json.load(f)
                st.caption(f"Profile date: {existing_profile.get('timestamp', 'Unknown')}")
                st.caption(f"Total columns: {existing_profile.get('total_columns', 0)}")

            if st.button("📊 Profile Database", key="profile"):
                st.caption("*Analyzes data samples to understand content and statistics.*")
                with st.spinner("Profiling database..."):
                    app_logger.log_operation("PROFILING_START", {
                        "database_id": selected_db,
                        "sample_size": sample_size
                    })

                    job_id = client.create_job(selected_db, 'profiling')
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    # Get tables
                    tables = client.discover_tables(db_info['project_id'], db_info['dataset_name'])

                    # Get key tables from config if available, otherwise use all tables
                    config = load_config()
                    db_config = config['databases'].get(selected_db, {})
                    key_tables = db_config.get('key_tables', [])

                    if key_tables:
                        # Filter to only key tables if specified in config
                        tables_to_profile = [t for t in tables if t['table_name'] in key_tables]
                        if not tables_to_profile:
                            # If no key tables found, use all tables
                            tables_to_profile = tables
                    else:
                        # Use all discovered tables
                        tables_to_profile = tables

                    st.info(f"📊 Found {len(tables)} tables. Profiling {len(tables_to_profile)} tables...")

                    all_profiles = []
                    failed_tables = []
                    for idx, table in enumerate(tables_to_profile):
                        table_name = table['table_name']
                        status_text.text(f"Profiling table {idx+1}/{len(tables_to_profile)}: {table_name}")

                        try:
                            # Sample table data
                            sample_df = client.sample_table_data(
                                db_info['project_id'],
                                db_info['dataset_name'],
                                table_name,
                                sample_size
                            )

                            if not sample_df.empty:
                                # Profile the data
                                profiles = profiler.profile_dataframe(sample_df, table_name, selected_db)
                                all_profiles.extend(profiles)
                                app_logger.log_operation("TABLE_PROFILED", {
                                    "table": table_name,
                                    "columns_profiled": len(profiles)
                                })
                            else:
                                app_logger.log_operation("TABLE_SKIPPED", {
                                    "table": table_name,
                                    "reason": "Empty or invalid sample"
                                }, level="WARNING")
                                failed_tables.append(table_name)

                        except Exception as e:
                            app_logger.log_operation("TABLE_PROFILE_FAILED", {
                                "table": table_name,
                                "error": str(e)
                            }, level="ERROR")
                            failed_tables.append(table_name)
                            # Continue with next table instead of failing entirely

                        progress = (idx + 1) / len(tables_to_profile)
                        progress_bar.progress(progress)
                        client.update_job_progress(job_id, idx + 1, len(tables_to_profile))

                    # Save profiling results to JSON
                    if all_profiles:
                        st.success(f"✅ Profiled {len(all_profiles)} columns successfully!")

                        # Show warning if some tables failed
                        if failed_tables:
                            st.warning(f"⚠️ Failed to profile {len(failed_tables)} tables: {', '.join(failed_tables)}")
                            st.info("Check logs for details. Continuing with successfully profiled tables.")

                        # Save to JSON
                        profile_data = {
                            "database_id": selected_db,
                            "timestamp": datetime.now().isoformat(),
                            "sample_size": sample_size,
                            "total_columns": len(all_profiles),
                            "tables_count": len(tables_to_profile),
                            "profiles": all_profiles
                        }

                        with open(profile_path, 'w') as f:
                            json.dump(profile_data, f, indent=2, default=str)

                        st.success(f"📁 Saved profiling results to: {profile_path}")

                        # Log profiling results
                        profiler_logger.log_profiling(
                            selected_db,
                            "all_tables",
                            len(all_profiles)
                        )

                        # Store in session for next step
                        st.session_state.profile_data = profile_data
                    else:
                        st.warning("No columns were profiled")

                    client.complete_job(job_id, success=True)
                    status_text.text("Profiling complete!")

        # Step 3: AI Column Selection (Separate from Profiling)
        with st.expander("Step 3: AI Column Selection", expanded=True):
            st.info("Use AI to intelligently select the most valuable columns for embedding")

            # Check if profile exists
            if not profile_path.exists():
                st.warning("⚠️ Please complete Step 2 (Data Profiling) first")
                st.stop()

            # Load profile data
            with open(profile_path, 'r') as f:
                profile_data = json.load(f)

            st.caption(f"Profile loaded: {profile_data['total_columns']} columns from {profile_data['tables_count']} tables")

            col1, col2 = st.columns(2)
            with col1:
                max_columns_per_table = st.number_input("Max columns per table", min_value=5, max_value=100, value=50)
            with col2:
                min_columns_per_table = st.number_input("Min columns per table", min_value=1, max_value=20, value=3)

            # Check if selection already exists
            selection_path = profiles_dir / f"{selected_db}_selection.json"
            if selection_path.exists():
                st.success(f"✅ Existing selection found: {selection_path}")
                with open(selection_path, 'r') as f:
                    existing_selection = json.load(f)
                st.caption(f"Selection date: {existing_selection.get('timestamp', 'Unknown')}")
                st.caption(f"Selected columns: {existing_selection.get('total_selected', 0)}")

            if st.button("🤖 Run AI Selection", key="ai_select"):
                st.caption("*Uses Gemini to intelligently select the most analytically valuable columns, reducing cost and noise.*")
                with st.spinner("Running AI column selection..."):
                    app_logger.log_operation("AI_SELECTION_START", {
                        "database_id": selected_db,
                        "total_columns": profile_data['total_columns']
                    })

                    # Initialize AI assistant
                    ai_analyst = AIAnalyst(client)

                    # Group profiles by table
                    all_profiles = profile_data['profiles']
                    tables_profiles = {}
                    for profile in all_profiles:
                        table_name = profile['table_name']
                        if table_name not in tables_profiles:
                            tables_profiles[table_name] = []
                        tables_profiles[table_name].append(profile)

                    # Run AI selection for each table
                    selected_profiles = []
                    selection_details = {}
                    total_selected = 0
                    total_excluded = 0

                    progress_bar = st.progress(0)
                    for idx, (table_name, table_profiles) in enumerate(tables_profiles.items()):
                        st.text(f"Analyzing table {idx+1}/{len(tables_profiles)}: {table_name}")

                        selection_result = ai_analyst.select_key_columns(
                            table_name,
                            table_profiles,
                            max_columns=max_columns_per_table,
                            min_columns=min_columns_per_table
                        )

                        if selection_result['success']:
                            st.info(f"🤖 AI selected {len(selection_result['selected_columns'])} of {len(table_profiles)} columns from {table_name}")

                            # Mark selected profiles with is_selected=True
                            for profile in table_profiles:
                                if profile['column_name'] in selection_result['selected_columns']:
                                    profile['is_selected'] = True
                                    profile['selection_reason'] = selection_result.get('reason', 'AI selected as valuable')
                                    selected_profiles.append(profile)
                                else:
                                    profile['is_selected'] = False

                            selection_details[table_name] = {
                                "total": len(table_profiles),
                                "selected": len(selection_result['selected_columns']),
                                "excluded": len(selection_result['excluded_columns']),
                                "selected_columns": selection_result['selected_columns'],
                                "excluded_columns": selection_result['excluded_columns']
                            }

                            total_selected += len(selection_result['selected_columns'])
                            total_excluded += len(selection_result['excluded_columns'])

                            # Log AI selection
                            ai_assistant_logger.log_ai_selection(
                                table_name,
                                len(table_profiles),
                                len(selection_result['selected_columns']),
                                len(selection_result['excluded_columns'])
                            )

                            # Show details in expander
                            with st.expander(f"Selection Details for {table_name}", expanded=False):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write("**Selected Columns:**")
                                    for col in selection_result['selected_columns'][:20]:  # Limit display
                                        st.write(f"✅ {col}")
                                with col2:
                                    st.write("**Excluded Columns:**")
                                    for col in selection_result['excluded_columns'][:20]:  # Limit display
                                        st.write(f"❌ {col}")
                        else:
                            st.error(f"❌ AI selection failed for {table_name}. Please check Gemini connection (connection_id='us.gemini_connection')")

                        progress_bar.progress((idx + 1) / len(tables_profiles))

                    # Calculate cost savings
                    if total_selected > 0 or total_excluded > 0:
                        ai_analyst = AIAnalyst(client)
                        savings = ai_analyst.estimate_cost_savings(
                            total_selected + total_excluded,
                            total_selected,
                            0.01  # Cost per 1k tokens
                        )

                        st.success(f"""
                        ✅ **AI Column Selection Complete!**
                        - Selected: {total_selected} columns
                        - Excluded: {total_excluded} columns
                        - Reduction: {savings['reduction_percentage']:.1f}%
                        - Estimated savings: ${savings['savings']:.4f}
                        """)

                    # Save selection results to JSON
                    selection_data = {
                        "database_id": selected_db,
                        "timestamp": datetime.now().isoformat(),
                        "total_columns": profile_data['total_columns'],
                        "total_selected": total_selected,
                        "total_excluded": total_excluded,
                        "selection_details": selection_details,
                        "selected_profiles": selected_profiles,
                        "cost_savings": savings if (total_selected > 0 or total_excluded > 0) else None
                    }

                    with open(selection_path, 'w') as f:
                        json.dump(selection_data, f, indent=2, default=str)

                    st.success(f"📁 Saved selection results to: {selection_path}")

                    # Insert selected profiles to database with is_selected=True
                    client.insert_enriched_metadata(selected_profiles, selected_db, is_selected=True)

                    st.success(f"✅ Inserted {len(selected_profiles)} selected columns to database")

                    # Store in session state for embedding step
                    st.session_state.selected_profiles = selected_profiles
                    st.session_state.selection_data = selection_data

        # Step 4: Generate Embeddings (Renamed from Step 3)
        with st.expander("Step 4: Generate Embeddings", expanded=True):
            # Show information about selected columns
            if 'selected_profiles' in st.session_state:
                selected_count = len(st.session_state.selected_profiles)
                st.info(f"📊 Ready to generate embeddings for {selected_count} selected columns")
            else:
                st.info("ℹ️ Run profiling step first to select columns")

            # Check existing embeddings
            existing_query = f"""
            SELECT COUNT(*) as existing_count
            FROM `{client.full_dataset_id}.enriched_metadata`
            WHERE database_id = '{selected_db}'
              AND embedding IS NOT NULL
              AND ARRAY_LENGTH(embedding) > 0
            """
            existing_result = client.execute_query(existing_query)
            existing_count = existing_result[0]['existing_count'] if existing_result else 0

            if existing_count > 0:
                st.warning(f"⚠️ Database already has {existing_count} embeddings")

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("""
                    **Existing Embeddings Behavior:**
                    - Default: Skip columns with embeddings
                    - Force Regenerate: Delete ALL embeddings first
                    - Use force regenerate when:
                      - Semantic context has improved
                      - Switching embedding models
                      - Testing different strategies
                    """)
                with col2:
                    force_regenerate = st.checkbox(
                        "🔄 Force Regenerate (Delete existing embeddings)",
                        key="force_regenerate",
                        help="WARNING: This will delete ALL existing embeddings for this database before generating new ones"
                    )

                    if force_regenerate:
                        st.error("⚠️ WARNING: All existing embeddings will be deleted!")
            else:
                force_regenerate = False
                st.success(f"✅ No existing embeddings found - ready to generate")

            model = st.selectbox("Embedding Model", ["text-embedding-005", "gemini-embedding-001"], key="embed_model")
            batch_size = st.number_input("Batch Size", min_value=10, max_value=200, value=50, key="embed_batch")

            col1, col2, col3 = st.columns(3)
            with col1:
                generate_btn = st.button("🧬 Generate Embeddings", type="primary", key="generate")
                st.caption("*Creates vector embeddings using ML.GENERATE_EMBEDDING so the data can be understood by the AI.*")
            with col2:
                if existing_count > 0:
                    delete_btn = st.button("🗑️ Delete Embeddings Only", type="secondary", key="delete_embeddings")
                else:
                    delete_btn = False
            with col3:
                if existing_count > 0:
                    st.metric("Existing Embeddings", existing_count)

            if generate_btn:
                with st.spinner("Generating embeddings..."):
                    job_id = client.create_job(selected_db, 'embedding')
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    # Set model
                    embeddings_gen.set_model(model)

                    # Create model if needed
                    embeddings_gen.create_embedding_model(model)

                    # Generate embeddings with progress callback
                    def progress_callback(progress, message):
                        progress_bar.progress(progress / 100)
                        status_text.text(message)

                    stats = embeddings_gen.generate_embeddings_batch(
                        selected_db,
                        batch_size=batch_size,
                        progress_callback=progress_callback,
                        force_regenerate=force_regenerate if 'force_regenerate' in locals() else False
                    )

                    if stats['successful'] > 0:
                        success_msg = f"""
                        ✅ Embedding generation complete!
                        - Processed: {stats['total_processed']} records
                        - Successful: {stats['successful']}
                        - Failed: {stats['failed']}
                        - Duration: {stats['duration_seconds']:.1f} seconds
                        - Estimated cost: ${stats['estimated_cost']:.4f}
                        """
                        if stats.get('force_regenerated') and stats.get('deleted_count'):
                            success_msg += f"\n- Deleted before regeneration: {stats['deleted_count']} embeddings"
                        st.success(success_msg)
                    else:
                        st.error("Embedding generation failed")

                    client.complete_job(job_id, success=stats['successful'] > 0)

            # Handle delete button
            if delete_btn:
                with st.spinner("Deleting embeddings..."):
                    delete_result = embeddings_gen.delete_embeddings(selected_db)
                    if delete_result['success']:
                        if delete_result['deleted_count'] > 0:
                            remaining = delete_result.get('verified_remaining', 0)
                            st.success(f"""
                            ✅ Embeddings deletion completed!
                            - Deleted: {delete_result['deleted_count']} embeddings
                            - Remaining: {remaining} embeddings
                            - Database: {delete_result['database_id']}

                            You can now:
                            1. Re-profile with better semantic context
                            2. Switch to a different embedding model
                            3. Generate new embeddings
                            """)
                        else:
                            st.info("No embeddings found to delete.")
                        st.rerun()  # Refresh to update the count
                    else:
                        st.error(f"Failed to delete embeddings: {delete_result.get('error')}")

        # Step 5: Create Vector Index (Renamed from Step 4)
        with st.expander("Step 5: Create Vector Index", expanded=False):
            # Check total embeddings across ALL databases
            total_query = f"""
            SELECT COUNT(*) as total_embeddings
            FROM `{client.full_dataset_id}.enriched_metadata`
            WHERE embedding IS NOT NULL
            AND ARRAY_LENGTH(embedding) = 768
            """
            total_result = client.execute_query(total_query)
            total_embeddings = total_result[0]['total_embeddings'] if total_result else 0

            # Check if index already exists (with fallback for old cached instances)
            if hasattr(embeddings_gen, 'check_vector_index_status'):
                index_status = embeddings_gen.check_vector_index_status()
            else:
                # Fallback for cached instances without the new method
                st.warning("⚠️ Please restart the app to enable new vector index features")
                index_status = {'exists': False}

            if index_status.get('exists'):
                st.success(f"✅ **Vector Index Already Active!**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Index Name", index_status.get('index_name', 'Unknown'))
                with col2:
                    st.metric("Rows Indexed", f"{index_status.get('rows_indexed', 0):,}")
                with col3:
                    st.metric("Status", index_status.get('status', 'Unknown'))
            else:
                st.info(f"📊 **Total Embeddings Across All Databases**: {total_embeddings:,} (need 5,000+)")

                if total_embeddings >= 5000:
                    st.success("✅ You have enough embeddings to create a vector index!")

                    if st.button("🚀 Create Vector Index", key="create_index", type="primary"):
                        with st.spinner("Creating vector index... This may take a few minutes..."):
                            if hasattr(embeddings_gen, 'create_vector_index'):
                                result = embeddings_gen.create_vector_index()
                            else:
                                result = {'success': False, 'message': 'Please restart the app to enable vector index creation'}

                            if result['success']:
                                st.success(result['message'])
                                st.balloons()
                                st.info("All databases will now benefit from 10-100x faster searches!")
                            else:
                                st.error(result['message'])
                                if 'error' in result:
                                    with st.expander("Error Details", expanded=True):
                                        st.code(result['error'])
                else:
                    remaining = 5000 - total_embeddings
                    progress = (total_embeddings / 5000) * 100

                    st.warning(f"⚠️ Need {remaining} more embeddings to enable vector index")
                    st.progress(progress / 100, text=f"Progress: {progress:.1f}%")

                    st.markdown("""
                    **How to get more embeddings:**
                    1. Add more databases from Configuration tab
                    2. Process additional tables in existing databases
                    3. Each database contributes to the total count
                    """)

            # Show benefits of vector index
            with st.expander("📚 Vector Index Benefits", expanded=False):
                st.markdown("""
                **Performance Improvements:**
                - 🚀 10-100x faster semantic search
                - 📊 Efficient handling of large-scale data
                - 🎯 Better relevance scoring

                **How it works:**
                - Uses IVF (Inverted File) indexing for fast similarity search
                - Pre-filters by database_id for isolation
                - Automatically maintained as data changes

                **Current Status:**
                - Without index: Using ML.DISTANCE (functional but slower)
                - With index: Using VECTOR_SEARCH (optimized performance)
                """)

def show_monitoring():
    """Monitoring page for tracking jobs and performance"""
    st.header("📈 Monitoring")

    client = st.session_state.bq_client

    # Get recent jobs
    query = f"""
    SELECT
        job_id,
        database_id,
        job_type,
        status,
        total_items,
        processed_items,
        progress_percentage,
        start_time,
        end_time,
        duration_seconds,
        embeddings_generated,
        estimated_cost_usd
    FROM `{client.full_dataset_id}.embedding_jobs`
    ORDER BY created_at DESC
    LIMIT 50
    """

    jobs_df = client.execute_query_to_df(query)

    if jobs_df.empty:
        st.info("No jobs have been run yet.")
        return

    # Job statistics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_jobs = len(jobs_df)
        st.metric("Total Jobs", total_jobs)

    with col2:
        successful = len(jobs_df[jobs_df['status'] == 'completed'])
        st.metric("Successful", successful)

    with col3:
        failed = len(jobs_df[jobs_df['status'] == 'failed'])
        st.metric("Failed", failed)

    with col4:
        total_cost = jobs_df['estimated_cost_usd'].sum()
        st.metric("Total Cost", f"${total_cost:.2f}")

    # Job history table
    st.subheader("Recent Jobs")

    # Format the dataframe
    display_df = jobs_df.copy()

    # Ensure start_time is datetime then formatted string
    display_df['start_time'] = pd.to_datetime(display_df['start_time'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M')

    # Optionally ensure numeric columns are numeric (coerce errors to NaN)
    display_df['progress_percentage'] = pd.to_numeric(display_df.get('progress_percentage'), errors='coerce')
    display_df['duration_seconds'] = pd.to_numeric(display_df.get('duration_seconds'), errors='coerce')
    display_df['embeddings_generated'] = pd.to_numeric(display_df.get('embeddings_generated'), errors='coerce')

    # Status badges
    status_colors = {
        'completed': '🟢',
        'running': '🟡',
        'failed': '🔴',
        'pending': '⚪'
    }
    display_df['status'] = display_df['status'].apply(lambda x: f"{status_colors.get(x, '⚪')} {x}")

    st.dataframe(
        display_df[['job_id', 'database_id', 'job_type', 'status', 'progress_percentage',
                'duration_seconds', 'embeddings_generated', 'start_time']],
        width='stretch',
        hide_index=True
    )

    # Performance charts
    if not jobs_df.empty:
        st.subheader("Performance Metrics")

        col1, col2 = st.columns(2)

        with col1:
            # Job types distribution
            job_types = jobs_df['job_type'].value_counts()
            st.bar_chart(job_types)

        with col2:
            # Success rate by database
            db_success = jobs_df.groupby('database_id')['status'].apply(
                lambda x: (x == 'completed').sum() / len(x) * 100
            )
            st.bar_chart(db_success)

if __name__ == "__main__":
    main()