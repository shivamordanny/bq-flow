"""
Embeddings Generator Module for BQ Flow System
Uses BigQuery's ML.GENERATE_EMBEDDING for efficient embedding generation
"""

from google.cloud import bigquery
from typing import Dict, List, Any, Optional
import logging
import pandas as pd
from datetime import datetime
import time
from .logger import embeddings_logger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingsGenerator:
    """
    Generate embeddings using BigQuery's ML.GENERATE_EMBEDDING
    """

    def __init__(self, client: bigquery.Client, project_id: str = None,
                 dataset_id: str = None):
        # Import config to get defaults from config.yaml
        import os
        import yaml

        # Try to load config
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config', 'config.yaml')
        config = {}
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

        self.client = client

        # Get project_id and dataset_id from environment if not provided
        if not project_id:
            project_id = os.getenv('PROJECT_ID')
            if not project_id:
                # Try loading from config as fallback
                if os.path.exists(config_path) and config:
                    project_id = config.get('bigquery', {}).get('project_id')
                if not project_id:
                    raise ValueError("PROJECT_ID must be provided or set in environment variables")

        if not dataset_id:
            dataset_id = os.getenv('DATASET_ID')
            if not dataset_id:
                # Try loading from config as fallback
                if os.path.exists(config_path) and config:
                    dataset_id = config.get('bigquery', {}).get('dataset_id')
                if not dataset_id:
                    raise ValueError("DATASET_ID must be provided or set in environment variables")

        self.project_id = project_id
        self.dataset_id = dataset_id
        self.full_dataset_id = f"{self.project_id}.{self.dataset_id}"

        # Model configurations
        self.model_configs = {
            'text-embedding-005': {
                'model_path': f'{self.full_dataset_id}.embedding_model_005',
                'dimensions': 768,
                'batch_size': 250, # Increased batch size for efficiency
                'cost_per_1k': 0.01
            },
            'gemini-embedding-001': {
                'model_path': f'{self.full_dataset_id}.embedding_model_gemini',
                'dimensions': 3072,
                'batch_size': 50,
                'cost_per_1k': 0.02
            }
        }
        self.current_model = 'text-embedding-005'

    def set_model(self, model_name: str):
        """Set the embedding model to use"""
        if model_name in self.model_configs:
            self.current_model = model_name
            logger.info(f"Embedding model set to: {model_name}")
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def delete_embeddings(self, database_id: str) -> Dict:
        """
        Delete all embeddings for a specific database by setting them to an empty array.
        """
        try:
            count_query = f"""
            SELECT COUNT(*) as count FROM `{self.full_dataset_id}.enriched_metadata`
            WHERE database_id = @database_id AND embedding IS NOT NULL AND ARRAY_LENGTH(embedding) > 0
            """
            job_config = bigquery.QueryJobConfig(
                query_parameters=[bigquery.ScalarQueryParameter("database_id", "STRING", database_id)]
            )
            result = self.client.query(count_query, job_config=job_config).to_dataframe()
            count_before = result['count'].iloc[0] if not result.empty else 0

            if count_before == 0:
                logger.info(f"No embeddings to delete for database {database_id}")
                return {'success': True, 'deleted_count': 0}

            delete_query = f"""
            UPDATE `{self.full_dataset_id}.enriched_metadata`
            SET embedding = [], embedding_model = NULL, embedding_dimensions = NULL, last_updated = CURRENT_TIMESTAMP()
            WHERE database_id = @database_id AND embedding IS NOT NULL AND ARRAY_LENGTH(embedding) > 0
            """
            job = self.client.query(delete_query, job_config=job_config)
            job.result()

            logger.info(f"Successfully deleted {job.num_dml_affected_rows} embeddings for database {database_id}")
            self._update_database_stats(database_id)
            return {'success': True, 'deleted_count': int(job.num_dml_affected_rows)}

        except Exception as e:
            logger.error(f"Failed to delete embeddings for {database_id}: {e}")
            return {'success': False, 'error': str(e)}

    def generate_embeddings_batch(self, database_id: str, batch_size: int = 100,
                                 progress_callback=None, force_regenerate: bool = False) -> Dict:
        """
        Generate embeddings for all metadata in batches using a robust multi-step DML process.
        """
        start_time = time.time()
        stats = {'total_processed': 0, 'successful': 0, 'failed': 0, 'batches': 0}

        try:
            if force_regenerate:
                logger.info(f"Force regenerate requested for {database_id}, deleting existing embeddings.")
                self.delete_embeddings(database_id)

            metadata_query = f"""
            SELECT DISTINCT database_id, table_name, column_name, semantic_context
            FROM `{self.full_dataset_id}.enriched_metadata`
            WHERE database_id = @database_id
              AND is_selected = TRUE
              AND semantic_context IS NOT NULL
              AND (embedding IS NULL OR ARRAY_LENGTH(embedding) = 0)
            """
            job_config = bigquery.QueryJobConfig(
                query_parameters=[bigquery.ScalarQueryParameter("database_id", "STRING", database_id)]
            )
            metadata_df = self.client.query(metadata_query, job_config=job_config).to_dataframe()
            total_rows = len(metadata_df)

            if total_rows == 0:
                logger.info(f"No selected metadata to embed for database {database_id}")
                return stats

            logger.info(f"Found {total_rows} selected metadata records to embed for {database_id}")

            model_config = self.model_configs[self.current_model]
            batch_size = min(batch_size, model_config['batch_size'])

            for i in range(0, total_rows, batch_size):
                batch_df = metadata_df.iloc[i:i + batch_size]
                stats['batches'] += 1
                try:
                    self._process_embedding_batch(batch_df, model_config)
                    stats['successful'] += len(batch_df)
                    if progress_callback:
                        processed_count = i + len(batch_df)
                        progress = (processed_count / total_rows) * 100
                        progress_callback(progress, f"Processed {processed_count}/{total_rows} records")
                    logger.info(f"Batch {stats['batches']}: Successfully embedded {len(batch_df)} records")
                except Exception as e:
                    logger.error(f"Failed to process and embed batch {stats['batches']}: {e}")
                    stats['failed'] += len(batch_df)
                time.sleep(0.5)

            stats['total_processed'] = total_rows
            stats['duration_seconds'] = time.time() - start_time
            stats['estimated_cost'] = self._calculate_cost(stats['successful'], model_config)
            self._update_database_stats(database_id)
            logger.info(f"Embedding generation complete: {stats}")
            return stats

        except Exception as e:
            logger.error(f"Embedding generation failed for {database_id}: {e}")
            stats['error'] = str(e)
            return stats

    def _process_embedding_batch(self, batch_df: pd.DataFrame, model_config: Dict):
        """
        Processes a single batch using temporary tables to avoid DML conflicts.
        """
        if batch_df.empty:
            return

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        temp_input_table = f"{self.full_dataset_id}.temp_embedding_input_{timestamp}"
        temp_output_table = f"{self.full_dataset_id}.temp_embedding_output_{timestamp}"

        try:
            # Step 1: Load batch data into a temporary input table
            load_df = batch_df[['database_id', 'table_name', 'column_name', 'semantic_context']].copy()
            load_df['semantic_context'] = load_df['semantic_context'].astype(str)
            job_config = bigquery.LoadJobConfig(
                schema=[
                    bigquery.SchemaField("database_id", "STRING"),
                    bigquery.SchemaField("table_name", "STRING"),
                    bigquery.SchemaField("column_name", "STRING"),
                    bigquery.SchemaField("semantic_context", "STRING"),
                ]
            )
            self.client.load_table_from_dataframe(load_df, temp_input_table, job_config=job_config).result()

            # Step 2: Generate embeddings and store results in a temporary output table
            create_output_query = f"""
            CREATE TABLE `{temp_output_table}` AS
            SELECT database_id, table_name, column_name, ml_generate_embedding_result AS embedding
            FROM ML.GENERATE_EMBEDDING(
                MODEL `{model_config['model_path']}`,
                (SELECT database_id, table_name, column_name, semantic_context AS content FROM `{temp_input_table}`),
                STRUCT(TRUE AS flatten_json_output)
            )
            """
            self.client.query(create_output_query).result()

            # Step 3: Merge the generated embeddings from the output table into the main table
            merge_query = f"""
            MERGE `{self.full_dataset_id}.enriched_metadata` AS target
            USING `{temp_output_table}` AS source
            ON target.database_id = source.database_id
               AND target.table_name = source.table_name
               AND target.column_name = source.column_name
            WHEN MATCHED THEN
                UPDATE SET
                    embedding = source.embedding,
                    embedding_model = '{self.current_model}',
                    embedding_dimensions = {model_config['dimensions']},
                    last_updated = CURRENT_TIMESTAMP()
            """
            self.client.query(merge_query).result()

        finally:
            # Step 4: Clean up both temporary tables
            self.client.delete_table(temp_input_table, not_found_ok=True)
            self.client.delete_table(temp_output_table, not_found_ok=True)
            logger.debug(f"Cleaned up temporary tables for batch {timestamp}")

    def _calculate_cost(self, num_embeddings: int, model_config: Dict) -> float:
        """Calculate estimated cost for embeddings"""
        return (num_embeddings / 1000) * model_config.get('cost_per_1k', 0)

    def _update_database_stats(self, database_id: str):
        """Update database registry with embedding statistics"""
        try:
            query = f"""
            UPDATE `{self.full_dataset_id}.database_registry` AS dr
            SET
                total_embeddings = (
                    SELECT COUNTIF(embedding IS NOT NULL AND ARRAY_LENGTH(embedding) > 0)
                    FROM `{self.full_dataset_id}.enriched_metadata`
                    WHERE database_id = dr.database_id
                ),
                last_embedded_at = CURRENT_TIMESTAMP()
            WHERE dr.database_id = @database_id
            """
            job_config = bigquery.QueryJobConfig(
                query_parameters=[bigquery.ScalarQueryParameter("database_id", "STRING", database_id)]
            )
            self.client.query(query, job_config=job_config).result()
        except Exception as e:
            logger.error(f"Failed to update database stats for {database_id}: {e}")

    def create_embedding_model(self, model_name: str = 'text-embedding-005') -> bool:
        """Create or replace the embedding model in BigQuery."""
        model_config = self.model_configs.get(model_name)
        if not model_config:
            raise ValueError(f"Unknown model: {model_name}")

        endpoint_map = {
            'text-embedding-005': 'text-embedding-005',
            'gemini-embedding-001': 'text-multimodal-embedding-001'
        }
        endpoint = endpoint_map.get(model_name, model_name)

        # Get connection from environment or use default
        import os
        connection_id = os.getenv('GEMINI_CONNECTION', 'us.gemini_connection')

        query = f"""
        CREATE OR REPLACE MODEL `{model_config['model_path']}`
        REMOTE WITH CONNECTION `{self.project_id}.{connection_id}`
        OPTIONS (endpoint = '{endpoint}')
        """
        try:
            self.client.query(query).result()
            logger.info(f"Successfully created or replaced embedding model: {model_config['model_path']}")
            return True
        except Exception as e:
            logger.error(f"Failed to create embedding model: {e}")
            return False

    def test_embedding_generation(self, test_text: str = "Sample column with customer data") -> bool:
        """Tests if the ML.GENERATE_EMBEDDING function works correctly."""
        try:
            model_config = self.model_configs[self.current_model]
            query = f"""
            SELECT ml_generate_embedding_result
            FROM ML.GENERATE_EMBEDDING(
                MODEL `{model_config['model_path']}`,
                (SELECT '{test_text}' AS content),
                STRUCT(TRUE AS flatten_json_output)
            )
            """
            result = self.client.query(query).to_dataframe()
            embedding = result['ml_generate_embedding_result'].iloc[0]
            if embedding and len(embedding) > 0:
                logger.info(f"Test embedding successful: {len(embedding)} dimensions generated.")
                return True
            logger.error("Test embedding failed: No embedding returned.")
            return False
        except Exception as e:
            logger.error(f"Test embedding failed with an exception: {e}")
            return False

    def create_vector_index(self) -> Dict:
        """
        Creates or replaces the vector index on the enriched metadata table.
        Requires at least 5000 embeddings across all databases.
        """
        try:
            # First check total embeddings across all databases
            count_query = f"""
            SELECT COUNT(*) as total_embeddings
            FROM `{self.full_dataset_id}.enriched_metadata`
            WHERE embedding IS NOT NULL
            AND ARRAY_LENGTH(embedding) = 768
            """

            count_result = self.client.query(count_query).to_dataframe()
            total_embeddings = count_result['total_embeddings'].iloc[0] if not count_result.empty else 0

            logger.info(f"Total embeddings across all databases: {total_embeddings}")

            if total_embeddings < 5000:
                message = f"Cannot create vector index: Only {total_embeddings} embeddings found. Need at least 5000."
                logger.warning(message)
                return {
                    'success': False,
                    'message': message,
                    'total_embeddings': int(total_embeddings),
                    'required_embeddings': 5000,
                    'remaining_needed': 5000 - int(total_embeddings)
                }

            # Create the vector index
            index_query = f"""
            CREATE OR REPLACE VECTOR INDEX enriched_embedding_index
            ON `{self.full_dataset_id}.enriched_metadata`(embedding)
            STORING(
                database_id,      -- Critical for database isolation
                table_name,
                column_name,
                data_type,
                semantic_context,
                example_values,
                distinct_count,
                null_percentage
            )
            OPTIONS(
                index_type = 'IVF',
                distance_type = 'COSINE',
                ivf_options = '{{"num_lists": 50}}'  -- Optimized for smaller datasets
            )
            """

            logger.info("Attempting to create vector index...")
            job = self.client.query(index_query)
            job.result()  # Wait for the job to complete

            message = f"Successfully created vector index 'enriched_embedding_index' with {total_embeddings} embeddings"
            logger.info(message)

            # Verify the index was created
            verify_query = f"""
            SELECT COUNT(*) as index_count
            FROM `{self.project_id}.{self.dataset_id}.INFORMATION_SCHEMA.VECTOR_INDEXES`
            WHERE index_name = 'enriched_embedding_index'
            """

            verify_result = self.client.query(verify_query).to_dataframe()
            index_exists = verify_result['index_count'].iloc[0] > 0 if not verify_result.empty else False

            if index_exists:
                return {
                    'success': True,
                    'message': message,
                    'total_embeddings': int(total_embeddings),
                    'index_created': True
                }
            else:
                return {
                    'success': False,
                    'message': "Index creation command executed but index not found in INFORMATION_SCHEMA",
                    'total_embeddings': int(total_embeddings)
                }

        except Exception as e:
            error_message = str(e)
            logger.error(f"Failed to create vector index: {error_message}")

            # Check if it's a known error about row count
            if "5000" in error_message or "rows" in error_message.lower():
                return {
                    'success': False,
                    'message': f"BigQuery requirement not met: {error_message}",
                    'total_embeddings': int(total_embeddings) if 'total_embeddings' in locals() else 0,
                    'error': error_message
                }
            else:
                return {
                    'success': False,
                    'message': f"Failed to create vector index: {error_message}",
                    'error': error_message
                }

    def check_vector_index_status(self) -> Dict:
        """
        Check if vector index exists and get its status
        """
        try:
            query = f"""
            SELECT
                index_name,
                table_name,
                index_status,
                creation_time,
                coverage_percentage,
                ddl
            FROM `{self.project_id}.{self.dataset_id}.INFORMATION_SCHEMA.VECTOR_INDEXES`
            WHERE table_name = 'enriched_metadata'
            """

            result = self.client.query(query).to_dataframe()

            if result.empty:
                return {
                    'exists': False,
                    'message': 'No vector index found'
                }

            index_info = result.iloc[0].to_dict()
            return {
                'exists': True,
                'index_name': index_info.get('index_name'),
                'status': index_info.get('index_status'),
                'creation_time': str(index_info.get('creation_time')),
                'coverage_percentage': float(index_info.get('coverage_percentage', 0)),
                'ddl': index_info.get('ddl', '')
            }

        except Exception as e:
            logger.error(f"Failed to check vector index status: {e}")
            return {
                'exists': False,
                'error': str(e)
            }