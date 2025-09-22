"""
BigQuery Client Module for BQ Flow Embeddings System
Handles all database operations with multi-tenant isolation
"""

from google.cloud import bigquery
from typing import Dict, List, Any, Optional, Tuple
import uuid
from datetime import datetime
import logging
import pandas as pd
import json
from pathlib import Path
from .logger import bigquery_logger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BigQueryClient:
    """
    BigQuery client with multi-tenant support and database_id isolation
    """

    def __init__(self, project_id: str = None, dataset_id: str = None):
        # Import config to get defaults from config.yaml
        import os
        import yaml

        # Try to load config
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config', 'config.yaml')
        config = {}
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

        # Get project_id and dataset_id from environment if not provided
        if not project_id:
            project_id = os.getenv('PROJECT_ID')
            if not project_id:
                raise ValueError("PROJECT_ID must be provided or set in environment variables")

        if not dataset_id:
            dataset_id = os.getenv('DATASET_ID')
            if not dataset_id:
                raise ValueError("DATASET_ID must be provided or set in environment variables")

        self.project_id = project_id
        self.dataset_id = dataset_id
        self.client = bigquery.Client(project=self.project_id)
        self.full_dataset_id = f"{self.project_id}.{self.dataset_id}"

    def execute_query(self, query: str, params: Optional[Dict] = None) -> List[Dict]:
        """Execute a query and return results as list of dicts"""
        try:
            if params:
                job_config = bigquery.QueryJobConfig(
                    query_parameters=[
                        bigquery.ScalarQueryParameter(key, "STRING", value)
                        for key, value in params.items()
                    ]
                )
                query_job = self.client.query(query, job_config=job_config)
            else:
                query_job = self.client.query(query)

            results = query_job.result()
            return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise

    def execute_query_to_df(self, query: str, params: Optional[Dict] = None) -> pd.DataFrame:
        """Execute a query and return results as DataFrame"""
        results = self.execute_query(query, params)
        return pd.DataFrame(results) if results else pd.DataFrame()

    def create_tables_if_not_exist(self):
        """Create all necessary tables if they don't exist"""
        sql_file = "sql/01_create_tables.sql"
        try:
            with open(sql_file, 'r') as f:
                sql_content = f.read()

            # Split by statement and execute each
            statements = [s.strip() for s in sql_content.split(';') if s.strip() and not s.strip().startswith('--')]

            for statement in statements:
                if statement:
                    try:
                        self.client.query(statement).result()
                        logger.info(f"Executed table creation statement successfully")
                    except Exception as e:
                        if "Already Exists" in str(e):
                            logger.info("Table already exists, skipping")
                        else:
                            logger.error(f"Error creating table: {e}")

            return True
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            return False

    def register_database(self, database_config: Dict) -> bool:
        """Register a new database in the registry"""
        try:
            query = f"""
            INSERT INTO `{self.full_dataset_id}.database_registry`
            (database_id, display_name, project_id, dataset_name, description,
             sample_size, embedding_model, profiling_strategy)
            VALUES
            (@database_id, @display_name, @project_id, @dataset_name, @description,
             @sample_size, @embedding_model, @profiling_strategy)
            """

            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("database_id", "STRING", database_config['database_id']),
                    bigquery.ScalarQueryParameter("display_name", "STRING", database_config.get('display_name', database_config['database_id'])),
                    bigquery.ScalarQueryParameter("project_id", "STRING", database_config['project_id']),
                    bigquery.ScalarQueryParameter("dataset_name", "STRING", database_config['dataset_name']),
                    bigquery.ScalarQueryParameter("description", "STRING", database_config.get('description', '')),
                    bigquery.ScalarQueryParameter("sample_size", "INT64", database_config.get('sample_size', 1000)),
                    bigquery.ScalarQueryParameter("embedding_model", "STRING", database_config.get('embedding_model', 'text-embedding-005')),
                    bigquery.ScalarQueryParameter("profiling_strategy", "STRING", database_config.get('profiling_strategy', 'auto')),
                ]
            )

            self.client.query(query, job_config=job_config).result()
            logger.info(f"Database {database_config['database_id']} registered successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to register database: {e}")
            return False

    def get_registered_databases(self) -> pd.DataFrame:
        """Get all registered databases"""
        query = f"""
        SELECT
            database_id,
            display_name,
            project_id,
            dataset_name,
            table_count,
            column_count,
            total_embeddings,
            last_profiled_at,
            last_embedded_at
        FROM `{self.full_dataset_id}.database_registry`
        WHERE is_active = TRUE
        ORDER BY database_id
        """
        return self.execute_query_to_df(query)

    def discover_tables(self, project_id: str, dataset_name: str) -> List[Dict]:
        """Discover all tables in a dataset"""
        try:
            query = f"""
            SELECT
                table_name,
                table_type,
                creation_time
            FROM `{project_id}.{dataset_name}.INFORMATION_SCHEMA.TABLES`
            WHERE table_type = 'BASE TABLE'
            ORDER BY table_name
            """
            tables = self.execute_query(query)
            logger.info(f"INFORMATION_SCHEMA discovered {len(tables)} tables: {[t['table_name'] for t in tables]}")
            return tables
        except Exception as e:
            logger.error(f"Failed to discover tables using INFORMATION_SCHEMA: {e}")
            logger.info(f"Attempting fallback method for {project_id}.{dataset_name}")
            # Fallback to listing tables directly
            try:
                dataset_ref = self.client.dataset(dataset_name, project=project_id)
                tables = self.client.list_tables(dataset_ref)
                raw_table_list = [{'table_name': table.table_id, 'table_type': 'BASE TABLE', 'creation_time': None} for table in tables]

                # Log all discovered tables for debugging
                logger.info(f"Fallback discovered {len(raw_table_list)} tables: {[t['table_name'] for t in raw_table_list]}")

                # Validate tables - filter out suspicious ones
                valid_tables = []
                for table in raw_table_list:
                    table_name = table['table_name']
                    # Skip tables with suspicious names
                    if '-' in table_name and table_name.startswith(dataset_name):
                        logger.warning(f"Skipping potentially invalid table: {table_name}")
                    elif table_name == dataset_name:
                        logger.warning(f"Skipping table with same name as dataset: {table_name}")
                    else:
                        valid_tables.append(table)

                logger.info(f"After validation, {len(valid_tables)} valid tables: {[t['table_name'] for t in valid_tables]}")
                return valid_tables
            except Exception as fallback_error:
                logger.error(f"Fallback method also failed: {fallback_error}")
                return []

    def discover_columns(self, project_id: str, dataset_name: str, table_name: str) -> List[Dict]:
        """Discover all columns in a table"""
        try:
            query = f"""
            SELECT
                column_name,
                ordinal_position,
                is_nullable,
                data_type,
                is_partitioning_column,
                clustering_ordinal_position
            FROM `{project_id}.{dataset_name}.INFORMATION_SCHEMA.COLUMNS`
            WHERE table_name = '{table_name}'
            ORDER BY ordinal_position
            """
            return self.execute_query(query)
        except Exception as e:
            logger.error(f"Failed to discover columns using INFORMATION_SCHEMA: {e}")
            logger.info(f"Attempting fallback method for {project_id}.{dataset_name}.{table_name}")
            # Fallback to getting table schema
            try:
                table_ref = self.client.dataset(dataset_name, project=project_id).table(table_name)
                table = self.client.get_table(table_ref)
                columns = [
                    {
                        'column_name': field.name,
                        'data_type': field.field_type,
                        'is_nullable': field.mode != 'REQUIRED',
                        'ordinal_position': idx + 1,
                        'is_partitioning_column': False,
                        'clustering_ordinal_position': None
                    }
                    for idx, field in enumerate(table.schema)
                ]
                logger.info(f"Successfully discovered {len(columns)} columns using fallback method")
                return columns
            except Exception as fallback_error:
                logger.error(f"Fallback method also failed: {fallback_error}")
                return []

    def create_job(self, database_id: str, job_type: str) -> str:
        """Create a new job entry and return job_id"""
        job_id = str(uuid.uuid4())

        query = f"""
        INSERT INTO `{self.full_dataset_id}.embedding_jobs`
        (job_id, database_id, job_type, status, start_time)
        VALUES
        ('{job_id}', '{database_id}', '{job_type}', 'running', CURRENT_TIMESTAMP())
        """

        self.client.query(query).result()
        return job_id

    def update_job_progress(self, job_id: str, processed: int, total: int, status: str = 'running'):
        """Update job progress"""
        progress = (processed / total * 100) if total > 0 else 0

        query = f"""
        UPDATE `{self.full_dataset_id}.embedding_jobs`
        SET
            processed_items = {processed},
            total_items = {total},
            progress_percentage = {progress},
            status = '{status}'
        WHERE job_id = '{job_id}'
        """

        self.client.query(query).result()

    def complete_job(self, job_id: str, success: bool = True, error_message: str = None):
        """Mark a job as completed"""
        status = 'completed' if success else 'failed'

        query = f"""
        UPDATE `{self.full_dataset_id}.embedding_jobs`
        SET
            status = '{status}',
            end_time = CURRENT_TIMESTAMP(),
            duration_seconds = TIMESTAMP_DIFF(CURRENT_TIMESTAMP(), start_time, SECOND)
            {f", error_message = '{error_message}'" if error_message else ""}
        WHERE job_id = '{job_id}'
        """

        self.client.query(query).result()

    def get_enriched_metadata_count(self, database_id: str) -> Dict[str, int]:
        """Get counts of enriched metadata for a database"""
        query = f"""
        SELECT
            COUNT(DISTINCT table_name) as table_count,
            COUNT(DISTINCT column_name) as column_count,
            COUNT(*) as total_rows,
            COUNTIF(embedding IS NOT NULL AND ARRAY_LENGTH(embedding) > 0) as embedded_count
        FROM `{self.full_dataset_id}.enriched_metadata`
        WHERE database_id = '{database_id}'
        """

        results = self.execute_query(query)
        return results[0] if results else {'table_count': 0, 'column_count': 0, 'total_rows': 0, 'embedded_count': 0}

    def delete_existing_metadata(self, database_id: str, table_name: Optional[str] = None) -> int:
        """Delete existing metadata records for a database before insertion"""
        try:
            # Check existing count first
            count_query = f"""
            SELECT COUNT(*) as count
            FROM `{self.full_dataset_id}.enriched_metadata`
            WHERE database_id = '{database_id}'
            """
            if table_name:
                count_query += f" AND table_name = '{table_name}'"

            result = self.execute_query(count_query)
            existing_count = result[0]['count'] if result else 0

            if existing_count > 0:
                # Delete existing records
                delete_query = f"""
                DELETE FROM `{self.full_dataset_id}.enriched_metadata`
                WHERE database_id = '{database_id}'
                """
                if table_name:
                    delete_query += f" AND table_name = '{table_name}'"

                self.client.query(delete_query).result()
                bigquery_logger.log_deletion(database_id, existing_count)
                logger.info(f"Deleted {existing_count} existing metadata records for {database_id}")

            return existing_count

        except Exception as e:
            logger.error(f"Failed to delete existing metadata: {e}")
            bigquery_logger.log_error("DELETE_METADATA", e, {"database_id": database_id})
            return 0

    def merge_enriched_metadata(self, records: List[Dict], database_id: str, is_selected: bool = False):
        """
        Merge (upsert) enriched metadata records using a robust DML pattern.
        This method avoids potential streaming buffer conflicts by using a temporary
        table populated via a load job, followed by a MERGE DML statement.
        """
        if not records:
            logger.info("No metadata records to merge for database %s.", database_id)
            return True

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        temp_table_id = f"{self.full_dataset_id}.temp_merge_{database_id}_{timestamp}"

        try:
            # Step 1: Prepare records and load into a DataFrame
            prepared_records = []
            for record in records:
                example_vals = record.get('example_values', [])
                prepared_record = {
                    'database_id': database_id,
                    'table_name': record.get('table_name'),
                    'column_name': record.get('column_name'),
                    'data_type': record.get('data_type'),
                    'semantic_context': record.get('semantic_context'),
                    'is_selected': record.get('is_selected', is_selected),
                    'selection_reason': record.get('selection_reason'),
                    'selection_score': record.get('selection_score'),
                    'distinct_count': record.get('distinct_count'),
                    'null_count': record.get('null_count'),
                    'null_percentage': record.get('null_percentage'),
                    'total_count': record.get('total_count'),
                    'min_value': str(record.get('min_value')) if record.get('min_value') is not None else None,
                    'max_value': str(record.get('max_value')) if record.get('max_value') is not None else None,
                    'avg_value': record.get('avg_value'),
                    'std_dev': record.get('std_dev'),
                    'example_values': [str(v) for v in example_vals if v is not None][:10] if isinstance(example_vals, list) else [],
                    'enriched_at': datetime.now().isoformat()
                }
                prepared_records.append(prepared_record)
            
            df = pd.DataFrame(prepared_records)
            if df.empty:
                return True

            # Step 2: Define schema and load data into temporary table
            schema = [
                bigquery.SchemaField("database_id", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("table_name", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("column_name", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("data_type", "STRING"),
                bigquery.SchemaField("semantic_context", "STRING"),
                bigquery.SchemaField("is_selected", "BOOLEAN"),
                bigquery.SchemaField("selection_reason", "STRING"),
                bigquery.SchemaField("selection_score", "FLOAT64"),
                bigquery.SchemaField("distinct_count", "INT64"),
                bigquery.SchemaField("null_count", "INT64"),
                bigquery.SchemaField("null_percentage", "FLOAT64"),
                bigquery.SchemaField("total_count", "INT64"),
                bigquery.SchemaField("min_value", "STRING"),
                bigquery.SchemaField("max_value", "STRING"),
                bigquery.SchemaField("avg_value", "FLOAT64"),
                bigquery.SchemaField("std_dev", "FLOAT64"),
                bigquery.SchemaField("example_values", "STRING", mode="REPEATED"),
                bigquery.SchemaField("enriched_at", "STRING"),
            ]
            job_config = bigquery.LoadJobConfig(schema=schema)
            load_job = self.client.load_table_from_dataframe(df, temp_table_id, job_config=job_config)
            load_job.result()
            logger.info(f"Loaded {len(df)} records into temporary table {temp_table_id}")

            # Step 3: Build and execute MERGE statement
            merge_query = f"""
            MERGE `{self.full_dataset_id}.enriched_metadata` AS target
            USING (
                SELECT *, PARSE_TIMESTAMP('%Y-%m-%dT%H:%M:%E*S', enriched_at) AS enriched_at_ts
                FROM `{temp_table_id}`
            ) AS source
            ON target.database_id = source.database_id
               AND target.table_name = source.table_name
               AND target.column_name = source.column_name
            WHEN MATCHED THEN
                UPDATE SET
                    data_type = source.data_type,
                    semantic_context = source.semantic_context,
                    is_selected = source.is_selected,
                    selection_reason = source.selection_reason,
                    selection_score = source.selection_score,
                    distinct_count = source.distinct_count,
                    null_count = source.null_count,
                    null_percentage = source.null_percentage,
                    total_count = source.total_count,
                    min_value = source.min_value,
                    max_value = source.max_value,
                    avg_value = source.avg_value,
                    std_dev = source.std_dev,
                    example_values = source.example_values,
                    enriched_at = source.enriched_at_ts,
                    last_updated = CURRENT_TIMESTAMP()
            WHEN NOT MATCHED THEN
                INSERT (
                    database_id, table_name, column_name, data_type, semantic_context,
                    is_selected, selection_reason, selection_score, distinct_count,
                    null_count, null_percentage, total_count, min_value, max_value,
                    avg_value, std_dev, example_values, enriched_at, last_updated
                ) VALUES (
                    source.database_id, source.table_name, source.column_name, source.data_type,
                    source.semantic_context, source.is_selected, source.selection_reason,
                    source.selection_score, source.distinct_count, source.null_count,
                    source.null_percentage, source.total_count, source.min_value,
                    source.max_value, source.avg_value, source.std_dev,
                    source.example_values, source.enriched_at_ts, CURRENT_TIMESTAMP()
                );
            """
            merge_job = self.client.query(merge_query)
            merge_job.result()

            num_dml_affected_rows = merge_job.num_dml_affected_rows
            bigquery_logger.log_insertion(database_id, len(records), "MERGE_UPSERT")
            logger.info(f"Successfully merged {len(records)} metadata records for database {database_id}, {num_dml_affected_rows} rows affected.")
            return True

        except Exception as e:
            logger.error(f"Failed to merge enriched metadata: {e}")
            raise
        finally:
            # Step 4: Clean up temporary table
            self.client.delete_table(temp_table_id, not_found_ok=True)
            logger.debug(f"Cleaned up temporary table {temp_table_id}")

    def insert_enriched_metadata(self, records: List[Dict], database_id: str, is_selected: bool = False):
        """Insert enriched metadata records for a database - now uses MERGE to avoid streaming conflicts"""
        return self.merge_enriched_metadata(records, database_id, is_selected)

    def sample_table_data(self, project_id: str, dataset_name: str, table_name: str,
                         sample_size: int = 1000) -> pd.DataFrame:
        """Sample data from a table for profiling"""
        if '-' in table_name and table_name.startswith(dataset_name):
            logger.error(f"Invalid table name detected: {table_name}. Skipping sampling.")
            return pd.DataFrame()

        if table_name == dataset_name:
            logger.error(f"Table name same as dataset name: {table_name}. This is likely invalid. Skipping sampling.")
            return pd.DataFrame()

        full_table_path = f"`{project_id}.{dataset_name}.{table_name}`"
        logger.info(f"Attempting to sample table: {full_table_path} with limit {sample_size}")

        query = f"SELECT * FROM {full_table_path} LIMIT {sample_size}"

        try:
            result_df = self.execute_query_to_df(query)
            logger.info(f"Successfully sampled {len(result_df)} rows from {table_name}")
            return result_df
        except Exception as e:
            error_msg = str(e)
            if "does not have a schema" in error_msg:
                logger.error(f"Table {full_table_path} does not have a schema. This table may not exist or may be corrupted.")
            else:
                logger.error(f"Failed to sample table {full_table_path}: {error_msg}")
            return pd.DataFrame()

    def _update_database_stats(self, database_id: str):
        """Update database registry with current embedding statistics"""
        try:
            query = f"""
            UPDATE `{self.full_dataset_id}.database_registry` AS dr
            SET
                total_embeddings = (
                    SELECT COUNTIF(embedding IS NOT NULL AND ARRAY_LENGTH(embedding) > 0)
                    FROM `{self.full_dataset_id}.enriched_metadata`
                    WHERE database_id = dr.database_id
                ),
                column_count = (
                    SELECT COUNT(*)
                    FROM `{self.full_dataset_id}.enriched_metadata`
                    WHERE database_id = dr.database_id
                ),
                table_count = (
                    SELECT COUNT(DISTINCT table_name)
                    FROM `{self.full_dataset_id}.enriched_metadata`
                    WHERE database_id = dr.database_id
                )
            WHERE dr.database_id = '{database_id}'
            """
            self.client.query(query).result()
            logger.info(f"Updated stats for database {database_id}")
        except Exception as e:
            logger.error(f"Failed to update database stats: {e}")

    def vector_search(self, query_text: str, database_id: str, top_k: int = 10) -> List[Dict]:
        """Perform vector search with database isolation"""
        # This will be implemented after embeddings are generated
        return []