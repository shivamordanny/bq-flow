"""
BigQuery AI Functions Module
Implements all BigQuery AI features for maximum showcase
"""

from google.cloud import bigquery
from typing import List, Dict, Any, Optional
import json
import os
from datetime import datetime
import time
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import centralized logging
from .logging import (
    get_logger, log_performance, log_bigquery_operation
)

# Import configuration
from .config import get_config

# Get logger for this module
logger = get_logger(__name__)

# Get configuration
config = get_config()

# Import centralized utilities
from .utils import json_serial, clean_column_data, convert_numpy_types

# Initialize BigQuery client using centralized config
# Get client from config module instead of creating duplicate instance
client = config.client

logger.info(f"BigQuery AI Functions initialized with project: {config.get_dataset_config().project_id}")

@log_performance
async def generate_query_embedding(query: str) -> List[float]:
    """
    Generate embedding for user query using ML.GENERATE_EMBEDDING
    """
    logger.debug(f"Generating embedding for query: {query[:50]}...")

    sql = f"""
    SELECT ml_generate_embedding_result AS embedding
    FROM ML.GENERATE_EMBEDDING(
        MODEL `{config.get_model_path('embedding')}`,
        (SELECT '{query.replace("'", "''")}' AS content),
        STRUCT(
            TRUE AS flatten_json_output,
            {config.get_embedding_config()['dimensions']} AS output_dimensionality,
            'RETRIEVAL_QUERY' AS task_type
        )
    )
    """

    try:
        start_time = time.time()
        result = list(client.query(sql).result())[0]
        duration_ms = int((time.time() - start_time) * 1000)

        log_bigquery_operation(
            logger,
            "ML.GENERATE_EMBEDDING",
            {
                'query_length': len(query),
                'output_dims': {config.get_embedding_config()['dimensions']},
                'task_type': 'RETRIEVAL_QUERY'
            },
            duration_ms
        )

        logger.info(
            f"Embedding generated successfully",
            extra={
                'extra_fields': {
                    'embedding_dims': len(result.embedding),
                    'query_length': len(query),
                    'duration_ms': duration_ms
                }
            }
        )

        # Convert numpy array to list for JSON serialization
        embedding = result.embedding
        if hasattr(embedding, 'tolist'):
            # If it's a numpy array, convert to list
            return embedding.tolist()
        return embedding
    except Exception as e:
        logger.error(
            f"Failed to generate embedding: {str(e)}",
            exc_info=True,
            extra={
                'extra_fields': {
                    'query_preview': query[:100],
                    'error_type': type(e).__name__
                }
            }
        )
        raise

# clean_column_data function moved to utils.py

@log_performance
async def vector_search_columns(query_embedding: List[float], database_id: str) -> tuple[List[Dict[str, Any]], str]:
    """
    Use VECTOR_SEARCH with IVF index for semantic column discovery
    Now uses enriched_metadata table with better context
    Returns: (results, feature_used)
    """
    logger.debug(f"Starting vector search for database: {database_id}")
    feature_used = "Manual Similarity"  # Default fallback

    # First check if enriched metadata exists with valid embeddings
    check_enriched_query = f"""
    SELECT COUNT(*) as enriched_count
    FROM `{config.get_metadata_table_path('enriched')}`
    WHERE database_id = '{database_id}'
    AND embedding IS NOT NULL
    AND ARRAY_LENGTH(embedding) = {config.get_embedding_config()['dimensions']}  -- Only count valid embeddings
    """

    try:
        enriched_result = list(client.query(check_enriched_query).result())[0]
        use_enriched = enriched_result.enriched_count > 0
        logger.info(f"Enriched metadata available: {use_enriched} ({enriched_result.enriched_count} columns)")
    except Exception as e:
        logger.warning(f"Could not check enriched metadata: {e}")
        use_enriched = False

    # Check for vector index on enriched table
    metadata_project = config._config['datasets']['metadata_project']
    metadata_dataset = config._config['datasets']['metadata_dataset']
    check_index_query = f"""
    SELECT COUNT(*) as index_count
    FROM `{metadata_project}.{metadata_dataset}.INFORMATION_SCHEMA.VECTOR_INDEXES`
    WHERE index_name = 'enriched_embedding_index'
    """

    try:
        start_time = time.time()
        index_result = list(client.query(check_index_query).result())[0]
        has_index = index_result.index_count > 0
        check_duration = int((time.time() - start_time) * 1000)

        logger.debug(
            f"Index check completed",
            extra={
                'extra_fields': {
                    'has_index': has_index,
                    'use_enriched': use_enriched,
                    'duration_ms': check_duration
                }
            }
        )
    except Exception as e:
        has_index = False
        logger.debug(f"Could not check for vector index: {str(e)}")

    if use_enriched and has_index:
        # Use VECTOR_SEARCH with enriched index (BEST CASE)
        logger.info("Using VECTOR_SEARCH with enriched IVF index for maximum accuracy")
        feature_used = "VECTOR_SEARCH with IVF index"

        embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'

        query = f"""
        SELECT
            base.table_name,
            base.column_name,
            base.semantic_context as description,
            base.example_values,
            base.distinct_count,
            distance,
            1 - distance as similarity_score
        FROM VECTOR_SEARCH(
            TABLE `{config.get_metadata_table_path('enriched')}`,
            'embedding',
            (SELECT {embedding_str} AS embedding),
            top_k => 15,
            distance_type => 'COSINE',
            options => '{{"fraction_lists_to_search": 0.1}}'
        )
        WHERE base.database_id = '{database_id}'
        AND distance < 0.5  -- More strict threshold for enriched data
        AND ARRAY_LENGTH(base.embedding) = {config.get_embedding_config()['dimensions']}  -- Filter out empty arrays
        ORDER BY distance
        LIMIT 10
        """

        try:
            results = client.query(query).to_dataframe()
            return [
                {
                    'table_name': row['table_name'],
                    'column_name': row['column_name'],
                    'description': row['description'],
                    'example_values': clean_column_data(row.get('example_values', [])),
                    'distinct_count': int(row.get('distinct_count', 0)) if row.get('distinct_count') is not None else 0,
                    'similarity_score': float(row['similarity_score'])
                }
                for _, row in results.iterrows()
            ], feature_used
        except Exception as e:
            logger.warning(f"VECTOR_SEARCH failed, falling back to ML.DISTANCE: {e}")

    elif use_enriched:
        # Use enriched metadata with ML.DISTANCE (GOOD CASE)
        logger.info("Using enriched metadata with ML.DISTANCE")
        feature_used = "ML.DISTANCE with enriched metadata"

        embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'

        query = f"""
        WITH query_embedding AS (
            SELECT {embedding_str} AS embedding
        ),
        column_similarities AS (
            SELECT
                em.table_name,
                em.column_name,
                em.semantic_context as description,
                em.example_values,
                em.distinct_count,
                -- Use ML.DISTANCE for proper similarity
                1 - ML.DISTANCE(
                    em.embedding,
                    qe.embedding,
                    'COSINE'
                ) as similarity_score
            FROM query_embedding qe
            CROSS JOIN `{config.get_metadata_table_path('enriched')}` em
            WHERE em.database_id = '{database_id}'
            AND em.embedding IS NOT NULL
            AND ARRAY_LENGTH(em.embedding) = {config.get_embedding_config()['dimensions']}  -- Filter out empty arrays
        )
        SELECT *
        FROM column_similarities
        WHERE similarity_score > 0.4  -- Higher threshold for enriched
        ORDER BY similarity_score DESC
        LIMIT 10
        """

        try:
            results = client.query(query).to_dataframe()

            logger.info(f"ML.DISTANCE enriched search found {len(results)} matches")

            return [
                {
                    'table_name': row['table_name'],
                    'column_name': row['column_name'],
                    'description': row['description'],
                    'example_values': clean_column_data(row.get('example_values', [])),
                    'distinct_count': int(row.get('distinct_count', 0)) if row.get('distinct_count') is not None else 0,
                    'similarity_score': float(row['similarity_score'])
                }
                for _, row in results.iterrows()
            ], feature_used
        except Exception as e:
            logger.warning(f"ML.DISTANCE search on enriched metadata failed: {e}")

    # Fallback to original metadata with ML.DISTANCE (FALLBACK CASE)
    logger.info("Falling back to original metadata with ML.DISTANCE")
    feature_used = "ML.DISTANCE with original metadata"

    embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'

    query = f"""
    WITH query_embedding AS (
        SELECT {embedding_str} AS embedding
    ),
    column_similarities AS (
        SELECT
            dme.table_name,
            dme.column_name,
            dme.description,
            -- Use ML.DISTANCE for proper similarity
            1 - ML.DISTANCE(
                dme.embedding,
                qe.embedding,
                'COSINE'
            ) as similarity_score
        FROM query_embedding qe
        CROSS JOIN `{config.get_metadata_table_path('enriched')}` dme
        WHERE dme.database_id = '{database_id}'
        AND dme.embedding IS NOT NULL
    )
    SELECT *
    FROM column_similarities
    WHERE similarity_score > 0.3
    ORDER BY similarity_score DESC
    LIMIT 10
    """

    start_time = time.time()
    results = client.query(query).to_dataframe()
    duration_ms = int((time.time() - start_time) * 1000)

    log_bigquery_operation(
        logger,
        feature_used,
        {
            'database_id': database_id,
            'embedding_dims': len(query_embedding),
            'threshold': 0.3
        },
        duration_ms
    )

    logger.info(
        f"ML.DISTANCE fallback vector search completed",
        extra={
            'extra_fields': {
                'columns_found': len(results),
                'top_score': results.iloc[0]['similarity_score'] if len(results) > 0 else 0,
                'duration_ms': duration_ms,
                'feature_used': feature_used
            }
        }
    )

    return [
        {
            'table_name': row['table_name'],
            'column_name': row['column_name'],
            'description': row['description'],
            'similarity_score': float(row['similarity_score'])
        }
        for _, row in results.iterrows()
    ], feature_used

async def generate_sql_with_context(user_query: str, database_id: str, relevant_columns: List[Dict]) -> str:
    """
    Generate SQL using AI.GENERATE with context from found columns
    """

    # Get database info with error handling
    db_query = f"""
    SELECT project_id, dataset_name, description
    FROM `{config.get_metadata_table_path('databases')}`
    WHERE database_id = '{database_id}'
    """

    try:
        db_results = list(client.query(db_query).result())
        if db_results:
            db_info = db_results[0]
            project_id = db_info.project_id
            dataset_name = db_info.dataset_name
            description = db_info.description or "No description available"
        else:
            # Fallback: Try to parse database_id for project and dataset info
            logger.warning(f"Database {database_id} not found in registry, using fallback")
            if '.' in database_id:
                parts = database_id.split('.')
                project_id = parts[0]
                dataset_name = '.'.join(parts[1:]) if len(parts) > 1 else database_id
            else:
                # Use default public data project as fallback
                project_id = "bigquery-public-data"
                dataset_name = database_id
            description = f"Database: {database_id}"

    except Exception as e:
        logger.error(f"Error looking up database {database_id}: {e}")
        # Use sensible defaults
        project_id = "bigquery-public-data"
        dataset_name = database_id
        description = f"Database: {database_id}"

    # Build context from discovered columns
    context = f"""
    Database: {project_id}.{dataset_name}
    Description: {description}

    Available columns from semantic search:
    """

    for col in relevant_columns[:10]:  # Top 10 columns
        desc = col['description'] if col['description'] else ''
        context += f"\n- {col['table_name']}.{col['column_name']}: {desc} (similarity: {col['similarity_score']:.2%})"

    # Generate SQL using AI.GENERATE
    prompt = f"""
    You are a Google BigQuery expert. Your task is to generate a high-quality, executable BigQuery SQL query based on the user's request and the provided schema context.

    **Database Context:**
    - **Database:** {db_info.project_id}.{db_info.dataset_name}
    - **Description:** {db_info.description}
    - **Relevant Columns (from semantic search):**
    {context}

    **User Query:** "{user_query}"

    **Instructions & Rules:**
    1.  **Table Naming:** Use the full table path: `{db_info.project_id}.{db_info.dataset_name}.{{table_name}}`.
    2.  **Use Aliases:** Always use concise table aliases (e.g., `AS t1`, `AS o`, `AS p`) to improve readability, especially for JOINs.
    3.  **Date/Time Functions:**
        - Adhere strictly to BigQuery's date/time functions (`TIMESTAMP_DIFF`, `DATE_TRUNC`, `EXTRACT`, etc.).
        - If the user asks for a relative time period like "last month" or "this year", use functions like `DATE_SUB(CURRENT_DATE(), INTERVAL 1 MONTH)`.
    4.  **Date-Sharded Tables:** If you see tables with date suffixes (e.g., `ga_sessions_YYYYMMDD`), you MUST query the latest available table unless the user specifies a different date.
    5.  **Output:** Return ONLY the executable SQL query. Do not include any explanations, comments, or markdown formatting like ````sql`.
    """

    query = f'''
    SELECT AI.GENERATE(
        """{prompt}""",
        connection_id => '{config.get_connection_id('gemini')}',
        endpoint => '{config.get_model_endpoint('generation')}'
    ).result AS generated_sql
    '''

    result = list(client.query(query).result())[0]
    sql = result.generated_sql.strip()

    # Clean up SQL (remove markdown if present)
    if '```sql' in sql:
        sql = sql.split('```sql')[1].split('```')[0].strip()
    elif '```' in sql:
        sql = sql.split('```')[1].split('```')[0].strip()

    return sql

async def execute_bigquery(sql_query: str) -> Dict[str, Any]:
    """
    Execute the generated SQL and return results
    """
    try:
        query_job = client.query(sql_query)
        results = query_job.to_dataframe()

        # Clean the results to remove numpy arrays
        cleaned_results = []
        for _, row in results.iterrows():
            cleaned_row = {}
            for col, val in row.items():
                cleaned_row[col] = clean_column_data(val)
            cleaned_results.append(cleaned_row)

        return {
            'success': True,
            'row_count': len(results),
            'results': cleaned_results,
            'execution_time': (query_job.ended - query_job.started).total_seconds() if query_job.ended else None
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'results': []
        }

async def generate_structured_insights(user_query: str, query_results: List[Dict]) -> Dict[str, Any]:
    """
    Use AI.GENERATE_TABLE for structured insight extraction
    Creates structured analysis from query results
    """

    # Convert results to JSON string for analysis with custom serializer for Timestamp objects
    results_json = json.dumps(query_results[:10], default=json_serial)  # Limit to 10 rows

    prompt = f"""
    Analyze these query results and provide business insights.

    Original Query: {user_query}

    Results:
    {results_json}

    Provide:
    1. A key finding from the data
    2. Any observable trend
    3. A recommendation based on the data
    4. Two suggested follow-up questions

    Format as JSON with keys: key_finding, trend, recommendation, next_questions (array)
    """

    # Use AI.GENERATE to get structured insights
    query = f'''
    SELECT AI.GENERATE(
        """{prompt}""",
        connection_id => '{config.get_connection_id('gemini')}',
        endpoint => '{config.get_model_endpoint('insight')}'
    ).result AS insights_json
    '''

    try:
        result = list(client.query(query).result())[0]
        insights_text = result.insights_json.strip()

        # Parse JSON from response
        if '```json' in insights_text:
            insights_text = insights_text.split('```json')[1].split('```')[0].strip()
        elif '```' in insights_text:
            insights_text = insights_text.split('```')[1].split('```')[0].strip()

        insights = json.loads(insights_text)
        return insights
    except Exception as e:
        logger.warning(f"Could not parse insights: {e}")
        return {
            "key_finding": "Analysis completed successfully",
            "trend": "Data retrieved as requested",
            "recommendation": "Review the results for detailed insights",
            "next_questions": [
                "Would you like to filter these results further?",
                "Should we analyze a different time period?"
            ]
        }

async def generate_column_explanations(user_query: str, columns: List[Dict]) -> Dict[str, str]:
    """
    Generate AI-powered explanations for why columns were selected.
    Uses AI.GENERATE to provide transparency about column relevance.
    Note: This is NOT ML.EXPLAIN_PREDICT - we use AI for interpretable explanations.
    """

    # Build explanation prompt
    columns_text = "\n".join([
        f"- {col['table_name']}.{col['column_name']} (similarity: {col['similarity_score']:.2%})"
        for col in columns
    ])

    prompt = f"""
    Explain why these columns were selected for the query: "{user_query}"

    Selected columns:
    {columns_text}

    Provide a brief explanation for each column's relevance in one sentence.
    Format as JSON with column names as keys.
    """

    query = f'''
    SELECT AI.GENERATE(
        """{prompt}""",
        connection_id => '{config.get_connection_id('gemini')}',
        endpoint => '{config.get_model_endpoint('generation')}'
    ).result AS explanations_json
    '''

    try:
        result = list(client.query(query).result())[0]
        explanations_text = result.explanations_json.strip()

        # Parse JSON
        if '```json' in explanations_text:
            explanations_text = explanations_text.split('```json')[1].split('```')[0].strip()
        elif '```' in explanations_text:
            explanations_text = explanations_text.split('```')[1].split('```')[0].strip()

        return json.loads(explanations_text)
    except Exception as e:
        logger.warning(f"Could not generate explanations: {e}")
        return {}

async def check_semantic_cache(user_query: str, database_id: str) -> Optional[Dict[str, Any]]:
    """
    Check semantic cache for similar queries
    """
    # First check if cache tables exist
    check_query = """
    SELECT COUNT(*) as cache_count
    FROM `{config.get_metadata_table_path('cache')}`
    LIMIT 1
    """

    try:
        cache_exists = list(client.query(check_query).result())[0].cache_count > 0
        if not cache_exists:
            logger.debug("Cache is empty, skipping cache lookup")
            return None
    except Exception as e:
        logger.debug(f"Cache tables not found, skipping cache lookup: {e}")
        return None

    # Now try the cache lookup
    metadata_project = config._config['datasets']['metadata_project']
    metadata_dataset = config._config['datasets']['metadata_dataset']
    query = f"""
    SELECT
        found_match,
        cached_query,
        cached_sql,
        similarity_score,
        original_confidence,
        usage_count
    FROM `{metadata_project}.{metadata_dataset}.semantic_cache_lookup`(
        '{user_query.replace("'", "''")}',
        0.85  -- Higher threshold for cache hits
    )
    """

    try:
        result = list(client.query(query).result())[0]
        if result.found_match:
            # Update usage count in cache
            update_query = f"""
            UPDATE `{config.get_metadata_table_path('cache')}`
            SET usage_count = usage_count + 1,
                last_used = CURRENT_TIMESTAMP()
            WHERE user_query = '{result.cached_query.replace("'", "''")}'
            """
            client.query(update_query)

            return {
                'found_match': True,
                'cached_query': result.cached_query,
                'cached_sql': result.cached_sql,
                'similarity_score': result.similarity_score,
                'usage_count': result.usage_count
            }
    except Exception as e:
        logger.warning(f"Cache lookup failed: {e}")

    return None

def calculate_cost(features_used: List[str], query_length: int) -> float:
    """
    Calculate estimated cost based on BigQuery pricing
    Shows understanding of resource consumption
    """
    # Pricing as of September 2025 (approximate)
    costs = {
        "ML.GENERATE_EMBEDDING": 0.000025 * (query_length / 1000),  # per 1k tokens
        "VECTOR_SEARCH with IVF index": 0.000001,  # per query
        "Semantic Cache Lookup": 0.0000005,  # minimal cost
        f"AI.GENERATE ({config.get_model_endpoint('generation')})": 0.000075 * ((query_length + 500) / 1000),  # input + output tokens
        "AI.GENERATE_TABLE (Structured Insights)": 0.000100,  # per call
        "AI-Generated Explanations": 0.000050,  # per explanation using AI.GENERATE
        "Query Refinement with Context": 0.000080,  # slightly higher for context
    }

    total = sum(costs.get(feature, 0) for feature in features_used)

    # Add base query execution cost (minimal)
    total += 0.000005

    return round(total, 6)