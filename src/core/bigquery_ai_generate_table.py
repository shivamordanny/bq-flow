"""
AI.GENERATE_TABLE Implementation for Structured Insights
Uses BigQuery's AI.GENERATE_TABLE for better structured responses
"""

from google.cloud import bigquery
from typing import Dict, List, Any, Optional
import json
from datetime import datetime
import time
import re

# Import configuration
from .config import get_config

# Import logging
from .logging import get_logger, log_performance, log_bigquery_operation

logger = get_logger(__name__)
config = get_config()
client = bigquery.Client(project=config.get_dataset_config().project_id)


def escape_sql_string(value: str) -> str:
    """
    Properly escape a string for safe inclusion in SQL queries.
    Handles single quotes, backslashes, newlines, and other special characters.
    """
    if value is None:
        return ''

    # First, strip leading/trailing whitespace to avoid issues
    # But preserve internal structure
    value = value.strip()

    # Replace backslashes first (must be done before other replacements)
    escaped = value.replace('\\', '\\\\')

    # Replace single quotes with doubled single quotes
    escaped = escaped.replace("'", "''")

    # Remove or escape other potentially problematic characters
    # Remove null bytes which can cause issues
    escaped = escaped.replace('\x00', '')

    # Replace newlines with spaces to keep SQL valid
    # This maintains readability while preventing SQL syntax errors
    escaped = escaped.replace('\n', ' ').replace('\r', ' ')

    # Collapse multiple spaces into single space
    import re
    escaped = re.sub(r'\s+', ' ', escaped)

    # Log if we're dealing with very long strings
    if len(escaped) > 10000:
        logger.warning(f"Escaping very long string: {len(escaped)} characters")

    return escaped


@log_performance
async def generate_structured_insights_v2(
    user_query: str,
    query_results: List[Dict],
    database_id: str
) -> Dict[str, Any]:
    """
    Use AI.GENERATE_TABLE for structured insight extraction
    Better than AI.GENERATE for structured responses
    """
    logger.debug(f"Generating structured insights for {len(query_results)} results")

    if not query_results:
        return {
            "key_finding": "No data to analyze",
            "trend": "N/A",
            "recommendation": "Try a different query",
            "next_questions": []
        }

    # Convert results to JSON for analysis
    results_json = json.dumps(query_results[:10])  # Limit to 10 rows
    model_config = config.get_model_config('insight')

    # Create temporary table with data for AI.GENERATE_TABLE
    temp_table_id = f"temp_insights_{int(time.time())}"
    temp_table_path = f"{config.get_dataset_config().metadata_project}.{config.get_dataset_config().metadata_dataset}.{temp_table_id}"

    try:
        # Create enhanced schema for richer business insights
        # Use single line format to avoid SQL parsing issues
        output_schema = (
            "key_finding STRING, "
            "trend STRING, "
            "trend_direction STRING, "
            "recommendation STRING, "
            "business_impact STRING, "
            "action_priority STRING, "
            "implementation_timeline STRING, "
            "confidence_score FLOAT64, "
            "data_quality STRING, "
            "risk_assessment STRING, "
            "opportunity_highlight STRING, "
            "next_question_1 STRING, "
            "next_question_2 STRING, "
            "next_question_3 STRING"
        )

        # Prepare enhanced prompt for richer insights
        # Use a clean format without problematic leading/trailing whitespace
        prompt = (
            f"Analyze these query results and provide comprehensive business insights. "
            f"Original Query: {user_query} "
            f"Database: {database_id} "
            f"Results Data: {results_json} "
            f"Provide structured analysis with: "
            f"1. Key finding - The most important discovery from the data "
            f"2. Trend - Observable pattern in the data "
            f"3. Trend direction - UPWARD/DOWNWARD/STABLE/VOLATILE "
            f"4. Recommendation - Actionable business advice "
            f"5. Business impact - Financial or operational impact assessment "
            f"6. Action priority - HIGH/MEDIUM/LOW based on urgency "
            f"7. Implementation timeline - IMMEDIATE (1-7 days) / SHORT_TERM (1-4 weeks) / LONG_TERM (1-3 months) "
            f"8. Confidence score - Your confidence in the analysis (0.0-1.0) "
            f"9. Data quality - Assessment of data completeness and reliability "
            f"10. Risk assessment - Potential risks if trends continue "
            f"11. Opportunity highlight - Key opportunity to leverage "
            f"12. Three strategic follow-up questions for deeper analysis"
        )

        # Create temporary table with prompt
        # Use comprehensive escaping to prevent SQL injection
        escaped_prompt = escape_sql_string(prompt)
        escaped_results = escape_sql_string(results_json)

        # Log the actual SQL query for debugging
        logger.debug(f"Creating temp table {temp_table_id} with prompt length: {len(escaped_prompt)}, results length: {len(escaped_results)}")

        # Use persistent table instead of TEMP TABLE to avoid session requirement
        create_temp_query = f"""
        CREATE OR REPLACE TABLE `{temp_table_path}` AS
        SELECT '{escaped_prompt}' as prompt, '{escaped_results}' as data
        """

        # Log first 500 chars of the query to debug syntax errors
        logger.debug(f"SQL Query (first 500 chars): {create_temp_query[:500]}")

        try:
            client.query(create_temp_query).result()
        except Exception as e:
            logger.error(f"Failed to create temp table. Error: {e}")
            logger.error(f"Problematic prompt (first 200 chars): {prompt[:200]}")
            logger.error(f"Problematic results (first 200 chars): {results_json[:200]}")
            raise

        # Use AI.GENERATE_TABLE for structured response
        # First, ensure we have a Gemini model to use
        model_path = f"{config.get_dataset_config().metadata_project}.{config.get_dataset_config().metadata_dataset}.gemini_flash_model"

        # Create or replace the model if it doesn't exist
        create_model_query = f"""
        CREATE OR REPLACE MODEL `{model_path}`
        REMOTE WITH CONNECTION `{config.get_connection_id('gemini')}`
        OPTIONS(
            endpoint = '{model_config.insight_model}'
        )
        """

        try:
            client.query(create_model_query).result()
            logger.debug(f"Created/verified Gemini model: {model_path}")
        except Exception as e:
            logger.warning(f"Model creation warning (may already exist): {e}")

        # Now use the correct AI.GENERATE_TABLE syntax
        generate_query = f"""
        SELECT *
        FROM AI.GENERATE_TABLE(
            MODEL `{model_path}`,
            TABLE `{temp_table_path}`,
            STRUCT(
                "{output_schema}" AS output_schema,
                8192 AS max_output_tokens,
                0.7 AS temperature
            )
        )
        """

        logger.debug(f"AI.GENERATE_TABLE query prepared for {temp_table_id}")

        start_time = time.time()
        results = client.query(generate_query).to_dataframe()
        duration_ms = int((time.time() - start_time) * 1000)

        log_bigquery_operation(
            logger,
            "AI.GENERATE_TABLE",
            {
                'database_id': database_id,
                'rows_analyzed': len(query_results),
                'model': model_config.insight_model
            },
            duration_ms
        )

        if len(results) > 0:
            row = results.iloc[0]
            insights = {
                "key_finding": row.get('key_finding', 'Analysis completed'),
                "trend": row.get('trend', 'Data retrieved successfully'),
                "trend_direction": row.get('trend_direction', 'STABLE'),
                "recommendation": row.get('recommendation', 'Review results'),
                "business_impact": row.get('business_impact', 'Based on the data analysis'),
                "action_priority": row.get('action_priority', 'MEDIUM'),
                "implementation_timeline": row.get('implementation_timeline', 'SHORT_TERM'),
                "confidence_score": float(row.get('confidence_score', 0.8)),
                "data_quality": row.get('data_quality', 'Good'),
                "risk_assessment": row.get('risk_assessment', 'Low risk based on current trends'),
                "opportunity_highlight": row.get('opportunity_highlight', 'Leverage data insights for strategic planning'),
                "next_questions": [
                    row.get('next_question_1', 'Would you like more details?'),
                    row.get('next_question_2', 'Should we analyze a different metric?'),
                    row.get('next_question_3', 'How can we track this metric over time?')
                ]
            }

            logger.info(f"Enhanced structured insights generated with confidence: {insights['confidence_score']:.2f}, priority: {insights['action_priority']}")
            return insights

    except Exception as e:
        logger.error(f"AI.GENERATE_TABLE failed: {e}")
        # Return structured error response instead of fallback
        return {
            "key_finding": "Unable to generate AI insights",
            "trend": "Analysis error occurred",
            "trend_direction": "UNKNOWN",
            "recommendation": "Please try again or contact support",
            "business_impact": "Unable to assess at this time",
            "action_priority": "HIGH",
            "implementation_timeline": "IMMEDIATE",
            "confidence_score": 0.0,
            "data_quality": "Error in processing",
            "risk_assessment": "Unable to determine",
            "opportunity_highlight": "N/A",
            "next_questions": [
                "Would you like to retry the analysis?",
                "Should we try a simpler query?",
                "Can we check the data format?"
            ],
            "error": str(e)
        }

    finally:
        # Clean up temp table if it exists
        try:
            drop_query = f"DROP TABLE IF EXISTS {temp_table_path}"
            client.query(drop_query).result()
            logger.debug(f"Cleaned up temp table: {temp_table_path}")
        except Exception as e:
            logger.warning(f"Failed to clean up temp table {temp_table_path}: {e}")
            # Not critical - table will be auto-cleaned by BigQuery eventually


# Removed fallback function - AI.GENERATE_TABLE should be the primary method
# If it fails, we return an error response directly


@log_performance
async def generate_explanation_table(
    user_query: str,
    columns: List[Dict],
    database_id: str
) -> Dict[str, str]:
    """
    Use AI.GENERATE_TABLE for column selection explanations
    More structured than using AI.GENERATE
    """
    if not columns:
        return {}

    model_config = config.get_model_config('generation')

    # Build structured input
    columns_data = []
    for col in columns[:5]:  # Top 5 columns
        columns_data.append({
            "column_name": f"{col['table_name']}.{col['column_name']}",
            "similarity_score": col['similarity_score'],
            "description": col.get('description', '')
        })

    columns_json = json.dumps(columns_data)

    # Define output schema for explanations
    output_schema = (
        "column_name STRING, "
        "relevance_explanation STRING, "
        "confidence FLOAT64"
    )

    # Build prompt without problematic newlines
    prompt = (
        f"Explain why these columns were selected for the query: '{user_query}' "
        f"Selected columns data: {columns_json} "
        f"For each column, provide: "
        f"1. Clear explanation of relevance "
        f"2. Confidence score (0-1)"
    )

    # Create unique temp table ID
    temp_table_id = f"temp_explain_{int(time.time())}"

    try:
        # Create temporary input
        # Use comprehensive escaping to prevent SQL syntax errors
        escaped_prompt = escape_sql_string(prompt)
        escaped_columns = escape_sql_string(columns_json)

        # Create full table path for persistent table
        temp_table_path = f"{config.get_dataset_config().metadata_project}.{config.get_dataset_config().metadata_dataset}.{temp_table_id}"

        # Log for debugging
        logger.debug(f"Creating explanation temp table {temp_table_id} with prompt length: {len(escaped_prompt)}, columns length: {len(escaped_columns)}")

        # Use persistent table instead of TEMP TABLE to avoid session requirement
        create_temp_query = f"""
        CREATE OR REPLACE TABLE `{temp_table_path}` AS
        SELECT '{escaped_prompt}' as prompt, '{escaped_columns}' as columns_data
        """

        # Log first 500 chars of the query to debug syntax errors
        logger.debug(f"Explanation SQL Query (first 500 chars): {create_temp_query[:500]}")

        try:
            client.query(create_temp_query).result()
        except Exception as e:
            logger.error(f"Failed to create explanation temp table. Error: {e}")
            logger.error(f"Problematic prompt (first 200 chars): {prompt[:200]}")
            logger.error(f"Problematic columns (first 200 chars): {columns_json[:200]}")
            raise

        # Generate explanations
        # First, ensure we have a Gemini model to use
        model_path = f"{config.get_dataset_config().metadata_project}.{config.get_dataset_config().metadata_dataset}.gemini_flash_model"

        # Create or replace the model if it doesn't exist
        create_model_query = f"""
        CREATE OR REPLACE MODEL `{model_path}`
        REMOTE WITH CONNECTION `{config.get_connection_id('gemini')}`
        OPTIONS(
            endpoint = '{model_config.generation_model}'
        )
        """

        try:
            client.query(create_model_query).result()
            logger.debug(f"Created/verified Gemini model for explanations: {model_path}")
        except Exception as e:
            logger.warning(f"Model creation warning (may already exist): {e}")

        # Now use the correct AI.GENERATE_TABLE syntax
        generate_query = f"""
        SELECT *
        FROM AI.GENERATE_TABLE(
            MODEL `{model_path}`,
            TABLE `{temp_table_path}`,
            STRUCT(
                "{output_schema}" AS output_schema,
                4096 AS max_output_tokens
            )
        )
        """

        logger.debug(f"Explanation AI.GENERATE_TABLE query prepared for {temp_table_id}")

        results = client.query(generate_query).to_dataframe()

        explanations = {}
        for _, row in results.iterrows():
            col_name = row.get('column_name', '')
            explanation = row.get('relevance_explanation', 'Relevant to query')
            confidence = row.get('confidence', 0.8)

            explanations[col_name] = f"{explanation} (confidence: {confidence:.0%})"

        logger.info(f"Generated explanations for {len(explanations)} columns")
        return explanations

    except Exception as e:
        logger.warning(f"AI.GENERATE_TABLE explanation failed: {e}")
        return {}

    finally:
        # Cleanup - using the full table path
        try:
            drop_query = f"DROP TABLE IF EXISTS `{temp_table_path}`"
            client.query(drop_query).result()
            logger.debug(f"Cleaned up temp table: {temp_table_path}")
        except Exception as e:
            logger.warning(f"Failed to clean up temp table {temp_table_path}: {e}")
            # Not critical - table will be auto-cleaned by BigQuery eventually


async def validate_output_schema(schema_str: str) -> bool:
    """
    Validate that the output schema is valid for AI.GENERATE_TABLE
    """
    valid_types = ['STRING', 'INT64', 'FLOAT64', 'BOOL', 'DATE', 'DATETIME', 'TIMESTAMP']

    try:
        lines = schema_str.strip().split(',')
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 2:
                return False
            field_name, field_type = parts
            if field_type not in valid_types:
                return False
        return True
    except (ValueError, AttributeError) as e:
        logger.debug(f"Invalid schema format: {e}")
        return False


# Export functions
generate_structured_insights_with_table = generate_structured_insights_v2
generate_explanations_with_table = generate_explanation_table

# Main exports for use by other modules
generate_structured_insights = generate_structured_insights_v2
generate_column_explanations = generate_explanation_table