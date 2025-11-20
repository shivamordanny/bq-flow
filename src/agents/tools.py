"""
ADK Tools for BQ Flow V2
Wraps existing BigQuery AI functions as ADK tools for agent orchestration
"""

from typing import Dict, Any, List
import json
import pandas as pd

# Import existing BQ Flow functions
from src.core.bigquery_ai import (
    generate_query_embedding,
    vector_search_columns,
    generate_sql_with_context,
    execute_bigquery
)
from src.core.bigquery_ai_forecast import generate_ai_forecast
from src.core.bigquery_ai_generate_table import generate_structured_insights_v2
from src.core.config import get_config
from src.core.logging import get_logger

# ADK imports
from google.genai.types import Tool, FunctionDeclaration

logger = get_logger(__name__)


def create_bq_flow_tools() -> List[Tool]:
    """
    Create ADK tools for BQ Flow agent

    Returns:
        List of Tool objects for ADK agent
    """

    # Tool 1: Search Data (Vector Search)
    search_data_tool = Tool(
        function_declarations=[
            FunctionDeclaration(
                name="search_data",
                description=(
                    "Search for relevant columns in a BigQuery database using semantic vector search. "
                    "Use this when you need to find which columns contain data related to the user's question. "
                    "For example: 'sales', 'customer demographics', 'order history'."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language description of the data to search for"
                        },
                        "database_id": {
                            "type": "string",
                            "description": "Database identifier (e.g., 'ecommerce', 'stackoverflow', 'bikeshare')"
                        }
                    },
                    "required": ["query", "database_id"]
                }
            )
        ]
    )

    # Tool 2: Query Data (SQL Generation + Execution)
    query_data_tool = Tool(
        function_declarations=[
            FunctionDeclaration(
                name="query_data",
                description=(
                    "Generate and execute a SQL query on BigQuery to answer the user's question. "
                    "Use this after searching for relevant columns, or when you have context about the data structure. "
                    "Returns actual data results from BigQuery."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "user_query": {
                            "type": "string",
                            "description": "The user's natural language question"
                        },
                        "database_id": {
                            "type": "string",
                            "description": "Database identifier"
                        },
                        "relevant_columns": {
                            "type": "array",
                            "description": "List of relevant columns from search_data (can be empty if not searched)",
                            "items": {"type": "object"}
                        }
                    },
                    "required": ["user_query", "database_id"]
                }
            )
        ]
    )

    # Tool 3: Forecast Data (Time-Series Prediction)
    forecast_data_tool = Tool(
        function_declarations=[
            FunctionDeclaration(
                name="forecast_data",
                description=(
                    "Generate time-series forecasts using BigQuery AI.FORECAST with TimesFM 2.0. "
                    "Use this when the data shows temporal patterns (dates, timestamps) and user asks for predictions. "
                    "Automatically detects time-series structure and generates forecasts."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "results_data": {
                            "type": "string",
                            "description": "JSON string of query results containing time-series data"
                        },
                        "user_query": {
                            "type": "string",
                            "description": "The user's original question for context"
                        },
                        "database_id": {
                            "type": "string",
                            "description": "Database identifier"
                        },
                        "horizon": {
                            "type": "integer",
                            "description": "Number of periods to forecast (default: 30)",
                            "default": 30
                        }
                    },
                    "required": ["results_data", "user_query", "database_id"]
                }
            )
        ]
    )

    # Tool 4: Generate Insights (Structured Analysis)
    generate_insights_tool = Tool(
        function_declarations=[
            FunctionDeclaration(
                name="generate_insights",
                description=(
                    "Generate structured business insights from query results using AI.GENERATE_TABLE. "
                    "Use this to provide actionable recommendations, trends, and analysis. "
                    "Returns JSON with key insights, confidence scores, and recommendations."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "user_query": {
                            "type": "string",
                            "description": "The user's original question"
                        },
                        "results_data": {
                            "type": "string",
                            "description": "JSON string of query results to analyze"
                        },
                        "database_id": {
                            "type": "string",
                            "description": "Database identifier for context"
                        }
                    },
                    "required": ["user_query", "results_data", "database_id"]
                }
            )
        ]
    )

    return [search_data_tool, query_data_tool, forecast_data_tool, generate_insights_tool]


# Tool implementation functions (called by agent)

async def search_data_impl(query: str, database_id: str) -> Dict[str, Any]:
    """
    Implementation of search_data tool
    """
    try:
        logger.info(f"[TOOL] search_data called: query='{query}', database_id='{database_id}'")

        # Generate embedding for query
        embedding = await generate_query_embedding(query)

        # Perform vector search
        columns, search_method = await vector_search_columns(embedding, database_id)

        logger.info(f"[TOOL] search_data completed: found {len(columns)} columns using {search_method}")

        return {
            "success": True,
            "columns": columns,
            "search_method": search_method,
            "count": len(columns)
        }
    except Exception as e:
        logger.error(f"[TOOL] search_data failed: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "columns": []
        }


async def query_data_impl(user_query: str, database_id: str, relevant_columns: List[Dict] = None) -> Dict[str, Any]:
    """
    Implementation of query_data tool
    """
    try:
        logger.info(f"[TOOL] query_data called: query='{user_query}', database_id='{database_id}'")

        if relevant_columns is None:
            relevant_columns = []

        # Generate SQL
        sql = await generate_sql_with_context(user_query, database_id, relevant_columns)

        # Execute query
        results = await execute_bigquery(sql)

        logger.info(f"[TOOL] query_data completed: {results.get('row_count', 0)} rows returned")

        return {
            "success": True,
            "sql": sql,
            "results": results.get("results", []),
            "row_count": results.get("row_count", 0),
            "column_names": results.get("column_names", [])
        }
    except Exception as e:
        logger.error(f"[TOOL] query_data failed: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "sql": None,
            "results": []
        }


async def forecast_data_impl(results_data: str, user_query: str, database_id: str, horizon: int = 30) -> Dict[str, Any]:
    """
    Implementation of forecast_data tool
    """
    try:
        logger.info(f"[TOOL] forecast_data called: database_id='{database_id}', horizon={horizon}")

        # Parse results data
        if isinstance(results_data, str):
            results = json.loads(results_data)
        else:
            results = results_data

        # Generate forecast
        forecast_result = await generate_ai_forecast(
            query_results=results,
            user_query=user_query,
            database_id=database_id,
            horizon=horizon
        )

        logger.info(f"[TOOL] forecast_data completed: {forecast_result.get('forecast_points', 0)} points forecasted")

        return {
            "success": True,
            "forecast": forecast_result
        }
    except Exception as e:
        logger.error(f"[TOOL] forecast_data failed: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "forecast": None
        }


async def generate_insights_impl(user_query: str, results_data: str, database_id: str) -> Dict[str, Any]:
    """
    Implementation of generate_insights tool
    """
    try:
        logger.info(f"[TOOL] generate_insights called: query='{user_query}', database_id='{database_id}'")

        # Parse results data
        if isinstance(results_data, str):
            results = json.loads(results_data)
        else:
            results = results_data

        # Generate structured insights
        insights = await generate_structured_insights_v2(
            user_query=user_query,
            query_results=results,
            database_id=database_id
        )

        logger.info(f"[TOOL] generate_insights completed: {len(insights.get('insights', []))} insights generated")

        return {
            "success": True,
            "insights": insights
        }
    except Exception as e:
        logger.error(f"[TOOL] generate_insights failed: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "insights": None
        }


# Tool dispatch mapping
TOOL_IMPLEMENTATIONS = {
    "search_data": search_data_impl,
    "query_data": query_data_impl,
    "forecast_data": forecast_data_impl,
    "generate_insights": generate_insights_impl
}


async def execute_tool(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute a tool by name with arguments

    Args:
        tool_name: Name of the tool to execute
        arguments: Dictionary of arguments for the tool

    Returns:
        Tool execution result
    """
    if tool_name not in TOOL_IMPLEMENTATIONS:
        return {
            "success": False,
            "error": f"Unknown tool: {tool_name}"
        }

    impl_func = TOOL_IMPLEMENTATIONS[tool_name]
    return await impl_func(**arguments)
