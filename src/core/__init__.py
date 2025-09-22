"""
Core Module - BQ Flow Business Logic
Contains BigQuery AI functions, configuration management, logging, utilities, and forecasting
"""

from . import bigquery_ai
from . import bigquery_ai_forecast
from . import config
from . import logging
from . import utils

# Export key functions for easier imports
from .bigquery_ai import (
    generate_query_embedding,
    vector_search_columns,
    generate_sql_with_context,
    execute_bigquery,
    generate_structured_insights,
    generate_column_explanations
)

# Export AI.FORECAST functions (new)
from .bigquery_ai_forecast import (
    generate_ai_forecast,
    validate_forecast_data,
    generate_forecast_insights_ai,
    detect_time_series
)

from .config import get_config, reload_config
from .logging import get_logger, setup_logging
from .utils import (
    json_serial,
    clean_column_data,
    convert_numpy_types,
    clean_dataframe_for_json,
    safe_json_dumps
)

__all__ = [
    # Modules
    "bigquery_ai",
    "bigquery_ai_forecast",
    "config",
    "logging",
    "utils",
    # Key BigQuery AI functions
    "generate_query_embedding",
    "vector_search_columns",
    "generate_sql_with_context",
    "execute_bigquery",
    "generate_structured_insights",
    "generate_column_explanations",
    # AI.FORECAST functions
    "generate_ai_forecast",
    "validate_forecast_data",
    "generate_forecast_insights_ai",
    "detect_time_series",
    # Config and logging
    "get_config",
    "reload_config",
    "get_logger",
    "setup_logging",
    # Utility functions
    "json_serial",
    "clean_column_data",
    "convert_numpy_types",
    "clean_dataframe_for_json",
    "safe_json_dumps"
]