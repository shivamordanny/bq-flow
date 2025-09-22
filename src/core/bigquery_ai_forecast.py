"""
BigQuery AI.FORECAST Module
Implements predictive forecasting using Google's TimesFM foundation model
No model creation or management required - direct data-to-forecast
"""

from google.cloud import bigquery
from typing import Dict, List, Any, Optional, Tuple
import time
from datetime import datetime
import pandas as pd
import numpy as np
import json

# Import centralized logging
from .logging import (
    get_logger, log_performance, log_bigquery_operation
)

# Import configuration
from .config import get_config

# Import centralized utilities
from .utils import json_serial, convert_numpy_types

# Get logger for this module
logger = get_logger(__name__)

# Get configuration
config = get_config()
client = config.client


@log_performance
def detect_time_series(results: List[Dict[str, Any]], min_data_points: int = 20) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Detect if query results contain time-series data suitable for forecasting

    Returns:
        Tuple of (is_forecastable, date_column, value_column)
    """
    logger.debug(f"Detecting time-series in {len(results)} rows")

    if not results or len(results) < min_data_points:
        logger.info(f"Insufficient data points: {len(results)} < {min_data_points}")
        return False, None, None

    # Get first row to analyze column types
    first_row = results[0]

    # Find date/timestamp columns
    date_columns = []
    for col_name, col_value in first_row.items():
        if col_value is None:
            continue

        # Check various date/time types
        value_type = str(type(col_value).__name__).lower()

        if any(dt in value_type for dt in ['date', 'timestamp', 'datetime']):
            date_columns.append(col_name)
        elif any(dt in col_name.lower() for dt in ['date', 'time', 'day', 'month', 'year', 'period']):
            # Check if it looks like a date string
            try:
                pd.to_datetime(col_value)
                date_columns.append(col_name)
            except (ValueError, TypeError):
                # Not a date, continue checking other columns
                pass

    # Find numeric columns (excluding obvious IDs)
    numeric_columns = []
    for col_name, col_value in first_row.items():
        if col_value is None:
            continue

        # Skip likely ID columns
        if any(id_term in col_name.lower() for id_term in ['id', 'key', 'code']):
            continue

        # Check if numeric
        if isinstance(col_value, (int, float, np.integer, np.floating)):
            numeric_columns.append(col_name)

    logger.info(
        f"Time-series detection results",
        extra={
            'extra_fields': {
                'date_columns': date_columns,
                'numeric_columns': numeric_columns,
                'row_count': len(results)
            }
        }
    )

    if date_columns and numeric_columns:
        # Return first date column and first numeric column
        return True, date_columns[0], numeric_columns[0]

    return False, None, None


def expand_sql_for_forecast(sql_query: str, date_column: str, min_months: int = 12) -> str:
    """
    Expand SQL query to ensure sufficient historical data for forecasting.
    Removes or modifies restrictive WHERE clauses on dates.

    Args:
        sql_query: Original SQL query
        date_column: Name of the date column (as it appears in SELECT, may be an alias)
        min_months: Minimum months of historical data needed

    Returns:
        Modified SQL with expanded date range
    """
    import re

    # Clean the SQL
    sql = sql_query.rstrip(';')

    # Check if this query has CTEs (WITH clauses) or FORMAT_DATE
    has_cte = bool(re.search(r'\bWITH\s+\w+\s+AS\s*\(', sql, re.IGNORECASE))
    has_format_date = bool(re.search(r'FORMAT_DATE\s*\(', sql, re.IGNORECASE))

    if has_cte or has_format_date:
        # For complex queries with CTEs or FORMAT_DATE, don't try to modify
        # These queries often have complex structures that are hard to modify safely
        logger.debug("Complex query detected (CTE or FORMAT_DATE), using original SQL for forecast")
        return sql

    # For simpler queries, try to find and expand date restrictions
    actual_date_column = date_column  # Default to the provided column name

    # Try to find the actual column if date_column is an alias
    alias_pattern = re.compile(
        r'([A-Za-z_][A-Za-z0-9_\.\(\),\s]*?)\s+as\s+' + re.escape(date_column),
        re.IGNORECASE | re.DOTALL
    )
    alias_match = alias_pattern.search(sql)

    if alias_match:
        actual_expr = alias_match.group(1)
        # Check if it's a function like DATE_TRUNC(created_at, MONTH)
        func_match = re.search(r'DATE_TRUNC\s*\(\s*([A-Za-z_][A-Za-z0-9_\.]*)', actual_expr, re.IGNORECASE)
        if func_match:
            actual_date_column = func_match.group(1)
            logger.info(f"Detected actual date column '{actual_date_column}' from DATE_TRUNC alias '{date_column}'")
        elif '(' not in actual_expr:
            # Simple alias: col AS alias
            actual_date_column = actual_expr.strip()
            logger.info(f"Detected actual date column '{actual_date_column}' from alias '{date_column}'")

    # Look for restrictive WHERE clauses on the date column
    # Pattern 1: WHERE DATE_TRUNC(column, MONTH) = 'specific_date'
    pattern1 = rf"WHERE\s+DATE_TRUNC\s*\(\s*{re.escape(actual_date_column)}\s*,\s*MONTH\s*\)\s*=\s*['\"]?\d{{4}}-\d{{2}}-\d{{2}}['\"]?"

    # Pattern 2: WHERE column >= 'recent_date' (within last 3 months)
    pattern2 = rf"WHERE\s+{re.escape(actual_date_column)}\s*>=\s*['\"]?202[4-9]-\d{{2}}-\d{{2}}['\"]?"

    # Check if we have a restrictive WHERE clause
    has_restrictive_where = False
    for pattern in [pattern1, pattern2]:
        if re.search(pattern, sql, re.IGNORECASE):
            has_restrictive_where = True
            # Replace with expanded date range
            replacement = f"WHERE DATE({actual_date_column}) >= DATE_SUB(CURRENT_DATE(), INTERVAL {min_months} MONTH)"
            sql = re.sub(pattern, replacement, sql, flags=re.IGNORECASE)
            logger.info(f"Expanded SQL date range to last {min_months} months for forecast")
            break

    # If no modification was made and there's no WHERE clause, return original
    if not has_restrictive_where:
        logger.debug("No restrictive WHERE clause found, using original SQL")

    return sql

@log_performance
async def generate_ai_forecast(
    sql_query: str,
    date_column: str,
    value_column: str,
    horizon: int = 30,
    confidence_level: float = 0.95,
    model: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Generate forecast using AI.FORECAST with TimesFM foundation model

    This is a model-free approach that doesn't require creating or managing models.
    TimesFM is pre-trained on billions of time-points and can be applied directly.

    Args:
        sql_query: SQL query that returns time-series data
        date_column: Name of the timestamp/date column
        value_column: Name of the numeric value column to forecast
        horizon: Number of time points to forecast (1-10000)
        confidence_level: Confidence level for prediction intervals (0-1)
        model: Model name (defaults to config if not specified)

    Returns:
        List of forecast results with predictions and confidence intervals
    """
    # Use model from config if not specified
    if model is None:
        model = config._config.get('forecast', {}).get('model', 'TimesFM 2.0')

    logger.info(f"Generating AI forecast with horizon={horizon}, confidence={confidence_level}, model={model}")

    # Validate parameters
    if horizon < 1 or horizon > 10000:
        raise ValueError(f"Horizon must be between 1 and 10000, got {horizon}")

    if confidence_level <= 0 or confidence_level >= 1:
        raise ValueError(f"Confidence level must be between 0 and 1, got {confidence_level}")

    # Expand SQL to ensure sufficient historical data for forecasting
    expanded_sql = expand_sql_for_forecast(sql_query, date_column, min_months=12)

    # Clean the SQL query
    clean_sql = expanded_sql.rstrip(';')

    # Determine if we need to parse date strings or cast dates
    # Check if the query uses FORMAT_DATE (which produces strings)
    import re
    # Use a simpler pattern that works with nested parentheses
    has_format_date = bool(re.search(rf"FORMAT_DATE.*?AS\s+{re.escape(date_column)}", sql_query, re.IGNORECASE | re.DOTALL))

    if has_format_date:
        # FORMAT_DATE with '%Y-%m' produces strings like '2024-01'
        # We need to parse them back to dates for AI.FORECAST
        date_conversion = f"PARSE_DATE('%Y-%m', {date_column})"
        logger.debug(f"Using PARSE_DATE for FORMAT_DATE column {date_column}")
    else:
        # Standard DATE/TIMESTAMP columns can be cast
        date_conversion = f"CAST({date_column} AS DATE)"
        logger.debug(f"Using CAST for date column {date_column}")

    # Build the AI.FORECAST query using WITH clause for the input data
    forecast_query = f"""
    WITH input_data AS (
        SELECT
            {date_conversion} as timestamp_col,
            {value_column} as data_col
        FROM ({clean_sql})
        WHERE {date_column} IS NOT NULL
        AND {value_column} IS NOT NULL
        ORDER BY {date_column}
    )
    SELECT
        forecast_timestamp,
        forecast_value,
        prediction_interval_lower_bound,
        prediction_interval_upper_bound
    FROM AI.FORECAST(
        TABLE input_data,
        data_col => 'data_col',
        timestamp_col => 'timestamp_col',
        model => '{model}',
        horizon => {horizon},
        confidence_level => {confidence_level}
    )
    ORDER BY forecast_timestamp
    """

    start_time = time.time()

    try:
        # Execute the AI.FORECAST query
        logger.debug(f"Executing AI.FORECAST query for {horizon} periods")
        results = client.query(forecast_query).to_dataframe()
        duration_ms = int((time.time() - start_time) * 1000)

        log_bigquery_operation(
            logger,
            "AI.FORECAST",
            {
                'model': model,
                'horizon': horizon,
                'confidence_level': confidence_level,
                'rows_generated': len(results)
            },
            duration_ms
        )

        # Convert results to standardized format
        forecast_results = []
        for _, row in results.iterrows():
            forecast_results.append({
                'forecast_date': convert_numpy_types(row['forecast_timestamp']),
                'predicted_value': convert_numpy_types(row['forecast_value']),
                'lower_bound': convert_numpy_types(row['prediction_interval_lower_bound']),
                'upper_bound': convert_numpy_types(row['prediction_interval_upper_bound']),
                'confidence_level': confidence_level
            })

        logger.info(
            f"AI.FORECAST completed successfully",
            extra={
                'extra_fields': {
                    'forecast_points': len(forecast_results),
                    'duration_ms': duration_ms,
                    'model': model
                }
            }
        )

        return forecast_results

    except Exception as e:
        logger.error(
            f"Failed to generate AI forecast: {str(e)}",
            exc_info=True,
            extra={
                'extra_fields': {
                    'model': model,
                    'horizon': horizon,
                    'error_type': type(e).__name__
                }
            }
        )
        raise


@log_performance
async def validate_forecast_data(
    sql_query: str,
    date_column: str,
    value_column: str,
    min_data_points: int = 10
) -> Tuple[bool, str]:
    """
    Validate that the data is suitable for AI.FORECAST

    Args:
        sql_query: SQL query to validate
        date_column: Date/timestamp column name
        value_column: Numeric value column name
        min_data_points: Minimum required data points

    Returns:
        Tuple of (is_valid, message)
    """
    logger.debug("Validating data for AI.FORECAST")

    # Expand SQL to ensure sufficient historical data for validation
    expanded_sql = expand_sql_for_forecast(sql_query, date_column, min_months=12)

    # Clean the SQL query - remove trailing semicolon if present
    clean_sql = expanded_sql.rstrip().rstrip(';')

    try:
        # Check data points count
        count_query = f"""
        SELECT COUNT(*) as row_count
        FROM ({clean_sql}) AS subquery
        WHERE subquery.{date_column} IS NOT NULL
        AND subquery.{value_column} IS NOT NULL
        """

        result = list(client.query(count_query).result())[0]
        row_count = result.row_count

        if row_count < min_data_points:
            return False, f"Insufficient data points: {row_count} < {min_data_points}"

        # Check for regular time intervals (optional but recommended)
        # Try to determine if date_column is a DATE or STRING type
        # If it's a string (like from FORMAT_DATE), we need to handle it differently

        try:
            # First, try with DATE casting (for DATE/TIMESTAMP columns)
            interval_query = f"""
            WITH data_source AS ({clean_sql}),
            time_diffs AS (
                SELECT
                    DATE_DIFF(
                        CAST(LEAD(data_source.{date_column}) OVER (ORDER BY data_source.{date_column}) AS DATE),
                        CAST(data_source.{date_column} AS DATE),
                        DAY
                    ) as day_diff
                FROM data_source
                WHERE data_source.{date_column} IS NOT NULL
                AND data_source.{value_column} IS NOT NULL
            )
            SELECT
                COUNT(DISTINCT day_diff) as unique_intervals,
                MIN(day_diff) as min_interval,
                MAX(day_diff) as max_interval,
                AVG(day_diff) as avg_interval
            FROM time_diffs
            WHERE day_diff IS NOT NULL
            """

            interval_result = list(client.query(interval_query).result())[0]

        except Exception as cast_error:
            # If casting fails, try parsing as string (for FORMAT_DATE results)
            logger.debug(f"Date casting failed, trying string parsing: {cast_error}")

            interval_query = f"""
            WITH data_source AS ({clean_sql}),
            time_diffs AS (
                SELECT
                    DATE_DIFF(
                        PARSE_DATE('%Y-%m', LEAD(data_source.{date_column}) OVER (ORDER BY data_source.{date_column})),
                        PARSE_DATE('%Y-%m', data_source.{date_column}),
                        DAY
                    ) as day_diff
                FROM data_source
                WHERE data_source.{date_column} IS NOT NULL
                AND data_source.{value_column} IS NOT NULL
            )
            SELECT
                COUNT(DISTINCT day_diff) as unique_intervals,
                MIN(day_diff) as min_interval,
                MAX(day_diff) as max_interval,
                AVG(day_diff) as avg_interval
            FROM time_diffs
            WHERE day_diff IS NOT NULL
            """

            try:
                interval_result = list(client.query(interval_query).result())[0]
            except:
                # If both fail, skip interval checking
                logger.warning("Could not determine date intervals, skipping interval check")
                interval_result = type('obj', (object,), {
                    'unique_intervals': None,
                    'avg_interval': 30  # Assume monthly
                })()


        logger.info(
            f"Data validation complete",
            extra={
                'extra_fields': {
                    'row_count': row_count,
                    'unique_intervals': interval_result.unique_intervals,
                    'avg_interval_days': float(interval_result.avg_interval) if interval_result.avg_interval else None
                }
            }
        )

        return True, f"Data validated: {row_count} points with average interval of {interval_result.avg_interval:.1f} days"

    except Exception as e:
        logger.error(f"Data validation failed: {e}")
        return False, f"Validation error: {str(e)}"


async def generate_forecast_insights_ai(
    forecast_results: List[Dict[str, Any]],
    original_query: str,
    value_column: str,
    horizon: int
) -> Dict[str, Any]:
    """
    Generate AI-powered insights for forecast results using AI.GENERATE

    Args:
        forecast_results: List of forecast predictions
        original_query: The original SQL query used
        value_column: Name of the value column being forecast
        horizon: Forecast horizon used

    Returns:
        Dictionary with structured insights
    """
    if not forecast_results:
        return {
            "summary": "No forecast results to analyze",
            "trend": "N/A",
            "confidence": "N/A",
            "recommendations": []
        }

    # Calculate comprehensive statistics
    predictions = [r['predicted_value'] for r in forecast_results]
    lower_bounds = [r['lower_bound'] for r in forecast_results]
    upper_bounds = [r['upper_bound'] for r in forecast_results]

    avg_prediction = sum(predictions) / len(predictions)
    max_prediction = max(predictions)
    min_prediction = min(predictions)

    # Calculate volatility
    variance = sum((p - avg_prediction) ** 2 for p in predictions) / len(predictions)
    std_deviation = variance ** 0.5
    volatility_index = (std_deviation / avg_prediction * 100) if avg_prediction != 0 else 0

    # Calculate growth rate
    if len(predictions) > 1:
        growth_rate = ((predictions[-1] - predictions[0]) / predictions[0] * 100) if predictions[0] != 0 else 0
    else:
        growth_rate = 0

    # Determine trend with more nuance
    first_third = predictions[:len(predictions)//3] if len(predictions) >= 3 else predictions
    last_third = predictions[-len(predictions)//3:] if len(predictions) >= 3 else predictions
    first_third_avg = sum(first_third) / len(first_third)
    last_third_avg = sum(last_third) / len(last_third)

    if last_third_avg > first_third_avg * 1.10:
        trend = "Strong Upward"
        trend_emoji = "📈"
    elif last_third_avg > first_third_avg * 1.03:
        trend = "Moderate Upward"
        trend_emoji = "📈"
    elif last_third_avg < first_third_avg * 0.90:
        trend = "Strong Downward"
        trend_emoji = "📉"
    elif last_third_avg < first_third_avg * 0.97:
        trend = "Moderate Downward"
        trend_emoji = "📉"
    else:
        trend = "Stable"
        trend_emoji = "➡️"

    # Calculate confidence interval analysis
    avg_interval_width = sum([(u - l) for u, l in zip(upper_bounds, lower_bounds)]) / len(forecast_results)
    confidence_ratio = (avg_interval_width / avg_prediction * 100) if avg_prediction != 0 else 0

    # Assess risk level
    if confidence_ratio < 10 and volatility_index < 20:
        risk_level = "Low"
    elif confidence_ratio < 25 and volatility_index < 40:
        risk_level = "Moderate"
    else:
        risk_level = "High"

    # Identify critical periods
    critical_periods = []
    for i, (pred, lower, upper) in enumerate(zip(predictions, lower_bounds, upper_bounds)):
        # Mark periods with high uncertainty or extreme values
        interval_width = upper - lower
        if interval_width > avg_interval_width * 1.5:
            critical_periods.append(f"Period {i+1}: High uncertainty")
        elif pred == max_prediction:
            critical_periods.append(f"Period {i+1}: Peak value ({pred:.2f})")
        elif pred == min_prediction:
            critical_periods.append(f"Period {i+1}: Minimum value ({pred:.2f})")

    # Limit critical periods to top 3
    critical_periods = critical_periods[:3]

    # Prepare data for AI.GENERATE
    forecast_context = {
        "metric_name": value_column,
        "forecast_horizon": horizon,
        "trend": trend,
        "growth_rate_percent": round(growth_rate, 2),
        "volatility_index": round(volatility_index, 2),
        "average_value": round(avg_prediction, 2),
        "min_value": round(min_prediction, 2),
        "max_value": round(max_prediction, 2),
        "confidence_interval_ratio": round(confidence_ratio, 2),
        "risk_level": risk_level,
        "sample_predictions": [round(p, 2) for p in predictions[:5]]  # First 5 predictions
    }

    # Create prompt for AI.GENERATE
    prompt = f"""
    Analyze this time-series forecast for {value_column} and provide specific business insights:

    Forecast Data:
    {json.dumps(forecast_context, indent=2)}

    Based on this forecast analysis, provide:
    1. One key business finding specific to {value_column}
    2. Three actionable recommendations that are specific to this metric and trend
    3. Suggested monitoring thresholds based on the prediction values
    4. Two follow-up analysis questions

    Consider the {trend} trend with {growth_rate:.1f}% growth rate and {risk_level} risk level.

    Format response as JSON with keys:
    - key_finding (string)
    - recommendations (array of 3 specific actions)
    - monitoring_thresholds (object with upper_alert and lower_alert values)
    - follow_up_questions (array of 2 questions)
    """

    try:
        # Use AI.GENERATE for intelligent insights
        # Update the prompt to request JSON format
        json_prompt = f"""{prompt}

Please format your response as valid JSON with the exact structure requested above."""

        insights_query = f'''
        SELECT AI.GENERATE(
            """{json_prompt}""",
            connection_id => '{config.get_connection_id('gemini')}',
            endpoint => '{config.get_model_endpoint('insight')}'
        ).result AS ai_response
        '''

        # Execute the query
        result = list(client.query(insights_query).result())[0]

        # Parse AI-generated response
        ai_response_text = result.ai_response.strip() if result.ai_response else ""

        # Clean up JSON from markdown if present
        if '```json' in ai_response_text:
            ai_response_text = ai_response_text.split('```json')[1].split('```')[0].strip()
        elif '```' in ai_response_text:
            ai_response_text = ai_response_text.split('```')[1].split('```')[0].strip()

        # Parse the JSON
        ai_response = json.loads(ai_response_text) if ai_response_text else {}

        # Extract insights with fallbacks
        ai_insights = {
            "key_finding": ai_response.get("key_finding", f"The {value_column} shows a {trend.lower()} trend with {growth_rate:.1f}% change"),
            "recommendations": ai_response.get("recommendations", [
                f"Monitor {value_column} for deviations from the {trend.lower()} trend",
                f"Prepare for {risk_level.lower()} volatility based on {volatility_index:.1f}% variance",
                f"Set alerts at {min_prediction:.2f} (lower) and {max_prediction:.2f} (upper) bounds"
            ]),
            "monitoring_thresholds": ai_response.get("monitoring_thresholds", {
                "upper_alert": round(avg_prediction * 1.2, 2),
                "lower_alert": round(avg_prediction * 0.8, 2)
            }),
            "follow_up_questions": ai_response.get("follow_up_questions", [
                f"What factors could cause {value_column} to deviate from the predicted trend?",
                f"How does this {value_column} forecast compare to historical patterns?"
            ])
        }

    except Exception as e:
        logger.error(f"AI.GENERATE failed: {e}")
        # Re-raise the exception to fail properly instead of using fallback
        # This ensures we rely on core functionality only
        raise Exception(f"Failed to generate AI insights for forecast: {e}")

    # Compile final insights
    insights = {
        "summary": f"{config._config.get('forecast', {}).get('model', 'TimesFM 2.0')} projects {trend.lower()} trajectory for {value_column} with {risk_level.lower()} confidence",
        "trend": f"{trend_emoji} {trend} ({growth_rate:+.1f}%)",
        "statistics": {
            "average_prediction": round(avg_prediction, 2),
            "max_prediction": round(max_prediction, 2),
            "min_prediction": round(min_prediction, 2),
            "volatility_index": round(volatility_index, 2),
            "growth_rate": round(growth_rate, 2),
            "confidence_ratio": round(confidence_ratio, 2)
        },
        "risk_assessment": {
            "level": risk_level,
            "volatility": f"{volatility_index:.1f}%",
            "confidence_width": f"{confidence_ratio:.1f}%"
        },
        "key_insights": [
            ai_insights["key_finding"],
            f"Expected value range: {min_prediction:.2f} to {max_prediction:.2f}",
            f"Risk level: {risk_level} (volatility: {volatility_index:.1f}%, confidence: {confidence_ratio:.1f}%)"
        ],
        "recommendations": ai_insights["recommendations"],
        "monitoring": {
            "thresholds": ai_insights["monitoring_thresholds"],
            "critical_periods": critical_periods if critical_periods else ["No critical periods identified"]
        },
        "follow_up_questions": ai_insights["follow_up_questions"],
        "model_info": {
            "model": config._config.get('forecast', {}).get('model', 'TimesFM 2.0'),
            "type": "Foundation Model",
            "confidence_level": "95%"
        }
    }

    logger.info(f"Generated enhanced forecast insights with {trend} trend and {risk_level} risk")

    return insights


# Export the main functions
__all__ = [
    'generate_ai_forecast',
    'validate_forecast_data',
    'generate_forecast_insights_ai',
    'detect_time_series',  # Re-exported from original module
    'expand_sql_for_forecast'  # Helper to expand SQL date ranges
]