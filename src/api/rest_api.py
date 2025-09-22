"""
FastAPI Backend for BQ Flow System
Showcases maximum BigQuery AI features with clear tracking
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import asyncio
from google.cloud import bigquery
import json
import os
from dotenv import load_dotenv
from datetime import datetime
import time
import uuid

# Import centralized logging
from src.core.logging import (
    setup_logging, get_logger, set_request_context, clear_request_context,
    log_performance, log_api_request, log_api_response, log_bigquery_operation
)

# Load environment variables
load_dotenv()

# Setup logging
setup_logging(
    log_level=os.getenv('LOG_LEVEL', 'INFO'),
    log_file='logs/api_endpoint.log',
    use_json=True,
    console_output=True
)

# Get logger for this module
logger = get_logger(__name__)

# Import configuration to use centralized settings
from src.core.config import get_config

# Get configuration instance
config = get_config()

# Use client from centralized config instead of creating duplicate
client = config.client
PROJECT_ID = config.get_dataset_config().project_id
DATASET_ID = config.get_dataset_config().metadata_dataset

logger.info(f"Starting API with PROJECT_ID: {PROJECT_ID}, DATASET_ID: {DATASET_ID}")

# Initialize FastAPI app
app = FastAPI(
    title="BQ Flow API",
    description="BQ Flow API showcasing maximum AI features",
    version="3.7"
)

# Enable CORS for Chainlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8506", "http://localhost:8505"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request tracking middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Middleware to log all requests and responses with correlation IDs
    """
    # Generate request ID and set context
    request_id = str(uuid.uuid4())
    context = set_request_context(request_id=request_id)

    # Log request start
    start_time = time.time()

    # Get request body for logging (if JSON)
    body = None
    if request.method in ["POST", "PUT", "PATCH"]:
        try:
            body = await request.body()
            request._body = body  # Store for later use
        except:
            pass

    logger.info(
        f"Request started: {request.method} {request.url.path}",
        extra={
            'extra_fields': {
                'method': request.method,
                'path': request.url.path,
                'query_params': dict(request.query_params),
                'client_host': request.client.host if request.client else None,
                'request_id': request_id
            }
        }
    )

    try:
        # Process request
        response = await call_next(request)

        # Calculate duration
        duration_ms = int((time.time() - start_time) * 1000)

        # Log response
        logger.info(
            f"Request completed: {request.method} {request.url.path} -> {response.status_code}",
            extra={
                'extra_fields': {
                    'status_code': response.status_code,
                    'duration_ms': duration_ms,
                    'request_id': request_id
                }
            }
        )

        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id

        return response

    except Exception as e:
        duration_ms = int((time.time() - start_time) * 1000)

        logger.error(
            f"Request failed: {request.method} {request.url.path}",
            exc_info=True,
            extra={
                'extra_fields': {
                    'error': str(e),
                    'error_type': type(e).__name__,
                    'duration_ms': duration_ms,
                    'request_id': request_id
                }
            }
        )
        raise

    finally:
        # Clear context
        clear_request_context()

# Request/Response Models
class QueryRequest(BaseModel):
    query: str
    database_id: str
    conversation_id: Optional[str] = None
    use_cache: Optional[bool] = True

class RefineRequest(BaseModel):
    query: str
    database_id: str
    conversation_id: str
    previous_sql: str

class QueryResponse(BaseModel):
    sql: str
    results: List[Dict[str, Any]]
    columns_used: List[Dict[str, Any]]
    bigquery_features_used: List[str]
    execution_time_ms: int
    cost_estimate: float
    insights: Optional[Dict[str, Any]] = None
    cache_hit: bool = False

class DatabaseInfo(BaseModel):
    database_id: str
    display_name: str
    description: str
    table_count: Optional[int] = None

class DatabaseListResponse(BaseModel):
    databases: List[DatabaseInfo]

class ForecastRequest(BaseModel):
    sql: str  # The SQL query that generated time-series results
    database_id: str
    date_column: str  # Column containing dates/timestamps
    value_column: str  # Column containing numeric values to forecast
    horizon: Optional[int] = 30  # Number of periods to forecast
    confidence_level: Optional[float] = 0.95  # Confidence level for intervals

class ForecastResponse(BaseModel):
    forecast_results: List[Dict[str, Any]]
    insights: Dict[str, Any]
    model_metrics: Optional[Dict[str, Any]] = None
    bigquery_features_used: List[str]
    execution_time_ms: int
    cost_estimate: float

# Import BigQuery AI functions
from src.core.bigquery_ai import (
    generate_query_embedding,
    vector_search_columns,
    generate_sql_with_context,
    execute_bigquery,
    generate_structured_insights as generate_structured_insights_v1,  # Keep v1 as fallback
    generate_column_explanations,
    calculate_cost
)

# Import v2 insights using AI.GENERATE_TABLE (better structured output)
from src.core.bigquery_ai_generate_table import (
    generate_structured_insights_v2,
    generate_explanation_table
)

# Import BigQuery Forecast functions
# Using new AI.FORECAST module (TimesFM) instead of ML.FORECAST (ARIMA)
from src.core.bigquery_ai_forecast import (
    generate_ai_forecast,
    validate_forecast_data,
    generate_forecast_insights_ai,
    detect_time_series  # Still using the same detection logic
)

# Configuration instance already imported and initialized above

@app.get("/")
@log_performance
async def root():
    """Health check endpoint"""
    logger.debug("Health check requested")

    # Get features from configuration
    features = config.get_bigquery_features()
    feature_names = [f['name'] for f in features]

    response = {
        "status": "healthy",
        "service": "BQ Flow API",
        "version": "5.0",
        "bigquery_features": feature_names
    }

    logger.info("Health check successful", extra={
        'extra_fields': {'features_count': 7}
    })

    return response

@app.get("/api/databases")
@log_performance
async def list_databases() -> DatabaseListResponse:
    """List available databases with metadata from new registry"""

    project_id = PROJECT_ID
    dataset_id = DATASET_ID

    logger.info(f"Listing available databases from {project_id}.{dataset_id} registry")

    try:
        query = f"""
        SELECT
            database_id,
            display_name,
            description,
            COALESCE(table_count, 0) as table_count
        FROM `{project_id}.{dataset_id}.database_registry`
        WHERE is_active = TRUE
        ORDER BY display_name
        """

        # Log BigQuery operation
        query_start = time.time()
        results = client.query(query).to_dataframe()
        query_duration = int((time.time() - query_start) * 1000)

        logger.info(
            f"BigQuery query completed for database listing",
            extra={
                'extra_fields': {
                    'operation': 'list_databases',
                    'rows_returned': len(results),
                    'duration_ms': query_duration
                }
            }
        )

        databases = [
            DatabaseInfo(
                database_id=row['database_id'],
                display_name=row['display_name'],
                description=row['description'],
                table_count=int(row['table_count']) if row['table_count'] else 0
            )
            for _, row in results.iterrows()
        ]

        logger.info(
            f"Successfully retrieved {len(databases)} databases",
            extra={
                'extra_fields': {
                    'database_count': len(databases),
                    'database_ids': [db.database_id for db in databases]
                }
            }
        )

        return DatabaseListResponse(databases=databases)

    except Exception as e:
        logger.error(
            f"Failed to list databases: {str(e)}",
            exc_info=True,
            extra={
                'extra_fields': {
                    'error_type': type(e).__name__,
                    'error_message': str(e)
                }
            }
        )
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/query")
@log_performance
async def process_query(request: QueryRequest) -> QueryResponse:
    """
    Main query processing endpoint showcasing maximum BigQuery AI features
    """
    # Set context for this query
    set_request_context(database_id=request.database_id)

    logger.info(
        f"Processing query for database {request.database_id}",
        extra={
            'extra_fields': {
                'query': request.query[:100],  # First 100 chars
                'database_id': request.database_id,
                'use_cache': request.use_cache,
                'conversation_id': request.conversation_id
            }
        }
    )

    features_used = []
    start_time = datetime.now()
    cache_hit = False
    operation_times = {}

    try:
        # Always generate fresh SQL for accuracy
        # Note: Semantic caching disabled for NL2SQL safety - small query variations need different SQL

        # 1. Generate embedding for query
        embed_start = time.time()
        features_used.append("ML.GENERATE_EMBEDDING")

        logger.debug("Generating query embedding")
        query_embedding = await generate_query_embedding(request.query)

        embed_duration = int((time.time() - embed_start) * 1000)
        operation_times['embedding_generation'] = embed_duration

        logger.info(
            f"Query embedding generated",
            extra={
                'extra_fields': {
                    'embedding_dims': len(query_embedding),
                    'duration_ms': embed_duration
                }
            }
        )

        # 2. Use VECTOR_SEARCH with IVF index
        search_start = time.time()

        logger.debug("Searching for relevant columns")
        # Handle both old format (list) and new format (tuple with feature)
        result = await vector_search_columns(
            query_embedding,
            request.database_id
        )
        if isinstance(result, tuple):
            relevant_columns, vector_feature = result
            # Use the specific vector feature returned (VECTOR_SEARCH, ML.DISTANCE, etc)
            features_used.append(vector_feature)
        else:
            # Old format compatibility - default feature
            relevant_columns = result
            features_used.append("VECTOR_SEARCH")

        search_duration = int((time.time() - search_start) * 1000)
        operation_times['vector_search'] = search_duration

        logger.info(
            f"Found {len(relevant_columns)} relevant columns",
            extra={
                'extra_fields': {
                    'columns_found': len(relevant_columns),
                    'top_column': relevant_columns[0]['column_name'] if relevant_columns else None,
                    'top_score': relevant_columns[0]['similarity_score'] if relevant_columns else 0,
                    'duration_ms': search_duration
                }
            }
        )

        # 3. Use AI.GENERATE for SQL generation
        sql_start = time.time()
        features_used.append("AI.GENERATE (Gemini 2.5 Flash)")

        logger.debug("Generating SQL with AI.GENERATE")
        generated_sql = await generate_sql_with_context(
            request.query,
            request.database_id,
            relevant_columns
        )

        sql_duration = int((time.time() - sql_start) * 1000)
        operation_times['sql_generation'] = sql_duration

        logger.info(
            f"SQL generated successfully",
            extra={
                'extra_fields': {
                    'sql_length': len(generated_sql),
                    'duration_ms': sql_duration
                }
            }
        )

        # 4. Execute the generated SQL
        exec_start = time.time()
        logger.debug(f"Executing SQL query")

        execution_result = await execute_bigquery(generated_sql)

        exec_duration = int((time.time() - exec_start) * 1000)
        operation_times['sql_execution'] = exec_duration

        if not execution_result['success']:
            # Calculate total execution time for error response
            end_time = datetime.now()
            total_execution_time_ms = int((end_time - start_time).total_seconds() * 1000)

            logger.warning(
                f"SQL execution failed - returning graceful error response",
                extra={
                    'extra_fields': {
                        'error': execution_result['error'],
                        'sql': generated_sql[:500],  # First 500 chars
                        'duration_ms': exec_duration
                    }
                }
            )

            # Extract error message for user-friendly response
            error_msg = execution_result['error']

            # Parse common errors
            if 'Unrecognized name' in error_msg:
                # Column name issue - provide helpful message
                user_message = (
                    "I couldn't execute the query due to a schema mismatch. "
                    "The generated SQL referenced columns that don't exist in the database. "
                    "This usually happens when the database metadata is incomplete. "
                    "Please try rephrasing your query or contact support."
                )
            elif 'Syntax error' in error_msg:
                user_message = (
                    "The generated SQL has a syntax error. "
                    "Please try rephrasing your query in simpler terms."
                )
            else:
                user_message = (
                    "I couldn't execute the query successfully. "
                    "Please try rephrasing your question or try a simpler query."
                )

            # Return graceful response instead of throwing error
            return QueryResponse(
                sql=generated_sql,
                results=[],
                columns_used=relevant_columns[:5],
                bigquery_features_used=features_used,
                execution_time_ms=total_execution_time_ms,  # Use calculated value
                cost_estimate=calculate_cost(features_used, len(request.query)),
                insights={
                    "error": True,
                    "message": user_message,
                    "technical_details": error_msg[:200] if len(error_msg) > 200 else error_msg
                },
                cache_hit=cache_hit
            )

        results = execution_result['results']

        logger.info(
            f"SQL executed successfully",
            extra={
                'extra_fields': {
                    'rows_returned': len(results),
                    'duration_ms': exec_duration
                }
            }
        )

        # 5. Generate structured insights (for non-empty results)
        insights = None
        if len(results) > 0 and not cache_hit:
            insights_start = time.time()
            features_used.append("AI.GENERATE_TABLE (Structured Insights Extraction)")

            logger.debug("Generating structured insights using AI.GENERATE_TABLE")
            try:
                # Use v2 with AI.GENERATE_TABLE for better structured output
                insights = await generate_structured_insights_v2(
                    request.query,
                    results[:10],  # Limit to first 10 rows for insights
                    request.database_id
                )
                logger.info("Successfully used AI.GENERATE_TABLE for insights")
            except Exception as v2_error:
                logger.warning(f"AI.GENERATE_TABLE failed, falling back to v1: {v2_error}")
                try:
                    # Fallback to v1 with AI.GENERATE
                    insights = await generate_structured_insights_v1(
                        request.query,
                        results[:10]
                    )
                    # Ensure v1 response has all the new fields for consistency
                    if not insights.get('trend_direction'):
                        insights['trend_direction'] = 'STABLE'
                    if not insights.get('business_impact'):
                        insights['business_impact'] = insights.get('recommendation', '')
                    if not insights.get('action_priority'):
                        insights['action_priority'] = 'MEDIUM'
                    if not insights.get('implementation_timeline'):
                        insights['implementation_timeline'] = 'SHORT_TERM'
                    if not insights.get('confidence_score'):
                        insights['confidence_score'] = 0.75
                    if not insights.get('data_quality'):
                        insights['data_quality'] = 'Good'
                    if not insights.get('risk_assessment'):
                        insights['risk_assessment'] = 'Based on available data'
                    if not insights.get('opportunity_highlight'):
                        insights['opportunity_highlight'] = 'Review insights for opportunities'
                    logger.info("Successfully used AI.GENERATE (v1) as fallback")
                except Exception as v1_error:
                    logger.error(f"Both v2 and v1 insights generation failed: v2={v2_error}, v1={v1_error}")
                    raise

                insights_duration = int((time.time() - insights_start) * 1000)
                operation_times['insights_generation'] = insights_duration

                logger.info(
                    f"Insights generated successfully",
                    extra={
                        'extra_fields': {
                            'has_key_finding': bool(insights.get('key_finding')),
                            'has_recommendation': bool(insights.get('recommendation')),
                            'duration_ms': insights_duration
                        }
                    }
                )
            except Exception as e:
                insights_duration = int((time.time() - insights_start) * 1000)
                logger.warning(
                    f"Could not generate insights: {str(e)}",
                    extra={
                        'extra_fields': {
                            'error': str(e),
                            'duration_ms': insights_duration
                        }
                    }
                )

        # 6. Optional: Generate AI explanations for transparency
        explanations = None
        if len(relevant_columns) > 0 and not cache_hit:
            features_used.append("AI-Generated Explanations")
            try:
                explanations = await generate_column_explanations(
                    request.query,
                    relevant_columns[:3]  # Top 3 columns
                )
                # Add explanations to columns
                for col in relevant_columns[:3]:
                    col['explanation'] = explanations.get(
                        f"{col['table_name']}.{col['column_name']}",
                        "No explanation available"
                    )
            except Exception as e:
                logger.warning(f"Could not generate explanations: {e}")

        # Calculate execution time
        end_time = datetime.now()
        execution_time_ms = int((end_time - start_time).total_seconds() * 1000)

        # Calculate cost estimate
        cost_estimate = calculate_cost(features_used, len(request.query))

        # Format results for response
        formatted_results = []
        for row in results[:100]:  # Limit to 100 rows for API response
            formatted_results.append({
                k: str(v) if v is not None else None
                for k, v in row.items()
            })

        # Log final summary
        logger.info(
            f"Query processing completed successfully",
            extra={
                'extra_fields': {
                    'total_duration_ms': execution_time_ms,
                    'features_used': features_used,
                    'features_count': len(features_used),
                    'cache_hit': cache_hit,
                    'rows_returned': len(results),
                    'cost_estimate': cost_estimate,
                    'operation_times': operation_times,
                    'database_id': request.database_id,
                    'query_length': len(request.query)
                }
            }
        )

        return QueryResponse(
            sql=generated_sql,
            results=formatted_results,
            columns_used=relevant_columns[:5],  # Top 5 columns
            bigquery_features_used=features_used,
            execution_time_ms=execution_time_ms,
            cost_estimate=cost_estimate,
            insights=insights,
            cache_hit=cache_hit
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/refine")
async def refine_query(request: RefineRequest) -> QueryResponse:
    """
    Refine a previous query with context
    Uses conversation history for better refinement
    """
    features_used = ["Query Refinement with Context"]

    try:
        # Build context from previous query
        context = f"""
        Previous SQL: {request.previous_sql}
        Refinement Request: {request.query}
        Database: {request.database_id}
        """

        # Use AI.GENERATE to refine the SQL
        features_used.append("AI.GENERATE (Query Refinement)")

        refined_request = QueryRequest(
            query=context,
            database_id=request.database_id,
            conversation_id=request.conversation_id,
            use_cache=False  # Don't use cache for refinements
        )

        # Process as a new query with context
        return await process_query(refined_request)

    except Exception as e:
        logger.error(f"Error refining query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/forecast")
@log_performance
async def forecast_time_series(request: ForecastRequest) -> ForecastResponse:
    """
    Generate time-series forecast using AI.FORECAST with TimesFM
    No model creation needed - direct data-to-forecast approach
    """
    start_time = time.time()
    features_used = []

    try:
        logger.info(
            f"AI.FORECAST request received",
            extra={
                'extra_fields': {
                    'database_id': request.database_id,
                    'date_column': request.date_column,
                    'value_column': request.value_column,
                    'horizon': request.horizon,
                    'method': 'AI.FORECAST'
                }
            }
        )

        # Step 1: Validate data (optional but recommended)
        logger.info("Validating data for AI.FORECAST")
        is_valid, validation_msg = await validate_forecast_data(
            sql_query=request.sql,
            date_column=request.date_column,
            value_column=request.value_column,
            min_data_points=10  # TimesFM can work with fewer points than ARIMA
        )

        if not is_valid:
            raise ValueError(f"Data validation failed: {validation_msg}")

        logger.debug(f"Data validation passed: {validation_msg}")

        # Step 2: Generate forecast directly with AI.FORECAST
        logger.info(f"Generating {request.horizon}-period forecast using TimesFM 2.0")
        features_used.append(f"AI.FORECAST (TimesFM 2.0, horizon={request.horizon})")

        forecast_results = await generate_ai_forecast(
            sql_query=request.sql,
            date_column=request.date_column,
            value_column=request.value_column,
            horizon=request.horizon,
            confidence_level=request.confidence_level,
            model="TimesFM 2.0"
        )

        # Step 3: Generate insights
        logger.info("Generating AI-powered forecast insights")
        insights = await generate_forecast_insights_ai(
            forecast_results=forecast_results,
            original_query=request.sql,
            value_column=request.value_column,
            horizon=request.horizon
        )
        features_used.append("AI-Generated Insights")

        # Step 4: Calculate cost (lower than ML.FORECAST since no model creation)
        # AI.FORECAST is typically cheaper as it doesn't require model storage
        cost_estimate = calculate_cost(features_used, len(request.sql)) * 0.7  # 30% cheaper

        # Note: No model cleanup needed with AI.FORECAST!
        # This is a major advantage over ML.FORECAST

        duration_ms = int((time.time() - start_time) * 1000)

        logger.info(
            f"AI.FORECAST completed successfully",
            extra={
                'extra_fields': {
                    'forecast_points': len(forecast_results),
                    'duration_ms': duration_ms,
                    'features_used': features_used,
                    'model': 'TimesFM 2.0'
                }
            }
        )

        return ForecastResponse(
            forecast_results=forecast_results,
            insights=insights,
            model_metrics={"model": "TimesFM 2.0", "type": "Foundation Model", "pre_trained": True},
            bigquery_features_used=features_used,
            execution_time_ms=duration_ms,
            cost_estimate=cost_estimate
        )

    except Exception as e:
        logger.error(
            f"AI.FORECAST error: {str(e)}",
            exc_info=True,
            extra={
                'extra_fields': {
                    'error_type': type(e).__name__,
                    'database_id': request.database_id,
                    'method': 'AI.FORECAST'
                }
            }
        )

        # No model cleanup needed with AI.FORECAST

        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate AI forecast: {str(e)}"
        )

@app.get("/api/features")
async def get_features():
    """
    Endpoint to showcase all BigQuery AI features being used
    """
    # Get features from configuration
    features = config.get_bigquery_features()

    # Create structured response from config features
    feature_dict = {}
    for feature in features:
        feature_name = feature['name'].lower().replace(' ', '_').replace('.', '_')
        if 'ML.GENERATE_EMBEDDING' in feature['name']:
            feature_dict['embeddings'] = {
                "name": feature['name'],
                "model": feature.get('model', 'text-embedding-005'),
                "dimensionality": feature.get('dimensionality', 768),
                "purpose": feature.get('description', 'Convert queries to vectors')
            }
        elif 'VECTOR_SEARCH' in feature['name']:
            feature_dict['vector_search'] = {
                "name": feature['name'],
                "index_type": feature.get('index_type', 'IVF'),
                "distance_type": feature.get('distance_type', 'COSINE'),
                "purpose": feature.get('description', 'Semantic column discovery')
            }
        elif 'AI.GENERATE' in feature['name'] and 'TABLE' not in feature['name']:
            feature_dict['sql_generation'] = {
                "name": feature['name'],
                "model": feature.get('model', config.get_model_endpoint('generation')),
                "purpose": feature.get('description', 'Natural language to SQL')
            }
        elif 'AI.GENERATE_TABLE' in feature['name']:
            feature_dict['insights'] = {
                "name": feature['name'],
                "model": feature.get('model', config.get_model_endpoint('insight')),
                "purpose": feature.get('description', 'Structured insights extraction')
            }
        elif 'AI.FORECAST' in feature['name']:
            feature_dict['forecast'] = {
                "name": feature['name'],
                "model": feature.get('model', config._config.get('forecast', {}).get('model', 'TimesFM 2.0')),
                "purpose": feature.get('description', 'Time-series prediction')
            }
        elif 'Explanations' in feature['name']:
            feature_dict['explainability'] = {
                "name": feature['name'],
                "purpose": feature.get('description', 'Transparency through AI-powered insights')
            }
        elif 'Cache' in feature['name']:
            feature_dict['caching'] = {
                "name": feature['name'],
                "purpose": feature.get('description', 'Performance optimization')
            }
        elif 'Cost' in feature['name']:
            feature_dict['cost_tracking'] = {
                "name": feature['name'],
                "purpose": feature.get('description', 'Resource awareness')
            }

    return {
        "features": feature_dict,
        "total_features": len(features),
        "status": "All features operational"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)