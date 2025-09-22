"""
Streaming Query Processor
Wraps existing BigQuery AI functions with progress updates
"""

import asyncio
import time
from typing import Dict, Any, List, Optional
from fastapi import WebSocket
import logging

# Import components
from .connection_manager import ConnectionManager
from .progress_tracker import ProgressTracker, QueryStage, SQL_BUILD_STAGES

# Import existing BigQuery functions
from src.core.bigquery_ai import (
    generate_query_embedding,
    vector_search_columns,
    generate_sql_with_context,
    execute_bigquery,
    generate_structured_insights as generate_structured_insights_v1  # Keep v1 as fallback
)

# Import v2 insights using AI.GENERATE_TABLE (better structured output)
from src.core.bigquery_ai_generate_table import (
    generate_structured_insights_v2
)

logger = logging.getLogger(__name__)


class StreamingQueryProcessor:
    """
    Processes queries with real-time progress updates via WebSocket.
    This class wraps ALL existing BigQuery AI functions without modifying them.
    """

    def __init__(self, websocket: WebSocket, connection_manager: ConnectionManager):
        self.websocket = websocket
        self.manager = connection_manager
        self.tracker = ProgressTracker()
        self.start_time = None

    async def send_progress(self, stage: QueryStage, **kwargs):
        """Send progress update for a stage"""
        progress_data = self.tracker.move_to_stage(stage)

        # Add any additional kwargs to the progress data
        progress_data.update(kwargs)

        await self.manager.send_progress_update(
            self.websocket,
            progress_data['stage'],
            progress_data['progress'],
            progress_data['message'],
            **{k: v for k, v in progress_data.items()
               if k not in ['stage', 'progress', 'message']}
        )

    async def send_detail(self, detail: str, **kwargs):
        """Send a detail update without changing stage"""
        progress_data = self.tracker.add_detail(detail)
        progress_data.update(kwargs)

        await self.manager.send_progress_update(
            self.websocket,
            progress_data['stage'],
            progress_data['progress'],
            progress_data['message'],
            **{k: v for k, v in progress_data.items()
               if k not in ['stage', 'progress', 'message']}
        )

    async def process_with_streaming(self, query: str, database_id: str):
        """
        Main processing function with streaming progress updates.
        Wraps existing BigQuery AI functions with progress notifications.
        """
        self.start_time = time.time()
        self.tracker.start()

        try:
            # Stage 1: Initialization (0-5%)
            await self.send_progress(
                QueryStage.INITIALIZATION,
                detail=f"Database: {database_id}"
            )
            await asyncio.sleep(0.5)  # Small delay for visual effect

            # Stage 2: Understanding Query (5-10%)
            await self.send_progress(
                QueryStage.UNDERSTANDING,
                detail=f'Query: "{query[:50]}{"..." if len(query) > 50 else ""}'
            )
            await asyncio.sleep(1.0)

            # Stage 3: Generate Embedding (10-15%)
            await self.send_progress(
                QueryStage.EMBEDDING,
                detail="Using text-embedding-005 (768 dimensions)"
            )

            # Call existing async function
            embedding = await generate_query_embedding(query)

            await self.send_detail(
                f"Embedding generated: {len(embedding)} dimensions"
            )

            # Stage 4: Vector Search (15-25%)
            await self.send_progress(
                QueryStage.SEARCHING,
                detail=f"Searching in database: {database_id}"
            )

            # Call existing async vector search and get feature used
            result = await vector_search_columns(
                embedding,
                database_id
            )

            # Handle both old format (list) and new format (tuple with feature)
            if isinstance(result, tuple):
                columns, vector_feature = result
            else:
                columns = result
                vector_feature = "Manual Similarity"  # Fallback for old format

            # Stage 5: Columns Found (25-35%)
            table_names = list(set(col.get('table_name', '') for col in columns[:5]))

            # Update tracker with actual vector search feature used
            if 'vector_feature' in locals():
                self.tracker.all_features_used.append(vector_feature)

            await self.send_progress(
                QueryStage.COLUMNS_FOUND,
                detail=f"Tables: {', '.join(table_names)}",
                columns_found=len(columns),
                top_columns=[{
                    'table': col.get('table_name'),
                    'column': col.get('column_name'),
                    'score': col.get('similarity_score', 0)
                } for col in columns[:3]],
                features_used=list(set(self.tracker.all_features_used))  # Send actual features
            )

            # Stage 6: SQL Generation Start (35-45%)
            await self.send_progress(
                QueryStage.SQL_GENERATION,
                detail="Creating optimized BigQuery SQL"
            )

            # Start SQL generation in background
            sql_task = asyncio.create_task(
                generate_sql_with_context(
                    query,
                    database_id,
                    columns
                )
            )

            # Stage 7: SQL Building with sub-stages (45-60%)
            await self.send_progress(QueryStage.SQL_BUILDING)

            # Simulate progressive SQL building
            for i, (progress, message) in enumerate(SQL_BUILD_STAGES):
                # Just update details, don't move to stage again
                await self.send_detail(
                    message,
                    progress=progress,
                    sql_step=f"Step {i+1}/6"
                )
                await asyncio.sleep(1.5)  # Visual effect

                # Check if SQL is ready
                if sql_task.done():
                    break

            # Wait for SQL generation to complete
            generated_sql = await sql_task

            # Show SQL preview
            sql_preview = generated_sql[:500] + "..." if len(generated_sql) > 500 else generated_sql

            # Stage 8: SQL Complete (60-65%)
            await self.send_progress(
                QueryStage.SQL_COMPLETE,
                detail=f"Query complexity: {self.analyze_sql_complexity(generated_sql)}",
                sql=generated_sql,
                sql_preview=sql_preview
            )

            # Stage 9: Execute Query (65-75%)
            await self.send_progress(
                QueryStage.EXECUTING,
                detail=f"Executing on BigQuery dataset: {database_id}"
            )

            # Start execution
            exec_task = asyncio.create_task(
                execute_bigquery(generated_sql)
            )

            # Stage 10: Execution Progress (75-85%)
            await self.send_progress(
                QueryStage.EXECUTION_PROGRESS,
                detail="Processing query on BigQuery..."
            )

            # Simulate execution progress
            for i in range(3):
                await asyncio.sleep(3.0)
                progress = 75 + (i + 1) * 3
                await self.send_detail(
                    f"Processing... ({progress}%)",
                    progress=progress
                )

                if exec_task.done():
                    break

            # Wait for execution to complete
            execution_result = await exec_task

            if not execution_result.get('success'):
                raise Exception(execution_result.get('error', 'Query execution failed'))

            results = execution_result.get('results', [])
            execution_time = execution_result.get('execution_time_ms', 0)

            # Stage 11: Results Ready (85-90%)
            await self.send_progress(
                QueryStage.RESULTS_READY,
                detail=f"Retrieved {len(results)} rows in {execution_time}ms",
                row_count=len(results)
            )

            # Stage 12: Generate Insights (90-95%)
            await self.send_progress(
                QueryStage.INSIGHTS,
                detail="Using AI.GENERATE_TABLE for insights"
            )

            # Generate insights if we have results using AI.GENERATE_TABLE
            insights = None
            if results:
                try:
                    insights = await generate_structured_insights_v2(
                        query,
                        results,
                        database_id  # Pass database_id for v2
                    )
                    logger.info("Successfully used AI.GENERATE_TABLE for insights")
                except Exception as v2_error:
                    logger.warning(f"AI.GENERATE_TABLE failed, falling back to v1: {v2_error}")
                    # Fallback to v1
                    insights = await generate_structured_insights_v1(
                        query,
                        results
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

            # Stage 13: Complete (95-100%)
            total_time = time.time() - self.start_time
            complete_data = self.tracker.complete(total_time)

            # Send final complete message with all data
            await self.manager.send_progress_update(
                self.websocket,
                'complete',
                100,
                complete_data['message'],
                total_time=complete_data['total_time'],
                features_used=complete_data['features_used'],
                sql=generated_sql,
                results=results[:100],  # Limit results for WebSocket
                insights=insights,
                execution_time_ms=execution_time,
                row_count=len(results),
                columns_used=len(columns),
                database_id=database_id
            )

            logger.info(f"Query processed successfully in {total_time:.1f}s")

        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)

            # Send error state
            error_data = self.tracker.set_error(str(e), type(e).__name__)
            await self.manager.send_error(
                self.websocket,
                error_data['message'],
                error_data.get('error_type')
            )

    def analyze_sql_complexity(self, sql: str) -> str:
        """Analyze SQL complexity for display"""
        complexity_factors = []

        sql_upper = sql.upper()

        # Check for various SQL features
        join_count = sql_upper.count(' JOIN ')
        if join_count > 0:
            complexity_factors.append(f"{join_count} JOIN{'s' if join_count > 1 else ''}")

        if 'WITH ' in sql_upper:
            cte_count = sql_upper.count('WITH ') + sql_upper.count(', ')
            complexity_factors.append(f"{cte_count} CTE{'s' if cte_count > 1 else ''}")

        if 'GROUP BY' in sql_upper:
            complexity_factors.append("Aggregations")

        if 'WHERE' in sql_upper:
            complexity_factors.append("Filters")

        if 'ORDER BY' in sql_upper:
            complexity_factors.append("Sorting")

        if 'PARTITION BY' in sql_upper:
            complexity_factors.append("Window functions")

        return ', '.join(complexity_factors) if complexity_factors else "Simple query"