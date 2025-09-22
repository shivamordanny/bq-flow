"""
WebSocket Streaming Server
Adds WebSocket endpoint to existing FastAPI app for real-time progress updates
"""

# Standard imports - no path manipulation needed with proper structure

from fastapi import WebSocket, WebSocketDisconnect
import asyncio
import json
import time
from typing import Dict, Any, Optional

# Import existing FastAPI app and functions
from src.api.rest_api import app
from src.core.bigquery_ai import (
    generate_query_embedding,
    vector_search_columns,
    generate_sql_with_context,
    execute_bigquery,
    generate_structured_insights
)

# Import our new components
from .connection_manager import ConnectionManager
from .progress_tracker import ProgressTracker, QueryStage
from .streaming_processor import StreamingQueryProcessor

# Initialize connection manager
manager = ConnectionManager()


@app.websocket("/ws/query/stream")
async def websocket_endpoint(websocket: WebSocket):
    """
    New WebSocket endpoint for streaming query progress.
    This endpoint is ADDED to the existing FastAPI app without modifying existing routes.
    """
    await manager.connect(websocket)

    try:
        while True:
            # Receive query request
            data = await websocket.receive_json()

            if data.get('type') == 'ping':
                # Handle ping/pong for connection keep-alive
                await manager.send_personal_message(websocket, {'type': 'pong'})
                continue

            # Extract query parameters
            query = data.get('query')
            database_id = data.get('database_id', 'thelook_ecommerce')
            session_id = data.get('session_id')

            if not query:
                await manager.send_error(websocket, "No query provided", "ValidationError")
                continue

            # Store session ID in connection metadata
            conn_info = manager.get_connection_info(websocket)
            conn_info['session_id'] = session_id

            # Create processor with WebSocket for progress updates
            processor = StreamingQueryProcessor(websocket, manager)

            # Process query with streaming updates
            try:
                await processor.process_with_streaming(query, database_id)
            except Exception as e:
                await manager.send_error(
                    websocket,
                    f"Error processing query: {str(e)}",
                    type(e).__name__
                )

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print(f"Client disconnected. Active connections: {manager.get_connection_count()}")

    except Exception as e:
        print(f"WebSocket error: {e}")
        await manager.send_error(websocket, str(e), type(e).__name__)
        manager.disconnect(websocket)


# Add a health check endpoint for WebSocket
@app.get("/ws/health")
async def websocket_health():
    """Check WebSocket server health"""
    return {
        "status": "healthy",
        "websocket_endpoint": "/ws/query/stream",
        "active_connections": manager.get_connection_count(),
        "features": [
            "Real-time progress updates",
            "SQL preview during generation",
            "BigQuery AI feature tracking",
            "Graceful error handling",
            "Automatic reconnection support"
        ]
    }


print("✅ WebSocket endpoint added to FastAPI app at /ws/query/stream")