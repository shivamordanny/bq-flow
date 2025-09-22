"""
WebSocket Connection Manager
Handles WebSocket connections and message broadcasting
"""

from fastapi import WebSocket
from typing import List, Dict, Any
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections and message distribution"""

    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.connection_metadata: Dict[WebSocket, Dict[str, Any]] = {}

    async def connect(self, websocket: WebSocket, metadata: Dict[str, Any] = None):
        """Accept and track a new WebSocket connection"""
        await websocket.accept()
        self.active_connections.append(websocket)

        # Store connection metadata
        self.connection_metadata[websocket] = metadata or {
            'connected_at': datetime.now().isoformat(),
            'session_id': None
        }

        logger.info(f"New WebSocket connection. Total connections: {len(self.active_connections)}")

        # Send welcome message
        await self.send_personal_message(websocket, {
            'type': 'connection',
            'status': 'connected',
            'message': 'WebSocket connection established',
            'timestamp': datetime.now().isoformat()
        })

    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

        if websocket in self.connection_metadata:
            del self.connection_metadata[websocket]

        logger.info(f"WebSocket disconnected. Remaining connections: {len(self.active_connections)}")

    async def send_personal_message(self, websocket: WebSocket, data: Dict[str, Any]):
        """Send a message to a specific client"""

        try:
            if isinstance(data, dict):
                # Ensure timestamp is included
                if 'timestamp' not in data:
                    data['timestamp'] = datetime.now().isoformat()
                await websocket.send_json(data)
            else:
                await websocket.send_text(str(data))
        except Exception as e:
            logger.error(f"Error sending message to client: {e}")
            # Connection might be closed, remove it
            self.disconnect(websocket)

    async def send_progress_update(
        self,
        websocket: WebSocket,
        stage: str,
        progress: int,
        message: str,
        **kwargs
    ):
        """Send a standardized progress update to a client"""
        update_data = {
            'type': 'progress',
            'stage': stage,
            'progress': progress,
            'message': message,
            'timestamp': datetime.now().isoformat(),
            **kwargs  # Additional data like detail, sql_preview, features_used, etc.
        }

        await self.send_personal_message(websocket, update_data)

    async def send_error(self, websocket: WebSocket, error_message: str, error_type: str = None):
        """Send an error message to a client"""
        error_data = {
            'type': 'error',
            'stage': 'error',
            'progress': 0,
            'message': error_message,
            'error_type': error_type or 'UnknownError',
            'timestamp': datetime.now().isoformat()
        }

        await self.send_personal_message(websocket, error_data)

    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast a message to all connected clients"""
        disconnected = []

        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")
                disconnected.append(connection)

        # Clean up disconnected clients
        for conn in disconnected:
            self.disconnect(conn)

    def get_connection_count(self) -> int:
        """Get the number of active connections"""
        return len(self.active_connections)

    def get_connection_info(self, websocket: WebSocket) -> Dict[str, Any]:
        """Get metadata for a specific connection"""
        return self.connection_metadata.get(websocket, {})