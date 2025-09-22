#!/usr/bin/env python3
"""
BQ Flow Application
Main entry point for REST API and WebSocket server
Version: 5.0
"""

import sys
import os
import uvicorn
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import the FastAPI app
from src.api.rest_api import app
# Import websocket to register endpoints (side effect import)
import src.websocket.streaming_server

def main():
    """
    Run the BQ Flow application
    Supports both REST API and WebSocket endpoints
    """
    print("=" * 70)
    print("BQ FLOW - Version 5.0")
    print("=" * 70)
    print("Starting server with REST API and WebSocket support...")
    print(f"REST API: http://localhost:8000/docs")
    print(f"WebSocket: ws://localhost:8000/ws/query/stream")
    print("=" * 70)

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

# Export app for uvicorn
app = app

if __name__ == "__main__":
    main()