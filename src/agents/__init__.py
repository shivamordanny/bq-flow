"""
BQ Flow V2 - Conversational Agent Module
Built for Google AI Agents Intensive Capstone

Demonstrates 3 core concepts:
1. TOOLS - Custom BigQuery AI functions
2. MEMORY - Session state management
3. EVALUATION - Metrics and observability
"""

from src.agents.bq_flow_agent import BQFlowAgent
from src.agents.tools import create_bq_flow_tools, execute_tool
from src.agents.memory import SessionMemory, Session
from src.agents.evaluation import AgentMetrics

__all__ = [
    "BQFlowAgent",
    "create_bq_flow_tools",
    "execute_tool",
    "SessionMemory",
    "Session",
    "AgentMetrics"
]

__version__ = "2.0.0"
__author__ = "Shivam Bhardwaj"
__description__ = "Conversational BigQuery Analytics with ADK"
