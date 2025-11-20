"""
BQ Flow V2 Agent - Conversational BigQuery Analytics
Built with Google ADK for AI Agents Intensive Capstone

Demonstrates:
1. TOOLS - 4 custom tools wrapping BigQuery AI functions
2. MEMORY - Session state for multi-turn conversations
3. EVALUATION - Metrics logging and tracking
"""

import asyncio
import json
import os
from typing import Dict, Any, List, Optional
from datetime import datetime

# Google Gen AI for Gemini
import google.genai as genai
from google.genai.types import GenerateContentConfig, Tool, FunctionCall

# BQ Flow imports
from src.core.config import get_config
from src.core.logging import get_logger

# Agent components
from src.agents.tools import execute_tool
from src.agents.memory import SessionMemory
from src.agents.evaluation import AgentMetrics

logger = get_logger(__name__)


class BQFlowAgent:
    """
    Conversational agent for BigQuery data analysis
    Powered by Google ADK + BigQuery AI
    """

    def __init__(self, model_name: str = "gemini-2.0-flash-exp"):
        """
        Initialize BQ Flow agent

        Args:
            model_name: Gemini model to use for agent reasoning
        """
        self.config = get_config()
        self.model_name = model_name
        self.logger = get_logger(__name__)

        # Initialize Gemini client
        # Use GEMINI_API_KEY or GOOGLE_API_KEY from environment
        api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
        if api_key:
            genai.configure(api_key=api_key)
        self.client = genai.Client()

        # Session memory (Concept 2: MEMORY)
        self.memory = SessionMemory()

        # Metrics tracker (Concept 3: EVALUATION)
        self.metrics = AgentMetrics()

        # Available databases
        self.available_databases = self._get_available_databases()

        # System prompt
        self.system_prompt = self._build_system_prompt()

        self.logger.info(f"BQFlowAgent initialized with model: {model_name}")
        self.logger.info(f"Available databases: {list(self.available_databases.keys())}")

    def _get_available_databases(self) -> Dict[str, str]:
        """Get list of available databases from config"""
        try:
            databases = self.config.get_all_databases()
            return {db["database_id"]: db.get("description", "") for db in databases}
        except Exception as e:
            self.logger.warning(f"Could not load databases: {e}")
            return {
                "ecommerce": "E-commerce sales data",
                "stackoverflow": "Stack Overflow developer survey",
                "bikeshare": "NYC Citi Bike trip data"
            }

    def _build_system_prompt(self) -> str:
        """Build system prompt for the agent"""
        db_list = "\n".join([f"- {db_id}: {desc}" for db_id, desc in self.available_databases.items()])

        return f"""You are a conversational BigQuery data analyst assistant powered by AI.

Your capabilities:
1. **search_data** - Find relevant columns using semantic vector search
2. **query_data** - Generate and execute SQL queries
3. **forecast_data** - Create time-series predictions
4. **generate_insights** - Extract actionable business insights

Available databases:
{db_list}

Conversation guidelines:
- Ask for database selection if not specified
- Use search_data to discover relevant columns before querying
- Be conversational and remember context across turns
- Explain your reasoning and tool choices
- Suggest forecasts when you detect time-series data
- Provide insights to add value beyond raw data

Remember: You're helping democratize data access. Make analytics accessible to non-SQL users."""

    async def chat(
        self,
        user_message: str,
        session_id: str = "default",
        database_id: Optional[str] = None
    ) -> str:
        """
        Process user message and return agent response

        Args:
            user_message: User's natural language query
            session_id: Session identifier for context retention
            database_id: Optional database to query

        Returns:
            Agent's response message
        """
        start_time = datetime.now()

        try:
            # Log conversation turn
            self.logger.info(f"[TURN] Session: {session_id}, Message: {user_message[:100]}")

            # Get or create session
            session = self.memory.get_session(session_id)

            # Update session with database if provided
            if database_id:
                session.update_state("current_database", database_id)

            # Add user message to history
            session.add_message("user", user_message)

            # Build conversation history for model
            conversation_history = self._build_conversation_history(session)

            # Generate response with function calling
            response = await self._generate_with_tools(
                conversation_history,
                session
            )

            # Add assistant response to history
            session.add_message("assistant", response)

            # Update session last activity
            session.last_activity = datetime.now()

            # Log metrics
            duration = (datetime.now() - start_time).total_seconds()
            self.metrics.log_conversation_turn(
                session_id=session_id,
                user_message=user_message,
                agent_response=response,
                duration_seconds=duration
            )

            return response

        except Exception as e:
            self.logger.error(f"[ERROR] Chat failed: {str(e)}", exc_info=True)
            self.metrics.log_error("chat", str(e))
            return f"I encountered an error: {str(e)}. Please try rephrasing your question."

    def _build_conversation_history(self, session) -> List[Dict[str, str]]:
        """Build conversation history for model context"""
        history = [{"role": "system", "content": self.system_prompt}]

        # Add session context
        state_summary = session.get_state_summary()
        if state_summary:
            history.append({
                "role": "system",
                "content": f"Session context: {state_summary}"
            })

        # Add message history (last 10 turns to manage context)
        for msg in session.messages[-20:]:  # 10 user + 10 assistant = 20 total
            history.append({
                "role": msg["role"],
                "content": msg["content"]
            })

        return history

    async def _generate_with_tools(
        self,
        conversation_history: List[Dict[str, str]],
        session
    ) -> str:
        """
        Generate response using Gemini with function calling

        Args:
            conversation_history: Full conversation context
            session: Current session object

        Returns:
            Final response text
        """
        # Define tools for function calling
        tools = self._define_tools()

        # Convert history to Gemini format
        contents = [{"role": msg["role"], "parts": [{"text": msg["content"]}]} for msg in conversation_history]

        # Generate with tools
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=2048,
                    tools=tools
                )
            )

            # Check if function calls were made
            if hasattr(response.candidates[0].content, 'parts'):
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'function_call') and part.function_call:
                        # Execute function call
                        function_response = await self._handle_function_call(
                            part.function_call,
                            session
                        )

                        # Continue conversation with function result
                        contents.append({
                            "role": "model",
                            "parts": [{"function_call": part.function_call}]
                        })
                        contents.append({
                            "role": "function",
                            "parts": [{"function_response": {
                                "name": part.function_call.name,
                                "response": function_response
                            }}]
                        })

                        # Generate final response
                        final_response = self.client.models.generate_content(
                            model=self.model_name,
                            contents=contents,
                            config=GenerateContentConfig(temperature=0.3)
                        )

                        return final_response.text

            # No function calls, return direct response
            return response.text

        except Exception as e:
            self.logger.error(f"Error in generate_with_tools: {str(e)}", exc_info=True)
            raise

    def _define_tools(self) -> List[Tool]:
        """Define tools for Gemini function calling"""
        return [
            Tool(function_declarations=[{
                "name": "search_data",
                "description": "Search for relevant columns in BigQuery database using semantic vector search",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language description of data to search for"
                        },
                        "database_id": {
                            "type": "string",
                            "description": "Database identifier (e.g., 'ecommerce', 'stackoverflow')"
                        }
                    },
                    "required": ["query", "database_id"]
                }
            }]),
            Tool(function_declarations=[{
                "name": "query_data",
                "description": "Generate and execute SQL query on BigQuery",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_query": {"type": "string", "description": "User's question"},
                        "database_id": {"type": "string", "description": "Database identifier"},
                        "relevant_columns": {
                            "type": "array",
                            "description": "Relevant columns from search (optional)",
                            "items": {"type": "object"}
                        }
                    },
                    "required": ["user_query", "database_id"]
                }
            }]),
            Tool(function_declarations=[{
                "name": "forecast_data",
                "description": "Generate time-series forecasts using BigQuery AI",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "results_data": {"type": "string", "description": "JSON of query results"},
                        "user_query": {"type": "string", "description": "Original question"},
                        "database_id": {"type": "string", "description": "Database identifier"},
                        "horizon": {"type": "integer", "description": "Forecast periods", "default": 30}
                    },
                    "required": ["results_data", "user_query", "database_id"]
                }
            }]),
            Tool(function_declarations=[{
                "name": "generate_insights",
                "description": "Generate structured business insights from query results",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_query": {"type": "string", "description": "User's question"},
                        "results_data": {"type": "string", "description": "JSON of results"},
                        "database_id": {"type": "string", "description": "Database identifier"}
                    },
                    "required": ["user_query", "results_data", "database_id"]
                }
            }])
        ]

    async def _handle_function_call(
        self,
        function_call: FunctionCall,
        session
    ) -> Dict[str, Any]:
        """
        Execute function call and update session state

        Args:
            function_call: Function call from model
            session: Current session

        Returns:
            Function execution result
        """
        tool_name = function_call.name
        arguments = dict(function_call.args)

        self.logger.info(f"[TOOL_CALL] {tool_name} with args: {list(arguments.keys())}")

        # Execute tool
        result = await execute_tool(tool_name, arguments)

        # Update session state based on tool results
        if result.get("success"):
            if tool_name == "search_data":
                session.update_state("discovered_columns", result.get("columns", []))
            elif tool_name == "query_data":
                session.update_state("last_results", result.get("results", []))
                session.update_state("last_sql", result.get("sql"))

        # Log metrics
        self.metrics.log_tool_call(
            tool_name=tool_name,
            success=result.get("success", False),
            error=result.get("error")
        )

        return result

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get current metrics summary"""
        return self.metrics.get_summary()

    def clear_session(self, session_id: str):
        """Clear a specific session"""
        self.memory.clear_session(session_id)
        self.logger.info(f"Session {session_id} cleared")


# Standalone execution for testing
async def main():
    """Test the agent"""
    agent = BQFlowAgent()

    print("=== BQ Flow V2 Agent Test ===")
    print("Testing conversational flow...\n")

    # Turn 1
    response1 = await agent.chat(
        "Show me sales data",
        session_id="test_session"
    )
    print(f"Turn 1: {response1}\n")

    # Turn 2
    response2 = await agent.chat(
        "Use the ecommerce database",
        session_id="test_session"
    )
    print(f"Turn 2: {response2}\n")

    # Metrics
    print("=== Metrics ===")
    print(json.dumps(agent.get_metrics_summary(), indent=2))


if __name__ == "__main__":
    asyncio.run(main())
