"""
Agent Evaluation & Metrics for BQ Flow V2
Implements Concept 3: OBSERVABILITY - Logging, Tracing, Metrics

Tracks:
- Tool call success rates
- SQL generation quality
- Conversation metrics
- Performance timing
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, field
from collections import defaultdict
import json


@dataclass
class ToolMetrics:
    """Metrics for a single tool"""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_duration_seconds: float = 0.0
    errors: List[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage"""
        if self.total_calls == 0:
            return 0.0
        return (self.successful_calls / self.total_calls) * 100

    @property
    def average_duration(self) -> float:
        """Calculate average duration in seconds"""
        if self.successful_calls == 0:
            return 0.0
        return self.total_duration_seconds / self.successful_calls

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary"""
        return {
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "success_rate": round(self.success_rate, 2),
            "average_duration_seconds": round(self.average_duration, 3),
            "error_count": len(self.errors)
        }


class AgentMetrics:
    """
    Centralized metrics tracking for BQ Flow agent
    Production-grade observability for enterprise evaluation
    """

    def __init__(self):
        # Tool-level metrics
        self.tool_metrics: Dict[str, ToolMetrics] = defaultdict(ToolMetrics)

        # Conversation metrics
        self.total_conversations = 0
        self.total_turns = 0
        self.conversation_durations: List[float] = []

        # SQL quality metrics
        self.sql_generated = 0
        self.sql_executed_successfully = 0
        self.sql_execution_failures = 0

        # General errors
        self.errors_by_type: Dict[str, int] = defaultdict(int)

        # Timestamps
        self.start_time = datetime.now()

    def log_tool_call(
        self,
        tool_name: str,
        success: bool,
        duration_seconds: float = 0.0,
        error: Optional[str] = None
    ):
        """
        Log a tool invocation

        Args:
            tool_name: Name of the tool called
            success: Whether the call succeeded
            duration_seconds: Time taken to execute
            error: Error message if failed
        """
        metrics = self.tool_metrics[tool_name]
        metrics.total_calls += 1

        if success:
            metrics.successful_calls += 1
            metrics.total_duration_seconds += duration_seconds
        else:
            metrics.failed_calls += 1
            if error:
                metrics.errors.append(f"{datetime.now().isoformat()}: {error}")

    def log_conversation_turn(
        self,
        session_id: str,
        user_message: str,
        agent_response: str,
        duration_seconds: float
    ):
        """
        Log a conversation turn

        Args:
            session_id: Session identifier
            user_message: User's message
            agent_response: Agent's response
            duration_seconds: Time taken for full turn
        """
        self.total_turns += 1
        self.conversation_durations.append(duration_seconds)

    def log_sql_generation(self, success: bool):
        """Log SQL generation event"""
        self.sql_generated += 1
        if success:
            self.sql_executed_successfully += 1
        else:
            self.sql_execution_failures += 1

    def log_error(self, error_type: str, message: str):
        """Log a general error"""
        self.errors_by_type[error_type] += 1

    def get_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive metrics summary

        Returns:
            Dictionary with all metrics
        """
        # Tool metrics
        tool_summary = {}
        for tool_name, metrics in self.tool_metrics.items():
            tool_summary[tool_name] = metrics.to_dict()

        # Conversation metrics
        avg_turn_duration = (
            sum(self.conversation_durations) / len(self.conversation_durations)
            if self.conversation_durations else 0.0
        )

        # SQL quality
        sql_success_rate = (
            (self.sql_executed_successfully / self.sql_generated * 100)
            if self.sql_generated > 0 else 0.0
        )

        # Uptime
        uptime_seconds = (datetime.now() - self.start_time).total_seconds()

        return {
            "tool_metrics": tool_summary,
            "conversation_metrics": {
                "total_turns": self.total_turns,
                "average_turn_duration_seconds": round(avg_turn_duration, 3)
            },
            "sql_quality": {
                "total_generated": self.sql_generated,
                "successful_executions": self.sql_executed_successfully,
                "failed_executions": self.sql_execution_failures,
                "success_rate_percentage": round(sql_success_rate, 2)
            },
            "errors": dict(self.errors_by_type),
            "system": {
                "uptime_seconds": round(uptime_seconds, 2),
                "start_time": self.start_time.isoformat()
            }
        }

    def get_tool_performance(self, tool_name: str) -> Dict[str, Any]:
        """Get metrics for a specific tool"""
        if tool_name in self.tool_metrics:
            return self.tool_metrics[tool_name].to_dict()
        return {"error": f"No metrics for tool: {tool_name}"}

    def export_metrics(self, filepath: str):
        """
        Export metrics to JSON file

        Args:
            filepath: Path to save metrics
        """
        with open(filepath, 'w') as f:
            json.dump(self.get_summary(), f, indent=2)

    def reset_metrics(self):
        """Reset all metrics (useful for testing)"""
        self.tool_metrics.clear()
        self.total_conversations = 0
        self.total_turns = 0
        self.conversation_durations.clear()
        self.sql_generated = 0
        self.sql_executed_successfully = 0
        self.sql_execution_failures = 0
        self.errors_by_type.clear()
        self.start_time = datetime.now()


# Example usage
if __name__ == "__main__":
    # Demo metrics tracking
    metrics = AgentMetrics()

    # Simulate tool calls
    metrics.log_tool_call("search_data", success=True, duration_seconds=1.2)
    metrics.log_tool_call("search_data", success=True, duration_seconds=0.9)
    metrics.log_tool_call("query_data", success=True, duration_seconds=2.5)
    metrics.log_tool_call("query_data", success=False, error="SQL syntax error")

    # Simulate conversations
    metrics.log_conversation_turn(
        session_id="demo",
        user_message="Show me sales",
        agent_response="Here are the sales...",
        duration_seconds=3.8
    )

    # SQL tracking
    metrics.log_sql_generation(success=True)
    metrics.log_sql_generation(success=True)
    metrics.log_sql_generation(success=False)

    # Print summary
    print("=== Agent Metrics Summary ===")
    print(json.dumps(metrics.get_summary(), indent=2))

    # Export to file
    metrics.export_metrics("agent_metrics.json")
    print("\nMetrics exported to agent_metrics.json")
