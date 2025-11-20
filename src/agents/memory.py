"""
Session Memory Management for BQ Flow V2
Implements Concept 2: SESSIONS & MEMORY

Provides:
- Session state retention across conversation turns
- Context tracking (database, queries, results)
- Multi-session support
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, field
import json


@dataclass
class Session:
    """
    Represents a single conversation session
    """
    session_id: str
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    state: Dict[str, Any] = field(default_factory=dict)
    messages: List[Dict[str, str]] = field(default_factory=list)

    def add_message(self, role: str, content: str):
        """Add a message to the conversation history"""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })

    def update_state(self, key: str, value: Any):
        """Update a state variable"""
        self.state[key] = value

    def get_state(self, key: str, default: Any = None) -> Any:
        """Get a state variable"""
        return self.state.get(key, default)

    def get_state_summary(self) -> str:
        """Get a human-readable summary of session state"""
        summary_parts = []

        if "current_database" in self.state:
            summary_parts.append(f"Database: {self.state['current_database']}")

        if "discovered_columns" in self.state:
            col_count = len(self.state["discovered_columns"])
            summary_parts.append(f"Discovered {col_count} relevant columns")

        if "last_sql" in self.state:
            summary_parts.append("Previous query executed")

        if "last_results" in self.state:
            row_count = len(self.state["last_results"])
            summary_parts.append(f"Last query returned {row_count} rows")

        return ", ".join(summary_parts) if summary_parts else "New conversation"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize session to dictionary"""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "state": self.state,
            "message_count": len(self.messages),
            "state_summary": self.get_state_summary()
        }


class SessionMemory:
    """
    In-memory session storage
    Simulates ADK's InMemorySessionService for hackathon demo
    """

    def __init__(self):
        self.sessions: Dict[str, Session] = {}

    def get_session(self, session_id: str) -> Session:
        """
        Get or create a session

        Args:
            session_id: Unique session identifier

        Returns:
            Session object
        """
        if session_id not in self.sessions:
            self.sessions[session_id] = Session(session_id=session_id)

        return self.sessions[session_id]

    def clear_session(self, session_id: str):
        """Remove a session from memory"""
        if session_id in self.sessions:
            del self.sessions[session_id]

    def get_all_sessions(self) -> List[Dict[str, Any]]:
        """Get all sessions (for debugging/monitoring)"""
        return [session.to_dict() for session in self.sessions.values()]

    def get_active_sessions(self, max_age_minutes: int = 30) -> List[str]:
        """
        Get sessions active within the last N minutes

        Args:
            max_age_minutes: Maximum age in minutes

        Returns:
            List of active session IDs
        """
        now = datetime.now()
        active = []

        for session_id, session in self.sessions.items():
            age_minutes = (now - session.last_activity).total_seconds() / 60
            if age_minutes <= max_age_minutes:
                active.append(session_id)

        return active

    def cleanup_old_sessions(self, max_age_hours: int = 24):
        """
        Remove sessions older than N hours

        Args:
            max_age_hours: Maximum age in hours before cleanup
        """
        now = datetime.now()
        to_remove = []

        for session_id, session in self.sessions.items():
            age_hours = (now - session.last_activity).total_seconds() / 3600
            if age_hours > max_age_hours:
                to_remove.append(session_id)

        for session_id in to_remove:
            del self.sessions[session_id]

        return len(to_remove)


# Example usage
if __name__ == "__main__":
    # Demo session memory
    memory = SessionMemory()

    # Create session
    session = memory.get_session("demo_session")

    # Simulate conversation
    session.add_message("user", "Show me sales data")
    session.update_state("current_database", "ecommerce")

    session.add_message("assistant", "I'll search for sales-related columns...")
    session.update_state("discovered_columns", [
        {"table_name": "orders", "column_name": "total_sales"},
        {"table_name": "products", "column_name": "revenue"}
    ])

    session.add_message("user", "For electronics category")
    session.update_state("last_sql", "SELECT * FROM orders WHERE category = 'electronics'")

    # Print session state
    print("=== Session State ===")
    print(json.dumps(session.to_dict(), indent=2))

    print("\n=== State Summary ===")
    print(session.get_state_summary())
