"""
Progress Tracking System
Manages query processing stages and progress estimation
"""

import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class QueryStage(Enum):
    """Enumeration of query processing stages"""
    INITIALIZATION = "initialization"
    UNDERSTANDING = "understanding"
    EMBEDDING = "embedding"
    SEARCHING = "searching"
    COLUMNS_FOUND = "columns_found"
    SQL_GENERATION = "sql_generation"
    SQL_BUILDING = "sql_building"
    SQL_COMPLETE = "sql_complete"
    EXECUTING = "executing"
    EXECUTION_PROGRESS = "execution_progress"
    RESULTS_READY = "results_ready"
    INSIGHTS = "insights"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class StageInfo:
    """Information about a processing stage"""
    name: str
    progress_start: int
    progress_end: int
    estimated_duration: float  # in seconds
    message: str
    icon: str
    features: List[str] = None


class ProgressTracker:
    """Tracks and estimates progress for query processing"""

    def __init__(self):
        # Define stage configurations with realistic timings for 37-second total
        self.stages = {
            QueryStage.INITIALIZATION: StageInfo(
                "initialization", 0, 5, 0.5,
                "Initializing query processor...", "🔄", []
            ),
            QueryStage.UNDERSTANDING: StageInfo(
                "understanding", 5, 10, 1.5,
                "Understanding your query...", "🔍", []
            ),
            QueryStage.EMBEDDING: StageInfo(
                "embedding", 10, 15, 2.0,
                "Generating semantic embedding...", "🧬",
                ["ML.GENERATE_EMBEDDING"]
            ),
            QueryStage.SEARCHING: StageInfo(
                "searching", 15, 25, 3.0,
                "Searching relevant columns...", "🔎", []
            ),
            QueryStage.COLUMNS_FOUND: StageInfo(
                "columns_found", 25, 35, 0.5,
                "Found relevant columns", "📊",
                []  # Features will be added dynamically based on actual usage
            ),
            QueryStage.SQL_GENERATION: StageInfo(
                "sql_generation", 35, 45, 3.0,
                "Generating SQL with Gemini 2.5 Flash...", "✨",
                ["AI.GENERATE"]
            ),
            QueryStage.SQL_BUILDING: StageInfo(
                "sql_building", 45, 60, 10.0,
                "Building query components...", "🔨", []
            ),
            QueryStage.SQL_COMPLETE: StageInfo(
                "sql_complete", 60, 65, 0.5,
                "SQL generated successfully!", "📝", []
            ),
            QueryStage.EXECUTING: StageInfo(
                "executing", 65, 75, 3.0,
                "Executing query on BigQuery...", "⚡", []
            ),
            QueryStage.EXECUTION_PROGRESS: StageInfo(
                "execution_progress", 75, 85, 10.0,
                "Processing rows...", "⚙️", []
            ),
            QueryStage.RESULTS_READY: StageInfo(
                "results_ready", 85, 90, 1.0,
                "Processing results...", "📊", []
            ),
            QueryStage.INSIGHTS: StageInfo(
                "insights", 90, 95, 2.0,
                "Generating AI insights...", "🎯",
                ["AI.GENERATE_TABLE"]
            ),
            QueryStage.COMPLETE: StageInfo(
                "complete", 95, 100, 0.5,
                "Query complete!", "✅", []
            )
        }

        self.current_stage: Optional[QueryStage] = None
        self.start_time: Optional[float] = None
        self.stage_start_time: Optional[float] = None
        self.completed_stages: List[QueryStage] = []
        self.stage_history: Dict[QueryStage, float] = {}
        self.all_features_used: List[str] = []

    def start(self):
        """Start tracking a new query"""
        self.start_time = time.time()
        self.stage_start_time = self.start_time
        self.current_stage = QueryStage.INITIALIZATION
        self.completed_stages = []
        self.stage_history = {}
        self.all_features_used = []

        logger.info("Progress tracking started")

    def move_to_stage(self, stage: QueryStage) -> Dict[str, Any]:
        """Move to a new processing stage"""
        if self.current_stage:
            # Record duration of previous stage
            duration = time.time() - self.stage_start_time
            self.stage_history[self.current_stage] = duration
            self.completed_stages.append(self.current_stage)

        self.current_stage = stage
        self.stage_start_time = time.time()

        stage_info = self.stages[stage]

        # Track features used
        if stage_info.features:
            self.all_features_used.extend(stage_info.features)

        return self.get_current_progress()

    def get_current_progress(self, include_estimate: bool = True) -> Dict[str, Any]:
        """Get current progress information"""
        if not self.current_stage:
            return {'progress': 0, 'message': 'Not started', 'stage': 'idle'}

        stage_info = self.stages[self.current_stage]
        elapsed = time.time() - self.start_time if self.start_time else 0

        # Calculate smooth progress within current stage
        stage_elapsed = time.time() - self.stage_start_time
        stage_progress_ratio = min(stage_elapsed / stage_info.estimated_duration, 1.0)

        # Interpolate progress within stage bounds
        progress = stage_info.progress_start + (
            (stage_info.progress_end - stage_info.progress_start) * stage_progress_ratio
        )

        result = {
            'stage': self.current_stage.value,
            'progress': min(int(progress), 100),
            'message': f"{stage_info.icon} {stage_info.message}",
            'elapsed': round(elapsed, 1),
            'features_used': list(set(self.all_features_used))  # Unique features
        }

        # Only include estimate if requested to avoid recursion
        if include_estimate:
            result['estimated_remaining'] = self.estimate_remaining_time()

        return result

    def estimate_remaining_time(self) -> Optional[float]:
        """Estimate remaining time based on progress"""
        if not self.current_stage or not self.start_time:
            return None

        # Calculate total estimated time
        total_estimated = sum(s.estimated_duration for s in self.stages.values())

        # Calculate completed time
        elapsed = time.time() - self.start_time

        # Get current progress percentage WITHOUT recursion
        current_progress_data = self.get_current_progress(include_estimate=False)
        current_progress = current_progress_data['progress']

        if current_progress > 0:
            # Estimate total time based on current progress
            estimated_total = (elapsed / current_progress) * 100
            remaining = max(0, estimated_total - elapsed)
            return round(remaining, 1)

        return round(total_estimated - elapsed, 1)

    def add_detail(self, detail: str) -> Dict[str, Any]:
        """Add detail message to current progress"""
        progress = self.get_current_progress()
        progress['detail'] = detail
        return progress

    def add_sql_preview(self, sql_preview: str) -> Dict[str, Any]:
        """Add SQL preview to current progress"""
        progress = self.get_current_progress()
        progress['sql_preview'] = sql_preview
        return progress

    def set_error(self, error_message: str, error_type: str = None) -> Dict[str, Any]:
        """Set error state"""
        self.current_stage = QueryStage.ERROR
        elapsed = time.time() - self.start_time if self.start_time else 0

        return {
            'stage': 'error',
            'progress': 0,
            'message': f"❌ {error_message}",
            'elapsed': round(elapsed, 1),
            'error_type': error_type or 'UnknownError'
        }

    def complete(self, total_time: float = None) -> Dict[str, Any]:
        """Mark query as complete"""
        self.current_stage = QueryStage.COMPLETE

        if not total_time and self.start_time:
            total_time = time.time() - self.start_time

        return {
            'stage': 'complete',
            'progress': 100,
            'message': "✅ Query complete!",
            'total_time': f"{total_time:.1f}s" if total_time else "N/A",
            'features_used': list(set(self.all_features_used))
        }

    def get_stage_timings(self) -> Dict[str, float]:
        """Get timing information for completed stages"""
        return {stage.value: duration for stage, duration in self.stage_history.items()}


# SQL building sub-stages for simulated progressive display
SQL_BUILD_STAGES = [
    (45, "Analyzing query requirements..."),
    (48, "Identifying necessary tables..."),
    (51, "Building JOIN relationships..."),
    (54, "Creating WHERE conditions..."),
    (57, "Adding aggregations and grouping..."),
    (60, "Optimizing query structure...")
]


def get_sql_build_progress(substage_index: int) -> tuple[int, str]:
    """Get progress and message for SQL building substages"""
    if substage_index < len(SQL_BUILD_STAGES):
        return SQL_BUILD_STAGES[substage_index]
    return (60, "Finalizing SQL...")