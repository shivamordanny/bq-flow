"""
Comprehensive Logging Module for BQ Flow Embeddings System
Provides detailed logging for all operations with proper tracking
"""

import logging
import sys
from datetime import datetime
from typing import Optional, Dict, Any
import json
from pathlib import Path

# Create logs directory
LOGS_DIR = Path(__file__).parent.parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)

class EmbeddingLogger:
    """Custom logger for embeddings system with detailed tracking"""

    def __init__(self, name: str, level: str = "INFO"):
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))

        # Remove existing handlers
        self.logger.handlers = []

        # Console handler with detailed format
        console_handler = logging.StreamHandler(sys.stdout)
        console_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)

        # File handler for persistent logs
        file_handler = logging.FileHandler(
            LOGS_DIR / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
        )
        file_format = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        self.logger.addHandler(file_handler)

        # JSON file handler for structured logs
        json_handler = logging.FileHandler(
            LOGS_DIR / f"{name}_{datetime.now().strftime('%Y%m%d')}.json"
        )
        json_handler.setFormatter(JsonFormatter())
        self.logger.addHandler(json_handler)

    def log_operation(self, operation: str, details: Dict[str, Any], level: str = "INFO"):
        """Log an operation with structured details"""
        message = f"[OPERATION: {operation}] "

        # Add details to message
        for key, value in details.items():
            if isinstance(value, (list, dict)):
                message += f"{key}={json.dumps(value, default=str)} "
            else:
                message += f"{key}={value} "

        getattr(self.logger, level.lower())(message)

    def log_profiling(self, database_id: str, table_name: str, columns_count: int,
                     selected_count: Optional[int] = None):
        """Log profiling operation details"""
        details = {
            "database_id": database_id,
            "table_name": table_name,
            "columns_profiled": columns_count
        }
        if selected_count is not None:
            details["columns_selected"] = selected_count
            details["reduction_pct"] = round((1 - selected_count/columns_count) * 100, 1) if columns_count > 0 else 0

        self.log_operation("PROFILING", details)

    def log_insertion(self, database_id: str, records_count: int, operation: str = "INSERT"):
        """Log database insertion details"""
        self.log_operation("DB_INSERTION", {
            "database_id": database_id,
            "operation": operation,
            "records_count": records_count,
            "timestamp": datetime.now().isoformat()
        })

    def log_deletion(self, database_id: str, deleted_count: int):
        """Log deletion operations"""
        self.log_operation("DB_DELETION", {
            "database_id": database_id,
            "deleted_count": deleted_count,
            "timestamp": datetime.now().isoformat()
        })

    def log_embedding_generation(self, database_id: str, total_processed: int,
                                successful: int, failed: int):
        """Log embedding generation results"""
        self.log_operation("EMBEDDING_GENERATION", {
            "database_id": database_id,
            "total_processed": total_processed,
            "successful": successful,
            "failed": failed,
            "success_rate": round(successful/total_processed * 100, 1) if total_processed > 0 else 0
        })

    def log_ai_selection(self, table_name: str, total_columns: int, selected_columns: int,
                        excluded_columns: int):
        """Log AI column selection results"""
        self.log_operation("AI_SELECTION", {
            "table_name": table_name,
            "total_columns": total_columns,
            "selected_columns": selected_columns,
            "excluded_columns": excluded_columns,
            "selection_rate": round(selected_columns/total_columns * 100, 1) if total_columns > 0 else 0
        })

    def log_error(self, operation: str, error: Exception, context: Optional[Dict] = None):
        """Log error with context"""
        details = {
            "operation": operation,
            "error_type": type(error).__name__,
            "error_message": str(error)
        }
        if context:
            details.update(context)

        self.log_operation("ERROR", details, level="ERROR")

    def log_duplicate_detection(self, database_id: str, existing_count: int, new_count: int):
        """Log duplicate detection and handling"""
        self.log_operation("DUPLICATE_DETECTION", {
            "database_id": database_id,
            "existing_records": existing_count,
            "new_records": new_count,
            "action": "DELETE_AND_INSERT"
        }, level="WARNING")

    def get_logger(self):
        """Get the underlying logger instance"""
        return self.logger


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging"""

    def format(self, record):
        log_obj = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage()
        }

        # Add extra fields if present
        if hasattr(record, 'extra'):
            log_obj.update(record.extra)

        return json.dumps(log_obj, default=str)


# Create singleton loggers for different components
profiler_logger = EmbeddingLogger("profiler")
embeddings_logger = EmbeddingLogger("embeddings")
bigquery_logger = EmbeddingLogger("bigquery")
ai_assistant_logger = EmbeddingLogger("ai_assistant")
app_logger = EmbeddingLogger("app")

# Export logger instances
__all__ = [
    'EmbeddingLogger',
    'profiler_logger',
    'embeddings_logger',
    'bigquery_logger',
    'ai_assistant_logger',
    'app_logger'
]