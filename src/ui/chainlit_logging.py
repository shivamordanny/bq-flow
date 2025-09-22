"""
Centralized Logging Configuration for BQ Flow System
Provides structured logging with correlation IDs and performance tracking
"""

import logging
import logging.handlers
import json
import sys
import os
from datetime import datetime
from typing import Dict, Any, Optional
import uuid
from contextvars import ContextVar

# Context variable for request correlation
request_id_var: ContextVar[Optional[str]] = ContextVar('request_id', default=None)
user_session_var: ContextVar[Optional[str]] = ContextVar('user_session', default=None)
database_id_var: ContextVar[Optional[str]] = ContextVar('database_id', default=None)

class StructuredFormatter(logging.Formatter):
    """
    Custom formatter that outputs structured JSON logs
    """

    def format(self, record: logging.LogRecord) -> str:
        # Base log structure
        log_obj = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'service': 'bq-flow-api',
            'environment': os.getenv('ENVIRONMENT', 'development'),
        }

        # Add correlation IDs from context
        request_id = request_id_var.get()
        if request_id:
            log_obj['request_id'] = request_id

        user_session = user_session_var.get()
        if user_session:
            log_obj['user_session'] = user_session

        database_id = database_id_var.get()
        if database_id:
            log_obj['database_id'] = database_id

        # Add location information
        log_obj['location'] = {
            'file': record.pathname,
            'line': record.lineno,
            'function': record.funcName
        }

        # Add any extra fields from the record
        if hasattr(record, 'extra_fields'):
            log_obj['metadata'] = record.extra_fields

        # Add performance metrics if present
        if hasattr(record, 'duration_ms'):
            log_obj['duration_ms'] = record.duration_ms

        if hasattr(record, 'features_used'):
            log_obj['features_used'] = record.features_used

        if hasattr(record, 'cost_estimate'):
            log_obj['cost_estimate'] = record.cost_estimate

        # Add exception info if present
        if record.exc_info:
            log_obj['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': self.formatException(record.exc_info)
            }

        return json.dumps(log_obj, default=str)

class HumanReadableFormatter(logging.Formatter):
    """
    Human-readable formatter for console output
    """

    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'

    def format(self, record: logging.LogRecord) -> str:
        # Add color to level name
        levelname = record.levelname
        if sys.stdout.isatty():  # Only add colors if output is terminal
            levelname = f"{self.COLORS.get(record.levelname, '')}{record.levelname}{self.RESET}"

        # Get correlation IDs
        request_id = request_id_var.get() or 'no-request'

        # Format timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

        # Build the message
        prefix = f"[{timestamp}] [{levelname}] [{record.name}] [req:{request_id[:8]}]"

        # Add the main message
        message = f"{prefix} {record.getMessage()}"

        # Add extra fields if present
        if hasattr(record, 'extra_fields'):
            message += f" | {json.dumps(record.extra_fields, default=str)}"

        # Add performance metrics
        if hasattr(record, 'duration_ms'):
            message += f" | duration={record.duration_ms}ms"

        # Add exception if present
        if record.exc_info:
            message += f"\n{self.formatException(record.exc_info)}"

        return message

class PerformanceLoggerAdapter(logging.LoggerAdapter):
    """
    Logger adapter that adds performance and context information
    """

    def process(self, msg, kwargs):
        # Add context variables to extra
        extra = kwargs.get('extra', {})
        extra['request_id'] = request_id_var.get()
        extra['user_session'] = user_session_var.get()
        extra['database_id'] = database_id_var.get()
        kwargs['extra'] = extra
        return msg, kwargs

def setup_logging(
    log_level: str = 'INFO',
    log_file: Optional[str] = 'logs/bq_flow.log',
    use_json: bool = True,
    console_output: bool = True
) -> None:
    """
    Setup comprehensive logging configuration

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (None to disable file logging)
        use_json: Use JSON format for file logs
        console_output: Enable console output
    """

    # Create logs directory if needed
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Clear existing handlers
    logging.root.handlers = []

    # Set base logging level
    logging.root.setLevel(getattr(logging, log_level.upper()))

    # Console Handler (Human Readable)
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(HumanReadableFormatter())
        logging.root.addHandler(console_handler)

    # File Handler (JSON or Human Readable)
    if log_file:
        # Use rotating file handler to prevent disk space issues
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.INFO)

        if use_json:
            file_handler.setFormatter(StructuredFormatter())
        else:
            file_handler.setFormatter(HumanReadableFormatter())

        logging.root.addHandler(file_handler)

    # Configure specific loggers
    configure_module_loggers()

    logging.info(
        "Logging system initialized",
        extra={
            'extra_fields': {
                'log_level': log_level,
                'log_file': log_file,
                'use_json': use_json,
                'console_output': console_output
            }
        }
    )

def configure_module_loggers():
    """
    Configure logging levels for specific modules
    """
    # Reduce noise from external libraries
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    logging.getLogger('uvicorn.access').setLevel(logging.INFO)
    logging.getLogger('chainlit').setLevel(logging.INFO)
    logging.getLogger('google.cloud.bigquery').setLevel(logging.WARNING)

    # Ensure our modules log at appropriate levels
    logging.getLogger('chainlit_app').setLevel(logging.INFO)

def get_logger(name: str) -> PerformanceLoggerAdapter:
    """
    Get a logger instance with performance tracking

    Args:
        name: Logger name (usually __name__)

    Returns:
        PerformanceLoggerAdapter instance
    """
    base_logger = logging.getLogger(name)
    return PerformanceLoggerAdapter(base_logger, {})

def set_request_context(
    request_id: Optional[str] = None,
    user_session: Optional[str] = None,
    database_id: Optional[str] = None
) -> Dict[str, str]:
    """
    Set context variables for request correlation

    Args:
        request_id: Unique request identifier
        user_session: User session identifier
        database_id: Current database context

    Returns:
        Dictionary with set context values
    """
    if request_id:
        request_id_var.set(request_id)
    else:
        request_id = str(uuid.uuid4())
        request_id_var.set(request_id)

    if user_session:
        user_session_var.set(user_session)

    if database_id:
        database_id_var.set(database_id)

    return {
        'request_id': request_id,
        'user_session': user_session,
        'database_id': database_id
    }

def clear_request_context():
    """
    Clear context variables after request completion
    """
    request_id_var.set(None)
    user_session_var.set(None)
    database_id_var.set(None)

def log_performance(func):
    """
    Decorator to log function performance

    Usage:
        @log_performance
        async def my_function():
            pass
    """
    import functools
    import time
    import asyncio

    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start_time = time.time()

        logger.debug(f"Starting {func.__name__}", extra={
            'extra_fields': {
                'function': func.__name__,
                'args_count': len(args),
                'kwargs_keys': list(kwargs.keys())
            }
        })

        try:
            result = await func(*args, **kwargs)
            duration_ms = int((time.time() - start_time) * 1000)

            logger.info(
                f"Completed {func.__name__} in {duration_ms}ms",
                extra={
                    'duration_ms': duration_ms,
                    'extra_fields': {
                        'function': func.__name__,
                        'status': 'success'
                    }
                }
            )
            return result

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)

            logger.error(
                f"Failed {func.__name__} after {duration_ms}ms: {str(e)}",
                exc_info=True,
                extra={
                    'duration_ms': duration_ms,
                    'extra_fields': {
                        'function': func.__name__,
                        'status': 'error',
                        'error_type': type(e).__name__
                    }
                }
            )
            raise

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start_time = time.time()

        logger.debug(f"Starting {func.__name__}")

        try:
            result = func(*args, **kwargs)
            duration_ms = int((time.time() - start_time) * 1000)

            logger.info(
                f"Completed {func.__name__} in {duration_ms}ms",
                extra={'duration_ms': duration_ms}
            )
            return result

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)

            logger.error(
                f"Failed {func.__name__} after {duration_ms}ms: {str(e)}",
                exc_info=True,
                extra={'duration_ms': duration_ms}
            )
            raise

    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper

# Utility functions for structured logging
def log_api_request(logger, method: str, path: str, body: Any = None):
    """Log API request details"""
    logger.info(
        f"API Request: {method} {path}",
        extra={
            'extra_fields': {
                'http_method': method,
                'path': path,
                'body_size': len(json.dumps(body, default=str)) if body else 0
            }
        }
    )

def log_api_response(logger, status_code: int, duration_ms: int):
    """Log API response details"""
    logger.info(
        f"API Response: {status_code} in {duration_ms}ms",
        extra={
            'extra_fields': {
                'status_code': status_code,
                'duration_ms': duration_ms
            }
        }
    )

def log_bigquery_operation(logger, operation: str, details: Dict[str, Any], duration_ms: int):
    """Log BigQuery operation details"""
    logger.info(
        f"BigQuery {operation} completed in {duration_ms}ms",
        extra={
            'extra_fields': {
                'operation': operation,
                'details': details,
                'duration_ms': duration_ms
            }
        }
    )