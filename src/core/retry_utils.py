"""
Retry utilities for BigQuery AI operations
Provides resilient error recovery for AI function calls
"""

import asyncio
import time
from typing import TypeVar, Callable, Optional, Any, Union
from functools import wraps
import random
from google.api_core import exceptions as google_exceptions

# Import logging
from .logging import get_logger

logger = get_logger(__name__)

T = TypeVar('T')


class RetryConfig:
    """Configuration for retry behavior"""

    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 30.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ):
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter


# Default retry configurations for different operations
RETRY_CONFIGS = {
    'embedding': RetryConfig(max_attempts=3, initial_delay=0.5),
    'vector_search': RetryConfig(max_attempts=2, initial_delay=0.5),
    'ai_generate': RetryConfig(max_attempts=3, initial_delay=1.0),
    'ai_forecast': RetryConfig(max_attempts=2, initial_delay=2.0),
    'bigquery_execute': RetryConfig(max_attempts=3, initial_delay=1.0),
}


def calculate_delay(attempt: int, config: RetryConfig) -> float:
    """Calculate exponential backoff delay with optional jitter"""
    delay = min(
        config.initial_delay * (config.exponential_base ** (attempt - 1)),
        config.max_delay
    )

    if config.jitter:
        # Add random jitter (±25% of delay)
        jitter_range = delay * 0.25
        delay += random.uniform(-jitter_range, jitter_range)

    return max(0.1, delay)  # Ensure minimum 100ms delay


def is_retryable_error(error: Exception) -> bool:
    """Determine if an error is retryable"""

    # Google Cloud specific errors
    if isinstance(error, google_exceptions.GoogleAPIError):
        # Retry on rate limit, timeout, and temporary failures
        if isinstance(error, (
            google_exceptions.TooManyRequests,
            google_exceptions.DeadlineExceeded,
            google_exceptions.ServiceUnavailable,
            google_exceptions.InternalServerError,
        )):
            return True

        # Don't retry on client errors (except rate limits)
        if isinstance(error, (
            google_exceptions.BadRequest,
            google_exceptions.InvalidArgument,
            google_exceptions.NotFound,
            google_exceptions.PermissionDenied,
        )):
            return False

    # Network and connection errors
    error_message = str(error).lower()
    retryable_patterns = [
        'rate limit',
        'quota exceeded',
        'timeout',
        'deadline exceeded',
        'connection reset',
        'connection refused',
        'temporary failure',
        '503',
        '502',
        '504',
        'resource exhausted',
    ]

    if any(pattern in error_message for pattern in retryable_patterns):
        return True

    # Default: don't retry unknown errors
    return False


def retry_async(
    operation_type: str = 'default',
    config: Optional[RetryConfig] = None
):
    """
    Async decorator for retrying BigQuery AI operations

    Usage:
        @retry_async('embedding')
        async def generate_embedding(text):
            ...
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            retry_config = config or RETRY_CONFIGS.get(
                operation_type,
                RetryConfig()  # Default config
            )

            last_error = None

            for attempt in range(1, retry_config.max_attempts + 1):
                try:
                    logger.debug(
                        f"Attempting {func.__name__} (attempt {attempt}/{retry_config.max_attempts})",
                        extra={
                            'extra_fields': {
                                'operation': operation_type,
                                'attempt': attempt,
                                'function': func.__name__
                            }
                        }
                    )

                    # Execute the function
                    result = await func(*args, **kwargs)

                    if attempt > 1:
                        logger.info(
                            f"Retry successful for {func.__name__} on attempt {attempt}",
                            extra={
                                'extra_fields': {
                                    'operation': operation_type,
                                    'attempt': attempt,
                                    'function': func.__name__
                                }
                            }
                        )

                    return result

                except Exception as e:
                    last_error = e

                    if not is_retryable_error(e):
                        logger.error(
                            f"Non-retryable error in {func.__name__}: {str(e)}",
                            extra={
                                'extra_fields': {
                                    'operation': operation_type,
                                    'error_type': type(e).__name__,
                                    'function': func.__name__
                                }
                            }
                        )
                        raise

                    if attempt == retry_config.max_attempts:
                        logger.error(
                            f"Max retries exceeded for {func.__name__} after {attempt} attempts",
                            exc_info=True,
                            extra={
                                'extra_fields': {
                                    'operation': operation_type,
                                    'attempts': attempt,
                                    'function': func.__name__,
                                    'final_error': str(e)
                                }
                            }
                        )
                        raise

                    # Calculate delay for next retry
                    delay = calculate_delay(attempt, retry_config)

                    logger.warning(
                        f"Retryable error in {func.__name__}, waiting {delay:.2f}s before retry",
                        extra={
                            'extra_fields': {
                                'operation': operation_type,
                                'attempt': attempt,
                                'delay_seconds': delay,
                                'error': str(e),
                                'error_type': type(e).__name__,
                                'function': func.__name__
                            }
                        }
                    )

                    # Wait before retry
                    await asyncio.sleep(delay)

            # Should not reach here, but handle it
            if last_error:
                raise last_error
            raise RuntimeError(f"Unexpected retry loop exit in {func.__name__}")

        return wrapper
    return decorator


def retry_sync(
    operation_type: str = 'default',
    config: Optional[RetryConfig] = None
):
    """
    Synchronous decorator for retrying operations

    Usage:
        @retry_sync('bigquery_execute')
        def execute_query(sql):
            ...
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args, **kwargs):
            retry_config = config or RETRY_CONFIGS.get(
                operation_type,
                RetryConfig()  # Default config
            )

            last_error = None

            for attempt in range(1, retry_config.max_attempts + 1):
                try:
                    result = func(*args, **kwargs)

                    if attempt > 1:
                        logger.info(f"Retry successful for {func.__name__} on attempt {attempt}")

                    return result

                except Exception as e:
                    last_error = e

                    if not is_retryable_error(e):
                        logger.error(f"Non-retryable error: {str(e)}")
                        raise

                    if attempt == retry_config.max_attempts:
                        logger.error(f"Max retries exceeded after {attempt} attempts")
                        raise

                    delay = calculate_delay(attempt, retry_config)
                    logger.warning(f"Retrying in {delay:.2f}s... (attempt {attempt}/{retry_config.max_attempts})")
                    time.sleep(delay)

            if last_error:
                raise last_error
            raise RuntimeError(f"Unexpected retry loop exit in {func.__name__}")

        return wrapper
    return decorator


class CircuitBreaker:
    """
    Circuit breaker pattern for preventing cascading failures
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = 'closed'  # closed, open, half-open

    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""

        # Check if circuit should be reset
        if self.state == 'open':
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = 'half-open'
                logger.info(f"Circuit breaker entering half-open state for {func.__name__}")

        if self.state == 'open':
            raise RuntimeError(f"Circuit breaker is open for {func.__name__}")

        try:
            result = func(*args, **kwargs)

            # Reset on success
            if self.state == 'half-open':
                self.state = 'closed'
                self.failure_count = 0
                logger.info(f"Circuit breaker closed for {func.__name__}")

            return result

        except self.expected_exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.failure_threshold:
                self.state = 'open'
                logger.error(f"Circuit breaker opened for {func.__name__} after {self.failure_count} failures")

            raise

    async def async_call(self, func: Callable, *args, **kwargs):
        """Execute async function with circuit breaker protection"""

        # Check if circuit should be reset
        if self.state == 'open':
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = 'half-open'
                logger.info(f"Circuit breaker entering half-open state for {func.__name__}")

        if self.state == 'open':
            raise RuntimeError(f"Circuit breaker is open for {func.__name__}")

        try:
            result = await func(*args, **kwargs)

            # Reset on success
            if self.state == 'half-open':
                self.state = 'closed'
                self.failure_count = 0
                logger.info(f"Circuit breaker closed for {func.__name__}")

            return result

        except self.expected_exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.failure_threshold:
                self.state = 'open'
                logger.error(f"Circuit breaker opened for {func.__name__} after {self.failure_count} failures")

            raise


# Export key components
__all__ = [
    'retry_async',
    'retry_sync',
    'RetryConfig',
    'RETRY_CONFIGS',
    'CircuitBreaker',
    'is_retryable_error'
]