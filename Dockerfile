# Multi-stage Dockerfile for BQ Flow
# Stage 1: Builder stage
FROM python:3.12-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VERSION=2.1.4 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 - --version $POETRY_VERSION
ENV PATH="$POETRY_HOME/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml poetry.lock ./

# Install Python dependencies
RUN poetry install --no-dev --no-root

# Stage 2: Runtime stage
FROM python:3.12-slim as runtime

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VERSION=2.1.4 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \
    PATH="/app/.venv/bin:$PATH"

# Install runtime dependencies and supervisor
RUN apt-get update && apt-get install -y \
    supervisor \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry in runtime (needed for running the app)
RUN curl -sSL https://install.python-poetry.org | python3 - --version $POETRY_VERSION
ENV PATH="$POETRY_HOME/bin:$PATH"

# Create app user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY --chown=appuser:appuser . /app

# Copy dependency files
COPY --chown=appuser:appuser pyproject.toml poetry.lock ./

# Create necessary directories
RUN mkdir -p /app/logs /var/log/supervisor && \
    chown -R appuser:appuser /app/logs /var/log/supervisor

# Copy supervisor configuration
COPY --chown=appuser:appuser supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Create a startup script for the container
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
# Check if Google credentials are provided\n\
if [ -n "$GOOGLE_APPLICATION_CREDENTIALS_JSON" ]; then\n\
    echo "Setting up Google Cloud credentials..."\n\
    echo "$GOOGLE_APPLICATION_CREDENTIALS_JSON" > /app/credentials.json\n\
    export GOOGLE_APPLICATION_CREDENTIALS=/app/credentials.json\n\
fi\n\
\n\
# Load environment variables from .env if it exists\n\
if [ -f /app/.env ]; then\n\
    export $(cat /app/.env | grep -v "^#" | xargs)\n\
fi\n\
\n\
# Start supervisord\n\
exec /usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf\n\
' > /app/docker-entrypoint.sh && \
    chmod +x /app/docker-entrypoint.sh && \
    chown appuser:appuser /app/docker-entrypoint.sh

# Expose ports
EXPOSE 8000 3000

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/docs || exit 1

# Set the entrypoint
ENTRYPOINT ["/app/docker-entrypoint.sh"]