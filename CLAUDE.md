# CLAUDE.md - BQ Flow Codebase Guide for AI Assistants

**Version:** 5.4
**Last Updated:** 2025-11-14
**Project:** BQ Flow - Natural Language to Predictive Insights powered by BigQuery AI

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture & Design](#architecture--design)
3. [Directory Structure](#directory-structure)
4. [Development Setup](#development-setup)
5. [Coding Conventions & Patterns](#coding-conventions--patterns)
6. [Key Technologies](#key-technologies)
7. [Common Development Workflows](#common-development-workflows)
8. [BigQuery AI Integration](#bigquery-ai-integration)
9. [Testing Strategy](#testing-strategy)
10. [Deployment](#deployment)
11. [Troubleshooting](#troubleshooting)
12. [Important Files Reference](#important-files-reference)

---

## Project Overview

### What is BQ Flow?

BQ Flow transforms BigQuery from a data warehouse into an AI-powered analytics engine that:
- Understands natural language queries
- Discovers semantic relationships in data
- Generates predictive insights
- All without moving data or managing external ML infrastructure

### Core Value Proposition

- **Natural Language to SQL**: Convert questions to optimized BigQuery queries using Gemini 2.5 Flash
- **Semantic Search**: Find relevant columns by meaning, not keywords
- **Automatic Forecasting**: Detect and predict time-series patterns using TimesFM 2.0
- **Real-time Progress**: 13-stage WebSocket streaming updates for transparency
- **Cost Tracking**: Monitor BigQuery AI function usage and costs

### Hackathon Context

This project was created for the **BigQuery AI Hackathon 2025** and demonstrates advanced integration of:
- ML.GENERATE_EMBEDDING (text-embedding-005)
- VECTOR_SEARCH with IVF indexing
- AI.GENERATE (Gemini 2.5 Flash)
- AI.GENERATE_TABLE (structured insights)
- AI.FORECAST (TimesFM 2.0)
- ML.DISTANCE (cosine similarity fallback)

---

## Architecture & Design

### System Architecture

```
┌─────────────────────────────────────────┐
│   User Interface Layer (Port 3000)      │
│   - Chainlit: Chat interface            │
│   - WebSocket: Real-time progress       │
└──────────────┬──────────────────────────┘
               │
┌──────────────┴──────────────────────────┐
│   Backend API Layer (Port 8000)         │
│   - FastAPI: REST endpoints             │
│   - WebSocket: /ws/query/stream         │
└──────────────┬──────────────────────────┘
               │
┌──────────────┴──────────────────────────┐
│   Business Logic Layer                  │
│   - BigQuery AI orchestration           │
│   - Progress tracking (13 stages)       │
│   - Result processing                   │
└──────────────┬──────────────────────────┘
               │
┌──────────────┴──────────────────────────┐
│   BigQuery AI Functions                 │
│   - ML.GENERATE_EMBEDDING               │
│   - VECTOR_SEARCH                       │
│   - AI.GENERATE                         │
│   - AI.GENERATE_TABLE                   │
│   - AI.FORECAST                         │
└──────────────┬──────────────────────────┘
               │
┌──────────────┴──────────────────────────┐
│   Data Layer                            │
│   - enriched_metadata (embeddings)      │
│   - database_registry                   │
│   - query_embeddings (cache)            │
└─────────────────────────────────────────┘
```

### Design Patterns

**1. Centralized Configuration**
- All configuration managed via `src/core/config.py` (BigQueryConfig class)
- NO hardcoded values - everything loaded from `config/config.yaml` or environment variables
- Dynamic database discovery from `database_registry` table

**2. Modular Architecture**
- Clear separation: Core logic (`src/core/`) vs API (`src/api/`) vs UI (`src/ui/`)
- Each module has single responsibility
- Websocket streaming wraps existing core functions without modifying them

**3. Fallback Chain Pattern**
- Primary: VECTOR_SEARCH with IVF index
- Secondary: ML.DISTANCE for cosine similarity
- Tertiary: Semantic cache (>0.85 similarity)

**4. Progress Tracking Wrapper**
- WebSocket streaming decorates core functions with progress updates
- 13-stage pipeline from query to insights
- Non-blocking, async-friendly design

**5. Multi-tenancy Support**
- `database_id` field isolates datasets
- Single metadata tables support multiple databases
- Dynamic configuration per database

---

## Directory Structure

```
/home/user/bq-flow/
│
├── src/                          # Main application source code
│   ├── api/                      # FastAPI REST endpoints
│   │   ├── rest_api.py          # Main API app, routes, CORS
│   │   └── __init__.py
│   │
│   ├── core/                     # Core business logic
│   │   ├── bigquery_ai.py       # Primary BigQuery AI functions
│   │   ├── bigquery_ai_forecast.py      # Time-series forecasting
│   │   ├── bigquery_ai_generate_table.py # Structured insights (v2)
│   │   ├── config.py            # Configuration management (CRITICAL)
│   │   ├── logging.py           # Structured logging system
│   │   ├── retry_utils.py       # Retry logic with exponential backoff
│   │   ├── utils.py             # Helper utilities (JSON, dataframes)
│   │   └── __init__.py
│   │
│   ├── websocket/                # WebSocket streaming layer
│   │   ├── streaming_server.py  # WebSocket endpoint registration
│   │   ├── streaming_processor.py # Query processing with progress
│   │   ├── progress_tracker.py  # 13-stage progress management
│   │   ├── connection_manager.py # WebSocket connection pooling
│   │   └── __init__.py
│   │
│   └── ui/                       # User interfaces
│       ├── chainlit_app.py      # Main chat interface (Chainlit)
│       ├── chainlit_logging.py  # UI-specific logging
│       ├── chainlit.md          # Chainlit config
│       └── __init__.py
│
├── bq_flow_onboarding/           # Data preparation & training system
│   ├── app.py                    # Streamlit onboarding UI (port 8501)
│   ├── core/                     # Onboarding logic
│   │   ├── bigquery_client.py   # Multi-tenant BigQuery operations
│   │   ├── profiler.py          # Data profiling & analysis
│   │   ├── embeddings.py        # Vector generation (ML.GENERATE_EMBEDDING)
│   │   ├── ai_assistant.py      # AI column selection (Gemini)
│   │   └── logger.py            # Onboarding-specific logging
│   ├── sql/                      # Database schema
│   │   ├── 01_create_tables.sql # Metadata tables creation
│   │   └── 02_create_model_and_index.sql # Vector index setup
│   ├── profiles/                 # Dataset configurations
│   │   └── [16+ JSON profiles for public datasets]
│   └── README-ONBORD.md         # Onboarding documentation
│
├── config/                       # Configuration files
│   ├── config.yaml              # Master configuration (CRITICAL)
│   └── databases.yaml           # Database registry (16+ datasets)
│
├── docs/                         # Documentation
│   ├── BENCHMARKING_STRATEGY.md # Spider 2.0 evaluation plan
│   └── BIGQUERY_AI_HACKATHON_SUBMISSION.md
│
├── arch-diagram/                 # Architecture diagrams
├── screenshots/                  # UI screenshots
├── video-walkthrough/            # Demo video
│
├── main.py                       # Application entry point (uvicorn)
├── startup.sh                    # Multi-process startup script
├── Dockerfile                    # Docker multi-stage build
├── pyproject.toml               # Poetry dependencies
├── .env.example                 # Environment template
└── README.md                    # Project documentation
```

### Key Directories Explained

**`src/core/`** - The brain of the application
- All BigQuery AI function calls originate here
- Configuration, logging, retry logic
- NO hardcoding - all values from config

**`src/api/`** - REST API layer
- FastAPI application
- CORS middleware for local development
- Routes: `/api/query`, `/api/forecast`, `/api/databases`, `/docs`

**`src/websocket/`** - Real-time streaming
- Wraps core functions with progress tracking
- 13-stage pipeline updates via WebSocket
- Endpoint: `ws://localhost:8000/ws/query/stream`

**`src/ui/`** - Chainlit chat interface
- Port 3000
- WebSocket streaming OR REST fallback
- Database selection, query history, visualizations

**`bq_flow_onboarding/`** - Separate Streamlit app
- Port 8501
- 4-step process: Discover → Profile → AI Selection → Generate Embeddings
- Prepares data for natural language queries

---

## Development Setup

### Prerequisites

```bash
# Required
- Python 3.12+
- Poetry (version 2.1.4+)
- Google Cloud Project with BigQuery API enabled
- Service account with BigQuery Admin role

# Optional but recommended
- Docker & Docker Compose
- gcloud CLI configured
```

### Initial Setup

```bash
# 1. Clone repository
git clone https://github.com/shivamordanny/bq-flow.git
cd bq-flow

# 2. Install Poetry (if not installed)
curl -sSL https://install.python-poetry.org | python3 -

# 3. Create virtual environment and install dependencies
poetry install

# 4. Configure environment
cp .env.example .env
# Edit .env with your Google Cloud credentials
```

### Required Environment Variables

**MUST SET:**
```bash
PROJECT_ID=your-gcp-project-id
DATASET_ID=your-bigquery-dataset-id
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
```

**OR use Application Default Credentials:**
```bash
gcloud auth application-default login
```

### Running the Application

**Option 1: All-in-one startup script (recommended)**
```bash
./startup.sh
# Starts backend (port 8000) + frontend (port 3000)
# Monitors processes and auto-restarts on failure
```

**Option 2: Run components separately**
```bash
# Terminal 1: Backend API
poetry run python main.py

# Terminal 2: Frontend UI
cd src/ui && poetry run chainlit run chainlit_app.py --port 3000

# Terminal 3: Data Onboarding (optional)
cd bq_flow_onboarding && poetry run streamlit run app.py --port 8501
```

**Option 3: Docker**
```bash
docker build -t bq-flow .
docker run -p 8000:8000 -p 3000:3000 \
  -e PROJECT_ID=your-project \
  -e DATASET_ID=your-dataset \
  -v /path/to/credentials.json:/app/credentials.json \
  -e GOOGLE_APPLICATION_CREDENTIALS=/app/credentials.json \
  bq-flow
```

### Access Points

- **Main Application**: http://localhost:3000
- **API Docs**: http://localhost:8000/docs
- **Data Onboarding**: http://localhost:8501
- **WebSocket**: ws://localhost:8000/ws/query/stream

---

## Coding Conventions & Patterns

### Python Style Guide

**1. Follow PEP 8**
- 4 spaces for indentation (NOT tabs)
- Max line length: 100 characters (flexible for readability)
- Snake_case for functions and variables
- PascalCase for classes

**2. Type Hints**
```python
from typing import Dict, List, Optional, Any

def generate_query_embedding(
    query: str,
    client: bigquery.Client,
    config: BigQueryConfig
) -> Optional[List[float]]:
    """Always include docstrings with type information"""
    pass
```

**3. Configuration Access Pattern**
```python
# CORRECT: Use config instance
from src.core.config import get_config
config = get_config()
model_name = config.get_model_endpoint('generation')

# WRONG: Hardcoded values
model_name = 'gemini-2.5-flash'  # DON'T DO THIS
```

**4. Error Handling**
```python
# Always use try-except with specific exceptions
try:
    result = client.query(sql).result()
except exceptions.NotFound as e:
    logger.error(f"Table not found: {e}")
    raise
except exceptions.GoogleAPIError as e:
    logger.error(f"BigQuery API error: {e}")
    # Retry logic or fallback
```

**5. Logging Pattern**
```python
from src.core.logging import get_logger
logger = get_logger(__name__)  # Use module name

# Structured logging with context
logger.info(f"Executing query for database: {database_id}", extra={
    'database_id': database_id,
    'query_length': len(query),
    'user_id': user_id
})
```

**6. BigQuery Client Usage**
```python
# CORRECT: Use parameterized queries (security)
job_config = bigquery.QueryJobConfig(
    query_parameters=[
        bigquery.ScalarQueryParameter("database_id", "STRING", database_id)
    ]
)
query = "SELECT * FROM table WHERE database_id = @database_id"
results = client.query(query, job_config=job_config).result()

# WRONG: String interpolation (SQL injection risk)
query = f"SELECT * FROM table WHERE database_id = '{database_id}'"
```

**7. Async Patterns**
```python
# FastAPI async endpoints
@app.post("/api/query")
async def execute_query(request: QueryRequest):
    # Use async where beneficial for I/O
    result = await some_async_operation()
    return result

# Chainlit async handlers
@cl.on_message
async def on_message(message: cl.Message):
    # Async message processing
    pass
```

### File Organization

**Import Order:**
```python
# 1. Standard library
import os
import sys
from pathlib import Path
from typing import Dict, List

# 2. Third-party packages
import pandas as pd
from google.cloud import bigquery
from fastapi import FastAPI

# 3. Local imports
from src.core.config import get_config
from src.core.logging import get_logger
```

**Module Structure:**
```python
"""
Module docstring explaining purpose
"""

# Imports
# Constants
# Classes
# Functions
# Main execution (if __name__ == "__main__")
```

---

## Key Technologies

### Core Stack

| Technology | Version | Purpose |
|------------|---------|---------|
| **Python** | 3.12+ | Primary language |
| **Poetry** | 2.1.4+ | Dependency management |
| **FastAPI** | 0.116.1+ | REST API framework |
| **Uvicorn** | 0.35.0+ | ASGI server |
| **Chainlit** | 2.8.0+ | Chat UI framework |
| **Streamlit** | 1.49.1+ | Data onboarding UI |
| **WebSockets** | 15.0.1+ | Real-time communication |
| **google-cloud-bigquery** | 3.37.0+ | BigQuery client |
| **Pandas** | 2.3.2+ | Data manipulation |
| **PyYAML** | 6.0.2+ | Configuration parsing |

### BigQuery AI Features

**ML.GENERATE_EMBEDDING**
- Model: `text-embedding-005`
- Dimensions: 768
- Cost: ~$0.01 per 1K tokens
- Purpose: Query and column vectorization

**VECTOR_SEARCH**
- Index Type: IVF (Inverted File Index)
- Distance: COSINE
- Minimum rows: 5000 (BigQuery requirement)
- Purpose: Semantic column discovery

**AI.GENERATE**
- Model: `gemini-2.5-flash`
- Temperature: 0.1 (SQL generation), 0.3 (insights)
- Max tokens: 1000
- Purpose: Natural language to SQL, text insights

**AI.GENERATE_TABLE**
- Model: `gemini-2.5-flash`
- Purpose: Structured JSON output with schema
- Use case: Business insights with defined structure

**AI.FORECAST**
- Model: `TimesFM 2.0`
- Default horizon: 30 days
- Confidence level: 0.95
- Purpose: Time-series predictions

---

## Common Development Workflows

### Adding a New BigQuery AI Function

1. **Add to `src/core/bigquery_ai.py`** (or appropriate module)
```python
def new_ai_function(
    client: bigquery.Client,
    config: BigQueryConfig,
    input_data: str
) -> Dict[str, Any]:
    """
    Description of what this does

    Args:
        client: BigQuery client instance
        config: Configuration instance
        input_data: Input description

    Returns:
        Dict with results
    """
    logger = get_logger(__name__)
    logger.info(f"Executing new AI function with input: {input_data}")

    # Use config for all parameters
    model = config.get_model_endpoint('generation')
    connection = config.get_connection_id('gemini')

    # Build parameterized query
    query = f"""
    SELECT ML.NEW_FUNCTION(
        @input,
        MODEL `{model}`,
        CONNECTION `{connection}`
    ) AS result
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("input", "STRING", input_data)
        ]
    )

    try:
        results = client.query(query, job_config=job_config).result()
        return process_results(results)
    except Exception as e:
        logger.error(f"Error in new_ai_function: {e}")
        raise
```

2. **Add REST endpoint in `src/api/rest_api.py`**
```python
@app.post("/api/new-function")
async def new_function_endpoint(request: NewFunctionRequest):
    """API endpoint for new function"""
    config = get_config()
    result = new_ai_function(config.client, config, request.input_data)
    return {"status": "success", "result": result}
```

3. **Add WebSocket support** (if needed) in `src/websocket/streaming_processor.py`

4. **Update configuration** in `config/config.yaml`

5. **Document in README.md**

### Adding a New Database

1. **Update `config/databases.yaml`**
```yaml
my_new_database:
  database_id: my_new_database
  project_id: bigquery-public-data
  dataset_name: my_dataset
  description: "Description of the database"
  profiling_strategy: comprehensive
  enabled: true
```

2. **Register in BigQuery** (run via onboarding app or SQL)
```sql
INSERT INTO `project.dataset.database_registry`
(database_id, project_id, dataset_name, description)
VALUES
('my_new_database', 'bigquery-public-data', 'my_dataset', 'Description');
```

3. **Create profile** in `bq_flow_onboarding/profiles/`
```json
{
  "database_id": "my_new_database",
  "project_id": "bigquery-public-data",
  "dataset_name": "my_dataset",
  "profiling_config": {
    "sample_size": 1000,
    "max_columns": 100
  }
}
```

4. **Run onboarding workflow** to generate embeddings

### Modifying Configuration

**Never hardcode values!** Always use the config system:

```python
# In config/config.yaml
models:
  generation:
    temperature: 0.2  # Change here

# OR in .env
GENERATION_TEMP=0.2

# Code automatically picks it up via:
config = get_config()
temp = config.get_model_config('generation').temperature
```

### Adding New Progress Stages

Modify `src/websocket/progress_tracker.py`:

```python
class QueryStage(Enum):
    # ... existing stages ...
    NEW_STAGE = "new_stage"

# Update stage_weights
stage_weights = {
    # ... existing weights ...
    QueryStage.NEW_STAGE: 5  # Percentage weight
}
```

---

## BigQuery AI Integration

### Core Integration Patterns

**1. Embedding Generation**
```python
# Location: src/core/bigquery_ai.py:generate_query_embedding()

sql = f"""
SELECT ML.GENERATE_EMBEDDING(
    @query,
    MODEL `{config.get_model_path('embedding')}`
) AS embedding
"""
```

**2. Vector Search**
```python
# Location: src/core/bigquery_ai.py:vector_search_columns()

search_sql = f"""
SELECT base.*
FROM VECTOR_SEARCH(
    TABLE `{metadata_table}`,
    'enriched_embedding',
    (SELECT embedding FROM query_emb),
    top_k => @top_k,
    distance_type => 'COSINE'
) AS base
"""
```

**3. SQL Generation**
```python
# Location: src/core/bigquery_ai.py:generate_sql_with_context()

prompt = f"""
Generate a SQL query for BigQuery that answers: {user_query}

Available tables and columns:
{table_context}

Requirements:
- Use fully qualified table names
- Follow BigQuery SQL syntax
- Optimize for performance
"""

sql = f"""
SELECT AI.GENERATE(
    @prompt,
    MODEL `{model}`,
    CONNECTION `{connection}`
) AS generated_sql
"""
```

**4. Structured Insights (AI.GENERATE_TABLE)**
```python
# Location: src/core/bigquery_ai_generate_table.py

schema = """
{
  "type": "ARRAY",
  "items": {
    "type": "OBJECT",
    "properties": {
      "insight": {"type": "STRING"},
      "confidence": {"type": "NUMBER"}
    }
  }
}
"""

sql = f"""
SELECT * FROM AI.GENERATE_TABLE(
    @prompt,
    MODEL `{model}`,
    CONNECTION `{connection}`,
    STRUCT('{escaped_schema}' AS schema)
)
"""
```

**5. Time-Series Forecasting**
```python
# Location: src/core/bigquery_ai_forecast.py

forecast_sql = f"""
SELECT * FROM AI.FORECAST(
    MODEL `{model_name}`,
    STRUCT(@horizon AS horizon, @confidence AS confidence_level)
)
"""
```

### Best Practices for BigQuery AI

1. **Always use parameterized queries**
```python
job_config = bigquery.QueryJobConfig(
    query_parameters=[
        bigquery.ScalarQueryParameter("param", "STRING", value)
    ]
)
```

2. **Handle BigQuery-specific errors**
```python
from google.cloud import exceptions

try:
    result = client.query(sql).result()
except exceptions.NotFound:
    # Table/model doesn't exist
except exceptions.Forbidden:
    # Permission denied
except exceptions.BadRequest:
    # Invalid SQL syntax
```

3. **Implement retry logic for transient failures**
```python
from src.core.retry_utils import retry_with_backoff

@retry_with_backoff(max_retries=3)
def execute_query(client, sql):
    return client.query(sql).result()
```

4. **Track costs**
```python
# Check bytes processed
job = client.query(sql)
bytes_processed = job.total_bytes_processed
estimated_cost = (bytes_processed / 1e12) * 5  # $5 per TB
logger.info(f"Query cost estimate: ${estimated_cost:.4f}")
```

---

## Testing Strategy

### Current State

The repository is **benchmarking-focused** rather than traditional unit testing:
- Primary evaluation: **Spider 2.0-lite** (1000+ BigQuery queries)
- Metrics: SQL accuracy, semantic correctness, execution success
- See `docs/BENCHMARKING_STRATEGY.md` for details

### Testing Approach

**1. Manual Testing via UI**
- Use Chainlit interface (port 3000) for end-to-end testing
- Check each stage of 13-stage progress pipeline
- Verify results, insights, and forecasts

**2. API Testing**
- FastAPI auto-generates interactive docs at `/docs`
- Test endpoints directly via Swagger UI
- Validate request/response schemas

**3. BigQuery AI Function Testing**
```python
# Test in BigQuery console first
SELECT ML.GENERATE_EMBEDDING(
    'test query',
    MODEL `project.dataset.embedding_model_005`
) AS embedding;

# Then integrate into Python
```

**4. Logging-Based Debugging**
```python
# All operations are logged with context
logger.info("Operation completed", extra={
    'operation': 'vector_search',
    'results_count': len(results),
    'execution_time': elapsed_time
})
```

### Recommended Testing Additions

When adding tests, use **pytest**:

```python
# tests/test_bigquery_ai.py
import pytest
from src.core.bigquery_ai import generate_query_embedding
from src.core.config import get_config

@pytest.fixture
def config():
    return get_config()

def test_generate_embedding(config):
    """Test embedding generation"""
    embedding = generate_query_embedding(
        "test query",
        config.client,
        config
    )
    assert embedding is not None
    assert len(embedding) == 768  # text-embedding-005 dimensions
```

---

## Deployment

### Local Deployment (Development)

**Using startup.sh (recommended):**
```bash
./startup.sh
# Handles:
# - Environment validation
# - Credential checking
# - Port availability
# - Process monitoring
# - Auto-restart on failure
```

**Manual process management:**
```bash
# Backend
poetry run python main.py

# Frontend
cd src/ui && poetry run chainlit run chainlit_app.py --port 3000
```

### Docker Deployment

**Build:**
```bash
docker build -t bq-flow:5.4 .
```

**Run:**
```bash
docker run -d \
  --name bq-flow \
  -p 8000:8000 \
  -p 3000:3000 \
  -e PROJECT_ID=your-project \
  -e DATASET_ID=your-dataset \
  -v $(pwd)/credentials.json:/app/credentials.json:ro \
  -e GOOGLE_APPLICATION_CREDENTIALS=/app/credentials.json \
  bq-flow:5.4
```

**Docker Compose (create `docker-compose.yml`):**
```yaml
version: '3.8'
services:
  bq-flow:
    build: .
    ports:
      - "8000:8000"
      - "3000:3000"
    environment:
      - PROJECT_ID=${PROJECT_ID}
      - DATASET_ID=${DATASET_ID}
      - GOOGLE_APPLICATION_CREDENTIALS=/app/credentials.json
    volumes:
      - ./credentials.json:/app/credentials.json:ro
      - ./logs:/app/logs
```

### Cloud Deployment Considerations

**Google Cloud Run:**
- Port: 8000 (API only, or combined with frontend)
- Environment variables: Set via Cloud Run console
- Service account: Attach with BigQuery Admin role
- Memory: Recommend 2GB+
- CPU: 2 vCPUs

**Google Compute Engine:**
- Use startup.sh for automatic service management
- Supervisor for process monitoring
- nginx as reverse proxy (optional)

**Kubernetes:**
- Separate deployments for backend and frontend
- ConfigMaps for environment variables
- Secrets for credentials

---

## Troubleshooting

### Common Issues

**1. "Missing required environment variables: PROJECT_ID, DATASET_ID"**
```bash
# Solution: Create/update .env file
cp .env.example .env
# Edit .env and set PROJECT_ID and DATASET_ID
```

**2. "Could not authenticate with Google Cloud"**
```bash
# Option 1: Use service account
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json

# Option 2: Use gcloud auth
gcloud auth application-default login
```

**3. "Table not found: enriched_metadata"**
```bash
# Solution: Run data onboarding first
cd bq_flow_onboarding
poetry run streamlit run app.py --port 8501
# Complete 4-step onboarding process
```

**4. "VECTOR_SEARCH error: Not enough rows"**
```bash
# BigQuery requires 5000+ rows for IVF index
# Solution: Add more columns to enriched_metadata
# OR: Use ML.DISTANCE fallback (automatic)
```

**5. "WebSocket connection failed"**
```bash
# Check backend is running
curl http://localhost:8000/docs

# Check CORS settings in config/config.yaml
# Ensure ui.backend.ws_url matches actual backend URL
```

**6. "ModuleNotFoundError"**
```bash
# Reinstall dependencies
poetry install

# Check Python version
python --version  # Should be 3.12+
```

**7. "BigQuery quota exceeded"**
```bash
# Check quotas in GCP console
# Reduce VECTOR_SEARCH_TOP_K in .env
# Enable caching: ENABLE_CACHING=true
```

### Debugging Tips

**1. Enable detailed logging**
```bash
# In .env
LOG_LEVEL=DEBUG

# Check logs
tail -f logs/app.log
tail -f logs/backend.log
tail -f logs/frontend.log
```

**2. Test BigQuery connection**
```python
from google.cloud import bigquery
client = bigquery.Client(project='your-project')
print([d.dataset_id for d in client.list_datasets()])
```

**3. Validate configuration**
```python
from src.core.config import get_config
config = get_config()
print(config.to_dict())  # Shows all loaded config
```

**4. Test individual components**
```python
# Test embedding generation
from src.core.bigquery_ai import generate_query_embedding
from src.core.config import get_config

config = get_config()
embedding = generate_query_embedding("test", config.client, config)
print(f"Embedding dimensions: {len(embedding)}")
```

**5. Check FastAPI logs**
```bash
# Backend runs with uvicorn
# Logs show all requests, errors, and tracebacks
# Look for:
# - 404 errors (routing issues)
# - 500 errors (server errors)
# - WebSocket connection logs
```

---

## Important Files Reference

### Configuration Files

| File | Purpose | When to Modify |
|------|---------|----------------|
| `config/config.yaml` | Master configuration | Changing models, thresholds, features |
| `.env` | Environment variables | Per-environment settings, credentials |
| `config/databases.yaml` | Database registry | Adding new databases |
| `pyproject.toml` | Python dependencies | Adding/updating packages |

### Core Application Files

| File | Lines | Purpose | Modify Frequency |
|------|-------|---------|------------------|
| `main.py` | 46 | Application entry point | Rarely |
| `src/core/config.py` | 559 | Configuration management | Medium |
| `src/core/bigquery_ai.py` | 558 | Core BigQuery AI functions | High |
| `src/api/rest_api.py` | Large | REST API endpoints | Medium |
| `src/ui/chainlit_app.py` | 1490 | Chat interface | Medium |
| `src/websocket/progress_tracker.py` | 259 | Progress tracking | Low |

### Data Onboarding Files

| File | Purpose |
|------|---------|
| `bq_flow_onboarding/app.py` | Streamlit onboarding UI |
| `bq_flow_onboarding/core/embeddings.py` | Embedding generation |
| `bq_flow_onboarding/sql/01_create_tables.sql` | Metadata schema |

### Documentation Files

| File | Purpose |
|------|---------|
| `README.md` | Project overview, setup guide |
| `CLAUDE.md` | This file - AI assistant guide |
| `docs/BENCHMARKING_STRATEGY.md` | Evaluation methodology |
| `bq_flow_onboarding/README-ONBORD.md` | Onboarding guide |

---

## Quick Reference Commands

### Development

```bash
# Install dependencies
poetry install

# Run backend
poetry run python main.py

# Run frontend
cd src/ui && poetry run chainlit run chainlit_app.py --port 3000

# Run onboarding
cd bq_flow_onboarding && poetry run streamlit run app.py --port 8501

# All-in-one
./startup.sh
```

### Configuration

```bash
# View current config
poetry run python -c "from src.core.config import get_config; print(get_config().to_dict())"

# Validate environment
cat .env | grep -v '^#' | grep -v '^$'

# Check BigQuery connection
gcloud auth application-default login
bq ls --project_id=your-project
```

### Database Operations

```bash
# List datasets
bq ls --project_id=your-project

# Query enriched_metadata
bq query --use_legacy_sql=false \
  'SELECT COUNT(*) FROM `project.dataset.enriched_metadata`'

# Check vector index
bq query --use_legacy_sql=false \
  'SELECT * FROM `project.dataset.vector_indexes_metadata`'
```

---

## Version History

- **5.4** (2025-11-14): Current version, added CLAUDE.md
- **5.0** (2025-09-22): Initial hackathon submission
- Features: 13-stage progress, semantic cache, AI.FORECAST

---

## Additional Resources

- **Main README**: `/home/user/bq-flow/README.md`
- **Onboarding Guide**: `/home/user/bq-flow/bq_flow_onboarding/README-ONBORD.md`
- **Benchmarking Strategy**: `/home/user/bq-flow/docs/BENCHMARKING_STRATEGY.md`
- **API Docs**: http://localhost:8000/docs (when running)
- **Video Walkthrough**: `/home/user/bq-flow/video-walkthrough/`
- **Architecture Diagrams**: `/home/user/bq-flow/arch-diagram/`

---

## Contact & Support

**Author**: Shivam Bhardwaj
**LinkedIn**: https://www.linkedin.com/in/shivamordanny/
**Hackathon**: BigQuery AI Hackathon 2025
**Repository**: https://github.com/shivamordanny/bq-flow

---

**Last Updated**: 2025-11-14
**Document Version**: 1.0
