-- =====================================================
-- BQ Flow Embeddings System - Table Creation
-- Single Multi-Tenant Architecture with database_id isolation
-- Note: Replace ${PROJECT_ID} and ${DATASET_ID} with actual values when executing
-- =====================================================

-- 1. Main enriched metadata table (multi-tenant)
CREATE OR REPLACE TABLE `${PROJECT_ID}.${DATASET_ID}.enriched_metadata` (
    -- Client isolation key
    database_id STRING NOT NULL,

    -- Core identifiers
    table_name STRING NOT NULL,
    column_name STRING NOT NULL,
    full_column_path STRING,  -- For nested columns like geoNetwork.region

    -- Basic metadata
    data_type STRING,
    description STRING,
    is_nullable BOOL,
    is_partitioning_column BOOL,
    is_clustering_column BOOL,

    -- Enriched data samples
    example_values ARRAY<STRING>,
    example_values_json STRING,  -- For complex types

    -- Statistical profile
    distinct_count INT64,
    null_count INT64,
    null_percentage FLOAT64,
    total_count INT64,

    -- Value distribution
    min_value STRING,
    max_value STRING,
    avg_value FLOAT64,
    std_dev FLOAT64,

    -- Semantic content for embedding
    semantic_context STRING,

    -- AI Selection tracking
    is_selected BOOL DEFAULT FALSE,
    selection_reason STRING,
    selection_score FLOAT64,

    -- Embeddings
    embedding ARRAY<FLOAT64>,
    embedding_model STRING,
    embedding_dimensions INT64,

    -- Metadata tracking
    enriched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
    enrichment_version STRING DEFAULT 'v2.0',

    -- Primary key for uniqueness
    PRIMARY KEY (database_id, table_name, column_name) NOT ENFORCED
) CLUSTER BY database_id, table_name;

-- 2. Database registry table
CREATE OR REPLACE TABLE `${PROJECT_ID}.${DATASET_ID}.database_registry` (
    database_id STRING NOT NULL,
    display_name STRING,
    project_id STRING NOT NULL,
    dataset_name STRING NOT NULL,
    description STRING,

    -- Configuration
    sample_size INT64 DEFAULT 1000,
    embedding_model STRING DEFAULT 'text-embedding-005',
    profiling_strategy STRING DEFAULT 'auto',  -- auto, ga_analytics, ecommerce, generic

    -- Statistics
    table_count INT64,
    column_count INT64,
    total_embeddings INT64,

    -- Tracking
    registered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
    last_profiled_at TIMESTAMP,
    last_embedded_at TIMESTAMP,
    is_active BOOL DEFAULT TRUE,

    PRIMARY KEY (database_id) NOT ENFORCED
);

-- 3. Embedding jobs tracking
CREATE OR REPLACE TABLE `${PROJECT_ID}.${DATASET_ID}.embedding_jobs` (
    job_id STRING NOT NULL,
    database_id STRING NOT NULL,

    -- Job details
    job_type STRING,  -- discovery, profiling, embedding, indexing
    status STRING,    -- pending, running, completed, failed

    -- Progress tracking
    total_items INT64,
    processed_items INT64,
    failed_items INT64,
    progress_percentage FLOAT64,

    -- Performance metrics
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    duration_seconds INT64,

    -- Cost tracking
    estimated_cost_usd FLOAT64,
    actual_cost_usd FLOAT64,
    embeddings_generated INT64,

    -- Error handling
    error_message STRING,
    retry_count INT64 DEFAULT 0,

    -- Metadata
    created_by STRING DEFAULT 'streamlit_app',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),

    PRIMARY KEY (job_id) NOT ENFORCED
) CLUSTER BY database_id;

-- 4. Vector index metadata
CREATE OR REPLACE TABLE `${PROJECT_ID}.${DATASET_ID}.vector_indexes_metadata` (
    index_name STRING NOT NULL,
    table_name STRING NOT NULL,
    column_name STRING NOT NULL,

    -- Index configuration
    index_type STRING DEFAULT 'IVF',
    distance_type STRING DEFAULT 'COSINE',
    num_lists INT64,

    -- Stored columns for pre-filtering
    stored_columns ARRAY<STRING>,

    -- Status
    index_status STRING,  -- building, active, failed, rebuilding
    coverage_percentage FLOAT64,

    -- Performance metrics
    avg_query_time_ms FLOAT64,
    recall_score FLOAT64,

    -- Tracking
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
    last_refresh_time TIMESTAMP,
    refresh_frequency STRING DEFAULT 'daily',

    PRIMARY KEY (index_name) NOT ENFORCED
);