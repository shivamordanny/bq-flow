-- =====================================================
-- Create Embedding Model and Vector Index
-- Note: Replace ${PROJECT_ID}, ${DATASET_ID}, and ${GEMINI_CONNECTION} with actual values when executing
-- =====================================================

-- 1. Create Vertex AI connection (if not exists)
-- Note: This needs to be run once by an admin with proper permissions
-- The connection provides access to Vertex AI for ML.GENERATE_EMBEDDING

/*
-- Run this if connection doesn't exist:
-- NOTE: Replace ${PROJECT_ID} and ${GEMINI_CONNECTION} with environment values
CREATE OR REPLACE CONNECTION `${PROJECT_ID}.${GEMINI_CONNECTION}`
OPTIONS(
  type='CLOUD_RESOURCE',
  location='US'
);

-- Grant permissions to the service account
-- Get the service account from:
-- SELECT * FROM ${PROJECT_ID}.INFORMATION_SCHEMA.CONNECTIONS
-- WHERE connection_name = 'vertex_connection';
*/

-- 2. Create remote model for text embeddings
-- NOTE: This is handled programmatically by the application using environment variables
-- The application will use GEMINI_CONNECTION from .env
CREATE OR REPLACE MODEL `${PROJECT_ID}.${DATASET_ID}.embedding_model_005`
REMOTE WITH CONNECTION `${PROJECT_ID}.${GEMINI_CONNECTION}`
OPTIONS (
  endpoint = 'text-embedding-005'
);

-- Alternative: Create model for gemini-embedding-001 (premium, 3072 dimensions)
-- CREATE OR REPLACE MODEL `${PROJECT_ID}.${DATASET_ID}.embedding_model_gemini`
-- REMOTE WITH CONNECTION `${PROJECT_ID}.${GEMINI_CONNECTION}`
-- OPTIONS (
--   endpoint = 'gemini-embedding-001'
-- );

-- 3. Create vector index on enriched_metadata table
-- Note: Table needs at least 5000 rows for index creation
-- This will be created programmatically from the Streamlit app once enough data is available

-- The following SQL is used by the create_vector_index() function:
/*
CREATE OR REPLACE VECTOR INDEX enriched_embedding_index
ON `${PROJECT_ID}.${DATASET_ID}.enriched_metadata`(embedding)
STORING(
    database_id,      -- Critical for database isolation and pre-filtering
    table_name,
    column_name,
    data_type,
    semantic_context,
    example_values,
    distinct_count,
    null_percentage
)
OPTIONS(
  index_type = 'IVF',
  distance_type = 'COSINE',
  ivf_options = '{"num_lists": 50}'  -- Optimized for smaller multi-database datasets
);

-- Benefits of this approach:
-- 1. database_id in STORING enables pre-filtering optimization
-- 2. IVF index works well for datasets under 10M rows
-- 3. Cumulative embeddings across databases can reach 5000 threshold
-- 4. Each database query remains isolated via WHERE clause
*/

-- 4. Helper view for easy querying with database isolation
CREATE OR REPLACE VIEW `${PROJECT_ID}.${DATASET_ID}.v_enriched_metadata` AS
SELECT
    database_id,
    table_name,
    column_name,
    data_type,
    semantic_context,
    example_values,
    distinct_count,
    null_percentage,
    embedding_model,
    enriched_at,
    ARRAY_LENGTH(embedding) as embedding_dims
FROM `${PROJECT_ID}.${DATASET_ID}.enriched_metadata`
WHERE embedding IS NOT NULL;

-- 5. Create function for vector search with database isolation
CREATE OR REPLACE FUNCTION `${PROJECT_ID}.${DATASET_ID}.search_columns`(
    query_text STRING,
    target_database_id STRING,
    top_k INT64
)
AS (
    (
        SELECT
            base.table_name,
            base.column_name,
            base.data_type,
            base.semantic_context,
            base.example_values,
            distance
        FROM VECTOR_SEARCH(
            TABLE `${PROJECT_ID}.${DATASET_ID}.enriched_metadata`,
            'embedding',
            (
                SELECT ML.GENERATE_EMBEDDING(
                    MODEL `${PROJECT_ID}.${DATASET_ID}.embedding_model_005`,
                    (SELECT query_text AS content),
                    STRUCT(768 AS output_dimensionality)
                ).embedding
            ),
            top_k => top_k
        )
        WHERE base.database_id = target_database_id  -- Client isolation
        ORDER BY distance
    )
);