"""
Configuration Management for BQ Flow System
Centralizes all configuration to eliminate hardcoding
"""

import os
import yaml
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import json
from google.cloud import bigquery

# Get logger (will be imported from logging_config)
from .logging import get_logger

logger = get_logger(__name__)


@dataclass
class ModelConfig:
    """Configuration for AI models"""
    embedding_model: str
    generation_model: str
    insight_model: str
    embedding_dimensions: int
    temperature: float
    max_output_tokens: int


@dataclass
class DatasetConfig:
    """Configuration for datasets"""
    project_id: str
    dataset_name: str
    metadata_project: str
    metadata_dataset: str


@dataclass
class DatabaseConfig:
    """Configuration for a specific database"""
    database_id: str
    project_id: str
    dataset_name: str
    table_pattern: Optional[str] = None
    description: Optional[str] = None


class BigQueryConfig:
    """
    Central configuration management for BQ Flow
    Eliminates all hardcoding and provides dynamic configuration
    """

    def __init__(self, config_file: Optional[str] = None):
        """Initialize configuration from file or environment"""
        self.config_file = config_file or os.getenv('CONFIG_FILE', 'config/config.yaml')
        self.client = None
        self._config = {}
        self._databases = {}
        self._models = {}

        # Validate required environment variables early
        self._validate_required_env_vars()

        # Load configuration
        self.load_config()

        # Initialize BigQuery client for dynamic queries
        self.init_client()

        logger.info(f"Configuration loaded from {self.config_file}")

    def _validate_required_env_vars(self):
        """Validate that required environment variables are set"""
        required_vars = ['PROJECT_ID', 'DATASET_ID']
        missing = []

        for var in required_vars:
            if not os.getenv(var):
                missing.append(var)

        if missing:
            error_msg = f"Missing required environment variables: {', '.join(missing)}. Please set them in your .env file."
            logger.error(error_msg)
            raise ValueError(error_msg)

    def load_config(self):
        """Load configuration from YAML file or use defaults"""
        config_path = Path(self.config_file)

        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    self._config = yaml.safe_load(f)
                logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                logger.warning(f"Could not load config file: {e}, using defaults")
                self._config = self.get_default_config()
        else:
            logger.info("No config file found, using environment and defaults")
            self._config = self.get_default_config()

        # Override with environment variables if present
        self.apply_env_overrides()

    def get_default_config(self) -> Dict:
        """Get default configuration with environment variable overrides"""
        # Validate required environment variables
        project_id = os.getenv('PROJECT_ID')
        dataset_id = os.getenv('DATASET_ID')

        if not project_id:
            raise ValueError("PROJECT_ID environment variable is required. Please set it in your .env file.")
        if not dataset_id:
            raise ValueError("DATASET_ID environment variable is required. Please set it in your .env file.")

        return {
            'bigquery': {
                'project_id': project_id,
                'dataset_id': dataset_id,
                'location': os.getenv('LOCATION', 'US'),
                'connections': {
                    'gemini_connection': {
                        'name': os.getenv('GEMINI_CONNECTION', 'us.gemini_connection'),
                        'location': 'us',
                        'type': 'CLOUD_RESOURCE'
                    }
                }
            },
            'models': {
                'embedding': {
                    'model_name': os.getenv('EMBEDDING_MODEL', 'text-embedding-005'),
                    'dimensions': int(os.getenv('EMBEDDING_DIMS', '768')),
                    'task_type': 'RETRIEVAL_QUERY',
                    'batch_size': 250,
                    'cost_per_1k_tokens': 0.01
                },
                'generation': {
                    'model_name': os.getenv('GENERATION_MODEL', 'gemini-2.5-flash'),
                    'temperature': float(os.getenv('GENERATION_TEMP', '0.1')),
                    'max_tokens': int(os.getenv('MAX_TOKENS', '1000')),
                    'connection_id': os.getenv('GEMINI_CONNECTION', 'us.gemini_connection')
                },
                'insight': {
                    'model_name': os.getenv('INSIGHT_MODEL', 'gemini-2.5-flash'),
                    'temperature': float(os.getenv('INSIGHT_TEMP', '0.3')),
                    'max_tokens': 500,
                    'connection_id': os.getenv('GEMINI_CONNECTION', 'us.gemini_connection')
                }
            },
            'datasets': {
                'metadata_project': os.getenv('METADATA_PROJECT', project_id),
                'metadata_dataset': os.getenv('METADATA_DATASET', dataset_id),
                'execution_project': os.getenv('EXECUTION_PROJECT', project_id),
                'tables': {
                    'enriched_metadata':  os.getenv('METADATA_TABLE', 'enriched_metadata'),
                    'database_registry': os.getenv('DATABASE_TABLE', 'database_registry'),
                    'embedding_jobs': os.getenv('EMBEDDING_JOBS_TABLE', 'embedding_jobs'),
                    'vector_indexes_metadata': os.getenv('VECTOR_INDEXES_TABLE', 'vector_indexes_metadata'),
                    'query_embeddings': os.getenv('QUERY_EMBEDDINGS', 'query_embeddings')
                }
            },
            'vector_search': {
                'index': {
                    'type': 'IVF',
                    'distance_type': 'COSINE',
                    'min_rows_required': 5000,
                    'default_num_lists': 50,
                    'auto_create': False,
                    'cumulative_mode': True
                },
                'search': {
                    'top_k': int(os.getenv('VECTOR_SEARCH_TOP_K', '15')),
                    'fraction_lists_to_search': 0.1
                }
            },
            'fallback': {
                'use_ml_functions': True,
                'max_fallback_levels': 2,
                'similarity_thresholds': {
                    'enriched': 0.4,
                    'original': 0.3,
                    'cache': 0.85
                }
            },
            'security': {
                'use_parameterized_queries': os.getenv('USE_PARAMETERIZED_QUERIES', 'true').lower() == 'true',
                'max_query_length': int(os.getenv('MAX_QUERY_LENGTH', '10000')),
                'allowed_projects': []  # Empty means all allowed
            },
            'performance': {
                'enable_caching': os.getenv('ENABLE_CACHING', 'true').lower() == 'true',
                'cache_ttl_seconds': int(os.getenv('CACHE_TTL_SECONDS', '3600')),
                'max_results': int(os.getenv('MAX_RESULTS', '1000')),
                'vector_search_top_k': int(os.getenv('VECTOR_SEARCH_TOP_K', '15'))
            },
            'forecast': {
                'method': os.getenv('FORECAST_METHOD', 'AI.FORECAST'),
                'model': os.getenv('FORECAST_MODEL', 'TimesFM 2.0'),
                'default_horizon': int(os.getenv('FORECAST_HORIZON', '30')),
                'max_horizon': 10000,
                'confidence_level': float(os.getenv('FORECAST_CONFIDENCE', '0.95')),
                'min_data_points': 10,
                'connection_id': os.getenv('GEMINI_CONNECTION', 'us.gemini_connection')
            },
            'bigquery_features': self.get_default_features()
        }

    def get_default_features(self) -> List[Dict[str, Any]]:
        """Get default BigQuery features configuration"""
        return [
            {
                "name": "ML.GENERATE_EMBEDDING",
                "description": "Convert queries to vectors",
                "model": "text-embedding-005",
                "dimensionality": 768
            },
            {
                "name": "VECTOR_SEARCH with IVF Index",
                "description": "Semantic column discovery",
                "index_type": "IVF",
                "distance_type": "COSINE"
            },
            {
                "name": "AI.GENERATE",
                "description": "Natural language to SQL",
                "model": "gemini-2.5-flash"
            },
            {
                "name": "AI.GENERATE_TABLE",
                "description": "Structured insights extraction",
                "model": "gemini-2.5-flash"
            },
            {
                "name": "AI.FORECAST",
                "description": "Time-series prediction",
                "model": "TimesFM 2.0"
            },
            {
                "name": "AI-Generated Explanations",
                "description": "Transparency through AI-powered insights"
            },
            {
                "name": "Semantic Cache",
                "description": "Performance optimization"
            },
            {
                "name": "Cost Tracking",
                "description": "Resource awareness"
            }
        ]

    def apply_env_overrides(self):
        """Apply environment variable overrides to configuration"""
        # Model overrides
        if 'EMBEDDING_MODEL' in os.environ:
            self._config.setdefault('models', {}).setdefault('embedding', {})['model_name'] = os.environ['EMBEDDING_MODEL']
        if 'GENERATION_MODEL' in os.environ:
            self._config.setdefault('models', {}).setdefault('generation', {})['model_name'] = os.environ['GENERATION_MODEL']

        # Ensure datasets section exists
        self._config.setdefault('datasets', {})

        # Project and dataset configuration from environment (required)
        project_id = os.getenv('PROJECT_ID')
        dataset_id = os.getenv('DATASET_ID')

        if project_id:
            self._config['datasets']['execution_project'] = project_id
            self._config['datasets']['metadata_project'] = os.getenv('METADATA_PROJECT', project_id)
            self._config.setdefault('bigquery', {})['project_id'] = project_id

        if dataset_id:
            self._config['datasets']['metadata_dataset'] = os.getenv('METADATA_DATASET', dataset_id)
            self._config.setdefault('bigquery', {})['dataset_id'] = dataset_id

        # Connection overrides
        if 'GEMINI_CONNECTION' in os.environ:
            self._config.setdefault('bigquery', {}).setdefault('connections', {}).setdefault('gemini_connection', {})['name'] = os.environ['GEMINI_CONNECTION']

    def init_client(self):
        """Initialize BigQuery client"""
        # Ensure datasets config exists
        if 'datasets' not in self._config or 'execution_project' not in self._config['datasets']:
            # Try to get from environment or use PROJECT_ID
            project = os.getenv('EXECUTION_PROJECT') or os.getenv('PROJECT_ID')
            if not project:
                raise ValueError("No execution project specified. Set PROJECT_ID in environment.")
            if 'datasets' not in self._config:
                self._config['datasets'] = {}
            self._config['datasets']['execution_project'] = project
        else:
            project = self._config['datasets']['execution_project']

        self.client = bigquery.Client(project=project)
        logger.info(f"Initialized BigQuery client with project: {project}")

    def get_model_config(self, model_type: str = 'generation') -> ModelConfig:
        """Get model configuration"""
        model_cfg = self._config['models'].get(model_type, self._config['models']['generation'])

        return ModelConfig(
            embedding_model=self._config['models']['embedding']['model_name'],
            generation_model=model_cfg.get('model_name', 'gemini-2.5-flash'),
            insight_model=self._config['models']['insight']['model_name'],
            embedding_dimensions=self._config['models']['embedding']['dimensions'],
            temperature=model_cfg.get('temperature', 0.1),
            max_output_tokens=model_cfg.get('max_tokens', 1000)
        )

    def get_dataset_config(self) -> DatasetConfig:
        """Get dataset configuration"""
        # Ensure datasets config exists with required values
        if 'datasets' not in self._config:
            self._config['datasets'] = {}

        ds_cfg = self._config.get('datasets', {})

        # Get values with fallbacks to environment
        execution_project = ds_cfg.get('execution_project') or os.getenv('EXECUTION_PROJECT') or os.getenv('PROJECT_ID')
        metadata_project = ds_cfg.get('metadata_project') or os.getenv('METADATA_PROJECT') or os.getenv('PROJECT_ID')
        metadata_dataset = ds_cfg.get('metadata_dataset') or os.getenv('METADATA_DATASET') or os.getenv('DATASET_ID')

        if not execution_project:
            raise ValueError("No execution project specified. Set PROJECT_ID in environment.")
        if not metadata_dataset:
            raise ValueError("No metadata dataset specified. Set DATASET_ID in environment.")

        return DatasetConfig(
            project_id=execution_project,
            dataset_name=metadata_dataset,
            metadata_project=metadata_project,
            metadata_dataset=metadata_dataset
        )

    def get_database_config(self, database_id: str) -> Optional[DatabaseConfig]:
        """
        Get configuration for a specific database
        Queries BigQuery to get actual configuration, not hardcoded
        """
        if database_id in self._databases:
            return self._databases[database_id]

        # Query for database configuration
        query = f"""
        SELECT
            database_id,
            project_id,
            dataset_name,
            description
        FROM `{self._config['datasets']['metadata_project']}.{self._config['datasets']['metadata_dataset']}.database_registry`
        WHERE database_id = @database_id
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("database_id", "STRING", database_id)
            ]
        )

        try:
            results = list(self.client.query(query, job_config=job_config).result())
            if results:
                row = results[0]
                config = DatabaseConfig(
                    database_id=row.database_id,
                    project_id=row.project_id,
                    dataset_name=row.dataset_name,
                    description=row.description
                )
                self._databases[database_id] = config
                return config
        except Exception as e:
            logger.error(f"Could not fetch database config for {database_id}: {e}")

        return None

    def get_ga_tables(self, database_id: str) -> List[str]:
        """
        Dynamically discover Google Analytics tables
        No hardcoding of ga_sessions_20170801
        """
        db_config = self.get_database_config(database_id)
        if not db_config:
            return []

        # Query INFORMATION_SCHEMA for actual tables
        query = f"""
        SELECT table_name
        FROM `{db_config.project_id}.{db_config.dataset_name}.INFORMATION_SCHEMA.TABLES`
        WHERE table_name LIKE 'ga_sessions_%'
        ORDER BY table_name DESC
        LIMIT 10
        """

        try:
            results = self.client.query(query).to_dataframe()
            tables = results['table_name'].tolist()
            logger.info(f"Found {len(tables)} GA tables for {database_id}: {tables[:3]}...")
            return tables
        except Exception as e:
            logger.warning(f"Could not fetch GA tables: {e}")
            return []

    def get_table_pattern(self, database_id: str) -> str:
        """Get table pattern for a database"""
        patterns = {
            'ga_sample': 'ga_sessions_%',
            'thelook_ecommerce': '%',  # All tables
            'stackoverflow': '%'  # All tables
        }
        return patterns.get(database_id, '%')

    def get_similarity_threshold(self, search_type: str = 'original') -> float:
        """Get similarity threshold for different search types"""
        return self._config['fallback']['similarity_thresholds'].get(search_type, 0.3)

    def get_model_endpoint(self, model_type: str = 'generation') -> str:
        """Get the model endpoint for BigQuery AI"""
        model_cfg = self._config['models'].get(model_type, {})
        return model_cfg.get('model_name', 'gemini-2.5-flash')

    def get_embedding_config(self) -> Dict[str, Any]:
        """Get embedding configuration for ML.GENERATE_EMBEDDING"""
        return {
            'model': self._config['models']['embedding']['model_name'],
            'dimensions': self._config['models']['embedding']['dimensions'],
            'task_type': self._config['models']['embedding'].get('task_type', 'RETRIEVAL_QUERY')
        }

    def should_use_ml_fallback(self) -> bool:
        """Check if ML functions should be used for fallback"""
        return self._config['fallback'].get('use_ml_functions', True)

    def get_connection_id(self, connection_type: str = 'gemini') -> str:
        """Get connection ID for BigQuery AI connections"""
        connections = self._config.get('bigquery', {}).get('connections', {})
        if connection_type == 'gemini':
            return connections.get('gemini_connection', {}).get('name', 'us.gemini_connection')
        return 'us.gemini_connection'  # default fallback

    def get_vector_search_config(self) -> Dict[str, Any]:
        """Get vector search configuration"""
        vs_config = self._config.get('vector_search', {})
        search_config = vs_config.get('search', {})
        return {
            'top_k': search_config.get('top_k', 15),
            'distance_type': vs_config.get('index', {}).get('distance_type', 'COSINE'),
            'fraction_lists_to_search': search_config.get('fraction_lists_to_search', 0.1)
        }

    def is_caching_enabled(self) -> bool:
        """Check if caching is enabled"""
        return self._config['performance'].get('enable_caching', True)

    def get_cache_ttl(self) -> int:
        """Get cache TTL in seconds"""
        return self._config['performance'].get('cache_ttl_seconds', 3600)

    def get_max_results(self) -> int:
        """Get maximum number of results to return"""
        return self._config['performance'].get('max_results', 1000)

    def use_parameterized_queries(self) -> bool:
        """Check if parameterized queries should be used"""
        return self._config['security'].get('use_parameterized_queries', True)

    def validate_project_access(self, project_id: str) -> bool:
        """Validate if project access is allowed"""
        allowed = self._config['security'].get('allowed_projects', [])
        if not allowed:  # Empty means all allowed
            return True
        return project_id in allowed

    def get_metadata_table_path(self, table_type: str = 'enriched') -> str:
        """Get the full path to metadata tables"""
        project = self._config['datasets']['metadata_project']
        dataset = self._config['datasets']['metadata_dataset']

        tables = {
            'enriched': f"{project}.{dataset}.enriched_metadata",
            'embeddings': f"{project}.{dataset}.query_embeddings",
            'databases': f"{project}.{dataset}.database_registry"
        }

        return tables.get(table_type, f"{project}.{dataset}.{table_type}")

    def get_model_path(self, model_type: str = 'embedding') -> str:
        """Get the full path to BigQuery ML models"""
        project = self._config['datasets']['metadata_project']
        dataset = self._config['datasets']['metadata_dataset']

        models = {
            'embedding': f"{project}.{dataset}.embedding_model_005",
            'insight': f"{project}.{dataset}.insight_model",
            'explanation': f"{project}.{dataset}.explanation_model"
        }

        return models.get(model_type, f"{project}.{dataset}.{model_type}_model")

    def get_project_id(self) -> str:
        """Get the BigQuery project ID"""
        return self._config.get('bigquery', {}).get('project_id') or os.getenv('PROJECT_ID')

    def get_metadata_dataset(self) -> str:
        """Get the metadata dataset ID"""
        return self._config.get('datasets', {}).get('metadata_dataset') or os.getenv('DATASET_ID')

    def get_bigquery_features(self) -> List[Dict[str, Any]]:
        """Get BigQuery features configuration for API responses"""
        features = self._config.get('bigquery_features', self.get_default_features())

        # Update features with current model configurations
        for feature in features:
            if feature['name'] == 'ML.GENERATE_EMBEDDING':
                feature['model'] = self._config.get('models', {}).get('embedding', {}).get('model_name', 'text-embedding-005')
                feature['dimensionality'] = self._config.get('models', {}).get('embedding', {}).get('dimensions', 768)
            elif feature['name'] == 'AI.GENERATE':
                feature['model'] = self._config.get('models', {}).get('generation', {}).get('model_name', 'gemini-2.5-flash')
            elif feature['name'] == 'AI.GENERATE_TABLE':
                feature['model'] = self._config.get('models', {}).get('insight', {}).get('model_name', 'gemini-2.5-flash')
            elif feature['name'] == 'AI.FORECAST':
                feature['model'] = self._config.get('forecast', {}).get('model', 'TimesFM 2.0')

        return features

    def to_dict(self) -> Dict:
        """Export configuration as dictionary"""
        return self._config

    def save_config(self, file_path: Optional[str] = None):
        """Save configuration to YAML file"""
        save_path = file_path or self.config_file

        with open(save_path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Configuration saved to {save_path}")


# Global configuration instance
_config_instance = None


def get_config() -> BigQueryConfig:
    """Get or create global configuration instance"""
    global _config_instance
    if _config_instance is None:
        _config_instance = BigQueryConfig()
    return _config_instance


def reload_config(config_file: Optional[str] = None):
    """Reload configuration from file"""
    global _config_instance
    _config_instance = BigQueryConfig(config_file)
    return _config_instance