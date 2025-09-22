"""
Data Profiler Module for BQ Flow Embeddings System
Profiles BigQuery tables to extract statistics and patterns for rich embeddings
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import logging
from .logger import profiler_logger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProfiler:
    """
    Unified data profiler that extracts statistics and patterns from BigQuery tables
    """

    def __init__(self, sample_size: int = 1000):
        self.sample_size = sample_size
        self.max_examples = 10
        self.max_unique_for_examples = 1000  # Don't extract examples if too many unique values

    def profile_dataframe(self, df: pd.DataFrame, table_name: str, database_id: str) -> List[Dict]:
        """Profile a DataFrame and return column metadata"""
        if df.empty:
            return []

        profiles = []

        for column_name in df.columns:
            profile = self.profile_column(df[column_name], column_name, table_name, database_id)
            profiles.append(profile)

        return profiles

    def profile_column(self, series: pd.Series, column_name: str, table_name: str, database_id: str) -> Dict:
        """Profile a single column and extract metadata"""
        total_count = len(series)
        null_count = series.isnull().sum()
        null_percentage = (null_count / total_count * 100) if total_count > 0 else 0

        # Get non-null values for analysis
        non_null_series = series.dropna()

        profile = {
            'database_id': database_id,
            'table_name': table_name,
            'column_name': column_name,
            'data_type': str(series.dtype),
            'total_count': int(total_count),
            'null_count': int(null_count),
            'null_percentage': float(null_percentage),
            'distinct_count': int(series.nunique()),
        }

        # Extract examples and statistics based on data type
        if len(non_null_series) > 0:
            if self._is_numeric(series):
                profile.update(self._profile_numeric(non_null_series))
            elif self._is_datetime(series):
                profile.update(self._profile_datetime(non_null_series))
            else:
                profile.update(self._profile_categorical(non_null_series))

        # Build semantic context
        profile['semantic_context'] = self._build_semantic_context(profile)

        return profile

    def _is_numeric(self, series: pd.Series) -> bool:
        """Check if series is numeric"""
        return pd.api.types.is_numeric_dtype(series)

    def _is_datetime(self, series: pd.Series) -> bool:
        """Check if series is datetime"""
        return pd.api.types.is_datetime64_any_dtype(series)

    def _profile_numeric(self, series: pd.Series) -> Dict:
        """Profile numeric column"""
        return {
            'min_value': str(series.min()),
            'max_value': str(series.max()),
            'avg_value': float(series.mean()),
            'std_dev': float(series.std()) if len(series) > 1 else 0,
            'example_values': self._get_numeric_examples(series),
            'data_type': 'NUMERIC'
        }

    def _profile_datetime(self, series: pd.Series) -> Dict:
        """Profile datetime column"""
        try:
            min_date = series.min()
            max_date = series.max()
            return {
                'min_value': str(min_date),
                'max_value': str(max_date),
                'example_values': self._get_datetime_examples(series),
                'data_type': 'DATETIME'
            }
        except:
            return {
                'example_values': self._get_top_values(series),
                'data_type': 'DATETIME'
            }

    def _profile_categorical(self, series: pd.Series) -> Dict:
        """Profile categorical/string column"""
        value_counts = series.value_counts()

        profile = {
            'example_values': self._get_top_values(series),
            'data_type': 'STRING'
        }

        # Add most common values if cardinality is reasonable
        if len(value_counts) <= 100:
            top_5 = value_counts.head(5)
            profile['top_values'] = {str(k): int(v) for k, v in top_5.items()}

        return profile

    def _get_numeric_examples(self, series: pd.Series) -> List[str]:
        """Get representative numeric examples"""
        examples = []

        # Get min, max, and some percentiles
        try:
            percentiles = [0, 25, 50, 75, 100]
            values = series.quantile([p/100 for p in percentiles]).unique()
            examples = [str(v) for v in values[:self.max_examples]]
        except:
            examples = self._get_top_values(series)

        return examples

    def _get_datetime_examples(self, series: pd.Series) -> List[str]:
        """Get representative datetime examples"""
        try:
            # Get evenly distributed dates
            sorted_dates = series.sort_values()
            indices = np.linspace(0, len(sorted_dates)-1, min(self.max_examples, len(sorted_dates)), dtype=int)
            examples = [str(sorted_dates.iloc[i]) for i in indices]
            return examples
        except:
            return self._get_top_values(series)

    def _get_top_values(self, series: pd.Series) -> List[str]:
        """Get top values for categorical columns"""
        value_counts = series.value_counts()

        if len(value_counts) <= self.max_examples:
            return [str(v) for v in value_counts.index.tolist()]
        else:
            # Get most common values
            return [str(v) for v in value_counts.head(self.max_examples).index.tolist()]

    def _build_semantic_context(self, profile: Dict) -> str:
        """Build rich semantic context for embedding generation using natural language"""

        # Start with column identification
        context = f"Column '{profile['column_name']}' from table '{profile['table_name']}' in database '{profile['database_id']}'. "

        # Add data type description
        data_type = profile['data_type']
        context += f"Data type is {data_type}. "

        # Describe cardinality in natural language
        distinct = profile.get('distinct_count', 0)
        total = profile.get('total_count', 0)

        if distinct == 1:
            context += "It contains a single constant value. "
        elif distinct > 0:
            if distinct < 10:
                context += f"It is a low-cardinality categorical field with {distinct} unique values. "
            elif distinct < 100:
                context += f"It has moderate cardinality with {distinct} unique values. "
            elif total > 0 and (distinct / total) > 0.9:
                context += f"It appears to be a high-cardinality identifier with {distinct} unique values. "
            else:
                context += f"It contains {distinct} distinct values. "

        # Describe null patterns naturally
        null_pct = profile.get('null_percentage', 0)
        if null_pct == 0:
            context += "This column has no null values. "
        elif null_pct > 90:
            context += f"This column is mostly empty with {null_pct:.1f}% null values. "
        elif null_pct > 50:
            context += f"This column has many nulls ({null_pct:.1f}%). "
        elif null_pct > 0:
            context += f"About {null_pct:.1f}% of values are null. "

        # Add examples in a descriptive way
        examples = profile.get('example_values', [])
        if examples:
            if data_type == 'STRING':
                example_str = ', '.join([f'"{e}"' for e in examples[:5]])
                context += f"Sample values include: {example_str}. "
            elif data_type in ['NUMERIC', 'INTEGER', 'FLOAT']:
                example_str = ', '.join([str(e) for e in examples[:5]])
                context += f"Sample numeric values include: {example_str}. "
            elif data_type in ['DATETIME', 'TIMESTAMP', 'DATE']:
                example_str = ', '.join([str(e) for e in examples[:3]])
                context += f"Sample dates include: {example_str}. "
            else:
                example_str = ', '.join([str(e) for e in examples[:5]])
                context += f"Example values are: {example_str}. "

        # Add type-specific insights
        if data_type in ['NUMERIC', 'INTEGER', 'FLOAT']:
            min_val = profile.get('min_value')
            max_val = profile.get('max_value')
            avg_val = profile.get('avg_value')

            if min_val is not None and max_val is not None:
                context += f"Values range from {min_val} to {max_val}. "
            if avg_val is not None:
                context += f"The average value is {avg_val:.2f}. "

            # Add business context hints
            col_lower = profile['column_name'].lower()
            if 'price' in col_lower or 'amount' in col_lower or 'cost' in col_lower:
                context += "This appears to be a monetary field. "
            elif 'count' in col_lower or 'quantity' in col_lower:
                context += "This appears to be a quantity or count field. "
            elif 'age' in col_lower or 'year' in col_lower:
                context += "This appears to be an age or year field. "
            elif 'score' in col_lower or 'rating' in col_lower:
                context += "This appears to be a score or rating field. "

        elif data_type in ['DATETIME', 'TIMESTAMP', 'DATE']:
            min_date = profile.get('min_value')
            max_date = profile.get('max_value')
            if min_date and max_date:
                context += f"Date values range from {min_date} to {max_date}. "

            # Add temporal context
            col_lower = profile['column_name'].lower()
            if 'created' in col_lower or 'registration' in col_lower:
                context += "This appears to be a creation or registration timestamp. "
            elif 'updated' in col_lower or 'modified' in col_lower:
                context += "This appears to be an update or modification timestamp. "
            elif 'birth' in col_lower or 'dob' in col_lower:
                context += "This appears to be a birth date field. "

        elif data_type == 'STRING':
            # Add categorical insights
            if distinct > 0 and distinct < 50:
                top_values = profile.get('top_values', {})
                if top_values:
                    top_3 = list(top_values.keys())[:3]
                    context += f"The most common values are: {', '.join(top_3)}. "

            # Add semantic hints based on column name
            col_lower = profile['column_name'].lower()
            if 'email' in col_lower:
                context += "This appears to be an email address field. "
            elif 'name' in col_lower:
                context += "This appears to be a name field. "
            elif 'status' in col_lower or 'state' in col_lower:
                context += "This appears to be a status or state field. "
            elif 'category' in col_lower or 'type' in col_lower:
                context += "This appears to be a categorical classification field. "
            elif 'country' in col_lower or 'region' in col_lower or 'city' in col_lower:
                context += "This appears to be a geographic location field. "
            elif 'description' in col_lower or 'comment' in col_lower:
                context += "This appears to be a free text description field. "

        return context.strip()

    def suggest_query_patterns(self, profile: Dict) -> List[str]:
        """Suggest common query patterns based on column profile"""
        patterns = []
        column_name = profile['column_name']
        data_type = profile['data_type']
        examples = profile.get('example_values', [])

        # Patterns based on data type
        if data_type == 'STRING':
            if examples:
                patterns.append(f"WHERE {column_name} = '{examples[0]}'")
                if len(examples) > 2:
                    patterns.append(f"WHERE {column_name} IN ('{examples[0]}', '{examples[1]}')")
                patterns.append(f"WHERE {column_name} LIKE '%pattern%'")
            patterns.append(f"WHERE {column_name} IS NOT NULL")

        elif data_type == 'NUMERIC':
            min_val = profile.get('min_value')
            max_val = profile.get('max_value')
            avg_val = profile.get('avg_value')

            if min_val and max_val:
                patterns.append(f"WHERE {column_name} BETWEEN {min_val} AND {max_val}")
            if avg_val:
                patterns.append(f"WHERE {column_name} > {avg_val}")
                patterns.append(f"WHERE {column_name} <= {avg_val}")
            patterns.append(f"GROUP BY {column_name}")
            patterns.append(f"ORDER BY {column_name} DESC")

        elif data_type == 'DATETIME':
            patterns.append(f"WHERE {column_name} >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)")
            patterns.append(f"WHERE {column_name} BETWEEN '2024-01-01' AND '2024-12-31'")
            patterns.append(f"GROUP BY DATE({column_name})")
            patterns.append(f"ORDER BY {column_name} DESC")

        # Add aggregation patterns for numeric columns
        if data_type == 'NUMERIC':
            patterns.extend([
                f"SUM({column_name})",
                f"AVG({column_name})",
                f"MAX({column_name})",
                f"COUNT(DISTINCT {column_name})"
            ])

        return patterns[:5]  # Return top 5 patterns