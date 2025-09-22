"""Centralized utility functions for BQ Flow system"""

import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Any, Dict, List, Union

# Import logging
from .logging import get_logger

logger = get_logger(__name__)


def json_serial(obj: Any) -> Any:
    """JSON serializer for objects not serializable by default
    
    Handles:
    - Numpy arrays and scalars
    - Pandas Timestamps
    - Datetime objects
    - NaN/None values
    
    Args:
        obj: Object to serialize
    
    Returns:
        JSON-serializable representation
    """
    # Check for numpy arrays first (before pd.isna which fails on arrays)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif hasattr(obj, 'item'):  # Generic numpy scalar
        return obj.item()
    elif isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    elif pd.isna(obj):
        return None
    elif hasattr(obj, '__str__'):
        return str(obj)
    raise TypeError(f"Type {type(obj)} not serializable")


def clean_column_data(value: Any) -> Any:
    """Clean numpy arrays and other non-serializable types from column data
    
    Recursively processes nested structures.
    
    Args:
        value: Value to clean
    
    Returns:
        JSON-serializable value
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        # If it's already a list, clean each element
        return [clean_column_data(v) for v in value]
    if hasattr(value, 'tolist'):
        # Convert numpy arrays to list
        return value.tolist()
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, pd.Timedelta):
        return str(value)
    # For primitive types, return as-is
    if isinstance(value, (str, int, float, bool)):
        return value
    # For other types, convert to string
    return str(value)


def convert_numpy_types(value: Any) -> Any:
    """Convert numpy types to Python native types for JSON serialization
    
    Args:
        value: Value to convert
    
    Returns:
        Python native type
    """
    # Check for numpy arrays first (before pd.isna which fails on arrays)
    if isinstance(value, np.ndarray):
        return value.tolist()
    elif isinstance(value, (np.integer, np.floating)):
        return float(value)
    elif hasattr(value, 'item'):  # Generic numpy scalar
        return value.item()
    elif isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    elif pd.isna(value):
        return None
    else:
        return value


def clean_dataframe_for_json(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Clean a pandas DataFrame for JSON serialization
    
    Args:
        df: DataFrame to clean
    
    Returns:
        List of dictionaries with cleaned values
    """
    cleaned_results = []
    for _, row in df.iterrows():
        cleaned_row = {}
        for col, val in row.items():
            cleaned_row[col] = clean_column_data(val)
        cleaned_results.append(cleaned_row)
    return cleaned_results


def safe_json_dumps(obj: Any, **kwargs) -> str:
    """Safely dump object to JSON string
    
    Args:
        obj: Object to serialize
        **kwargs: Additional arguments for json.dumps
    
    Returns:
        JSON string
    """
    return json.dumps(obj, default=json_serial, **kwargs)


def validate_json_serializable(obj: Any) -> bool:
    """Check if an object is JSON serializable
    
    Args:
        obj: Object to check
    
    Returns:
        True if serializable, False otherwise
    """
    try:
        json.dumps(obj, default=json_serial)
        return True
    except (TypeError, ValueError) as e:
        logger.debug(f"Object not JSON serializable: {e}")
        return False