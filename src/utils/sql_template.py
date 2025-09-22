"""
SQL Template Utility
Replaces placeholders in SQL files with actual environment values
"""

import os
import re
from typing import Dict, Optional

def get_template_values() -> Dict[str, str]:
    """Get template values from environment"""
    return {
        'PROJECT_ID': os.getenv('PROJECT_ID', ''),
        'DATASET_ID': os.getenv('DATASET_ID', ''),
        'GEMINI_CONNECTION': os.getenv('GEMINI_CONNECTION', 'us.gemini_connection'),
        'LOCATION': os.getenv('LOCATION', 'US')
    }

def replace_placeholders(sql: str, values: Optional[Dict[str, str]] = None) -> str:
    """
    Replace ${VARIABLE} placeholders in SQL with actual values

    Args:
        sql: SQL string with placeholders
        values: Dictionary of placeholder values (defaults to environment)

    Returns:
        SQL string with placeholders replaced

    Example:
        sql = "SELECT * FROM `${PROJECT_ID}.${DATASET_ID}.table`"
        result = replace_placeholders(sql)
        # Returns: "SELECT * FROM `my-project.my-dataset.table`"
    """
    if values is None:
        values = get_template_values()

    # Validate required values
    if not values.get('PROJECT_ID'):
        raise ValueError("PROJECT_ID is required but not set in environment")
    if not values.get('DATASET_ID'):
        raise ValueError("DATASET_ID is required but not set in environment")

    # Replace all ${VARIABLE} patterns
    for key, value in values.items():
        pattern = f'${{{{?{key}}}}}?'  # Matches ${KEY} or ${KEY}
        sql = re.sub(pattern, value, sql)

    # Check for any remaining placeholders
    remaining = re.findall(r'\$\{[^}]+\}', sql)
    if remaining:
        raise ValueError(f"Unresolved placeholders in SQL: {', '.join(remaining)}")

    return sql

def process_sql_file(file_path: str, output_path: Optional[str] = None) -> str:
    """
    Process a SQL file and replace placeholders

    Args:
        file_path: Path to SQL file with placeholders
        output_path: Optional path to write processed SQL (if not provided, returns string)

    Returns:
        Processed SQL string
    """
    with open(file_path, 'r') as f:
        sql = f.read()

    processed_sql = replace_placeholders(sql)

    if output_path:
        with open(output_path, 'w') as f:
            f.write(processed_sql)
        print(f"Processed SQL written to: {output_path}")

    return processed_sql

def validate_sql_templates(directory: str = 'bq_flow_onboarding/sql') -> bool:
    """
    Validate that all SQL template files can be processed

    Args:
        directory: Directory containing SQL files

    Returns:
        True if all files are valid
    """
    import glob

    sql_files = glob.glob(f"{directory}/*.sql")
    all_valid = True

    for file_path in sql_files:
        try:
            with open(file_path, 'r') as f:
                sql = f.read()
            # Try to replace placeholders
            processed = replace_placeholders(sql)
            print(f"✅ {file_path} - Valid")
        except Exception as e:
            print(f"❌ {file_path} - Error: {str(e)}")
            all_valid = False

    return all_valid