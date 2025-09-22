"""
AI Assistant Module for Intelligent Column Selection
Uses BigQuery's AI.GENERATE to select analytically valuable columns
"""

import json
import logging
from typing import List, Dict, Any, Optional
from .logger import ai_assistant_logger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AIAnalyst:
    """
    AI-powered analyst that selects the most valuable columns for embedding
    """

    def __init__(self, bq_client):
        self.bq_client = bq_client
        self.model = 'gemini-2.0-flash-exp'  # Using latest Gemini model

    def _build_prompt(self, table_name: str, profiles: List[Dict]) -> str:
        """Build a structured prompt for AI column selection"""

        # Build column descriptions
        profile_lines = []
        for p in profiles:
            # Extract key information
            col_name = p['column_name']
            data_type = p['data_type']
            distinct = p.get('distinct_count', 0)
            null_pct = p.get('null_percentage', 0)
            examples = p.get('example_values', [])[:3]

            # Format example values based on type
            if examples and data_type == 'STRING':
                example_str = ', '.join([f'"{e}"' for e in examples])
            else:
                example_str = ', '.join([str(e) for e in examples]) if examples else 'N/A'

            # Create description line
            line = f"- {col_name}: {data_type}, {distinct} unique values, {null_pct:.1f}% null. Examples: [{example_str}]"
            profile_lines.append(line)

        profile_text = "\n".join(profile_lines)

        # Build the comprehensive prompt
        prompt = f"""You are an expert data analyst evaluating database columns for a semantic search system.

Analyze this table schema and select columns most valuable for business intelligence queries:

Table: {table_name}
Columns:
{profile_text}

Selection Criteria:
INCLUDE columns that are:
- Categorical fields with business meaning (status, category, type, country, etc.)
- Numeric measures useful for aggregation (amount, quantity, price, score, etc.)
- Descriptive text fields with moderate cardinality
- Date/time fields useful for temporal analysis
- Foreign keys that represent relationships

EXCLUDE columns that are:
- Primary keys or unique identifiers (id, uuid, key)
- System timestamps (created_at, updated_at, modified_at)
- High-cardinality free text (descriptions, comments) unless categorical
- Binary or technical fields with no business meaning
- Columns with >90% null values

Your task: Select the columns that would be most useful for answering typical analytical questions like:
- "Show me sales by category"
- "What are the top products?"
- "Filter by status and region"
- "Compare metrics over time"

Return ONLY a JSON array of column names. No explanation needed.
Example format: ["column1", "column2", "column3"]

Important: Be selective. Choose only columns with clear analytical value."""

        return prompt

    def select_key_columns(self, table_name: str, profiles: List[Dict],
                          max_columns: int = 50, min_columns: int = 3) -> Dict[str, Any]:
        """
        Use AI to select the most valuable columns for embedding

        Returns:
            Dictionary with:
            - selected_columns: List of column names to embed
            - excluded_columns: List of column names to skip
            - selection_reason: Brief explanation (if available)
            - success: Boolean indicating if AI selection worked
        """

        result = {
            'selected_columns': [],
            'excluded_columns': [],
            'selection_reason': '',
            'success': False
        }

        # Build the AI prompt
        prompt = self._build_prompt(table_name, profiles)

        # Escape the prompt for SQL - replace backslashes and quotes
        escaped_prompt = prompt.replace('\\', '\\\\').replace('"""', r'\"""')

        # Use AI.GENERATE with the same pattern as the main module
        # Get connection from environment or use default
        import os
        connection_id = os.getenv('GEMINI_CONNECTION', 'us.gemini_connection')

        query = f'''
        SELECT AI.GENERATE(
            """{escaped_prompt}""",
            connection_id => '{connection_id}',
            endpoint => 'gemini-2.5-flash'
        ).result AS selected_columns_json
        '''

        logger.info(f"Requesting AI column selection for table: {table_name}")

        # Execute the query
        query_result = self.bq_client.execute_query(query)

        if not query_result or len(query_result) == 0:
            raise Exception("No response from AI.GENERATE")

        # Extract the generated text
        json_string = query_result[0].get('selected_columns_json', '[]')
        logger.info(f"Raw AI response: {json_string[:500]}...")  # Log first 500 chars

        # Clean and parse JSON
        json_string = json_string.strip()

        # Find JSON array in the response (in case there's extra text)
        import re
        json_match = re.search(r'\[.*?\]', json_string, re.DOTALL)
        if json_match:
            json_string = json_match.group(0)
            logger.info(f"Extracted JSON: {json_string}")
        else:
            logger.warning(f"Could not find JSON array in response: {json_string}")

        try:
            # Parse the JSON array
            selected = json.loads(json_string)
            logger.info(f"Parsed {len(selected)} columns from AI response")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            logger.error(f"JSON string was: {json_string}")
            selected = []

        # Validate the selection
        all_columns = [p['column_name'] for p in profiles]
        logger.info(f"Available columns: {all_columns}")
        logger.info(f"AI selected columns: {selected}")

        # Check for case sensitivity issues
        all_columns_lower = {col.lower(): col for col in all_columns}
        validated_selected = []
        for col in selected:
            if col in all_columns:
                validated_selected.append(col)
            elif col.lower() in all_columns_lower:
                # Try case-insensitive match
                validated_selected.append(all_columns_lower[col.lower()])
                logger.info(f"Case-corrected column: {col} -> {all_columns_lower[col.lower()]}")
            else:
                logger.warning(f"Column '{col}' not found in available columns")

        selected = validated_selected
        logger.info(f"Validated {len(selected)} columns")

        if not selected:
            # Fallback: Select all columns if AI fails
            logger.warning("AI did not select any valid columns, falling back to selecting all columns")
            selected = [p['column_name'] for p in profiles]
            result['selection_reason'] = f"Fallback: selected all {len(selected)} columns (AI selection failed)"

        # Apply min/max constraints
        if len(selected) < min_columns and len(all_columns) >= min_columns:
            # Too few selected, add some more based on distinct count
            remaining = [p for p in profiles if p['column_name'] not in selected]
            remaining.sort(key=lambda x: x.get('distinct_count', 0), reverse=True)
            for p in remaining[:min_columns - len(selected)]:
                selected.append(p['column_name'])

        elif len(selected) > max_columns:
            # Too many selected, trim to max
            selected = selected[:max_columns]

        # Build excluded list
        excluded = [col for col in all_columns if col not in selected]

        result['selected_columns'] = selected
        result['excluded_columns'] = excluded
        if not result['selection_reason']:  # Only set if not already set by fallback
            result['selection_reason'] = f"AI selected {len(selected)}/{len(all_columns)} columns based on analytical value"
        result['success'] = True

        logger.info(f"AI selected {len(selected)} columns out of {len(all_columns)}")

        return result


    def filter_profiles(self, profiles: List[Dict], selected_columns: List[str]) -> List[Dict]:
        """
        Filter profiles to only include selected columns
        """
        return [p for p in profiles if p['column_name'] in selected_columns]

    def estimate_cost_savings(self, total_columns: int, selected_columns: int,
                            cost_per_1k: float = 0.01) -> Dict[str, float]:
        """
        Calculate estimated cost savings from column selection
        """
        reduction_pct = (1 - selected_columns / total_columns) * 100 if total_columns > 0 else 0

        # Estimate tokens per column (rough estimate)
        avg_tokens_per_column = 50

        total_cost = (total_columns * avg_tokens_per_column / 1000) * cost_per_1k
        reduced_cost = (selected_columns * avg_tokens_per_column / 1000) * cost_per_1k
        savings = total_cost - reduced_cost

        return {
            'total_cost': total_cost,
            'reduced_cost': reduced_cost,
            'savings': savings,
            'reduction_percentage': reduction_pct
        }