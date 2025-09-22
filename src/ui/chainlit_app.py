"""
Enhanced Chainlit UI with WebSocket Streaming Support
Provides real-time progress updates during query processing
"""

import chainlit as cl
import aiohttp
import asyncio
import json
import os
import uuid
from typing import Dict, Any, Optional, List
import httpx
import yaml
import pandas as pd
import sys
from pathlib import Path

# Add project root to sys.path for imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# Import forecast detection function from AI.FORECAST module
from src.core.bigquery_ai_forecast import detect_time_series

# Import centralized logging from current directory
from chainlit_logging import (
        setup_logging,
        get_logger,
        set_request_context,
        clear_request_context,
        log_performance,
        log_api_request,
        log_api_response,
    )

# Setup logging
setup_logging(
    log_level=os.getenv('LOG_LEVEL', 'INFO'),
    log_file='logs/chainlit_ui.log',
    use_json=False,  # Human-readable for UI logs
    console_output=True,
)

# Get logger for this module
logger = get_logger(__name__)

config_file = project_root / os.getenv('CONFIG_FILE', 'config/config.yaml')
if not config_file.exists():
    logger.error(f'Configuration file not found: {config_file}')
    raise FileNotFoundError(f'Configuration file not found: {config_file}')
with open(config_file, 'r') as f:
    config = yaml.safe_load(f)

# Get the backend configuration dictionary
backend_config = config['ui']['backend']

# Assign values to your variables
WS_URL = backend_config['ws_url']
REST_URL = backend_config['rest_url']
FORECAST_URL = backend_config['forecast_url']
API_BASE = backend_config['api_base']
# HTTP client for REST fallback
httpx_client = httpx.AsyncClient(timeout=120.0)


class StreamingQueryHandler:
    """Handles queries with WebSocket streaming or REST fallback"""

    def __init__(self):
        self.ws_url = WS_URL
        self.rest_url = REST_URL
        self.ws_session: Optional[aiohttp.ClientSession] = None
        self.ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self.use_streaming = True

    @log_performance
    async def connect_websocket(self) -> bool:
        """Try to establish WebSocket connection"""
        logger.debug('Attempting WebSocket connection', extra={'extra_fields': {'url': self.ws_url}})
        try:
            if not self.ws_session:
                self.ws_session = aiohttp.ClientSession()

            self.ws = await self.ws_session.ws_connect(self.ws_url)
            logger.info('WebSocket connected successfully', extra={'extra_fields': {'url': self.ws_url}})
            return True
        except Exception as e:
            logger.error(f'WebSocket connection failed: {str(e)}', extra={'extra_fields': {'url': self.ws_url}})
            self.use_streaming = False
            return False

    @log_performance
    async def disconnect_websocket(self):
        """Close WebSocket connection"""
        logger.debug('Disconnecting WebSocket', extra={'extra_fields': {'url': self.ws_url}})
        if self.ws:
            await self.ws.close()
            self.ws = None
            logger.info('WebSocket connection closed', extra={'extra_fields': {'url': self.ws_url}})
        # Keep session for reuse within same chat session

    async def cleanup_session(self):
        """Clean up WebSocket session - call this when chat ends"""
        if self.ws:
            await self.disconnect_websocket()
        if self.ws_session:
            await self.ws_session.close()
            self.ws_session = None
            logger.info('WebSocket session cleaned up')

    @log_performance
    async def send_heartbeat(self):
        """Send periodic heartbeat to keep connection alive"""
        logger.debug('Starting WebSocket heartbeat', extra={'extra_fields': {'interval': '20s'}})
        try:
            while self.ws and not self.ws.closed:
                await asyncio.sleep(20)  # Send heartbeat every 20 seconds
                if self.ws and not self.ws.closed:
                    await self.ws.send_json({'type': 'ping'})
                    logger.debug('Sent WebSocket heartbeat', extra={'extra_fields': {'type': 'ping'}})
        except Exception as e:
            logger.error(f'Heartbeat error: {str(e)}', extra={'extra_fields': {'interval': '20s'}})

    @log_performance
    async def process_query(self, query: str, database_id: str, session_id: str = None) -> Dict[str, Any]:
        """Process query with streaming or fallback to REST"""
        logger.info('Processing user query', extra={'extra_fields': {'query': query[:100], 'database_id': database_id}})
        if self.use_streaming and await self.connect_websocket():
            return await self.process_with_streaming(query, database_id, session_id)
        else:
            return await self.process_with_rest(query, database_id)

    def _format_error_message(self, error: str, job_id: Optional[str] = None) -> str:
        """Format error messages for user-friendly display"""
        base_message = '❌ **Query Failed**'
        job_id_info = f'\n\n**Support Reference**: Job ID {job_id}' if job_id else ''

        # Handle specific error types
        if '404 Not found' in error and 'Table' in error:
            table_name = error.split('Table ')[1].split(' was')[0] if 'Table ' in error else 'unknown'
            return f"""{base_message}

**Issue**: Table `{table_name}` not found in the selected dataset.

**💡 Possible Solutions**:
1. **Check Dataset Selection**: Ensure you selected the correct pre-embedded dataset
2. **Verify Onboarding**: The table might not be onboarded yet - run the BQ Flow Data Onboarding tool
3. **Try Different Query**: Use a simpler query or ask about available tables
4. **Check Spelling**: Verify table and column names in your query

**🔄 Quick Fix**: Try asking "What tables are available?" or select a different dataset.{job_id_info}"""

        elif 'permission' in error.lower() or 'access' in error.lower():
            return f"""{base_message}

**Issue**: Insufficient permissions to access the requested data.

**💡 Possible Solutions**:
1. **Check Credentials**: Verify your Google Cloud authentication
2. **Dataset Access**: Ensure you have BigQuery access to the selected dataset
3. **Try Public Datasets**: Use pre-embedded public datasets like `thelook_ecommerce`

**🔄 Quick Fix**: Try a different pre-embedded dataset or contact your administrator.{job_id_info}"""

        elif 'timeout' in error.lower() or 'deadline' in error.lower():
            return f"""{base_message}

**Issue**: Query took too long to execute and timed out.

**💡 Possible Solutions**:
1. **Simplify Query**: Try a more specific question with fewer data points
2. **Add Filters**: Include date ranges or limits (e.g., "last 30 days", "top 10")
3. **Break Down Query**: Split complex questions into smaller parts

**🔄 Quick Fix**: Try asking for a smaller subset of data or add time/count limits.{job_id_info}"""

        elif 'syntax' in error.lower() or 'invalid' in error.lower():
            return f"""{base_message}

**Issue**: The generated SQL has syntax errors or invalid references.

**💡 Possible Solutions**:
1. **Rephrase Query**: Try asking your question differently
2. **Be More Specific**: Include specific column names or table references
3. **Check Data Onboarding**: Ensure the dataset is properly onboarded with embeddings

**🔄 Quick Fix**: Try rephrasing your question more clearly or ask about available data first.{job_id_info}"""

        elif 'embedding' in error.lower() or 'vector' in error.lower():
            return f"""{base_message}

**Issue**: Problem with semantic search or embeddings.

**💡 Possible Solutions**:
1. **Run Onboarding**: The dataset may not have proper embeddings - use the BQ Flow Data Onboarding tool
2. **Check Vector Indexes**: Ensure VECTOR_SEARCH indexes are built
3. **Try Simpler Query**: Start with basic questions about the data

**🔄 Quick Fix**: Go to `cd bq_flow_onboarding && ./run.sh` to properly onboard the dataset.{job_id_info}"""

        else:
            # Generic error handling
            return f"""{base_message}

**Issue**: {error}

**💡 General Solutions**:
1. **Try Again**: Sometimes temporary issues resolve themselves
2. **Rephrase Query**: Ask your question differently
3. **Check Dataset**: Ensure you selected a properly onboarded dataset
4. **Simplify**: Start with a basic question about the data

**🔄 Quick Fix**: Try rephrasing your question or selecting a different dataset.{job_id_info}"""

    @log_performance
    async def process_with_streaming(self, query: str, database_id: str, session_id: str) -> Dict[str, Any]:
        """Process using WebSocket with real-time updates"""
        # Create progress message
        progress_msg = cl.Message(content='🔍 Starting query processing...')
        await progress_msg.send()

        sql_msg = None
        feature_msg = None
        last_progress = 0
        heartbeat_task = None

        try:
            # Start heartbeat task
            heartbeat_task = asyncio.create_task(self.send_heartbeat())

            # Log and send query through WebSocket
            log_api_request(logger, 'WS', self.ws_url, {'query': query, 'database_id': database_id})
            await self.ws.send_json(
                {
                    'type': 'query',
                    'query': query,
                    'database_id': database_id,
                    'session_id': session_id,
                }
            )

            # Listen for updates
            async for msg in self.ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)

                    # Update progress message
                    progress = data.get('progress', 0)
                    stage = data.get('stage', 'unknown')
                    message = data.get('message', 'Processing...')
                    detail = data.get('detail', '')
                    elapsed = data.get('elapsed', 0)

                    # Create enhanced progress visualizations
                    progress_bar = self.create_progress_bar(progress)
                    stage_visual = self.create_stage_visualization(stage, progress)

                    # Format enhanced progress message with visual pipeline
                    progress_content = f"""### 🎆 {message}

{progress_bar}

{stage_visual}

**⏱️ Time Elapsed**: {elapsed}s | **🎯 Current Stage**: `{stage}`

{f"🔍 **Live Update**: {detail}" if detail else ""}
"""

                    # Enhanced feature badges with descriptions
                    if 'features_used' in data and data['features_used']:
                        if not feature_msg:
                            feature_msg = cl.Message(content='## 🤖 **BigQuery AI Features Activated**\n')
                            await feature_msg.send()

                        # Track features in session
                        session_features = cl.user_session.get('features_used', set())
                        session_features.update(data['features_used'])
                        cl.user_session.set('features_used', session_features)

                        # Create visual feature cards
                        features_display = self.format_feature_cards(data['features_used'])
                        feature_msg.content = f'## 🤖 **BigQuery AI Features Activated**\n\n{features_display}\n\n'
                        await feature_msg.update()

                    progress_msg.content = progress_content
                    await progress_msg.update()

                    # Show SQL preview when available
                    if 'sql' in data and data['sql'] and not sql_msg:
                        sql_preview = data['sql'][:1000] + '...' if len(data['sql']) > 1000 else data['sql']
                        sql_msg = cl.Message(content=sql_preview, language='sql')
                        await sql_msg.send()

                    # Handle completion
                    if stage == 'complete':
                        # Show simple completion message
                        completion_msg = "🎯 **Query Processing Complete!**"
                        await cl.Message(content=completion_msg).send()

                        await progress_msg.remove()
                        if feature_msg:
                            await feature_msg.remove()

                        # Ensure database_id is in the data
                        data['database_id'] = database_id
                        await self.display_results(data)
                        log_api_response(logger, 200, int(data.get('elapsed', 0) * 1000))
                        return data

                    # Handle error with proper cleanup
                    if stage == 'error':
                        await progress_msg.remove()  # Remove progress message
                        if feature_msg:
                            await feature_msg.remove()  # Remove feature message

                        job_id = data.get('job_id', None)
                        user_message = self._format_error_message(message, job_id)
                        await cl.Message(content=user_message).send()
                        log_api_response(logger, 500, int(data.get('elapsed', 0) * 1000))
                        return {'error': message}

                    last_progress = progress

                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(
                        f'WebSocket error received: {msg.data}', extra={'extra_fields': {'stage': 'streaming'}}
                    )
                    raise Exception(f'WebSocket error: {msg.data}')

        except Exception as e:
            logger.error(
                f'Streaming error: {str(e)}',
                extra={'extra_fields': {'query': query[:100], 'database_id': database_id}},
            )

            # Clean up any remaining progress messages
            try:
                await progress_msg.remove()
            except:
                pass
            try:
                if feature_msg:
                    await feature_msg.remove()
            except:
                pass

            # Check if it's a connection error - try REST fallback
            if 'connection' in str(e).lower() or 'websocket' in str(e).lower():
                await cl.Message(content='⚠️ **Connection Issue**: WebSocket failed, switching to standard mode...').send()
                return await self.process_with_rest(query, database_id)

            # For other errors, show formatted error message
            job_id = str(e).split('Job ID: ')[1] if 'Job ID: ' in str(e) else None
            user_message = self._format_error_message(str(e), job_id)
            await cl.Message(content=user_message).send()
            return {'error': str(e)}

        finally:
            # Cancel heartbeat task if running
            if heartbeat_task and not heartbeat_task.done():
                heartbeat_task.cancel()
            await self.disconnect_websocket()

    def _format_forecast_error(self, error: str, horizon: int, date_col: str, value_col: str) -> str:
        """Format forecast error messages for user-friendly display"""
        base_message = '❌ **Forecast Generation Failed**'

        if 'timeout' in error.lower() or 'deadline' in error.lower():
            return f"""{base_message}

**Issue**: Forecast generation took too long and timed out.

**💡 Solutions**:
1. **Reduce Horizon**: Try forecasting fewer periods (e.g., 7 or 30 instead of {horizon})
2. **Simplify Data**: Use a simpler time-series query with fewer data points
3. **Check Data Quality**: Ensure your time-series data has consistent intervals

**🔄 Quick Fix**: Try the 7-period forecast button or run a simpler time-series query."""

        elif 'insufficient' in error.lower() or 'not enough' in error.lower():
            return f"""{base_message}

**Issue**: Not enough historical data for reliable forecasting.

**💡 Solutions**:
1. **Expand Time Range**: Include more historical data in your query
2. **Check Data Quality**: Ensure the `{date_col}` column has sufficient data points
3. **Verify Data**: Make sure `{value_col}` contains numeric values

**🔄 Quick Fix**: Try asking for data over a longer time period (e.g., "last 2 years" instead of "last month")."""

        elif 'model' in error.lower() or 'timesfm' in error.lower():
            return f"""{base_message}

**Issue**: Problem with the AI.FORECAST model (TimesFM 2.0).

**💡 Solutions**:
1. **Retry**: Sometimes model issues are temporary
2. **Check Data Format**: Ensure dates are properly formatted and values are numeric
3. **Simplify Query**: Try a different time-series query

**🔄 Quick Fix**: Wait a moment and try the forecast again, or try a different time period."""

        elif 'permission' in error.lower() or 'access' in error.lower():
            return f"""{base_message}

**Issue**: Insufficient permissions to use AI.FORECAST.

**💡 Solutions**:
1. **Check Credentials**: Verify BigQuery AI functions are enabled in your project
2. **Contact Admin**: You may need AI.FORECAST permissions
3. **Try Different Dataset**: Use a pre-configured dataset

**🔄 Quick Fix**: Contact your Google Cloud administrator to enable AI.FORECAST access."""

        elif 'sql' in error.lower() or 'syntax' in error.lower():
            return f"""{base_message}

**Issue**: Problem generating the forecast SQL query.

**💡 Solutions**:
1. **Check Original Query**: Ensure your time-series query worked correctly
2. **Verify Columns**: Make sure `{date_col}` and `{value_col}` exist and are correct types
3. **Try Different Query**: Ask for time-series data in a different way

**🔄 Quick Fix**: Try running a simpler time-series query first, then forecast."""

        else:
            # Generic forecast error
            return f"""{base_message}

**Issue**: {error}

**💡 General Solutions**:
1. **Retry Forecast**: Try the forecast again after a moment
2. **Check Data**: Ensure your query returns proper time-series data
3. **Reduce Horizon**: Try forecasting fewer periods ({horizon} → 7 or 30)
4. **Different Query**: Ask for time-series data in a different way

**💡 Data Requirements**:
- Date/time column: `{date_col}`
- Numeric value column: `{value_col}`
- At least 20 historical data points
- Consistent time intervals

**🔄 Quick Fix**: Try running your original query again, then use a shorter forecast horizon."""

    @log_performance
    async def process_with_rest(self, query: str, database_id: str) -> Dict[str, Any]:
        """Fallback to REST API processing"""
        status_msg = cl.Message(content='📄 Processing query (standard mode)...')
        await status_msg.send()

        try:
            log_api_request(logger, 'POST', self.rest_url, {'query': query, 'database_id': database_id})
            response = await httpx_client.post(
                self.rest_url,
                json={'query': query, 'database_id': database_id, 'use_cache': False},
            )
            response.raise_for_status()
            result = response.json()
            # Ensure database_id is in the result
            result['database_id'] = database_id

            await status_msg.remove()
            await self.display_results(result)
            log_api_response(logger, response.status_code, int(result.get('elapsed', 0) * 1000))
            return result

        except Exception as e:
            logger.error(
                f'REST API error: {str(e)}',
                extra={'extra_fields': {'query': query[:100], 'database_id': database_id}},
            )

            # Clean up status message
            try:
                await status_msg.remove()
            except:
                pass

            job_id = str(e).split('Job ID: ')[1] if 'Job ID: ' in str(e) else None
            user_message = self._format_error_message(str(e), job_id)
            await cl.Message(content=user_message).send()
            log_api_response(logger, 500, 0)
            return {'error': str(e)}

    async def display_results(self, data: Dict[str, Any]):
        """Display final results with enhanced hackathon-winning formatting"""
        # Extract data
        results = data.get('results', [])
        insights = data.get('insights')
        sql = data.get('sql', data.get('generated_sql', ''))
        features = data.get('features_used', data.get('bigquery_features_used', []))

        # Create results header
        results_content = ""

        # Format features with descriptions (check most specific patterns first)
        seen_features = set()
        for feature in features:
            # Determine the feature type and avoid duplicates
            feature_key = None
            feature_line = None

            if 'ML.GENERATE_EMBEDDING' in feature:
                feature_key = 'embedding'
                feature_line = "🧬 **ML.GENERATE_EMBEDDING** - Text to semantic vectors\n"
            elif 'VECTOR_SEARCH' in feature and 'IVF' in feature:
                feature_key = 'vector_search'
                feature_line = "🔍 **VECTOR_SEARCH with IVF index** - Semantic similarity search with optimization\n"
            elif 'AI.GENERATE_TABLE' in feature:
                feature_key = 'generate_table'
                feature_line = "📊 **AI.GENERATE_TABLE** - Structured insights extraction\n"
            elif 'AI.GENERATE' in feature and 'Gemini' in feature:
                feature_key = 'generate_sql'
                feature_line = "✨ **AI.GENERATE** - Natural language to SQL translation (Gemini 2.5 Flash)\n"
            elif 'AI.GENERATE' in feature and 'Query Refinement' in feature:
                feature_key = 'generate_refine'
                feature_line = "🔧 **AI.GENERATE** - Query refinement and optimization\n"
            elif 'AI.GENERATE' in feature:
                feature_key = 'generate_other'
                feature_line = "✨ **AI.GENERATE** - AI-powered generation\n"
            elif 'ML.DISTANCE' in feature:
                feature_key = 'distance'
                feature_line = "📏 **ML.DISTANCE** - Vector similarity computation\n"
            elif 'AI-Generated Explanations' in feature:
                feature_key = 'explanations'
                feature_line = "💡 **AI-Generated Explanations** - Column selection reasoning\n"
            elif 'AI-Generated Insights' in feature:
                feature_key = 'insights_generic'
                feature_line = "💡 **AI-Generated Insights** - Business intelligence\n"
            else:
                feature_key = feature
                feature_line = f"✅ **{feature}**\n"

            # Only add if not already seen
            if feature_key not in seen_features:
                seen_features.add(feature_key)
                results_content += feature_line
        
        if not features:
            results_content += "No AI features tracked\n"

        results_content += f"""

---

### 📝 **Generated SQL** _(Powered by BigQuery AI)_

```sql
{sql}
```

---

### 📊 **Query Results**

{self.format_enhanced_results_table(results)}
"""

        # Add insights if available
        if insights:
            if isinstance(insights, str):
                try:
                    # Try to parse JSON string
                    insights_obj = json.loads(insights)
                    insights_text = self.format_insights(insights_obj)
                except:
                    insights_text = insights
            elif isinstance(insights, dict):
                insights_text = self.format_insights(insights)
            else:
                insights_text = str(insights)

            results_content += f"""

---

### 💡 **AI-Generated Business Intelligence** _(Powered by AI.GENERATE_TABLE)_

{insights_text}

---
"""

        # Send main results message
        main_msg = cl.Message(content=results_content)
        await main_msg.send()

        # Check if results are forecastable and add button if applicable
        await self.check_and_add_forecast_button(results, sql, data)

    def create_progress_bar(self, progress: int) -> str:
        """Create an enhanced visual progress bar with stage indicators"""
        filled = int(progress / 5)  # 20 segments total
        empty = 20 - filled

        # Create gradient effect with different block characters
        if filled > 0:
            bar = '█' * (filled - 1) + '▓' + '░' * empty if filled < 20 else '█' * 20
        else:
            bar = '░' * 20

        # Add percentage indicator
        percentage = f"{progress:3d}%"

        # Create the full progress visualization
        return f'⟦{bar}⟧ {percentage}'

    def format_feature_cards(self, features: List[str]) -> str:
        """Create visual cards for BigQuery AI features"""
        feature_info = {
            'ML.GENERATE_EMBEDDING': ('🧬', 'Text → 768-dim vectors', 'Semantic understanding'),
            'VECTOR_SEARCH': ('🔎', 'IVF-indexed search', 'Find similar columns'),
            'AI.GENERATE': ('✨', 'Gemini 2.5 Flash', 'Natural language → SQL'),
            'AI.GENERATE_TABLE': ('📊', 'Structured extraction', 'Business insights'),
            'AI.FORECAST': ('🔮', 'TimesFM 2.0', 'Time-series predictions'),
            'ML.DISTANCE': ('📏', 'Cosine similarity', 'Fallback matching')
        }

        cards = []
        for feature in features:
            for key, (emoji, tech, desc) in feature_info.items():
                if key in feature:
                    cards.append(f"{emoji} **{key}**\n   `{tech}` - {desc}")
                    break
            else:
                # Generic feature
                cards.append(f"✅ **{feature}**")

        return '\n\n'.join(cards)

    def create_stage_visualization(self, current_stage: str, progress: int) -> str:
        """Create visual representation of the 13-stage pipeline"""
        stages = [
            ('🔍', 'init', 'initialization'),
            ('🧠', 'understand', 'understanding'),
            ('🧬', 'embed', 'embedding'),
            ('🔎', 'search', 'searching'),
            ('📋', 'columns', 'columns_found'),
            ('✨', 'generate', 'sql_generation'),
            ('🔨', 'build', 'sql_building'),
            ('✅', 'validate', 'sql_complete'),
            ('⚡', 'execute', 'executing'),
            ('📊', 'progress', 'execution_progress'),
            ('📈', 'results', 'results_ready'),
            ('💡', 'insights', 'insights'),
            ('🎯', 'complete', 'complete')
        ]

        visualization = '\n**Pipeline Progress:**\n'
        visualization += '```\n'

        for i, (emoji, short_name, full_name) in enumerate(stages):
            is_current = full_name == current_stage
            is_completed = i < len(stages) * progress // 100

            if is_current:
                visualization += f' ➤ {emoji} {short_name.upper()} '
            elif is_completed:
                visualization += f' ✓ {emoji} {short_name} '
            else:
                visualization += f' ○ {emoji} {short_name} '

            if (i + 1) % 4 == 0 and i < len(stages) - 1:
                visualization += '\n'

        visualization += '\n```'
        return visualization

    def format_results_table(self, results: List[Dict], limit: int = 10) -> str:
        """Format query results as markdown table"""
        if not results:
            return '_No results found_'

        df = pd.DataFrame(results[:limit])

        # Format the dataframe as markdown
        table = df.to_markdown(index=False)

        if len(results) > limit:
            table += f'\n\n_Showing first {limit} of {len(results)} rows_'

        return table

    def format_enhanced_results_table(self, results: List[Dict], limit: int = 10) -> str:
        """Format query results with enhanced visualization hints"""
        if not results:
            return '🔍 _No results found - try adjusting your query_'

        df = pd.DataFrame(results[:limit])

        # Add visual indicators for different data types
        result_str = ""

        # Check if results contain time-series data
        has_dates = any('date' in col.lower() or 'time' in col.lower() for col in df.columns)
        has_numbers = any(df[col].dtype in ['int64', 'float64'] for col in df.columns)

        if has_dates and has_numbers:
            result_str += "📈 **Time-Series Data Detected** - _Forecast button will appear below_\n\n"

        # Format the dataframe as markdown with emojis for column types
        table_lines = df.to_markdown(index=False).split('\n')
        if len(table_lines) > 0:
            # Add emojis to header
            header = table_lines[0]
            for col in df.columns:
                if 'date' in col.lower() or 'time' in col.lower():
                    header = header.replace(col, f'📅 {col}')
                elif df[col].dtype in ['int64', 'float64']:
                    header = header.replace(col, f'🔢 {col}')
                elif 'id' in col.lower():
                    header = header.replace(col, f'🆔 {col}')
                elif 'name' in col.lower() or 'title' in col.lower():
                    header = header.replace(col, f'🏷️ {col}')
            table_lines[0] = header

        table = '\n'.join(table_lines)

        if len(results) > limit:
            table += f'\n\n📄 _Showing first {limit} of **{len(results)}** total rows_'

        return result_str + table

    def format_insights(self, insights: dict) -> str:
        """Format enhanced AI insights as markdown"""
        formatted = ''

        # Key finding with confidence
        if 'key_finding' in insights:
            confidence = insights.get('confidence_score', 0.8)
            formatted += f"**🎯 Key Finding** (Confidence: {confidence:.0%}):\n{insights['key_finding']}\n\n"

        # Trend analysis with direction
        if 'trend' in insights:
            direction = insights.get('trend_direction', '')
            trend_emoji = {
                'UPWARD': '↗️', 'DOWNWARD': '↘️',
                'STABLE': '→', 'VOLATILE': '⚡'
            }.get(direction, '📈')
            formatted += f"**{trend_emoji} Trend Analysis:**\n{insights['trend']}\n"
            if direction:
                formatted += f"*Direction: {direction}*\n\n"
            else:
                formatted += "\n"

        # Business Impact and Priority
        if 'business_impact' in insights:
            priority = insights.get('action_priority', 'MEDIUM')
            priority_emoji = {'HIGH': '🔴', 'MEDIUM': '🟡', 'LOW': '🟢'}.get(priority, '⚪')
            formatted += f"**💼 Business Impact {priority_emoji}:**\n{insights['business_impact']}\n"
            formatted += f"*Priority: {priority}*\n\n"

        # Recommendation with timeline
        if 'recommendation' in insights:
            timeline = insights.get('implementation_timeline', '')
            timeline_text = {
                'IMMEDIATE': '1-7 days',
                'SHORT_TERM': '1-4 weeks',
                'LONG_TERM': '1-3 months'
            }.get(timeline, '')
            formatted += f"**💡 Recommendation:**\n{insights['recommendation']}\n"
            if timeline_text:
                formatted += f"*Timeline: {timeline_text}*\n\n"
            else:
                formatted += "\n"

        # Risk Assessment
        if 'risk_assessment' in insights and insights['risk_assessment'] != 'N/A':
            formatted += f"**⚠️ Risk Assessment:**\n{insights['risk_assessment']}\n\n"

        # Opportunity Highlight
        if 'opportunity_highlight' in insights and insights['opportunity_highlight'] != 'N/A':
            formatted += f"**🚀 Opportunity:**\n{insights['opportunity_highlight']}\n\n"

        # Data Quality indicator
        if 'data_quality' in insights:
            quality_emoji = {
                'Excellent': '✅', 'Good': '👍',
                'Limited': '⚠️', 'Poor': '❌'
            }.get(insights['data_quality'], '📊')
            formatted += f"**{quality_emoji} Data Quality:** {insights['data_quality']}\n\n"

        # Next questions
        if 'next_questions' in insights and isinstance(insights['next_questions'], list):
            formatted += '**❓ Strategic Next Questions:**\n'
            for i, question in enumerate(insights['next_questions'], 1):
                formatted += f'{i}. {question}\n'

        return formatted if formatted else json.dumps(insights, indent=2)

    async def check_and_add_forecast_button(self, results: List[Dict], sql: str, original_data: Dict):
        """Check if results contain time-series data and add forecast button"""
        if not results:
            return

        if len(results) < 20:
            # Show helpful message about forecast requirements
            await cl.Message(content=f"""📈 **Time-Series Info**: Your query returned {len(results)} rows.

💡 **For AI.FORECAST**: You need at least 20 data points with date and numeric columns.

**Try**: Expanding your time range (e.g., "last 2 years" instead of "last month")""").send()
            return

        # Detect time-series data
        try:
            is_forecastable, date_col, value_col = detect_time_series(results, min_data_points=20)
        except Exception as e:
            logger.warning(f"Time-series detection failed: {str(e)}")
            return

        if is_forecastable:
            # Store query context for forecast
            cl.user_session.set("last_query_sql", sql)
            cl.user_session.set("last_query_results", results)
            cl.user_session.set("last_query_data", original_data)
            cl.user_session.set("date_column", date_col)
            cl.user_session.set("value_column", value_col)
            cl.user_session.set("database_id", original_data.get('database_id', 'unknown'))

            # Create forecast button with proper payload
            actions = [
                cl.Action(
                    name="forecast_30",
                    payload={"horizon": 30},
                    label="🔮 Forecast Next 30 Periods"
                ),
                cl.Action(
                    name="forecast_7",
                    payload={"horizon": 7},
                    label="📊 Forecast Next 7 Periods"
                ),
                cl.Action(
                    name="forecast_90",
                    payload={"horizon": 90},
                    label="📈 Forecast Next 90 Periods"
                )
            ]

            forecast_msg = cl.Message(
                content=f"""### 🔮 **Time-Series Data Detected!**

**Date Column:** `{date_col}`
**Value Column:** `{value_col}`
**Data Points:** {len(results)}

Click below to generate a forecast using **AI.FORECAST** with Google's TimesFM 2.0 foundation model:""",
                actions=actions
            )
            await forecast_msg.send()

    async def handle_forecast(self, horizon: int):
        """Handle forecast request when button is clicked"""
        # Retrieve stored query context
        sql = cl.user_session.get("last_query_sql")
        date_col = cl.user_session.get("date_column")
        value_col = cl.user_session.get("value_column")
        database_id = cl.user_session.get("database_id")

        logger.info(f"Forecast request - SQL: {sql[:100] if sql else None}, date_col: {date_col}, value_col: {value_col}, db: {database_id}")

        if not all([sql, date_col, value_col]):
            await cl.Message(content="""❌ **Forecast Context Missing**

**Issue**: Required forecast data is not available.

**💡 Solution**: Please run a time-series query first that returns:
- A date/time column
- A numeric value column
- At least 20 data points

**Example**: "Show me monthly sales over the last year""").send()
            return

        # Create loading message with progress animation
        loading_msg = cl.Message(content=f"""
🔮 **Generating {horizon}-Period Forecast**

⌛ Processing with AI.FORECAST (TimesFM 2.0)...

[███░░░░░░░░░░░░░░░░░] 15%

**Steps:**
1. ✅ Expanding SQL for historical data
2. ⏳ Running time-series analysis
3. ⏳ Generating predictions
4. ⏳ Calculating confidence intervals
""")
        await loading_msg.send()

        try:
            # Make forecast API request
            forecast_request = {
                "sql": sql,
                "database_id": database_id,
                "date_column": date_col,
                "value_column": value_col,
                "horizon": horizon,
                "confidence_level": 0.95
            }
            logger.info(f"Sending forecast request: {json.dumps(forecast_request, indent=2)[:500]}")

            response = await httpx_client.post(
                FORECAST_URL,
                json=forecast_request,
                timeout=60.0  # Increased timeout for SQL expansion
            )
            response.raise_for_status()
            forecast_data = response.json()

            await loading_msg.remove()
            await self.display_forecast_results(forecast_data, value_col, horizon)

        except Exception as e:
            # Clean up loading message
            try:
                await loading_msg.remove()
            except:
                pass

            logger.error(f"Forecast error: {str(e)}")

            # Format user-friendly error message
            error_msg = self._format_forecast_error(str(e), horizon, date_col, value_col)
            await cl.Message(content=error_msg).send()

    def safe_get(self, data: Any, key: str, default: Any = None) -> Any:
        """Safely extract value from dictionary or nested structure"""
        if data is None:
            return default
        
        if isinstance(data, dict):
            value = data.get(key, default)
        else:
            value = default
        
        # Handle lists/arrays by taking first element
        if isinstance(value, (list, tuple)):
            if len(value) > 0:
                value = value[0]
            else:
                return default
        
        # Handle numpy types
        if hasattr(value, 'item'):
            value = value.item()
        elif hasattr(value, 'tolist'):
            value = value.tolist()
            if isinstance(value, list) and len(value) > 0:
                value = value[0]
        
        return value if value is not None else default

    def format_number(self, value: Any, decimals: int = 2, default: str = 'N/A') -> str:
        """Format a number safely with proper decimal places"""
        if value is None or value == 'N/A':
            return default
        
        try:
            # Handle various numeric types
            if isinstance(value, (list, tuple)) and len(value) > 0:
                value = value[0]
            
            if hasattr(value, 'item'):
                value = value.item()
            elif hasattr(value, 'tolist'):
                value = value.tolist()
                if isinstance(value, list) and len(value) > 0:
                    value = value[0]
            
            return f"{float(value):.{decimals}f}"
        except (TypeError, ValueError):
            return default

    async def display_forecast_results(self, forecast_data: Dict, value_col: str, horizon: int):
        """Display forecast results with enhanced formatting and AI insights"""
        logger.info("=== Starting display_forecast_results ===")
        logger.info(f"forecast_data keys: {forecast_data.keys()}")
        
        results = forecast_data.get('forecast_results', [])
        insights_raw = forecast_data.get('insights', {})
        metrics = forecast_data.get('model_metrics', {})
        features = forecast_data.get('bigquery_features_used', [])
        exec_time = forecast_data.get('execution_time_ms', 0)
        cost = forecast_data.get('cost_estimate', 0)

        # EXTENSIVE DEBUG LOGGING FOR INSIGHTS
        logger.info(f"insights_raw type: {type(insights_raw)}")
        logger.info(f"insights_raw content (first 500 chars): {str(insights_raw)[:500]}")
        
        # Humanize the column name for display
        display_col_name = value_col.replace('_', ' ').title()

        # Format forecast table FIRST
        forecast_table = "_No forecast results_"
        if results:
            df = pd.DataFrame(results)
            # Format dates if they're strings
            if 'forecast_date' in df.columns:
                df['forecast_date'] = pd.to_datetime(df['forecast_date']).dt.strftime('%Y-%m-%d')
            # Round numeric columns
            for col in ['predicted_value', 'lower_bound', 'upper_bound']:
                if col in df.columns:
                    df[col] = df[col].round(2)
            forecast_table = df.head(10).to_markdown(index=False)
            if len(results) > 10:
                forecast_table += f'\n\n_Showing first 10 of {len(results)} forecasted periods_'

        # Build the main forecast message
        forecast_content = f"""## ✅ AI.FORECAST Analysis for {display_col_name}

### 📈 Predicted Values ({horizon} Periods)

{forecast_table}

"""

        # CRITICAL: Process insights with the CORRECT structure
        if insights_raw:
            logger.info("Processing insights_raw...")

            try:
                # Handle the new structure from generate_forecast_insights_ai
                # Extract summary as interpretation
                interpretation = insights_raw.get('summary', '')

                # Extract trend information
                trend_str = insights_raw.get('trend', '')
                trend_emoji = '→'
                trend_direction = 'Stable'
                percent_change = 0

                # Parse trend string (e.g., "↗️ Upward (+5.2%)")
                if trend_str and isinstance(trend_str, str):
                    import re
                    # Extract emoji
                    if '↗' in trend_str or '📈' in trend_str:
                        trend_emoji = '↗️'
                        trend_direction = 'Upward'
                    elif '↘' in trend_str or '📉' in trend_str:
                        trend_emoji = '↘️'
                        trend_direction = 'Downward'
                    elif '→' in trend_str or '➡' in trend_str:
                        trend_emoji = '→'
                        trend_direction = 'Stable'

                    # Extract percentage
                    match = re.search(r'([+-]?\d+\.?\d*)%', trend_str)
                    if match:
                        try:
                            percent_change = float(match.group(1))
                        except ValueError:
                            percent_change = 0

                logger.info(f"Trend data - direction: {trend_direction}, emoji: {trend_emoji}, change: {percent_change}")

                # Extract recommendations (it's a list)
                recommendations_list = insights_raw.get('recommendations', [])
                if not isinstance(recommendations_list, list):
                    recommendations_list = []

                # Extract key insights as action items
                key_insights = insights_raw.get('key_insights', [])
                if not isinstance(key_insights, list):
                    key_insights = []
                action_item_1 = key_insights[0] if len(key_insights) > 0 else ''
                action_item_2 = key_insights[1] if len(key_insights) > 1 else ''

                # Extract risk assessment properly (it's a dict)
                risk_dict = insights_raw.get('risk_assessment', {})
                if isinstance(risk_dict, dict):
                    risk_level = risk_dict.get('level', 'Moderate')
                    volatility = risk_dict.get('volatility', '0%')
                    confidence_width = risk_dict.get('confidence_width', '0%')
                    risk_assessment = f"{risk_level} risk (volatility: {volatility}, confidence width: {confidence_width})"
                else:
                    risk_assessment = str(risk_dict) if risk_dict else ''

                # Extract statistics for business context
                stats = insights_raw.get('statistics', {})
                if isinstance(stats, dict):
                    avg_pred = stats.get('average_prediction', 0)
                    growth_rate = stats.get('growth_rate', 0)
                    business_impact = f"Expected average value: {avg_pred:.2f} with {growth_rate:+.1f}% growth"
                else:
                    business_impact = ''

                # Extract monitoring info for confidence assessment
                monitoring = insights_raw.get('monitoring', {})
                if isinstance(monitoring, dict):
                    thresholds = monitoring.get('thresholds', {})
                    if isinstance(thresholds, dict):
                        upper = thresholds.get('upper_alert', 0)
                        lower = thresholds.get('lower_alert', 0)
                        if upper and lower:
                            confidence_assessment = f"Monitor between {lower:.2f} (lower alert) and {upper:.2f} (upper alert)"
                        else:
                            confidence_assessment = ''
                    else:
                        confidence_assessment = ''
                else:
                    confidence_assessment = ''

                # Extract follow-up questions as opportunity highlight
                follow_up = insights_raw.get('follow_up_questions', [])
                if isinstance(follow_up, list) and len(follow_up) > 0:
                    opportunity_highlight = str(follow_up[0])
                else:
                    opportunity_highlight = ''

                forecast_quality_score = 0  # Not provided in current structure

            except Exception as e:
                logger.error(f"Error processing insights: {e}")
                # Set defaults on error
                interpretation = insights_raw.get('summary', '') if isinstance(insights_raw.get('summary'), str) else ''
                trend_direction = 'Unknown'
                trend_emoji = '→'
                percent_change = 0
                recommendations_list = []
                action_item_1 = ''
                action_item_2 = ''
                risk_assessment = ''
                business_impact = ''
                confidence_assessment = ''
                opportunity_highlight = ''
                forecast_quality_score = 0
            
            # DISPLAY KEY INSIGHT
            if interpretation and isinstance(interpretation, str) and interpretation.strip():
                logger.info(f"Adding Key Insight section with interpretation of length {len(interpretation)}")
                forecast_content += f"""### 💡 Key Insight

{interpretation.strip()}

"""
            else:
                logger.warning("No interpretation found or it was empty!")

            # DISPLAY TREND
            if trend_direction and isinstance(trend_direction, str) and trend_direction.strip():
                logger.info(f"Adding Trend section: {trend_direction}")
                try:
                    percent_val = float(percent_change) if percent_change else 0
                    sign = '+' if percent_val > 0 else ''
                    percent_str = f"{sign}{percent_val:.1f}%"
                except:
                    percent_str = "0.0%"
                
                # Use provided emoji or create one
                if not trend_emoji or trend_emoji.strip() == '':
                    if percent_val > 0:
                        trend_emoji = '↗️'
                    elif percent_val < 0:
                        trend_emoji = '↘️'
                    else:
                        trend_emoji = '→'
                
                # Use regular text size for trend (not heading)
                forecast_content += f"""{trend_emoji} **Trend: {trend_direction.strip().upper()} {percent_str}**

"""

            # DISPLAY RECOMMENDED ACTIONS
            # First check action items from key insights
            if action_item_1 and isinstance(action_item_1, str) and action_item_1.strip():
                logger.info("Adding Recommended Actions from key insights")
                forecast_content += "### 🎯 Recommended Actions\n\n"
                forecast_content += f"1. {action_item_1.strip()}\n"
                if action_item_2 and isinstance(action_item_2, str) and action_item_2.strip():
                    forecast_content += f"2. {action_item_2.strip()}\n"
                forecast_content += "\n"
            # Otherwise use recommendations list
            elif recommendations_list and isinstance(recommendations_list, list) and len(recommendations_list) > 0:
                logger.info(f"Adding Recommended Actions from list ({len(recommendations_list)} items)")
                forecast_content += "### 🎯 Recommended Actions\n\n"
                for i, rec in enumerate(recommendations_list[:3], 1):
                    if rec and isinstance(rec, str) and rec.strip():
                        forecast_content += f"{i}. {rec.strip()}\n"
                forecast_content += "\n"

            # DISPLAY BUSINESS CONTEXT - filter out generic/default values
            context_parts = []
            
            # Only show business impact if it's not the generic default
            if (business_impact and isinstance(business_impact, str) and business_impact.strip() and 
                business_impact not in ['Monitor trend for business planning', 'Standard monitoring applies']):
                context_parts.append(f"**Impact:** {business_impact.strip()}")
                
            # Only show risk if it's not the generic default
            if (risk_assessment and isinstance(risk_assessment, str) and risk_assessment.strip() and 
                risk_assessment not in ['Standard forecast uncertainty applies', 'Normal forecast uncertainty']):
                context_parts.append(f"**Risk:** {risk_assessment.strip()}")
                
            # Only show opportunity if it's not the generic default
            if (opportunity_highlight and isinstance(opportunity_highlight, str) and opportunity_highlight.strip() and 
                opportunity_highlight not in ['Leverage trend for strategic planning', 'Use insights for planning']):
                context_parts.append(f"**Opportunity:** {opportunity_highlight.strip()}")
            
            if context_parts:
                logger.info(f"Adding Business Context with {len(context_parts)} parts")
                forecast_content += "### 💼 Business Context\n\n"
                forecast_content += "\n\n".join(context_parts)
                forecast_content += "\n\n"

            # DISPLAY CONFIDENCE ASSESSMENT - only if not generic
            if (confidence_assessment and isinstance(confidence_assessment, str) and confidence_assessment.strip() and 
                confidence_assessment not in ['Moderate confidence', 'Standard confidence level']):
                logger.info("Adding Confidence Assessment")
                forecast_content += f"""### 🔍 Confidence Assessment

{confidence_assessment.strip()}

"""

            # DISPLAY QUALITY SCORE
            if forecast_quality_score and forecast_quality_score > 0:
                stars = '⭐' * max(1, min(5, int(forecast_quality_score * 5)))
                forecast_content += f"**Forecast Quality:** {stars} ({forecast_quality_score:.2f})\n\n"
                
        else:
            logger.warning("No insights dictionary available after processing!")

        # Add horizontal rule
        forecast_content += "---\n\n"

        # Add BigQuery features section
        forecast_content += "### 🤖 BigQuery AI Features Used\n\n"
        for feature in features:
            if 'AI.FORECAST' in feature:
                forecast_content += "🔮 **AI.FORECAST (TimesFM 2.0)** - Foundation model predictions\n"
            elif 'TimesFM' in feature:
                forecast_content += "🧠 **TimesFM 2.0** - Google's pre-trained time-series model\n"
            elif 'AI-Generated' in feature:
                forecast_content += "💡 **AI.GENERATE** - AI-powered business insights\n"
            elif 'AI.GENERATE' in feature:
                forecast_content += "✨ **AI.GENERATE** - AI-powered business insights\n"
            else:
                forecast_content += f"✅ **{feature}**\n"

        # Add execution summary
        forecast_content += f"""
### 📊 Execution Summary
**Time:** {exec_time}ms | **Model:** TimesFM 2.0 | **Horizon:** {horizon} periods | **Cost:** ${cost:.6f}

---
_Powered by **BigQuery AI.FORECAST** with TimesFM 2.0 foundation model_
"""

        logger.info(f"Final forecast_content length: {len(forecast_content)} chars")
        logger.info("=== Ending display_forecast_results ===")

        # Send the formatted message
        await cl.Message(content=forecast_content).send()


# Global handler instance
handler = StreamingQueryHandler()


@log_performance
@cl.on_stop
async def cleanup():
    """Clean up when the application/chat stops"""
    logger.info('Application/chat stopping, performing cleanup')
    await handler.cleanup_session()
    clear_request_context()
    logger.debug('Cleanup completed')


@log_performance
@cl.on_chat_start
async def start():
    """Initialize the chat session with enhanced welcome experience"""
    session_id = str(uuid.uuid4())
    cl.user_session.set('session_id', session_id)
    cl.user_session.set('selected_database', None)
    cl.user_session.set('conversation_id', str(uuid.uuid4())[:8])
    cl.user_session.set('features_used', set())

    # Set request context for logging
    set_request_context(user_session=session_id)
    logger.info(
        'Chat session started',
        extra={'extra_fields': {'session_id': session_id, 'conversation_id': cl.user_session.get('conversation_id')}},
    )

    # Get available databases
    try:
        log_api_request(logger, 'GET', f'{API_BASE}/api/databases')
        response = await httpx_client.get(f'{API_BASE}/api/databases')
        response.raise_for_status()
        databases = response.json().get('databases', [])
        cl.user_session.set('databases', databases)
        log_api_response(logger, response.status_code, 0)
    except Exception as e:
        databases = []
        logger.error(
            f'Failed to fetch databases: {str(e)}', extra={'extra_fields': {'endpoint': f'{API_BASE}/api/databases'}}
        )

    # Enhanced welcome message with hackathon branding
    welcome_msg = """# **BQ FLOW**
## _Where Natural Language Meets Data Intelligence_

---

### 🎯 **Competing in: AI Architect + Semantic Detective Tracks**
#### -- Shivam Bhardwaj | [Linkedin](https://www.linkedin.com/in/shivamordanny/)

---

## 📋 **Prerequisites: Data Onboarding & Schema Embedding**

> **Important**: Datasets must be onboarded before querying using **BQ Flow Data Onboarding & AI Training**
>
> 🔧 **Run onboarding**: `cd bq_flow_onboarding && ./run.sh` (port 8501)
>
> This creates semantic embeddings and vector indexes that enable natural language understanding

---

## 🎮 **How to Use**
1. **Select a pre-embedded dataset** (e.g., type `thelook_ecommerce`)
2. **Ask questions in plain English**: "What is the average order value by month?"
3. **Watch the 13-stage pipeline** transform your query in real-time
4. **Get AI-powered insights** and business intelligence
5. **📈 Time-series detection**: If your results contain dates and values, you'll see forecast buttons

### 🤖 **BigQuery AI Features**
- **ML.GENERATE_EMBEDDING** - Text to semantic vectors (768-dim)
- **VECTOR_SEARCH** - Find relevant columns with IVF indexing
- **AI.GENERATE** - Natural language to SQL (Gemini 2.5 Flash)
- **AI.GENERATE_TABLE** - Extract structured insights
- **AI.FORECAST** - Time-series predictions (TimesFM 2.0)
- **ML.DISTANCE** - Similarity calculations for fallback

---

## 🗄️ **Pre-embedded Datasets**
_These datasets have been onboarded and indexed for semantic search:_"""

    # Enhanced database display
    db_info = {
        'google_trends': ('📈', 'Search trends & popularity data'),
        'thelook_ecommerce': ('🛍️', 'E-commerce transactions & customers'),
        'stackoverflow': ('💻', 'Developer Q&A & community data'),
        'hackernews': ('📰', 'Tech news & discussions')
    }

    for db in databases:
        db_id = db['database_id']
        emoji, desc = db_info.get(db_id, ('📊', db['description']))
        welcome_msg += f"\n- {emoji} **`{db_id}`** - {desc}"

    welcome_msg += """

---

**Ready to start!** Type a dataset name above to begin querying.

---"""

    await cl.Message(content=welcome_msg).send()


@log_performance
@cl.on_message
async def main(message: cl.Message):
    """Handle user messages"""
    user_input = message.content.strip()
    selected_database = cl.user_session.get('selected_database')
    session_id = cl.user_session.get('session_id')

    logger.info(
        'Received user message',
        extra={'extra_fields': {'input_length': len(user_input), 'database_id': selected_database}},
    )

    # Check if user is selecting a database
    if user_input.lower() in ['google_trends', 'thelook_ecommerce', 'stackoverflow', 'hackernews']:
        await select_database(user_input.lower())
        return

    # Check if database is selected
    if not selected_database:
        await cl.Message(
            content='⚠️ Please select a pre-embedded dataset first by typing: **google_trends**, **thelook_ecommerce**, **stackoverflow** or **hackernews**\n\n💡 **Note**: Datasets must be onboarded first using the BQ Flow Data Onboarding tool (port 8501)',
        ).send()
        logger.warning('No database selected for query', extra={'extra_fields': {'user_input': user_input[:100]}})
        return

    # Process query with streaming
    await handler.process_query(user_input, selected_database, session_id)


@log_performance
async def select_database(database_id: str):
    """Handle database selection"""
    databases = cl.user_session.get('databases', [])

    # Find selected database
    selected_db = next((db for db in databases if db['database_id'] == database_id), None)

    if not selected_db:
        # Create a default entry if not found
        selected_db = {
            'database_id': database_id,
            'display_name': database_id.replace('_', ' ').title(),
            'description': f'{database_id} database',
        }

    # Store in session
    cl.user_session.set('selected_database', database_id)
    set_request_context(database_id=database_id)
    logger.info(
        'Database selected',
        extra={'extra_fields': {'database_id': database_id, 'display_name': selected_db['display_name']}},
    )

    # Example queries for each database
    examples = {
        'thelook_ecommerce': [
            'What is the average order value by month?',
            'List top 5 customers by total spend',
            'Which products have the highest profit margin?',
            'Find customers who spent over $500',
        ],
        'stackoverflow': [
            'What are the highest scored Python questions?',
            'Show posts with most views this year',
            'Find unanswered questions about machine learning',
            'Top contributors by reputation',
        ],
    }

    example_queries = examples.get(database_id, ['Ask any question about the data!'])
    examples_formatted = '\n'.join(f'• {q}' for q in example_queries)

    await cl.Message(
        content=f"""## ✅ Database Selected: **{selected_db['display_name']}**

**Description**: {selected_db['description']}

### 💡 **Power Queries for {selected_db['display_name']}**
{examples_formatted}

---

### 🏆 **BigQuery AI Features Status**

| Feature | Status | Description |
|---------|--------|-------------|
| 🧬 ML.GENERATE_EMBEDDING | 🟢 Ready | Semantic vectorization |
| 🔎 VECTOR_SEARCH | 🟢 Ready | IVF-indexed search |
| ✨ AI.GENERATE | 🟢 Ready | Gemini 2.5 Flash SQL |
| 📊 AI.GENERATE_TABLE | 🟢 Ready | Structured insights |
| 🔮 AI.FORECAST | 🟢 Ready | TimesFM predictions |
| 📏 ML.DISTANCE | 🟢 Ready | Fallback matching |

**🌐 Real-time Streaming Active**

---
_All systems operational. Ready for natural language queries!_"""
    ).send()


@cl.action_callback("forecast_30")
async def on_forecast_30(action: cl.Action):
    """Handle 30-period forecast request"""
    horizon = action.payload.get("horizon", 30) if action.payload else 30
    await handler.handle_forecast(horizon)


@cl.action_callback("forecast_7")
async def on_forecast_7(action: cl.Action):
    """Handle 7-period forecast request"""
    horizon = action.payload.get("horizon", 7) if action.payload else 7
    await handler.handle_forecast(horizon)


@cl.action_callback("forecast_90")
async def on_forecast_90(action: cl.Action):
    """Handle 90-period forecast request"""
    horizon = action.payload.get("horizon", 90) if action.payload else 90
    await handler.handle_forecast(horizon)