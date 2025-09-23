# BQ Flow - Natural Language to Data Intelligence 🚀

## Welcome to BQ Flow!

BQ Flow transforms your natural language questions into powerful data insights using BigQuery's cutting-edge AI capabilities.

### 📋 Prerequisites

Before using BQ Flow, ensure your datasets are properly onboarded:

1. **Run the Data Onboarding Tool**
   ```bash
   cd bq_flow_onboarding
   ./run.sh  # Opens at http://localhost:8501
   ```

2. **Why Onboarding Matters**
   - Creates semantic embeddings for each column
   - Builds vector indexes for fast search
   - Enables natural language understanding
   - Significantly reduces costs through intelligent column selection

### 🎯 Quick Start

1. **Select a Pre-embedded Dataset**
   - Type: `thelook_ecommerce`, `google_trends`, `stackoverflow`, or `hackernews`

2. **Ask Questions in Natural Language**
   - "What are our monthly sales trends?"
   - "Show me top customers by lifetime value"
   - "Find products with highest profit margins"

3. **Watch the 13-Stage Pipeline**
   - Real-time progress visualization
   - See each BigQuery AI feature in action
   - Understand how your query transforms

4. **Get AI-Powered Insights**
   - Structured business intelligence
   - Trend analysis and recommendations
   - Confidence scores and risk assessments

5. **Forecast Future Trends**
   - Automatic time-series detection
   - One-click predictions with AI.FORECAST
   - 7, 30, or 90-period forecasts

### 🤖 BigQuery AI Features

| Feature | Description | Use Case |
|---------|------------|----------|
| **ML.GENERATE_EMBEDDING** | Text → 768-dim vectors | Semantic understanding |
| **VECTOR_SEARCH** | IVF-indexed search | Find relevant columns |
| **AI.GENERATE** | Gemini 2.5 Flash | Natural language → SQL |
| **AI.GENERATE_TABLE** | Structured extraction | Business insights |
| **AI.FORECAST** | TimesFM 2.0 | Predictive analytics |
| **ML.DISTANCE** | Cosine similarity | Fallback matching |

### 💡 Pro Tips

- **Time-Series Detection**: If your results contain dates and values, forecast buttons will automatically appear
- **Cost Optimization**: Our semantic caching significantly reduces costs
- **Performance**: Fast query processing with high first-attempt success rate

### 🏆 BigQuery AI Hackathon 2025

This project showcases ALL 6 BigQuery AI functions in production:
- **AI Architect Track**: Workflow automation with generative AI
- **Semantic Detective Track**: Deep data understanding through embeddings

---

**Need Help?** Check our [Documentation](https://docs.bqflow.ai) or [GitHub](https://github.com/bqflow)