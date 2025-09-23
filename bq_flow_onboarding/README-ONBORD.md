# BQ Flow: Data Onboarding & AI Training

> The intelligent data preparation system that transforms your BigQuery datasets into AI-ready knowledge bases.

## 🎯 Purpose

This onboarding tool is the **crucial first step** in enabling natural language queries with BQ Flow. Before the AI can understand and query your data, it needs to:
- **Map** your database structure
- **Understand** the semantic meaning of columns
- **Create** searchable vector representations
- **Optimize** for performance and cost

## 🤔 Why This Matters

### Without Proper Onboarding:
- ❌ The AI won't understand your column meanings
- ❌ Natural language queries will fail to find relevant tables
- ❌ Performance will be slow without vector indexes
- ❌ Costs will be higher processing unnecessary columns
- ❌ Results will be inaccurate or incomplete

### With BQ Flow Onboarding:
- ✅ AI understands the semantic context of every column
- ✅ Natural language instantly maps to the right tables
- ✅ Vector indexes enable sub-second semantic search
- ✅ Intelligent column selection significantly reduces costs
- ✅ Accurate SQL generation from the first query

## 📋 The 4-Step Process

### Step 1: Discovery 🔍
**"Scans the database schema to find all available data."**
- Connects to your BigQuery dataset
- Discovers all tables and columns
- Maps data types and relationships
- Creates initial metadata catalog

### Step 2: Profiling 📊
**"Analyzes data samples to understand content and statistics."**
- Samples data from each table
- Calculates statistics (nulls, cardinality, distributions)
- Extracts example values for context
- Identifies patterns and data quality issues

### Step 3: AI Selection 🤖
**"Uses Gemini to intelligently select the most analytically valuable columns."**
- Leverages AI.GENERATE with Gemini 2.5 Flash
- Identifies business-relevant columns
- Excludes system fields and IDs
- Substantially reduces embedding costs
- Focuses on columns users actually query

### Step 4: Embedding Generation 🧬
**"Creates vector embeddings using ML.GENERATE_EMBEDDING so the data can be understood by the AI."**
- Generates 768-dimensional vectors with text-embedding-005
- Creates semantic context for each column
- Builds searchable vector representations
- Enables VECTOR_SEARCH with IVF indexing
- Prepares data for natural language understanding

## 🚀 Quick Start

### Prerequisites
- Google Cloud Project with BigQuery API enabled
- Service account with BigQuery Admin permissions
- Python 3.12+ with Poetry installed

### Installation & Running

```bash
# Navigate to the onboarding tool
cd bq_flow_onboarding

# Install dependencies (if not already done)
poetry install

# Run the onboarding interface
poetry run streamlit run app.py --server.port 8501

# Or use the convenient script
./run.sh
```

The interface will open at http://localhost:8501

### Configuration

1. **Select Database**: Choose from pre-configured BigQuery datasets
2. **Configure Settings**: Adjust profiling and embedding parameters
3. **Run Workflow**: Execute the 4-step process
4. **Monitor Progress**: Track embedding generation in real-time

## 📊 What Gets Created

### Metadata Tables
- `enriched_metadata` - Column descriptions with semantic context
- `database_registry` - Catalog of onboarded databases
- `embedding_jobs` - Processing history and status

### Vector Indexes
- IVF indexes for VECTOR_SEARCH (when >5000 embeddings)
- Optimized for semantic similarity search
- COSINE distance for best accuracy

### AI Models
- `embedding_model_005` - Text embedding model (768 dims)
- Optional: `embedding_model_gemini` - Premium model (3072 dims)

## 💡 Best Practices

### Column Selection Strategy
- **Include**: Status fields, categories, amounts, dates, names
- **Exclude**: IDs, UUIDs, timestamps, hashes, tokens
- **Let AI Decide**: Use Step 3 for intelligent selection

### Performance Optimization
- Profile samples of 1000 rows (adjustable)
- Batch embeddings in groups of 100
- Use AI selection to reduce unnecessary embeddings
- Create indexes after reaching 5000 embeddings

### Cost Management
- AI selection significantly reduces embedding costs
- Track costs in real-time during generation
- Use caching for repeated embeddings
- Monitor daily budget limits

## 🔧 Advanced Configuration

### Custom Datasets
Add your datasets to `databases.yaml`:
```yaml
your_dataset:
  project_id: your-project
  dataset_id: your-dataset
  display_name: "Your Dataset"
  description: "Description of your data"
```

### Model Selection
Choose embedding models in `config.yaml`:
```yaml
embedding_models:
  default: text-embedding-005  # Fast, 768 dimensions
  # or
  default: gemini-embedding-001  # Premium, 3072 dimensions
```

## 📈 Metrics & Monitoring

### Processing Statistics
- Tables discovered
- Columns profiled
- Columns selected by AI
- Embeddings generated
- Vector index status

### Performance Tracking
- Embedding generation speed
- API call latency
- Cost per operation
- Cache hit rates

## 🤝 Integration with Main App

Once onboarding is complete:

1. **Vector Search Ready**: Natural language queries can find relevant columns
2. **Semantic Understanding**: AI comprehends your data's meaning
3. **Optimized Performance**: Indexes enable fast searches
4. **Cost Efficient**: Only valuable columns are processed

The main BQ Flow application (port 3000) can now:
- Convert natural language to SQL
- Find relevant tables semantically
- Generate accurate queries
- Provide business insights

## 🐛 Troubleshooting

### Common Issues

**"Not enough rows for vector index"**
- Need minimum 5000 embeddings across all databases
- Use cumulative mode to combine multiple datasets

**"AI selection failed"**
- Check Gemini API quotas
- Verify connection to Vertex AI
- Review error logs in `logs/` directory

**"Embeddings generation slow"**
- Reduce batch size in configuration
- Check BigQuery slot availability
- Monitor API rate limits

## 📚 Technical Details

### BigQuery AI Functions Used
- `ML.GENERATE_EMBEDDING` - Creates vector representations
- `AI.GENERATE` - Powers intelligent column selection
- `VECTOR_SEARCH` - Enables semantic search (after onboarding)
- `CREATE VECTOR INDEX` - Builds IVF indexes

### Architecture
```
Data Source (BigQuery)
    ↓
Discovery & Profiling
    ↓
AI Column Selection (Gemini)
    ↓
Embedding Generation (text-embedding-005)
    ↓
Vector Index Creation
    ↓
Ready for SQL Generation
```

## 📄 License

Part of the BQ Flow project for BigQuery AI Hackathon 2025.

---

<div align="center">

**Prepare your data for AI. Enable natural language queries. Transform your analytics.**

[Main Application](http://localhost:3000) | [Documentation](../README.md) | [Support](mailto:team@bqflow.ai)

</div>