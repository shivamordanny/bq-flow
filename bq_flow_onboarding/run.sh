#!/bin/bash

# BQ Flow Onboarding - Streamlit App Runner

echo "🚀 Starting BQ Flow Onboarding..."

# Check if we're in the right directory
if [ ! -f "app.py" ]; then
    echo "Error: app.py not found. Please run from bq_flow_onboarding directory"
    exit 1
fi

# Install dependencies
echo "📦 Installing dependencies...
poetry install"

# Run the Streamlit app
echo "🔍 Launching Streamlit application..."
poetry run streamlit run app.py --server.port 8501 --server.address localhost

echo "✅ Application stopped"