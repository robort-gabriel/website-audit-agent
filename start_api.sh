#!/bin/bash

# Website SEO Audit API Startup Script

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Install dependencies if needed
if [ ! -f ".venv/.deps_installed" ]; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
    touch .venv/.deps_installed
fi

# Load environment variables if .env exists
if [ -f ".env" ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Start the API
echo "Starting Website SEO Audit API..."
python main.py

