#!/bin/bash
set -e
echo "Starting build.sh..."
echo "Installing dependencies..."
pip install -r requirements.txt
echo "Starting Streamlit app..."
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
