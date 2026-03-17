#!/usr/bin/env bash
set -e

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Generating model files..."
python generate_models.py

echo "Build complete!"
