#!/bin/bash
set -e

echo "=== GNR 638 Assignment 2 - Environment Setup ==="

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

echo "=== Environment setup complete ==="
echo "Activate with: source venv/bin/activate"
