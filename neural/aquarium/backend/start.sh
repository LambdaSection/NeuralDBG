#!/bin/bash
# Start Neural Aquarium Backend API

echo "Starting Neural Aquarium Backend API..."
echo "API will be available at http://localhost:5000"
echo ""

cd "$(dirname "$0")"

if [ ! -d "../../../.venv" ] && [ ! -d "../../../venv" ]; then
    echo "Warning: Virtual environment not found. Installing dependencies globally..."
    pip install -r requirements.txt
else
    if [ -d "../../../.venv" ]; then
        source ../../../.venv/bin/activate
    else
        source ../../../venv/bin/activate
    fi
    echo "Virtual environment activated"
fi

export FLASK_APP=api.py
export FLASK_ENV=development

python api.py
