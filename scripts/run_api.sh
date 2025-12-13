#!/bin/bash
# Script to run Neural API server with all services

set -e

echo "🚀 Starting Neural API Server..."

# Check if Redis is running
if ! redis-cli ping > /dev/null 2>&1; then
    echo "❌ Redis is not running!"
    echo "Please start Redis first:"
    echo "  docker run -d -p 6379:6379 redis:7-alpine"
    echo "  OR: redis-server"
    exit 1
fi

echo "✅ Redis is running"

# Start Celery worker in background
echo "🔧 Starting Celery worker..."
celery -A neural.api.celery_app worker --loglevel=info --concurrency=4 &
CELERY_PID=$!

# Wait for Celery to start
sleep 3

# Start API server
echo "🌐 Starting API server..."
uvicorn neural.api.main:app --host 0.0.0.0 --port 8000 --reload &
API_PID=$!

echo ""
echo "=" * 80
echo "✅ Neural API Server Started Successfully!"
echo "=" * 80
echo "API Server: http://localhost:8000"
echo "API Docs: http://localhost:8000/docs"
echo "Health Check: http://localhost:8000/health"
echo ""
echo "To stop the server, press Ctrl+C"
echo "=" * 80
echo ""

# Handle shutdown
trap "echo 'Shutting down...'; kill $CELERY_PID $API_PID; exit" INT TERM

# Wait for processes
wait
