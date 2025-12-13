#!/bin/bash
# Start both frontend and backend for development

echo "Starting Neural Aquarium Development Environment"
echo "================================================"
echo ""

cd "$(dirname "$0")"

trap 'kill $(jobs -p)' EXIT

echo "Starting Backend API..."
(cd backend && bash start.sh) &
BACKEND_PID=$!

sleep 3

echo ""
echo "Starting Frontend..."
npm start &
FRONTEND_PID=$!

echo ""
echo "Development servers started:"
echo "  Frontend: http://localhost:3000"
echo "  Backend:  http://localhost:5000"
echo ""
echo "Press Ctrl+C to stop both servers"

wait
