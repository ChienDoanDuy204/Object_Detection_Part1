#!/bin/bash
# ==============================================================================
# OverFeat AI Studio - Startup Script
# Starts FastAPI backend (port 8000) and Next.js frontend (port 3000)
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_EXEC="/home/chiendoan/miniconda3/envs/study_env/bin/python"
NODE_BIN="/home/chiendoan/.local/node/bin"
export PATH="$NODE_BIN:$PATH"

echo "=== Starting OverFeat AI Studio ==="

# 1. Clean up any existing processes on ports 8000 or 3000 to prevent EADDRINUSE
cleanup_port() {
    local port=$1
    local pids=$(lsof -ti :$port 2>/dev/null)
    if [ -n "$pids" ]; then
        echo "Port $port is currently in use by PID(s): $pids. Releasing port..."
        kill -9 $pids 2>/dev/null || true
        sleep 0.5
    fi
}

cleanup_port 8000
cleanup_port 3000

# 2. Start FastAPI backend
echo "[1/2] Starting FastAPI backend on http://localhost:8000..."
$PYTHON_EXEC -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Wait for backend to be ready
echo "Waiting for backend to initialize..."
for i in {1..30}; do
    if curl -s http://localhost:8000/ > /dev/null 2>&1; then
        echo "Backend is ready!"
        break
    fi
    sleep 0.5
done

# 3. Start Next.js frontend
echo "[2/2] Starting Next.js frontend on http://localhost:3000..."
cd "$SCRIPT_DIR/frontend"
npm run dev -- -p 3000 &
FRONTEND_PID=$!

echo ""
echo "============================================="
echo "  🚀 OverFeat AI Studio is Running!"
echo "  👉 Frontend: http://localhost:3000"
echo "  👉 Backend:  http://localhost:8000"
echo "  👉 API Docs: http://localhost:8000/docs"
echo "============================================="
echo "Press CTRL+C at any time to stop both servers."

cleanup() {
    echo ""
    echo "Stopping OverFeat AI Studio..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null || true
    cleanup_port 8000
    cleanup_port 3000
    echo "All servers stopped."
    exit 0
}

trap cleanup INT TERM
wait
