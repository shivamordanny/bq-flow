#!/bin/bash
# BQ Flow Startup Script
# Launches both backend API and Chainlit frontend

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_message() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

# Cleanup function to kill processes on exit
cleanup() {
    print_warning "Shutting down services..."

    # Kill backend process if running
    if [ ! -z "$BACKEND_PID" ] && kill -0 $BACKEND_PID 2>/dev/null; then
        print_message "Stopping backend server (PID: $BACKEND_PID)..."
        kill $BACKEND_PID
    fi

    # Kill frontend process if running
    if [ ! -z "$FRONTEND_PID" ] && kill -0 $FRONTEND_PID 2>/dev/null; then
        print_message "Stopping frontend server (PID: $FRONTEND_PID)..."
        kill $FRONTEND_PID
    fi

    # Wait for processes to terminate
    wait $BACKEND_PID 2>/dev/null
    wait $FRONTEND_PID 2>/dev/null

    print_message "All services stopped."
    exit 0
}

# Set up trap to call cleanup on script exit
trap cleanup EXIT INT TERM

# Check if poetry is installed
if ! command -v poetry &> /dev/null; then
    print_error "Poetry is not installed. Please install poetry first."
    exit 1
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    print_warning ".env file not found. Creating from .env.example..."
    if [ -f ".env.example" ]; then
        cp .env.example .env
        print_message "Created .env file. Please update it with your configuration."
    else
        print_error ".env.example not found. Cannot create .env file."
        exit 1
    fi
fi

# Load environment variables
if [ -f ".env" ]; then
    set +e  # Temporarily disable exit on error for env loading
    export $(cat .env | grep -v '^#' | grep -v '^$' | xargs) 2>/dev/null || true
    set -e  # Re-enable exit on error
    print_message "Environment variables loaded from .env"
fi

# Check Google credentials
if [ -z "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
    print_warning "GOOGLE_APPLICATION_CREDENTIALS not set."
    print_message "Will use Application Default Credentials (gcloud auth)."
    # Check if user is authenticated with gcloud
    if gcloud auth list --filter=status:ACTIVE --format="value(account)" &> /dev/null; then
        ACTIVE_ACCOUNT=$(gcloud auth list --filter=status:ACTIVE --format="value(account)")
        print_message "Using gcloud auth with account: $ACTIVE_ACCOUNT"
    else
        print_warning "No active gcloud authentication found. BigQuery operations may fail."
    fi
else
    if [ ! -f "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
        print_error "Service account file not found at: $GOOGLE_APPLICATION_CREDENTIALS"
        exit 1
    fi
    print_message "Using service account credentials from: $GOOGLE_APPLICATION_CREDENTIALS"
fi

print_message "==================================="
print_message "Starting BQ Flow Services"
print_message "==================================="

# Check if ports are already in use
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0  # Port is in use
    else
        return 1  # Port is free
    fi
}

# Kill existing processes on ports if needed
if check_port 8000; then
    print_warning "Port 8000 is already in use."
    read -p "Kill existing process on port 8000? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_message "Killing existing process on port 8000..."
        lsof -ti:8000 | xargs kill -9 2>/dev/null || true
        sleep 2
    else
        print_error "Cannot start backend server. Port 8000 is in use."
        exit 1
    fi
fi

if check_port 3000; then
    print_warning "Port 3000 is already in use."
    read -p "Kill existing process on port 3000? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_message "Killing existing process on port 3000..."
        lsof -ti:3000 | xargs kill -9 2>/dev/null || true
        sleep 2
    else
        print_error "Cannot start frontend server. Port 3000 is in use."
        exit 1
    fi
fi

# Install dependencies if needed
print_message "Checking dependencies..."
if ! poetry install --no-interaction --quiet 2>/dev/null; then
    print_warning "Some dependencies may not be fully installed, but continuing..."
fi
print_message "Dependencies check complete."

# Create logs directory if it doesn't exist
mkdir -p logs

# Start backend server
print_message "Starting backend API server..."
poetry run python main.py > logs/backend.log 2>&1 &
BACKEND_PID=$!
print_message "Backend server started (PID: $BACKEND_PID)"
print_message "API available at: http://localhost:8000"
print_message "API docs at: http://localhost:8000/docs"

# Wait for backend to be ready
print_message "Waiting for backend to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:8000/docs > /dev/null 2>&1; then
        print_message "Backend is ready!"
        break
    fi
    if [ $i -eq 30 ]; then
        print_error "Backend failed to start after 30 seconds"
        exit 1
    fi
    sleep 1
done

# Start frontend server
print_message "Starting Chainlit UI..."
cd src/ui
poetry run chainlit run chainlit_app.py --port 3000 --host 0.0.0.0 > ../../logs/frontend.log 2>&1 &
FRONTEND_PID=$!
cd ../..
print_message "Frontend server started (PID: $FRONTEND_PID)"
print_message "UI available at: http://localhost:3000"

print_message "==================================="
print_message "All services are running!"
print_message "==================================="
print_message ""
print_message "Services:"
print_message "  • Backend API: http://localhost:8000"
print_message "  • API Docs: http://localhost:8000/docs"
print_message "  • WebSocket: ws://localhost:8000/ws/query/stream"
print_message "  • Chainlit UI: http://localhost:3000"
print_message ""
print_message "Logs:"
print_message "  • Backend: logs/backend.log"
print_message "  • Frontend: logs/frontend.log"
print_message ""
print_message "Press Ctrl+C to stop all services"

# Monitor processes
while true; do
    # Check if backend is still running
    if ! kill -0 $BACKEND_PID 2>/dev/null; then
        print_error "Backend server crashed! Check logs/backend.log for details"
        exit 1
    fi

    # Check if frontend is still running
    if ! kill -0 $FRONTEND_PID 2>/dev/null; then
        print_error "Frontend server crashed! Check logs/frontend.log for details"
        exit 1
    fi

    sleep 5
done