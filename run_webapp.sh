#!/usr/bin/env bash
# Run the Flask web app, creating a venv and installing Flask if needed.
# Usage: ./run_webapp.sh   (from stock_analysis directory)

set -e
cd "$(dirname "$0")"

VENV=".venv"
if [[ ! -d "$VENV" ]]; then
  echo "Creating virtual environment in $VENV ..."
  python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"

if ! python -c "import flask" 2>/dev/null; then
  echo "Installing Flask ..."
  pip install -q flask
fi
if ! python -c "import pandas" 2>/dev/null; then
  echo "Installing pandas (required for Compare) ..."
  pip install -q pandas
fi
if ! python -c "import yfinance" 2>/dev/null; then
  echo "Installing yfinance (required for Calendar) ..."
  pip install -q yfinance
fi

# Use 5001 to avoid conflict with macOS AirPlay Receiver (which uses 5000)
PORT=5001
if command -v lsof >/dev/null 2>&1; then
  for try_port in 5001 5002 5003; do
    PIDS=$(lsof -ti :"$try_port" 2>/dev/null || true)
    if [[ -n "$PIDS" ]]; then
      echo "Port $try_port in use. Stopping existing process(es): $PIDS"
      echo "$PIDS" | xargs kill -9 2>/dev/null || true
      sleep 2
    fi
    if ! lsof -ti :"$try_port" >/dev/null 2>&1; then
      PORT=$try_port
      break
    fi
    echo "Port $try_port still in use, trying next..."
  done
fi

export PORT
echo "Starting web app at http://127.0.0.1:$PORT"
echo "  (For ngrok/localtunnel: use port $PORT, e.g. ngrok http 127.0.0.1:$PORT)"
echo "  (If not accessible: keep this terminal open; restart with ./run_webapp.sh if the app stops.)"
python app.py
