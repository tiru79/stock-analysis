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
  echo "Installing pandas (required for Compare symbols) ..."
  pip install -q pandas
fi
if ! python -c "import yfinance" 2>/dev/null; then
  echo "Installing yfinance (required for Calendar) ..."
  pip install -q yfinance
fi

PORT=5000
if command -v lsof >/dev/null 2>&1; then
  PIDS=$(lsof -ti :"$PORT" 2>/dev/null || true)
  if [[ -n "$PIDS" ]]; then
    echo "Port $PORT in use. Stopping existing process(es): $PIDS"
    echo "$PIDS" | xargs kill -9 2>/dev/null || true
    sleep 1
  fi
fi

echo "Starting web app at http://127.0.0.1:$PORT"
python app.py
