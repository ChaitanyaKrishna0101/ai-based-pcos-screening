#!/usr/bin/env bash
# One-command local setup + run.
set -e

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate

pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

if [ ! -f "app/model/pcos_model.pkl" ]; then
  echo "Training model (first run only)..."
  python app/model/train.py
fi

if [ ! -f ".env" ] && [ -f ".env.example" ]; then
  cp .env.example .env
  echo "Created .env — add your GROQ_API_KEY to it for AI-generated advice (optional)."
fi

echo "Starting app at http://localhost:7860"
python -m app.main
