#!/bin/bash
# Create venv and install dependencies (use this on macOS/Homebrew when pip says "externally-managed-environment")
cd "$(dirname "$0")"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
echo ""
echo "Done. To use the venv, run:"
echo "  source .venv/bin/activate"
echo "Then run clean_data.py or train_live_birth.py as usual."
