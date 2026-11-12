#!/usr/bin/env bash
set -euo pipefail

echo "================================================================================"
echo "  Business Data Case Study (Team 6): Machine Learning for Derivative Pricing"
echo "  Team: Ramitha MV (PES1UG24AM392), Gayathri Priya V (PES1UG24AM105),"
echo "        Gonu Guntla Hema Kowshik (PES1UG24AM106)"
echo "================================================================================"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if command -v /opt/homebrew/bin/python3.12 >/dev/null 2>&1; then
    PYTHON_BIN="/opt/homebrew/bin/python3.12"
else
    PYTHON_BIN="python3"
fi

echo "[INFO] Running Derivative Pricing ML pipeline using $PYTHON_BIN..."
"$PYTHON_BIN" run_pipeline.py

echo "[INFO] Verifying generated datasets, results, and figures..."
ls -lh data/ results/ results/figures/ report/*.pdf

echo "[SUCCESS] All datasets, EDA statistics, 5 frameworks + PI-BSNet, and figures verified."
