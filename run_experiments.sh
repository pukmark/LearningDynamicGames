


#!/bin/bash

set -e

source .venv/bin/activate

PYTHON_SCRIPT="LDG_Simulation.py"

echo "Run 1: Normalized GNE + Nash Bargaining"
python3 "$PYTHON_SCRIPT" \
    --cooperative-selection nash_bargaining \
    --bargaining-gamma1 0.3333333333333333 \
    --bargaining-gamma2 0.3333333333333333 \
    --movie LDG_normalized_GNE_nash.mp4


echo "Run 2: Multiple GNEs + Nash Bargaining"
python3 "$PYTHON_SCRIPT" \
    --cooperative-selection nash_bargaining \
    --bargaining-gamma1 \
        0.3333333333333333 0.35 0.30 0.35 0.40 0.30 0.30 0.50 0.25 0.25 0.80 0.10 0.10 \
    --bargaining-gamma2 \
        0.3333333333333333 0.30 0.35 0.35 0.30 0.40 0.30 0.25 0.50 0.25 0.10 0.80 0.10 \
    --movie LDG_multiple_GNE_nash.mp4


echo "Run 3: Multiple GNEs + Weighted Sum"
python3 "$PYTHON_SCRIPT" \
    --cooperative-selection weighted_sum \
    --cooperative-cost-weights \
        0.3333333333333333 0.3333333333333333 0.3333333333333333 \
    --bargaining-gamma1 \
        0.3333333333333333 0.35 0.30 0.35 0.40 0.30 0.40 0.50 0.25 0.25 0.80 0.10 0.10 \
    --bargaining-gamma2 \
        0.3333333333333333 0.30 0.35 0.35 0.30 0.40 0.40 0.25 0.50 0.25 0.10 0.80 0.10 \
    --movie LDG_multiple_GNE_weighted_sum.mp4


echo "========================================"
echo "All simulations completed."
echo "========================================"
