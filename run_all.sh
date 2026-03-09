#!/bin/bash
set -e

echo "=== GNR 638 Assignment 2 - Running All Scenarios ==="
echo "Note: Scenario scripts now support arguments for faster testing!"
echo "Example: python3 scenario_2_finetuning.py --model resnet50 --epochs 10"
echo "------------------------------------------------------------"
echo "Start time: $(date)"

# Activate venv
source venv/bin/activate

echo ""
echo "============================================"
echo "  SCENARIO 1: Linear Probe Transfer"
echo "============================================"
python3 scenario_1_linear_probe.py

echo ""
echo "============================================"
echo "  SCENARIO 2: Fine-Tuning Strategies"
echo "============================================"
python3 scenario_2_finetuning.py

echo ""
echo "============================================"
echo "  SCENARIO 3: Few-Shot Learning"
echo "============================================"
python3 scenario_3_fewshot.py

echo ""
echo "============================================"
echo "  SCENARIO 4: Corruption Robustness"
echo "============================================"
python3 scenario_4_corruption.py

echo ""
echo "============================================"
echo "  SCENARIO 5: Layer-Wise Feature Probing"
echo "============================================"
python3 scenario_5_layerwise.py

echo ""
echo "=== All scenarios complete! ==="
echo "End time: $(date)"
echo "Results saved in outputs/ directory"
