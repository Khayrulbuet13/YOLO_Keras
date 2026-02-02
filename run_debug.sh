#!/bin/bash
# Comprehensive debugging script for KD training issues

# Activate virtual environment
source ~/.virtualenvs/yolo/bin/activate

echo "========================================================================="
echo "  KNOWLEDGE DISTILLATION DEBUGGING SUITE"
echo "========================================================================="

# Debug 1: Verify teacher model knowledge
echo ""
echo "=========================================================================";
echo " DEBUG 1: Verifying Teacher Model Knowledge"
echo "=========================================================================";
echo ""
python3 debug_teacher.py

echo ""
echo "Press Enter to continue to Debug 2 (or Ctrl+C to stop)..."
read

# Debug 2: Test with full-precision student (identical architecture)
echo ""
echo "=========================================================================";
echo " DEBUG 2: Testing KD with Full-Precision Student (10 epochs)"
echo "=========================================================================";
echo "This tests if the KD pipeline works when both models have the same architecture"
echo ""

python3 main_keras.py \
    --train \
    --kd \
    --debug-fp-student \
    --teacher-weights results/rect_256x128_cleaned/best.weights.h5 \
    --input-size 128x256 \
    --batch-size 4 \
    --epochs 10 \
    --kd-temperature 3.0 \
    --kd-alpha 0.5 \
    --kd-beta 0.5 \
    --save-path ./results/rect_256x128_kd_debug_fp \
    --dataset-dir ./Dataset/bionano_cellv2

echo ""
echo "=========================================================================";
echo " DEBUG 2 COMPLETE"
echo "=========================================================================";
echo "Check results/rect_256x128_kd_debug_fp/step.csv to see if metrics improved"
echo ""
echo "If full-precision student learned (metrics > 0):"
echo "  → KD pipeline works, quantization is the issue"
echo ""
echo "If full-precision student also has NaN metrics:"
echo "  → KD pipeline or teacher has fundamental issues"
echo ""
echo "=========================================================================";
