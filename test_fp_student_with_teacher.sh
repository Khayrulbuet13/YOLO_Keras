#!/bin/bash
# Test KD with full-precision student AND teacher evaluation

# Activate virtual environment
source ~/.virtualenvs/yolo/bin/activate

echo "========================================================================="
echo "  KD Training: Full-Precision Student + Teacher Evaluation"
echo "========================================================================="
echo "This will:"
echo "1. Evaluate teacher model (should show good metrics)"
echo "2. Train full-precision student from teacher (to test KD pipeline)"
echo ""
echo "If teacher metrics are good BUT student stays at NaN:"
echo "  → KD loss or pipeline has issues"
echo ""
echo "If both teacher and student have NaN:"
echo "  → Teacher loading or evaluation has issues"
echo "========================================================================="
echo ""

python3 main_keras.py \
    --train \
    --kd \
    --debug-fp-student \
    --teacher-weights results/rect_256x128_cleaned/best.weights.h5 \
    --input-size 128x256 \
    --batch-size 4 \
    --epochs 1000 \
    --kd-temperature 1.0 \
    --kd-alpha 0.5 \
    --kd-beta 0.5 \
    --save-path ./results/rect_256x128_kd_fp_with_teacherv2 \
    --dataset-dir ./Dataset/bionano_cellv2

echo ""
echo "========================================================================="
echo "Check results/rect_256x128_kd_fp_with_teacher/step.csv"
echo ""
echo "Compare teacher vs student metrics:"
echo "  - Teacher columns show teacher performance (should be good)"
echo "  - Student columns show if KD learning works"
echo "========================================================================="
