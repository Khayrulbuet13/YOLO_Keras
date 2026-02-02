#!/bin/bash
# Test KD training with teacher evaluation to verify teacher is loaded correctly

# Activate virtual environment
source ~/.virtualenvs/yolo/bin/activate

echo "========================================================================="
echo "  KD Training with Teacher Evaluation"
echo "========================================================================="
echo "This will:"
echo "1. Load the teacher model"
echo "2. Evaluate teacher on train/val sets to verify it works"
echo "3. Start KD training (quantized student)"
echo ""
echo "Teacher metrics will be shown at the start and logged in step.csv"
echo "========================================================================="
echo ""

python3 main_keras.py \
    --train \
    --kd \
    --teacher-weights results/rect_256x128_cleaned/best.weights.h5 \
    --quantized \
    --input-size 128x256 \
    --batch-size 4 \
    --epochs 10 \
    --kd-temperature 3.0 \
    --kd-alpha 0.5 \
    --kd-beta 0.5 \
    --save-path ./results/rect_256x128_kd_with_teacher_eval \
    --dataset-dir ./Dataset/bionano_cellv2

echo ""
echo "========================================================================="
echo "Training complete! Check results:"
echo "  - results/rect_256x128_kd_with_teacher_eval/step.csv"
echo ""
echo "The CSV will show teacher metrics (constant across epochs) and"
echo "student metrics (should improve if learning works)"
echo "========================================================================="
