#!/bin/bash
# Knowledge Distillation Training Script
#
# This script trains a quantized student model using Knowledge Distillation
# from a pretrained full-precision teacher model.

# Activate virtual environment
source ~/.virtualenvs/yolo/bin/activate

# Configuration
TEACHER_WEIGHTS="results/rect_256x128_cleaned/best.weights.h5"
INPUT_SIZE="128x256"
BATCH_SIZE=4
EPOCHS=200
KD_TEMPERATURE=3.0
KD_ALPHA=0.5
KD_BETA=0.5
SAVE_PATH="./results/rect_256x128_kd"
DATASET_DIR="./Dataset/bionano_cellv2"

# Check if teacher weights exist
if [ ! -f "$TEACHER_WEIGHTS" ]; then
    echo "Error: Teacher weights not found at $TEACHER_WEIGHTS"
    echo "Please train a teacher model first or specify the correct path."
    exit 1
fi

echo "Starting Knowledge Distillation Training"
echo "========================================"
echo "Teacher weights: $TEACHER_WEIGHTS"
echo "Input size: $INPUT_SIZE"
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "KD temperature: $KD_TEMPERATURE"
echo "KD alpha (box): $KD_ALPHA"
echo "KD beta (class): $KD_BETA"
echo "Save path: $SAVE_PATH"
echo "========================================"

python3 main_keras.py \
    --train \
    --kd \
    --teacher-weights "$TEACHER_WEIGHTS" \
    --quantized \
    --input-size "$INPUT_SIZE" \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --kd-temperature $KD_TEMPERATURE \
    --kd-alpha $KD_ALPHA \
    --kd-beta $KD_BETA \
    --save-path "$SAVE_PATH" \
    --dataset-dir "$DATASET_DIR"
