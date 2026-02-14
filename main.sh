#!/bin/bash

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Create results directory if it doesn't exist
mkdir -p results

# Run training
echo "Starting YOLO training..."
python3 main_keras.py --train --input-size 128x256 --epochs 100 --batch-size 8 > training_log_keras.txt 2>&1

echo "Training completed. Check training_log_keras.txt for results."


# Functional API
python main_keras.py --train --quantized --yaml_file utils/args_bionano.yaml --save-path ./results/quantized_integration_from_pretrained --input-size 128x256 --pretrained-weights results/rect_256x128_functional/best.weights.h5

# ============================================================================
# KNOWLEDGE DISTILLATION TRAINING EXAMPLES
# ============================================================================

# Example 1: KD training with LymphoML-aligned hyperparameters (recommended starting point)
# Trains quantized student using knowledge from full-precision teacher
# - Teacher: Pre-trained float32 functional model
# - Student: Quantized model initialized from teacher weights (warmstart)
# - KD alpha: 1.0 (pure KD loss, aligned with LymphoML Student2)
# - Temperature: 1.0 (no softening, aligned with LymphoML Student2)
# - Learning rate: 0.001 (constant, from args_bionano_kd.yaml)
# - Weight decay: 0.0001 (from args_bionano_kd.yaml)
python main_kd.py \
    --train \
    --input-size 256 \
    --batch-size 4 \
    --epochs 3000 \
    --teacher-weights results/rect_256x128_functional/best.weights.h5 \
    --pretrained-weights results/rect_256x128_functional/best.weights.h5 \
    --kd-alpha 1.0 \
    --kd-temperature 1.0 \
    --kd-box-weight 1.0 \
    --kd-cls-weight 1.0 \
    --yaml_file utils/args_bionano_kd.yaml \
    --save-path ./results/kd_quantized_lymphoml \
    --dataset-dir ./Dataset/bionano_cellv2