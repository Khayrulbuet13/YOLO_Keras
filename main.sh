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
