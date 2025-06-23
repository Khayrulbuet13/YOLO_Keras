#!/bin/bash

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Create results directory if it doesn't exist
mkdir -p results

# Run PyTorch implementation
echo "Running PyTorch implementation..."
python main.py --train --input-size 128x256 --epochs 2 > debug_log_pytorch.txt 2>&1

# Run Keras implementation  
echo "Running Keras implementation..."
python main_keras.py --train --input-size 128x256 --epochs 2 > debug_log_keras.txt 2>&1

echo "Training completed. Check debug_log_pytorch.txt and debug_log_keras.txt for results."



