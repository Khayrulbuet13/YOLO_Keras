"""
Quantized Functional API YOLO model for QKeras and HLS4ml synthesis.

This is a quantized version of tinysimov35_keras_hls4ml.py using QKeras layers.
Uses pure Functional API for compatibility with QKeras utilities.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
import math
from qkeras import QConv2D, QActivation, quantized_bits, quantized_relu


def build_yolo_quantized_functional(num_classes=1, img_size=(256, 256), 
                                     weight_bits=8, activation_bits=8, dtype=tf.float32):
    """
    Build quantized YOLO model using Keras Functional API with QKeras layers.
    
    Args:
        num_classes: Number of detection classes
        img_size: Input image size (height, width)
        weight_bits: Bit width for weight quantization
        activation_bits: Bit width for activation quantization
        dtype: Model dtype
        
    Returns:
        tf.keras.Model with QKeras quantized layers
    """
    if isinstance(img_size, int):
        img_size = (img_size, img_size)
    
    h, w = img_size
    dfl_ch = 16
    
    # Quantizer configuration
    # Fixed alpha=1 for training stability
    # Integer bits: 3 for weights (range [-8, 8]), 4 for activations (range [0, 16])
    weight_quantizer = quantized_bits(weight_bits, 3, symmetric=1, alpha=1)
    activation_quantizer = quantized_relu(activation_bits, 4)
    
    # Input layer
    x_in = layers.Input(shape=(h, w, 3), dtype=dtype, name='input')
    x = x_in
    
    # Backbone: Sequential QConv blocks matching widths=[3,4,8,16,64], depths=[1,1,1,1]
    # Block 0: 3 -> 4 channels, stride 2
    x = QConv2D(4, 3, strides=2, padding='same', use_bias=False,
                kernel_quantizer=weight_quantizer,
                dtype=dtype, name='backbone_qconv1')(x)
    x = layers.BatchNormalization(epsilon=0.001, momentum=0.03,
                                   dtype=dtype, name='backbone_bn1')(x)
    x = QActivation(activation_quantizer, name='backbone_qrelu1')(x)
    
    # Block 1: 4 -> 8 channels, stride 2
    x = QConv2D(8, 3, strides=2, padding='same', use_bias=False,
                kernel_quantizer=weight_quantizer,
                dtype=dtype, name='backbone_qconv2')(x)
    x = layers.BatchNormalization(epsilon=0.001, momentum=0.03,
                                   dtype=dtype, name='backbone_bn2')(x)
    x = QActivation(activation_quantizer, name='backbone_qrelu2')(x)
    
    # Block 2: 8 -> 16 channels, stride 2
    x = QConv2D(16, 3, strides=2, padding='same', use_bias=False,
                kernel_quantizer=weight_quantizer,
                dtype=dtype, name='backbone_qconv3')(x)
    x = layers.BatchNormalization(epsilon=0.001, momentum=0.03,
                                   dtype=dtype, name='backbone_bn3')(x)
    x = QActivation(activation_quantizer, name='backbone_qrelu3')(x)
    
    # Block 3: 16 -> 64 channels, stride 1 (final backbone layer)
    backbone_out = QConv2D(64, 3, strides=1, padding='same', use_bias=False,
                           kernel_quantizer=weight_quantizer,
                           dtype=dtype, name='backbone_qconv4')(x)
    backbone_out = layers.BatchNormalization(epsilon=0.001, momentum=0.03,
                                              dtype=dtype, name='backbone_bn4')(backbone_out)
    backbone_out = QActivation(activation_quantizer, name='backbone_qrelu4')(backbone_out)
    
    # Prediction heads with quantization
    bias_quantizer = quantized_bits(weight_bits, 3, symmetric=1, alpha=1)
    
    box_output = QConv2D(4 * dfl_ch, 1,
                         kernel_quantizer=weight_quantizer,
                         bias_quantizer=bias_quantizer,
                         dtype=dtype, name='box_qconv')(backbone_out)
    
    cls_output = QConv2D(num_classes, 1,
                         kernel_quantizer=weight_quantizer,
                         bias_quantizer=bias_quantizer,
                         dtype=dtype, name='cls_qconv')(backbone_out)
    
    # Concatenate outputs (training format)
    output = layers.Concatenate(axis=-1, name='output_concat')([box_output, cls_output])
    
    # Build model
    model = Model(inputs=x_in, outputs=output, name='yolo_quantized_functional')
    
    # Attach metadata (needed by ComputeLoss)
    model.nc = num_classes
    model.dfl_ch = dfl_ch
    model.no = num_classes + 4 * dfl_ch
    
    # Calculate stride by running dummy forward pass
    dummy = tf.zeros((1, h, w, 3), dtype=dtype)
    features = model(dummy, training=False)
    feature_h, feature_w = features.shape[1], features.shape[2]
    stride_h = h / feature_h
    stride_w = w / feature_w
    
    if not math.isclose(stride_h, stride_w, rel_tol=1e-5):
        print(f"Warning: Stride mismatch: h={stride_h}, w={stride_w}. Using average.")
    
    model.stride = tf.constant([(stride_h + stride_w) / 2], dtype=dtype)
    
    # Initialize biases
    s = model.stride[0].numpy()
    
    # Box bias: set to 1.0
    box_layer = model.get_layer('box_qconv')
    if box_layer.bias is not None:
        box_layer.bias.assign(tf.ones_like(box_layer.bias))
    
    # Class bias: log(5 / nc / (640 / s)^2)
    cls_layer = model.get_layer('cls_qconv')
    if cls_layer.bias is not None:
        bias_value = tf.math.log(5 / num_classes / (640 / s) ** 2)
        cls_layer.bias.assign(tf.ones_like(cls_layer.bias) * tf.cast(bias_value, cls_layer.bias.dtype))
    
    return model


def yolo_v8_s_quantized_functional(num_classes=1, img_size=(256, 256), 
                                    weight_bits=8, activation_bits=8, dtype=tf.float32):
    """
    Factory function for quantized functional YOLO model.
    
    Args:
        num_classes: Number of detection classes
        img_size: Input image size, can be int (square) or tuple (h, w)
        weight_bits: Bit width for weight quantization (default: 8)
        activation_bits: Bit width for activation quantization (default: 8)
        dtype: Model dtype
        
    Returns:
        Keras Functional API model with QKeras quantized layers
    """
    return build_yolo_quantized_functional(num_classes, img_size, weight_bits, activation_bits, dtype)
