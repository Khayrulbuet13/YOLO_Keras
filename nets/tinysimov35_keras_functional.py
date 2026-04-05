"""
Functional API YOLO model for QKeras compatibility and HLS4ml synthesis.

This module provides a pure Keras Functional API implementation of the YOLO model,
enabling seamless integration with QKeras utilities like model_quantize(), 
print_qstats(), and quantized_model_debug().

Key differences from the subclassed version:
- No custom Layer/Model subclasses - pure functional API
- DFL decoding moved to standalone function (no learnable parameters)
- All layers explicitly named for easy QKeras quantization mapping
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
import math
import numpy as np


def build_yolo_functional(num_classes=1, img_size=(256, 256), dtype=tf.float32):
    """
    Build YOLO model using Keras Functional API.
    
    Args:
        num_classes: Number of detection classes
        img_size: Input image size (height, width)
        dtype: Model dtype
        
    Returns:
        tf.keras.Model with raw predictions (B, H, W, 4*dfl_ch + nc)
    """
    if isinstance(img_size, int):
        img_size = (img_size, img_size)
    
    h, w = img_size
    dfl_ch = 16
    
    # Input layer
    x_in = layers.Input(shape=(h, w, 3), dtype=dtype, name='input')
    x = x_in
    
    # Backbone: Sequential Conv blocks matching widths=[3,4,8,16,64,128], depths=[1,1,1,1]
    # With depths=[1,1,1,1], we get 4 blocks total (one per depth entry)
    # Block 0: 3 -> 4 channels, stride 2
    x = layers.Conv2D(4, 3, strides=2, padding='same', use_bias=False, 
                      dtype=dtype, name='backbone_conv1')(x)
    x = layers.BatchNormalization(epsilon=0.001, momentum=0.03, 
                                   dtype=dtype, name='backbone_bn1')(x)
    x = layers.ReLU(dtype=dtype, name='backbone_relu1')(x)
    
    # Block 1: 4 -> 8 channels, stride 2
    x = layers.Conv2D(8, 3, strides=2, padding='same', use_bias=False,
                      dtype=dtype, name='backbone_conv2')(x)
    x = layers.BatchNormalization(epsilon=0.001, momentum=0.03,
                                   dtype=dtype, name='backbone_bn2')(x)
    x = layers.ReLU(dtype=dtype, name='backbone_relu2')(x)
    
    # Block 2: 8 -> 16 channels, stride 2
    x = layers.Conv2D(16, 3, strides=2, padding='same', use_bias=False,
                      dtype=dtype, name='backbone_conv3')(x)
    x = layers.BatchNormalization(epsilon=0.001, momentum=0.03,
                                   dtype=dtype, name='backbone_bn3')(x)
    x = layers.ReLU(dtype=dtype, name='backbone_relu3')(x)
    
    # Block 3: 16 -> 64 channels, stride 1 (final backbone layer)
    backbone_out = layers.Conv2D(64, 3, strides=1, padding='same', use_bias=False,
                                  dtype=dtype, name='backbone_conv4')(x)
    backbone_out = layers.BatchNormalization(epsilon=0.001, momentum=0.03,
                                              dtype=dtype, name='backbone_bn4')(backbone_out)
    backbone_out = layers.ReLU(dtype=dtype, name='backbone_relu4')(backbone_out)
    
    # Prediction heads applied directly to backbone output (64 channels)
    # Note: head.conv is defined in subclassed model but never used in training!
    box_output = layers.Conv2D(4 * dfl_ch, 1, dtype=dtype, name='box_conv')(backbone_out)
    cls_output = layers.Conv2D(num_classes, 1, dtype=dtype, name='cls_conv')(backbone_out)
    
    # Concatenate outputs (training format)
    output = layers.Concatenate(axis=-1, name='output_concat')([box_output, cls_output])
    
    # Build model
    model = Model(inputs=x_in, outputs=output, name='yolo_functional')
    
    # Attach metadata (needed by ComputeLoss)
    model.nc = num_classes
    model.dfl_ch = dfl_ch
    model.no = num_classes + 4 * dfl_ch
    
    # Calculate stride by running dummy forward pass -- force CPU to avoid CuDNN version mismatches
    with tf.device('/CPU:0'):
        dummy = tf.zeros((1, h, w, 3), dtype=dtype)
        features = model(dummy, training=False)
    feature_h, feature_w = features.shape[1], features.shape[2]
    stride_h = h / feature_h
    stride_w = w / feature_w
    
    if not math.isclose(stride_h, stride_w, rel_tol=1e-5):
        print(f"Warning: Stride mismatch: h={stride_h}, w={stride_w}. Using average.")
    
    model.stride = tf.constant([(stride_h + stride_w) / 2], dtype=dtype)
    
    # Initialize biases using numpy/math (avoid GPU dispatch for scalar ops)
    import math as _math
    import numpy as _np
    s = float(model.stride[0].numpy())

    box_layer = model.get_layer('box_conv')
    if box_layer.bias is not None:
        box_layer.bias.assign(tf.ones_like(box_layer.bias))

    cls_layer = model.get_layer('cls_conv')
    if cls_layer.bias is not None:
        bias_value = float(_math.log(5 / num_classes / (640 / s) ** 2))
        cls_layer.bias.assign(
            tf.constant(_np.full(cls_layer.bias.shape, bias_value), dtype=cls_layer.bias.dtype)
        )
    
    return model


def decode_predictions(raw_output, stride, nc, dfl_ch=16, dtype=tf.float32):
    """
    Decode raw model predictions to bounding boxes (inference mode).
    
    This is a standalone function with no learnable parameters, equivalent to
    the inference path in the subclassed Head/YOLO model.
    
    Args:
        raw_output: Raw model output (B, H, W, 4*dfl_ch + nc)
        stride: Feature map stride (scalar or tensor)
        nc: Number of classes
        dfl_ch: DFL channels (default 16)
        dtype: Output dtype
        
    Returns:
        Decoded predictions (B, HW, 4+nc) where 4 = [x, y, w, h] in pixel coords
    """
    b = tf.shape(raw_output)[0]
    h = tf.shape(raw_output)[1]
    w = tf.shape(raw_output)[2]
    
    # Split into box and class predictions
    box = raw_output[..., :4*dfl_ch]  # (B, H, W, 64)
    cls = raw_output[..., 4*dfl_ch:]  # (B, H, W, nc)
    
    # Flatten spatial dimensions
    box_flat = tf.reshape(box, [b, -1, 4 * dfl_ch])  # (B, HW, 64)
    cls_flat = tf.reshape(cls, [b, -1, nc])          # (B, HW, nc)
    
    # DFL decoding: reshape to (B, HW, 4, 16), apply softmax, weighted sum
    box_processed = tf.reshape(box_flat, [b, -1, 4, dfl_ch])  # (B, HW, 4, 16)
    box_processed = tf.nn.softmax(box_processed, axis=-1)
    
    # Weighted sum with [0..15] to get distance predictions
    dfl_weights = tf.range(dfl_ch, dtype=dtype)
    box_dfl = tf.reduce_sum(box_processed * dfl_weights, axis=-1)  # (B, HW, 4)
    
    # Generate anchors (grid centers in feature map coordinates)
    grid_x = tf.cast(tf.range(w), dtype=dtype) + 0.5
    grid_y = tf.cast(tf.range(h), dtype=dtype) + 0.5
    grid = tf.stack(tf.meshgrid(grid_x, grid_y, indexing='xy'), axis=-1)  # (H, W, 2)
    anchors = tf.reshape(grid, [1, -1, 2])  # (1, HW, 2)
    
    # Decode boxes: split into left-top and right-bottom distances
    lt, rb = tf.split(box_dfl, 2, axis=-1)  # Each (B, HW, 2)
    
    # Convert to center + size format
    # PyTorch: cat(((anchors - lt + anchors + rb) / 2, lt + rb), -1)
    boxes = tf.concat([
        (anchors - lt + anchors + rb) / 2,  # Center: (x, y)
        (lt + rb)                            # Size: (w, h)
    ], axis=-1)  # (B, HW, 4)
    
    # Scale by stride to get pixel coordinates
    if isinstance(stride, (int, float)):
        stride = tf.constant(stride, dtype=dtype)
    boxes = boxes * stride
    
    # Apply sigmoid to class scores
    cls_scores = tf.sigmoid(cls_flat)  # (B, HW, nc)
    
    # Concatenate boxes and class scores
    output = tf.concat([boxes, cls_scores], axis=-1)  # (B, HW, 4+nc)
    
    return output


def yolo_v8_s_functional(num_classes=1, img_size=(256, 256), dtype=tf.float32):
    """
    Factory function for functional YOLO model (drop-in replacement for yolo_v8_s).
    
    Args:
        num_classes: Number of detection classes
        img_size: Input image size, can be int (square) or tuple (h, w)
        dtype: Model dtype
        
    Returns:
        Keras Functional API model
    """
    return build_yolo_functional(num_classes, img_size, dtype)
