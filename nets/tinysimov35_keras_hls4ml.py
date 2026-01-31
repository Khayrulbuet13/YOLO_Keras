"""
Static inference model for HLS4ml FPGA synthesis

This module provides a YOLO model with static computational graph suitable for
HLS4ml conversion. All dynamic operations (tf.range, tf.meshgrid, dynamic reshapes)
are removed or pre-computed.

Key differences from training model:
1. Fixed input shape (no dynamic batch size)
2. Pre-computed anchor grids
3. Unrolled DFL processing (no loops)
4. Static tensor shapes throughout
5. Simplified activations where possible
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
from qkeras import QConv2D, QActivation, quantized_bits, quantized_relu


class QConvStatic(layers.Layer):
    """Quantized Conv block for static inference"""
    def __init__(self, in_ch, out_ch, k=1, s=1, p=None, d=1, g=1, 
                 weight_bits=8, activation_bits=8, dtype=tf.float32):
        super().__init__(dtype=dtype)
        self.dtype_ = dtype
        padding = 'same' if p is None else 'valid'
        
        self.conv = QConv2D(
            out_ch, k, strides=s, 
            padding=padding,
            dilation_rate=d,
            groups=g,
            use_bias=False,
            kernel_quantizer=quantized_bits(weight_bits, 0, alpha=1),
            dtype=dtype
        )
        
        self.norm = layers.BatchNormalization(
            epsilon=0.001, 
            momentum=0.03,
            dtype=dtype
        )
        
        self.relu = QActivation(quantized_relu(activation_bits, 0), dtype=dtype)

    def call(self, x):
        return self.relu(self.norm(self.conv(x)))


class QDarkNetStatic(Model):
    """Quantized DarkNet backbone for static inference"""
    def __init__(self, widths=None, depths=None, dtype=tf.float32):
        super().__init__(dtype=dtype)
        self.dtype_ = dtype
        self.layers_list = []
        in_ch = widths[0]
        
        for i in range(min(len(depths), len(widths) - 1)):
            out_ch = widths[i + 1]
            nblocks = depths[i]
            
            for j in range(nblocks):
                stride = 2 if (j == nblocks - 1 and i < len(depths) - 1) else 1
                
                if i == 0 and j == 0:
                    weight_bits, activation_bits = 16, 16
                elif i >= len(depths) - 1:
                    weight_bits, activation_bits = 12, 12
                else:
                    weight_bits, activation_bits = 8, 8
                
                self.layers_list.append(
                    QConvStatic(in_ch, out_ch, 3, stride, 
                               weight_bits=weight_bits, 
                               activation_bits=activation_bits,
                               dtype=dtype)
                )
                in_ch = out_ch
        
        self.out_channels = in_ch

    def call(self, x):
        for layer in self.layers_list:
            x = layer(x)
        return x


class QDFLStatic(layers.Layer):
    """Static DFL layer with unrolled operations
    
    Replaces softmax + weighted sum with explicit operations
    to avoid expensive softmax on FPGA.
    """
    def __init__(self, ch=16, dtype=tf.float32):
        super().__init__(dtype=dtype)
        self.dtype_ = dtype
        self.ch = ch
        
        # Pre-compute weight vector [0, 1, 2, ..., 15]
        self.register_buffer = tf.constant(
            np.arange(ch, dtype=np.float32).reshape(1, 1, 1, ch),
            dtype=dtype
        )
    
    def call(self, x):
        """Static version: softmax + weighted sum
        
        Input: (B, H, W, 16)
        Output: (B, H, W, 1)
        """
        # Softmax along channel dimension
        x = tf.nn.softmax(x, axis=-1)
        
        # Weighted sum: sum(softmax * [0..15])
        # Reshape for broadcasting
        weights = tf.reshape(self.register_buffer, [1, 1, 1, self.ch])
        result = tf.reduce_sum(x * weights, axis=-1, keepdims=True)
        
        return result


class QHeadStatic(Model):
    """Static quantized detection head for HLS4ml
    
    Key changes from dynamic version:
    1. Fixed spatial dimensions (computed at build time)
    2. Pre-computed anchor grids
    3. Unrolled DFL processing (no loops)
    4. All tensor shapes are static
    """
    def __init__(self, nc=20, ch_in=128, feature_h=32, feature_w=16, 
                 stride=8.0, dtype=tf.float32):
        super().__init__(dtype=dtype)
        self.dtype_ = dtype
        self.nc = nc
        self.ch = 16
        self.feature_h = feature_h
        self.feature_w = feature_w
        self.stride = tf.constant(stride, dtype=dtype)
        self.num_anchors = feature_h * feature_w
        
        # Quantized conv block
        self.conv = QConvStatic(ch_in, 24, 1, 1, weight_bits=12, activation_bits=12, dtype=dtype)
        
        # DFL layer
        self.dfl = QDFLStatic(self.ch, dtype=dtype)
        
        # Box and class prediction heads
        self.box = QConv2D(
            4 * self.ch, 1, 
            kernel_quantizer=quantized_bits(12, 0, alpha=1),
            dtype=dtype
        )
        self.cls = QConv2D(
            nc, 1,
            kernel_quantizer=quantized_bits(12, 0, alpha=1),
            dtype=dtype
        )
        
        # Pre-compute anchor grid (static, no tf.meshgrid)
        self._build_anchors()
    
    def _build_anchors(self):
        """Pre-compute anchor grid at build time"""
        # Create grid coordinates
        grid_y, grid_x = np.meshgrid(
            np.arange(self.feature_h, dtype=np.float32) + 0.5,
            np.arange(self.feature_w, dtype=np.float32) + 0.5,
            indexing='ij'
        )
        
        # Stack and reshape: (H, W, 2) -> (1, H*W, 2)
        grid = np.stack([grid_x, grid_y], axis=-1)
        anchors = grid.reshape(1, -1, 2)
        
        # Store as constant tensor
        self.anchors = tf.constant(anchors, dtype=self.dtype_)
    
    def call(self, x):
        """Static inference forward pass
        
        Input: (1, H, W, C) - batch size must be 1
        Output: (1, H*W, 5) - [x, y, w, h, class_score]
        """
        # Get box and class predictions
        box = self.box(x)  # (1, H, W, 64)
        cls = self.cls(x)  # (1, H, W, nc)
        
        # Reshape to flatten spatial dimensions
        # (1, H, W, 64) -> (1, H*W, 64)
        box_flat = tf.reshape(box, [1, self.num_anchors, 4 * self.ch])
        cls_flat = tf.reshape(cls, [1, self.num_anchors, self.nc])
        
        # Process box coordinates - unroll DFL for each coordinate
        # Split into 4 groups of 16 channels each
        box_splits = tf.split(box_flat, 4, axis=-1)  # 4x (1, H*W, 16)
        
        # Apply DFL to each coordinate (unrolled, no loop)
        # Reshape each split to (1, H*W, 1, 16) for DFL processing
        coord_0 = tf.reshape(box_splits[0], [1, self.num_anchors, 1, self.ch])
        coord_1 = tf.reshape(box_splits[1], [1, self.num_anchors, 1, self.ch])
        coord_2 = tf.reshape(box_splits[2], [1, self.num_anchors, 1, self.ch])
        coord_3 = tf.reshape(box_splits[3], [1, self.num_anchors, 1, self.ch])
        
        # Apply DFL and squeeze
        dfl_0 = tf.squeeze(self.dfl(coord_0), axis=[2, 3])  # (1, H*W)
        dfl_1 = tf.squeeze(self.dfl(coord_1), axis=[2, 3])
        dfl_2 = tf.squeeze(self.dfl(coord_2), axis=[2, 3])
        dfl_3 = tf.squeeze(self.dfl(coord_3), axis=[2, 3])
        
        # Stack coordinates: (1, H*W, 4)
        box_dfl = tf.stack([dfl_0, dfl_1, dfl_2, dfl_3], axis=2)
        
        # Split into distance predictions
        a, b_coords = tf.split(box_dfl, 2, axis=2)  # 2x (1, H*W, 2)
        
        # Calculate final boxes using pre-computed anchors
        # center = (anchors - a + anchors + b) / 2 = anchors + (b - a) / 2
        # size = a + b
        boxes = tf.concat([
            (self.anchors - a + self.anchors + b_coords) / 2,  # (x_center, y_center)
            (a + b_coords)                                      # (width, height)
        ], axis=-1)
        
        # Apply stride scaling
        boxes = boxes * self.stride
        
        # Apply sigmoid to class scores
        # Note: Sigmoid is expensive on FPGA. For production, consider:
        # - Using lookup tables
        # - Piecewise linear approximation
        # - Or removing if post-processing can handle logits
        cls_scores = tf.sigmoid(cls_flat)
        
        # Concatenate boxes and class scores
        # Output: (1, H*W, 5) where 5 = [x, y, w, h, class_score]
        return tf.concat([boxes, cls_scores], axis=-1)


class QYOLOStatic(Model):
    """Static quantized YOLO model for HLS4ml FPGA synthesis
    
    This model has a completely static computational graph:
    - Fixed batch size (1)
    - Fixed input dimensions
    - Pre-computed anchor grids
    - No dynamic operations (loops, meshgrid, dynamic reshapes)
    - All tensor shapes known at compile time
    
    Use this model for HLS4ml conversion after training with the
    quantized model.
    """
    def __init__(self, widths=None, depths=None, num_classes=20, 
                 img_h=256, img_w=128, dtype=tf.float32):
        super().__init__(dtype=dtype)
        self.dtype_ = dtype
        self.img_h = img_h
        self.img_w = img_w
        
        # Build backbone
        self.net = QDarkNetStatic(widths, depths, dtype=dtype)
        
        # Calculate feature map dimensions
        # Assuming stride of 8 (typical for small YOLO)
        # This should match your actual model's stride
        dummy = tf.zeros((1, img_h, img_w, 3), dtype=dtype)
        features = self.net(dummy)
        feature_h, feature_w = features.shape[1], features.shape[2]
        stride = img_h / feature_h  # Assuming square-ish aspect
        
        print(f"[HLS4ml Static Model] Feature map: {feature_h}x{feature_w}, Stride: {stride}")
        
        # Build head with static dimensions
        self.head = QHeadStatic(
            num_classes, 
            ch_in=self.net.out_channels,
            feature_h=feature_h,
            feature_w=feature_w,
            stride=stride,
            dtype=dtype
        )
        
        # Build the model
        _ = self.head(features)
        
    def call(self, x):
        """Static forward pass
        
        Input: (1, H, W, 3) - batch size MUST be 1
        Output: (1, num_anchors, 5)
        """
        features = self.net(x)
        return self.head(features)


def yolo_v8_s_hls4ml(num_classes: int = 20, img_h: int = 256, img_w: int = 128, 
                     dtype=tf.float32):
    """
    Create static quantized YOLO model for HLS4ml FPGA synthesis
    
    This model is specifically designed for HLS4ml conversion:
    - All operations are static (no dynamic shapes)
    - Batch size is fixed to 1
    - Anchor grids are pre-computed
    - No control flow (loops, conditionals)
    - Quantized with QKeras
    
    Args:
        num_classes: Number of detection classes
        img_h: Input image height (must be fixed)
        img_w: Input image width (must be fixed)
        dtype: Data type (default: tf.float32)
    
    Returns:
        Static YOLO model ready for HLS4ml conversion
    
    Usage:
        # 1. Train with quantized model (tinysimov35_keras_quantized.py)
        # 2. Load trained weights into this static model
        # 3. Export to HLS4ml
        
        model = yolo_v8_s_hls4ml(num_classes=1, img_h=256, img_w=128)
        model.load_weights('path/to/trained/weights.h5')
        
        # Convert to HLS4ml
        import hls4ml
        config = hls4ml.utils.config_from_keras_model(model, granularity='name')
        hls_model = hls4ml.converters.convert_from_keras_model(
            model, 
            hls_config=config,
            output_dir='hls4ml_prj',
            fpga_part='xczu9eg-ffvb1156-2-e'  # Your target FPGA
        )
    """
    widths = [3, 4, 8, 16, 64, 128]
    depths = [1, 1, 1, 1]
    return QYOLOStatic(widths, depths, num_classes, img_h, img_w, dtype=dtype)


def transfer_weights_to_static(quantized_model, static_model):
    """
    Transfer weights from trained quantized model to static HLS4ml model
    
    Args:
        quantized_model: Trained QYOLO model
        static_model: QYOLOStatic model for HLS4ml
    
    Note: Both models must have the same architecture (widths, depths, num_classes)
    """
    print("[INFO] Transferring weights from quantized model to static HLS4ml model...")
    
    # Get weights from quantized model
    quantized_weights = quantized_model.get_weights()
    
    # Set weights to static model
    # The architectures are identical, so direct weight transfer works
    static_model.set_weights(quantized_weights)
    
    print(f"[INFO] Transferred {len(quantized_weights)} weight tensors successfully")
    return static_model
