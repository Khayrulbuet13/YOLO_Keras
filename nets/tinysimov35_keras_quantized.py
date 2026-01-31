import tensorflow as tf
from tensorflow.keras import layers, Model
import math
from qkeras import QConv2D, QActivation, quantized_bits, quantized_relu

class QConv(layers.Layer):
    """Quantized Conv block with QKeras layers"""
    def __init__(self, in_ch, out_ch, k=1, s=1, p=None, d=1, g=1, 
                 weight_bits=8, activation_bits=8, dtype=tf.float32):
        super().__init__(dtype=dtype)
        self.dtype_ = dtype
        padding = 'same' if p is None else 'valid'
        
        # Quantized convolution
        # Use more bits for first layer (more sensitive)
        self.conv = QConv2D(
            out_ch, k, strides=s, 
            padding=padding,
            dilation_rate=d,
            groups=g,
            use_bias=False,
            kernel_quantizer=quantized_bits(weight_bits, 0, alpha=1),
            dtype=dtype
        )
        
        # Batch normalization (no quantization needed - fused with Conv in HLS4ml)
        self.norm = layers.BatchNormalization(
            epsilon=0.001, 
            momentum=0.03,
            dtype=dtype
        )
        
        # Quantized ReLU activation
        self.relu = QActivation(quantized_relu(activation_bits, 0), dtype=dtype)

    def call(self, x):
        return self.relu(self.norm(self.conv(x)))

class QDarkNet(Model):
    """Quantized DarkNet backbone"""
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
                
                # Use 16-bit for first layer (more sensitive to quantization)
                # Use 8-bit for middle layers
                # Use 12-bit for last layers (before head)
                if i == 0 and j == 0:
                    weight_bits, activation_bits = 16, 16
                elif i >= len(depths) - 1:
                    weight_bits, activation_bits = 12, 12
                else:
                    weight_bits, activation_bits = 8, 8
                
                self.layers_list.append(
                    QConv(in_ch, out_ch, 3, stride, 
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

class QDFL(layers.Layer):
    """Quantized Distribution Focal Loss layer
    
    Note: This layer uses softmax which is expensive on FPGA.
    For HLS4ml synthesis, consider replacing with simpler operations.
    """
    def __init__(self, ch=16, dtype=tf.float32):
        super().__init__(dtype=dtype)
        self.dtype_ = dtype
        self.ch = ch
        
        # Quantized convolution for DFL
        self.conv = QConv2D(
            1, 1, 
            use_bias=False, 
            kernel_initializer='zeros', 
            trainable=False,
            kernel_quantizer=quantized_bits(8, 0, alpha=1),
            dtype=dtype
        )
    
    def build(self, input_shape):
        kernel = tf.reshape(tf.range(self.ch, dtype=self.dtype_), [1, 1, self.ch, 1])
        self.conv.build(input_shape)
        self.conv.kernel.assign(kernel)
    
    def call(self, x):
        """PyTorch equivalent: sum(softmax(channel_axis) * [0..15])
        
        WARNING: Softmax is resource-intensive on FPGA.
        For production FPGA deployment, consider approximating or replacing.
        """
        x = tf.nn.softmax(x, axis=-1)
        return self.conv(x)

class QHead(Model):
    """Quantized detection head
    
    Training mode: Returns raw predictions for loss computation
    Inference mode: Returns processed detections (boxes + class scores)
    """
    def __init__(self, nc=20, ch_in=128, dtype=tf.float32):
        super().__init__(dtype=dtype)
        self.dtype_ = dtype
        self.nc = nc
        self.ch = 16
        self.no = nc + self.ch * 4
        
        # Quantized conv block before head
        self.conv = QConv(ch_in, 24, 1, 1, weight_bits=12, activation_bits=12, dtype=dtype)
        
        # DFL layer
        self.dfl = QDFL(self.ch, dtype=dtype)
        
        # Box and class prediction heads (quantized)
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

    def call(self, x, training=False):
        if training:
            # Training path: simple concatenation for loss computation
            return tf.concat([self.box(x), self.cls(x)], axis=-1)
        else:
            # Inference path with dynamic operations
            # NOTE: This path has operations that are problematic for HLS4ml:
            # - tf.range loops
            # - Dynamic reshaping
            # - tf.meshgrid
            # For FPGA synthesis, use the static inference model instead
            b = tf.shape(x)[0]
            
            box = self.box(x)
            cls = self.cls(x)
            
            box_flat = tf.reshape(box, [b, -1, 4 * self.ch])
            cls_flat = tf.reshape(cls, [b, -1, self.nc])
            
            box_processed = tf.reshape(box_flat, [b, -1, 4, self.ch])
            
            box_dfl = []
            for i in tf.range(4):
                coord = box_processed[:, :, i, :]
                coord = tf.expand_dims(coord, 2)
                dfl_out = self.dfl(coord)
                box_dfl.append(tf.squeeze(dfl_out, [2,3]))
                
            box_dfl = tf.stack(box_dfl, axis=2)
            
            h, w = tf.shape(x)[1], tf.shape(x)[2]
            grid_x = tf.range(w, dtype=self.dtype_) + 0.5
            grid_y = tf.range(h, dtype=self.dtype_) + 0.5
            grid = tf.stack(tf.meshgrid(grid_x, grid_y), axis=-1)
            anchors = tf.reshape(grid, [1, -1, 2])
            
            a, b_coords = tf.split(box_dfl, 2, axis=2)
            boxes = tf.concat([
                (anchors - a + anchors + b_coords) / 2,
                (a + b_coords)
            ], axis=-1)
            
            boxes = boxes * self.stride
            
            # WARNING: Sigmoid is expensive on FPGA
            return tf.concat([boxes, tf.sigmoid(cls_flat)], axis=-1)

class QYOLO(Model):
    """Quantized YOLO model for HLS4ml synthesis
    
    This model uses QKeras quantization-aware training.
    For FPGA deployment, export using the static inference path.
    """
    def __init__(self, widths=None, depths=None, num_classes=20, img_size=(256, 256), dtype=tf.float32):
        super().__init__(dtype=dtype)
        self.dtype_ = dtype
        self.net = QDarkNet(widths, depths, dtype=dtype)
        self.head = QHead(num_classes, ch_in=self.net.out_channels, dtype=dtype)
        
        # Convert img_size to (h, w, c) format
        if isinstance(img_size, int):
            img_size_with_channels = (img_size, img_size, 3)
        else:
            img_size_with_channels = (*img_size, 3)

        # Build layers by running a dummy forward pass
        dummy = tf.zeros((1, *img_size_with_channels), dtype=self.dtype_)
        _ = self.head(self.net(dummy), training=True)

        # Initialize strides
        self.stride = self.calculate_stride(img_size_with_channels)
        self.head.stride = self.stride
        
        # Initialize biases
        self.initialize_biases()
        
    def call(self, x, training=False):
        features = self.net(x)
        return self.head(features, training=training)
    
    def calculate_stride(self, img_size):
        dummy_img = tf.zeros((1, *img_size), dtype=self.dtype_)
        features = self.net(dummy_img)
        feature_h, feature_w = features.shape[1], features.shape[2]
        stride_h = img_size[0] / feature_h
        stride_w = img_size[1] / feature_w
        
        if not math.isclose(stride_h, stride_w, rel_tol=1e-5):
            print(f"Warning: Stride mismatch: h={stride_h}, w={stride_w}. Using average.")
            
        return tf.constant([(stride_h + stride_w) / 2], dtype=self.dtype_)
    
    def initialize_biases(self):
        s = self.stride[0].numpy()
        # Box bias
        if hasattr(self.head.box, 'bias') and self.head.box.bias is not None:
            self.head.box.bias.assign(tf.ones_like(self.head.box.bias))
        # Class bias
        if hasattr(self.head.cls, 'bias') and self.head.cls.bias is not None:
            b = self.head.cls.bias
            bias_value = tf.math.log(5 / self.head.nc / (640 / s) ** 2)
            b.assign(tf.ones_like(b) * tf.cast(bias_value, b.dtype))


def yolo_v8_s_quantized(num_classes: int = 20, img_size=(256, 256), dtype=tf.float32):
    """
    Quantized Small YOLO v8 model for HLS4ml FPGA synthesis
    
    Uses QKeras quantization-aware training with:
    - 16-bit for first layer (most sensitive)
    - 8-bit for middle layers (most efficient)
    - 12-bit for last layers and head (balance accuracy/resources)
    
    Args:
        num_classes: Number of classes to detect
        img_size: Input image size, can be int (square) or tuple (h, w)
        dtype: Data type for the model (default: tf.float32)
    
    Returns:
        Quantized YOLO model ready for training or HLS4ml conversion
    """
    widths = [3, 4, 8, 16, 64, 128]
    depths = [1, 1, 1, 1]
    return QYOLO(widths, depths, num_classes, img_size, dtype=dtype)
