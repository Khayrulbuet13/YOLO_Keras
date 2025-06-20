import tensorflow as tf
from tensorflow.keras import layers, Model
import math

class Conv(layers.Layer):
    def __init__(self, in_ch, out_ch, k=1, s=1, p=None, d=1, g=1, dtype=tf.float32):
        super().__init__(dtype=dtype)
        self.dtype_ = dtype  # Store as different attribute name
        padding = 'same' if p is None else 'valid'
        self.conv = layers.Conv2D(
            out_ch, k, strides=s, 
            padding=padding,
            dilation_rate=d,
            groups=g,
            use_bias=False,
            dtype=dtype
        )
        self.norm = layers.BatchNormalization(epsilon=0.001, momentum=0.03, dtype=dtype)
        self.relu = layers.ReLU(dtype=dtype)

    def call(self, x):
        return self.relu(self.norm(self.conv(x)))

class DarkNet(Model):
    def __init__(self, widths=None, depths=None, dtype=tf.float32):
        super().__init__(dtype=dtype)
        self.dtype_ = dtype  # Store as different attribute name
        self.layers_list = []
        in_ch = widths[0]
        
        for i in range(min(len(depths), len(widths) - 1)):
            out_ch = widths[i + 1]
            nblocks = depths[i]
            
            for j in range(nblocks):
                stride = 2 if (j == nblocks - 1 and i < len(depths) - 1) else 1
                self.layers_list.append(Conv(in_ch, out_ch, 3, stride, dtype=dtype))
                in_ch = out_ch
        
        self.out_channels = in_ch

    def call(self, x):
        for layer in self.layers_list:
            x = layer(x)
        return x

# class DFL(layers.Layer):
#     def __init__(self, ch=16):
#         super().__init__()
#         self.ch = ch
#         self.conv = layers.Conv2D(
#             filters=ch, 
#             kernel_size=1, 
#             padding='same', 
#             use_bias=False,
#             trainable=False  # Non-trainable weights
#         )

#     def build(self, input_shape):
#         super().build(input_shape)
#         in_channels = input_shape[-1]
        
#         # Create fixed weights: [0, 1, 2, ..., ch-1] repeated for each input channel
#         kernel = tf.range(self.ch, dtype=tf.float32)
#         kernel = tf.reshape(kernel, [1, 1, 1, self.ch])
#         kernel = tf.repeat(kernel, repeats=in_channels, axis=2)
        
#         # Set weights after building conv layer
#         self.conv.build(input_shape)
#         self.conv.set_weights([kernel])

#     def call(self, x):
#         return self.conv(x)

class DFL(layers.Layer):
    def __init__(self, ch=16, dtype=tf.float32):
        super().__init__(dtype=dtype)
        self.dtype_ = dtype  # Store as different attribute name
        self.ch = ch
        self.conv = layers.Conv2D(1, 1, use_bias=False, kernel_initializer='zeros', trainable=False, dtype=dtype)
    
    def build(self, input_shape):
        kernel = tf.reshape(tf.range(self.ch, dtype=self.dtype_), [1, 1, self.ch, 1])  # (1,1,16,1)
        self.conv.build(input_shape)
        self.conv.kernel.assign(kernel)
    
    def call(self, x):
        """PyTorch equivalent: sum(softmax(channel_axis) * [0..15])"""
        x = tf.nn.softmax(x, axis=-1)
        return self.conv(x) 



class Head(Model):
    def __init__(self, nc=20, ch_in=128, dtype=tf.float32):
        super().__init__(dtype=dtype)
        self.dtype_ = dtype  # Store as different attribute name
        self.nc = nc
        self.ch = 16
        self.no = nc + self.ch * 4  # 65
        self.conv = Conv(ch_in, 24, 1, 1, dtype=dtype)
        self.dfl = DFL(self.ch, dtype=dtype)
        self.box = layers.Conv2D(4 * self.ch, 1, dtype=dtype)
        self.cls = layers.Conv2D(nc, 1, dtype=dtype)

    def call(self, x, training=False):
        if training:
            return tf.concat([self.box(x), self.cls(x)], axis=-1)  # Normal training path
        else:
            # Inference: Match PyTorch's reshape flow
            b = tf.shape(x)[0]
            
            # Get predictions
            box = self.box(x)  # (B, H, W, 64)
            cls = self.cls(x)  # (B, H, W, 1)
            
            # Reshape to anchors-first format (B, HW, 64)
            box_flat = tf.reshape(box, [b, -1, 4 * self.ch])  # (B, H*W, 64)
            cls_flat = tf.reshape(cls, [b, -1, self.nc])      # (B, H*W, 1)
            
            # Process box coordinates (PyTorch eq: box.view(b,4,self.ch,-1))
            box_processed = tf.reshape(box_flat, [b, -1, 4, self.ch])  # (B, HW, 4, 16)
            
            # Apply DFL per coordinate (split is implicit in reshape)
            box_dfl = []
            for i in tf.range(4):
                coord = box_processed[:, :, i, :]  # (B, HW, 16)
                coord = tf.expand_dims(coord, 2)   # Add dummy width dim (B, HW, 1, 16)
                dfl_out = self.dfl(coord)           # (B, HW, 1, 1) 
                box_dfl.append(tf.squeeze(dfl_out, [2,3]))  # (B, HW)
                
            box_dfl = tf.stack(box_dfl, axis=2)  # (B, HW, 4)
            
            # Generate anchors (different from training)
            h, w = tf.shape(x)[1], tf.shape(x)[2]
            grid_x = (tf.range(w, dtype=self.dtype_) + 0.5) * self.stride
            grid_y = (tf.range(h, dtype=self.dtype_) + 0.5) * self.stride
            grid = tf.stack(tf.meshgrid(grid_x, grid_y), axis=-1)  # (H, W, 2)
            anchors = tf.reshape(grid, [1, -1, 2])  # (1, HW, 2)
            
            # Calculate final boxes
            a, b_coords = tf.split(box_dfl, 2, axis=2)  # 2x (B, HW, 2)
            boxes = tf.concat([
                (anchors - a + anchors + b_coords) / 2,  # (x_center, y_center)
                (b_coords - a)                           # (width, height)
            ], axis=-1)
            
            return tf.concat([boxes, tf.sigmoid(cls_flat)], axis=-1)  # (B, HW, 5)




class YOLO(Model):
    def __init__(self, widths=None, depths=None, num_classes=20, img_size=(256, 256), dtype=tf.float32):
        super().__init__(dtype=dtype)
        self.dtype_ = dtype  # Store as different attribute name
        self.net = DarkNet(widths, depths, dtype=dtype)
        self.head = Head(num_classes, ch_in=self.net.out_channels, dtype=dtype)
        
        # Convert img_size to (h, w, c) format
        if isinstance(img_size, int):
            img_size_with_channels = (img_size, img_size, 3)
        else:
            img_size_with_channels = (*img_size, 3)

        # Build layers by running a dummy forward pass
        dummy = tf.zeros((1, *img_size_with_channels), dtype=self.dtype_)
        _ = self.head(self.net(dummy), training=True)  # build layers

        # Initialize strides
        self.stride = self.calculate_stride(img_size_with_channels)
        self.head.stride = self.stride

        # Initialize biases now that layers exist
        self.initialize_biases()

        # Debug info about initialized biases and initial logits
        print("[nets/tinysimov35_keras.YOLO.__init__] Box bias mean:",
              float(tf.reduce_mean(self.head.box.bias)))
        print("[nets/tinysimov35_keras.YOLO.__init__] Cls bias mean:",
              float(tf.reduce_mean(self.head.cls.bias)))
        logits = self.head(self.net(dummy))
        print(
            f"[nets/tinysimov35_keras.YOLO.__init__] Initial logits min/max: {float(tf.reduce_min(logits)):.4f}/{float(tf.reduce_max(logits)):.4f}"
        )
        
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
            print("[nets/tinysimov35_keras.initialize_biases] Box bias initialized")
        # Class bias
        if hasattr(self.head.cls, 'bias') and self.head.cls.bias is not None:
            b = self.head.cls.bias
            bias_value = tf.math.log(5 / self.head.nc / (640 / s) ** 2)
            b.assign(tf.ones_like(b) * tf.cast(bias_value, b.dtype))
            print("[nets/tinysimov35_keras.initialize_biases] Cls bias initialized")
    
    
    # def initialize_biases(self):
    #     s = self.stride[0].numpy()
    #     if self.head.box.bias is not None:
    #         self.head.box.bias.assign(tf.ones_like(self.head.box.bias))
    #     if self.head.cls.bias is not None:
    #         # Use Keras-compatible initializer
    #         b_init = tf.math.log(5 / self.head.nc / (640 / s) ** 2)
    #         self.head.cls.bias.assign(tf.ones_like(self.head.cls.bias) * b_init)

def yolo_v8_s(num_classes: int = 20, img_size=(256, 256), dtype=tf.float32):
    """
    Small YOLO v8 model
    
    Args:
        num_classes: Number of classes to detect
        img_size: Input image size, can be int (square) or tuple (h, w) for rectangular
        dtype: Data type for the model (default: tf.float32)
    """
    widths = [3, 4, 8, 16, 64, 128]
    depths = [1, 1, 1, 1]
    return YOLO(widths, depths, num_classes, img_size, dtype=dtype)
