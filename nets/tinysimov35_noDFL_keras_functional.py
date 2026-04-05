"""
YOLO26-style functional model: no DFL, no NMS (end-to-end).

Key differences from tinysimov35_keras_functional.py:
- Box head outputs 4 channels (direct LTRB distances) instead of 4*16 DFL bins
- Dual head: one-to-many (training) + one-to-one (inference)
- Inference uses top-k selection instead of IoU-NMS
- decode_predictions_noDFL: no softmax/integral, raw LTRB -> xywh
- postprocess_e2e: confidence filter + top-k, returns list of (K, 6) tensors

Architecture mirrors YOLO26 with reg_max=1 and end2end=True.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
import math
import numpy as np


def build_yolo_e2e_functional(num_classes=1, img_size=(256, 256), dtype=tf.float32):
    """
    Build YOLO end-to-end model (YOLO26-style) using Keras Functional API.

    Returns a model with two outputs:
        - outputs[0]: one-to-many head  (B, H, W, 4+nc) -- used in training
        - outputs[1]: one-to-one head   (B, H, W, 4+nc) -- used in inference

    Metadata attached to model:
        model.nc, model.dfl_ch (=1), model.no, model.stride
    """
    if isinstance(img_size, int):
        img_size = (img_size, img_size)

    h, w = img_size

    x_in = layers.Input(shape=(h, w, 3), dtype=dtype, name='input')
    x = x_in

    # --- Backbone (identical to functional model) ---
    x = layers.Conv2D(4, 3, strides=2, padding='same', use_bias=False,
                      dtype=dtype, name='backbone_conv1')(x)
    x = layers.BatchNormalization(epsilon=0.001, momentum=0.03,
                                   dtype=dtype, name='backbone_bn1')(x)
    x = layers.ReLU(dtype=dtype, name='backbone_relu1')(x)

    x = layers.Conv2D(8, 3, strides=2, padding='same', use_bias=False,
                      dtype=dtype, name='backbone_conv2')(x)
    x = layers.BatchNormalization(epsilon=0.001, momentum=0.03,
                                   dtype=dtype, name='backbone_bn2')(x)
    x = layers.ReLU(dtype=dtype, name='backbone_relu2')(x)

    x = layers.Conv2D(16, 3, strides=2, padding='same', use_bias=False,
                      dtype=dtype, name='backbone_conv3')(x)
    x = layers.BatchNormalization(epsilon=0.001, momentum=0.03,
                                   dtype=dtype, name='backbone_bn3')(x)
    x = layers.ReLU(dtype=dtype, name='backbone_relu3')(x)

    backbone_out = layers.Conv2D(64, 3, strides=1, padding='same', use_bias=False,
                                  dtype=dtype, name='backbone_conv4')(x)
    backbone_out = layers.BatchNormalization(epsilon=0.001, momentum=0.03,
                                              dtype=dtype, name='backbone_bn4')(backbone_out)
    backbone_out = layers.ReLU(dtype=dtype, name='backbone_relu4')(backbone_out)

    # --- One-to-many head (training branch) ---
    o2m_box = layers.Conv2D(4, 1, dtype=dtype, name='o2m_box_conv')(backbone_out)
    o2m_cls = layers.Conv2D(num_classes, 1, dtype=dtype, name='o2m_cls_conv')(backbone_out)
    o2m_output = layers.Concatenate(axis=-1, name='o2m_concat')([o2m_box, o2m_cls])

    # --- One-to-one head (inference branch, independent weights) ---
    # Stop-gradient on backbone features so backbone is trained only by o2m loss.
    # The o2o head itself still receives gradients from the o2o loss branch.
    backbone_detached = layers.Lambda(
        lambda x: tf.stop_gradient(x), name='o2o_detach'
    )(backbone_out)
    o2o_box = layers.Conv2D(4, 1, dtype=dtype, name='o2o_box_conv')(backbone_detached)
    o2o_cls = layers.Conv2D(num_classes, 1, dtype=dtype, name='o2o_cls_conv')(backbone_detached)
    o2o_output = layers.Concatenate(axis=-1, name='o2o_concat')([o2o_box, o2o_cls])

    model = Model(inputs=x_in, outputs=[o2m_output, o2o_output], name='yolo_e2e_functional')

    # Attach metadata (mirrors functional model convention, used by loss/inference)
    model.nc = num_classes
    model.dfl_ch = 1         # no DFL bins; kept for API compatibility
    model.no = num_classes + 4

    # Compute stride via dummy forward -- force CPU to avoid CuDNN version mismatches
    with tf.device('/CPU:0'):
        dummy = tf.zeros((1, h, w, 3), dtype=dtype)
        o2m_feat, _ = model(dummy, training=False)
    feat_h, feat_w = o2m_feat.shape[1], o2m_feat.shape[2]
    stride_h = h / feat_h
    stride_w = w / feat_w
    if not math.isclose(stride_h, stride_w, rel_tol=1e-5):
        print(f"Warning: stride mismatch h={stride_h}, w={stride_w}. Using average.")
    model.stride = tf.constant([(stride_h + stride_w) / 2], dtype=dtype)

    # Initialize biases using numpy/math (avoid GPU dispatch for scalar ops)
    s = float(model.stride[0].numpy())
    import math as _math
    cls_bias_val = float(_math.log(5 / num_classes / (640 / s) ** 2))
    for head_prefix in ('o2m', 'o2o'):
        box_layer = model.get_layer(f'{head_prefix}_box_conv')
        if box_layer.bias is not None:
            box_layer.bias.assign(tf.ones_like(box_layer.bias))
        cls_layer = model.get_layer(f'{head_prefix}_cls_conv')
        if cls_layer.bias is not None:
            import numpy as _np
            cls_layer.bias.assign(
                tf.constant(_np.full(cls_layer.bias.shape, cls_bias_val),
                            dtype=cls_layer.bias.dtype)
            )

    return model


def decode_predictions_noDFL(raw_output, stride, nc, dtype=tf.float32):
    """
    Decode raw one-to-one head output to bounding boxes (no DFL, no softmax).

    Args:
        raw_output: (B, H, W, 4+nc) -- first 4 channels are direct LTRB distances
        stride: scalar stride (feature map -> image pixels)
        nc: number of classes
        dtype: computation dtype

    Returns:
        (B, HW, 4+nc) where 4 = [cx, cy, w, h] in pixel coords, nc = sigmoid class scores
    """
    b = tf.shape(raw_output)[0]
    h = tf.shape(raw_output)[1]
    w = tf.shape(raw_output)[2]

    # Split box (LTRB) and class logits
    box_raw = raw_output[..., :4]    # (B, H, W, 4)
    cls_raw = raw_output[..., 4:]    # (B, H, W, nc)

    box_flat = tf.reshape(box_raw, [b, -1, 4])    # (B, HW, 4)
    cls_flat = tf.reshape(cls_raw, [b, -1, nc])   # (B, HW, nc)

    # Build anchor grid (feature-map coordinates + 0.5 offset)
    grid_x = tf.cast(tf.range(w), dtype) + 0.5
    grid_y = tf.cast(tf.range(h), dtype) + 0.5
    grid = tf.stack(tf.meshgrid(grid_x, grid_y, indexing='xy'), axis=-1)  # (H, W, 2)
    anchors = tf.reshape(grid, [1, -1, 2])  # (1, HW, 2)

    # Decode LTRB -> center+size (same dist2bbox logic as ultralytics)
    lt, rb = tf.split(box_flat, 2, axis=-1)        # each (B, HW, 2)
    boxes = tf.concat([
        (anchors - lt + anchors + rb) / 2,          # center (cx, cy)
        lt + rb                                       # size (w, h)
    ], axis=-1)  # (B, HW, 4)

    # Scale by stride to pixel coordinates
    if isinstance(stride, (int, float)):
        stride = tf.constant(stride, dtype=dtype)
    boxes = boxes * tf.cast(stride, dtype)

    cls_scores = tf.sigmoid(cls_flat)
    return tf.concat([boxes, cls_scores], axis=-1)  # (B, HW, 4+nc)


def postprocess_e2e(decoded_output, max_det=300, conf_threshold=0.25, dtype=tf.float32):
    """
    Top-k post-processing for end-to-end model (replaces IoU-NMS).

    Mirrors ultralytics Detect.postprocess / get_topk_index:
      1. Find max class score per anchor
      2. Select top-k anchors globally
      3. Confidence threshold filter

    Args:
        decoded_output: (B, HW, 4+nc) xywh + class scores
        max_det: maximum detections to keep
        conf_threshold: minimum confidence to keep after top-k
        dtype: computation dtype

    Returns:
        list of length B, each element is a numpy-able tensor (K, 6):
        [x1, y1, x2, y2, conf, cls_id]
    """
    decoded_output = tf.cast(decoded_output, dtype)
    batch_size = tf.shape(decoded_output)[0]
    num_anchors = tf.shape(decoded_output)[1]
    nc = decoded_output.shape[-1] - 4

    boxes_xywh = decoded_output[..., :4]   # (B, HW, 4)
    scores = decoded_output[..., 4:]        # (B, HW, nc)

    # Convert center-size to xyxy
    cx, cy = boxes_xywh[..., 0], boxes_xywh[..., 1]
    bw, bh = boxes_xywh[..., 2], boxes_xywh[..., 3]
    x1 = cx - bw / 2
    y1 = cy - bh / 2
    x2 = cx + bw / 2
    y2 = cy + bh / 2
    boxes_xyxy = tf.stack([x1, y1, x2, y2], axis=-1)  # (B, HW, 4)

    # Max score and class id per anchor
    max_scores = tf.reduce_max(scores, axis=-1)       # (B, HW)
    cls_ids = tf.cast(tf.argmax(scores, axis=-1), dtype)  # (B, HW)

    # Top-k selection
    k = tf.minimum(max_det, num_anchors)
    topk_vals, topk_idx = tf.math.top_k(max_scores, k=k)   # (B, k)

    # Gather boxes, scores, class ids per image
    batch_size_static = decoded_output.shape[0]
    outputs = []
    for b_i in range(batch_size_static if batch_size_static is not None else decoded_output.shape[0]):
        idx = topk_idx[b_i]                                      # (k,)
        sel_boxes = tf.gather(boxes_xyxy[b_i], idx)              # (k, 4)
        sel_conf = tf.gather(max_scores[b_i], idx)               # (k,)
        sel_cls = tf.gather(cls_ids[b_i], idx)                   # (k,)
        det = tf.concat([sel_boxes,
                         sel_conf[:, tf.newaxis],
                         sel_cls[:, tf.newaxis]], axis=-1)        # (k, 6)
        # Confidence filter
        keep = det[:, 4] > conf_threshold
        det = tf.boolean_mask(det, keep)
        outputs.append(det)

    return outputs


def postprocess_e2e_dynamic(decoded_output, max_det=300, conf_threshold=0.25, dtype=tf.float32):
    """
    Dynamic-batch version of postprocess_e2e (uses tf.while_loop).
    Useful when batch size is unknown at graph-build time.
    """
    decoded_output = tf.cast(decoded_output, dtype)
    nc = decoded_output.shape[-1] - 4

    boxes_xywh = decoded_output[..., :4]
    scores = decoded_output[..., 4:]

    cx, cy = boxes_xywh[..., 0], boxes_xywh[..., 1]
    bw, bh = boxes_xywh[..., 2], boxes_xywh[..., 3]
    boxes_xyxy = tf.stack([cx - bw/2, cy - bh/2, cx + bw/2, cy + bh/2], axis=-1)

    max_scores = tf.reduce_max(scores, axis=-1)
    cls_ids = tf.cast(tf.argmax(scores, axis=-1), dtype)

    num_anchors = tf.shape(decoded_output)[1]
    k = tf.minimum(max_det, num_anchors)
    topk_vals, topk_idx = tf.math.top_k(max_scores, k=k)

    batch_size = tf.shape(decoded_output)[0]
    outputs = []
    for b_i in tf.range(batch_size):
        idx = topk_idx[b_i]
        sel_boxes = tf.gather(boxes_xyxy[b_i], idx)
        sel_conf = tf.gather(max_scores[b_i], idx)
        sel_cls = tf.gather(cls_ids[b_i], idx)
        det = tf.concat([sel_boxes, sel_conf[:, tf.newaxis], sel_cls[:, tf.newaxis]], axis=-1)
        keep = det[:, 4] > conf_threshold
        outputs.append(tf.boolean_mask(det, keep))

    return outputs


def yolo_v8_s_e2e(num_classes=1, img_size=(256, 256), dtype=tf.float32):
    """Factory alias matching main_keras.py naming convention."""
    return build_yolo_e2e_functional(num_classes, img_size, dtype)
