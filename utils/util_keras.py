import tensorflow as tf
import numpy as np
import math
import random
import time
import os
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.image import non_max_suppression as tf_nms
from tensorflow import one_hot

# Define global dtype for consistent type handling
DTYPE = tf.float32


# utils/util_keras.py  (put it near the top, after imports)
def cast_like(tensor, ref):
    """Return `tensor` cast to the dtype of `ref` if they differ."""
    # Handle case where ref is a list of tensors
    if isinstance(ref, (list, tuple)) and len(ref) > 0:
        ref = ref[0]  # Use the first tensor in the list
    return tf.cast(tensor, ref.dtype) if tensor.dtype != ref.dtype else tensor


def generate_colors(num_classes):
    """
    Helper function to generate random colors for each class.
    Ensures colors are created for all possible class IDs up to num_classes.
    """
    np.random.seed(42)  # For consistent colors
    colors = {}
    for i in range(num_classes):
        colors[i] = tuple(np.random.randint(0, 256, 3).tolist())
    return colors




def visualize_predictions(samples, outputs, gt_boxes, shapes, params, class_colors, results_dir, index):
    """
    Visualize first 5 images from the val set: draws ground-truth boxes on the
    left, predicted boxes on the right, and saves side-by-side images.
    """
    # Convert to central dtype
    samples = tf.cast(samples, DTYPE)
    # Convert single image to CPU numpy
    img = samples[0].numpy()

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = (img * 255).astype(np.uint8)

    # Remove padding, resize back
    original_h, original_w = shapes[0][0].numpy()
    pad = shapes[0][1].numpy() 
    pad_h, pad_w = pad[0], pad[1]  # Assuming pad is [pad_h, pad_w]
    
    h, w = img.shape[:2]
    img = img[int(pad_h):int(h - pad_h), int(pad_w):int(w - pad_w)]
    img = cv2.resize(img, (int(original_w), int(original_h)))

    img_gt = img.copy()
    img_pred = img.copy()

    # Draw GT
    if tf.size(gt_boxes) > 0:
        gt_boxes_np = gt_boxes.numpy()
        
        for i, gt in enumerate(gt_boxes_np):
            # Check if this is a valid box (non-zero)
            if np.sum(gt) == 0:
                continue
                
            cls_gt = int(gt[1])  # Class is at index 1 in Keras format [img_idx, cls, x, y, w, h]
            coords = gt[2:]  # Get coordinates [x, y, w, h]
            x_c, y_c = coords[0], coords[1]
            w_b, h_b = coords[2], coords[3]

            # Convert to corner coordinates
            x1 = int(x_c - w_b / 2)
            y1 = int(y_c - h_b / 2)
            x2 = int(x_c + w_b / 2)
            y2 = int(y_c + h_b / 2)
            
            # Check if coordinates are valid
            if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0:
                continue
                
            if cls_gt in class_colors:
                color = tuple(map(int, class_colors[cls_gt]))
            else:
                color = (0, 255, 0)  # Default green
                
            cv2.rectangle(img_gt, (x1, y1), (x2, y2), color, 2)
            
            if cls_gt in params['names']:
                cls_name = params['names'][cls_gt]
            else:
                cls_name = str(cls_gt)
            cv2.putText(img_gt, cls_name, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Draw predictions
    if outputs[0] is not None:
        # Scale predictions using util.scale like in test function
        det_clone = outputs[0].numpy().copy()
        
        # Convert Keras shapes format to PyTorch format for scale function
        # Keras: shapes[0]=[orig_h, orig_w], shapes[1]=[r, r], shapes[2]=[pad_w, pad_h]
        # PyTorch: shapes[0][0]=original_shape, shapes[0][1]=[[ratio, ratio], [pad_w, pad_h]]
        original_shape = shapes[0][0].numpy()  # [orig_h, orig_w]
        ratio_info = shapes[0][1].numpy()      # [r, r]
        pad_info = shapes[0][2].numpy()        # [pad_w, pad_h]
        pytorch_format = [[ratio_info[0], ratio_info[1]], [pad_info[0], pad_info[1]]]
        
        scale(det_clone[:, :4], samples[0].shape[1:], original_shape, pytorch_format)
        
        for i, det in enumerate(det_clone):
            if len(det) < 6:
                continue
                
            x1, y1, x2, y2, conf, cls_id = det
            cls_id = int(cls_id)
            
            # Check if coordinates are valid
            if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0:
                continue
                
            if cls_id in class_colors:
                color = tuple(map(int, class_colors[cls_id]))
            else:
                color = (255, 0, 0)  # Default red
                
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            
            if cls_id in params['names']:
                cls_name = params['names'][cls_id]
            else:
                cls_name = str(cls_id)
            
            cv2.rectangle(img_pred, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img_pred, f"{cls_name} {conf:.2f}",
                        (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Save side-by-side
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB))
    plt.title('Ground Truth')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(img_pred, cv2.COLOR_BGR2RGB))
    plt.title('Predictions')
    plt.axis('off')

    save_path = os.path.join(results_dir, f'result_{index}.png')
    plt.savefig(save_path)
    plt.close()




def setup_seed(seed=0):
    """Setup random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['PYTHONHASHSEED'] = str(seed)

def make_anchors_tf(x, strides, offset=0.5):
    """
    Generate anchors from features (TensorFlow version)
    """
    assert x is not None and isinstance(x, (list, tuple)) and len(x) > 0, "Input x must be a non-empty list of feature maps"
    anchor_points, stride_tensor = [], []
    for i, stride in enumerate(strides):
        _, _, h, w = x[i].shape
        sx = tf.range(w, dtype=x[i].dtype) + offset  # shift x
        sy = tf.range(h, dtype=x[i].dtype) + offset  # shift y
        sy, sx = tf.meshgrid(sy, sx, indexing='ij')
        anchors = tf.stack([sx, sy], axis=-1)
        anchor_points.append(tf.reshape(anchors, [-1, 2]))
        stride_tensor.append(tf.fill((h * w, 1), stride))
    return tf.concat(anchor_points, axis=0), tf.concat(stride_tensor, axis=0)


def scale(coords, shape1, shape2, ratio_pad=None):
    """Scale bounding box coordinates from resized image back to original"""
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(shape1[0] / shape2[0], shape1[1] / shape2[1])  # gain  = old / new
        pad = (shape1[1] - shape2[1] * gain) / 2, (shape1[0] - shape2[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain

    # Use numpy clip instead of PyTorch clamp_
    coords[:, 0] = np.clip(coords[:, 0], 0, shape2[1])  # x1
    coords[:, 1] = np.clip(coords[:, 1], 0, shape2[0])  # y1
    coords[:, 2] = np.clip(coords[:, 2], 0, shape2[1])  # x2
    coords[:, 3] = np.clip(coords[:, 3], 0, shape2[0])  # y2
    return coords

def box_iou(box1, box2):
    """
    Compute IoU between two sets of boxes (xyxy format)
    box1: [N, 4]
    box2: [M, 4]
    Returns: [N, M] matrix of IoU values
    """
    # Compute intersection
    b1 = tf.expand_dims(box1, 1)
    b2 = tf.expand_dims(box2, 0)
    
    inter_xmin = tf.maximum(b1[..., 0], b2[..., 0])
    inter_ymin = tf.maximum(b1[..., 1], b2[..., 1])
    inter_xmax = tf.minimum(b1[..., 2], b2[..., 2])
    inter_ymax = tf.minimum(b1[..., 3], b2[..., 3])
    
    inter_w = tf.maximum(inter_xmax - inter_xmin, 0)
    inter_h = tf.maximum(inter_ymax - inter_ymin, 0)
    intersection = inter_w * inter_h
    
    # Compute areas
    area1 = (b1[..., 2] - b1[..., 0]) * (b1[..., 3] - b1[..., 1])
    area2 = (b2[..., 2] - b2[..., 0]) * (b2[..., 3] - b2[..., 1])
    
    union = area1 + area2 - intersection
    iou = intersection / (union + 1e-7)
    return iou

def wh2xy(x):
    """Convert [x_center, y_center, width, height] to [x1, y1, x2, y2]"""
    y = tf.identity(x)
    y = tf.concat([
        x[..., 0:1] - x[..., 2:3] / 2,  # x1
        x[..., 1:2] - x[..., 3:4] / 2,  # y1
        x[..., 0:1] + x[..., 2:3] / 2,  # x2
        x[..., 1:2] + x[..., 3:4] / 2   # y2
    ], axis=-1)
    return y




def non_max_suppression(prediction, conf_threshold=0.25, iou_threshold=0.45):
    """
    Performs Non-Maximum Suppression (NMS) on inference results
    
    Args:
        prediction: tensor of shape [batch_size, num_boxes, num_classes + 5]
                    where 5 represents (x, y, w, h, obj_conf)
        conf_threshold: confidence threshold
        iou_threshold: IoU threshold for NMS
        
    Returns:
        list of detections, on (n,6) tensor per image [x1, y1, x2, y2, conf, cls]
    """
    # Add dtype consistency
    prediction = tf.cast(prediction, DTYPE)
    print(f"\n--- [utils/util_keras.py::non_max_suppression] KERAS NMS DEBUG ---")
    print(f"[utils/util_keras.py::non_max_suppression] Prediction shape: {prediction.shape}")
    print(f"[utils/util_keras.py::non_max_suppression] Conf threshold: {conf_threshold}, IoU threshold: {iou_threshold}")
    
    # Settings
    max_wh = 7680  # (pixels) maximum box width and height
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into TF NMS
    
    # nc = prediction.shape[1] - 4  # number of classes
    # prediction shape is (batch, num_boxes, num_classes + 4)
    # Use the last dimension to determine number of classes
    nc = prediction.shape[-1] - 4  # number of classes
    print(f"[utils/util_keras.py::non_max_suppression] Number of classes: {nc}")
    
    
    start = time.time()
    batch_size = prediction.shape[0]
    outputs = [tf.zeros((0, 6), dtype=prediction.dtype)] * batch_size
    
    # Process each image in batch
    for index in range(batch_size):
        # Get image predictions
        x = prediction[index]
        
        # Apply confidence threshold on class scores
        class_scores = x[:, 4:4+nc]
        xc = tf.reduce_max(class_scores, axis=1) > conf_threshold
        x = tf.boolean_mask(x, xc)
        
        # If no boxes remain, skip this image
        if tf.shape(x)[0] == 0:
            continue
            
        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = x[:, :4]
        cls = x[:, 4:4+nc]
        box = wh2xy(box)
        
        if nc > 1:
            # Multiple classes
            # Get indices where class confidence > threshold
            class_indices = tf.where(cls > conf_threshold)
            boxes_i = tf.gather_nd(box, class_indices[:, 0:1])
            scores_i = tf.gather_nd(class_scores, class_indices)
            classes_i = tf.cast(class_indices[:, 1:2], tf.float32)
            
            # Combine as [boxes, scores, classes]
            x = tf.concat([boxes_i, scores_i[:, None], classes_i], axis=1)
        else:
            # Single class - get the max confidence
            conf = tf.reduce_max(cls, axis=1, keepdims=True)
            j = tf.argmax(cls, axis=1, output_type=tf.float32)[:, None]
            
            # Filter by confidence threshold
            mask = tf.squeeze(conf > conf_threshold)
            x = tf.concat([box, conf, j], axis=1)
            x = tf.boolean_mask(x, mask)
        
        # Check if we have any boxes
        if tf.shape(x)[0] == 0:
            continue
            
        # Sort by confidence (descending) and limit to max_nms
        x = tf.gather(x, tf.argsort(x[:, 4], direction='DESCENDING'))
        x = x[:max_nms]
        
        # Apply NMS (using TensorFlow's built-in NMS)
        # For TF NMS, we need separate tensors for boxes and scores
        nms_boxes = x[:, :4]
        nms_scores = x[:, 4]
        
        # NMS needs float32 boxes (TensorFlow quirk)
        nms_boxes = cast_like(nms_boxes, tf.constant(0., dtype=tf.float32))
        nms_scores = cast_like(nms_scores, tf.constant(0., dtype=tf.float32))
        
        # Add offset to boxes based on class for class-aware NMS
        c = x[:, 5:6] * max_wh
        nms_boxes_with_offset = nms_boxes + c
        
        # Perform NMS
        selected_indices = tf.image.non_max_suppression(
            nms_boxes_with_offset, 
            nms_scores, 
            max_det, 
            iou_threshold=iou_threshold
        )
        
        # Get selected predictions
        x_filtered = tf.gather(x, selected_indices)
        outputs[index] = x_filtered
        
        # Check if time limit exceeded
        if (time.time() - start) > 0.5 + 0.05 * batch_size:
            print(f"WARNING ⚠️ NMS time limit {0.5 + 0.05 * batch_size:.3f}s exceeded")
            break
    
    return outputs
def smooth(y, f=0.05):
    """Smooth a 1D array using a box filter"""
    nf = int(round(len(y) * f * 2) // 2) + 1  # filter size (odd)
    p = np.ones(nf // 2)  # padding
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # padded y
    return np.convolve(yp, np.ones(nf) / nf, mode='valid')  # smoothed



def compute_ap(tp, conf, pred_cls, target_cls, eps=1e-16):
    """
    Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Object-ness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
    # Returns
        The average precision
    """
    
    # Sort by object-ness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # number of classes

    # Create Precision-Recall curve and compute AP for each class
    p = np.zeros((nc, 1000))
    r = np.zeros((nc, 1000))
    ap = np.zeros((nc, tp.shape[1]))
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        nl = nt[ci]  # number of labels
        no = i.sum()  # number of outputs
        if no == 0 or nl == 0:
            continue

        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        # Recall
        recall = tpc / (nl + eps)  # recall curve
        r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)

        # Precision
        precision = tpc / (tpc + fpc)  # precision curve
        p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)

        # AP from recall-precision curve
        for j in range(tp.shape[1]):
            m_rec = np.concatenate(([0.0], recall[:, j], [1.0]))
            m_pre = np.concatenate(([1.0], precision[:, j], [0.0]))
            m_pre = np.flip(np.maximum.accumulate(np.flip(m_pre)))
            x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
            ap[ci, j] = np.trapz(np.interp(x, m_rec, m_pre), x)

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + eps)
    i = smooth(f1.mean(0), 0.1).argmax()  # max F1 index
    p, r, f1 = p[:, i], r[:, i], f1[:, i]
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
    m_pre, m_rec = p.mean(), r.mean()
    map50, mean_ap = ap50.mean(), ap.mean()
    return tp, fp, m_pre, m_rec, map50, mean_ap

def clip_gradients(model, max_norm=10.0):
    """Clip gradients during training"""
    optimizer = model.optimizer
    if hasattr(optimizer, 'clipnorm'):
        optimizer.clipnorm = max_norm
    elif hasattr(optimizer, 'clipvalue'):
        optimizer.clipvalue = max_norm

class EMA:
    """Exponential Moving Average for model weights"""
    def __init__(self, model, decay=0.9999, tau=2000):
        self.model = model
        self.decay_fn = lambda x: decay * (1 - math.exp(-x / tau))
        self.updates = 0
        self.ema_weights = [tf.Variable(w, dtype=model.dtype) for w in model.get_weights()]
        
    def update(self):
        self.updates += 1
        d = self.decay_fn(self.updates)
        model_weights = self.model.get_weights()
        
        for i, w in enumerate(self.ema_weights):
            if tf.as_dtype(w.dtype).is_floating:
                w.assign(w * d + (1 - d) * model_weights[i])
    
    def set_weights_to_model(self, model=None):
        """Apply EMA weights to the model"""
        target_model = model if model else self.model
        target_model.set_weights([w.numpy() for w in self.ema_weights])
        
    def restore(self):
        """Restore original model weights"""
        self.ema_weights = [tf.Variable(w) for w in self.model.get_weights()]


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
import tensorflow as tf
import math



# import tensorflow as tf
# import numpy as np
# import math
from tensorflow.keras.layers import Layer


class ComputeLoss(Layer):
    def __init__(self, model, params, dtype=tf.float32, **kwargs):
        super(ComputeLoss, self).__init__(dtype=dtype, **kwargs)  # Set layer dtype
        self.dtype_ = dtype  # Store as different attribute name
        self.params = params
        
        # Extract parameters from model head
        m = model.layers[-1] if hasattr(model, 'layers') else model.head
        
        self.stride = m.stride
        if not isinstance(self.stride, (list, tuple)):
            self.stride = [self.stride]
        self.stride = [float(s) for s in self.stride]
        
        
        self.nc = m.nc  # number of classes
        self.no = m.no
        self.dfl_ch = m.dfl.ch
        
        # Task-aligned assigner config
        self.top_k = 10
        self.alpha = 0.5
        self.beta = 6.0
        self.eps = 1e-9
        
        # DFL project
        self.project = tf.range(self.dfl_ch, dtype=self.dtype_)
        
        # Initialize counters
        self.bs = 1
        self.num_max_boxes = 0

    def call(self, outputs, targets):
        # Ensure consistent typing
        outputs = tf.cast(outputs, self.dtype_)
        targets = tf.cast(targets, self.dtype_)

        # ensure we always have a *list* of feature maps
        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]

        # convert every map to NCHW and collect
        x = [tf.transpose(o, [0, 3, 1, 2]) for o in outputs]
        
        # Debugging outputs
        print(f"\n--- [utils/util.py::ComputeLoss.__call__] KERAS LOSS COMPUTATION DEBUG ---")
        print(f"[utils/util.py::ComputeLoss.__call__] Number of output tensors: {len(outputs)}")
        for i, o in enumerate(outputs):
            print(f"[utils/util.py::ComputeLoss.__call__] Output {i} shape: {o.shape}")
        output = tf.concat([tf.reshape(i, (i.shape[0], self.no, -1)) for i in x], axis=2)
        print(f"[utils/util.py::ComputeLoss.__call__] Concatenated output shape: {output.shape}")
        
        pred_output, pred_scores = tf.split(output, [4 * self.dfl_ch, self.nc], axis=1)
        print(f"[utils/util.py::ComputeLoss.__call__] pred_output shape: {pred_output.shape}")
        print(f"[utils/util.py::ComputeLoss.__call__] pred_scores shape: {pred_scores.shape}")
        
        pred_output = tf.transpose(pred_output, [0, 2, 1])
        pred_scores = tf.transpose(pred_scores, [0, 2, 1])
        print(f"[utils/util.py::ComputeLoss.__call__] After permute - pred_output shape: {pred_output.shape}")
        print(f"[utils/util.py::ComputeLoss.__call__] After permute - pred_scores shape: {pred_scores.shape}")
        
        
        # The feature maps have been converted to NCHW format above so the
        # spatial dimensions are at indices 2 and 3. Use these to compute the
        # feature map size (height, width) before reversing to (width, height).
        hw = tf.cast(tf.shape(x[0])[2:4], dtype=pred_scores.dtype)  # (h, w)
        # size = tf.reverse(hw, axis=[0]) * self.stride[0]  # (w, h) * stride
        size = hw * self.stride[0]  # (h, w) * stride
        print(f"[utils/util.py::ComputeLoss.__call__] Size tensor: {size}")
        
        anchor_points, stride_tensor = make_anchors_tf(x, self.stride, 0.5)
        # Use cast_like for critical operations
        anchor_points = cast_like(anchor_points, outputs)
        stride_tensor = cast_like(stride_tensor, outputs)
        print(f"[utils/util.py::ComputeLoss.__call__] Anchor points shape: {anchor_points.shape}")
        print(f"[utils/util.py::ComputeLoss.__call__] Stride tensor shape: {stride_tensor.shape}")

        
        # Process targets
        print(f"\n--- [utils/util.py::ComputeLoss.__call__] TARGET PROCESSING ---")
        print(f"[utils/util.py::ComputeLoss.__call__] Targets shape: {targets.shape}")
        
        if tf.shape(targets)[0] == 0:
            gt = tf.zeros((pred_scores.shape[0], 0, 5), dtype=pred_scores.dtype)
            print(f"[utils/util.py::ComputeLoss.__call__] No targets - gt shape: {gt.shape}")
        else:
            i = targets[:, 0]  # image index
            _, _, counts = tf.unique_with_counts(i)
            max_count = tf.reduce_max(counts)
            gt = tf.zeros((pred_scores.shape[0], max_count, 5), dtype=pred_scores.dtype)
            
            for j in range(pred_scores.shape[0]):
                matches = tf.equal(i, j)
                n = tf.reduce_sum(tf.cast(matches, tf.int32))
                if n > 0:
                    selected = tf.boolean_mask(targets, matches)[:, 1:]
                    padded = tf.pad(selected, [[0, max_count - n], [0, 0]])  # shape: [max_count, 5]
                    padded = tf.expand_dims(padded, axis=0)                 # shape: [1, max_count, 5]
                    gt = tf.tensor_scatter_nd_update(gt, [[j]], padded)

            # Convert WH to XYXY
            boxes = gt[..., 1:5] * tf.concat([size[1:2], size[0:1], size[1:2], size[0:1]], axis=0)
            gt_boxes = wh2xy(boxes)
            gt = tf.concat([gt[..., 0:1], gt_boxes], axis=-1)
        
        gt_labels, gt_bboxes = tf.split(gt, [1, 4], axis=2)
        mask_gt = tf.reduce_sum(gt_bboxes, axis=2, keepdims=True) > 0
        
        # Box decoding
        b, a, c = pred_output.shape
        print(f"[utils/util.py::ComputeLoss.__call__] Pred output dimensions - b: {b}, a: {a}, c: {c}")
        pred_bboxes = tf.reshape(pred_output, (b, a, 4, c // 4))
        pred_bboxes = tf.nn.softmax(pred_bboxes, axis=3)
        print(f"[utils/util.py::ComputeLoss.__call__] Pred bboxes after softmax shape: {pred_bboxes.shape}")
        pred_bboxes = tf.tensordot(pred_bboxes, self.project, axes=[[3], [0]])
        print(f"[utils/util.py::ComputeLoss.__call__] Pred bboxes after matmul shape: {pred_bboxes.shape}")
        
        a_tensor, b_tensor = tf.split(pred_bboxes, 2, axis=-1)
        pred_bboxes = tf.concat([anchor_points - a_tensor, anchor_points + b_tensor], axis=-1)
        print(f"[utils/util.py::ComputeLoss.__call__] Final pred bboxes shape: {pred_bboxes.shape}")
        
        # Detach for assignment (stop gradient)
        scores = tf.stop_gradient(tf.sigmoid(pred_scores))
        bboxes = tf.stop_gradient(pred_bboxes * stride_tensor)
        
        print(f"[utils/util.py::ComputeLoss.__call__] Scores shape: {scores.shape}")
        print(f"[utils/util.py::ComputeLoss.__call__] Bboxes shape: {bboxes.shape}")
        
        # Task-aligned assignment
        print(f"\n--- [utils/util.py::ComputeLoss.__call__] ASSIGNMENT PHASE ---")
        target_bboxes, target_scores, fg_mask = self.assign(
            scores, bboxes, gt_labels, gt_bboxes, mask_gt, anchor_points * stride_tensor
        )
        target_bboxes /= stride_tensor
        target_scores_sum = tf.reduce_sum(target_scores)
        
        # Classification loss
        loss_cls = tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=target_scores, logits=pred_scores)
        ) / tf.maximum(target_scores_sum, 1)
        
        # Box and DFL losses
        loss_box = tf.zeros(1, dtype=tf.float32)
        loss_dfl = tf.zeros(1, dtype=tf.float32)
        fg_count = tf.reduce_sum(tf.cast(fg_mask, tf.float32))
        
        if fg_count > 0:
            # IoU loss
            weight = tf.boolean_mask(tf.reduce_sum(target_scores, axis=-1), fg_mask)[:, tf.newaxis]
            iou = self.compute_iou(tf.boolean_mask(pred_bboxes, fg_mask), 
                                 tf.boolean_mask(target_bboxes, fg_mask))
            loss_box = tf.reduce_sum((1.0 - iou) * weight) / tf.maximum(target_scores_sum, 1)
            
            # # DFL loss
            # a, b = tf.split(target_bboxes, 2, axis=-1)
            # target_lt_rb = tf.concat([anchor_points - a, b - anchor_points], axis=-1)
            # target_lt_rb = tf.clip_by_value(target_lt_rb, 0, self.dfl_ch - 1.01)
            # loss_dfl = self.df_loss(tf.boolean_mask(pred_output, fg_mask), 
            #                       tf.boolean_mask(target_lt_rb, fg_mask))
            # loss_dfl = tf.reduce_sum(loss_dfl * weight) / tf.maximum(target_scores_sum, 1)
            
            
            # DFL loss
            a, b = tf.split(target_bboxes, 2, axis=-1)
            target_lt_rb = tf.concat([anchor_points - a, b - anchor_points], axis=-1)
            target_lt_rb = tf.clip_by_value(target_lt_rb, 0, self.dfl_ch - 1.01)
            
            # Reshape predictions to match PyTorch format
            fg_pred_output = tf.boolean_mask(pred_output, fg_mask)
            fg_pred_output = tf.reshape(fg_pred_output, [-1, self.dfl_ch])
            
            # Reshape targets to match
            fg_target_lt_rb = tf.reshape(tf.boolean_mask(target_lt_rb, fg_mask), [-1])
            
            loss_dfl = self.df_loss(fg_pred_output, fg_target_lt_rb)
            loss_dfl = tf.reduce_sum(loss_dfl * weight) / tf.maximum(target_scores_sum, 1)
                    
        
            
            

        
        # Apply loss weights
        loss_cls *= self.params['cls']
        loss_box *= self.params['box']
        loss_dfl *= self.params['dfl']
        total_loss = loss_cls + loss_box + loss_dfl
        
        # Debugging outputs
        # In ComputeLoss.call() method:
        print(f"\n--- [utils/util.py::ComputeLoss.__call__] LOSS COMPONENTS DEBUG ---")
        print(f"[utils/util.py::ComputeLoss.__call__] Raw loss_cls: {loss_cls.numpy().item():.6f}")
        print(f"[utils/util.py::ComputeLoss.__call__] Raw loss_box: {loss_box.numpy().item():.6f}")
        print(f"[utils/util.py::ComputeLoss.__call__] Raw loss_dfl: {loss_dfl.numpy().item():.6f}")


        print(f"[utils/util.py::ComputeLoss.__call__] Loss weights - cls: {self.params['cls']}, box: {self.params['box']}, dfl: {self.params['dfl']}")
        print(f"[utils/util.py::ComputeLoss.__call__] target_scores_sum: {target_scores_sum.numpy().item():.6f}")
        print(f"[utils/util.py::ComputeLoss.__call__] fg_mask sum: {fg_count.numpy()}")
        print(f"[utils/util.py::ComputeLoss.__call__] Weighted loss_cls: {loss_cls.numpy().item():.6f}")
        print(f"[utils/util.py::ComputeLoss.__call__] Weighted loss_box: {loss_box.numpy().item():.6f}")
        print(f"[utils/util.py::ComputeLoss.__call__] Weighted loss_dfl: {loss_dfl.numpy().item():.6f}")
        print(f"[utils/util.py::ComputeLoss.__call__] Total loss: {total_loss.numpy().item():.6f}")
        print(f"--- [utils/util.py::ComputeLoss.__call__] END KERAS LOSS COMPUTATION DEBUG ---\n")
        
        return total_loss


    # def one_hot(indices, depth, dtype=tf.float32):
    #     # identical semantic to the helper used in PyTorch code
    #     return tf.one_hot(indices, depth, dtype=dtype)


    # --------------------------------------------------------------------------- #
    #  Replace your existing `assign` method with this one
    # --------------------------------------------------------------------------- #
    def assign(
        self,
        pred_scores,   # (B, A, C)
        pred_bboxes,   # (B, A, 4)
        true_labels,   # (B, N, 1)
        true_bboxes,   # (B, N, 4)
        true_mask,     # (B, N, 1)   – 1 for valid boxes, 0 for padding
        anchors        # (A, 2)
    ):
        """
        Task-aligned One-stage Object Detection assigner – TensorFlow version that
        mirrors the original PyTorch implementation *exactly*, including all debug
        messages and tensor shapes.
        """

        # --------------------------- DEBUG HEADER --------------------------- #
        print("\n--- [utils/util.py::ComputeLoss.assign] KERAS ASSIGNMENT DEBUG ---")
        print(f"[utils/util.py::ComputeLoss.assign] pred_scores shape: {pred_scores.shape}")
        print(f"[utils/util.py::ComputeLoss.assign] pred_bboxes shape: {pred_bboxes.shape}")
        print(f"[utils/util.py::ComputeLoss.assign] true_labels shape: {true_labels.shape}")
        print(f"[utils/util.py::ComputeLoss.assign] true_bboxes shape: {true_bboxes.shape}")
        print(f"[utils.util.py::ComputeLoss.assign] true_mask shape: {true_mask.shape}")
        print(f"[utils/util.py::ComputeLoss.assign] anchors shape: {anchors.shape}")

        # ------------------------------------------------------------------- #
        # basic sizes
        self.bs           = tf.shape(pred_scores)[0]
        self.num_max_boxes = tf.shape(true_bboxes)[1]
        print(f"[utils/util.py::ComputeLoss.assign] Batch size: {self.bs}, Max boxes: {self.num_max_boxes}")

        # ------------------------------------------------------------------- #
        # no GT case --------------------------------------------------------- #
        if tf.equal(self.num_max_boxes, 0):
            print(f"[utils/util.py::ComputeLoss.assign] No ground truth boxes, returning zeros")
            return (
                tf.zeros_like(pred_bboxes),
                tf.zeros_like(pred_scores),
                tf.cast(tf.zeros_like(pred_scores[..., 0]), tf.bool)
            )

        # ------------------------------------------------------------------- #
        # indices tensor i[0], i[1] ----------------------------------------- #
        i0 = tf.tile(
            tf.reshape(tf.range(self.bs, dtype=tf.int64), (-1, 1)),
            [1, self.num_max_boxes]
        )                          # (B, N)
        i1 = tf.cast(tf.squeeze(true_labels, -1), tf.int64)  # (B, N)
        i = tf.stack([i0, i1], axis=0)                       # (2, B, N)
        print(f"[utils/util.py::ComputeLoss.assign] Created indices tensor with shape: {i.shape}")

        # ------------------------------------------------------------------- #
        # IoU (CIoU) between all GT and anchors ----------------------------- #
        print("\n--- [utils/util.py::ComputeLoss.assign] IOUs DEBUG ---")
        print(f"[utils/util.py::ComputeLoss.assign] true_bboxes shape: {true_bboxes.shape}")
        print(f"[utils/util.py::ComputeLoss.assign] pred_bboxes shape: {pred_bboxes.shape}")

        overlaps = self.compute_iou(
            tf.expand_dims(true_bboxes, 2),     # (B, N, 1, 4)
            tf.expand_dims(pred_bboxes, 1)      # (B, 1, A, 4)
        )
        overlaps = tf.clip_by_value(tf.squeeze(overlaps, 3), 0.0, 1.0)  # (B, N, A)
        print(f"[utils/util.py::ComputeLoss.assign] overlaps shape: {overlaps.shape}")
        if tf.executing_eagerly():
            print(f"[utils/util.py::ComputeLoss.assign] overlaps min: {tf.reduce_min(overlaps).numpy()}, "
                f"max: {tf.reduce_max(overlaps).numpy()}")

        # ------------------------------------------------------------------- #
        # alignment metric --------------------------------------------------- #
        # pred_scores[i[0], :, i[1]]  ->  (B, N, A)
        scores_cls = tf.gather(
            pred_scores,     # (B, A, C)
            i1,              # (B, N)
            axis=2,
            batch_dims=1
        )                    # (B, A, N)
        scores_cls = tf.transpose(scores_cls, [0, 2, 1])  # (B, N, A)

        align_metric = tf.pow(scores_cls, self.alpha) * tf.pow(overlaps, self.beta)
        print(f"[utils/util.py::ComputeLoss.assign] align_metric shape: {align_metric.shape}")
        if tf.executing_eagerly():
            print(f"[utils/util.py::ComputeLoss.assign] align_metric min: {tf.reduce_min(align_metric).numpy()}, "
                f"max: {tf.reduce_max(align_metric).numpy()}")

        # ------------------------------------------------------------------- #
        # mask_in_gts -------------------------------------------------------- #
        bs          = tf.shape(true_bboxes)[0]
        n_boxes     = tf.shape(true_bboxes)[1]
        anchors_exp = tf.expand_dims(anchors, 0)            # (1, A, 2)

        lt, rb = tf.split(tf.reshape(true_bboxes, [-1, 1, 4]), 2, axis=-1)  # (B*N,1,2)
        bbox_deltas = tf.concat([anchors_exp - lt, rb - anchors_exp], axis=2)  # (B*N, A, 4)
        bbox_deltas = tf.reshape(bbox_deltas, [bs, n_boxes, -1, 4])
        mask_in_gts = tf.reduce_min(bbox_deltas, axis=-1) > 1e-9            # (B, N, A)
        print(f"[utils/util.py::ComputeLoss.assign] mask_in_gts shape: {mask_in_gts.shape}")
        if tf.executing_eagerly():
            print(f"[utils/util.py::ComputeLoss.assign] mask_in_gts sum: {tf.reduce_sum(tf.cast(mask_in_gts, tf.int32)).numpy()}")

        # ------------------------------------------------------------------- #
        # metrics & Top-k ---------------------------------------------------- #
        metrics   = align_metric * tf.cast(mask_in_gts, align_metric.dtype)
        print(f"[utils/util.py::ComputeLoss.assign] metrics shape: {metrics.shape}")

        top_k_mask = tf.tile(tf.cast(true_mask, tf.bool), [1, 1, self.top_k])  # (B, N, K)
        print(f"[utils/util.py::ComputeLoss.assign] top_k_mask shape: {top_k_mask.shape}")

        num_anchors          = tf.shape(metrics)[-1]
        top_k_metrics, top_k_indices = tf.math.top_k(metrics, k=self.top_k, sorted=True)
        print(f"[utils/util.py::ComputeLoss.assign] top_k_metrics shape: {top_k_metrics.shape}")
        print(f"[utils/util.py::ComputeLoss.assign] top_k_indices shape: {top_k_indices.shape}")

        # if top_k_mask were None in PyTorch, we would build it here – we already have it
        top_k_indices = tf.where(top_k_mask, top_k_indices, tf.zeros_like(top_k_indices))

        is_in_top_k = tf.reduce_sum(one_hot(top_k_indices, num_anchors, dtype=tf.int32), axis=-2)  # (B, N, A)
        print(f"[utils/util.py::ComputeLoss.assign] is_in_top_k shape: {is_in_top_k.shape}")

        # filter invalid boxes
        is_in_top_k = tf.where(is_in_top_k > 1, 0, is_in_top_k)
        mask_top_k  = tf.cast(is_in_top_k, metrics.dtype)                                         # (B, N, A)
        print(f"[utils/util.py::ComputeLoss.assign] mask_top_k shape: {mask_top_k.shape}")

        # ------------------------------------------------------------------- #
        # positive mask ------------------------------------------------------ #
        mask_pos = mask_top_k * tf.cast(mask_in_gts, mask_top_k.dtype) * tf.cast(true_mask, mask_top_k.dtype)
        print(f"[utils/util.py::ComputeLoss.assign] mask_pos shape: {mask_pos.shape}")

        fg_mask = tf.reduce_sum(mask_pos, axis=1)                   # (B, A)
        print(f"[utils/util.py::ComputeLoss.assign] fg_mask shape: {fg_mask.shape}")
        if tf.executing_eagerly():
            print(f"[utils/util.py::ComputeLoss.assign] fg_mask sum: {tf.reduce_sum(fg_mask).numpy()}")

        # resolve anchors that hit multiple GTs ----------------------------- #
        if tf.reduce_max(fg_mask) > 1:
            print(f"[utils/util.py::ComputeLoss.assign] Detected anchors assigned to multiple GT boxes")
            mask_multi_gts = tf.tile(tf.expand_dims(fg_mask > 1, 1), [1, self.num_max_boxes, 1])  # (B,N,A)

            max_overlaps_idx = tf.argmax(overlaps, axis=1)                    # (B, A)
            is_max_overlaps  = one_hot(max_overlaps_idx, self.num_max_boxes, dtype=mask_pos.dtype)
            is_max_overlaps  = tf.transpose(is_max_overlaps, [0, 2, 1])       # (B, N, A)

            mask_pos = tf.where(mask_multi_gts, is_max_overlaps, mask_pos)
            fg_mask  = tf.reduce_sum(mask_pos, axis=1)
            if tf.executing_eagerly():
                print(f"[utils/util.py::ComputeLoss.assign] After resolving multiple assignments - fg_mask sum: "
                    f"{tf.reduce_sum(fg_mask).numpy()}")

        # ------------------------------------------------------------------- #
        # which GT each anchor serves --------------------------------------- #
        target_gt_idx = tf.argmax(mask_pos, axis=1)                           # (B, A)
        print(f"[utils/util.py::ComputeLoss.assign] target_gt_idx shape: {target_gt_idx.shape}")

        # # gather GT labels --------------------------------------------------- #
        batch_index = tf.reshape(tf.range(self.bs, dtype=tf.int64), (-1, 1))
        # # Cast to ensure compatible types for multiplication
        num_max_boxes_cast = cast_like(self.num_max_boxes, batch_index)
        target_gt_idx_flat = target_gt_idx + batch_index * num_max_boxes_cast    # broadcast
        # target_labels = tf.reshape(tf.cast(true_labels, tf.int64), [-1])[target_gt_idx_flat]  # (B, A)
        # print(f"[utils/util.py::ComputeLoss.assign] target_labels shape: {target_labels.shape}")

        # # gather GT boxes ---------------------------------------------------- #
        # target_bboxes = tf.reshape(true_bboxes, [-1, 4])[target_gt_idx_flat]     # (B, A, 4)
        # print(f"[utils/util.py::ComputeLoss.assign] target_bboxes shape: {target_bboxes.shape}")
        
        
        
        # ------------------------------------------------------------------- #
        # gather GT labels (shape: B × A) ----------------------------------- #
        flat_labels  = tf.reshape(true_labels, [-1])           # (B·N,)
        flat_bboxes  = tf.reshape(true_bboxes, [-1, 4])        # (B·N,4)

        # indices matrix (B,A) → flatten → gather → reshape back
        idx_flat     = tf.reshape(target_gt_idx_flat, [-1])    # (B·A,)
        labels_flat  = tf.gather(flat_labels, idx_flat)        # (B·A,)
        bboxes_flat  = tf.gather(flat_bboxes, idx_flat)        # (B·A,4)

        target_labels = tf.reshape(labels_flat, tf.shape(target_gt_idx))      # (B,A)
        target_bboxes = tf.reshape(bboxes_flat,
                                tf.concat([tf.shape(target_gt_idx), [4]], 0))  # (B,A,4)

        print(f"[utils/util.py::ComputeLoss.assign] target_labels shape: {target_labels.shape}")
        print(f"[utils/util.py::ComputeLoss.assign] target_bboxes shape: {target_bboxes.shape}")
        
        
        
        
        # Step 1: Make sure self.nc is correct
        print(f"[utils/util_keras.py::ComputeLoss.assign] self.nc: {self.nc}")  # Should be >1

        # Step 2: Correct one-hot encoding
        target_labels_clamped = tf.clip_by_value(target_labels, 0, self.nc - 1)
        target_scores = tf.one_hot(tf.cast(target_labels_clamped, tf.int32), self.nc, dtype=tf.float32)  # (B, A, C)

        # Step 3: Mask out background anchors
        fg_scores_mask = tf.tile(tf.expand_dims(fg_mask > 0, -1), [1, 1, self.nc])
        target_scores = tf.where(fg_scores_mask, target_scores, 0.0)
        print(f"[utils/util_keras.py::ComputeLoss.assign] target_scores shape: {target_scores.shape}")

        # Step 4: Normalize using align_metric
        align_metric *= mask_pos  # (B, N, A)
        pos_align_metrics = tf.reduce_max(align_metric, axis=-1, keepdims=True)  # (B, N, 1)
        pos_overlaps = tf.reduce_max(overlaps * mask_pos, axis=-1, keepdims=True)  # (B, N, 1)

        norm_align_metric = tf.reduce_max(
            align_metric * pos_overlaps / (pos_align_metrics + self.eps),
            axis=1,  # reduce over N
            keepdims=False
        )  # → shape: (B, A)

        norm_align_metric = tf.expand_dims(norm_align_metric, axis=-1)  # (B, A, 1)

        # Step 5: Apply to scores
        target_scores = target_scores * norm_align_metric  # Broadcasting works correctly
        print(f"[utils/util_keras.py::ComputeLoss.assign] Normalized target_scores shape: {target_scores.shape}")


                
                
                
        
        # ------------------------------------------------------------------- #
        # normalisation ------------------------------------------------------ #
        # align_metric *= mask_pos
        # pos_align_metrics = tf.reduce_max(align_metric, axis=-1, keepdims=True)
        # pos_overlaps      = tf.reduce_max(overlaps * mask_pos, axis=-1, keepdims=True)
        # norm_align_metric = tf.reduce_max(
        #     align_metric * pos_overlaps / (pos_align_metrics + self.eps), axis=-2, keepdims=True
        # )  # (B, 1, A)
        # target_scores = target_scores * norm_align_metric
        # print(f"[utils/util.py::ComputeLoss.assign] Normalized target_scores shape: {target_scores.shape}")
        
        
        
        if tf.executing_eagerly():
            print(f"[utils/util.py::ComputeLoss.assign] target_scores sum: {tf.reduce_sum(target_scores).numpy()}")

        print("--- [utils/util.py::ComputeLoss.assign] END KERAS ASSIGNMENT DEBUG ---\n")
        return target_bboxes, target_scores, tf.cast(fg_mask > 0, tf.bool)




    # def df_loss(self, pred_dist, target):
        
    #     print(f"\n--- [utils/util_keras.py::ComputeLoss.df_loss] PYTORCH DFL LOSS DEBUG ---")
    #     print(f"[utils/util_keras.py::ComputeLoss.df_loss] pred_dist shape: {pred_dist.shape}, dtype: {pred_dist.dtype}")
    #     print(f"[utils/util_keras.py::ComputeLoss.df_loss] target shape: {target.shape}, dtype: {target.dtype}")
    #     tl = tf.cast(target, tf.int32)  # target left
    #     tr = tl + 1  # target right
    #     # Cast tr back to the same dtype as target to avoid type mismatch
    #     tr_float = tf.cast(tr, target.dtype)
    #     wl = tr_float - target  # weight left
    #     wr = 1.0 - wl  # weight right
        
    #     # Get left and right values
    #     left_loss = tf.keras.losses.sparse_categorical_crossentropy(
    #         tl, pred_dist, from_logits=True, axis=-1
    #     )
    #     right_loss = tf.keras.losses.sparse_categorical_crossentropy(
    #         tr, pred_dist, from_logits=True, axis=-1
    #     )
        
    #     # Reshape and combine losses
    #     left_loss = tf.reshape(left_loss, tl.shape)
    #     right_loss = tf.reshape(right_loss, tl.shape)
        
    #     print(f"[utils/util_keras.py::ComputeLoss.df_loss] DFL loss shape: {loss.shape}")
    #     print(f"[utils/util_keras.py::ComputeLoss.df_loss] DFL loss mean: {loss.mean().item():.6f}")
    #     print(f"--- [utils/util_keras.py::ComputeLoss.df_loss] END PYTORCH DFL LOSS DEBUG ---\n")
        
    #     return tf.reduce_mean(left_loss * wl + right_loss * wr, axis=-1, keepdims=True)



    # def df_loss(self, pred_dist, target):
    #     print(f"\n--- [utils/util_keras.py::ComputeLoss.df_loss] KERAS DFL LOSS DEBUG ---")
    #     print(f"[utils/util_keras.py::ComputeLoss.df_loss] pred_dist shape: {pred_dist.shape}, dtype: {pred_dist.dtype}")
    #     print(f"[utils/util_keras.py::ComputeLoss.df_loss] target shape: {target.shape}, dtype: {target.dtype}")
        
    #     tl = tf.cast(target, tf.int32)  # target left
    #     tr = tl + 1  # target right
    #     # Cast tr back to the same dtype as target to avoid type mismatch
    #     tr_float = tf.cast(tr, target.dtype)
    #     wl = tr_float - target  # weight left
    #     wr = 1.0 - wl  # weight right
        
    #     # Get left and right values
    #     left_loss = tf.keras.losses.sparse_categorical_crossentropy(
    #         tl, pred_dist, from_logits=True, axis=-1
    #     )
    #     right_loss = tf.keras.losses.sparse_categorical_crossentropy(
    #         tr, pred_dist, from_logits=True, axis=-1
    #     )
        
    #     # Combine losses
    #     loss = left_loss * wl + right_loss * wr
        
    #     print(f"[utils/util_keras.py::ComputeLoss.df_loss] DFL loss shape: {loss.shape}")
    #     if tf.executing_eagerly():
    #         print(f"[utils/util_keras.py::ComputeLoss.df_loss] DFL loss mean: {tf.reduce_mean(loss).numpy().item():.6f}")
    #     print(f"--- [utils/util_keras.py::ComputeLoss.df_loss] END KERAS DFL LOSS DEBUG ---\n")
        
    #     return loss


    def df_loss(self, pred_dist, target):
        print(f"\n--- [utils/util_keras.py::ComputeLoss.df_loss] KERAS DFL LOSS DEBUG ---")
        print(f"[utils/util_keras.py::ComputeLoss.df_loss] pred_dist shape: {pred_dist.shape}, dtype: {pred_dist.dtype}")
        print(f"[utils/util_keras.py::ComputeLoss.df_loss] target shape: {target.shape}, dtype: {target.dtype}")
        
        tl = tf.cast(target, tf.int32)  # target left
        tr = tl + 1  # target right
        tr_float = tf.cast(tr, target.dtype)
        wl = tr_float - target  # weight left
        wr = 1.0 - wl  # weight right
        
        # Compute losses
        left_loss = tf.keras.losses.sparse_categorical_crossentropy(
            tl, pred_dist, from_logits=True, axis=-1
        )
        right_loss = tf.keras.losses.sparse_categorical_crossentropy(
            tr, pred_dist, from_logits=True, axis=-1
        )
        
        # Combine losses
        loss_per_coord = left_loss * wl + right_loss * wr
        
        # Reshape to (n_fg, 4) and compute mean per anchor
        n_fg = tf.shape(loss_per_coord)[0] // 4
        loss_per_anchor = tf.reshape(loss_per_coord, (n_fg, 4))
        loss = tf.reduce_mean(loss_per_anchor, axis=1, keepdims=True)  # Shape: (n_fg, 1)
        
        print(f"[utils/util_keras.py::ComputeLoss.df_loss] DFL loss shape: {loss.shape}")
        if tf.executing_eagerly():
            print(f"[utils/util_keras.py::ComputeLoss.df_loss] DFL loss mean: {tf.reduce_mean(loss).numpy().item():.6f}")
        print(f"--- [utils/util_keras.py::ComputeLoss.df_loss] END KERAS DFL LOSS DEBUG ---\n")
        
        return loss

    def compute_iou(self, box1, box2, eps=1e-7):
        """
        Complete-IoU (CIoU) with verbose debug prints that exactly match the
        PyTorch implementation’s shapes and log locations.

        Args
        ----
        box1 : Tensor[..., 4]   – broadcastable to box2
        box2 : Tensor[..., 4]
        eps  : float            – numerical stability term

        Returns
        -------
        ciou : Tensor[..., 1]   – same broadcasted shape as intersection/union
        """
        # ---------- DEBUG LOG HEADER ----------
        print("\n--- [utils/util.py::ComputeLoss.iou] KERAS IOU DEBUG ---")
        print(f"[utils/util.py::ComputeLoss.iou] box1 shape: {box1.shape}")
        print(f"[utils/util.py::ComputeLoss.iou] box2 shape: {box2.shape}")

        # Coordinate split (x1, y1, x2, y2)
        b1_x1, b1_y1, b1_x2, b1_y2 = tf.split(box1, 4, axis=-1)
        b2_x1, b2_y1, b2_x2, b2_y2 = tf.split(box2, 4, axis=-1)

        # Width / height (+eps exactly like PyTorch)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
        print(f"[utils/util.py::ComputeLoss.iou] Box1 dimensions - w: {w1.shape}, h: {h1.shape}")
        print(f"[utils/util.py::ComputeLoss.iou] Box2 dimensions - w: {w2.shape}, h: {h2.shape}")

        # -------- Intersection --------
        inter_w = tf.maximum(tf.minimum(b1_x2, b2_x2) - tf.maximum(b1_x1, b2_x1), 0.0)
        inter_h = tf.maximum(tf.minimum(b1_y2, b2_y2) - tf.maximum(b1_y1, b2_y1), 0.0)
        intersection = inter_w * inter_h
        print(f"[utils/util.py::ComputeLoss.iou] Intersection shape: {intersection.shape}")

        # -------- Union & IoU --------
        union = w1 * h1 + w2 * h2 - intersection + eps
        print(f"[utils/util.py::ComputeLoss.iou] Union shape: {union.shape}")

        iou = intersection / union
        print(f"[utils/util.py::ComputeLoss.iou] IoU shape: {iou.shape}")
        if tf.size(iou) > 0 and tf.executing_eagerly():
            print(f"[utils/util.py::ComputeLoss.iou] IoU min: {tf.reduce_min(iou).numpy():.6f}, "
                f"max: {tf.reduce_max(iou).numpy():.6f}")

        # -------- CIoU terms --------
        cw = tf.maximum(b1_x2, b2_x2) - tf.minimum(b1_x1, b2_x1)          # convex width
        ch = tf.maximum(b1_y2, b2_y2) - tf.minimum(b1_y1, b2_y1)          # convex height
        c2 = cw**2 + ch**2 + eps                                          # convex diag²
        print(f"[utils/util.py::ComputeLoss.iou] Convex diagonal squared shape: {c2.shape}")

        rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2)**2 +
                (b2_y1 + b2_y2 - b1_y1 - b1_y2)**2) / 4.0
        print(f"[utils/util.py::ComputeLoss.iou] Center distance squared shape: {rho2.shape}")

        v = (4.0 / (math.pi**2)) * tf.square(tf.atan(w2 / (h2)) - tf.atan(w1 / (h1)))
        print(f"[utils/util.py::ComputeLoss.iou] Aspect ratio term shape: {v.shape}")

        alpha = tf.stop_gradient(v / (v - iou + (1.0 + eps)))             # matches torch.no_grad()

        ciou = iou - (rho2 / c2 + v * alpha)
        print(f"[utils/util.py::ComputeLoss.iou] CIoU shape: {ciou.shape}")
        if tf.size(ciou) > 0 and tf.executing_eagerly():
            print(f"[utils/util.py::ComputeLoss.iou] CIoU min: {tf.reduce_min(ciou).numpy():.6f}, "
                f"max: {tf.reduce_max(ciou).numpy():.6f}")
        print("--- [utils/util.py::ComputeLoss.iou] END KERAS IOU DEBUG ---\n")

        return ciou  # do NOT squeeze – shape identical to PyTorch
