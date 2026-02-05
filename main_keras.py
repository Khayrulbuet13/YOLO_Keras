import argparse
import csv
import math
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, callbacks
import yaml
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

from nets.tinysimov35_keras import yolo_v8_s
from nets.tinysimov35_keras_quantized import yolo_v8_s_quantized
from utils.dataset_keras import Dataset
from utils.util_keras import (
    generate_colors, 
    visualize_predictions,
    ComputeLoss,
    EMA,
    AverageMeter,
    non_max_suppression,
    compute_ap
)

# Import QKeras utilities for quantized model handling
try:
    from qkeras.estimate import print_qstats
    from qkeras.utils import model_save_quantized_weights
    QKERAS_AVAILABLE = True
except ImportError:
    QKERAS_AVAILABLE = False
    print("[WARNING] QKeras utilities not available. Quantized model saving may not work properly.")

# Define global dtype for consistent type handling
DTYPE = tf.float32  # Central dtype definition (change to float16 for mixed-precision)

# Set consistent random seeds for reproducible results
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

def learning_rate(args, params):
    def fn(epoch):
        return (1 - epoch / args.epochs) * (1.0 - params['lrf']) + params['lrf']
    return fn

class MultiGroupOptimizer:
    """Custom optimizer that mimics PyTorch's parameter group behavior"""
    def __init__(self, model, lr0, momentum, weight_decay, nesterov=True):
        self.lr0 = lr0
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov

        # Group variables by name
        self.bias_params = set()
        self.weight_params = set()
        self.bn_params = set()
        all_vars = model.trainable_variables
        for var in all_vars:
            var_name = var.name.lower()
            if var_name.endswith('/bias:0'):
                self.bias_params.add(var.name)
            elif any(bn in var_name for bn in ['/batch_normalization', '/bn', '/batchnorm']):
                # All BN variables (gamma, beta, moving_mean, moving_variance)
                self.bn_params.add(var.name)
            elif var_name.endswith('/kernel:0'):
                self.weight_params.add(var.name)
        # Remove BN weights from weight_params (so they're not double-counted)
        self.weight_params = self.weight_params - self.bn_params

        # Initialize momentum variables as tf.Variable
        # Use variable name as key (variables are not hashable, but names are)
        self.momentum_vars = {}
        for var in all_vars:
            self.momentum_vars[var.name] = tf.Variable(tf.zeros_like(var), trainable=False)

        self.lr_bias = lr0
        self.lr_weight = lr0
        self.lr_bn = lr0

    def set_learning_rates(self, lr_bias, lr_weight, lr_bn):
        self.lr_bias = lr_bias
        self.lr_weight = lr_weight
        self.lr_bn = lr_bn

    def apply_gradients(self, gradients_and_vars):
        for grad, var in gradients_and_vars:
            if grad is None:
                continue
            var_name = var.name
            if var_name in self.bias_params:
                lr = self.lr_bias
            elif var_name in self.bn_params:
                lr = self.lr_bn
            else:
                lr = self.lr_weight
            # Apply weight decay to weights (not bias or batch norm)
            if var_name in self.weight_params:
                grad = grad + self.weight_decay * var
            # Apply momentum using variable name as key
            # Create momentum variable if it doesn't exist or has wrong shape
            if var_name not in self.momentum_vars:
                self.momentum_vars[var_name] = tf.Variable(tf.zeros_like(var), trainable=False)
            momentum_var = self.momentum_vars[var_name]
            # Check if shapes match, recreate if needed
            if momentum_var.shape != grad.shape:
                self.momentum_vars[var_name] = tf.Variable(tf.zeros_like(grad), trainable=False)
                momentum_var = self.momentum_vars[var_name]
            momentum_var.assign(self.momentum * momentum_var + grad)
            # Apply update
            if self.nesterov:
                update = self.momentum * momentum_var + grad
            else:
                update = momentum_var
            var.assign_sub(lr * update)

def train(args, params):
    # Initialize with central dtype
    num_classes = len(params['names'].values())
    
    # Use quantized model if specified
    if args.quantized:
        print("[INFO] Using quantized model (QKeras) for HLS4ml FPGA synthesis")
        model = yolo_v8_s_quantized(num_classes, img_size=args.img_size, dtype=DTYPE)
    else:
        model = yolo_v8_s(num_classes, img_size=args.img_size, dtype=DTYPE)  # Pass dtype to model
    
    # Create model directory
    if args.local_rank == 0:
        os.makedirs(args.save_path, exist_ok=True)
    
    # Optimizer & Scheduler - MATCH PYTORCH BEHAVIOR
    accumulate = max(round(64 / (args.batch_size)), 1)
    params['weight_decay'] *= args.batch_size * accumulate / 64

    # Use custom multi-group optimizer to match PyTorch behavior
    optimizer = MultiGroupOptimizer(
        model, 
        params['lr0'], 
        params['momentum'], 
        params['weight_decay'], 
        nesterov=True
    )

    lr_func = learning_rate(args, params)
    
    # EMA
    ema = EMA(model) if args.local_rank == 0 else None
    
    # Datasets
    train_filenames = []
    with open(os.path.join(args.dataset_dir, 'train.txt')) as ftrain:
        for line in ftrain.readlines():
            line = line.rstrip().split('/')[-1]
            train_filenames.append(os.path.join(args.dataset_dir, 'images/train', line))

    # Datasets with dtype
    train_dataset = Dataset(train_filenames, args.img_size, params, True, dtype=DTYPE)
    def data_generator():
        # Mimic PyTorch's collate_fn behaviour. Each object's first column must
        # contain the index of the image in the current batch so that loss
        # calculation knows which prediction it belongs to.
        batch_samples = []
        batch_targets = []
        batch_shapes = []
        batch_size = args.batch_size
        
        
        for i in range(len(train_dataset)):
            sample, target, shapes = train_dataset[i]
            
            if i < 2:
                if hasattr(shapes, 'shape'):
                    if shapes.shape[0] >= 2 and shapes.shape[1] >= 2:
                        # Check if shapes[0] is all zeros (which would indicate mosaic was used)
                        is_mosaic = np.all(shapes[0] == 0)
            
            if i < 5:
                if target.shape[0] > 0:
                    # Check if there are multiple objects with the same class ID (which would suggest they came from different images)
                    if target.shape[0] > 1:
                        class_ids = target[:, 1].numpy() if hasattr(target, 'numpy') else target[:, 1]
                        unique_classes = np.unique(class_ids)
            
            # Insert the batch index in the first column like the PyTorch collate
            # function. This allows loss computation to know which image each
            # target belongs to after concatenation.
            batch_index = len(batch_samples)
            if target.shape[0] > 0:
                img_idx = tf.fill([tf.shape(target)[0], 1],
                                   tf.cast(batch_index, target.dtype))
                target = tf.concat([img_idx, target[:, 1:]], axis=1)

            batch_samples.append(sample)
            if target.shape[0] > 0:
                batch_targets.append(target)
            batch_shapes.append(shapes)
            
            # When we have a complete batch, yield it (drop incomplete last batch)
            if (i + 1) % batch_size == 0:
                # Stack samples
                stacked_samples = tf.cast(np.stack(batch_samples, axis=0), DTYPE)
                
                # Concatenate targets (flattened approach like PyTorch)
                if batch_targets:
                    stacked_targets = tf.cast(np.concatenate(batch_targets, axis=0), DTYPE)
                else:
                    # Empty targets case
                    stacked_targets = tf.zeros((0, 6), dtype=DTYPE)  # 6 columns: batch_idx + 5 target values
                
                # Stack shapes
                stacked_shapes = tf.cast(np.stack(batch_shapes, axis=0), tf.float32)  # Shapes remain float32
                
                yield stacked_samples, stacked_targets, stacked_shapes
                
                # Reset batch containers
                batch_samples = []
                batch_targets = []
                batch_shapes = []
    
    train_loader = tf.data.Dataset.from_generator(
        data_generator,
        output_types=(DTYPE, DTYPE, DTYPE),
        output_shapes=(
            (args.batch_size, args.img_size[0], args.img_size[1], 3),
            (None, 6),
            (args.batch_size, 3, 2)
        )
    ).prefetch(tf.data.AUTOTUNE)
    
    # Loss function with dtype
    criterion = ComputeLoss(model, params, dtype=DTYPE)
    
    # Training loop
    best = 0
    patience = 50 if args.quantized else 20  # Quantized models need more patience
    patience_counter = 0
    if args.quantized:
        print(f"[INFO] Using increased patience for quantized training: {patience} epochs")
    num_batch = len(train_dataset) // args.batch_size  # Floor division since we drop incomplete batches
    num_warmup = max(round(params['warmup_epochs'] * num_batch), 1000)
    
    # CSV logger
    csv_path = os.path.join(args.save_path, 'step.csv')
    csv_file = open(csv_path, 'w', newline='')
    writer = csv.DictWriter(csv_file, fieldnames=[
        'epoch', 
        'train_mAP@50', 'train_mAP', 'train_Precision', 'train_Recall', 'train_F1',
        'val_mAP@50', 'val_mAP', 'val_Precision', 'val_Recall', 'val_F1'
    ])
    writer.writeheader()

    for epoch in range(args.epochs):
        m_loss = AverageMeter()
        
        # Turn off mosaic for last 10 epochs
        if args.epochs - epoch == 10:
            train_dataset.mosaic = False
            print(f"\n[main_keras.py::train] TURNING OFF MOSAIC at epoch {epoch+1}/{args.epochs}")

        p_bar = tqdm(enumerate(train_loader), total=num_batch, desc=f'Epoch {epoch+1}/{args.epochs}')
        
        for i, (samples, targets, shapes) in p_bar:
            x = i + num_batch * epoch
            
            # Warmup - MATCH PYTORCH BEHAVIOR
            if x <= num_warmup:
                xp = [0, num_warmup]
                fp = [1, 64 / args.batch_size]
                accumulate = max(1, np.interp(x, xp, fp).round())
                
                # Set different learning rates for each parameter group (MATCH PYTORCH)
                # Group 0 (bias): warmup_bias_lr -> lr0 * lr_func(epoch)
                lr_bias = np.interp(x, xp, [params['warmup_bias_lr'], params['lr0'] * lr_func(epoch)])
                # Group 1 (weights): 0.0 -> lr0 * lr_func(epoch)
                lr_weight = np.interp(x, xp, [0.0, params['lr0'] * lr_func(epoch)])
                # Group 2 (batch norm): 0.0 -> lr0 * lr_func(epoch)
                lr_bn = np.interp(x, xp, [0.0, params['lr0'] * lr_func(epoch)])
                
                optimizer.set_learning_rates(lr_bias, lr_weight, lr_bn)
                
                # Adjust momentum
                momentum = np.interp(x, xp, [params['warmup_momentum'], params['momentum']])
                # Note: momentum adjustment would need to be implemented in MultiGroupOptimizer
            else:
                # Post-warmup: all groups use the same learning rate
                lr = params['lr0'] * lr_func(epoch)
                optimizer.set_learning_rates(lr, lr, lr)

            # Forward pass
            with tf.GradientTape() as tape:
                outputs = model(samples, training=True)
                loss = criterion(outputs, targets)
                
                # Scale loss for multi-GPU (MATCH PYTORCH BEHAVIOR)
                loss *= args.batch_size
                # Note: world_size not available in Keras args, using batch_size only for single-GPU
                
                m_loss.update(loss.numpy(), samples.shape[0])

            # Backward pass
            gradients = tape.gradient(loss, model.trainable_variables)
            
            # Clip gradients for quantized models to prevent explosion
            if args.quantized:
                gradients, global_norm = tf.clip_by_global_norm(gradients, 1.0)
            
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            # Update EMA
            if ema:
                ema.update()
            
            # Update progress bar
            p_bar.set_postfix({'loss': m_loss.avg})

        # Evaluation
        if args.local_rank == 0:
            # Create EMA model for evaluation if EMA is enabled
            eval_model = model
            if ema:
                # Create a temporary model with EMA weights
                if args.quantized:
                    eval_model = yolo_v8_s_quantized(num_classes, img_size=args.img_size)
                else:
                    eval_model = yolo_v8_s(num_classes, img_size=args.img_size)
                # Directly apply EMA weights
                eval_model.set_weights([w.numpy() for w in ema.ema_weights])
            
            # Train evaluation
            train_tp, train_fp, train_precision, train_recall, train_map50, train_mean_ap = test(args, params, eval_model, is_train=True)
            train_f1 = 2 * train_precision * train_recall / (train_precision + train_recall + 1e-16)
            
            # Validation evaluation
            val_tp, val_fp, val_precision, val_recall, val_map50, val_mean_ap = test(args, params, eval_model, is_train=False)
            val_f1 = 2 * val_precision * val_recall / (val_precision + val_recall + 1e-16)

            # Log results
            writer.writerow({
                'epoch': str(epoch + 1).zfill(3),
                'train_mAP@50': f'{train_map50:.3f}',
                'train_mAP': f'{train_mean_ap:.3f}',
                'train_Precision': f'{train_precision:.3f}',
                'train_Recall': f'{train_recall:.3f}',
                'train_F1': f'{train_f1:.3f}',
                'val_mAP@50': f'{val_map50:.3f}',
                'val_mAP': f'{val_mean_ap:.3f}',
                'val_Precision': f'{val_precision:.3f}',
                'val_Recall': f'{val_recall:.3f}',
                'val_F1': f'{val_f1:.3f}'
            })
            csv_file.flush()

            # Save model
            if val_mean_ap > best:
                best = val_mean_ap
                patience_counter = 0  # Reset counter on improvement
                model.save_weights(os.path.join(args.save_path, 'best.weights.h5'))
                
                # Save quantized weights if using quantized model
                if args.quantized and QKERAS_AVAILABLE:
                    print("[INFO] Saving quantized weights for HLS4ml deployment...")
                    model_save_quantized_weights(model)
                    
                print(f'Epoch {epoch + 1}: New best model saved (mAP: {best:.4f})')
            else:
                patience_counter += 1
            
            model.save_weights(os.path.join(args.save_path, 'last.weights.h5'))
            
            # Early stopping check
            if patience_counter >= patience:
                print(f'\nEarly stopping triggered after {epoch + 1} epochs (no improvement for {patience} epochs)')
                break

    csv_file.close()
    
    # Print quantization statistics for quantized models
    if args.quantized and QKERAS_AVAILABLE and args.local_rank == 0:
        print("\n" + "="*80)
        print("QUANTIZATION STATISTICS")
        print("="*80)
        try:
            print_qstats(model)
        except (AttributeError, Exception) as e:
            print(f"Note: print_qstats() not available for custom Model classes")
            print(f"This is a known limitation - the model trained successfully")
            print(f"Quantized weights have been saved and can be used for HLS4ml conversion")
        print("="*80)

def test(args, params, model=None, is_train=False):
    # Load dataset
    filenames = []
    split = 'train' if is_train else 'val'
    with open(os.path.join(args.dataset_dir, f'{split}.txt')) as f:
        for line in f.readlines():
            line = line.rstrip().split('/')[-1]
            filenames.append(os.path.join(args.dataset_dir, f'images/{split}', line))

    # Load dataset with dtype
    dataset = Dataset(filenames, args.img_size, params, False, dtype=DTYPE)
    def test_data_generator():
        for i in range(len(dataset)):
            sample, target, shapes = dataset[i]
            
            # Pad targets to maximum possible size (e.g., 100 objects max)
            max_objects = 100
            padded_target = np.zeros((max_objects, 6), dtype=np.float32)
            if target.shape[0] > 0:
                n_objects = min(target.shape[0], max_objects)
                padded_target[:n_objects] = target[:n_objects]
            
            sample = tf.cast(sample, DTYPE)
            padded_target = tf.cast(padded_target, DTYPE)
            shapes = tf.cast(shapes, tf.float32)
            
            yield sample, padded_target, shapes
    
    loader = tf.data.Dataset.from_generator(
        test_data_generator,
        output_types=(DTYPE, DTYPE, DTYPE),
        output_shapes=(
            (args.img_size[0], args.img_size[1], 3),
            (100, 6),
            (3, 2)
        )
    ).batch(8).prefetch(tf.data.AUTOTUNE)

    # Load model if not provided
    if model is None:
        model_path = os.path.join(args.save_path, 'best.weights.h5')
        if args.quantized:
            model = yolo_v8_s_quantized(len(params['names']), img_size=args.img_size)
        else:
            model = yolo_v8_s(len(params['names']), img_size=args.img_size)
        model.load_weights(model_path)


    # Prepare for evaluation
    class_colors = generate_colors(len(params['names']))
    results_dir = os.path.join(args.save_path, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    metrics = []
    vis_count = 0

    for samples, targets, shapes in tqdm(loader, desc='Evaluating'):
        outputs = model(samples, training=False)
        detections = non_max_suppression(outputs, 0.25, 0.45)
        
        # Scale GT coordinates from normalized to pixel space (matching PyTorch line 416)
        _, h, w, _ = samples.shape
        # targets format: [batch_idx, cls, x, y, w, h] -> scale cols 2:6 by (w, h, w, h)
        scale_tensor = tf.constant([1.0, 1.0, float(w), float(h), float(w), float(h)], dtype=targets.dtype)
        targets = targets * scale_tensor
        
        # Process each image in batch
        for i in range(samples.shape[0]):
            # Get predictions and ground truth
            pred = detections[i]

            # Extract ground truth for current image in batch (following PyTorch approach)
            # In PyTorch: targets shape is (N, 6) where N is total objects across batch
            # In Keras: targets shape is (batch_size, max_objects, 6) 
            # We need to extract valid objects for image i
            gt_for_image = targets[i]  # Shape: (max_objects, 6)
            # Filter out zero/padded entries (where all values are 0)
            valid_mask = tf.reduce_sum(tf.abs(gt_for_image), axis=1) > 0
            gt = tf.boolean_mask(gt_for_image, valid_mask)  # Shape: (num_valid_objects, 6)
            # Keep gt as is - already in correct format (class, x, y, w, h) from dataset
            if tf.shape(gt)[0] == 0:
                gt = tf.zeros((0, 5), dtype=tf.float32)
                    
            # Visualization
            if not is_train and vis_count < 5:
                # Format single-sample versions for visualization
                single_sample = samples[i:i+1]
                single_out = [pred]  # wrap in list for index usage
                single_shapes = shapes[i:i+1]
                
                # Scale ground truth boxes to original image size (like PyTorch version)
                single_gt = []
                if tf.shape(gt)[0] > 0:
                    # GT coordinates are now already in pixel space (scaled earlier)
                    gt_boxes = tf.identity(gt)  # Clone the tensor
                    h0, w0 = shapes[i][0]  # Original height, width
                    pad_info = shapes[i][2]  # Get padding values [pad_w, pad_h]
                    pad_w, pad_h = pad_info[0], pad_info[1]
                    
                    # Get current image dimensions
                    _, h, w, _ = samples.shape
                    
                    class_ids = gt_boxes[:, 1]  # Class column
                    # Calculate scaled dimensions after removing padding
                    scaled_w = float(w) - 2 * pad_w
                    scaled_h = float(h) - 2 * pad_h

                    # Compute scaling factors from resized to original
                    scale_x = w0 / scaled_w if scaled_w > 0 else 1.0
                    scale_y = h0 / scaled_h if scaled_h > 0 else 1.0

                    # gt format: [batch_idx, cls, x_center, y_center, width, height] in PIXEL coords
                    coords = gt_boxes[:, 2:]  # Get [x_center, y_center, width, height]
                    
                    # Coords are already in pixel space, just need to undo padding and scale to original
                    x_center = (coords[:, 0] - pad_w) * scale_x
                    y_center = (coords[:, 1] - pad_h) * scale_y
                    width = coords[:, 2] * scale_x
                    height = coords[:, 3] * scale_y
                    
                    # Recombine: [img_idx, cls, x_center, y_center, width, height]
                    for j in range(tf.shape(gt_boxes)[0]):
                        single_gt.append(tf.stack([
                            tf.constant(0.0, dtype=tf.float32),  # img_idx
                            class_ids[j],  # class
                            x_center[j],   # x_center
                            y_center[j],   # y_center
                            width[j],      # width
                            height[j]      # height
                        ]))
                    single_gt = tf.stack(single_gt)
                else:
                    single_gt = tf.zeros((0, 6), dtype=tf.float32)
                
                visualize_predictions(
                    single_sample,
                    single_out,
                    single_gt,      # Pass the correctly scaled ground truth boxes
                    single_shapes,
                    params,
                    class_colors,
                    results_dir,
                    vis_count
                )
                vis_count += 1
            
            # Calculate metrics
            if pred is None or len(pred) == 0:
                if len(gt) > 0:
                    # if we have labels but no detections
                    correct = np.zeros((0, 10), dtype=bool)
                    metrics.append((
                        correct,
                        np.zeros(0),  # conf
                        np.zeros(0),  # pred_cls
                        np.zeros(0)   # target_cls
                    ))
                continue
                
            # Calculate AP metrics - adapted from PyTorch version
            # Convert TensorFlow tensors to numpy for processing
            pred_np = pred.numpy() if hasattr(pred, 'numpy') else pred
            gt_np = gt.numpy() if hasattr(gt, 'numpy') else gt
            
            # Scale detection boxes back to original shape
            det_clone = pred_np.copy()
            # Note: scale function would need to be adapted for numpy arrays
            # For now, we'll use a simplified version
            
            # Convert label xywh -> xyxy
            if len(gt_np) > 0:
                # gt_np format: [batch_idx, cls, x, y, w, h] (6 columns)
                # We need: label_boxes = [cls, x1, y1, x2, y2] (5 columns)
                label_boxes = np.zeros((len(gt_np), 5), dtype=gt_np.dtype)
                label_boxes[:, 0] = gt_np[:, 1]  # cls
                label_boxes[:, 1] = gt_np[:, 2] - gt_np[:, 4] / 2.0  # x1 = x - w/2
                label_boxes[:, 2] = gt_np[:, 3] - gt_np[:, 5] / 2.0  # y1 = y - h/2
                label_boxes[:, 3] = gt_np[:, 2] + gt_np[:, 4] / 2.0  # x2 = x + w/2
                label_boxes[:, 4] = gt_np[:, 3] + gt_np[:, 5] / 2.0  # y2 = y + h/2
                
                # IoU vector for mAP@0.5:0.95
                iou_v = np.linspace(0.5, 0.95, 10)
                n_iou = len(iou_v)
                
                # Compute IoU matching
                correct = np.zeros((det_clone.shape[0], n_iou), dtype=bool)
                t_tensor = label_boxes[:, :5]  # (class, x1, y1, x2, y2)
                
                # Calculate IoU between predictions and ground truth
                # Simplified IoU calculation
                for j in range(len(iou_v)):
                    iou_threshold = iou_v[j]
                    
                    # For each detection, find best matching ground truth
                    for det_idx in range(det_clone.shape[0]):
                        det_box = det_clone[det_idx, :4]
                        det_class = det_clone[det_idx, 5]
                        
                        best_iou = 0
                        best_gt_idx = -1
                        
                        for gt_idx in range(len(t_tensor)):
                            gt_box = t_tensor[gt_idx, 1:5]
                            gt_class = t_tensor[gt_idx, 0]
                            
                            # Check class match (round to handle floating point)
                            if int(round(det_class)) == int(round(gt_class)):
                                # Calculate IoU (simplified)
                                x1 = max(det_box[0], gt_box[0])
                                y1 = max(det_box[1], gt_box[1])
                                x2 = min(det_box[2], gt_box[2])
                                y2 = min(det_box[3], gt_box[3])
                                
                                if x2 > x1 and y2 > y1:
                                    intersection = (x2 - x1) * (y2 - y1)
                                    det_area = (det_box[2] - det_box[0]) * (det_box[3] - det_box[1])
                                    gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
                                    union = det_area + gt_area - intersection
                                    iou = intersection / union if union > 0 else 0
                                    
                                    if iou > best_iou:
                                        best_iou = iou
                                        best_gt_idx = gt_idx
                        
                        if best_iou >= iou_threshold:
                            correct[det_idx, j] = True
                
                # Gather conf, pred_cls, true_cls
                conf = det_clone[:, 4]
                pred_cls = det_clone[:, 5]
                true_cls = t_tensor[:, 0]
                
                metrics.append((correct, conf, pred_cls, true_cls))
            else:
                # No labels => no matches
                correct = np.zeros((det_clone.shape[0], 10), dtype=bool)
                conf = det_clone[:, 4]
                pred_cls = det_clone[:, 5]
                target_cls = np.zeros(0)  # no ground truths
                metrics.append((correct, conf, pred_cls, target_cls))
            
    # Compute final metrics
    if len(metrics) > 0:
        metrics = [np.concatenate(x, 0) for x in zip(*metrics)]
        tp, fp, m_pre, m_rec, map50, mean_ap = compute_ap(*metrics)
    else:
        tp = fp = m_pre = m_rec = map50 = mean_ap = 0

    print(f'Precision: {m_pre:.3f}, Recall: {m_rec:.3f}, mAP50: {map50:.3f}, mAP: {mean_ap:.3f}')
    return tp, fp, m_pre, m_rec, map50, mean_ap

def main():
    parser = argparse.ArgumentParser()
    
    # Set global policy (for mixed-precision)
    tf.keras.mixed_precision.set_global_policy(
        'mixed_float16' if DTYPE == tf.float16 else 'float32'
    )
    parser.add_argument('--input-size', default='256', type=str)
    parser.add_argument('--batch-size', default=4, type=int)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--quantized', action='store_true', help='Use QKeras quantized model for HLS4ml FPGA synthesis')
    parser.add_argument('--yaml_file', type=str, default='utils/args_bionano.yaml')
    parser.add_argument('--save-path', type=str, default='./results/rect_256x128_cleaned')
    parser.add_argument('--dataset-dir', type=str, default='./Dataset/bionano_cellv2')

    args = parser.parse_args()

    # Parse image size
    if 'x' in args.input_size:
        h, w = map(int, args.input_size.split('x'))
        args.img_size = (h, w)
    else:
        size = int(args.input_size)
        args.img_size = (size, size)

    # Load params
    with open(args.yaml_file, 'r') as f:
        params = yaml.safe_load(f)

    if args.train:
        train(args, params)
    if args.test:
        test(args, params)

if __name__ == '__main__':
    main()
