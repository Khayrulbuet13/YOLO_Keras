"""
Knowledge Distillation Training Script for YOLO

Trains a quantized student model using knowledge distillation from a full-precision teacher.
Combines task loss (standard YOLO loss) with KD loss (KL divergence on outputs).
"""

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

from nets.tinysimov35_keras_functional import yolo_v8_s_functional, decode_predictions
from nets.tinysimov35_keras_quantized_functional import yolo_v8_s_quantized_functional
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
from utils.kd_loss import compute_kd_loss

# Import QKeras utilities for quantized model handling
try:
    from qkeras.estimate import print_qstats
    from qkeras.utils import model_save_quantized_weights
    QKERAS_AVAILABLE = True
except ImportError:
    QKERAS_AVAILABLE = False
    print("[WARNING] QKeras utilities not available. Quantized model saving may not work properly.")

# Define global dtype for consistent type handling
DTYPE = tf.float32

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
            elif any(bn in var_name for bn in ['/batch_normalization', '/bn', '/batchnorm', '_bn']):
                self.bn_params.add(var.name)
            elif var_name.endswith('/kernel:0'):
                self.weight_params.add(var.name)
        self.weight_params = self.weight_params - self.bn_params

        # Initialize momentum variables as tf.Variable
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
    
    # Load teacher model (frozen)
    print("[INFO] Loading teacher model (full-precision functional)")
    teacher = yolo_v8_s_functional(num_classes, img_size=args.img_size, dtype=DTYPE)
    teacher.load_weights(args.teacher_weights)
    teacher.trainable = False  # Freeze teacher
    print(f"[INFO] Teacher weights loaded from: {args.teacher_weights}")
    print("[INFO] Teacher model frozen (not trainable)")
    
    # Load student model (trainable, quantized)
    print("[INFO] Loading student model (quantized)")
    student = yolo_v8_s_quantized_functional(num_classes, img_size=args.img_size, dtype=DTYPE)
    
    # Transfer weights from teacher or pretrained checkpoint if available
    if args.pretrained_weights:
        print(f"[INFO] Initializing student from pre-trained weights: {args.pretrained_weights}")
        # Try to load directly first (if weights are from quantized model)
        try:
            student.load_weights(args.pretrained_weights)
            print("[INFO] Student weights loaded directly")
        except Exception as e:
            print(f"[INFO] Direct loading failed, attempting weight transfer from float32 model...")
            # Load float32 model and transfer weights
            float_model = yolo_v8_s_functional(num_classes, img_size=args.img_size, dtype=DTYPE)
            float_model.load_weights(args.pretrained_weights)
            
            float_layer_dict = {layer.name: layer for layer in float_model.layers}
            transferred = 0
            skipped = 0
            
            for qlayer in student.layers:
                # Map quantized layer names to float32 layer names
                float_name = qlayer.name.replace('qconv', 'conv').replace('qrelu', 'relu')
                
                if float_name in float_layer_dict:
                    flayer = float_layer_dict[float_name]
                    try:
                        if len(flayer.get_weights()) > 0:
                            qlayer.set_weights(flayer.get_weights())
                            print(f"  ✓ {flayer.name} -> {qlayer.name}")
                            transferred += 1
                    except Exception as e:
                        print(f"  ⚠ Failed {flayer.name} -> {qlayer.name}: {e}")
                        skipped += 1
            
            print(f"[INFO] Weight transfer complete: {transferred} layers transferred, {skipped} skipped")
            del float_model
    else:
        print("[INFO] Student initialized with random weights (no warmstart)")
    
    # Create model directory
    if args.local_rank == 0:
        os.makedirs(args.save_path, exist_ok=True)
    
    # Optimizer & Scheduler
    accumulate = max(round(64 / (args.batch_size)), 1)
    params['weight_decay'] *= args.batch_size * accumulate / 64

    # Use custom multi-group optimizer for student only
    optimizer = MultiGroupOptimizer(
        student, 
        params['lr0'], 
        params['momentum'], 
        params['weight_decay'], 
        nesterov=True
    )

    lr_func = learning_rate(args, params)
    
    # EMA for student
    ema = EMA(student) if args.local_rank == 0 else None
    
    # Datasets
    train_filenames = []
    with open(os.path.join(args.dataset_dir, 'train.txt')) as ftrain:
        for line in ftrain.readlines():
            line = line.rstrip().split('/')[-1]
            train_filenames.append(os.path.join(args.dataset_dir, 'images/train', line))

    train_dataset = Dataset(train_filenames, args.img_size, params, True, dtype=DTYPE)
    def data_generator():
        batch_samples = []
        batch_targets = []
        batch_shapes = []
        batch_size = args.batch_size
        
        for i in range(len(train_dataset)):
            sample, target, shapes = train_dataset[i]
            
            # Insert the batch index in the first column
            batch_index = len(batch_samples)
            if target.shape[0] > 0:
                img_idx = tf.fill([tf.shape(target)[0], 1],
                                   tf.cast(batch_index, target.dtype))
                target = tf.concat([img_idx, target[:, 1:]], axis=1)

            batch_samples.append(sample)
            if target.shape[0] > 0:
                batch_targets.append(target)
            batch_shapes.append(shapes)
            
            # When we have a complete batch, yield it
            if (i + 1) % batch_size == 0:
                stacked_samples = tf.cast(np.stack(batch_samples, axis=0), DTYPE)
                
                if batch_targets:
                    stacked_targets = tf.cast(np.concatenate(batch_targets, axis=0), DTYPE)
                else:
                    stacked_targets = tf.zeros((0, 6), dtype=DTYPE)
                
                stacked_shapes = tf.cast(np.stack(batch_shapes, axis=0), tf.float32)
                
                yield stacked_samples, stacked_targets, stacked_shapes
                
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
    criterion = ComputeLoss(student, params, dtype=DTYPE)
    
    # Training loop
    best = 0
    patience = 200  # Increased patience for KD training
    patience_counter = 0
    print(f"[INFO] Using patience for KD training: {patience} epochs")
    num_batch = len(train_dataset) // args.batch_size
    num_warmup = max(round(params['warmup_epochs'] * num_batch), 1000)
    
    # CSV logger
    csv_path = os.path.join(args.save_path, 'step.csv')
    csv_file = open(csv_path, 'w', newline='')
    writer = csv.DictWriter(csv_file, fieldnames=[
        'epoch', 
        'task_loss', 'kd_loss', 'box_kd', 'cls_kd', 'total_loss',
        'train_mAP@50', 'train_mAP', 'train_Precision', 'train_Recall', 'train_F1',
        'val_mAP@50', 'val_mAP', 'val_Precision', 'val_Recall', 'val_F1'
    ])
    writer.writeheader()

    print(f"\n{'='*80}")
    print(f"KNOWLEDGE DISTILLATION TRAINING")
    print(f"{'='*80}")
    print(f"Teacher: Full-precision functional model")
    print(f"Student: Quantized functional model")
    print(f"KD Alpha: {args.kd_alpha}")
    print(f"KD Temperature: {args.kd_temperature}")
    print(f"Box KD Weight: {args.kd_box_weight}")
    print(f"Cls KD Weight: {args.kd_cls_weight}")
    print(f"{'='*80}\n")

    for epoch in range(args.epochs):
        m_task_loss = AverageMeter()
        m_kd_loss = AverageMeter()
        m_box_kd = AverageMeter()
        m_cls_kd = AverageMeter()
        m_total_loss = AverageMeter()
        
        # Turn off mosaic for last 10 epochs
        if args.epochs - epoch == 10:
            train_dataset.mosaic = False
            print(f"\n[main_kd.py::train] TURNING OFF MOSAIC at epoch {epoch+1}/{args.epochs}")

        p_bar = tqdm(enumerate(train_loader), total=num_batch, desc=f'Epoch {epoch+1}/{args.epochs}')
        
        for i, (samples, targets, shapes) in p_bar:
            x = i + num_batch * epoch
            
            # Warmup
            if x <= num_warmup:
                xp = [0, num_warmup]
                fp = [1, 64 / args.batch_size]
                accumulate = max(1, np.interp(x, xp, fp).round())
                
                lr_bias = np.interp(x, xp, [params['warmup_bias_lr'], params['lr0'] * lr_func(epoch)])
                lr_weight = np.interp(x, xp, [0.0, params['lr0'] * lr_func(epoch)])
                lr_bn = np.interp(x, xp, [0.0, params['lr0'] * lr_func(epoch)])
                
                optimizer.set_learning_rates(lr_bias, lr_weight, lr_bn)
                
                momentum = np.interp(x, xp, [params['warmup_momentum'], params['momentum']])
            else:
                lr = params['lr0'] * lr_func(epoch)
                optimizer.set_learning_rates(lr, lr, lr)

            # Forward pass: teacher (frozen) + student (trainable)
            with tf.GradientTape() as tape:
                # Teacher forward pass (inference mode, frozen)
                teacher_output = teacher(samples, training=False)
                
                # Student forward pass (training mode)
                student_output = student(samples, training=True)
                
                # Task loss (standard YOLO loss on student)
                task_loss = criterion(student_output, targets)
                task_loss *= args.batch_size  # Scale for multi-GPU compatibility
                
                # KD loss (KL divergence between teacher and student outputs)
                kd_losses = compute_kd_loss(
                    teacher_output, 
                    student_output,
                    temperature=args.kd_temperature,
                    box_weight=args.kd_box_weight,
                    cls_weight=args.kd_cls_weight,
                    dfl_ch=student.dfl_ch
                )
                
                kd_loss_total = kd_losses['total']
                box_kd = kd_losses['box']
                cls_kd = kd_losses['cls']
                
                # Combined loss
                total_loss = (1-args.kd_alpha)*task_loss + args.kd_alpha * kd_loss_total
                
                # Update meters
                m_task_loss.update(task_loss.numpy(), samples.shape[0])
                m_kd_loss.update(kd_loss_total.numpy(), samples.shape[0])
                m_box_kd.update(box_kd.numpy(), samples.shape[0])
                m_cls_kd.update(cls_kd.numpy(), samples.shape[0])
                m_total_loss.update(total_loss.numpy(), samples.shape[0])

            # Backward pass (only through student)
            gradients = tape.gradient(total_loss, student.trainable_variables)
            
            # Clip gradients for stability
            gradients, global_norm = tf.clip_by_global_norm(gradients, 1.0)
            
            optimizer.apply_gradients(zip(gradients, student.trainable_variables))
            
            # Update EMA
            if ema:
                ema.update()
            
            # Update progress bar with detailed loss info
            # Convert to scalar if needed (handle numpy arrays)
            def to_scalar(val):
                return float(val.item()) if hasattr(val, 'item') else float(val)
            
            p_bar.set_postfix({
                'total': f'{to_scalar(m_total_loss.avg):.3f}',
                'task': f'{to_scalar(m_task_loss.avg):.3f}',
                'kd': f'{to_scalar(m_kd_loss.avg):.3f}',
                'box_kd': f'{to_scalar(m_box_kd.avg):.3f}',
                'cls_kd': f'{to_scalar(m_cls_kd.avg):.3f}'
            })

        # Evaluation
        if args.local_rank == 0:
            # Create EMA model for evaluation if EMA is enabled
            eval_model = student
            if ema:
                eval_model = yolo_v8_s_quantized_functional(num_classes, img_size=args.img_size)
                eval_model.set_weights([w.numpy() for w in ema.ema_weights])
            
            # Train evaluation
            train_tp, train_fp, train_precision, train_recall, train_map50, train_mean_ap = test(args, params, eval_model, is_train=True)
            train_f1 = 2 * train_precision * train_recall / (train_precision + train_recall + 1e-16)
            
            # Validation evaluation
            val_tp, val_fp, val_precision, val_recall, val_map50, val_mean_ap = test(args, params, eval_model, is_train=False)
            val_f1 = 2 * val_precision * val_recall / (val_precision + val_recall + 1e-16)

            # Log results
            # Helper to convert to scalar
            def to_scalar(val):
                return float(val.item()) if hasattr(val, 'item') else float(val)
            
            writer.writerow({
                'epoch': str(epoch + 1).zfill(3),
                'task_loss': f'{to_scalar(m_task_loss.avg):.4f}',
                'kd_loss': f'{to_scalar(m_kd_loss.avg):.4f}',
                'box_kd': f'{to_scalar(m_box_kd.avg):.4f}',
                'cls_kd': f'{to_scalar(m_cls_kd.avg):.4f}',
                'total_loss': f'{to_scalar(m_total_loss.avg):.4f}',
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
                patience_counter = 0
                
                save_model = student if not ema else eval_model
                save_model.save_weights(os.path.join(args.save_path, 'best.weights.h5'))
                print(f'Epoch {epoch + 1}: New best model saved (mAP: {best:.4f})')
                
                # Save quantized weights
                if QKERAS_AVAILABLE:
                    print("[INFO] Saving quantized weights for HLS4ml deployment...")
                    model_save_quantized_weights(save_model)
            else:
                patience_counter += 1
            
            # Save last weights
            student.save_weights(os.path.join(args.save_path, 'last.weights.h5'))
            
            # Early stopping check
            if patience_counter >= patience:
                print(f'\nEarly stopping triggered after {epoch + 1} epochs (no improvement for {patience} epochs)')
                break

    csv_file.close()
    
    # Print quantization statistics
    if QKERAS_AVAILABLE and args.local_rank == 0:
        print("\n" + "="*80)
        print("QUANTIZATION STATISTICS")
        print("="*80)
        try:
            print_qstats(student)
        except (AttributeError, Exception) as e:
            print(f"Note: print_qstats() not available for custom Model classes")
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

    dataset = Dataset(filenames, args.img_size, params, False, dtype=DTYPE)
    def test_data_generator():
        for i in range(len(dataset)):
            sample, target, shapes = dataset[i]
            
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
        model = yolo_v8_s_quantized_functional(len(params['names']), img_size=args.img_size)
        model.load_weights(model_path)

    # Prepare for evaluation
    class_colors = generate_colors(len(params['names']))
    results_dir = os.path.join(args.save_path, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    metrics = []
    vis_count = 0

    for samples, targets, shapes in tqdm(loader, desc='Evaluating'):
        outputs = model(samples, training=False)
        
        # Decode predictions if using functional model
        if len(outputs.shape) == 4:
            outputs = decode_predictions(outputs, model.stride, model.nc, model.dfl_ch, dtype=DTYPE)
        
        detections = non_max_suppression(outputs, 0.25, 0.45)
        
        # Scale GT coordinates
        _, h, w, _ = samples.shape
        scale_tensor = tf.constant([1.0, 1.0, float(w), float(h), float(w), float(h)], dtype=targets.dtype)
        targets = targets * scale_tensor
        
        # Process each image in batch
        for i in range(samples.shape[0]):
            pred = detections[i]

            gt_for_image = targets[i]
            valid_mask = tf.reduce_sum(tf.abs(gt_for_image), axis=1) > 0
            gt = tf.boolean_mask(gt_for_image, valid_mask)
            if tf.shape(gt)[0] == 0:
                gt = tf.zeros((0, 5), dtype=tf.float32)
                    
            # Visualization (skip for train set)
            if not is_train and vis_count < 5:
                single_sample = samples[i:i+1]
                single_out = [pred]
                single_shapes = shapes[i:i+1]
                
                single_gt = []
                if tf.shape(gt)[0] > 0:
                    gt_boxes = tf.identity(gt)
                    h0, w0 = shapes[i][0]
                    pad_info = shapes[i][2]
                    pad_w, pad_h = pad_info[0], pad_info[1]
                    
                    _, h, w, _ = samples.shape
                    
                    class_ids = gt_boxes[:, 1]
                    scaled_w = float(w) - 2 * pad_w
                    scaled_h = float(h) - 2 * pad_h

                    scale_x = w0 / scaled_w if scaled_w > 0 else 1.0
                    scale_y = h0 / scaled_h if scaled_h > 0 else 1.0

                    coords = gt_boxes[:, 2:]
                    
                    x_center = (coords[:, 0] - pad_w) * scale_x
                    y_center = (coords[:, 1] - pad_h) * scale_y
                    width = coords[:, 2] * scale_x
                    height = coords[:, 3] * scale_y
                    
                    for j in range(tf.shape(gt_boxes)[0]):
                        single_gt.append(tf.stack([
                            tf.constant(0.0, dtype=tf.float32),
                            class_ids[j],
                            x_center[j],
                            y_center[j],
                            width[j],
                            height[j]
                        ]))
                    single_gt = tf.stack(single_gt)
                else:
                    single_gt = tf.zeros((0, 6), dtype=tf.float32)
                
                visualize_predictions(
                    single_sample,
                    single_out,
                    single_gt,
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
                    correct = np.zeros((0, 10), dtype=bool)
                    metrics.append((
                        correct,
                        np.zeros(0),
                        np.zeros(0),
                        np.zeros(0)
                    ))
                continue
                
            pred_np = pred.numpy() if hasattr(pred, 'numpy') else pred
            gt_np = gt.numpy() if hasattr(gt, 'numpy') else gt
            
            det_clone = pred_np.copy()
            
            if len(gt_np) > 0:
                label_boxes = np.zeros((len(gt_np), 5), dtype=gt_np.dtype)
                label_boxes[:, 0] = gt_np[:, 1]
                label_boxes[:, 1] = gt_np[:, 2] - gt_np[:, 4] / 2.0
                label_boxes[:, 2] = gt_np[:, 3] - gt_np[:, 5] / 2.0
                label_boxes[:, 3] = gt_np[:, 2] + gt_np[:, 4] / 2.0
                label_boxes[:, 4] = gt_np[:, 3] + gt_np[:, 5] / 2.0
                
                iou_v = np.linspace(0.5, 0.95, 10)
                n_iou = len(iou_v)
                
                correct = np.zeros((det_clone.shape[0], n_iou), dtype=bool)
                t_tensor = label_boxes[:, :5]
                
                for j in range(len(iou_v)):
                    iou_threshold = iou_v[j]
                    
                    for det_idx in range(det_clone.shape[0]):
                        det_box = det_clone[det_idx, :4]
                        det_class = det_clone[det_idx, 5]
                        
                        best_iou = 0
                        best_gt_idx = -1
                        
                        for gt_idx in range(len(t_tensor)):
                            gt_box = t_tensor[gt_idx, 1:5]
                            gt_class = t_tensor[gt_idx, 0]
                            
                            if int(round(det_class)) == int(round(gt_class)):
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
                
                conf = det_clone[:, 4]
                pred_cls = det_clone[:, 5]
                true_cls = t_tensor[:, 0]
                
                metrics.append((correct, conf, pred_cls, true_cls))
            else:
                correct = np.zeros((det_clone.shape[0], 10), dtype=bool)
                conf = det_clone[:, 4]
                pred_cls = det_clone[:, 5]
                target_cls = np.zeros(0)
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
    
    # Set global policy
    tf.keras.mixed_precision.set_global_policy(
        'mixed_float16' if DTYPE == tf.float16 else 'float32'
    )
    
    # Standard arguments
    parser.add_argument('--input-size', default='256', type=str)
    parser.add_argument('--batch-size', default=4, type=int)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--yaml_file', type=str, default='utils/args_bionano.yaml')
    parser.add_argument('--save-path', type=str, default='./results/kd_quantized')
    parser.add_argument('--dataset-dir', type=str, default='./Dataset/bionano_cellv2')
    
    # KD-specific arguments
    parser.add_argument('--teacher-weights', type=str, required=True,
                        help='Path to pre-trained teacher (float32) weights')
    parser.add_argument('--pretrained-weights', type=str, default=None,
                        help='Path to pre-trained weights for student initialization (optional warmstart)')
    parser.add_argument('--kd-alpha', type=float, default=0.5,
                        help='Weight for KD loss (default: 0.5)')
    parser.add_argument('--kd-temperature', type=float, default=4.0,
                        help='Temperature for softening distributions (default: 4.0)')
    parser.add_argument('--kd-box-weight', type=float, default=1.0,
                        help='Weight for box KD loss (default: 1.0)')
    parser.add_argument('--kd-cls-weight', type=float, default=1.0,
                        help='Weight for class KD loss (default: 1.0)')

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
