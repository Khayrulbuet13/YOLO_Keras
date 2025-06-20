import argparse
import csv
import math
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, callbacks
import yaml
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

from nets.tinysimov35_keras import yolo_v8_s
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

# Define global dtype for consistent type handling
DTYPE = tf.float32  # Central dtype definition (change to float16 for mixed-precision)

def learning_rate(args, params):
    def fn(epoch):
        return (1 - epoch / args.epochs) * (1.0 - params['lrf']) + params['lrf']
    return fn

def train(args, params):
    # Initialize with central dtype
    num_classes = len(params['names'].values())
    print(f"[main_keras.py::train] Number of classes: {num_classes}")
    model = yolo_v8_s(num_classes, img_size=args.img_size, dtype=DTYPE)  # Pass dtype to model
    print(f"[main_keras.py::train] Model created with input size: {args.img_size}")
    
    # Create model directory
    if args.local_rank == 0:
        os.makedirs(args.save_path, exist_ok=True)
    
    # Optimizer & Scheduler
    accumulate = max(round(64 / (args.batch_size)), 1)
    params['weight_decay'] *= args.batch_size * accumulate / 64

    optimizer = optimizers.SGD(
        learning_rate=params['lr0'],
        momentum=params['momentum'],
        nesterov=True
    )

    lr_scheduler = callbacks.LearningRateScheduler(learning_rate(args, params))
    
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
            
            # DEBUG: Print dataset output for first few samples
            if i < 2:
                print(f"\n--- [main_keras.py::data_generator] DATASET OUTPUT DEBUG (sample {i}) ---")
                print(f"[main_keras.py::data_generator] Sample shape: {sample.shape}")
                print(f"[main_keras.py::data_generator] Target shape: {target.shape}")
                print(f"[main_keras.py::data_generator] Shapes from dataset: {shapes}")
                print(f"[main_keras.py::data_generator] Shapes type: {type(shapes)}")
                print(f"[main_keras.py::data_generator] Shapes dtype: {shapes.dtype if hasattr(shapes, 'dtype') else 'N/A'}")
                if hasattr(shapes, 'shape'):
                    print(f"[main_keras.py::data_generator] Shapes shape: {shapes.shape}")
                    if shapes.shape[0] >= 2 and shapes.shape[1] >= 2:
                        print(f"[main_keras.py::data_generator] Original size: [{shapes[0, 0]}, {shapes[0, 1]}]")
                        print(f"[main_keras.py::data_generator] Ratio/Padding: [{shapes[1, 0]}, {shapes[1, 1]}]")
                    else:
                        print(f"[main_keras.py::data_generator] Shapes content: {shapes}")
                else:
                    print(f"[main_keras.py::data_generator] Shapes content: {shapes}")
            
            # DEBUG: Print target info for first few samples
            if i < 5 and target.shape[0] > 0:
                print(f"[main_keras.py::data_generator] Target shape: {target.shape}")
                print(f"[main_keras.py::data_generator] Target sample: {target[:3] if len(target) > 0 else 'None'}")
            
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
            
            # When we have a complete batch, yield it
            if (i + 1) % batch_size == 0 or i == len(train_dataset) - 1:
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
                
                # Debug dtype info
                if i < 2:
                    print(f"Batch samples dtype: {stacked_samples.dtype}")
                    print(f"Batch targets dtype: {stacked_targets.dtype}")
                
                # Debug info
                if i < 5:
                    print(f"[main_keras.py::data_generator] Batch samples shape: {stacked_samples.shape}")
                    print(f"[main_keras.py::data_generator] Batch targets shape: {stacked_targets.shape}")
                    print(f"[main_keras.py::data_generator] Batch shapes shape: {stacked_shapes.shape}")
                
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
    num_batch = math.ceil(len(train_dataset) / args.batch_size)
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

        p_bar = tqdm(enumerate(train_loader), total=num_batch, desc=f'Epoch {epoch+1}/{args.epochs}')
        
        for i, (samples, targets, shapes) in p_bar:
            x = i + num_batch * epoch
            
            # Debug: Print data info for first few batches
            if epoch == 0 and i < 3:
                print(f"\n--- [main_keras.py::train] KERAS BATCH {i} DEBUG ---")
                print(f"[main_keras.py::train] Samples shape: {samples.shape}")
                print(f"[main_keras.py::train] Samples dtype: {samples.dtype}")
                print(f"[main_keras.py::train] Samples min/max: {tf.reduce_min(samples):.3f}/{tf.reduce_max(samples):.3f}")
                print(f"[main_keras.py::train] Targets shape: {targets.shape}")
                print(f"[main_keras.py::train] Targets dtype: {targets.dtype}")
                print(f"[main_keras.py::train] Number of targets: {tf.shape(targets)[0]}")
                if tf.reduce_sum(tf.cast(targets[:, 0] > 0, tf.int32)) > 0:
                    valid_targets = tf.boolean_mask(targets, targets[:, 0] > 0)
                    print(f"[main_keras.py::train] Target sample: {valid_targets[:5] if len(valid_targets) > 0 else 'None'}")
                print(f"[main_keras.py::train] Shapes: {shapes}")
                print(f"[main_keras.py::train] Shapes type: {type(shapes)}")
                print(f"[main_keras.py::train] Shapes dtype: {shapes.dtype}")
                print(f"[main_keras.py::train] Shapes shape: {shapes.shape}")
                
                # Detailed shapes analysis
                for batch_idx in range(min(shapes.shape[0], 2)):  # Check first 2 items in batch
                    print(f"[main_keras.py::train] Batch item {batch_idx} shapes: {shapes[batch_idx]}")
                    if shapes.shape[-1] >= 2 and shapes.shape[-2] >= 2:
                        print(f"[main_keras.py::train] Batch item {batch_idx} original size: [{shapes[batch_idx, 0, 0]}, {shapes[batch_idx, 0, 1]}]")
                        print(f"[main_keras.py::train] Batch item {batch_idx} ratio/padding: [{shapes[batch_idx, 1, 0]}, {shapes[batch_idx, 1, 1]}]")
                    else:
                        print(f"[main_keras.py::train] Batch item {batch_idx} unexpected shapes format: {shapes[batch_idx]}")
            
            # Warmup
            if x <= num_warmup:
                xp = [0, num_warmup]
                fp = [1, 64 / args.batch_size]
                accumulate = max(1, np.interp(x, xp, fp).round())
                
                # Adjust learning rate
                lr = np.interp(x, xp, [params['warmup_bias_lr'], params['lr0'] * learning_rate(args, params)(epoch)])
                optimizer.learning_rate = lr
                
                # Adjust momentum
                if hasattr(optimizer, 'momentum'):
                    optimizer.momentum = np.interp(x, xp, [params['warmup_momentum'], params['momentum']])

            # Forward pass
            with tf.GradientTape() as tape:
                outputs = model(samples, training=True)
                
                # Debug: Print model outputs for first few batches
                if epoch == 0 and i < 3:
                    print(f"\n--- [main_keras.py::train] KERAS MODEL OUTPUT DEBUG ---")
                    if isinstance(outputs, (list, tuple)):
                        print(f"[main_keras.py::train] Number of output tensors: {len(outputs)}")
                        for idx, out in enumerate(outputs):
                            print(f"[main_keras.py::train] Output {idx} shape: {out.shape}")
                            print(f"[main_keras.py::train] Output {idx} min/max: {tf.reduce_min(out):.6f}/{tf.reduce_max(out):.6f}")
                            print(f"[main_keras.py::train] Output {idx} mean/std: {tf.reduce_mean(out):.6f}/{tf.math.reduce_std(out):.6f}")
                    else:
                        print(f"[main_keras.py::train] Output shape: {outputs.shape}")
                        print(f"[main_keras.py::train] Output min/max: {tf.reduce_min(outputs):.6f}/{tf.reduce_max(outputs):.6f}")
                        print(f"[main_keras.py::train] Output mean/std: {tf.reduce_mean(outputs):.6f}/{tf.math.reduce_std(outputs):.6f}")
                
                loss = criterion(outputs, targets)
                
                # Debug: Print loss details for first few batches
                if epoch == 0 and i < 3:
                    print(f"\n--- [main_keras.py::train] KERAS LOSS DEBUG ---")
                    print(f"[main_keras.py::train] Loss value: {loss.numpy().item():.6f}")
                    print(f"[main_keras.py::train] Loss dtype: {loss.dtype}")
                
                m_loss.update(loss.numpy(), samples.shape[0])

            # Backward pass
            gradients = tape.gradient(loss, model.trainable_variables)
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
                model.save_weights(os.path.join(args.save_path, 'best.h5'))
            model.save_weights(os.path.join(args.save_path, 'last.h5'))

    csv_file.close()

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
            
            # DEBUG: Print test dataset output for first few samples
            if i < 5:
                print(f"\n--- [main_keras.py::test_data_generator] TEST DATASET OUTPUT DEBUG (sample {i}) ---")
                print(f"[main_keras.py::test_data_generator] Sample shape: {sample.shape}")
                print(f"[main_keras.py::test_data_generator] Target shape: {target.shape}")
                print(f"[main_keras.py::test_data_generator] Shapes from dataset: {shapes}")
                print(f"[main_keras.py::test_data_generator] Shapes type: {type(shapes)}")
                print(f"[main_keras.py::test_data_generator] Shapes dtype: {shapes.dtype if hasattr(shapes, 'dtype') else 'N/A'}")
                if hasattr(shapes, 'shape'):
                    print(f"[main_keras.py::test_data_generator] Shapes shape: {shapes.shape}")
                    if shapes.shape[0] >= 2 and shapes.shape[1] >= 2:
                        print(f"[main_keras.py::test_data_generator] Original size: [{shapes[0, 0]}, {shapes[0, 1]}]")
                        print(f"[main_keras.py::test_data_generator] Ratio/Padding: [{shapes[1, 0]}, {shapes[1, 1]}]")
                    else:
                        print(f"[main_keras.py::test_data_generator] Shapes content: {shapes}")
                else:
                    print(f"[main_keras.py::test_data_generator] Shapes content: {shapes}")
            
            # Pad targets to maximum possible size (e.g., 100 objects max)
            max_objects = 100
            padded_target = np.zeros((max_objects, 6), dtype=np.float32)
            if target.shape[0] > 0:
                n_objects = min(target.shape[0], max_objects)
                padded_target[:n_objects] = target[:n_objects]
                
            # DEBUG: Print padded target info for first few samples
            if i < 5:
                print(f"[main_keras.py::test_data_generator] Padded target shape: {padded_target.shape}")
                print(f"[main_keras.py::test_data_generator] Non-zero targets: {np.sum(padded_target[:, 0] > 0)}")
                if np.sum(padded_target[:, 0] > 0) > 0:
                    valid_targets = padded_target[padded_target[:, 0] > 0]
                    print(f"[main_keras.py::test_data_generator] Valid targets sample: {valid_targets[:3] if len(valid_targets) > 0 else 'None'}")
                    
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
        model_path = os.path.join(args.save_path, 'best.h5')
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
        print(f"[main_keras.py::test] Model outputs shape: {outputs.shape if isinstance(outputs, tf.Tensor) else [o.shape for o in outputs]}")
        detections = non_max_suppression(outputs, 0.25, 0.45)
        
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
                    # Scale coordinates back to original image size
                    gt_boxes = tf.identity(gt)  # Clone the tensor
                    h0, w0 = shapes[i][0]  # Original height, width
                    pad_info = shapes[i][2]  # Get padding values [pad_w, pad_h]
                    pad_w, pad_h = pad_info[0], pad_info[1]
                    
                    # Get current image dimensions
                    _, h, w, _ = samples.shape
                    
                    class_ids = gt_boxes[:, 1]  # Class column
                    # Calculate scaled dimensions after removing padding
                    scaled_w = w - 2 * pad_w
                    scaled_h = h - 2 * pad_h

                    # Compute scaling factors
                    scale_x = w0 / scaled_w if scaled_w > 0 else 1.0
                    scale_y = h0 / scaled_h if scaled_h > 0 else 1.0

                    # Adjust coordinates (gt format: [img_idx, cls, x_center, y_center, width, height])
                    coords = gt_boxes[:, 2:]  # Get [x_center, y_center, width, height]
                    
                    # Convert normalized coordinates to pixel coordinates and scale
                    x_center = (coords[:, 0] * w - pad_w) * scale_x  # x_center
                    y_center = (coords[:, 1] * h - pad_h) * scale_y  # y_center
                    width = coords[:, 2] * w * scale_x  # width
                    height = coords[:, 3] * h * scale_y  # height
                    
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

                # DEBUG: Print shapes before visualize_predictions
                print(f"\n=== DEBUG: visualize_predictions input shapes (vis_count={vis_count}) ===")
                print(f"single_sample shape: {single_sample.shape}")
                print(f"single_sample dtype: {single_sample.dtype}")
                print(f"single_sample min/max: {tf.reduce_min(single_sample):.3f}/{tf.reduce_max(single_sample):.3f}")
                
                print(f"single_out type: {type(single_out)}")
                if isinstance(single_out, list):
                    print(f"single_out length: {len(single_out)}")
                    for idx, out_item in enumerate(single_out):
                        if out_item is not None:
                            print(f"single_out[{idx}] shape: {out_item.shape}")
                            print(f"single_out[{idx}] dtype: {out_item.dtype}")
                            if hasattr(out_item, 'shape') and tf.size(out_item) > 0:
                                print(f"single_out[{idx}] min/max: {tf.reduce_min(out_item):.6f}/{tf.reduce_max(out_item):.6f}")
                            else:
                                print(f"single_out[{idx}]: empty tensor")
                        else:
                            print(f"single_out[{idx}]: None")
                else:
                    print(f"single_out shape: {single_out.shape}")
                    print(f"single_out dtype: {single_out.dtype}")
                    if hasattr(single_out, 'shape') and tf.size(single_out) > 0:
                        print(f"single_out min/max: {tf.reduce_min(single_out):.6f}/{tf.reduce_max(single_out):.6f}")
                    else:
                        print(f"single_out: empty tensor")
                
                print(f"single_gt shape: {single_gt.shape}")
                print(f"single_gt dtype: {single_gt.dtype}")
                if hasattr(single_gt, 'shape') and tf.size(single_gt) > 0:
                    print(f"single_gt content (first few):")
                    print(single_gt[:tf.minimum(3, tf.shape(single_gt)[0])])
                else:
                    print("single_gt: empty tensor")
                
                print(f"single_shapes: {single_shapes}")
                print(f"shapes shape: {shapes.shape}")
                print(f"single_shapes type: {type(single_shapes)}")
                print(f"single_shapes dtype: {single_shapes.dtype}")
                print(f"single_shapes shape: {single_shapes.shape}")
                if hasattr(single_shapes, 'shape') and len(single_shapes.shape) > 0:
                    print(f"single_shapes[0]: {single_shapes[0]}")
                    if single_shapes.shape[-1] >= 2 and single_shapes.shape[-2] >= 2:
                        print(f"single_shapes[0] original size: [{single_shapes[0, 0, 0]}, {single_shapes[0, 0, 1]}]")
                        print(f"single_shapes[0] ratio/padding: [{single_shapes[0, 1, 0]}, {single_shapes[0, 1, 1]}]")
                    else:
                        print(f"single_shapes[0] unexpected format: {single_shapes[0]}")
                print("=== END DEBUG ===")
                
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
                # gt_np: (cls, x, y, w, h)
                # convert to xyxy
                label_boxes = gt_np.copy()
                label_boxes[:, 1] = gt_np[:, 1] - gt_np[:, 3] / 2.0
                label_boxes[:, 2] = gt_np[:, 2] - gt_np[:, 4] / 2.0
                label_boxes[:, 3] = gt_np[:, 1] + gt_np[:, 3] / 2.0
                label_boxes[:, 4] = gt_np[:, 2] + gt_np[:, 4] / 2.0
                
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
                            
                            # Check class match
                            if det_class == gt_class:
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
    parser.add_argument('--epochs', default=2, type=int)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--yaml_file', type=str, default='utils/args_bionano.yaml')
    parser.add_argument('--save-path', type=str, default='./results/rect_256x128_v5')
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
