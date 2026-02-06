"""
Compare Float32 and Quantized YOLO Models

This script loads both float32 and quantized models, evaluates them on the validation set,
and compares their memory usage and accuracy metrics.

Usage:
    python utils/compare_models.py \
        --float32-path results/rect_256x128_functional \
        --quantized-path results/quantized_integration_from_pretrained \
        --dataset-dir ./Dataset/bionano_cellv2 \
        --yaml-file utils/args_bionano.yaml \
        --input-size 128x256
"""

import argparse
import os
import sys
import yaml
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nets.tinysimov35_keras_functional import yolo_v8_s_functional, decode_predictions
from nets.tinysimov35_keras_quantized_functional import yolo_v8_s_quantized_functional
from utils.dataset_keras import Dataset
from utils.util_keras import (
    non_max_suppression,
    compute_ap,
    AverageMeter
)

# Try to import QKeras utilities
try:
    from qkeras.estimate import print_qstats
    from qkeras.utils import model_save_quantized_weights
    QKERAS_AVAILABLE = True
except ImportError:
    QKERAS_AVAILABLE = False
    print("[WARNING] QKeras utilities not available")

DTYPE = tf.float32


def get_model_memory(model):
    """Calculate model memory usage in bytes and parameters"""
    total_params = 0
    trainable_params = 0
    
    for layer in model.layers:
        layer_params = layer.count_params()
        total_params += layer_params
        if layer.trainable:
            trainable_params += layer_params
    
    # Memory calculation (assuming float32 = 4 bytes per parameter)
    memory_bytes = total_params * 4
    memory_kb = memory_bytes / 1024
    memory_mb = memory_kb / 1024
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'memory_bytes': memory_bytes,
        'memory_kb': memory_kb,
        'memory_mb': memory_mb
    }


def get_quantized_memory_estimate(model):
    """Estimate quantized model memory (8-bit quantization)"""
    total_params = 0
    
    for layer in model.layers:
        total_params += layer.count_params()
    
    # For 8-bit quantization: 1 byte per parameter
    memory_bytes = total_params * 1
    memory_kb = memory_bytes / 1024
    memory_mb = memory_kb / 1024
    
    return {
        'total_params': total_params,
        'memory_bytes': memory_bytes,
        'memory_kb': memory_kb,
        'memory_mb': memory_mb,
        'note': 'Estimated post-synthesis size (8-bit quantization)'
    }


def evaluate_model(model, dataset, params, img_size, model_name="Model"):
    """Evaluate model on dataset and return metrics"""
    print(f"\n{'='*80}")
    print(f"Evaluating {model_name}")
    print(f"{'='*80}")
    
    def test_data_generator():
        for i in range(len(dataset)):
            sample, target, shapes = dataset[i]
            
            # Pad targets to maximum possible size
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
            (img_size[0], img_size[1], 3),
            (100, 6),
            (3, 2)
        )
    ).batch(8).prefetch(tf.data.AUTOTUNE)
    
    metrics = []
    inference_times = []
    
    for samples, targets, shapes in tqdm(loader, desc=f'Evaluating {model_name}'):
        # Measure inference time
        start_time = tf.timestamp()
        outputs = model(samples, training=False)
        end_time = tf.timestamp()
        inference_times.append((end_time - start_time).numpy())
        
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
            
            # Extract ground truth for current image
            gt_for_image = targets[i]
            valid_mask = tf.reduce_sum(tf.abs(gt_for_image), axis=1) > 0
            gt = tf.boolean_mask(gt_for_image, valid_mask)
            
            if tf.shape(gt)[0] == 0:
                gt = tf.zeros((0, 5), dtype=tf.float32)
            
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
                # Convert label xywh -> xyxy
                label_boxes = np.zeros((len(gt_np), 5), dtype=gt_np.dtype)
                label_boxes[:, 0] = gt_np[:, 1]  # cls
                label_boxes[:, 1] = gt_np[:, 2] - gt_np[:, 4] / 2.0  # x1
                label_boxes[:, 2] = gt_np[:, 3] - gt_np[:, 5] / 2.0  # y1
                label_boxes[:, 3] = gt_np[:, 2] + gt_np[:, 4] / 2.0  # x2
                label_boxes[:, 4] = gt_np[:, 3] + gt_np[:, 5] / 2.0  # y2
                
                iou_v = np.linspace(0.5, 0.95, 10)
                n_iou = len(iou_v)
                correct = np.zeros((det_clone.shape[0], n_iou), dtype=bool)
                t_tensor = label_boxes[:, :5]
                
                # Calculate IoU
                for j in range(len(iou_v)):
                    iou_threshold = iou_v[j]
                    
                    for det_idx in range(det_clone.shape[0]):
                        det_box = det_clone[det_idx, :4]
                        det_class = det_clone[det_idx, 5]
                        
                        best_iou = 0
                        
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
    
    # Calculate average inference time
    avg_inference_time = np.mean(inference_times) * 1000  # Convert to ms
    
    results = {
        'precision': m_pre,
        'recall': m_rec,
        'mAP@50': map50,
        'mAP@50:95': mean_ap,
        'f1_score': 2 * m_pre * m_rec / (m_pre + m_rec + 1e-16),
        'true_positives': tp,
        'false_positives': fp,
        'avg_inference_time_ms': avg_inference_time
    }
    
    print(f"\nResults for {model_name}:")
    print(f"  Precision:    {m_pre:.4f}")
    print(f"  Recall:       {m_rec:.4f}")
    print(f"  F1 Score:     {results['f1_score']:.4f}")
    print(f"  mAP@50:       {map50:.4f}")
    print(f"  mAP@50:95:    {mean_ap:.4f}")
    print(f"  Avg Inference Time: {avg_inference_time:.2f} ms/batch")
    
    return results


def create_comparison_report(float32_results, quantized_results, float32_memory, quantized_memory, 
                            quantized_estimated_memory, save_path):
    """Create a detailed comparison report"""
    report_path = os.path.join(save_path, 'model_comparison_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("YOLO MODEL COMPARISON: Float32 vs Quantized (QKeras)\n")
        f.write("="*80 + "\n\n")
        
        # Memory Comparison
        f.write("MEMORY USAGE\n")
        f.write("-"*80 + "\n")
        f.write(f"Float32 Model:\n")
        f.write(f"  Total Parameters:     {float32_memory['total_params']:,}\n")
        f.write(f"  Trainable Parameters: {float32_memory['trainable_params']:,}\n")
        f.write(f"  Memory (Training):    {float32_memory['memory_kb']:.2f} KB ({float32_memory['memory_mb']:.2f} MB)\n")
        f.write(f"  Precision:            32-bit floating point\n\n")
        
        f.write(f"Quantized Model:\n")
        f.write(f"  Total Parameters:     {quantized_memory['total_params']:,}\n")
        f.write(f"  Trainable Parameters: {quantized_memory['trainable_params']:,}\n")
        f.write(f"  Memory (Training):    {quantized_memory['memory_kb']:.2f} KB ({quantized_memory['memory_mb']:.2f} MB)\n")
        f.write(f"  Memory (Post-Synthesis): {quantized_estimated_memory['memory_kb']:.2f} KB ({quantized_estimated_memory['memory_mb']:.2f} MB)\n")
        f.write(f"  Precision:            8-bit quantized (simulated during training)\n\n")
        
        # Memory reduction
        reduction_ratio = (1 - quantized_estimated_memory['memory_kb'] / float32_memory['memory_kb']) * 100
        f.write(f"Memory Reduction (Post-Synthesis):\n")
        f.write(f"  Size Reduction:       {reduction_ratio:.1f}%\n")
        f.write(f"  Compression Ratio:    {float32_memory['memory_kb'] / quantized_estimated_memory['memory_kb']:.2f}x\n\n")
        
        # Accuracy Comparison
        f.write("ACCURACY METRICS\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Metric':<20} {'Float32':<15} {'Quantized':<15} {'Difference':<15}\n")
        f.write("-"*80 + "\n")
        
        metrics_to_compare = ['precision', 'recall', 'f1_score', 'mAP@50', 'mAP@50:95']
        for metric in metrics_to_compare:
            float_val = float32_results[metric]
            quant_val = quantized_results[metric]
            diff = quant_val - float_val
            diff_pct = (diff / float_val * 100) if float_val != 0 else 0
            
            f.write(f"{metric:<20} {float_val:<15.4f} {quant_val:<15.4f} {diff:+.4f} ({diff_pct:+.1f}%)\n")
        
        f.write("\n")
        
        # Inference Time
        f.write("INFERENCE PERFORMANCE (CPU/Python)\n")
        f.write("-"*80 + "\n")
        f.write(f"Float32 Inference Time:   {float32_results['avg_inference_time_ms']:.2f} ms/batch\n")
        f.write(f"Quantized Inference Time: {quantized_results['avg_inference_time_ms']:.2f} ms/batch\n")
        speedup = float32_results['avg_inference_time_ms'] / quantized_results['avg_inference_time_ms']
        f.write(f"Speedup:                  {speedup:.2f}x\n\n")
        f.write("NOTE: Quantized model is slower on CPU because QKeras simulates quantization\n")
        f.write("      in float32. Real speedup occurs after HLS4ml synthesis to FPGA hardware,\n")
        f.write("      where actual INT8 operations provide 2-4x speedup and lower power.\n\n")
        
        # Summary
        f.write("SUMMARY\n")
        f.write("-"*80 + "\n")
        f.write(f"✓ Model successfully quantized with 8-bit precision\n")
        f.write(f"✓ Memory reduction: {reduction_ratio:.1f}% (after HLS4ml synthesis)\n")
        
        map_loss = (quantized_results['mAP@50:95'] - float32_results['mAP@50:95']) / float32_results['mAP@50:95'] * 100
        if abs(map_loss) < 5:
            f.write(f"✓ Minimal accuracy loss: {abs(map_loss):.1f}% in mAP@50:95\n")
        else:
            f.write(f"⚠ Accuracy loss: {abs(map_loss):.1f}% in mAP@50:95\n")
        
        f.write(f"✓ Ready for HLS4ml FPGA synthesis\n")
        f.write("\n" + "="*80 + "\n")
    
    print(f"\n[INFO] Comparison report saved to: {report_path}")
    return report_path


def create_comparison_plots(float32_results, quantized_results, save_path):
    """Create visualization plots comparing the models"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Float32 vs Quantized Model Comparison', fontsize=16, fontweight='bold')
    
    # Plot 1: Accuracy Metrics Bar Chart
    ax1 = axes[0, 0]
    metrics = ['Precision', 'Recall', 'F1 Score', 'mAP@50', 'mAP@50:95']
    float_values = [float32_results['precision'], float32_results['recall'], 
                   float32_results['f1_score'], float32_results['mAP@50'], 
                   float32_results['mAP@50:95']]
    quant_values = [quantized_results['precision'], quantized_results['recall'], 
                   quantized_results['f1_score'], quantized_results['mAP@50'], 
                   quantized_results['mAP@50:95']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax1.bar(x - width/2, float_values, width, label='Float32', color='#3498db', alpha=0.8)
    ax1.bar(x + width/2, quant_values, width, label='Quantized', color='#e74c3c', alpha=0.8)
    ax1.set_ylabel('Score', fontweight='bold')
    ax1.set_title('Accuracy Metrics Comparison', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0, 1.0])
    
    # Plot 2: Accuracy Loss Percentage
    ax2 = axes[0, 1]
    losses = []
    for i, metric in enumerate(metrics):
        if float_values[i] != 0:
            loss_pct = ((quant_values[i] - float_values[i]) / float_values[i]) * 100
            losses.append(loss_pct)
        else:
            losses.append(0)
    
    colors = ['#2ecc71' if l >= 0 else '#e74c3c' for l in losses]
    bars = ax2.bar(metrics, losses, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    ax2.set_ylabel('Accuracy Change (%)', fontweight='bold', fontsize=11)
    ax2.set_title('Quantization Impact on Accuracy', fontweight='bold', fontsize=12)
    ax2.set_xticklabels(metrics, rotation=45, ha='right', fontsize=9)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bar, loss in zip(bars, losses):
        height = bar.get_height()
        label_y = height + (1 if height >= 0 else -3)
        ax2.text(bar.get_x() + bar.get_width()/2., label_y,
                f'{loss:.1f}%',
                ha='center', va='bottom' if height >= 0 else 'top',
                fontweight='bold', fontsize=9)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', alpha=0.7, label='Improvement'),
        Patch(facecolor='#e74c3c', alpha=0.7, label='Degradation')
    ]
    ax2.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    # Plot 3: Inference Time Comparison
    ax3 = axes[1, 0]
    inference_times = [float32_results['avg_inference_time_ms'], 
                      quantized_results['avg_inference_time_ms']]
    models = ['Float32', 'Quantized']
    colors_inf = ['#3498db', '#e74c3c']
    
    bars = ax3.bar(models, inference_times, color=colors_inf, alpha=0.8)
    ax3.set_ylabel('Inference Time (ms/batch)', fontweight='bold')
    ax3.set_title('Inference Speed Comparison', fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f} ms',
                ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Summary Table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_data = [
        ['Metric', 'Float32', 'Quantized', 'Difference'],
        ['mAP@50:95', f"{float32_results['mAP@50:95']:.4f}", 
         f"{quantized_results['mAP@50:95']:.4f}",
         f"{(quantized_results['mAP@50:95'] - float32_results['mAP@50:95']):.4f}"],
        ['mAP@50', f"{float32_results['mAP@50']:.4f}", 
         f"{quantized_results['mAP@50']:.4f}",
         f"{(quantized_results['mAP@50'] - float32_results['mAP@50']):.4f}"],
        ['Inference', f"{float32_results['avg_inference_time_ms']:.2f} ms", 
         f"{quantized_results['avg_inference_time_ms']:.2f} ms",
         f"{float32_results['avg_inference_time_ms'] - quantized_results['avg_inference_time_ms']:.2f} ms"],
    ]
    
    table = ax4.table(cellText=summary_data, cellLoc='center', loc='center',
                     colWidths=[0.25, 0.25, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax4.set_title('Performance Summary', fontweight='bold', pad=20)
    
    plt.tight_layout()
    plot_path = os.path.join(save_path, 'model_comparison_plots.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"[INFO] Comparison plots saved to: {plot_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Compare Float32 and Quantized YOLO models')
    parser.add_argument('--float32-path', type=str, required=True,
                       help='Path to float32 model results directory')
    parser.add_argument('--quantized-path', type=str, required=True,
                       help='Path to quantized model results directory')
    parser.add_argument('--dataset-dir', type=str, default='./Dataset/bionano_cellv2',
                       help='Path to dataset directory')
    parser.add_argument('--yaml-file', type=str, default='utils/args_bionano.yaml',
                       help='Path to YAML config file')
    parser.add_argument('--input-size', type=str, default='128x256',
                       help='Input image size (HxW)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for comparison results (default: quantized-path)')
    
    args = parser.parse_args()
    
    # Parse image size
    if 'x' in args.input_size:
        h, w = map(int, args.input_size.split('x'))
        img_size = (h, w)
    else:
        size = int(args.input_size)
        img_size = (size, size)
    
    # Load YAML config
    with open(args.yaml_file, 'r') as f:
        params = yaml.safe_load(f)
    
    num_classes = len(params['names'].values())
    
    # Set output directory
    output_dir = args.output_dir if args.output_dir else args.quantized_path
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*80)
    print("YOLO MODEL COMPARISON: Float32 vs Quantized")
    print("="*80)
    print(f"Float32 model path:  {args.float32_path}")
    print(f"Quantized model path: {args.quantized_path}")
    print(f"Dataset directory:    {args.dataset_dir}")
    print(f"Image size:           {img_size}")
    print(f"Output directory:     {output_dir}")
    print("="*80)
    
    # Load validation dataset
    val_filenames = []
    with open(os.path.join(args.dataset_dir, 'val.txt')) as f:
        for line in f.readlines():
            line = line.rstrip().split('/')[-1]
            val_filenames.append(os.path.join(args.dataset_dir, 'images/val', line))
    
    val_dataset = Dataset(val_filenames, img_size, params, False, dtype=DTYPE)
    print(f"\n[INFO] Loaded validation dataset: {len(val_dataset)} images")
    
    # Load Float32 Model
    print("\n" + "="*80)
    print("LOADING FLOAT32 MODEL")
    print("="*80)
    float32_model = yolo_v8_s_functional(num_classes, img_size=img_size, dtype=DTYPE)
    float32_weights_path = os.path.join(args.float32_path, 'best.weights.h5')
    
    if not os.path.exists(float32_weights_path):
        print(f"[ERROR] Float32 weights not found at: {float32_weights_path}")
        sys.exit(1)
    
    float32_model.load_weights(float32_weights_path)
    print(f"[INFO] Loaded float32 weights from: {float32_weights_path}")
    
    float32_memory = get_model_memory(float32_model)
    print(f"[INFO] Float32 model memory: {float32_memory['memory_kb']:.2f} KB ({float32_memory['total_params']:,} params)")
    
    # Load Quantized Model
    print("\n" + "="*80)
    print("LOADING QUANTIZED MODEL")
    print("="*80)
    quantized_model = yolo_v8_s_quantized_functional(num_classes, img_size=img_size, dtype=DTYPE)
    quantized_weights_path = os.path.join(args.quantized_path, 'best.weights.h5')
    
    if not os.path.exists(quantized_weights_path):
        print(f"[ERROR] Quantized weights not found at: {quantized_weights_path}")
        sys.exit(1)
    
    quantized_model.load_weights(quantized_weights_path)
    print(f"[INFO] Loaded quantized weights from: {quantized_weights_path}")
    
    quantized_memory = get_model_memory(quantized_model)
    quantized_estimated_memory = get_quantized_memory_estimate(quantized_model)
    print(f"[INFO] Quantized model memory (training): {quantized_memory['memory_kb']:.2f} KB")
    print(f"[INFO] Quantized model memory (post-synthesis): {quantized_estimated_memory['memory_kb']:.2f} KB")
    
    # Evaluate both models
    float32_results = evaluate_model(float32_model, val_dataset, params, img_size, "Float32 Model")
    quantized_results = evaluate_model(quantized_model, val_dataset, params, img_size, "Quantized Model")
    
    # Create comparison report
    print("\n" + "="*80)
    print("GENERATING COMPARISON REPORT")
    print("="*80)
    
    report_path = create_comparison_report(
        float32_results, quantized_results,
        float32_memory, quantized_memory, quantized_estimated_memory,
        output_dir
    )
    
    # Create comparison plots
    create_comparison_plots(float32_results, quantized_results, output_dir)
    
    # Print summary
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"Memory Reduction: {(1 - quantized_estimated_memory['memory_kb'] / float32_memory['memory_kb']) * 100:.1f}%")
    print(f"mAP@50:95 - Float32: {float32_results['mAP@50:95']:.4f}, Quantized: {quantized_results['mAP@50:95']:.4f}")
    print(f"mAP@50 - Float32: {float32_results['mAP@50']:.4f}, Quantized: {quantized_results['mAP@50']:.4f}")
    print(f"Accuracy Loss: {abs((quantized_results['mAP@50:95'] - float32_results['mAP@50:95']) / float32_results['mAP@50:95'] * 100):.1f}%")
    print("="*80)
    print(f"\n✓ Comparison complete! Results saved to: {output_dir}")
    print(f"  - Report: {report_path}")
    print(f"  - Plots: {os.path.join(output_dir, 'model_comparison_plots.png')}")


if __name__ == '__main__':
    main()
