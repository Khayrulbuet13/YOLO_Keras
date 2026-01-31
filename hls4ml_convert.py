"""
HLS4ml Conversion and Profiling Script

This script provides utilities for:
1. Converting trained quantized YOLO model to HLS4ml
2. Profiling model precision requirements
3. Generating synthesis-ready C++ code
4. Running C simulation for validation

Usage:
    # Profile model to determine optimal precision
    python hls4ml_convert.py --profile --weights results/quantized/best.weights.h5
    
    # Convert to HLS4ml
    python hls4ml_convert.py --convert --weights results/quantized/best.weights.h5 \
                             --output-dir hls4ml_project --fpga xczu9eg-ffvb1156-2-e
    
    # Run C simulation
    python hls4ml_convert.py --csim --hls-dir hls4ml_project
"""

import argparse
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import yaml

# Import models
from nets.tinysimov35_keras_quantized import yolo_v8_s_quantized
from nets.tinysimov35_keras_hls4ml import yolo_v8_s_hls4ml, transfer_weights_to_static


def profile_model(model, test_data, output_dir='hls4ml_profile'):
    """
    Profile model to determine optimal precision for each layer
    
    Uses hls4ml's profiling tools to analyze weight and activation distributions
    """
    try:
        import hls4ml
        from hls4ml.model.profiling import numerical
    except ImportError:
        print("[ERROR] hls4ml not installed. Install with: pip install hls4ml[profiling]")
        return
    
    print("[INFO] Profiling model with test data...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Run inference on test data to collect activations
    print("[INFO] Running inference to collect activation statistics...")
    predictions = model.predict(test_data, verbose=1)
    
    # Profile the model
    print("[INFO] Generating profiling report...")
    profile_data = numerical(model, test_data, output_dir=output_dir)
    
    print(f"[INFO] Profiling complete. Results saved to {output_dir}/")
    print("[INFO] Review the profiling plots to determine optimal bit-widths")
    print("[INFO] Look for:")
    print("  - Grey boxes should cover the data distribution")
    print("  - Reduce bit-width if grey boxes are much larger than data range")
    print("  - Increase bit-width if data extends beyond grey boxes")
    
    return profile_data


def create_hls4ml_config(model, granularity='name', reuse_factor=1, precision='ap_fixed<16,6>'):
    """
    Create HLS4ml configuration with optimized settings
    
    Args:
        model: Keras model to convert
        granularity: 'name' or 'type' - how to configure layers
        reuse_factor: Resource reuse factor (higher = less resources, more latency)
        precision: Default precision for weights/activations
    
    Returns:
        HLS4ml configuration dictionary
    """
    try:
        import hls4ml
    except ImportError:
        print("[ERROR] hls4ml not installed. Install with: pip install hls4ml")
        return None
    
    # Generate base config from model
    config = hls4ml.utils.config_from_keras_model(model, granularity=granularity)
    
    # Set global defaults
    config['Model']['ReuseFactor'] = reuse_factor
    config['Model']['Precision'] = precision
    config['Model']['Strategy'] = 'Latency'  # or 'Resource' for smaller designs
    
    # Layer-specific optimizations
    # These can be tuned based on profiling results
    layer_config = {
        # First layer: higher precision (more sensitive)
        'q_conv_0': {
            'Precision': 'ap_fixed<16,8>',
            'ReuseFactor': 1
        },
        # Middle layers: lower precision (more efficient)
        'q_conv_.*': {
            'Precision': 'ap_fixed<12,6>',
            'ReuseFactor': reuse_factor
        },
        # Head layers: medium precision
        'q_head.*': {
            'Precision': 'ap_fixed<14,7>',
            'ReuseFactor': 1
        },
        # Activations: can use lower precision
        'activation.*': {
            'Precision': 'ap_fixed<12,6>'
        },
        # Softmax: expensive, consider alternatives
        'softmax': {
            'Implementation': 'stable',  # More resource-efficient
            'Precision': 'ap_fixed<16,8>'
        },
        # Sigmoid: expensive, consider lookup table
        'sigmoid': {
            'Implementation': 'lut',  # Use lookup table
            'Precision': 'ap_fixed<16,8>'
        }
    }
    
    # Apply layer-specific configs
    for layer_pattern, layer_settings in layer_config.items():
        if layer_pattern in config['LayerName']:
            config['LayerName'][layer_pattern].update(layer_settings)
    
    return config


def convert_to_hls4ml(model, output_dir='hls4ml_prj', fpga_part='xczu9eg-ffvb1156-2-e',
                      clock_period=5, io_type='io_parallel', reuse_factor=1):
    """
    Convert Keras model to HLS4ml C++ code
    
    Args:
        model: Static Keras model (from tinysimov35_keras_hls4ml)
        output_dir: Directory for HLS4ml project
        fpga_part: Target FPGA part number
        clock_period: Target clock period in ns
        io_type: 'io_parallel' (fast, more resources) or 'io_stream' (slower, fewer resources)
        reuse_factor: Resource reuse factor
    """
    try:
        import hls4ml
    except ImportError:
        print("[ERROR] hls4ml not installed. Install with: pip install hls4ml")
        return None
    
    print("[INFO] Converting model to HLS4ml...")
    print(f"  Output directory: {output_dir}")
    print(f"  Target FPGA: {fpga_part}")
    print(f"  Clock period: {clock_period} ns")
    print(f"  IO type: {io_type}")
    print(f"  Reuse factor: {reuse_factor}")
    
    # Create configuration
    config = create_hls4ml_config(model, reuse_factor=reuse_factor)
    
    # Set HLS configuration
    hls_config = {
        'Model': {
            'Precision': 'ap_fixed<16,6>',
            'ReuseFactor': reuse_factor,
            'Strategy': 'Latency',
            'BramFactor': 100000  # Use BRAM for weights
        }
    }
    
    # Convert model
    print("[INFO] Running hls4ml conversion...")
    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=hls_config,
        output_dir=output_dir,
        fpga_part=fpga_part,
        clock_period=clock_period,
        io_type=io_type
    )
    
    print(f"[SUCCESS] HLS4ml project created at {output_dir}/")
    print("[INFO] Next steps:")
    print(f"  1. Review the generated code in {output_dir}/")
    print(f"  2. Run C simulation: hls_model.build(csim=True)")
    print(f"  3. Synthesize: hls_model.build(synth=True)")
    print(f"  4. Check resource usage in {output_dir}/vivado_hls.log")
    
    return hls_model


def run_csim(hls_model, test_data, keras_model):
    """
    Run C simulation and compare with Keras model
    
    Args:
        hls_model: HLS4ml model
        test_data: Test input data
        keras_model: Original Keras model for comparison
    """
    print("[INFO] Running C simulation...")
    
    # Get predictions from both models
    keras_pred = keras_model.predict(test_data, verbose=0)
    hls_pred = hls_model.predict(test_data)
    
    # Compare results
    mse = np.mean((keras_pred - hls_pred) ** 2)
    mae = np.mean(np.abs(keras_pred - hls_pred))
    max_error = np.max(np.abs(keras_pred - hls_pred))
    
    print("\n[INFO] C Simulation Results:")
    print(f"  Mean Squared Error: {mse:.6f}")
    print(f"  Mean Absolute Error: {mae:.6f}")
    print(f"  Max Absolute Error: {max_error:.6f}")
    
    # Check if accuracy is acceptable
    if mae < 0.01:
        print("[SUCCESS] C simulation matches Keras model well!")
    elif mae < 0.1:
        print("[WARNING] C simulation has moderate error. Consider increasing precision.")
    else:
        print("[ERROR] C simulation has large error. Increase precision or check model.")
    
    return {'mse': mse, 'mae': mae, 'max_error': max_error}


def main():
    parser = argparse.ArgumentParser(description='HLS4ml conversion and profiling for YOLO')
    
    # Mode selection
    parser.add_argument('--profile', action='store_true', help='Profile model precision')
    parser.add_argument('--convert', action='store_true', help='Convert to HLS4ml')
    parser.add_argument('--csim', action='store_true', help='Run C simulation')
    
    # Model parameters
    parser.add_argument('--weights', type=str, required=True, help='Path to trained weights')
    parser.add_argument('--num-classes', type=int, default=1, help='Number of classes')
    parser.add_argument('--img-size', type=str, default='256x128', help='Input size (HxW)')
    
    # HLS4ml parameters
    parser.add_argument('--output-dir', type=str, default='hls4ml_prj', help='HLS4ml output directory')
    parser.add_argument('--fpga', type=str, default='xczu9eg-ffvb1156-2-e', help='Target FPGA part')
    parser.add_argument('--clock-period', type=int, default=5, help='Clock period in ns')
    parser.add_argument('--reuse-factor', type=int, default=1, help='Resource reuse factor')
    parser.add_argument('--io-type', type=str, default='io_parallel', 
                       choices=['io_parallel', 'io_stream'], help='IO type')
    
    # Test data
    parser.add_argument('--test-samples', type=int, default=100, help='Number of test samples')
    
    args = parser.parse_args()
    
    # Parse image size
    if 'x' in args.img_size:
        img_h, img_w = map(int, args.img_size.split('x'))
    else:
        img_h = img_w = int(args.img_size)
    
    print(f"[INFO] Loading model: {args.num_classes} classes, {img_h}x{img_w} input")
    
    # Create static model for HLS4ml
    static_model = yolo_v8_s_hls4ml(
        num_classes=args.num_classes,
        img_h=img_h,
        img_w=img_w
    )
    
    # Load trained weights
    if os.path.exists(args.weights):
        print(f"[INFO] Loading weights from {args.weights}")
        # First load into quantized model, then transfer to static
        quantized_model = yolo_v8_s_quantized(
            num_classes=args.num_classes,
            img_size=(img_h, img_w)
        )
        quantized_model.load_weights(args.weights)
        static_model = transfer_weights_to_static(quantized_model, static_model)
    else:
        print(f"[WARNING] Weights file not found: {args.weights}")
        print("[WARNING] Using random initialization")
    
    # Generate test data
    print(f"[INFO] Generating {args.test_samples} test samples...")
    test_data = np.random.randn(args.test_samples, img_h, img_w, 3).astype(np.float32)
    test_data = test_data / 255.0  # Normalize
    
    # Execute requested operations
    if args.profile:
        profile_model(static_model, test_data, output_dir=args.output_dir + '_profile')
    
    if args.convert:
        hls_model = convert_to_hls4ml(
            static_model,
            output_dir=args.output_dir,
            fpga_part=args.fpga,
            clock_period=args.clock_period,
            io_type=args.io_type,
            reuse_factor=args.reuse_factor
        )
        
        if hls_model and args.csim:
            # Use small subset for C simulation (it's slow)
            test_subset = test_data[:10]
            run_csim(hls_model, test_subset, static_model)
    
    print("\n[INFO] Done!")


if __name__ == '__main__':
    main()
