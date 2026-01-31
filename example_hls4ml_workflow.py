"""
Example workflow for HLS4ml FPGA synthesis

This script demonstrates the complete workflow from training to FPGA synthesis.
"""

import os
import numpy as np
import tensorflow as tf
from nets.tinysimov35_keras import yolo_v8_s
from nets.tinysimov35_keras_quantized import yolo_v8_s_quantized
from nets.tinysimov35_keras_hls4ml import yolo_v8_s_hls4ml, transfer_weights_to_static


def example_1_compare_models():
    """
    Example 1: Compare float32, quantized, and static models
    """
    print("\n" + "="*70)
    print("Example 1: Comparing Model Variants")
    print("="*70)
    
    num_classes = 1
    img_h, img_w = 256, 128
    
    # Create models
    print("\n[1/3] Creating float32 model...")
    model_float32 = yolo_v8_s(num_classes, img_size=(img_h, img_w))
    
    print("[2/3] Creating quantized model...")
    model_quantized = yolo_v8_s_quantized(num_classes, img_size=(img_h, img_w))
    
    print("[3/3] Creating static HLS4ml model...")
    model_static = yolo_v8_s_hls4ml(num_classes, img_h=img_h, img_w=img_w)
    
    # Generate test input
    test_input = np.random.randn(1, img_h, img_w, 3).astype(np.float32) / 255.0
    
    # Run inference
    print("\n[INFO] Running inference on all models...")
    out_float32 = model_float32(test_input, training=False)
    out_quantized = model_quantized(test_input, training=False)
    out_static = model_static(test_input)
    
    # Compare outputs
    print("\n[INFO] Output shapes:")
    print(f"  Float32:   {out_float32.shape}")
    print(f"  Quantized: {out_quantized.shape}")
    print(f"  Static:    {out_static.shape}")
    
    # Note: Random weights will give different outputs
    # After training, they should be similar
    print("\n[INFO] Models created successfully!")
    print("[INFO] Train the quantized model, then transfer weights to static model for HLS4ml")


def example_2_weight_transfer():
    """
    Example 2: Transfer weights from quantized to static model
    """
    print("\n" + "="*70)
    print("Example 2: Weight Transfer Workflow")
    print("="*70)
    
    num_classes = 1
    img_h, img_w = 256, 128
    
    # Simulate trained quantized model
    print("\n[1/3] Creating and 'training' quantized model...")
    model_quantized = yolo_v8_s_quantized(num_classes, img_size=(img_h, img_w))
    
    # In practice, you would load trained weights:
    # model_quantized.load_weights('results/quantized_model/best.weights.h5')
    print("[INFO] (In practice, load trained weights here)")
    
    # Create static model
    print("\n[2/3] Creating static HLS4ml model...")
    model_static = yolo_v8_s_hls4ml(num_classes, img_h=img_h, img_w=img_w)
    
    # Transfer weights
    print("\n[3/3] Transferring weights...")
    model_static = transfer_weights_to_static(model_quantized, model_static)
    
    # Verify transfer
    test_input = np.random.randn(1, img_h, img_w, 3).astype(np.float32) / 255.0
    out_quantized = model_quantized(test_input, training=False)
    out_static = model_static(test_input)
    
    diff = np.abs(out_quantized.numpy() - out_static.numpy())
    print(f"\n[INFO] Output difference after weight transfer:")
    print(f"  Max difference: {np.max(diff):.6f}")
    print(f"  Mean difference: {np.mean(diff):.6f}")
    
    if np.max(diff) < 1e-5:
        print("[SUCCESS] Weight transfer successful! Models produce identical outputs.")
    else:
        print("[WARNING] Small differences expected due to numerical precision.")


def example_3_model_summary():
    """
    Example 3: Display model architecture and parameter counts
    """
    print("\n" + "="*70)
    print("Example 3: Model Architecture Analysis")
    print("="*70)
    
    num_classes = 1
    img_h, img_w = 256, 128
    
    # Create models
    model_float32 = yolo_v8_s(num_classes, img_size=(img_h, img_w))
    model_quantized = yolo_v8_s_quantized(num_classes, img_size=(img_h, img_w))
    model_static = yolo_v8_s_hls4ml(num_classes, img_h=img_h, img_w=img_w)
    
    print("\n[INFO] Float32 Model:")
    print(f"  Total parameters: {model_float32.count_params():,}")
    
    print("\n[INFO] Quantized Model:")
    print(f"  Total parameters: {model_quantized.count_params():,}")
    
    print("\n[INFO] Static HLS4ml Model:")
    print(f"  Total parameters: {model_static.count_params():,}")
    
    # Estimate FPGA resources (rough approximation)
    params = model_static.count_params()
    
    # Rough estimates for 8-bit quantization
    estimated_brams = params * 8 / (18 * 1024)  # 18Kb BRAMs
    estimated_luts = params * 10  # Very rough estimate
    estimated_dsps = params / 100  # Very rough estimate
    
    print("\n[INFO] Estimated FPGA Resources (8-bit quantization, ReuseFactor=1):")
    print(f"  BRAMs: ~{estimated_brams:.1f}")
    print(f"  LUTs: ~{estimated_luts:,.0f}")
    print(f"  DSPs: ~{estimated_dsps:.0f}")
    print("\n[NOTE] These are rough estimates. Run HLS synthesis for accurate numbers.")


def example_4_save_for_hls4ml():
    """
    Example 4: Save model in format ready for HLS4ml conversion
    """
    print("\n" + "="*70)
    print("Example 4: Preparing Model for HLS4ml Conversion")
    print("="*70)
    
    num_classes = 1
    img_h, img_w = 256, 128
    output_dir = "hls4ml_ready_model"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create and prepare static model
    print("\n[1/4] Creating static model...")
    model_static = yolo_v8_s_hls4ml(num_classes, img_h=img_h, img_w=img_w)
    
    # In practice, load trained weights
    print("[2/4] Loading trained weights...")
    print("[INFO] (Skipping - use model_static.load_weights('path/to/weights.h5'))")
    
    # Save model
    print(f"\n[3/4] Saving model to {output_dir}/...")
    model_path = os.path.join(output_dir, "yolo_static_model.h5")
    model_static.save(model_path)
    print(f"[SUCCESS] Model saved to {model_path}")
    
    # Save model config
    print("\n[4/4] Saving model configuration...")
    config = {
        'num_classes': num_classes,
        'img_height': img_h,
        'img_width': img_w,
        'input_shape': (1, img_h, img_w, 3),
        'output_shape': model_static.output_shape,
        'quantization': '8-16 bit mixed precision',
        'notes': 'Static model ready for HLS4ml conversion'
    }
    
    import json
    config_path = os.path.join(output_dir, "model_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"[SUCCESS] Configuration saved to {config_path}")
    
    print("\n[INFO] Next steps:")
    print(f"  1. Use hls4ml_convert.py to convert this model")
    print(f"  2. Run: python hls4ml_convert.py --convert --weights {model_path}")


def example_5_quantization_comparison():
    """
    Example 5: Compare different quantization bit-widths
    """
    print("\n" + "="*70)
    print("Example 5: Quantization Bit-width Analysis")
    print("="*70)
    
    print("\n[INFO] Quantization strategy used in this implementation:")
    print("\n  Layer Type              | Weight Bits | Activation Bits | Rationale")
    print("  " + "-"*75)
    print("  First Conv Layer        |     16      |       16        | Most sensitive")
    print("  Middle Conv Layers      |      8      |        8        | Most efficient")
    print("  Last Conv + Head        |     12      |       12        | Balance")
    print("  DFL Conv                |      8      |        8        | Efficient")
    
    print("\n[INFO] Trade-offs:")
    print("  - Lower bits → Less FPGA resources, faster synthesis")
    print("  - Higher bits → Better accuracy, more resources")
    print("  - Mixed precision → Best balance")
    
    print("\n[INFO] To adjust quantization:")
    print("  1. Edit nets/tinysimov35_keras_quantized.py")
    print("  2. Modify weight_bits and activation_bits in QConv layers")
    print("  3. Re-train the model with new quantization")
    print("  4. Profile with hls4ml_convert.py --profile")


def main():
    """Run all examples"""
    print("\n" + "="*70)
    print("HLS4ml FPGA Synthesis - Example Workflows")
    print("="*70)
    print("\nThis script demonstrates the complete workflow for preparing")
    print("YOLO models for FPGA synthesis using HLS4ml.")
    
    try:
        example_1_compare_models()
        example_2_weight_transfer()
        example_3_model_summary()
        example_4_save_for_hls4ml()
        example_5_quantization_comparison()
        
        print("\n" + "="*70)
        print("All Examples Completed Successfully!")
        print("="*70)
        print("\n[INFO] Next steps:")
        print("  1. Train quantized model: python main_keras.py --train --quantized")
        print("  2. Profile model: python hls4ml_convert.py --profile --weights <path>")
        print("  3. Convert to HLS4ml: python hls4ml_convert.py --convert --weights <path>")
        print("  4. Synthesize with Vivado HLS")
        print("\n[INFO] See HLS4ML_GUIDE.md for detailed instructions.")
        
    except Exception as e:
        print(f"\n[ERROR] Example failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
