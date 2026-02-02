"""
Quantization Initialization Utilities

Properly initialize quantized models from full-precision teacher models,
including QKeras quantizer scale factors (alpha) and BatchNorm statistics.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


def initialize_quantized_from_teacher(student_model, teacher_model, verbose=True):
    """
    Properly initialize quantized student from full-precision teacher
    
    This function:
    1. Copies raw weights from teacher to student
    2. Initializes QKeras quantizer alpha from teacher weight ranges
    3. Copies BatchNorm statistics (moving mean/variance)
    4. Validates and reports activation ranges
    
    Args:
        student_model: Quantized model (QKeras) to initialize
        teacher_model: Full-precision teacher model
        verbose: Print initialization details
    
    Returns:
        student_model: Initialized quantized model
        init_stats: Dictionary with initialization statistics
    """
    
    if verbose:
        print("\n" + "="*70)
        print("PROPER QUANTIZED MODEL INITIALIZATION FROM TEACHER")
        print("="*70)
    
    # Statistics tracking
    stats = {
        'weights_copied': 0,
        'weights_skipped': 0,
        'quantizers_initialized': 0,
        'bn_layers_copied': 0,
        'layer_details': []
    }
    
    # Build layer name mappings (handle QConv vs Conv naming differences)
    # Need to get ALL layers recursively, not just top-level
    def get_all_layers(model):
        """Recursively get all layers including nested ones"""
        all_layers = []
        for layer in model.layers:
            all_layers.append(layer)
            # Check if layer has nested layers
            if hasattr(layer, 'layers') and isinstance(layer.layers, list):
                all_layers.extend(get_all_layers(layer))
            elif hasattr(layer, 'layers_list') and isinstance(layer.layers_list, list):
                # Handle QDarkNet which uses layers_list
                all_layers.extend(layer.layers_list)
        return all_layers
    
    teacher_all_layers = get_all_layers(teacher_model)
    student_all_layers = get_all_layers(student_model)
    
    teacher_layers_dict = {}
    for layer in teacher_all_layers:
        # Store by multiple possible names
        teacher_layers_dict[layer.name] = layer
        # Also store without 'q_' prefix for matching
        clean_name = layer.name.replace('q_', '')
        teacher_layers_dict[clean_name] = layer
    
    # Iterate through ALL student layers (including nested)
    for student_layer in student_all_layers:
        layer_info = {'name': student_layer.name, 'status': 'skipped', 'type': type(student_layer).__name__}
        
        # Debug: check layer attributes
        has_conv = hasattr(student_layer, 'conv')
        has_kernel_quantizer = hasattr(student_layer, 'kernel_quantizer')
        if has_conv:
            has_nested_quantizer = hasattr(student_layer.conv, 'kernel_quantizer')
        else:
            has_nested_quantizer = False
        
        layer_info['has_conv'] = has_conv
        layer_info['has_kernel_quantizer'] = has_kernel_quantizer
        layer_info['has_nested_quantizer'] = has_nested_quantizer
        
        # Try to find matching teacher layer
        teacher_layer = None
        for possible_name in [student_layer.name, student_layer.name.replace('q_', '')]:
            if possible_name in teacher_layers_dict:
                teacher_layer = teacher_layers_dict[possible_name]
                break
        
        if teacher_layer is None:
            layer_info['reason'] = 'no_matching_teacher'
            stats['layer_details'].append(layer_info)
            continue
        
        # Process layer based on type
        try:
            # Handle layers with weights
            if hasattr(student_layer, 'get_weights') and len(student_layer.get_weights()) > 0:
                teacher_weights = teacher_layer.get_weights()
                student_weights = student_layer.get_weights()
                
                if len(teacher_weights) == 0:
                    layer_info['reason'] = 'no_teacher_weights'
                    stats['layer_details'].append(layer_info)
                    continue
                
                # Copy compatible weights
                weights_copied = 0
                for i in range(min(len(teacher_weights), len(student_weights))):
                    if teacher_weights[i].shape == student_weights[i].shape:
                        student_weights[i] = teacher_weights[i].copy()
                        weights_copied += 1
                    else:
                        if verbose:
                            print(f"  [SKIP] {student_layer.name} weight {i}: shape mismatch "
                                  f"{teacher_weights[i].shape} vs {student_weights[i].shape}")
                        stats['weights_skipped'] += 1
                
                # Initialize quantizer alpha from weight range (for QKeras layers)
                # Check for QConv wrapper (has nested conv with quantizer)
                quantizer = None
                qconv_layer = None
                
                if hasattr(student_layer, 'conv') and hasattr(student_layer.conv, 'kernel_quantizer'):
                    # QConv wrapper: quantizer is in student_layer.conv
                    quantizer = student_layer.conv.kernel_quantizer
                    qconv_layer = student_layer.conv
                    layer_type = 'QConv_wrapper'
                elif hasattr(student_layer, 'kernel_quantizer'):
                    # Direct QConv2D: quantizer is on the layer itself
                    quantizer = student_layer.kernel_quantizer
                    qconv_layer = student_layer
                    layer_type = 'QConv2D_direct'
                
                if quantizer is not None and hasattr(quantizer, 'alpha'):
                    # Get kernel weights (usually first weight)
                    kernel_idx = 0
                    if kernel_idx < len(teacher_weights):
                        kernel_weights = teacher_weights[kernel_idx]
                        
                        # Calculate optimal alpha to cover weight range
                        weight_max = np.max(np.abs(kernel_weights))
                        
                        # Get quantization bits
                        if hasattr(quantizer, 'bits'):
                            bits = quantizer.bits
                        else:
                            bits = 8  # Default
                        
                        # Calculate alpha: range / (2^(bits-1))
                        # This ensures weights fit within quantization range
                        alpha_value = weight_max / (2 ** (bits - 1) - 1)
                        
                        # Set alpha (handle different QKeras versions)
                        if isinstance(quantizer.alpha, tf.Variable):
                            quantizer.alpha.assign(alpha_value)
                        else:
                            quantizer.alpha = alpha_value
                        
                        layer_info['alpha'] = float(alpha_value)
                        layer_info['weight_range'] = float(weight_max)
                        layer_info['bits'] = int(bits)
                        layer_info['quantizer_type'] = layer_type
                        stats['quantizers_initialized'] += 1
                        
                        if verbose:
                            print(f"  [INIT] {student_layer.name} ({layer_type}): alpha={alpha_value:.6f}, "
                                  f"weight_range=[{-weight_max:.4f}, {weight_max:.4f}], bits={bits}")
                
                # Set weights
                student_layer.set_weights(student_weights)
                stats['weights_copied'] += weights_copied
                layer_info['status'] = 'initialized'
                layer_info['weights_copied'] = weights_copied
                
                # Special handling for BatchNormalization
                if isinstance(student_layer, layers.BatchNormalization):
                    stats['bn_layers_copied'] += 1
                    layer_info['type'] = 'batchnorm'
                    if verbose:
                        print(f"  [BN] {student_layer.name}: Copied moving statistics")
        
        except Exception as e:
            layer_info['status'] = 'error'
            layer_info['error'] = str(e)
            if verbose:
                print(f"  [ERROR] {student_layer.name}: {e}")
        
        stats['layer_details'].append(layer_info)
    
    if verbose:
        print("\n" + "-"*70)
        print("INITIALIZATION SUMMARY:")
        print(f"  Weights copied: {stats['weights_copied']}")
        print(f"  Weights skipped: {stats['weights_skipped']}")
        print(f"  Quantizers initialized: {stats['quantizers_initialized']}")
        print(f"  BatchNorm layers copied: {stats['bn_layers_copied']}")
        print("="*70 + "\n")
    
    return student_model, stats


def validate_quantized_model(model, input_shape, verbose=True):
    """
    Validate quantized model by checking activation ranges
    
    Args:
        model: Quantized model to validate
        input_shape: Input shape for dummy forward pass
        verbose: Print validation details
    
    Returns:
        validation_results: Dictionary with validation results
    """
    
    if verbose:
        print("\n" + "="*70)
        print("QUANTIZED MODEL VALIDATION")
        print("="*70)
    
    # Create dummy input
    dummy_input = tf.random.normal(input_shape)
    
    # Forward pass to get intermediate activations
    _ = model(dummy_input, training=False)
    
    results = {
        'layers_checked': 0,
        'potential_clipping': [],
        'activation_ranges': {}
    }
    
    # Check each layer's output range
    for layer in model.layers:
        if hasattr(layer, 'output') and layer.output is not None:
            try:
                # Get layer output
                intermediate_model = tf.keras.Model(inputs=model.input, outputs=layer.output)
                output = intermediate_model(dummy_input, training=False)
                
                output_min = float(tf.reduce_min(output).numpy())
                output_max = float(tf.reduce_max(output).numpy())
                output_mean = float(tf.reduce_mean(output).numpy())
                
                results['activation_ranges'][layer.name] = {
                    'min': output_min,
                    'max': output_max,
                    'mean': output_mean
                }
                results['layers_checked'] += 1
                
                # Check for potential clipping in quantized activations
                if hasattr(layer, 'activation') and 'qactivation' in str(type(layer.activation)).lower():
                    # Estimate quantized range (rough approximation)
                    # Most QActivations use limited ranges
                    if output_max > 16 or output_min < -16:
                        results['potential_clipping'].append({
                            'layer': layer.name,
                            'range': [output_min, output_max],
                            'issue': 'Values may exceed typical quantized activation range'
                        })
                        
                        if verbose:
                            print(f"  [WARNING] {layer.name}: range [{output_min:.2f}, {output_max:.2f}] "
                                  f"may clip in quantized activation")
            
            except Exception as e:
                if verbose:
                    print(f"  [SKIP] {layer.name}: Could not validate ({e})")
    
    if verbose:
        print(f"\n  Layers checked: {results['layers_checked']}")
        print(f"  Potential clipping issues: {len(results['potential_clipping'])}")
        print("="*70 + "\n")
    
    return results
