"""
Knowledge Distillation Loss Functions for YOLO

This module provides KL divergence-based loss functions for knowledge distillation
between a teacher (full-precision) and student (quantized) YOLO model.

The KD loss operates on raw model outputs before decoding:
- Box predictions: DFL distribution logits (B, H, W, 64) for 4 edges × 16 bins
- Class predictions: Classification logits (B, H, W, nc)
"""

import tensorflow as tf


def kl_divergence_loss(teacher_logits, student_logits, temperature=1.0):
    """
    Compute KL divergence loss between teacher and student logits.
    
    KL(P||Q) = sum(P * log(P/Q)) where P=teacher, Q=student
    
    Args:
        teacher_logits: Teacher model logits (any shape)
        student_logits: Student model logits (same shape as teacher)
        temperature: Temperature for softening distributions (default: 1.0)
        
    Returns:
        Scalar KL divergence loss
    """
    # Apply temperature scaling
    teacher_probs = tf.nn.softmax(teacher_logits / temperature, axis=-1)
    student_log_probs = tf.nn.log_softmax(student_logits / temperature, axis=-1)
    
    # KL divergence: sum(P * (log(P) - log(Q)))
    kl = tf.reduce_sum(
        teacher_probs * (tf.math.log(teacher_probs + 1e-10) - student_log_probs),
        axis=-1
    )
    
    # Scale by temperature^2 (standard practice in KD literature)
    kl = kl * (temperature ** 2)
    
    # Return mean over all spatial locations and batch
    return tf.reduce_mean(kl)


def box_kd_loss(teacher_box, student_box, temperature=4.0):
    """
    Compute KL divergence loss for DFL box predictions.
    
    The box predictions are DFL (Distribution Focal Loss) logits that represent
    the distribution over possible distances for each of the 4 box edges.
    
    Args:
        teacher_box: Teacher box logits (B, H, W, 64) where 64 = 4 edges × 16 bins
        student_box: Student box logits (B, H, W, 64)
        temperature: Temperature for softening distributions (default: 4.0)
        
    Returns:
        Scalar KL divergence loss for box predictions
    """
    # Get shape information
    batch_size = tf.shape(teacher_box)[0]
    height = tf.shape(teacher_box)[1]
    width = tf.shape(teacher_box)[2]
    
    # Reshape to (B*H*W, 4, 16) - separate the 4 edges and 16 bins
    teacher_dfl = tf.reshape(teacher_box, [-1, 4, 16])
    student_dfl = tf.reshape(student_box, [-1, 4, 16])
    
    # Apply temperature-scaled softmax to get probability distributions
    teacher_probs = tf.nn.softmax(teacher_dfl / temperature, axis=-1)
    student_log_probs = tf.nn.log_softmax(student_dfl / temperature, axis=-1)
    
    # Compute KL divergence for each edge
    # KL(P||Q) = sum(P * (log(P) - log(Q)))
    kl = tf.reduce_sum(
        teacher_probs * (tf.math.log(teacher_probs + 1e-10) - student_log_probs),
        axis=-1  # Sum over the 16 bins
    )
    
    # Scale by temperature^2
    kl = kl * (temperature ** 2)
    
    # Average over all edges and spatial locations
    return tf.reduce_mean(kl)


def class_kd_loss(teacher_cls, student_cls, temperature=4.0):
    """
    Compute KL divergence loss for classification predictions.
    
    For binary/multi-class classification, we convert sigmoid/softmax outputs
    to probability distributions and compute KL divergence.
    
    Args:
        teacher_cls: Teacher class logits (B, H, W, nc)
        student_cls: Student class logits (B, H, W, nc)
        temperature: Temperature for softening distributions (default: 4.0)
        
    Returns:
        Scalar KL divergence loss for class predictions
    """
    # Get number of classes
    num_classes = tf.shape(teacher_cls)[-1]
    
    if num_classes == 1:
        # Binary classification: use sigmoid and create [p, 1-p] distribution
        teacher_sigmoid = tf.sigmoid(teacher_cls / temperature)
        student_sigmoid = tf.sigmoid(student_cls / temperature)
        
        # Create binary distributions [p, 1-p]
        teacher_probs = tf.concat([teacher_sigmoid, 1 - teacher_sigmoid], axis=-1)
        student_probs = tf.concat([student_sigmoid, 1 - student_sigmoid], axis=-1)
        
        # Compute KL divergence
        kl = tf.reduce_sum(
            teacher_probs * tf.math.log((teacher_probs + 1e-10) / (student_probs + 1e-10)),
            axis=-1
        )
    else:
        # Multi-class: use softmax directly
        teacher_probs = tf.nn.softmax(teacher_cls / temperature, axis=-1)
        student_log_probs = tf.nn.log_softmax(student_cls / temperature, axis=-1)
        
        kl = tf.reduce_sum(
            teacher_probs * (tf.math.log(teacher_probs + 1e-10) - student_log_probs),
            axis=-1
        )
    
    # Scale by temperature^2
    kl = kl * (temperature ** 2)
    
    # Average over all spatial locations and batch
    return tf.reduce_mean(kl)


def compute_kd_loss(teacher_output, student_output, temperature=4.0, 
                    box_weight=1.0, cls_weight=1.0, dfl_ch=16):
    """
    Compute combined knowledge distillation loss.
    
    Splits the raw model outputs into box and class predictions, then computes
    separate KL divergence losses for each component.
    
    Args:
        teacher_output: Teacher model output (B, H, W, 4*dfl_ch + nc)
        student_output: Student model output (B, H, W, 4*dfl_ch + nc)
        temperature: Temperature for softening distributions (default: 4.0)
        box_weight: Weight for box KD loss (default: 1.0)
        cls_weight: Weight for class KD loss (default: 1.0)
        dfl_ch: Number of DFL channels (default: 16)
        
    Returns:
        Dictionary containing:
            - 'total': Combined KD loss
            - 'box': Box KD loss component
            - 'cls': Class KD loss component
    """
    # Split outputs into box and class predictions
    box_channels = 4 * dfl_ch  # 64 for default dfl_ch=16
    
    teacher_box = teacher_output[..., :box_channels]  # (B, H, W, 64)
    teacher_cls = teacher_output[..., box_channels:]  # (B, H, W, nc)
    
    student_box = student_output[..., :box_channels]  # (B, H, W, 64)
    student_cls = student_output[..., box_channels:]  # (B, H, W, nc)
    
    # Compute individual KD losses
    box_kd = box_kd_loss(teacher_box, student_box, temperature)
    cls_kd = class_kd_loss(teacher_cls, student_cls, temperature)
    
    # Weighted combination
    total_kd = box_weight * box_kd + cls_weight * cls_kd
    
    return {
        'total': total_kd,
        'box': box_kd,
        'cls': cls_kd
    }


def compute_kd_loss_simple(teacher_output, student_output, temperature=4.0):
    """
    Simplified interface that returns only the total KD loss.
    
    Args:
        teacher_output: Teacher model output (B, H, W, 4*dfl_ch + nc)
        student_output: Student model output (B, H, W, 4*dfl_ch + nc)
        temperature: Temperature for softening distributions (default: 4.0)
        
    Returns:
        Scalar total KD loss
    """
    kd_losses = compute_kd_loss(teacher_output, student_output, temperature)
    return kd_losses['total']
