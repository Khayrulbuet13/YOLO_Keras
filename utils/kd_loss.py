"""
Knowledge Distillation Loss Module for YOLO

Implements KL divergence loss between teacher (full-precision) and student (quantized) models.
Distills knowledge from both box predictions (DFL distributions) and class scores.
"""

import tensorflow as tf
from tensorflow.keras.layers import Layer


class KDLoss(Layer):
    """
    Knowledge Distillation Loss using KL Divergence
    
    Computes KL divergence between teacher and student outputs for:
    - Box predictions: DFL distribution (4 coordinates × 16 bins each)
    - Class scores: Sigmoid probability distributions
    
    Args:
        temperature: Temperature for softening distributions (default: 3.0)
        alpha: Weight for box KD loss (default: 0.5)
        beta: Weight for class KD loss (default: 0.5)
        dtype: Data type for computations (default: tf.float32)
    """
    
    def __init__(self, temperature=3.0, alpha=0.5, beta=0.5, dtype=tf.float32, **kwargs):
        super(KDLoss, self).__init__(dtype=dtype, **kwargs)
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.dtype_ = dtype
        self.eps = 1e-7
        
    def call(self, student_outputs, teacher_outputs):
        """
        Compute KD loss between student and teacher outputs
        
        Args:
            student_outputs: Raw outputs from student model (B, H, W, 65)
                           Format: [box_preds (64 channels), class_scores (1 channel)]
            teacher_outputs: Raw outputs from teacher model (B, H, W, 65)
        
        Returns:
            Total KD loss (scalar tensor)
        """
        # Ensure consistent typing
        student_outputs = tf.cast(student_outputs, self.dtype_)
        teacher_outputs = tf.cast(teacher_outputs, self.dtype_)
        
        # Split into box predictions and class scores
        # Box: 64 channels (4 coords × 16 bins), Class: nc channels (1 for single class)
        nc = 1  # number of classes (can be extracted from shape if needed)
        student_box, student_cls = tf.split(student_outputs, [64, nc], axis=-1)
        teacher_box, teacher_cls = tf.split(teacher_outputs, [64, nc], axis=-1)
        
        # Compute KL divergence for box predictions
        loss_box = self.kl_divergence_boxes(student_box, teacher_box)
        
        # Compute KL divergence for class scores
        loss_cls = self.kl_divergence_classes(student_cls, teacher_cls)
        
        # Weighted combination
        total_loss = self.alpha * loss_box + self.beta * loss_cls
        
        return total_loss
    
    def kl_divergence_boxes(self, student_box, teacher_box):
        """
        Compute KL divergence for box coordinate distributions
        
        Box predictions are DFL distributions: 4 coordinates, each with 16-bin softmax
        
        Args:
            student_box: Student box predictions (B, H, W, 64)
            teacher_box: Teacher box predictions (B, H, W, 64)
            
        Returns:
            KL divergence loss (scalar)
        """
        # Reshape from (B, H, W, 64) to (B, H, W, 4, 16)
        # This separates the 4 coordinates, each with 16 bins
        shape = tf.shape(student_box)
        B, H, W = shape[0], shape[1], shape[2]
        
        student_box_reshaped = tf.reshape(student_box, [B, H, W, 4, 16])
        teacher_box_reshaped = tf.reshape(teacher_box, [B, H, W, 4, 16])
        
        # Apply temperature scaling and softmax to get soft distributions
        # Temperature scaling: logits / T before softmax
        student_dist = tf.nn.softmax(student_box_reshaped / self.temperature, axis=-1)
        teacher_dist = tf.nn.softmax(teacher_box_reshaped / self.temperature, axis=-1)
        
        # Compute KL divergence: KL(teacher || student) = sum(teacher * log(teacher / student))
        # Using log-space for numerical stability
        kl_per_bin = teacher_dist * (
            tf.math.log(teacher_dist + self.eps) - tf.math.log(student_dist + self.eps)
        )
        
        # Sum over bins (16), then average over coordinates (4), spatial dimensions (H, W), and batch (B)
        kl_loss = tf.reduce_sum(kl_per_bin, axis=-1)  # Sum over 16 bins
        kl_loss = tf.reduce_mean(kl_loss)  # Average over all other dimensions
        
        # Scale by temperature^2 (standard in KD literature)
        kl_loss = kl_loss * (self.temperature ** 2)
        
        return kl_loss
    
    def kl_divergence_classes(self, student_cls, teacher_cls):
        """
        Compute KL divergence for class score distributions
        
        Class scores are logits that go through sigmoid for binary classification
        
        Args:
            student_cls: Student class scores (B, H, W, nc)
            teacher_cls: Teacher class scores (B, H, W, nc)
            
        Returns:
            KL divergence loss (scalar)
        """
        # Apply temperature scaling and sigmoid to get soft probabilities
        # For sigmoid with temperature: sigma(x/T)
        student_probs = tf.nn.sigmoid(student_cls / self.temperature)
        teacher_probs = tf.nn.sigmoid(teacher_cls / self.temperature)
        
        # For binary classification, we need to handle both positive and negative classes
        # P(class=1) = p, P(class=0) = 1-p
        # KL divergence for Bernoulli: teacher*log(teacher/student) + (1-teacher)*log((1-teacher)/(1-student))
        
        kl_pos = teacher_probs * (
            tf.math.log(teacher_probs + self.eps) - tf.math.log(student_probs + self.eps)
        )
        kl_neg = (1 - teacher_probs) * (
            tf.math.log(1 - teacher_probs + self.eps) - tf.math.log(1 - student_probs + self.eps)
        )
        
        kl_loss = kl_pos + kl_neg
        
        # Average over all dimensions
        kl_loss = tf.reduce_mean(kl_loss)
        
        # Scale by temperature^2
        kl_loss = kl_loss * (self.temperature ** 2)
        
        return kl_loss
    
    def get_config(self):
        """Return configuration for serialization"""
        config = super().get_config()
        config.update({
            'temperature': self.temperature,
            'alpha': self.alpha,
            'beta': self.beta,
            'dtype': self.dtype_
        })
        return config
