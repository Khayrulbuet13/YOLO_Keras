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
    - (Optional) Intermediate features: Per-layer feature distributions
    - (Optional) Confidence calibration: Penalizes false positive predictions
    
    Args:
        temperature: Temperature for softening distributions (default: 3.0)
        alpha: Weight for box KD loss (default: 0.5)
        beta: Weight for class KD loss (default: 0.5)
        use_intermediate_kd: Enable per-layer KD on intermediate features (default: False)
        intermediate_weight: Weight for intermediate KD loss (default: 0.3)
        calibration_weight: Weight for confidence calibration loss (default: 0.5)
        conf_threshold: Confidence threshold for calibration (default: 0.25)
        dtype: Data type for computations (default: tf.float32)
    """
    
    def __init__(self, temperature=3.0, alpha=0.5, beta=0.5, 
                 use_intermediate_kd=False, intermediate_weight=0.3,
                 calibration_weight=0.5, conf_threshold=0.25,
                 dtype=tf.float32, **kwargs):
        super(KDLoss, self).__init__(dtype=dtype, **kwargs)
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.use_intermediate_kd = use_intermediate_kd
        self.intermediate_weight = intermediate_weight
        self.calibration_weight = calibration_weight
        self.conf_threshold = conf_threshold
        self.dtype_ = dtype
        self.eps = 1e-7
        
    def call(self, student_outputs, teacher_outputs, 
             student_intermediates=None, teacher_intermediates=None):
        """
        Compute KD loss between student and teacher outputs
        
        Args:
            student_outputs: Raw outputs from student model (B, H, W, 65)
                           Format: [box_preds (64 channels), class_scores (1 channel)]
            teacher_outputs: Raw outputs from teacher model (B, H, W, 65)
            student_intermediates: Optional dict of intermediate features from student
            teacher_intermediates: Optional dict of intermediate features from teacher
        
        Returns:
            Total KD loss (scalar tensor)
        """
        # Compute output-level KD loss
        loss_output = self.compute_output_kd(student_outputs, teacher_outputs)
        
        # Add intermediate KD loss if enabled
        if self.use_intermediate_kd and student_intermediates is not None and teacher_intermediates is not None:
            loss_intermediate = self.compute_intermediate_kd(student_intermediates, teacher_intermediates)
            total_loss = loss_output + self.intermediate_weight * loss_intermediate
            return total_loss
        
        return loss_output
    
    def compute_output_kd(self, student_outputs, teacher_outputs):
        """
        Compute KD loss on final outputs
        
        Args:
            student_outputs: Raw outputs from student model (B, H, W, 65)
            teacher_outputs: Raw outputs from teacher model (B, H, W, 65)
        
        Returns:
            Output KD loss (scalar tensor)
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
        
        # Compute confidence calibration loss (penalizes false positives)
        loss_calibration = self.compute_calibration_loss(student_cls, teacher_cls)
        
        # Weighted combination
        total_loss = self.alpha * loss_box + self.beta * loss_cls + self.calibration_weight * loss_calibration
        
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
    
    def compute_calibration_loss(self, student_cls, teacher_cls):
        """
        Compute confidence calibration loss to reduce false positives
        
        Penalizes the student when it has high confidence predictions
        that the teacher doesn't have (false positives).
        
        Args:
            student_cls: Student class logits (B, H, W, nc)
            teacher_cls: Teacher class logits (B, H, W, nc)
            
        Returns:
            Calibration loss (scalar)
        """
        # Get probabilities (no temperature - we want actual confidence)
        student_probs = tf.nn.sigmoid(student_cls)
        teacher_probs = tf.nn.sigmoid(teacher_cls)
        
        # Identify false positive regions:
        # Where student is confident but teacher is not
        student_above_thresh = tf.cast(student_probs > self.conf_threshold, self.dtype_)
        teacher_below_thresh = tf.cast(teacher_probs < self.conf_threshold, self.dtype_)
        
        # False positive mask: student confident AND teacher not confident
        false_positive_mask = student_above_thresh * teacher_below_thresh
        
        # Penalize: push student confidence down toward teacher's
        # Use MSE on the false positive predictions
        conf_diff = (student_probs - teacher_probs) ** 2
        
        # Apply mask and average
        masked_loss = conf_diff * false_positive_mask
        
        # Also add a softer penalty for all cases where student > teacher
        # This encourages the student to not over-predict
        over_prediction = tf.maximum(student_probs - teacher_probs, 0.0)
        over_prediction_loss = tf.reduce_mean(over_prediction ** 2)
        
        # Combine: strong penalty for false positives + soft penalty for over-prediction
        num_false_positives = tf.reduce_sum(false_positive_mask) + self.eps
        fp_loss = tf.reduce_sum(masked_loss) / num_false_positives
        
        calibration_loss = fp_loss + 0.1 * over_prediction_loss
        
        return calibration_loss
    
    def compute_intermediate_kd(self, student_feats, teacher_feats):
        """
        Compute KL divergence on intermediate feature activations
        
        Args:
            student_feats: Dict of intermediate features from student
            teacher_feats: Dict of intermediate features from teacher
        
        Returns:
            Average KL divergence across all matching layers (scalar tensor)
        """
        total_loss = tf.constant(0.0, dtype=self.dtype_)
        num_layers = 0
        
        # Iterate through student features and match with teacher
        for layer_name in student_feats.keys():
            # Skip the output layer (already handled in output KD)
            if layer_name == 'output':
                continue
                
            if layer_name in teacher_feats:
                s_feat = tf.cast(student_feats[layer_name], self.dtype_)
                t_feat = tf.cast(teacher_feats[layer_name], self.dtype_)
                
                # Normalize features to probability-like distributions
                # Apply softmax over spatial dimensions (H, W) for each channel
                # Reshape to (B, H*W, C) for softmax
                shape = tf.shape(s_feat)
                B, H, W, C = shape[0], shape[1], shape[2], shape[3]
                
                s_feat_flat = tf.reshape(s_feat, [B, H * W, C])
                t_feat_flat = tf.reshape(t_feat, [B, H * W, C])
                
                # Apply temperature scaling and softmax over spatial dimension
                s_dist = tf.nn.softmax(s_feat_flat / self.temperature, axis=1)
                t_dist = tf.nn.softmax(t_feat_flat / self.temperature, axis=1)
                
                # Compute KL divergence: KL(teacher || student)
                kl = t_dist * (
                    tf.math.log(t_dist + self.eps) - tf.math.log(s_dist + self.eps)
                )
                
                # Average over all dimensions and scale by temperature^2
                layer_loss = tf.reduce_mean(kl) * (self.temperature ** 2)
                
                total_loss = total_loss + layer_loss
                num_layers += 1
        
        # Return average loss across all layers
        if num_layers > 0:
            return total_loss / tf.cast(num_layers, self.dtype_)
        else:
            return total_loss
    
    def get_config(self):
        """Return configuration for serialization"""
        config = super().get_config()
        config.update({
            'temperature': self.temperature,
            'alpha': self.alpha,
            'beta': self.beta,
            'use_intermediate_kd': self.use_intermediate_kd,
            'intermediate_weight': self.intermediate_weight,
            'calibration_weight': self.calibration_weight,
            'conf_threshold': self.conf_threshold,
            'dtype': self.dtype_
        })
        return config
