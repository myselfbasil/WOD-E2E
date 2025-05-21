"""
Loss functions for Waymo Open Dataset Challenge.
"""
import tensorflow as tf
import numpy as np
from typing import Dict, List, Tuple, Optional


def waypoint_loss(
    pred_waypoints: tf.Tensor, 
    true_waypoints: tf.Tensor,
    mask: Optional[tf.Tensor] = None
) -> tf.Tensor:
    """
    L2 loss between predicted and true waypoints.
    
    Args:
        pred_waypoints: Predicted waypoints [batch_size, horizon, 2]
        true_waypoints: Ground truth waypoints [batch_size, horizon, 2]
        mask: Optional mask for valid waypoints [batch_size, horizon]
        
    Returns:
        Mean squared error
    """
    squared_error = tf.reduce_sum(tf.square(pred_waypoints - true_waypoints), axis=-1)
    
    if mask is not None:
        # Apply mask
        squared_error = squared_error * mask
        # Compute mean over valid points
        loss = tf.reduce_sum(squared_error) / tf.maximum(tf.reduce_sum(mask), 1.0)
    else:
        # Compute mean over all points
        loss = tf.reduce_mean(squared_error)
    
    return loss


def average_displacement_error(
    pred_waypoints: tf.Tensor, 
    true_waypoints: tf.Tensor,
    mask: Optional[tf.Tensor] = None
) -> tf.Tensor:
    """
    Average displacement error between predicted and true waypoints.
    
    Args:
        pred_waypoints: Predicted waypoints [batch_size, horizon, 2]
        true_waypoints: Ground truth waypoints [batch_size, horizon, 2]
        mask: Optional mask for valid waypoints [batch_size, horizon]
        
    Returns:
        Average displacement error
    """
    # Compute L2 distance
    l2_dist = tf.sqrt(tf.reduce_sum(tf.square(pred_waypoints - true_waypoints), axis=-1))
    
    if mask is not None:
        # Apply mask
        l2_dist = l2_dist * mask
        # Compute mean over valid points
        ade = tf.reduce_sum(l2_dist) / tf.maximum(tf.reduce_sum(mask), 1.0)
    else:
        # Compute mean over all points
        ade = tf.reduce_mean(l2_dist)
    
    return ade


def final_displacement_error(
    pred_waypoints: tf.Tensor, 
    true_waypoints: tf.Tensor,
    mask: Optional[tf.Tensor] = None
) -> tf.Tensor:
    """
    Final displacement error between predicted and true waypoints.
    
    Args:
        pred_waypoints: Predicted waypoints [batch_size, horizon, 2]
        true_waypoints: Ground truth waypoints [batch_size, horizon, 2]
        mask: Optional mask for valid waypoints [batch_size, horizon]
        
    Returns:
        Final displacement error
    """
    # Get final waypoints
    final_pred = pred_waypoints[:, -1]
    final_true = true_waypoints[:, -1]
    
    # Compute L2 distance
    l2_dist = tf.sqrt(tf.reduce_sum(tf.square(final_pred - final_true), axis=-1))
    
    if mask is not None:
        # Get final mask
        final_mask = mask[:, -1]
        # Apply mask
        l2_dist = l2_dist * final_mask
        # Compute mean over valid points
        fde = tf.reduce_sum(l2_dist) / tf.maximum(tf.reduce_sum(final_mask), 1.0)
    else:
        # Compute mean over all points
        fde = tf.reduce_mean(l2_dist)
    
    return fde


def scenario_classification_loss(
    pred_logits: tf.Tensor, 
    true_labels: tf.Tensor
) -> tf.Tensor:
    """
    Scenario classification loss.
    
    Args:
        pred_logits: Predicted scenario logits [batch_size, num_classes]
        true_labels: Ground truth scenario labels [batch_size]
        
    Returns:
        Cross entropy loss
    """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=true_labels,
        logits=pred_logits
    )
    return tf.reduce_mean(loss)


def trust_region_loss(
    pred_waypoints: tf.Tensor, 
    true_waypoints: tf.Tensor, 
    initial_speed: tf.Tensor
) -> tf.Tensor:
    """
    Trust region loss based on longitudinal and lateral thresholds.
    
    Args:
        pred_waypoints: Predicted waypoints [batch_size, horizon, 2]
        true_waypoints: Ground truth waypoints [batch_size, horizon, 2]
        initial_speed: Initial speed [batch_size]
        
    Returns:
        Trust region loss
    """
    batch_size = tf.shape(pred_waypoints)[0]
    horizon = tf.shape(pred_waypoints)[1]
    
    # Compute delta between consecutive true waypoints
    true_diffs = true_waypoints[:, 1:] - true_waypoints[:, :-1]
    
    # Compute headings (assuming x is forward, y is left)
    headings = tf.math.atan2(true_diffs[..., 1], true_diffs[..., 0])
    
    # Heading at each waypoint (pad first with same as second)
    padded_headings = tf.concat([
        headings[:, :1],  # First heading
        headings
    ], axis=1)
    
    # Create rotation matrices for each waypoint
    cos_h = tf.cos(padded_headings)
    sin_h = tf.sin(padded_headings)
    
    # Create rotation matrices [batch, horizon, 2, 2]
    rot_matrices = tf.stack([
        tf.stack([cos_h, sin_h], axis=-1),
        tf.stack([-sin_h, cos_h], axis=-1)
    ], axis=-1)
    rot_matrices = tf.transpose(rot_matrices, [0, 1, 3, 2])
    
    # Compute delta between predicted and true waypoints
    delta = pred_waypoints - true_waypoints  # [batch, horizon, 2]
    
    # Reshape delta for matrix multiplication
    delta_reshaped = tf.expand_dims(delta, axis=-1)  # [batch, horizon, 2, 1]
    
    # Rotate delta to get longitudinal and lateral components
    rotated_delta = tf.matmul(rot_matrices, delta_reshaped)  # [batch, horizon, 2, 1]
    rotated_delta = tf.squeeze(rotated_delta, axis=-1)  # [batch, horizon, 2]
    
    # Extract longitudinal (x) and lateral (y) components
    lon_delta = rotated_delta[..., 0]  # [batch, horizon]
    lat_delta = rotated_delta[..., 1]  # [batch, horizon]
    
    # Compute longitudinal and lateral thresholds based on initial speed
    # Scale initial speed from m/s to km/h for threshold calculation
    speed_kmh = initial_speed * 3.6
    
    # Define scaling function based on speed
    def scale_fn(speed):
        cond1 = speed < 1.4  # 5 km/h
        cond2 = tf.logical_and(speed >= 1.4, speed < 11.1)  # 5-40 km/h
        cond3 = speed >= 11.1  # 40+ km/h
        
        result = tf.zeros_like(speed)
        result = tf.where(cond1, 0.5, result)
        
        # Linear interpolation for middle range
        mid_scale = 0.5 + (speed - 1.4) * (1.0 - 0.5) / (11.1 - 1.4)
        result = tf.where(cond2, mid_scale, result)
        
        result = tf.where(cond3, 1.0, result)
        return result
    
    scale = scale_fn(speed_kmh)
    
    # Compute t=3 thresholds (base case)
    lat_thresh_base = 1.0
    lon_thresh_base = 4.0
    
    # Scale thresholds for t=5
    t3_to_t5_lat_scale = 1.8
    t3_to_t5_lon_scale = 7.2
    
    # Create time-dependent threshold scaling
    # Linearly interpolate from t=0 to t=horizon
    t = tf.range(horizon, dtype=tf.float32) / tf.cast(horizon - 1, tf.float32)
    t = tf.reshape(t, [1, -1])  # [1, horizon]
    
    # Scale thresholds over time
    lat_thresh_scale = lat_thresh_base + t * (t3_to_t5_lat_scale - lat_thresh_base)
    lon_thresh_scale = lon_thresh_base + t * (t3_to_t5_lon_scale - lon_thresh_base)
    
    # Apply speed scaling to thresholds
    lat_thresh = tf.expand_dims(scale, axis=-1) * lat_thresh_scale  # [batch, horizon]
    lon_thresh = tf.expand_dims(scale, axis=-1) * lon_thresh_scale  # [batch, horizon]
    
    # Compute penalty when outside trust region
    # Use smooth L1 loss for better gradients
    def smooth_l1_outside_thresh(delta, thresh):
        abs_delta = tf.abs(delta)
        outside_mask = tf.cast(abs_delta > thresh, tf.float32)
        delta_outside = tf.maximum(abs_delta - thresh, 0.0)
        
        # Smooth L1 for large deviations outside threshold
        smooth_l1 = tf.where(
            delta_outside < 1.0,
            0.5 * tf.square(delta_outside),
            delta_outside - 0.5
        )
        
        return outside_mask * smooth_l1
    
    # Compute loss components
    lon_loss = smooth_l1_outside_thresh(lon_delta, lon_thresh)
    lat_loss = smooth_l1_outside_thresh(lat_delta, lat_thresh)
    
    # Sum losses over trajectory
    total_loss = tf.reduce_mean(lon_loss + 2.0 * lat_loss)  # Penalize lateral more
    
    return total_loss


def rater_feedback_score_loss(
    pred_waypoints: tf.Tensor, 
    true_waypoints: tf.Tensor, 
    initial_speed: tf.Tensor
) -> tf.Tensor:
    """
    Loss function approximating the Rater Feedback Score metric.
    Encourages predictions to stay within trust regions of ground truth trajectories.
    
    Args:
        pred_waypoints: Predicted waypoints [batch_size, horizon, 2]
        true_waypoints: Ground truth waypoints [batch_size, horizon, 2]
        initial_speed: Initial speed [batch_size]
        
    Returns:
        Loss approximating RFS metric
    """
    # We simplify the actual RFS metric for trainable loss
    # by focusing on the trust region concept
    
    # Trust region loss already implements the core concept
    trust_loss = trust_region_loss(pred_waypoints, true_waypoints, initial_speed)
    
    # Also include basic ADE for overall alignment
    ade = average_displacement_error(pred_waypoints, true_waypoints)
    
    # Combine losses - trust region concept is more important
    loss = 0.7 * trust_loss + 0.3 * ade
    
    return loss


def combined_loss(outputs: Dict, targets: Dict, config: Dict) -> Dict:
    """
    Combined loss function for training.
    
    Args:
        outputs: Dict of model outputs
        targets: Dict of targets
        config: Model configuration
        
    Returns:
        Dict of loss components and total loss
    """
    # Get predictions and targets
    pred_waypoints = outputs['pred_waypoints']
    true_waypoints = targets['future_waypoints']
    
    # Extract initial speed from past states
    past_states = targets.get('past_states')
    if past_states is not None:
        initial_speed = tf.sqrt(
            tf.square(past_states[:, -1, 4]) +  # velocity_x
            tf.square(past_states[:, -1, 5])    # velocity_y
        )
    else:
        # Default to mid-range speed if not available
        initial_speed = tf.ones([tf.shape(pred_waypoints)[0]]) * 5.0
    
    # Compute basic waypoint loss
    wp_loss = waypoint_loss(pred_waypoints, true_waypoints)
    
    # Compute ADE
    ade = average_displacement_error(pred_waypoints, true_waypoints)
    
    # Compute trust region loss
    trust_loss = trust_region_loss(pred_waypoints, true_waypoints, initial_speed)
    
    # Compute RFS-approximating loss
    rfs_loss = rater_feedback_score_loss(pred_waypoints, true_waypoints, initial_speed)
    
    # Initialize loss components
    loss_components = {
        'waypoint_loss': wp_loss,
        'ade': ade,
        'trust_loss': trust_loss,
        'rfs_loss': rfs_loss
    }
    
    # Add scenario classification loss if available
    if 'scenario_logits' in outputs and 'scenario_type' in targets:
        scenario_loss = scenario_classification_loss(
            outputs['scenario_logits'],
            targets['scenario_type']
        )
        loss_components['scenario_loss'] = scenario_loss
    else:
        scenario_loss = 0.0
    
    # Compute weighted sum of losses
    weights = config['loss_weights']
    total_loss = (
        weights['waypoint_loss'] * wp_loss +
        weights['ade_loss'] * ade +
        weights['trust_region_loss'] * trust_loss
    )
    
    # Add scenario classification loss if available
    if 'scenario_logits' in outputs and 'scenario_type' in targets:
        total_loss += weights['scenario_classification_loss'] * scenario_loss
    
    loss_components['total_loss'] = total_loss
    
    return loss_components
