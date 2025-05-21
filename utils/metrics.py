"""
Evaluation metrics for Waymo Open Dataset Challenge.
Implements the Rater Feedback Score (RFS) and other metrics.
"""
import tensorflow as tf
import numpy as np
from typing import Dict, List, Tuple, Optional, Union


def average_displacement_error(
    pred_waypoints: Union[tf.Tensor, np.ndarray], 
    true_waypoints: Union[tf.Tensor, np.ndarray],
    mask: Optional[Union[tf.Tensor, np.ndarray]] = None
) -> Union[tf.Tensor, float]:
    """
    Average displacement error between predicted and true waypoints.
    
    Args:
        pred_waypoints: Predicted waypoints [batch_size, horizon, 2]
        true_waypoints: Ground truth waypoints [batch_size, horizon, 2]
        mask: Optional mask for valid waypoints [batch_size, horizon]
        
    Returns:
        Average displacement error
    """
    # Convert to numpy if tensors
    if isinstance(pred_waypoints, tf.Tensor):
        pred_waypoints = pred_waypoints.numpy()
    if isinstance(true_waypoints, tf.Tensor):
        true_waypoints = true_waypoints.numpy()
    if mask is not None and isinstance(mask, tf.Tensor):
        mask = mask.numpy()
    
    # Compute L2 distance
    l2_dist = np.sqrt(np.sum(np.square(pred_waypoints - true_waypoints), axis=-1))
    
    if mask is not None:
        # Apply mask
        l2_dist = l2_dist * mask
        # Compute mean over valid points
        ade = np.sum(l2_dist) / np.maximum(np.sum(mask), 1.0)
    else:
        # Compute mean over all points
        ade = np.mean(l2_dist)
    
    return float(ade)


def final_displacement_error(
    pred_waypoints: Union[tf.Tensor, np.ndarray], 
    true_waypoints: Union[tf.Tensor, np.ndarray],
    mask: Optional[Union[tf.Tensor, np.ndarray]] = None
) -> Union[tf.Tensor, float]:
    """
    Final displacement error between predicted and true waypoints.
    
    Args:
        pred_waypoints: Predicted waypoints [batch_size, horizon, 2]
        true_waypoints: Ground truth waypoints [batch_size, horizon, 2]
        mask: Optional mask for valid waypoints [batch_size, horizon]
        
    Returns:
        Final displacement error
    """
    # Convert to numpy if tensors
    if isinstance(pred_waypoints, tf.Tensor):
        pred_waypoints = pred_waypoints.numpy()
    if isinstance(true_waypoints, tf.Tensor):
        true_waypoints = true_waypoints.numpy()
    if mask is not None and isinstance(mask, tf.Tensor):
        mask = mask.numpy()
    
    # Get final waypoints
    final_pred = pred_waypoints[:, -1]
    final_true = true_waypoints[:, -1]
    
    # Compute L2 distance
    l2_dist = np.sqrt(np.sum(np.square(final_pred - final_true), axis=-1))
    
    if mask is not None:
        # Get final mask
        final_mask = mask[:, -1]
        # Apply mask
        l2_dist = l2_dist * final_mask
        # Compute mean over valid points
        fde = np.sum(l2_dist) / np.maximum(np.sum(final_mask), 1.0)
    else:
        # Compute mean over all points
        fde = np.mean(l2_dist)
    
    return float(fde)


def compute_trust_region_thresholds(t: int, initial_speed: float) -> Tuple[float, float]:
    """
    Compute longitudinal and lateral thresholds for trust region.
    
    Args:
        t: Time index (e.g., 0-3 for 0-3 seconds, 4-5 for 4-5 seconds)
        initial_speed: Initial speed in m/s
        
    Returns:
        Tuple of (lateral_threshold, longitudinal_threshold)
    """
    # Convert speed to km/h for threshold calculation
    speed_kmh = initial_speed * 3.6
    
    # Base thresholds for t=3
    if t <= 3:
        lat_base = 1.0
        lon_base = 4.0
    else:  # t=4 or t=5
        lat_base = 1.8
        lon_base = 7.2
    
    # Scale factor based on speed
    if speed_kmh < 5.0:
        scale = 0.5
    elif speed_kmh >= 40.0:
        scale = 1.0
    else:
        # Linear interpolation between 5 and 40 km/h
        scale = 0.5 + (speed_kmh - 5.0) * 0.5 / 35.0
    
    # Apply scaling
    lateral_threshold = lat_base * scale
    longitudinal_threshold = lon_base * scale
    
    return lateral_threshold, longitudinal_threshold


def transform_to_local_frame(
    delta: np.ndarray, 
    heading: float
) -> np.ndarray:
    """
    Transform delta from global frame to local frame.
    
    Args:
        delta: Delta in global frame [x, y]
        heading: Heading in radians
        
    Returns:
        Delta in local frame [longitudinal, lateral]
    """
    # Create rotation matrix
    cos_h = np.cos(-heading)
    sin_h = np.sin(-heading)
    rotation = np.array([
        [cos_h, -sin_h],
        [sin_h, cos_h]
    ])
    
    # Transform delta
    local_delta = np.dot(rotation, delta)
    
    return local_delta


def is_within_trust_region(
    pred_waypoint: np.ndarray,
    true_waypoint: np.ndarray,
    heading: float,
    t: int,
    initial_speed: float
) -> bool:
    """
    Check if predicted waypoint is within trust region.
    
    Args:
        pred_waypoint: Predicted waypoint [x, y]
        true_waypoint: Ground truth waypoint [x, y]
        heading: Heading in radians
        t: Time index
        initial_speed: Initial speed in m/s
        
    Returns:
        Whether predicted waypoint is within trust region
    """
    # Compute delta
    delta = pred_waypoint - true_waypoint
    
    # Transform to local frame
    local_delta = transform_to_local_frame(delta, heading)
    
    # Get longitudinal and lateral components
    lon_delta = np.abs(local_delta[0])
    lat_delta = np.abs(local_delta[1])
    
    # Get thresholds
    lat_thresh, lon_thresh = compute_trust_region_thresholds(t, initial_speed)
    
    # Check if within trust region
    return lon_delta <= lon_thresh and lat_delta <= lat_thresh


def compute_rater_feedback_score(
    pred_waypoints: np.ndarray,
    rater_waypoints_list: List[np.ndarray],
    rater_scores: List[float],
    headings: np.ndarray,
    initial_speed: float
) -> float:
    """
    Compute Rater Feedback Score (RFS) for predicted trajectory.
    
    Args:
        pred_waypoints: Predicted waypoints [horizon, 2]
        rater_waypoints_list: List of rater-specified trajectories [num_raters, horizon, 2]
        rater_scores: List of rater scores [num_raters]
        headings: Heading at each timestep [horizon]
        initial_speed: Initial speed in m/s
        
    Returns:
        Rater Feedback Score
    """
    num_raters = len(rater_waypoints_list)
    horizon = pred_waypoints.shape[0]
    
    # For each rater trajectory, check if prediction is within trust region
    best_rater_idx = -1
    best_rater_score = 0.0
    min_distance = float('inf')
    
    for r in range(num_raters):
        rater_waypoints = rater_waypoints_list[r]
        
        # Check if all waypoints are within trust region
        all_within = True
        total_distance = 0.0
        
        for t in range(horizon):
            delta = pred_waypoints[t] - rater_waypoints[t]
            total_distance += np.sqrt(np.sum(np.square(delta)))
            
            within = is_within_trust_region(
                pred_waypoints[t],
                rater_waypoints[t],
                headings[t],
                t,
                initial_speed
            )
            
            if not within:
                all_within = False
                break
        
        if all_within:
            # If within trust region of multiple raters, choose closest
            if total_distance < min_distance:
                min_distance = total_distance
                best_rater_idx = r
                best_rater_score = rater_scores[r]
    
    # If not within any trust region, compute score based on closest rater
    if best_rater_idx == -1:
        # Find closest rater trajectory by ADE
        for r in range(num_raters):
            rater_waypoints = rater_waypoints_list[r]
            distance = np.mean(np.sqrt(np.sum(np.square(pred_waypoints - rater_waypoints), axis=-1)))
            
            if distance < min_distance:
                min_distance = distance
                best_rater_idx = r
        
        # Compute exponentially decaying score based on distance
        normalized_dist = min_distance / 10.0  # Normalize by 10 meters
        score = 4.0 + 6.0 * np.exp(-normalized_dist)
        score = min(max(4.0, score), 10.0)  # Clip to range [4, 10]
    else:
        score = best_rater_score
    
    return float(score)


def compute_scenario_specific_scores(
    predictions: Dict[str, np.ndarray],
    targets: Dict[str, np.ndarray],
    scenario_types: np.ndarray
) -> Dict[str, float]:
    """
    Compute scenario-specific ADE scores.
    
    Args:
        predictions: Dictionary of prediction arrays
        targets: Dictionary of target arrays
        scenario_types: Array of scenario type IDs
        
    Returns:
        Dictionary of scenario-specific scores
    """
    # Map scenario type IDs to names
    scenario_map = {
        0: "construction",
        1: "intersection",
        2: "pedestrian",
        3: "cyclist",
        4: "multi_lane_maneuver",
        5: "single_lane_maneuver",
        6: "cut_in",
        7: "foreign_object_debris",
        8: "special_vehicle",
        9: "spotlight",
        10: "others"
    }
    
    # Initialize scores
    scenario_scores = {name: [] for name in scenario_map.values()}
    
    # Get predictions and targets
    pred_waypoints = predictions["pred_waypoints"]
    true_waypoints = targets["future_waypoints"]
    
    # Compute ADE for each example
    for i in range(len(scenario_types)):
        scenario_id = scenario_types[i]
        
        # Handle different scenario_id formats
        if hasattr(scenario_id, 'numpy'):
            # TensorFlow tensor
            scenario_id_value = scenario_id.numpy()
        elif isinstance(scenario_id, np.ndarray):
            # Already a numpy array - for one-hot encoded, get the index
            if len(scenario_id.shape) > 0 and scenario_id.shape[0] > 1:
                scenario_id_value = np.argmax(scenario_id)
            else:
                scenario_id_value = int(scenario_id)
        else:
            # Regular Python value
            scenario_id_value = scenario_id
            
        # Get scenario name or default to "others"
        scenario_name = scenario_map.get(scenario_id_value, "others")
        
        # Compute ADE for this example
        ade = average_displacement_error(
            pred_waypoints[i:i+1],
            true_waypoints[i:i+1]
        )
        
        # Add to scenario-specific scores
        scenario_scores[scenario_name].append(ade)
    
    # Compute mean scores for each scenario
    scenario_means = {}
    for name, scores in scenario_scores.items():
        if scores:
            scenario_means[name] = float(np.mean(scores))
        else:
            scenario_means[name] = 0.0
    
    # Add overall ADE
    scenario_means["average"] = float(average_displacement_error(pred_waypoints, true_waypoints))
    
    # Add ADE at 3 and 5 seconds if horizon is at least 5 seconds
    horizon = pred_waypoints.shape[1]
    if horizon >= 30:  # Assuming 10Hz, 3 seconds = 30 frames
        ade_3s = average_displacement_error(
            pred_waypoints[:, :30],
            true_waypoints[:, :30]
        )
        scenario_means["ade_at_3_seconds"] = float(ade_3s)
    
    if horizon >= 50:  # Assuming 10Hz, 5 seconds = 50 frames
        ade_5s = average_displacement_error(
            pred_waypoints,
            true_waypoints
        )
        scenario_means["ade_at_5_seconds"] = float(ade_5s)
    
    return scenario_means
