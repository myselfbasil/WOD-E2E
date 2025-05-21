"""
Visualization utilities for Waymo Open Dataset Challenge.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
import tensorflow as tf
from typing import Dict, List, Optional, Tuple, Union
import io
import cv2


def plot_trajectory(
    past_states: np.ndarray,
    future_states: Optional[np.ndarray] = None,
    pred_waypoints: Optional[np.ndarray] = None,
    route_waypoints: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (10, 10),
    title: str = "Trajectory Visualization",
    show_trust_region: bool = False,
    initial_speed: Optional[float] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot agent trajectory with past, future and predicted paths.
    
    Args:
        past_states: Past agent states [seq_len, state_dim]
        future_states: Ground truth future states [horizon, state_dim] (optional)
        pred_waypoints: Predicted waypoints [horizon, 2] (optional)
        route_waypoints: Route waypoints [num_waypoints, 2] (optional)
        figsize: Figure size
        title: Plot title
        show_trust_region: Whether to show trust region
        initial_speed: Initial speed for trust region (required if show_trust_region=True)
        save_path: Path to save figure (optional)
        
    Returns:
        Matplotlib figure
    """
    from .metrics import compute_trust_region_thresholds
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Extract past positions
    past_x, past_y = past_states[:, 0], past_states[:, 1]
    
    # Plot past trajectory
    ax.plot(past_x, past_y, 'b-', linewidth=2, label='Past Trajectory')
    ax.scatter(past_x[-1], past_y[-1], c='b', s=100, marker='o', label='Current Position')
    
    # Plot future trajectory if available
    if future_states is not None:
        future_x, future_y = future_states[:, 0], future_states[:, 1]
        ax.plot(future_x, future_y, 'g-', linewidth=2, label='Ground Truth')
        ax.scatter(future_x[-1], future_y[-1], c='g', s=100, marker='x', label='Future End')
    
    # Plot predicted trajectory if available
    if pred_waypoints is not None:
        pred_x, pred_y = pred_waypoints[:, 0], pred_waypoints[:, 1]
        ax.plot(pred_x, pred_y, 'r-', linewidth=2, label='Prediction')
        ax.scatter(pred_x[-1], pred_y[-1], c='r', s=100, marker='+', label='Predicted End')
        
        # Show trust region if requested
        if show_trust_region and future_states is not None and initial_speed is not None:
            # Get heading from future states
            if future_states.shape[1] > 3:  # If heading is included
                headings = future_states[:, 3]
            else:
                # Compute headings from positions
                dx = np.diff(np.concatenate([past_x[-1:], future_x]))
                dy = np.diff(np.concatenate([past_y[-1:], future_y]))
                headings = np.arctan2(dy, dx)
                headings = np.concatenate([headings[0:1], headings])
            
            # Plot trust regions for each timestep
            for i in range(len(future_x)):
                t = min(i // 10, 5)  # Time index (assuming 10Hz)
                lat_thresh, lon_thresh = compute_trust_region_thresholds(t, initial_speed)
                
                # Create a rotated ellipse for the trust region
                angle = np.degrees(headings[i])
                ellipse = patches.Ellipse(
                    (future_x[i], future_y[i]), 
                    width=2*lon_thresh, 
                    height=2*lat_thresh,
                    angle=angle,
                    fill=False, 
                    edgecolor='gray', 
                    alpha=0.5
                )
                ax.add_patch(ellipse)
    
    # Plot route if available
    if route_waypoints is not None:
        route_x, route_y = route_waypoints[:, 0], route_waypoints[:, 1]
        ax.plot(route_x, route_y, 'y--', linewidth=1, label='Route')
    
    # Set equal aspect and grid
    ax.set_aspect('equal')
    ax.grid(True)
    
    # Set title and legend
    ax.set_title(title)
    ax.legend()
    
    # Set axis labels
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def visualize_scenario_scores(scenario_scores: Dict[str, float], save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualize scenario-specific scores.
    
    Args:
        scenario_scores: Dictionary of scenario scores
        save_path: Path to save figure (optional)
        
    Returns:
        Matplotlib figure
    """
    # Remove non-scenario entries
    plot_scores = {k: v for k, v in scenario_scores.items() 
                  if k not in ['average', 'ade_at_3_seconds', 'ade_at_5_seconds']}
    
    # Sort by score
    sorted_items = sorted(plot_scores.items(), key=lambda x: x[1])
    scenarios, scores = zip(*sorted_items)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create horizontal bar plot
    bars = ax.barh(scenarios, scores, color='skyblue')
    
    # Add labels
    ax.set_xlabel('Score')
    ax.set_title('Scenario-Specific Scores')
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.1, bar.get_y() + bar.get_height()/2, f'{width:.2f}',
                ha='left', va='center')
    
    # Add overall average
    if 'average' in scenario_scores:
        avg = scenario_scores['average']
        ax.axvline(x=avg, color='r', linestyle='--', label=f'Average: {avg:.2f}')
        ax.legend()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def visualize_camera_images(images: np.ndarray, titles: Optional[List[str]] = None) -> plt.Figure:
    """
    Visualize camera images from all cameras.
    
    Args:
        images: Camera images [num_cameras, height, width, channels]
        titles: List of titles for each camera (optional)
        
    Returns:
        Matplotlib figure
    """
    num_cameras = images.shape[0]
    
    # Determine grid layout
    if num_cameras <= 4:
        nrows, ncols = 2, 2
    else:
        nrows = int(np.ceil(num_cameras / 3))
        ncols = min(num_cameras, 3)
    
    # Create figure
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
    axes = axes.flatten() if num_cameras > 1 else [axes]
    
    # Plot each camera view
    for i in range(num_cameras):
        axes[i].imshow(images[i])
        axes[i].set_title(titles[i] if titles else f'Camera {i}')
        axes[i].axis('off')
    
    # Hide unused axes
    for i in range(num_cameras, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig


def plot_to_tensor(figure: plt.Figure) -> tf.Tensor:
    """
    Convert matplotlib figure to tensorflow image tensor.
    Useful for TensorBoard visualization.
    
    Args:
        figure: Matplotlib figure
        
    Returns:
        Image tensor
    """
    # Save figure to buffer
    buf = io.BytesIO()
    figure.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    
    # Read image from buffer
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    
    # Convert to float32 and scale
    image = tf.cast(image, tf.float32) / 255.0
    
    return image


def overlay_trajectory_on_image(
    image: np.ndarray,
    past_states: np.ndarray,
    future_states: Optional[np.ndarray] = None,
    pred_waypoints: Optional[np.ndarray] = None,
    transform_matrix: Optional[np.ndarray] = None,
    image_size: Tuple[int, int] = (640, 480)
) -> np.ndarray:
    """
    Overlay trajectory on camera image using transformation matrix.
    
    Args:
        image: Camera image [height, width, channels]
        past_states: Past agent states [seq_len, state_dim]
        future_states: Ground truth future states [horizon, state_dim] (optional)
        pred_waypoints: Predicted waypoints [horizon, 2] (optional)
        transform_matrix: 3x3 transformation matrix from world to image coordinates
        image_size: Size of the image (width, height)
        
    Returns:
        Image with overlaid trajectory
    """
    # Make a copy of the image
    result = image.copy()
    width, height = image_size
    
    # If no transform matrix provided, create a simple top-down view
    if transform_matrix is None:
        # Simple scaling to fit in image
        pos_xs = np.concatenate([past_states[:, 0], 
                                future_states[:, 0] if future_states is not None else [],
                                pred_waypoints[:, 0] if pred_waypoints is not None else []])
        pos_ys = np.concatenate([past_states[:, 1], 
                                future_states[:, 1] if future_states is not None else [],
                                pred_waypoints[:, 1] if pred_waypoints is not None else []])
        
        min_x, max_x = np.min(pos_xs), np.max(pos_xs)
        min_y, max_y = np.min(pos_ys), np.max(pos_ys)
        
        # Add margin
        range_x = max(1.0, max_x - min_x) * 1.1
        range_y = max(1.0, max_y - min_y) * 1.1
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        min_x, max_x = center_x - range_x/2, center_x + range_x/2
        min_y, max_y = center_y - range_y/2, center_y + range_y/2
        
        # Create simple transform
        def world_to_image(points):
            points_img = np.zeros_like(points)
            points_img[:, 0] = (points[:, 0] - min_x) / range_x * width
            points_img[:, 1] = height - (points[:, 1] - min_y) / range_y * height
            return points_img
    else:
        # Use provided transform
        def world_to_image(points):
            # Homogeneous coordinates
            points_h = np.concatenate([points, np.ones((len(points), 1))], axis=1)
            points_img_h = np.dot(points_h, transform_matrix.T)
            points_img = points_img_h[:, :2] / points_img_h[:, 2:]
            return points_img
    
    # Transform points
    past_img = world_to_image(past_states[:, :2])
    
    # Draw past trajectory
    for i in range(1, len(past_img)):
        pt1 = tuple(past_img[i-1].astype(np.int32))
        pt2 = tuple(past_img[i].astype(np.int32))
        cv2.line(result, pt1, pt2, (255, 0, 0), 2)
    
    # Draw current position
    cv2.circle(result, tuple(past_img[-1].astype(np.int32)), 5, (255, 0, 0), -1)
    
    # Draw future trajectory if available
    if future_states is not None:
        future_img = world_to_image(future_states[:, :2])
        for i in range(1, len(future_img)):
            pt1 = tuple(future_img[i-1].astype(np.int32))
            pt2 = tuple(future_img[i].astype(np.int32))
            cv2.line(result, pt1, pt2, (0, 255, 0), 2)
    
    # Draw predicted trajectory if available
    if pred_waypoints is not None:
        pred_img = world_to_image(pred_waypoints)
        for i in range(1, len(pred_img)):
            pt1 = tuple(pred_img[i-1].astype(np.int32))
            pt2 = tuple(pred_img[i].astype(np.int32))
            cv2.line(result, pt1, pt2, (0, 0, 255), 2)
    
    return result
