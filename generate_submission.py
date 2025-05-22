#!/usr/bin/env python
# coding: utf-8

"""
Submission generator for Waymo Open Dataset End-to-End Driving Challenge.
This script takes a trained model and generates predictions in the required submission format.
"""

import os
import tensorflow as tf
import numpy as np
import glob
import argparse
import yaml
import tqdm
from datetime import datetime

# Try to import the Waymo-specific protobuf libraries
try:
    from waymo_open_dataset.protos import wod_e2ed_submission_pb2
    WAYMO_PROTOS_AVAILABLE = True
except ImportError:
    print("Warning: Waymo Open Dataset protos not available. Using fallback submission format.")
    WAYMO_PROTOS_AVAILABLE = False

# Import project modules
from models.e2e_driving_model import build_model
from data.raw_tfrecord_loader import RawTFRecordLoader


def load_model(config, checkpoint_path):
    """
    Load the trained model from checkpoint.
    
    Args:
        config: Model configuration
        checkpoint_path: Path to model checkpoint
        
    Returns:
        Loaded model
    """
    # Build model
    model = build_model(config)
    
    # Load weights
    if os.path.exists(checkpoint_path):
        model.load_weights(checkpoint_path)
        print(f"Loaded weights from {checkpoint_path}")
    else:
        raise ValueError(f"Checkpoint not found at {checkpoint_path}")
    
    return model


def generate_predictions(model, data_loader, test_files, config):
    """
    Generate predictions for test data.
    
    Args:
        model: Trained model
        data_loader: Data loader
        test_files: List of test TFRecord files
        config: Configuration
        
    Returns:
        Dictionary of predictions
    """
    # Create test dataset
    test_dataset = data_loader.create_dataset(
        test_files,
        is_training=False,
        is_test=True,
        batch_size=config['training'].get('batch_size', 8),
        shuffle=False
    )
    
    # Generate predictions
    all_predictions = []
    all_frame_names = []
    
    for batch in tqdm.tqdm(test_dataset, desc="Generating predictions"):
        # For test data, we only have inputs
        inputs = batch
        
        # Forward pass
        outputs = model(inputs, training=False)
        
        # Extract predictions - waypoints are the key output for the challenge
        pred_waypoints = outputs['pred_waypoints'].numpy()
        
        # In a real implementation, we would extract frame names from the TFRecord
        # For now, generate placeholder frame names
        batch_size = pred_waypoints.shape[0]
        frame_names = [f"test_frame_{i}" for i in range(batch_size)]
        
        all_predictions.append(pred_waypoints)
        all_frame_names.extend(frame_names)
    
    # Concatenate predictions
    if all_predictions:
        all_predictions = np.concatenate(all_predictions, axis=0)
    else:
        all_predictions = np.array([])
    
    return {
        'predictions': all_predictions,
        'frame_names': all_frame_names
    }


def create_submission_proto(predictions, frame_names):
    """
    Create submission in Waymo proto format.
    
    Args:
        predictions: Numpy array of shape [N, 20, 2] for N scenarios, 20 waypoints (5s at 4Hz), x/y coords
        frame_names: List of frame names
        
    Returns:
        List of FrameTrajectoryPredictions
    """
    if not WAYMO_PROTOS_AVAILABLE:
        print("Warning: Waymo protos not available. Using dictionary format instead.")
        return [
            {
                'frame_name': frame_name,
                'trajectory': {
                    'pos_x': predictions[i, :, 0].tolist(),
                    'pos_y': predictions[i, :, 1].tolist()
                }
            }
            for i, frame_name in enumerate(frame_names)
        ]
    
    frame_trajectories = []
    for i, frame_name in enumerate(frame_names):
        # Create trajectory prediction - must be exactly 20 points at 4Hz
        trajectory = wod_e2ed_submission_pb2.TrajectoryPrediction(
            pos_x=predictions[i, :, 0].astype(np.float32),
            pos_y=predictions[i, :, 1].astype(np.float32)
        )
        
        # Create frame trajectory
        frame_trajectory = wod_e2ed_submission_pb2.FrameTrajectoryPredictions(
            frame_name=frame_name,
            trajectory=trajectory
        )
        
        frame_trajectories.append(frame_trajectory)
    
    return frame_trajectories


def save_submission(frame_trajectories, output_path, num_shards=1):
    """
    Save submission to file(s).
    
    Args:
        frame_trajectories: List of FrameTrajectoryPredictions
        output_path: Base path for output files
        num_shards: Number of shards to split submission into
    """
    if not WAYMO_PROTOS_AVAILABLE:
        # Fallback to JSON format
        import json
        with open(f"{output_path}.json", 'w') as f:
            json.dump(frame_trajectories, f)
        print(f"Saved submission to {output_path}.json")
        return
    
    # Split predictions into shards
    trajectories_per_shard = len(frame_trajectories) // num_shards
    
    for shard_idx in range(num_shards):
        start_idx = shard_idx * trajectories_per_shard
        end_idx = start_idx + trajectories_per_shard if shard_idx < num_shards - 1 else len(frame_trajectories)
        
        shard_trajectories = frame_trajectories[start_idx:end_idx]
        
        # Create submission
        submission = wod_e2ed_submission_pb2.E2EDSubmission()
        submission.frame_trajectories.extend(shard_trajectories)
        
        # Save to file
        output_file = f"{output_path}_{shard_idx:03d}.bin" if num_shards > 1 else f"{output_path}.bin"
        with open(output_file, 'wb') as f:
            f.write(submission.SerializeToString())
        
        print(f"Saved submission shard {shard_idx + 1}/{num_shards} to {output_file}")


def main(args):
    """Main function."""
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get all TFRecord files in the test directory
    test_files = sorted(glob.glob(os.path.join(args.test_dir, '*.tfrecord*')))
    if not test_files:
        print(f"No TFRecord files found in {args.test_dir}")
        return
    
    print(f"Found {len(test_files)} TFRecord files in {args.test_dir}")
    
    # Create data loader
    data_loader = RawTFRecordLoader(config)
    
    # Load model
    model = load_model(config, args.checkpoint)
    
    # Generate predictions
    print("Generating predictions...")
    prediction_results = generate_predictions(model, data_loader, test_files, config)
    
    # Create submission
    print("Creating submission...")
    frame_trajectories = create_submission_proto(
        prediction_results['predictions'],
        prediction_results['frame_names']
    )
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Save submission
    save_submission(frame_trajectories, args.output, args.num_shards)
    
    print("Submission generation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate submission for Waymo E2E Driving Challenge")
    parser.add_argument("--config", type=str, default="configs/model_config.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--test_dir", type=str, required=True, help="Directory containing test TFRecord files")
    parser.add_argument("--output", type=str, default="submission/waymo_e2e_submission", help="Output path for submission")
    parser.add_argument("--num_shards", type=int, default=1, help="Number of shards to split submission into")
    
    args = parser.parse_args()
    main(args)
