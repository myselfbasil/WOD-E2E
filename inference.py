"""
Inference script for Waymo Open Dataset Challenge.
Loads a trained model and generates predictions on test data.
"""
import os
import sys
import time
import argparse
import yaml
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import glob
import json
from datetime import datetime

# Add the project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from data.data_loader import WaymoDataLoader
from models.e2e_driving_model import build_model


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def predict(model, test_dataset):
    """Generate predictions for test dataset."""
    all_predictions = []
    all_scenario_ids = []
    
    for batch in tqdm(test_dataset, desc="Generating predictions"):
        inputs = batch  # For test data, batch is just inputs (no targets)
        
        # Forward pass
        outputs = model(inputs, training=False)
        
        # Get predictions
        pred_waypoints = outputs['pred_waypoints'].numpy()
        
        # Get scenario IDs if available
        if 'scenario_id' in inputs:
            scenario_ids = inputs['scenario_id'].numpy()
        else:
            # Generate placeholder IDs if not available
            scenario_ids = np.array([f"scenario_{i}" for i in range(pred_waypoints.shape[0])])
        
        # Store predictions and IDs
        all_predictions.append(pred_waypoints)
        all_scenario_ids.extend(scenario_ids)
    
    # Concatenate batch predictions
    predictions = np.concatenate(all_predictions, axis=0)
    
    return predictions, all_scenario_ids


def format_predictions_for_submission(predictions, scenario_ids, output_path):
    """Format predictions in the required format for submission."""
    submission = {}
    
    for i, scenario_id in enumerate(scenario_ids):
        # Convert scenario_id to string if it's not already
        scenario_id_str = str(scenario_id)
        
        # Extract waypoints for this scenario
        waypoints = predictions[i]  # Shape [horizon, 2]
        
        # Format as required for submission
        submission[scenario_id_str] = waypoints.tolist()
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(submission, f)
    
    print(f"Saved predictions for {len(submission)} scenarios to {output_path}")
    return submission


def main(args):
    """Main inference function."""
    # Load configuration
    config = load_config(args.config)
    
    # Get current directory (where inference.py is located)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Convert relative paths to absolute paths
    data_dir = os.path.join(current_dir, config['paths']['data_dir'])
    checkpoint_dir = os.path.join(current_dir, args.checkpoint_dir if args.checkpoint_dir else config['paths']['checkpoint_dir'])
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(current_dir, 'predictions', f"pred_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create data loader
    data_loader = WaymoDataLoader(config)
    
    # Get test TFRecord files
    if args.test_files:
        # Use specific test files if provided
        test_files = args.test_files.split(',')
        test_files = [os.path.join(data_dir, f) if not os.path.isabs(f) else f for f in test_files]
    else:
        # Use all files in the data directory by default
        test_files = sorted(glob.glob(os.path.join(data_dir, '*.tfrecord*')))
    
    print(f"Found {len(test_files)} test files")
    
    # Create test dataset
    test_dataset = data_loader.create_dataset(
        test_files,
        is_training=False,
        is_test=True,  # This is important to indicate we're in test mode
        batch_size=args.batch_size,
        shuffle=False
    )
    
    # Build model
    model = build_model(config)
    
    # Load checkpoint
    checkpoint = tf.train.Checkpoint(model=model)
    
    if os.path.isdir(args.checkpoint_path):
        # It's a checkpoint directory, find the latest checkpoint
        latest_checkpoint = tf.train.latest_checkpoint(args.checkpoint_path)
        if latest_checkpoint is None:
            raise ValueError(f"No checkpoints found in directory: {args.checkpoint_path}")
        checkpoint_path = latest_checkpoint
    else:
        # It's a specific checkpoint file
        checkpoint_path = args.checkpoint_path
    
    # Restore checkpoint
    checkpoint.restore(checkpoint_path).expect_partial()
    print(f"Restored model from checkpoint: {checkpoint_path}")
    
    # Generate predictions
    predictions, scenario_ids = predict(model, test_dataset)
    
    # Format and save predictions
    output_path = os.path.join(output_dir, 'predictions.json')
    format_predictions_for_submission(predictions, scenario_ids, output_path)
    
    print("Inference complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate predictions for Waymo Challenge")
    parser.add_argument("--config", type=str, default="configs/model_config.yaml",
                        help="Path to config file")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to checkpoint file or directory with checkpoints")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Override checkpoint directory from config")
    parser.add_argument("--test_files", type=str, default=None,
                        help="Comma-separated list of specific test files to use (optional)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for inference")
    args = parser.parse_args()
    
    main(args)
