#!/usr/bin/env python3
"""
End-to-end workflow for Waymo Open Dataset Challenge.
This script runs the complete pipeline:
1. Train model
2. Generate predictions
3. Prepare submission
"""
import os
import sys
import argparse
import subprocess
from datetime import datetime

def main(args):
    """Run the complete pipeline."""
    # Set up paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, args.config)
    
    # Create a run ID for this pipeline execution
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Starting pipeline run: {run_id}")
    
    # Step 1: Train model (if requested)
    if not args.skip_training:
        print("\n=== STEP 1: TRAINING MODEL ===")
        train_cmd = [
            sys.executable,
            os.path.join(current_dir, "train.py"),
            "--config", config_path
        ]
        
        print(f"Running command: {' '.join(train_cmd)}")
        train_process = subprocess.run(train_cmd)
        
        if train_process.returncode != 0:
            print("Training failed. Exiting pipeline.")
            return
        
        # Find latest checkpoint
        checkpoints_dir = os.path.join(current_dir, "outputs", f"run_{run_id}", "checkpoints")
        if not os.path.exists(checkpoints_dir):
            checkpoints_dir = os.path.join(current_dir, "checkpoints")
        
        checkpoint_path = checkpoints_dir
    else:
        # Use specified checkpoint
        checkpoint_path = args.checkpoint_path
        if not checkpoint_path:
            print("ERROR: Must specify --checkpoint_path when using --skip_training")
            return
    
    # Step 2: Generate predictions
    if not args.skip_inference:
        print("\n=== STEP 2: GENERATING PREDICTIONS ===")
        inference_cmd = [
            sys.executable,
            os.path.join(current_dir, "inference.py"),
            "--config", config_path,
            "--checkpoint_path", checkpoint_path,
            "--batch_size", str(args.batch_size)
        ]
        
        if args.test_files:
            inference_cmd.extend(["--test_files", args.test_files])
        
        print(f"Running command: {' '.join(inference_cmd)}")
        inference_process = subprocess.run(inference_cmd)
        
        if inference_process.returncode != 0:
            print("Inference failed. Exiting pipeline.")
            return
        
        # Find latest predictions
        predictions_dir = os.path.join(current_dir, "predictions")
        prediction_dirs = [os.path.join(predictions_dir, d) for d in os.listdir(predictions_dir) 
                         if os.path.isdir(os.path.join(predictions_dir, d))]
        latest_prediction_dir = max(prediction_dirs, key=os.path.getmtime)
        predictions_path = os.path.join(latest_prediction_dir, "predictions.json")
    else:
        # Use specified predictions file
        predictions_path = args.predictions_path
        if not predictions_path:
            print("ERROR: Must specify --predictions_path when using --skip_inference")
            return
    
    # Step 3: Prepare submission
    if not args.skip_submission_prep:
        print("\n=== STEP 3: PREPARING SUBMISSION ===")
        submission_cmd = [
            sys.executable,
            os.path.join(current_dir, "scripts", "prepare_submission.py"),
            "--predictions", predictions_path,
            "--model_name", f"waymo_submission_{run_id}"
        ]
        
        print(f"Running command: {' '.join(submission_cmd)}")
        submission_process = subprocess.run(submission_cmd)
        
        if submission_process.returncode != 0:
            print("Submission preparation failed.")
        else:
            print("\nPipeline completed successfully!")
            print("\nNEXT STEPS:")
            print("1. Upload your submission ZIP file to the Waymo Challenge website")
            print("2. Wait for evaluation results")
            print("3. Iterate based on feedback")
    else:
        print("\nPipeline completed without submission preparation.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run end-to-end Waymo Challenge pipeline")
    parser.add_argument("--config", type=str, default="configs/model_config.yaml",
                        help="Path to config file")
    parser.add_argument("--skip_training", action="store_true",
                        help="Skip training step")
    parser.add_argument("--skip_inference", action="store_true",
                        help="Skip inference step")
    parser.add_argument("--skip_submission_prep", action="store_true",
                        help="Skip submission preparation step")
    parser.add_argument("--checkpoint_path", type=str,
                        help="Path to checkpoint (required if skipping training)")
    parser.add_argument("--predictions_path", type=str,
                        help="Path to predictions file (required if skipping inference)")
    parser.add_argument("--test_files", type=str,
                        help="Comma-separated list of test files")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for inference")
    args = parser.parse_args()
    
    main(args)
