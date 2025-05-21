"""
Prepare submission for Waymo Open Dataset Challenge.
This script:
1. Validates the prediction format
2. Creates a submission package
3. Performs sanity checks on the predictions
"""
import os
import sys
import json
import argparse
import numpy as np
import zipfile
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def validate_predictions(predictions_path):
    """Validate prediction file format and contents."""
    print(f"Validating predictions file: {predictions_path}")
    
    # Check if file exists
    if not os.path.exists(predictions_path):
        print(f"ERROR: Predictions file not found at {predictions_path}")
        return False
    
    # Load predictions
    try:
        with open(predictions_path, 'r') as f:
            predictions = json.load(f)
    except json.JSONDecodeError:
        print("ERROR: Failed to parse predictions file as JSON")
        return False
    
    # Check if predictions is a dictionary
    if not isinstance(predictions, dict):
        print("ERROR: Predictions must be a dictionary mapping scenario IDs to trajectories")
        return False
    
    # Check number of scenarios
    num_scenarios = len(predictions)
    print(f"Number of scenarios in predictions: {num_scenarios}")
    
    # Validate structure for each scenario
    valid = True
    for scenario_id, trajectory in predictions.items():
        # Check trajectory is a list
        if not isinstance(trajectory, list):
            print(f"ERROR in scenario {scenario_id}: Trajectory must be a list of waypoints")
            valid = False
            continue
        
        # Check number of waypoints (should be 50 for 5 seconds at 10Hz)
        if len(trajectory) != 50:
            print(f"WARNING in scenario {scenario_id}: Expected 50 waypoints (5s at 10Hz), found {len(trajectory)}")
        
        # Check each waypoint is a list of 2 coordinates
        for i, waypoint in enumerate(trajectory):
            if not isinstance(waypoint, list) or len(waypoint) != 2:
                print(f"ERROR in scenario {scenario_id}, waypoint {i}: Waypoint must be [x, y]")
                valid = False
                break
            
            # Check coordinates are numbers
            for j, coord in enumerate(waypoint):
                if not isinstance(coord, (int, float)):
                    print(f"ERROR in scenario {scenario_id}, waypoint {i}: Coordinate {j} is not a number")
                    valid = False
    
    # Check for any NaN or infinity values
    for scenario_id, trajectory in predictions.items():
        trajectory_array = np.array(trajectory)
        if np.any(np.isnan(trajectory_array)) or np.any(np.isinf(trajectory_array)):
            print(f"ERROR in scenario {scenario_id}: Trajectory contains NaN or infinity values")
            valid = False
    
    if valid:
        print("✅ Predictions file format is valid")
    else:
        print("❌ Predictions file has format errors")
    
    return valid


def create_submission_package(predictions_path, output_dir=None, model_name=None):
    """Create a submission package (zip file) with predictions and metadata."""
    if output_dir is None:
        # Use directory containing predictions file
        output_dir = os.path.dirname(predictions_path)
    
    if model_name is None:
        # Use timestamp as model name
        model_name = f"waymo_submission_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create submission package path
    package_path = os.path.join(output_dir, f"{model_name}.zip")
    
    # Create metadata
    metadata = {
        "model_name": model_name,
        "timestamp": datetime.now().isoformat(),
        "prediction_file": os.path.basename(predictions_path)
    }
    
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Create zip file
    with zipfile.ZipFile(package_path, 'w') as zipf:
        # Add predictions file
        zipf.write(predictions_path, arcname=os.path.basename(predictions_path))
        
        # Add metadata
        zipf.write(metadata_path, arcname="metadata.json")
    
    # Remove temporary metadata file
    os.remove(metadata_path)
    
    print(f"Created submission package: {package_path}")
    return package_path


def run_sanity_checks(predictions_path):
    """Run sanity checks on predictions to catch common issues."""
    print(f"Running sanity checks on predictions: {predictions_path}")
    
    # Load predictions
    with open(predictions_path, 'r') as f:
        predictions = json.load(f)
    
    # Check for unrealistic distances between consecutive waypoints
    max_step_distance = 0
    max_scenario_id = None
    max_step_index = None
    
    for scenario_id, trajectory in predictions.items():
        trajectory_array = np.array(trajectory)
        
        # Compute distances between consecutive waypoints
        diffs = np.diff(trajectory_array, axis=0)
        distances = np.sqrt(np.sum(np.square(diffs), axis=1))
        
        # Check maximum step
        scenario_max = np.max(distances)
        if scenario_max > max_step_distance:
            max_step_distance = scenario_max
            max_scenario_id = scenario_id
            max_step_index = np.argmax(distances)
    
    print(f"Maximum step distance: {max_step_distance:.2f} meters")
    print(f"  Found in scenario {max_scenario_id} between waypoints {max_step_index} and {max_step_index+1}")
    
    if max_step_distance > 10.0:
        print("⚠️ WARNING: Maximum step distance is very large (> 10m). This might indicate prediction issues.")
    else:
        print("✅ Step distances appear reasonable")
    
    # Check for stationary predictions
    min_total_distance = float('inf')
    min_scenario_id = None
    
    for scenario_id, trajectory in predictions.items():
        trajectory_array = np.array(trajectory)
        
        # Compute total distance traveled
        diffs = np.diff(trajectory_array, axis=0)
        distances = np.sqrt(np.sum(np.square(diffs), axis=1))
        total_distance = np.sum(distances)
        
        if total_distance < min_total_distance:
            min_total_distance = total_distance
            min_scenario_id = scenario_id
    
    print(f"Minimum total trajectory distance: {min_total_distance:.2f} meters")
    print(f"  Found in scenario {min_scenario_id}")
    
    if min_total_distance < 1.0:
        print("⚠️ WARNING: Some trajectories barely move (<1m). This might indicate prediction issues.")
    else:
        print("✅ All trajectories show movement")
    
    return True


def main(args):
    """Main function."""
    # Validate predictions
    valid = validate_predictions(args.predictions)
    if not valid and not args.force:
        print("Validation failed. Use --force to continue anyway.")
        return
    
    # Run sanity checks
    if not args.skip_sanity:
        run_sanity_checks(args.predictions)
    
    # Create submission package
    create_submission_package(args.predictions, args.output_dir, args.model_name)
    
    print("Submission preparation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare submission for Waymo Challenge")
    parser.add_argument("--predictions", type=str, required=True,
                        help="Path to predictions JSON file")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for submission package")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Model name for submission")
    parser.add_argument("--force", action="store_true",
                        help="Force submission preparation even if validation fails")
    parser.add_argument("--skip_sanity", action="store_true",
                        help="Skip sanity checks")
    args = parser.parse_args()
    
    main(args)
