# Waymo Open Dataset End-to-End Driving Challenge

This repository contains a complete training pipeline for the Waymo Open Dataset Vision-based End-to-End Driving Challenge. The code handles data processing from TFRecord files, model training using a state-of-the-art architecture with attention mechanisms, and evaluation using the official Waymo metrics.

## Challenge Overview

The Waymo End-to-End Driving Challenge requires participants to predict 5-second future waypoints for autonomous driving based on camera images and historical data. Success is measured using the Rater Feedback Score (RFS) as the primary evaluation metric.

## Repository Structure

- `/configs` - Configuration files for model architecture and training
- `/data` - Data loading and processing modules
- `/models` - Model architecture definitions
- `/utils` - Utilities for losses, metrics, and visualization
- `/tfrecords` - Directory for TFRecord files (not included in repo)
- `/checkpoints` - Directory for saving model checkpoints
- `/outputs` - Directory for saving logs and visualizations

## Setup Instructions

1. **Create a Virtual Environment** (recommended):
   ```bash
   python -m venv wod_env
   source wod_env/bin/activate  # On Windows: wod_env\Scripts\activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare Data**:
   - Place your Waymo Open Dataset TFRecord files in the `tfrecords` directory
   - The code automatically handles train/validation splitting

## Training Pipeline

### 1. Start Model Training

```bash
python train.py --config configs/model_config.yaml
```

This will:
- Load TFRecord files and prepare training/validation datasets
- Build the model architecture as specified in the config
- Train the model with the specified parameters
- Log metrics and save checkpoints automatically

The model is configured specifically for the Waymo Challenge requirements:
- 5-second future trajectory prediction
- 4Hz prediction frequency (20 total waypoints)
- Evaluation using Rater Feedback Score (RFS)

### 2. Training Configuration

The `configs/model_config.yaml` file contains all settings for the model and training process including:

- Model architecture parameters
- Learning rate and optimizer settings
- Input/output specifications
- Loss function weights
- Evaluation metrics

### 3. Generating Predictions and Submissions

Once you have a trained model, generate predictions for the Waymo Challenge using the submission script:

```bash
python generate_submission.py --checkpoint checkpoints/model_best.h5 --test_dir tfrecords/test --output submission/waymo_submission
```

This will:
- Load your trained model from the checkpoint
- Process test TFRecord files
- Generate predictions in the exact format required by the Waymo Challenge:
  - 20 waypoints per scenario (5 seconds at 4Hz)
  - Proper proto formatting following the official submission guidelines
- Save the submission file(s) ready for upload to the challenge website

Where:
- `--checkpoint_path` specifies the checkpoint to use (replace XX with the checkpoint number)
- `--batch_size` sets the batch size for inference

This will:
- Load the specified model checkpoint
- Process test data from the `tfrecords` directory
- Generate predictions and save them to the `predictions` directory

### 3. Preparing Submission

Validate and package your predictions for submission:

```bash
python scripts/prepare_submission.py --predictions predictions/pred_YYYYMMDD_HHMMSS/predictions.json
```

This will:
- Validate the prediction format
- Run sanity checks on your predictions
- Create a ZIP file ready for submission

### 4. Submitting Results

1. Go to the Waymo Challenge website
2. Upload your submission ZIP file (created in step 3)
3. Wait for the evaluation results
4. Check your position on the leaderboard

## Using the All-in-One Pipeline

For convenience, you can run the entire pipeline at once:

```bash
python run_pipeline.py
```

Options:
- `--skip_training`: Skip the training step (requires `--checkpoint_path`)
- `--skip_inference`: Skip the inference step (requires `--predictions_path`)
- `--skip_submission_prep`: Skip the submission preparation step

Example of running only inference and submission preparation:
```bash
python run_pipeline.py --skip_training --checkpoint_path checkpoints/ckpt-20
```

## Key Files and Directories

- `/training/configs/model_config.yaml`: Model and training configuration
- `/training/tfrecords/`: TFRecord data files
- `/training/models/`: Model architecture implementation
- `/training/data/`: Data loading and processing
- `/training/utils/`: Utilities for losses, metrics, and visualization
- `/training/checkpoints/`: Saved model checkpoints
- `/training/outputs/`: Training outputs and visualizations
- `/training/predictions/`: Generated predictions for submission

## Model Architecture

The model uses an end-to-end architecture with:
- EfficientNet B3 backbone for processing camera images
- Attention mechanisms for feature fusion
- LSTM for temporal modeling
- Scenario-specific prediction heads

## Evaluation Metrics

The primary metrics for the challenge are:
- Rater Feedback Score (RFS): Main ranking metric
- Average Displacement Error (ADE): Secondary metric
- Scenario-specific scores for different driving conditions

## Troubleshooting

- **Out of memory errors**: Reduce batch size in the configuration
- **Slow training**: Ensure you're using GPU acceleration if available
- **Missing dependencies**: Check that all packages in requirements.txt are installed
- **Path errors**: Verify that the directory structure matches what's expected
