# Waymo Open Dataset End-to-End Driving Challenge

This repository contains a complete pipeline for the Waymo Open Dataset Vision-based End-to-End Driving Challenge. The pipeline handles data preprocessing, model training, inference, and submission preparation.

## Quick Start

To run the entire pipeline in sequence:

```bash
cd /Users/basilshaji/Projects/wod-challenges/training
python run_pipeline.py
```

## Setup Instructions

1. **Install Dependencies**:
   ```bash
   cd /Users/basilshaji/Projects/wod-challenges/training
   pip install -r requirements.txt
   ```

2. **Verify Data**:
   - Ensure your TFRecord files are in the `/training/tfrecords` directory
   - The pipeline automatically splits these files for training and validation

## Step-by-Step Execution

### 1. Training the Model

```bash
cd /Users/basilshaji/Projects/wod-challenges/training
python train.py --config configs/model_config.yaml
```

This will:
- Split your data into training (80%) and validation (20%) sets
- Train the model according to the configuration
- Save checkpoints to the `checkpoints` directory
- Log training metrics and visualizations to the `outputs` directory

**Training Parameters**:
- Model architecture is defined in the configuration file
- Default batch size is 16
- Default learning rate is 0.0003 with cosine decay
- Training runs for 100 epochs by default

### 2. Generating Predictions

Once you have a trained model, generate predictions on test data:

```bash
python inference.py --checkpoint_path checkpoints/ckpt-XX --batch_size 32
```

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
