"""
Main training script for Waymo Open Dataset Challenge.
"""
import os
import sys
import time
import argparse
import yaml
import tensorflow as tf
import numpy as np
from datetime import datetime
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import CosineDecay
import tqdm
import glob

# Add the project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from data.raw_tfrecord_loader import RawTFRecordLoader
from models.e2e_driving_model import build_model
from utils.losses import combined_loss
from utils.metrics import average_displacement_error, final_displacement_error, compute_scenario_specific_scores
from utils.visualization import plot_trajectory, visualize_scenario_scores, plot_to_tensor


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_optimizer(config):
    """Create optimizer based on configuration."""
    optimizer_config = config['optimizer']
    optimizer_name = optimizer_config['name']
    
    # Get learning rate
    learning_rate = config['training']['learning_rate']
    
    # Use learning rate schedule if specified
    if config['training'].get('lr_scheduler') == 'cosine':
        # Cosine decay schedule
        initial_lr = learning_rate
        epochs = config['training']['num_epochs']
        warmup_epochs = config['training']['warmup_epochs']
        steps_per_epoch = 100  # This will be updated when we know dataset size
        
        # Warmup followed by cosine decay
        warmup_steps = warmup_epochs * steps_per_epoch
        decay_steps = (epochs - warmup_epochs) * steps_per_epoch
        
        # Use TensorFlow's LearningRateSchedule class for proper callback
        class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
            def __init__(self, initial_lr, warmup_steps, decay_steps):
                super().__init__()
                self.initial_lr = initial_lr
                self.warmup_steps = warmup_steps
                self.decay_steps = decay_steps
                
            def __call__(self, step):
                # Convert to tensor to ensure graph compatibility
                step = tf.cast(step, tf.float32)
                warmup_steps = tf.cast(self.warmup_steps, tf.float32)
                decay_steps = tf.cast(self.decay_steps, tf.float32)
                initial_lr = tf.cast(self.initial_lr, tf.float32)
                
                # Linear warmup calculation
                warmup_lr = initial_lr * (step / warmup_steps)
                
                # Cosine decay calculation
                decay_step = step - warmup_steps
                decay_ratio = decay_step / decay_steps
                cosine_lr = initial_lr * 0.5 * (1 + tf.cos(tf.constant(np.pi) * decay_ratio))
                
                # Use tf.cond instead of if/else for graph compatibility
                lr = tf.cond(step < warmup_steps, 
                             lambda: warmup_lr,
                             lambda: cosine_lr)
                
                return lr
            
            def get_config(self):
                return {
                    'initial_lr': self.initial_lr,
                    'warmup_steps': self.warmup_steps,
                    'decay_steps': self.decay_steps
                }
        
        learning_rate = WarmupCosineDecay(initial_lr, warmup_steps, decay_steps)
    
    # Create optimizer
    if optimizer_name == 'adam':
        optimizer = Adam(
            learning_rate=learning_rate,
            beta_1=optimizer_config.get('beta1', 0.9),
            beta_2=optimizer_config.get('beta2', 0.999)
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    return optimizer


@tf.function
def train_step(model, optimizer, inputs, targets, config):
    """Single training step."""
    with tf.GradientTape() as tape:
        # Forward pass
        outputs = model(inputs, training=True)
        
        # Compute loss
        loss_dict = combined_loss(outputs, targets, config)
        total_loss = loss_dict['total_loss']
    
    # Compute gradients
    gradients = tape.gradient(total_loss, model.trainable_variables)
    
    # Apply gradients
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return outputs, loss_dict


@tf.function
def validation_step(model, inputs, targets, config):
    """Single validation step."""
    # Forward pass
    outputs = model(inputs, training=False)
    
    # Compute loss
    loss_dict = combined_loss(outputs, targets, config)
    
    return outputs, loss_dict


def evaluate_model(model, dataset, config):
    """Evaluate model on dataset."""
    all_predictions = []
    all_targets = []
    all_scenario_types = []
    all_losses = []
    
    for batch in dataset:
        inputs, targets = batch
        outputs, loss_dict = validation_step(model, inputs, targets, config)
        
        # Collect predictions and targets
        all_predictions.append({
            'pred_waypoints': outputs['pred_waypoints'].numpy()
        })
        all_targets.append({
            'future_waypoints': targets['future_waypoints'].numpy()
        })
        all_scenario_types.append(inputs['scenario_type'].numpy())
        all_losses.append({k: v.numpy() for k, v in loss_dict.items()})
    
    # Concatenate batches
    predictions = {
        'pred_waypoints': np.concatenate([p['pred_waypoints'] for p in all_predictions])
    }
    targets = {
        'future_waypoints': np.concatenate([t['future_waypoints'] for t in all_targets])
    }
    scenario_types = np.concatenate(all_scenario_types)
    
    # Compute metrics
    metrics = {}
    
    # Average metrics across batches
    losses = {k: np.mean([loss[k] for loss in all_losses]) for k in all_losses[0].keys()}
    metrics.update(losses)
    
    # ADE and FDE
    metrics['ade'] = average_displacement_error(
        predictions['pred_waypoints'],
        targets['future_waypoints']
    )
    metrics['fde'] = final_displacement_error(
        predictions['pred_waypoints'],
        targets['future_waypoints']
    )
    
    # Compute scenario-specific scores
    scenario_scores = compute_scenario_specific_scores(
        predictions,
        targets,
        scenario_types
    )
    metrics.update(scenario_scores)
    
    return metrics, predictions, targets, scenario_types


def main(args):
    """Main training function."""
    # Load configuration
    config = load_config(args.config)
    
    # Get current directory (where train.py is located)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Convert relative paths to absolute paths
    data_dir = os.path.join(current_dir, config['paths']['data_dir'])
    checkpoint_dir = os.path.join(current_dir, config['paths']['checkpoint_dir'])
    output_dir_base = os.path.join(current_dir, config['paths']['output_dir'])
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(output_dir_base, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save configuration
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    
    # Set up TensorBoard
    log_dir = os.path.join(output_dir, 'logs')
    summary_writer = tf.summary.create_file_writer(log_dir)
    
    # Create data loader
    data_loader = RawTFRecordLoader(config)
    
    # Get all TFRecord files in the data directory
    tfrecord_files = sorted(glob.glob(os.path.join(data_dir, '*.tfrecord*')))
    print(f"Found {len(tfrecord_files)} TFRecord files in {data_dir}")
    
    if not tfrecord_files:
        raise ValueError(f"No TFRecord files found in {data_dir}")
    
    # Split files into training and validation sets
    train_val_split_ratio = config['data'].get('train_val_split_ratio', 0.8)
    split_idx = int(len(tfrecord_files) * train_val_split_ratio)
    
    train_files = tfrecord_files[:split_idx]
    val_files = tfrecord_files[split_idx:]
    
    print(f"Split data: {len(train_files)} files for training, {len(val_files)} files for validation")
    
    # Create training and validation datasets
    train_dataset = data_loader.create_dataset(
        train_files,  # Pass the files directly instead of a glob pattern
        is_training=True,
        batch_size=config['training']['batch_size'],
        shuffle=True
    )
    
    val_dataset = data_loader.create_dataset(
        val_files,  # Pass the files directly instead of a glob pattern
        is_training=False,
        batch_size=config['training']['batch_size'],
        shuffle=False
    )
    
    # Update steps per epoch for learning rate schedule
    train_size = sum(1 for _ in train_dataset)
    steps_per_epoch = train_size
    print(f"Training dataset size: {train_size} batches")
    
    # Update learning rate schedule if using one
    if config['training'].get('lr_scheduler') == 'cosine':
        config['steps_per_epoch'] = steps_per_epoch
    
    # Build model
    model = build_model(config)
    
    # Create optimizer
    optimizer = create_optimizer(config)
    
    # Create checkpoint manager
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint, 
        checkpoint_dir, 
        max_to_keep=5
    )
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(config['training']['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['training']['num_epochs']}")
        start_time = time.time()
        
        # Training
        train_losses = {}
        
        for batch in tqdm.tqdm(train_dataset, desc="Training"):
            inputs, targets = batch
            outputs, loss_dict = train_step(model, optimizer, inputs, targets, config)
            
            # Accumulate losses
            for k, v in loss_dict.items():
                if k in train_losses:
                    train_losses[k].append(v.numpy())
                else:
                    train_losses[k] = [v.numpy()]
        
        # Average training losses
        avg_train_losses = {k: np.mean(v) for k, v in train_losses.items()}
        train_loss_str = ", ".join([f"{k}: {v:.4f}" for k, v in avg_train_losses.items()])
        print(f"Training losses: {train_loss_str}")
        
        # Log training metrics
        with summary_writer.as_default():
            for k, v in avg_train_losses.items():
                tf.summary.scalar(f"train/{k}", v, step=epoch)
            # Get current learning rate - this is compatible with newer TF versions
            try:
                # First attempt - most common in newer versions
                if hasattr(optimizer.learning_rate, 'numpy'):
                    current_lr = optimizer.learning_rate.numpy()
                # Second attempt - for LearningRateSchedule objects
                elif hasattr(optimizer.learning_rate, '__call__'):
                    current_lr = float(optimizer.learning_rate(optimizer.iterations))
                # Fallback
                else:
                    current_lr = float(optimizer.learning_rate)
            except:
                # If all else fails, just use a default value
                current_lr = 0.001
            
            tf.summary.scalar("train/learning_rate", current_lr, step=epoch)
        
        # Validation if needed
        if (epoch + 1) % config['training']['val_freq'] == 0:
            print("Validating...")
            val_metrics, val_predictions, val_targets, val_scenario_types = evaluate_model(
                model, 
                val_dataset, 
                config
            )
            
            val_metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in val_metrics.items() 
                                        if k in ['total_loss', 'ade', 'fde']])
            print(f"Validation metrics: {val_metrics_str}")
            
            # Log validation metrics
            with summary_writer.as_default():
                for k, v in val_metrics.items():
                    tf.summary.scalar(f"val/{k}", v, step=epoch)
            
            # Save visualizations
            if (epoch + 1) % 5 == 0:
                # Visualize scenario scores
                scenario_fig = visualize_scenario_scores(
                    {k: v for k, v in val_metrics.items() if k in val_metrics and k not in ['total_loss', 'ade', 'fde']},
                    save_path=os.path.join(output_dir, f"scenario_scores_epoch_{epoch+1}.png")
                )
                
                # Log visualization
                with summary_writer.as_default():
                    tf.summary.image("scenario_scores", plot_to_tensor(scenario_fig)[None], step=epoch)
                
                # Visualize sample trajectories
                num_samples = min(5, len(val_predictions['pred_waypoints']))
                
                for i in range(num_samples):
                    # Get sample data
                    batch_idx = i // config['training']['batch_size']
                    idx_in_batch = i % config['training']['batch_size']
                    
                    # Get speed
                    past_states = val_dataset.take(batch_idx + 1).as_numpy_iterator().next()[0]['past_states'][idx_in_batch]
                    initial_speed = np.sqrt(past_states[-1, 4]**2 + past_states[-1, 5]**2)
                    
                    # Create visualization
                    traj_fig = plot_trajectory(
                        past_states=past_states,
                        future_states=val_targets['future_waypoints'][i],
                        pred_waypoints=val_predictions['pred_waypoints'][i],
                        title=f"Sample Trajectory {i+1} (Epoch {epoch+1})",
                        show_trust_region=True,
                        initial_speed=initial_speed,
                        save_path=os.path.join(output_dir, f"trajectory_{i+1}_epoch_{epoch+1}.png")
                    )
                    
                    # Log visualization
                    with summary_writer.as_default():
                        tf.summary.image(f"trajectory_{i+1}", plot_to_tensor(traj_fig)[None], step=epoch)
            
            # Save checkpoint if better than previous best
            if val_metrics['total_loss'] < best_val_loss:
                best_val_loss = val_metrics['total_loss']
                # Save TensorFlow native checkpoint
                checkpoint_path = checkpoint_manager.save()
                
                # Also save the model in H5 format to the main checkpoints directory
                main_checkpoint_dir = os.path.join(current_dir, config['paths']['checkpoint_dir'])
                h5_path = os.path.join(main_checkpoint_dir, 'model_best.h5')
                model.save(h5_path)
                print(f"Saved checkpoint to {checkpoint_path} and H5 model to {h5_path} (best validation loss: {best_val_loss:.4f})")
        
        # Save regular checkpoint
        if (epoch + 1) % config['training']['save_freq'] == 0:
            checkpoint_path = checkpoint_manager.save()
            print(f"Saved checkpoint to {checkpoint_path}")
        
        # Print epoch time
        time_per_epoch = time.time() - start_time
        print(f"Time taken for epoch: {time_per_epoch:.2f}s")
    
    print("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train E2E Driving Model for Waymo Challenge")
    parser.add_argument("--config", type=str, default="/Users/basilshaji/Projects/wod-challenges/training/configs/model_config.yaml",
                        help="Path to config file")
    args = parser.parse_args()
    
    main(args)
