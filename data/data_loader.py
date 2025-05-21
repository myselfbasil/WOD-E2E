"""
Data loader for Waymo Open Dataset Challenge.
Handles reading and processing of TFRecord files.
"""
import os
import glob
import tensorflow as tf
import numpy as np
from typing import Dict, List, Tuple, Optional, Union


class WaymoDataLoader:
    """Data loader for Waymo Open Dataset Challenge."""
    
    def __init__(self, config: Dict):
        """
        Initialize data loader with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.image_height = config['data']['image_height']
        self.image_width = config['data']['image_width']
        self.num_cameras = config['data']['num_cameras']
        self.history_seconds = config['data']['history_seconds']
        self.future_seconds = config['data']['future_seconds']
        self.fps = config['data']['fps']
        self.normalize_poses = config['data']['normalize_poses']
        
        # Calculate sequence lengths based on fps
        self.history_steps = self.history_seconds * self.fps
        self.future_steps = self.future_seconds * self.fps
        
    def _parse_tfrecord(self, example) -> Dict:
        """
        Parse TFRecord example to extract features.
        
        Args:
            example: TFRecord example
            
        Returns:
            Dictionary of parsed features
        """
        # Define feature description for parsing
        feature_description = {
            'cameras/image': tf.io.FixedLenFeature([], tf.string),
            'cameras/intrinsics': tf.io.FixedLenFeature([9 * self.num_cameras], tf.float32),
            'cameras/extrinsics': tf.io.FixedLenFeature([16 * self.num_cameras], tf.float32),
            'state/past/x': tf.io.FixedLenFeature([self.history_steps], tf.float32),
            'state/past/y': tf.io.FixedLenFeature([self.history_steps], tf.float32),
            'state/past/z': tf.io.FixedLenFeature([self.history_steps], tf.float32),
            'state/past/heading': tf.io.FixedLenFeature([self.history_steps], tf.float32),
            'state/past/velocity_x': tf.io.FixedLenFeature([self.history_steps], tf.float32),
            'state/past/velocity_y': tf.io.FixedLenFeature([self.history_steps], tf.float32),
            'routing/waypoints/x': tf.io.VarLenFeature(tf.float32),
            'routing/waypoints/y': tf.io.VarLenFeature(tf.float32),
            'scenario_type': tf.io.FixedLenFeature([], tf.int64),
        }
        
        # For training data, include future states
        if not self.is_test:
            feature_description.update({
                'state/future/x': tf.io.FixedLenFeature([self.future_steps], tf.float32),
                'state/future/y': tf.io.FixedLenFeature([self.future_steps], tf.float32),
                'state/future/z': tf.io.FixedLenFeature([self.future_steps], tf.float32),
                'state/future/heading': tf.io.FixedLenFeature([self.future_steps], tf.float32),
                'state/future/velocity_x': tf.io.FixedLenFeature([self.future_steps], tf.float32),
                'state/future/velocity_y': tf.io.FixedLenFeature([self.future_steps], tf.float32),
            })
        
        # Parse example
        parsed_features = tf.io.parse_single_example(example, feature_description)
        
        # Process images
        images = tf.io.decode_jpeg(parsed_features['cameras/image'])
        images = tf.reshape(images, [self.num_cameras, self.image_height, self.image_width, 3])
        images = tf.cast(images, tf.float32) / 255.0
        
        # Process camera intrinsics and extrinsics
        intrinsics = tf.reshape(parsed_features['cameras/intrinsics'], [self.num_cameras, 3, 3])
        extrinsics = tf.reshape(parsed_features['cameras/extrinsics'], [self.num_cameras, 4, 4])
        
        # Process agent state history
        past_states = tf.stack([
            parsed_features['state/past/x'],
            parsed_features['state/past/y'],
            parsed_features['state/past/z'],
            parsed_features['state/past/heading'],
            parsed_features['state/past/velocity_x'],
            parsed_features['state/past/velocity_y']
        ], axis=1)
        
        # Process routing information
        route_x = tf.sparse.to_dense(parsed_features['routing/waypoints/x'])
        route_y = tf.sparse.to_dense(parsed_features['routing/waypoints/y'])
        route = tf.stack([route_x, route_y], axis=1)
        
        # Get scenario type
        scenario_type = parsed_features['scenario_type']
        
        # Create features dictionary
        features = {
            'images': images,
            'intrinsics': intrinsics,
            'extrinsics': extrinsics,
            'past_states': past_states,
            'route': route,
            'scenario_type': scenario_type
        }
        
        # For training data, include future states
        if not self.is_test:
            future_states = tf.stack([
                parsed_features['state/future/x'],
                parsed_features['state/future/y'],
                parsed_features['state/future/z'],
                parsed_features['state/future/heading'],
                parsed_features['state/future/velocity_x'],
                parsed_features['state/future/velocity_y']
            ], axis=1)
            features['future_states'] = future_states
            
        return features
    
    def _preprocess(self, features: Dict) -> Tuple[Dict, Optional[Dict]]:
        """
        Preprocess features for model input.
        
        Args:
            features: Dictionary of parsed features
            
        Returns:
            Tuple of (inputs, targets)
        """
        # Extract features
        images = features['images']
        past_states = features['past_states']
        route = features['route']
        scenario_type = features['scenario_type']
        
        # Preprocess images - already normalized in _parse_tfrecord
        
        # Normalize past states if configured
        if self.normalize_poses:
            # Use the last historical state as reference for normalization
            ref_x = past_states[-1, 0]
            ref_y = past_states[-1, 1]
            ref_heading = past_states[-1, 3]
            
            # Create rotation matrix for heading normalization
            cos_h = tf.cos(-ref_heading)
            sin_h = tf.sin(-ref_heading)
            rot_matrix = tf.stack([
                tf.stack([cos_h, -sin_h], axis=0),
                tf.stack([sin_h, cos_h], axis=0)
            ], axis=0)
            
            # Normalize past positions (x, y)
            past_xy = past_states[:, 0:2] - tf.stack([ref_x, ref_y], axis=0)
            past_xy = tf.matmul(past_xy, rot_matrix)
            past_states = tf.concat([
                past_xy,
                past_states[:, 2:3],  # z remains unchanged
                past_states[:, 3:4] - ref_heading,  # normalize heading
                tf.matmul(past_states[:, 4:6], rot_matrix)  # rotate velocities
            ], axis=1)
            
            # Normalize route
            route_xy = route - tf.stack([ref_x, ref_y], axis=0)
            route = tf.matmul(route_xy, rot_matrix)
        
        # Create model inputs
        inputs = {
            'images': images,
            'past_states': past_states,
            'route': route,
            'scenario_type': scenario_type
        }
        
        # Create targets for training
        targets = None
        if 'future_states' in features:
            future_states = features['future_states']
            
            # Normalize future states if configured
            if self.normalize_poses:
                # Use same reference as for past states
                ref_x = past_states[-1, 0]
                ref_y = past_states[-1, 1]
                ref_heading = past_states[-1, 3]
                
                # Create rotation matrix
                cos_h = tf.cos(-ref_heading)
                sin_h = tf.sin(-ref_heading)
                rot_matrix = tf.stack([
                    tf.stack([cos_h, -sin_h], axis=0),
                    tf.stack([sin_h, cos_h], axis=0)
                ], axis=0)
                
                # Normalize future positions (x, y)
                future_xy = future_states[:, 0:2] - tf.stack([ref_x, ref_y], axis=0)
                future_xy = tf.matmul(future_xy, rot_matrix)
                future_states = tf.concat([
                    future_xy,
                    future_states[:, 2:3],  # z remains unchanged
                    future_states[:, 3:4] - ref_heading,  # normalize heading
                    tf.matmul(future_states[:, 4:6], rot_matrix)  # rotate velocities
                ], axis=1)
            
            targets = {
                'future_states': future_states,
                # Extract just x,y coordinates for waypoint prediction
                'future_waypoints': future_states[:, 0:2]
            }
        
        return inputs, targets
    
    def create_dataset(
        self, 
        tfrecord_files_or_pattern: Union[str, List[str]], 
        is_training: bool = True, 
        is_test: bool = False,
        batch_size: int = 16, 
        shuffle: bool = True,
        cache: bool = False
    ) -> tf.data.Dataset:
        """
        Create TensorFlow dataset from TFRecord files.
        
        Args:
            tfrecord_files_or_pattern: Either a glob pattern for TFRecord files or a list of TFRecord file paths
            is_training: Whether this is for training
            is_test: Whether this is for testing (no ground truth)
            batch_size: Batch size
            shuffle: Whether to shuffle examples
            cache: Whether to cache the dataset
            
        Returns:
            TensorFlow dataset
        """
        self.is_test = is_test
        
        # Get TFRecord files
        if isinstance(tfrecord_files_or_pattern, str):
            # It's a glob pattern
            tfrecord_files = sorted(glob.glob(tfrecord_files_or_pattern))
            if not tfrecord_files:
                raise ValueError(f"No TFRecord files found with pattern: {tfrecord_files_or_pattern}")
        else:
            # It's already a list of files
            tfrecord_files = tfrecord_files_or_pattern
            if not tfrecord_files:
                raise ValueError("Empty list of TFRecord files provided")
        
        # Create dataset from TFRecord files
        dataset = tf.data.TFRecordDataset(tfrecord_files)
        
        # Parse and preprocess examples
        dataset = dataset.map(
            self._parse_tfrecord,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Preprocess features
        dataset = dataset.map(
            self._preprocess,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Cache dataset if requested
        if cache:
            dataset = dataset.cache()
        
        # Shuffle dataset if requested
        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000)
        
        # Batch dataset
        dataset = dataset.batch(batch_size)
        
        # Prefetch for performance
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
