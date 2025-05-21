"""
Data loader for Waymo End-to-End Driving Challenge.
Handles the specific Waymo proto format in TFRecord files.
"""
import os
import glob
import tensorflow as tf
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from waymo_open_dataset.protos import end_to_end_driving_data_pb2 as wod_e2ed_pb2


class WaymoE2EDataLoader:
    """Data loader for Waymo End-to-End Driving Challenge."""
    
    def __init__(self, config: Dict):
        """
        Initialize data loader with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.image_height = config['data'].get('image_height', 480)
        self.image_width = config['data'].get('image_width', 640)
        self.num_cameras = config['data'].get('num_cameras', 8)
        self.history_seconds = config['data'].get('history_seconds', 12)
        self.future_seconds = config['data'].get('future_seconds', 5)
        self.fps = config['data'].get('fps', 10)
        self.normalize_poses = config['data'].get('normalize_poses', True)
        
        # Default front camera order (front_left, front, front_right)
        self.camera_order = [2, 1, 3]
        
    def parse_tfrecord(self, serialized_example):
        """
        Parse TFRecord example to extract features using Waymo's protocol buffers.
        
        Args:
            serialized_example: Serialized example from TFRecord
            
        Returns:
            Parsed E2EDFrame and extracted features
        """
        # Parse using Waymo's proto format
        data = wod_e2ed_pb2.E2EDFrame()
        data.ParseFromString(serialized_example.numpy())
        
        # Extract features as needed for the model
        features = self._extract_features(data)
        
        return features
    
    def _extract_features(self, data: wod_e2ed_pb2.E2EDFrame):
        """
        Extract features from E2EDFrame.
        
        Args:
            data: Parsed E2EDFrame
            
        Returns:
            Dictionary of extracted features
        """
        # Extract camera images
        images = []
        camera_calibrations = []
        
        # First collect all cameras
        camera_dict = {}
        for index, image_content in enumerate(data.frame.images):
            camera_name = image_content.name
            calibration = data.frame.context.camera_calibrations[index]
            image = tf.io.decode_image(image_content.image)
            camera_dict[camera_name] = (image, calibration)
        
        # Then arrange them in the desired order
        for camera_name in self.camera_order:
            if camera_name in camera_dict:
                image, calibration = camera_dict[camera_name]
                images.append(image)
                camera_calibrations.append(calibration)
        
        # Stack images into a single tensor
        images = tf.stack(images, axis=0)
        
        # Extract past states
        past_x = np.array(data.past_states.pos_x)
        past_y = np.array(data.past_states.pos_y)
        past_z = np.array(data.past_states.pos_z)
        past_heading = np.array(data.past_states.heading)
        past_vel_x = np.array(data.past_states.vel_x)
        past_vel_y = np.array(data.past_states.vel_y)
        
        past_states = np.stack([past_x, past_y, past_z, past_heading, past_vel_x, past_vel_y], axis=1)
        
        # Extract routing information
        route_x = []
        route_y = []
        for point in data.route:
            route_x.append(point.pos_x)
            route_y.append(point.pos_y)
        
        if not route_x:  # If no routing info, provide dummy values
            route_x = [0.0]
            route_y = [0.0]
            
        route = np.stack([np.array(route_x), np.array(route_y)], axis=1)
        
        # Get scenario ID and type
        scenario_id = data.frame.context.name
        scenario_type = 0  # Default type, modify as needed
        
        # Create features dictionary
        features = {
            'images': images,
            'past_states': past_states,
            'route': route,
            'scenario_id': scenario_id,
            'scenario_type': scenario_type
        }
        
        # For training data, include future states if available
        if hasattr(data, 'future_states') and data.future_states.pos_x:
            future_x = np.array(data.future_states.pos_x)
            future_y = np.array(data.future_states.pos_y)
            future_z = np.array(data.future_states.pos_z)
            future_heading = np.array(data.future_states.heading)
            future_vel_x = np.array(data.future_states.vel_x)
            future_vel_y = np.array(data.future_states.vel_y)
            
            future_states = np.stack([future_x, future_y, future_z, 
                                    future_heading, future_vel_x, future_vel_y], axis=1)
            features['future_states'] = future_states
            features['future_waypoints'] = np.stack([future_x, future_y], axis=1)
            
        return features
    
    def _preprocess(self, features):
        """
        Preprocess features for model input.
        
        Args:
            features: Dictionary of extracted features
            
        Returns:
            Tuple of (inputs, targets)
        """
        # Extract features
        images = features['images']
        past_states = features['past_states']
        route = features['route']
        scenario_type = features['scenario_type']
        
        # Normalize images to [0, 1]
        images = tf.cast(images, tf.float32) / 255.0
        
        # Normalize past states if configured
        if self.normalize_poses:
            # Use the last historical state as reference for normalization
            ref_x = past_states[-1, 0]
            ref_y = past_states[-1, 1]
            ref_heading = past_states[-1, 3]
            
            # Create rotation matrix for heading normalization
            cos_h = np.cos(-ref_heading)
            sin_h = np.sin(-ref_heading)
            rot_matrix = np.array([
                [cos_h, -sin_h],
                [sin_h, cos_h]
            ])
            
            # Normalize past positions (x, y)
            past_xy = past_states[:, 0:2] - np.array([ref_x, ref_y])
            past_xy = np.matmul(past_xy, rot_matrix)
            past_states = np.concatenate([
                past_xy,
                past_states[:, 2:3],  # z remains unchanged
                past_states[:, 3:4] - ref_heading,  # normalize heading
                np.matmul(past_states[:, 4:6], rot_matrix)  # rotate velocities
            ], axis=1)
            
            # Normalize route
            if route.shape[0] > 0:
                route_xy = route - np.array([ref_x, ref_y])
                route = np.matmul(route_xy, rot_matrix)
        
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
            future_waypoints = features['future_waypoints']
            
            # Normalize future states if configured
            if self.normalize_poses:
                # Use same reference as for past states
                ref_x = past_states[-1, 0]
                ref_y = past_states[-1, 1]
                ref_heading = past_states[-1, 3]
                
                # Create rotation matrix
                cos_h = np.cos(-ref_heading)
                sin_h = np.sin(-ref_heading)
                rot_matrix = np.array([
                    [cos_h, -sin_h],
                    [sin_h, cos_h]
                ])
                
                # Normalize future positions (x, y)
                future_xy = future_waypoints - np.array([ref_x, ref_y])
                future_xy = np.matmul(future_xy, rot_matrix)
                future_waypoints = future_xy
                
                # Normalize future states
                future_xy = future_states[:, 0:2] - np.array([ref_x, ref_y])
                future_xy = np.matmul(future_xy, rot_matrix)
                future_states = np.concatenate([
                    future_xy,
                    future_states[:, 2:3],  # z remains unchanged
                    future_states[:, 3:4] - ref_heading,  # normalize heading
                    np.matmul(future_states[:, 4:6], rot_matrix)  # rotate velocities
                ], axis=1)
            
            targets = {
                'future_states': future_states,
                'future_waypoints': future_waypoints
            }
        
        return inputs, targets
    
    def parse_and_preprocess(self, serialized_example):
        """
        Parse and preprocess a single example.
        
        Args:
            serialized_example: Serialized example from TFRecord
            
        Returns:
            Tuple of (inputs, targets)
        """
        # Parse using tf.py_function to handle the proto parsing
        features = tf.py_function(
            self.parse_tfrecord,
            [serialized_example],
            Tout=[tf.float32, tf.float32, tf.float32, tf.string, tf.int64, 
                  tf.float32, tf.float32]
        )
        
        # Convert to dictionary
        feature_dict = {
            'images': features[0],
            'past_states': features[1],
            'route': features[2],
            'scenario_id': features[3],
            'scenario_type': features[4],
        }
        
        # Add future states if available (for training)
        if len(features) > 5:
            feature_dict['future_states'] = features[5]
            feature_dict['future_waypoints'] = features[6]
        
        # Preprocess
        inputs, targets = tf.py_function(
            self._preprocess,
            [feature_dict],
            Tout=[tf.float32, tf.float32, tf.float32, tf.int64, 
                  tf.float32, tf.float32]
        )
        
        # Convert to dictionary
        input_dict = {
            'images': inputs[0],
            'past_states': inputs[1],
            'route': inputs[2],
            'scenario_type': inputs[3]
        }
        
        target_dict = None
        if targets is not None:
            target_dict = {
                'future_states': targets[0],
                'future_waypoints': targets[1]
            }
        
        return input_dict, target_dict
    
    def create_dataset(
        self, 
        tfrecord_files: Union[str, List[str]], 
        is_training: bool = True, 
        is_test: bool = False,
        batch_size: int = 16, 
        shuffle: bool = True,
        cache: bool = False
    ) -> tf.data.Dataset:
        """
        Create TensorFlow dataset from TFRecord files.
        
        Args:
            tfrecord_files: Either a glob pattern for TFRecord files or a list of TFRecord file paths
            is_training: Whether this is for training
            is_test: Whether this is for testing (no ground truth)
            batch_size: Batch size
            shuffle: Whether to shuffle examples
            cache: Whether to cache the dataset
            
        Returns:
            TensorFlow dataset
        """
        # Get TFRecord files
        if isinstance(tfrecord_files, str):
            # It's a glob pattern
            tfrecord_files = sorted(glob.glob(tfrecord_files))
            if not tfrecord_files:
                raise ValueError(f"No TFRecord files found with pattern: {tfrecord_files}")
        else:
            # It's already a list of files
            if not tfrecord_files:
                raise ValueError("Empty list of TFRecord files provided")
        
        # Create dataset from TFRecord files
        dataset = tf.data.TFRecordDataset(tfrecord_files)
        
        # Parse and preprocess examples
        dataset = dataset.map(
            self.parse_and_preprocess,
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
