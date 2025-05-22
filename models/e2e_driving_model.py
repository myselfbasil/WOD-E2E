"""
End-to-End Driving Model for Waymo Open Dataset Challenge.
This model architecture aims to predict future waypoints based on camera inputs,
historical poses, and routing information.
"""
import tensorflow as tf
from tensorflow.keras import layers, Model
import tensorflow_hub as hub
from typing import Dict, List, Tuple, Optional, Union


class EnhancedAttentionModule(layers.Layer):
    """Enhanced self-attention module with feed-forward network for better feature extraction."""
    
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1, ff_dim: int = None):
        """
        Initialize enhanced attention module.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
            ff_dim: Feed-forward dimension (if None, use 4x embed_dim)
        """
        super().__init__()
        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            dropout=dropout
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)
        
        # Feed-forward network for better feature transformation
        ff_dim = ff_dim or 4 * embed_dim
        self.ff = tf.keras.Sequential([
            layers.Dense(ff_dim, activation='gelu'),
            layers.Dropout(dropout),
            layers.Dense(embed_dim)
        ])
    
    def call(self, x: tf.Tensor, training: bool = False, attention_mask: Optional[tf.Tensor] = None) -> tf.Tensor:
        """
        Apply enhanced self-attention to input tensor.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, embed_dim]
            training: Whether in training mode
            attention_mask: Optional mask for attention
            
        Returns:
            Output tensor of same shape as input
        """
        # Multi-head attention with residual connection and normalization
        attn_output = self.mha(x, x, x, attention_mask=attention_mask, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        
        # Feed-forward network with residual connection and normalization
        ff_output = self.ff(out1, training=training)
        ff_output = self.dropout2(ff_output, training=training)
        out2 = self.layernorm2(out1 + ff_output)
        
        return out2


class AttentionModule(EnhancedAttentionModule):
    """Self-attention module for feature enhancement (backward compatibility)."""
    pass


class EnhancedCrossAttentionModule(layers.Layer):
    """Enhanced cross-attention module with feed-forward network for better feature fusion."""
    
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1, ff_dim: int = None):
        """
        Initialize enhanced cross-attention module.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
            ff_dim: Feed-forward dimension (if None, use 4x embed_dim)
        """
        super().__init__()
        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            dropout=dropout
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)
        
        # Feed-forward network for better feature transformation
        ff_dim = ff_dim or 4 * embed_dim
        self.ff = tf.keras.Sequential([
            layers.Dense(ff_dim, activation='gelu'),
            layers.Dropout(dropout),
            layers.Dense(embed_dim)
        ])
    
    def call(self, x: tf.Tensor, context: tf.Tensor, training: bool = False, attention_mask: Optional[tf.Tensor] = None) -> tf.Tensor:
        """
        Apply enhanced cross-attention to input tensor using context.
        
        Args:
            x: Query tensor of shape [batch_size, seq_len, embed_dim]
            context: Key/value tensor of shape [batch_size, context_len, embed_dim]
            training: Whether in training mode
            attention_mask: Optional mask for attention
            
        Returns:
            Output tensor of same shape as x
        """
        # Multi-head attention with residual connection and normalization
        attn_output = self.mha(x, context, context, attention_mask=attention_mask, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        
        # Feed-forward network with residual connection and normalization
        ff_output = self.ff(out1, training=training)
        ff_output = self.dropout2(ff_output, training=training)
        out2 = self.layernorm2(out1 + ff_output)
        
        return out2


class CrossAttentionModule(EnhancedCrossAttentionModule):
    """Cross-attention module for feature fusion (backward compatibility)."""
    pass


class ImageEncoder(layers.Layer):
    """Encodes images using pretrained backbone."""
    
    def __init__(self, config: Dict):
        """
        Initialize image encoder.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config
        self.output_dim = config['model']['image_embedding_dim']
        
        # Load pretrained backbone
        backbone_name = config['model']['image_backbone']
        if backbone_name == 'efficientnet_b3':
            base_model = tf.keras.applications.EfficientNetB3(
                include_top=False,
                weights='imagenet' if config['model']['use_pretrained'] else None,
                input_shape=(None, None, 3)
            )
        elif backbone_name == 'resnet50':
            base_model = tf.keras.applications.ResNet50(
                include_top=False,
                weights='imagenet' if config['model']['use_pretrained'] else None,
                input_shape=(None, None, 3)
            )
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        self.backbone = base_model
        self.pooling = layers.GlobalAveragePooling2D()
        self.projection = layers.Dense(self.output_dim)
        
        # Use attention if specified
        self.use_attention = config['model']['use_attention']
        if self.use_attention:
            self.attention = AttentionModule(
                embed_dim=self.output_dim,
                num_heads=4,
                dropout=config['model']['dropout_rate']
            )
    
    def call(self, images: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Encode images to embeddings.
        
        Args:
            images: Input images of shape [batch_size, num_cameras, height, width, channels]
            training: Whether in training mode
            
        Returns:
            Image embeddings of shape [batch_size, num_cameras, embed_dim]
        """
        batch_size = tf.shape(images)[0]
        num_cameras = tf.shape(images)[1]
        height = tf.shape(images)[2]
        width = tf.shape(images)[3]
        
        # Reshape to process all images
        images_flat = tf.reshape(images, [-1, height, width, 3])
        
        # Apply backbone
        features = self.backbone(images_flat, training=training)
        
        # Apply pooling
        embeddings = self.pooling(features)
        
        # Project to embedding dimension
        embeddings = self.projection(embeddings)
        
        # Reshape back to [batch_size, num_cameras, embed_dim]
        embeddings = tf.reshape(embeddings, [batch_size, num_cameras, self.output_dim])
        
        # Apply self-attention if specified
        if self.use_attention:
            embeddings = self.attention(embeddings, training=training)
        
        return embeddings


class PoseEncoder(layers.Layer):
    """Encodes agent pose history."""
    
    def __init__(self, config: Dict):
        """
        Initialize pose encoder.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config
        self.output_dim = config['model']['pose_embedding_dim']
        
        # MLP for encoding poses
        self.mlp = tf.keras.Sequential([
            layers.Dense(128, activation='relu'),
            layers.Dropout(config['model']['dropout_rate']),
            layers.Dense(256, activation='relu'),
            layers.Dropout(config['model']['dropout_rate']),
            layers.Dense(self.output_dim, activation=None)
        ])
        
        # Use attention if specified
        self.use_attention = config['model']['use_attention']
        if self.use_attention:
            self.attention = AttentionModule(
                embed_dim=self.output_dim,
                num_heads=4,
                dropout=config['model']['dropout_rate']
            )
    
    def call(self, poses: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Encode pose history to embeddings.
        
        Args:
            poses: Pose history of shape [batch_size, seq_len, pose_dim]
            training: Whether in training mode
            
        Returns:
            Pose embeddings of shape [batch_size, seq_len, embed_dim]
        """
        # Apply MLP to each timestep
        batch_size = tf.shape(poses)[0]
        seq_len = tf.shape(poses)[1]
        poses_flat = tf.reshape(poses, [-1, tf.shape(poses)[-1]])
        
        embeddings = self.mlp(poses_flat, training=training)
        embeddings = tf.reshape(embeddings, [batch_size, seq_len, self.output_dim])
        
        # Apply self-attention if specified
        if self.use_attention:
            embeddings = self.attention(embeddings, training=training)
        
        return embeddings


class RouteEncoder(layers.Layer):
    """Encodes routing information."""
    
    def __init__(self, config: Dict):
        """
        Initialize route encoder.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config
        self.output_dim = config['model']['route_embedding_dim']
        
        # MLP for encoding route
        self.mlp = tf.keras.Sequential([
            layers.Dense(64, activation='relu'),
            layers.Dropout(config['model']['dropout_rate']),
            layers.Dense(self.output_dim, activation=None)
        ])
        
        # Use attention if specified
        self.use_attention = config['model']['use_attention']
        if self.use_attention:
            self.attention = AttentionModule(
                embed_dim=self.output_dim,
                num_heads=2,
                dropout=config['model']['dropout_rate']
            )
    
    def call(self, route: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Encode route waypoints to embeddings.
        
        Args:
            route: Route waypoints of shape [batch_size, num_waypoints, 2]
            training: Whether in training mode
            
        Returns:
            Route embeddings of shape [batch_size, num_waypoints, embed_dim]
        """
        # Apply MLP to each waypoint
        batch_size = tf.shape(route)[0]
        num_waypoints = tf.shape(route)[1]
        route_flat = tf.reshape(route, [-1, tf.shape(route)[-1]])
        
        embeddings = self.mlp(route_flat, training=training)
        embeddings = tf.reshape(embeddings, [batch_size, num_waypoints, self.output_dim])
        
        # Apply self-attention if specified
        if self.use_attention:
            embeddings = self.attention(embeddings, training=training)
        
        return embeddings


class MultimodalFusion(layers.Layer):
    """Fuses multimodal embeddings using attention."""
    
    def __init__(self, config: Dict):
        """
        Initialize multimodal fusion module.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config
        self.fusion_dim = config['model']['fusion_dim']
        
        # Project all embeddings to common fusion space
        self.image_proj = layers.Dense(self.fusion_dim)
        self.pose_proj = layers.Dense(self.fusion_dim)
        self.route_proj = layers.Dense(self.fusion_dim)
        
        # Cross-attention modules
        self.img_pose_attn = CrossAttentionModule(
            embed_dim=self.fusion_dim,
            num_heads=4,
            dropout=config['model']['dropout_rate']
        )
        self.pose_route_attn = CrossAttentionModule(
            embed_dim=self.fusion_dim,
            num_heads=4,
            dropout=config['model']['dropout_rate']
        )
        
        # Final fusion layer
        self.fusion_layer = layers.Dense(self.fusion_dim)
    
    def call(
        self, 
        image_embed: tf.Tensor, 
        pose_embed: tf.Tensor, 
        route_embed: tf.Tensor,
        training: bool = False
    ) -> tf.Tensor:
        """
        Fuse multimodal embeddings.
        
        Args:
            image_embed: Image embeddings [batch_size, num_cameras, img_dim]
            pose_embed: Pose embeddings [batch_size, seq_len, pose_dim]
            route_embed: Route embeddings [batch_size, num_waypoints, route_dim]
            training: Whether in training mode
            
        Returns:
            Fused embeddings [batch_size, seq_len, fusion_dim]
        """
        # Project to common fusion space
        img_proj = self.image_proj(image_embed)  # [batch_size, num_cameras, fusion_dim]
        pose_proj = self.pose_proj(pose_embed)   # [batch_size, seq_len, fusion_dim]
        route_proj = self.route_proj(route_embed)  # [batch_size, num_waypoints, fusion_dim]
        
        # Cross-attention: pose features attend to image features
        pose_img = self.img_pose_attn(pose_proj, img_proj, training=training)
        
        # Cross-attention: pose features attend to route features
        pose_route = self.pose_route_attn(pose_proj, route_proj, training=training)
        
        # Combine all features
        fused = pose_proj + pose_img + pose_route
        fused = self.fusion_layer(fused)
        
        return fused


class ScenarioClassifier(layers.Layer):
    """Classifies driving scenario type."""
    
    def __init__(self, config: Dict):
        """
        Initialize scenario classifier.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config
        self.num_classes = config['model']['num_scenario_classes']
        
        # Classification layers
        self.global_pooling = layers.GlobalAveragePooling1D()
        self.mlp = tf.keras.Sequential([
            layers.Dense(256, activation='relu'),
            layers.Dropout(config['model']['dropout_rate']),
            layers.Dense(self.num_classes, activation=None)
        ])
    
    def call(self, features: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Classify scenario type from fused features.
        
        Args:
            features: Fused features [batch_size, seq_len, fusion_dim]
            training: Whether in training mode
            
        Returns:
            Scenario logits [batch_size, num_classes]
        """
        # Global pooling over sequence dimension
        pooled = self.global_pooling(features)
        
        # Apply classifier
        logits = self.mlp(pooled, training=training)
        
        return logits


class TemporalEncoder(layers.Layer):
    """Encodes temporal dynamics using LSTM."""
    
    def __init__(self, config: Dict):
        """
        Initialize temporal encoder.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config
        self.hidden_dim = config['model']['lstm_hidden_dim']
        self.num_layers = config['model']['lstm_num_layers']
        
        # Stack of LSTM layers
        self.lstm_layers = [
            layers.LSTM(
                self.hidden_dim, 
                return_sequences=True,
                dropout=config['model']['dropout_rate'] if i < self.num_layers - 1 else 0
            )
            for i in range(self.num_layers)
        ]
    
    def call(self, features: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Encode temporal dynamics.
        
        Args:
            features: Input features [batch_size, seq_len, feature_dim]
            training: Whether in training mode
            
        Returns:
            Temporal encodings [batch_size, seq_len, hidden_dim]
        """
        x = features
        for lstm in self.lstm_layers:
            x = lstm(x, training=training)
        
        return x


class WaypointDecoder(layers.Layer):
    """Decodes waypoints from temporal encodings."""
    
    def __init__(self, config: Dict):
        """
        Initialize waypoint decoder.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config
        self.pred_horizon = config['model']['prediction_horizon']
        self.pred_dim = config['model']['prediction_dim']
        
        # Waypoint decoder
        self.decoder = tf.keras.Sequential([
            layers.Dense(256, activation='relu'),
            layers.Dropout(config['model']['dropout_rate']),
            layers.Dense(self.pred_horizon * self.pred_dim)
        ])
    
    def call(self, features: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Decode waypoints from features.
        
        Args:
            features: Input features [batch_size, seq_len, feature_dim]
            training: Whether in training mode
            
        Returns:
            Predicted waypoints [batch_size, pred_horizon, pred_dim]
        """
        # Use last timestep features for prediction
        last_features = features[:, -1]
        
        # Decode waypoints
        pred_flat = self.decoder(last_features, training=training)
        pred = tf.reshape(pred_flat, [-1, self.pred_horizon, self.pred_dim])
        
        return pred


class E2EDrivingModel(Model):
    """End-to-End Driving Model for Waymo Open Dataset Challenge."""
    
    def __init__(self, config: Dict):
        """
        Initialize model with configuration.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config
        
        # Feature encoders
        self.image_encoder = ImageEncoder(config)
        self.pose_encoder = PoseEncoder(config)
        self.route_encoder = RouteEncoder(config)
        
        # Multimodal fusion
        self.fusion = MultimodalFusion(config)
        
        # Scenario classifier if enabled
        self.use_scenario_classifier = config['model']['scenario_classification']
        if self.use_scenario_classifier:
            self.scenario_classifier = ScenarioClassifier(config)
        
        # Temporal encoder
        self.temporal_encoder = TemporalEncoder(config)
        
        # Waypoint decoder
        self.waypoint_decoder = WaypointDecoder(config)
    
    def call(self, inputs: Dict, training: bool = False) -> Dict:
        """
        Forward pass of the model.
        
        Args:
            inputs: Dictionary of model inputs
            training: Whether in training mode
            
        Returns:
            Dictionary of model outputs
        """
        # Extract inputs
        images = inputs['images']  # [batch_size, num_cameras, height, width, channels]
        past_states = inputs['past_states']  # [batch_size, seq_len, state_dim]
        route = inputs['route']  # [batch_size, num_waypoints, 2]
        
        # Encode features
        image_embeddings = self.image_encoder(images, training=training)
        pose_embeddings = self.pose_encoder(past_states, training=training)
        route_embeddings = self.route_encoder(route, training=training)
        
        # Fuse multimodal embeddings
        fused_features = self.fusion(
            image_embeddings, 
            pose_embeddings, 
            route_embeddings,
            training=training
        )
        
        # Classify scenario if enabled
        scenario_logits = None
        if self.use_scenario_classifier:
            scenario_logits = self.scenario_classifier(fused_features, training=training)
        
        # Encode temporal dynamics
        temporal_features = self.temporal_encoder(fused_features, training=training)
        
        # Decode waypoints
        pred_waypoints = self.waypoint_decoder(temporal_features, training=training)
        
        # Create outputs
        outputs = {
            'pred_waypoints': pred_waypoints
        }
        
        if scenario_logits is not None:
            outputs['scenario_logits'] = scenario_logits
        
        return outputs
    
    def get_config(self):
        """Get model configuration."""
        return self.config


def build_model(config: Dict) -> E2EDrivingModel:
    """
    Build the End-to-End Driving Model.
    
    Args:
        config: Model configuration
        
    Returns:
        Instantiated model
    """
    model = E2EDrivingModel(config)
    
    # Dummy forward pass to build the model
    batch_size = 2
    num_cameras = config['data']['num_cameras']
    image_height = config['data']['image_height']
    image_width = config['data']['image_width']
    history_steps = config['data']['history_seconds'] * config['data']['fps']
    
    dummy_inputs = {
        'images': tf.zeros([batch_size, num_cameras, image_height, image_width, 3]),
        'past_states': tf.zeros([batch_size, history_steps, 6]),
        'route': tf.zeros([batch_size, 10, 2])  # Assuming 10 route waypoints
    }
    
    _ = model(dummy_inputs, training=False)
    
    return model
