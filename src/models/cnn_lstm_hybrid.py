"""
CNN-LSTM Hybrid Model for Traffic Prediction
Spatial-temporal pattern recognition based on Agagu & Ajobiewe (2025) research
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, LSTM, Dense, Dropout,
    Flatten, Concatenate, Reshape
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from typing import Tuple, Optional, Dict
from pathlib import Path
from ..utils.logger import get_logger
from ..utils.metrics import calculate_metrics

logger = get_logger(__name__)


class CNNLSTMHybrid:
    """
    Hybrid CNN-LSTM model for spatial-temporal traffic prediction.
    Based on research by Agagu & Ajobiewe (2025) achieving 88.97% accuracy.
    """
    
    def __init__(
        self,
        sequence_length: int = 60,
        spatial_features: int = 5,
        temporal_features: int = 10,
        cnn_filters: list = [64, 32],
        cnn_kernel_size: int = 3,
        cnn_pool_size: int = 2,
        lstm_units: list = [128, 64],
        lstm_dropout: float = 0.3,
        lstm_recurrent_dropout: float = 0.2,
        learning_rate: float = 0.0005
    ):
        """
        Initialize CNN-LSTM hybrid model.
        
        Args:
            sequence_length: Number of time steps
            spatial_features: Number of spatial features (road segments)
            temporal_features: Number of temporal features
            cnn_filters: List of CNN filter sizes
            cnn_kernel_size: CNN kernel size
            cnn_pool_size: Max pooling size
            lstm_units: List of LSTM layer sizes
            lstm_dropout: LSTM dropout rate
            lstm_recurrent_dropout: LSTM recurrent dropout rate
            learning_rate: Learning rate
        """
        self.sequence_length = sequence_length
        self.spatial_features = spatial_features
        self.temporal_features = temporal_features
        self.cnn_filters = cnn_filters
        self.cnn_kernel_size = cnn_kernel_size
        self.cnn_pool_size = cnn_pool_size
        self.lstm_units = lstm_units
        self.lstm_dropout = lstm_dropout
        self.lstm_recurrent_dropout = lstm_recurrent_dropout
        self.learning_rate = learning_rate
        self.model = None
        self.history = None
    
    def build_model(self, output_dim: int = 1) -> keras.Model:
        """
        Build CNN-LSTM hybrid model architecture.
        
        Args:
            output_dim: Dimension of output
        
        Returns:
            Compiled Keras model
        """
        # Input shape: (sequence_length, spatial_features, temporal_features)
        input_shape = (self.sequence_length, self.spatial_features, self.temporal_features)
        inputs = Input(shape=input_shape)
        
        # Reshape for CNN: (sequence_length, spatial_features * temporal_features)
        reshaped = Reshape((self.sequence_length, self.spatial_features * self.temporal_features))(inputs)
        
        # CNN layers for spatial feature extraction
        x = reshaped
        for filters in self.cnn_filters:
            x = Conv1D(
                filters=filters,
                kernel_size=self.cnn_kernel_size,
                activation='relu',
                padding='same'
            )(x)
            x = MaxPooling1D(pool_size=self.cnn_pool_size)(x)
        
        # LSTM layers for temporal pattern recognition
        for i, units in enumerate(self.lstm_units):
            return_sequences = (i < len(self.lstm_units) - 1)
            x = LSTM(
                units=units,
                return_sequences=return_sequences,
                dropout=self.lstm_dropout,
                recurrent_dropout=self.lstm_recurrent_dropout
            )(x)
        
        # Output layer
        outputs = Dense(output_dim, activation='linear')(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'rmse']
        )
        
        self.model = model
        logger.info(f"Built CNN-LSTM hybrid model with {model.count_params()} parameters")
        return model
    
    def prepare_sequences(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Prepare sequences for CNN-LSTM input.
        
        Args:
            X: Input features (time_steps, spatial_features, temporal_features)
            y: Target values or None
        
        Returns:
            Tuple of (X_sequences, y_sequences)
        """
        X_sequences = []
        y_sequences = []
        
        for i in range(len(X) - self.sequence_length):
            X_sequences.append(X[i:i + self.sequence_length])
            if y is not None:
                y_sequences.append(y[i + self.sequence_length])
        
        X_sequences = np.array(X_sequences)
        if y is not None:
            y_sequences = np.array(y_sequences)
        else:
            y_sequences = None
        
        return X_sequences, y_sequences
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 150,
        batch_size: int = 32,
        validation_split: float = 0.2,
        callbacks: Optional[list] = None,
        verbose: int = 1
    ) -> keras.callbacks.History:
        """
        Train the CNN-LSTM model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            epochs: Number of epochs
            batch_size: Batch size
            validation_split: Validation split
            callbacks: Keras callbacks
            verbose: Verbosity level
        
        Returns:
            Training history
        """
        if self.model is None:
            output_dim = y_train.shape[-1] if len(y_train.shape) > 1 else 1
            self.build_model(output_dim=output_dim)
        
        # Prepare sequences
        X_seq, y_seq = self.prepare_sequences(X_train, y_train)
        
        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            X_val_seq, y_val_seq = self.prepare_sequences(X_val, y_val)
            validation_data = (X_val_seq, y_val_seq)
        
        # Default callbacks
        if callbacks is None:
            callbacks = [
                EarlyStopping(
                    monitor='val_loss' if validation_data else 'loss',
                    patience=15,
                    restore_best_weights=True
                )
            ]
        
        # Train model
        self.history = self.model.fit(
            X_seq,
            y_seq,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            validation_split=validation_split if validation_data is None else None,
            callbacks=callbacks,
            verbose=verbose
        )
        
        logger.info("CNN-LSTM hybrid model training completed")
        return self.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input features
        
        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not built or loaded.")
        
        X_seq, _ = self.prepare_sequences(X)
        predictions = self.model.predict(X_seq, verbose=0)
        
        return predictions
    
    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X: Input features
            y: True targets
        
        Returns:
            Evaluation metrics
        """
        predictions = self.predict(X)
        X_seq, y_seq = self.prepare_sequences(X, y)
        
        # Flatten if needed
        if len(predictions.shape) > 1 and predictions.shape[1] == 1:
            predictions = predictions.flatten()
        if len(y_seq.shape) > 1 and y_seq.shape[1] == 1:
            y_seq = y_seq.flatten()
        
        metrics = calculate_metrics(y_seq, predictions)
        logger.info(f"CNN-LSTM model evaluation metrics: {metrics}")
        return metrics
    
    def save_model(self, filepath: str):
        """Save model to file."""
        if self.model is None:
            raise ValueError("No model to save")
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(filepath)
        logger.info(f"Saved CNN-LSTM model to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model from file."""
        self.model = keras.models.load_model(filepath)
        logger.info(f"Loaded CNN-LSTM model from {filepath}")

