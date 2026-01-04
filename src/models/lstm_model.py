"""
LSTM Model for Traffic Prediction
Time-series traffic prediction using LSTM networks
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from typing import Tuple, Optional, Dict
from pathlib import Path
from ..utils.logger import get_logger
from ..utils.metrics import calculate_metrics

logger = get_logger(__name__)


class LSTMModel:
    """
    LSTM model for traffic prediction.
    """
    
    def __init__(
        self,
        sequence_length: int = 60,
        features: int = 10,
        hidden_units: list = [128, 64],
        dropout: float = 0.2,
        learning_rate: float = 0.001
    ):
        """
        Initialize LSTM model.
        
        Args:
            sequence_length: Number of time steps to look back
            features: Number of input features
            hidden_units: List of LSTM layer sizes
            dropout: Dropout rate
            learning_rate: Learning rate for optimizer
        """
        self.sequence_length = sequence_length
        self.features = features
        self.hidden_units = hidden_units
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.model = None
        self.history = None
    
    def build_model(self, output_dim: int = 1) -> keras.Model:
        """
        Build LSTM model architecture.
        
        Args:
            output_dim: Dimension of output (number of prediction targets)
        
        Returns:
            Compiled Keras model
        """
        model = Sequential()
        
        # Input layer
        model.add(Input(shape=(self.sequence_length, self.features)))
        
        # LSTM layers
        for i, units in enumerate(self.hidden_units):
            return_sequences = (i < len(self.hidden_units) - 1)
            model.add(LSTM(
                units=units,
                return_sequences=return_sequences,
                dropout=self.dropout,
                recurrent_dropout=self.dropout
            ))
        
        # Output layer
        model.add(Dense(output_dim, activation='linear'))
        
        # Compile model
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'rmse']
        )
        
        self.model = model
        logger.info(f"Built LSTM model with {model.count_params()} parameters")
        return model
    
    def prepare_sequences(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Prepare sequences for LSTM input.
        
        Args:
            X: Input features (time_steps, features)
            y: Target values (time_steps, targets) or None
        
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
        epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.2,
        callbacks: Optional[list] = None,
        verbose: int = 1
    ) -> keras.callbacks.History:
        """
        Train the LSTM model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            epochs: Number of training epochs
            batch_size: Batch size
            validation_split: Validation split ratio (if X_val not provided)
            callbacks: List of Keras callbacks
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
                    patience=10,
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
        
        logger.info("LSTM model training completed")
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
            raise ValueError("Model not built or loaded. Call build_model() or load_model() first.")
        
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
            Dictionary of evaluation metrics
        """
        predictions = self.predict(X)
        X_seq, y_seq = self.prepare_sequences(X, y)
        
        # Flatten if needed
        if len(predictions.shape) > 1 and predictions.shape[1] == 1:
            predictions = predictions.flatten()
        if len(y_seq.shape) > 1 and y_seq.shape[1] == 1:
            y_seq = y_seq.flatten()
        
        metrics = calculate_metrics(y_seq, predictions)
        logger.info(f"Model evaluation metrics: {metrics}")
        return metrics
    
    def save_model(self, filepath: str):
        """
        Save model to file.
        
        Args:
            filepath: Path to save model
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(filepath)
        logger.info(f"Saved model to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load model from file.
        
        Args:
            filepath: Path to model file
        """
        self.model = keras.models.load_model(filepath)
        logger.info(f"Loaded model from {filepath}")

