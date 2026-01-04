"""
Model Training Pipeline
Unified training interface for all models
"""

import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from datetime import datetime
from sklearn.model_selection import train_test_split

from .lstm_model import LSTMModel
from .cnn_lstm_hybrid import CNNLSTMHybrid
from .graph_neural_network import GNNTrafficPredictor
from ..utils.logger import get_logger
from ..utils.metrics import calculate_metrics, MetricsCalculator
from ..data_processing.feature_engineering import FeatureEngineer

logger = get_logger(__name__)


class ModelTrainer:
    """
    Unified trainer for all traffic prediction models.
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        model_type: str = "lstm"
    ):
        """
        Initialize model trainer.
        
        Args:
            config_path: Path to model_config.yaml
            model_type: Type of model to train (lstm, cnn_lstm, gnn, ensemble)
        """
        self.config_path = config_path
        self.model_type = model_type.lower()
        self.config = self._load_config()
        self.models = {}
        self.feature_engineer = FeatureEngineer(
            normalization_method=self.config.get('preprocessing', {}).get('normalization', 'min_max')
        )
        self.metrics_calculator = MetricsCalculator()
    
    def _load_config(self) -> Dict:
        """Load model configuration."""
        default_config = {
            'lstm': {
                'sequence_length': 60,
                'features': 10,
                'hidden_units': [128, 64],
                'dropout': 0.2,
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 100
            },
            'preprocessing': {
                'normalization': 'min_max'
            }
        }
        
        if self.config_path and Path(self.config_path).exists():
            try:
                with open(self.config_path, 'r') as f:
                    file_config = yaml.safe_load(f)
                    default_config.update(file_config)
            except Exception as e:
                logger.warning(f"Could not load model config: {e}")
        
        return default_config
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        target_column: str = "speed",
        sequence_length: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare data for training.
        
        Args:
            df: Input DataFrame
            target_column: Target column name
            sequence_length: Sequence length for time series models
        
        Returns:
            Tuple of (X, y, feature_names)
        """
        # Feature engineering
        df = self.feature_engineer.extract_temporal_features(df)
        df = self.feature_engineer.extract_spatial_features(df)
        df = self.feature_engineer.extract_traffic_features(df)
        
        # Select features
        feature_cols = [col for col in df.columns 
                       if col not in [target_column, 'timestamp', 'is_outlier']]
        
        # Normalize features
        df = self.feature_engineer.normalize_features(df, columns=feature_cols, fit=True)
        
        # Prepare target
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame")
        
        y = df[target_column].values
        X = df[feature_cols].values
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0)
        y = np.nan_to_num(y, nan=0.0)
        
        logger.info(f"Prepared data: X shape {X.shape}, y shape {y.shape}")
        return X, y, feature_cols
    
    def train_lstm(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_split: float = 0.2
    ) -> LSTMModel:
        """
        Train LSTM model.
        
        Args:
            X: Input features
            y: Target values
            validation_split: Validation split ratio
        
        Returns:
            Trained LSTM model
        """
        config = self.config.get('lstm', {})
        
        model = LSTMModel(
            sequence_length=config.get('sequence_length', 60),
            features=X.shape[1] if len(X.shape) > 1 else 1,
            hidden_units=config.get('hidden_units', [128, 64]),
            dropout=config.get('dropout', 0.2),
            learning_rate=config.get('learning_rate', 0.001)
        )
        
        model.build_model(output_dim=1)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, shuffle=False
        )
        
        # Train
        model.train(
            X_train, y_train,
            X_val=X_val, y_val=y_val,
            epochs=config.get('epochs', 100),
            batch_size=config.get('batch_size', 32),
            verbose=1
        )
        
        self.models['lstm'] = model
        logger.info("LSTM model training completed")
        return model
    
    def train_cnn_lstm(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_split: float = 0.2
    ) -> CNNLSTMHybrid:
        """
        Train CNN-LSTM hybrid model.
        
        Args:
            X: Input features
            y: Target values
            validation_split: Validation split ratio
        
        Returns:
            Trained CNN-LSTM model
        """
        config = self.config.get('cnn_lstm', {})
        
        # Reshape X for CNN-LSTM (needs spatial and temporal dimensions)
        # Assuming X is (time_steps, features), reshape to (time_steps, spatial_features, temporal_features)
        sequence_length = config.get('sequence_length', 60)
        spatial_features = config.get('spatial_features', 5)
        temporal_features = config.get('temporal_features', 10)
        
        # Reshape data
        if len(X.shape) == 2:
            # Reshape to (samples, sequence_length, spatial_features, temporal_features)
            # This is a simplified reshaping - actual implementation would need proper structure
            n_samples = len(X) // sequence_length
            X_reshaped = X[:n_samples * sequence_length].reshape(
                n_samples, sequence_length, spatial_features, temporal_features
            )
            y_reshaped = y[:n_samples * sequence_length:sequence_length]
        else:
            X_reshaped = X
            y_reshaped = y
        
        model = CNNLSTMHybrid(
            sequence_length=sequence_length,
            spatial_features=spatial_features,
            temporal_features=temporal_features,
            cnn_filters=config.get('cnn_filters', [64, 32]),
            cnn_kernel_size=config.get('cnn_kernel_size', 3),
            cnn_pool_size=config.get('cnn_pool_size', 2),
            lstm_units=config.get('lstm_units', [128, 64]),
            lstm_dropout=config.get('lstm_dropout', 0.3),
            lstm_recurrent_dropout=config.get('lstm_recurrent_dropout', 0.2),
            learning_rate=config.get('learning_rate', 0.0005)
        )
        
        model.build_model(output_dim=1)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_reshaped, y_reshaped, test_size=validation_split, shuffle=False
        )
        
        # Train
        model.train(
            X_train, y_train,
            X_val=X_val, y_val=y_val,
            epochs=config.get('epochs', 150),
            batch_size=config.get('batch_size', 32),
            verbose=1
        )
        
        self.models['cnn_lstm'] = model
        logger.info("CNN-LSTM model training completed")
        return model
    
    def train_ensemble(
        self,
        X: np.ndarray,
        y: np.ndarray,
        models: List[str] = None,
        weights: Optional[Dict[str, float]] = None
    ):
        """
        Train ensemble of models.
        
        Args:
            X: Input features
            y: Target values
            models: List of model types to train
            weights: Weights for ensemble voting
        """
        if models is None:
            models = ['lstm', 'cnn_lstm']
        
        if weights is None:
            weights = {model: 1.0 / len(models) for model in models}
        
        # Train each model
        for model_type in models:
            if model_type == 'lstm':
                self.train_lstm(X, y)
            elif model_type == 'cnn_lstm':
                self.train_cnn_lstm(X, y)
        
        self.models['ensemble'] = {
            'models': models,
            'weights': weights
        }
        
        logger.info(f"Ensemble training completed with models: {models}")
    
    def evaluate_model(
        self,
        model_name: str,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate a trained model.
        
        Args:
            model_name: Name of model to evaluate
            X: Test features
            y: True targets
        
        Returns:
            Evaluation metrics
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        model = self.models[model_name]
        
        if model_name == 'ensemble':
            # Ensemble prediction
            predictions = []
            for sub_model_name in model['models']:
                if sub_model_name in self.models:
                    pred = self.models[sub_model_name].predict(X)
                    predictions.append(pred * model['weights'].get(sub_model_name, 1.0))
            
            predictions = np.mean(predictions, axis=0)
        else:
            predictions = model.predict(X)
        
        # Calculate metrics
        if len(predictions.shape) > 1 and predictions.shape[1] == 1:
            predictions = predictions.flatten()
        if len(y.shape) > 1 and y.shape[1] == 1:
            y = y.flatten()
        
        metrics = calculate_metrics(y, predictions)
        logger.info(f"Model '{model_name}' evaluation: {metrics}")
        return metrics
    
    def save_models(self, base_path: str = "data/models"):
        """
        Save all trained models.
        
        Args:
            base_path: Base path for saving models
        """
        base_path = Path(base_path)
        base_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for model_name, model in self.models.items():
            if model_name == 'ensemble':
                continue
            
            filepath = base_path / f"{model_name}_{timestamp}.h5"
            model.save_model(str(filepath))
        
        logger.info(f"Saved all models to {base_path}")

