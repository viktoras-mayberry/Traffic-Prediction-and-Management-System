"""
Traffic Predictor
Main prediction service for traffic conditions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import yaml

from ..models.lstm_model import LSTMModel
from ..models.cnn_lstm_hybrid import CNNLSTMHybrid
from ..data_processing.feature_engineering import FeatureEngineer
from ..utils.logger import get_logger
from ..utils.metrics import calculate_metrics

logger = get_logger(__name__)


class TrafficPredictor:
    """
    Main traffic prediction service.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        model_type: str = "lstm",
        config_path: Optional[str] = None
    ):
        """
        Initialize traffic predictor.
        
        Args:
            model_path: Path to trained model file
            model_type: Type of model (lstm, cnn_lstm, ensemble)
            config_path: Path to configuration file
        """
        self.model_type = model_type.lower()
        self.config_path = config_path
        self.config = self._load_config()
        self.model = None
        self.feature_engineer = FeatureEngineer()
        
        if model_path:
            self.load_model(model_path)
    
    def _load_config(self) -> Dict:
        """Load configuration."""
        default_config = {
            'prediction': {
                'horizon_minutes': [15, 30, 60],
                'update_interval': 60,
                'cache_ttl': 300
            }
        }
        
        if self.config_path and Path(self.config_path).exists():
            try:
                with open(self.config_path, 'r') as f:
                    file_config = yaml.safe_load(f)
                    default_config.update(file_config)
            except Exception as e:
                logger.warning(f"Could not load config: {e}")
        
        return default_config
    
    def load_model(self, model_path: str):
        """
        Load trained model.
        
        Args:
            model_path: Path to model file
        """
        try:
            if self.model_type == "lstm":
                self.model = LSTMModel()
                self.model.load_model(model_path)
            elif self.model_type == "cnn_lstm":
                self.model = CNNLSTMHybrid()
                self.model.load_model(model_path)
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
            
            logger.info(f"Loaded {self.model_type} model from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def predict(
        self,
        data: pd.DataFrame,
        horizon_minutes: Optional[int] = None
    ) -> Dict[str, any]:
        """
        Predict traffic conditions.
        
        Args:
            data: Input traffic data
            horizon_minutes: Prediction horizon in minutes
        
        Returns:
            Dictionary with predictions
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        if horizon_minutes is None:
            horizon_minutes = self.config.get('prediction', {}).get('horizon_minutes', [15])[0]
        
        try:
            # Feature engineering
            data = self.feature_engineer.extract_temporal_features(data)
            data = self.feature_engineer.extract_spatial_features(data)
            data = self.feature_engineer.extract_traffic_features(data)
            
            # Prepare input
            feature_cols = [col for col in data.columns 
                           if col not in ['timestamp', 'is_outlier']]
            X = data[feature_cols].values
            
            # Normalize
            X = self.feature_engineer.normalize_features(
                pd.DataFrame(X, columns=feature_cols),
                columns=feature_cols,
                fit=False
            ).values
            
            # Make prediction
            predictions = self.model.predict(X)
            
            # Format output
            result = {
                'timestamp': datetime.now().isoformat(),
                'horizon_minutes': horizon_minutes,
                'predictions': {
                    'speed': float(predictions[0]) if len(predictions.shape) == 1 else float(predictions[0][0]),
                    'confidence': 0.85  # Placeholder
                },
                'model_type': self.model_type
            }
            
            logger.info(f"Generated prediction for {horizon_minutes} minute horizon")
            return result
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            raise
    
    def predict_multiple_horizons(
        self,
        data: pd.DataFrame,
        horizons: List[int] = None
    ) -> Dict[str, any]:
        """
        Predict for multiple time horizons.
        
        Args:
            data: Input traffic data
            horizons: List of prediction horizons in minutes
        
        Returns:
            Dictionary with predictions for each horizon
        """
        if horizons is None:
            horizons = self.config.get('prediction', {}).get('horizon_minutes', [15, 30, 60])
        
        results = {}
        for horizon in horizons:
            results[f"{horizon}_minutes"] = self.predict(data, horizon_minutes=horizon)
        
        return results
    
    def predict_travel_time(
        self,
        origin: Tuple[float, float],
        destination: Tuple[float, float],
        current_traffic: pd.DataFrame
    ) -> Dict[str, any]:
        """
        Predict travel time between two points.
        
        Args:
            origin: (latitude, longitude) of origin
            destination: (latitude, longitude) of destination
            current_traffic: Current traffic conditions
        
        Returns:
            Dictionary with travel time prediction
        """
        try:
            # Predict traffic conditions along route
            predictions = self.predict(current_traffic)
            
            # Calculate distance (simplified - would use actual routing)
            from ..data_ingestion.gps_processor import GPSProcessor
            gps_processor = GPSProcessor()
            
            distance = gps_processor._haversine_distance(
                pd.Series([origin[0]]),
                pd.Series([origin[1]]),
                destination[0],
                destination[1]
            ).iloc[0]
            
            # Estimate travel time
            predicted_speed = predictions['predictions']['speed']
            travel_time_minutes = (distance / predicted_speed * 60) if predicted_speed > 0 else 0
            
            result = {
                'origin': origin,
                'destination': destination,
                'distance_km': float(distance),
                'predicted_speed_kmh': float(predicted_speed),
                'travel_time_minutes': float(travel_time_minutes),
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Predicted travel time: {travel_time_minutes:.2f} minutes")
            return result
            
        except Exception as e:
            logger.error(f"Error predicting travel time: {e}")
            raise

