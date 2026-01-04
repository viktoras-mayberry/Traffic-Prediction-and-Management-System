"""
Feature Engineering
Extracts temporal, spatial, and traffic features from raw data
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime
from ..utils.logger import get_logger

logger = get_logger(__name__)


class FeatureEngineer:
    """
    Engine for creating features from raw traffic data.
    """
    
    def __init__(self, normalization_method: str = "min_max"):
        """
        Initialize feature engineer.
        
        Args:
            normalization_method: Method for normalization (min_max, standard, robust)
        """
        self.normalization_method = normalization_method
        self.feature_stats = {}  # Store statistics for normalization
    
    def extract_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract temporal features from timestamp.
        
        Args:
            df: DataFrame with timestamp column
        
        Returns:
            DataFrame with added temporal features
        """
        if df.empty or 'timestamp' not in df.columns:
            return df
        
        df = df.copy()
        timestamps = pd.to_datetime(df['timestamp'])
        
        # Basic temporal features
        df['hour'] = timestamps.dt.hour
        df['day_of_week'] = timestamps.dt.dayofweek
        df['day_of_month'] = timestamps.dt.day
        df['month'] = timestamps.dt.month
        df['quarter'] = timestamps.dt.quarter
        df['year'] = timestamps.dt.year
        
        # Cyclical encoding for periodic features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Binary features
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_weekday'] = (df['day_of_week'] < 5).astype(int)
        df['is_morning'] = ((df['hour'] >= 6) & (df['hour'] < 12)).astype(int)
        df['is_afternoon'] = ((df['hour'] >= 12) & (df['hour'] < 18)).astype(int)
        df['is_evening'] = ((df['hour'] >= 18) & (df['hour'] < 22)).astype(int)
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] < 6)).astype(int)
        
        # Rush hour indicators
        df['is_rush_hour_morning'] = ((df['hour'] >= 7) & (df['hour'] < 10)).astype(int)
        df['is_rush_hour_evening'] = ((df['hour'] >= 17) & (df['hour'] < 20)).astype(int)
        df['is_rush_hour'] = (df['is_rush_hour_morning'] | df['is_rush_hour_evening']).astype(int)
        
        logger.info("Extracted temporal features")
        return df
    
    def extract_spatial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract spatial features from geographic coordinates.
        
        Args:
            df: DataFrame with latitude and longitude columns
        
        Returns:
            DataFrame with added spatial features
        """
        if df.empty or 'latitude' not in df.columns or 'longitude' not in df.columns:
            return df
        
        df = df.copy()
        
        # Basic spatial features (if not already present)
        if 'road_type' not in df.columns:
            # Placeholder - would need actual road network data
            df['road_type'] = 'unknown'
        
        if 'num_lanes' not in df.columns:
            df['num_lanes'] = 2  # Default
        
        if 'speed_limit' not in df.columns:
            df['speed_limit'] = 50  # Default km/h
        
        # Distance from city center (example: NYC)
        # This would be customized based on the city
        city_center_lat = 40.7128
        city_center_lon = -74.0060
        
        df['distance_from_center'] = self._haversine_distance(
            df['latitude'],
            df['longitude'],
            city_center_lat,
            city_center_lon
        )
        
        logger.info("Extracted spatial features")
        return df
    
    def extract_traffic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract traffic-related features.
        
        Args:
            df: DataFrame with traffic data
        
        Returns:
            DataFrame with added traffic features
        """
        if df.empty:
            return df
        
        df = df.copy()
        
        # Calculate flow rate if vehicle count and time interval available
        if 'vehicle_count' in df.columns and 'timestamp' in df.columns:
            df = df.sort_values('timestamp')
            time_diff = df['timestamp'].diff().dt.total_seconds()
            df['flow_rate'] = df['vehicle_count'] / (time_diff / 3600 + 1e-6)  # vehicles per hour
            df['flow_rate'] = df['flow_rate'].fillna(0)
        
        # Calculate density if speed and flow rate available
        if 'speed' in df.columns and 'flow_rate' in df.columns:
            # Density = Flow / Speed (vehicles per km)
            df['density'] = df['flow_rate'] / (df['speed'] + 1e-6)
            df['density'] = df['density'].fillna(0)
        
        # Congestion indicators
        if 'speed' in df.columns and 'speed_limit' in df.columns:
            df['speed_ratio'] = df['speed'] / (df['speed_limit'] + 1e-6)
            df['is_congested'] = (df['speed_ratio'] < 0.5).astype(int)
            df['congestion_level'] = pd.cut(
                df['speed_ratio'],
                bins=[0, 0.3, 0.5, 0.7, 1.0, np.inf],
                labels=['severe', 'high', 'medium', 'low', 'none']
            )
        
        # Occupancy-based features
        if 'occupancy' in df.columns:
            df['occupancy_normalized'] = df['occupancy'] / 100.0
            df['is_high_occupancy'] = (df['occupancy'] > 80).astype(int)
        
        # Rolling statistics
        numeric_cols = ['speed', 'vehicle_count', 'flow_rate', 'occupancy']
        for col in numeric_cols:
            if col in df.columns:
                df[f'{col}_rolling_mean_5'] = df[col].rolling(window=5, min_periods=1).mean()
                df[f'{col}_rolling_std_5'] = df[col].rolling(window=5, min_periods=1).std()
                df[f'{col}_rolling_mean_10'] = df[col].rolling(window=10, min_periods=1).mean()
        
        logger.info("Extracted traffic features")
        return df
    
    def normalize_features(
        self,
        df: pd.DataFrame,
        columns: List[str] = None,
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Normalize features using specified method.
        
        Args:
            df: DataFrame to normalize
            columns: List of columns to normalize (if None, normalizes all numeric columns)
            fit: Whether to fit normalization parameters (True) or use existing (False)
        
        Returns:
            DataFrame with normalized features
        """
        if df.empty:
            return df
        
        df = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
            # Exclude temporal features that are already encoded
            exclude = ['hour', 'day_of_week', 'month', 'year', 'day_of_month', 'quarter']
            columns = [col for col in columns if col not in exclude]
        
        for col in columns:
            if col not in df.columns:
                continue
            
            if fit:
                if self.normalization_method == "min_max":
                    min_val = df[col].min()
                    max_val = df[col].max()
                    self.feature_stats[col] = {"min": min_val, "max": max_val}
                    
                    if max_val > min_val:
                        df[col] = (df[col] - min_val) / (max_val - min_val)
                    else:
                        df[col] = 0.0
                
                elif self.normalization_method == "standard":
                    mean_val = df[col].mean()
                    std_val = df[col].std()
                    self.feature_stats[col] = {"mean": mean_val, "std": std_val}
                    
                    if std_val > 0:
                        df[col] = (df[col] - mean_val) / std_val
                    else:
                        df[col] = 0.0
                
                elif self.normalization_method == "robust":
                    median_val = df[col].median()
                    iqr = df[col].quantile(0.75) - df[col].quantile(0.25)
                    self.feature_stats[col] = {"median": median_val, "iqr": iqr}
                    
                    if iqr > 0:
                        df[col] = (df[col] - median_val) / iqr
                    else:
                        df[col] = 0.0
            else:
                # Use existing statistics
                if col in self.feature_stats:
                    stats = self.feature_stats[col]
                    
                    if self.normalization_method == "min_max":
                        min_val = stats["min"]
                        max_val = stats["max"]
                        if max_val > min_val:
                            df[col] = (df[col] - min_val) / (max_val - min_val)
                    
                    elif self.normalization_method == "standard":
                        mean_val = stats["mean"]
                        std_val = stats["std"]
                        if std_val > 0:
                            df[col] = (df[col] - mean_val) / std_val
                    
                    elif self.normalization_method == "robust":
                        median_val = stats["median"]
                        iqr = stats["iqr"]
                        if iqr > 0:
                            df[col] = (df[col] - median_val) / iqr
        
        logger.info(f"Normalized features using {self.normalization_method} method")
        return df
    
    def create_lag_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        lags: List[int] = [1, 2, 3, 5, 10]
    ) -> pd.DataFrame:
        """
        Create lag features for time series prediction.
        
        Args:
            df: DataFrame with time series data
            columns: Columns to create lags for
            lags: List of lag values (number of time steps)
        
        Returns:
            DataFrame with added lag features
        """
        if df.empty or 'timestamp' not in df.columns:
            return df
        
        df = df.copy()
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        for col in columns:
            if col not in df.columns:
                continue
            
            for lag in lags:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        logger.info(f"Created lag features for {len(columns)} columns")
        return df
    
    def _haversine_distance(
        self,
        lat1: pd.Series,
        lon1: pd.Series,
        lat2: float,
        lon2: float
    ) -> pd.Series:
        """
        Calculate Haversine distance between points.
        
        Args:
            lat1: Latitude of first points
            lon1: Longitude of first points
            lat2: Latitude of second point
            lon2: Longitude of second point
        
        Returns:
            Series of distances in kilometers
        """
        R = 6371  # Earth radius in kilometers
        
        phi1 = np.radians(lat1)
        phi2 = np.radians(lat2)
        delta_phi = np.radians(lat2 - lat1)
        delta_lambda = np.radians(lon2 - lon1)
        
        a = (np.sin(delta_phi / 2) ** 2 +
             np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        
        return R * c

