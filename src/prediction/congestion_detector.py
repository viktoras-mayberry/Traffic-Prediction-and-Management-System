"""
Congestion Detector
Detects and predicts traffic congestion
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from ..utils.logger import get_logger

logger = get_logger(__name__)


class CongestionDetector:
    """
    Detects and predicts traffic congestion.
    """
    
    def __init__(
        self,
        speed_threshold: float = 0.5,
        occupancy_threshold: float = 80.0,
        density_threshold: float = 50.0
    ):
        """
        Initialize congestion detector.
        
        Args:
            speed_threshold: Speed ratio threshold (speed/speed_limit) for congestion
            occupancy_threshold: Occupancy percentage threshold
            density_threshold: Vehicle density threshold (vehicles/km)
        """
        self.speed_threshold = speed_threshold
        self.occupancy_threshold = occupancy_threshold
        self.density_threshold = density_threshold
    
    def detect_congestion(
        self,
        data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Detect congestion in current traffic data.
        
        Args:
            data: Traffic data DataFrame
        
        Returns:
            DataFrame with congestion indicators
        """
        df = data.copy()
        
        # Speed-based congestion
        if 'speed' in df.columns and 'speed_limit' in df.columns:
            df['speed_ratio'] = df['speed'] / (df['speed_limit'] + 1e-6)
            df['congested_by_speed'] = (df['speed_ratio'] < self.speed_threshold).astype(int)
        elif 'speed' in df.columns:
            # Assume default speed limit of 50 km/h
            df['speed_ratio'] = df['speed'] / 50.0
            df['congested_by_speed'] = (df['speed_ratio'] < self.speed_threshold).astype(int)
        
        # Occupancy-based congestion
        if 'occupancy' in df.columns:
            df['congested_by_occupancy'] = (df['occupancy'] > self.occupancy_threshold).astype(int)
        
        # Density-based congestion
        if 'density' in df.columns:
            df['congested_by_density'] = (df['density'] > self.density_threshold).astype(int)
        
        # Overall congestion indicator
        congestion_cols = [col for col in df.columns if col.startswith('congested_by_')]
        if congestion_cols:
            df['is_congested'] = df[congestion_cols].any(axis=1).astype(int)
        else:
            df['is_congested'] = 0
        
        # Congestion severity
        df['congestion_severity'] = self._calculate_severity(df)
        
        logger.info(f"Detected {df['is_congested'].sum()} congested locations")
        return df
    
    def _calculate_severity(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate congestion severity level.
        
        Args:
            df: DataFrame with congestion indicators
        
        Returns:
            Series with severity levels
        """
        severity = pd.Series(['none'] * len(df), index=df.index)
        
        if 'speed_ratio' in df.columns:
            severity[df['speed_ratio'] < 0.3] = 'severe'
            severity[(df['speed_ratio'] >= 0.3) & (df['speed_ratio'] < 0.5)] = 'high'
            severity[(df['speed_ratio'] >= 0.5) & (df['speed_ratio'] < 0.7)] = 'medium'
            severity[(df['speed_ratio'] >= 0.7) & (df['speed_ratio'] < 1.0)] = 'low'
        
        return severity
    
    def predict_congestion(
        self,
        current_data: pd.DataFrame,
        predicted_speeds: np.ndarray
    ) -> Dict[str, any]:
        """
        Predict future congestion based on predicted speeds.
        
        Args:
            current_data: Current traffic data
            predicted_speeds: Predicted speeds from model
        
        Returns:
            Dictionary with congestion predictions
        """
        try:
            predictions = []
            
            for i, speed in enumerate(predicted_speeds):
                speed_limit = current_data['speed_limit'].iloc[i] if 'speed_limit' in current_data.columns else 50.0
                speed_ratio = speed / speed_limit
                
                is_congested = speed_ratio < self.speed_threshold
                
                if speed_ratio < 0.3:
                    severity = 'severe'
                elif speed_ratio < 0.5:
                    severity = 'high'
                elif speed_ratio < 0.7:
                    severity = 'medium'
                elif speed_ratio < 1.0:
                    severity = 'low'
                else:
                    severity = 'none'
                
                predictions.append({
                    'location_id': current_data.index[i] if 'location_id' not in current_data.columns else current_data['location_id'].iloc[i],
                    'predicted_speed': float(speed),
                    'speed_limit': float(speed_limit),
                    'speed_ratio': float(speed_ratio),
                    'is_congested': bool(is_congested),
                    'severity': severity
                })
            
            result = {
                'timestamp': datetime.now().isoformat(),
                'predictions': predictions,
                'summary': {
                    'total_locations': len(predictions),
                    'congested_count': sum(1 for p in predictions if p['is_congested']),
                    'severe_count': sum(1 for p in predictions if p['severity'] == 'severe')
                }
            }
            
            logger.info(f"Predicted congestion for {len(predictions)} locations")
            return result
            
        except Exception as e:
            logger.error(f"Error predicting congestion: {e}")
            raise
    
    def get_congestion_hotspots(
        self,
        data: pd.DataFrame,
        top_n: int = 10
    ) -> List[Dict]:
        """
        Get top congestion hotspots.
        
        Args:
            data: Traffic data with congestion indicators
            top_n: Number of hotspots to return
        
        Returns:
            List of hotspot dictionaries
        """
        try:
            # Ensure congestion is detected
            if 'is_congested' not in data.columns:
                data = self.detect_congestion(data)
            
            # Filter congested locations
            congested = data[data['is_congested'] == 1].copy()
            
            if congested.empty:
                return []
            
            # Sort by severity and other factors
            if 'congestion_severity' in congested.columns:
                severity_order = {'severe': 4, 'high': 3, 'medium': 2, 'low': 1, 'none': 0}
                congested['severity_score'] = congested['congestion_severity'].map(severity_order)
                congested = congested.sort_values('severity_score', ascending=False)
            
            # Get top N
            top_hotspots = congested.head(top_n)
            
            hotspots = []
            for idx, row in top_hotspots.iterrows():
                hotspot = {
                    'location_id': row.get('location_id', idx),
                    'latitude': float(row.get('latitude', 0)),
                    'longitude': float(row.get('longitude', 0)),
                    'severity': row.get('congestion_severity', 'unknown'),
                    'speed': float(row.get('speed', 0)),
                    'speed_limit': float(row.get('speed_limit', 50)),
                    'occupancy': float(row.get('occupancy', 0))
                }
                hotspots.append(hotspot)
            
            logger.info(f"Identified {len(hotspots)} congestion hotspots")
            return hotspots
            
        except Exception as e:
            logger.error(f"Error getting congestion hotspots: {e}")
            return []

