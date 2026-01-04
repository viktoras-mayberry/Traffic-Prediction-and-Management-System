"""
GPS Signal Processing
Processes GPS trajectory data for route analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timedelta
import json
from ..utils.logger import get_logger

logger = get_logger(__name__)


class GPSProcessor:
    """
    Processor for GPS trajectory and location data.
    """
    
    def __init__(self, data_path: str = "data/raw/gps"):
        """
        Initialize GPS processor.
        
        Args:
            data_path: Path to GPS data files
        """
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)
    
    def load_gpx_file(self, filepath: str) -> pd.DataFrame:
        """
        Load GPS data from GPX file format.
        
        Args:
            filepath: Path to GPX file
        
        Returns:
            DataFrame with GPS trajectory data
        """
        try:
            # Simple GPX parser (for production, use gpxpy library)
            import xml.etree.ElementTree as ET
            
            tree = ET.parse(filepath)
            root = tree.getroot()
            
            # GPX namespace
            ns = {'gpx': 'http://www.topografix.com/GPX/1/1'}
            
            points = []
            for trkpt in root.findall('.//gpx:trkpt', ns):
                lat = float(trkpt.get('lat'))
                lon = float(trkpt.get('lon'))
                
                ele_elem = trkpt.find('gpx:ele', ns)
                ele = float(ele_elem.text) if ele_elem is not None else None
                
                time_elem = trkpt.find('gpx:time', ns)
                timestamp = pd.to_datetime(time_elem.text) if time_elem is not None else datetime.now()
                
                points.append({
                    'timestamp': timestamp,
                    'latitude': lat,
                    'longitude': lon,
                    'elevation': ele
                })
            
            df = pd.DataFrame(points)
            logger.info(f"Loaded {len(df)} GPS points from {filepath}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading GPX file: {e}")
            return pd.DataFrame()
    
    def load_csv_trajectory(self, filepath: str) -> pd.DataFrame:
        """
        Load GPS trajectory from CSV file.
        
        Args:
            filepath: Path to CSV file
        
        Returns:
            DataFrame with GPS trajectory data
        """
        try:
            df = pd.read_csv(filepath)
            
            # Standardize column names
            column_mapping = {
                'lat': 'latitude',
                'lon': 'longitude',
                'lng': 'longitude',
                'time': 'timestamp',
                'datetime': 'timestamp',
                'date_time': 'timestamp'
            }
            
            df = df.rename(columns=column_mapping)
            
            # Ensure timestamp is datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            else:
                df['timestamp'] = datetime.now()
            
            # Ensure numeric columns
            for col in ['latitude', 'longitude']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            logger.info(f"Loaded {len(df)} GPS points from CSV")
            return df
            
        except Exception as e:
            logger.error(f"Error loading CSV trajectory: {e}")
            return pd.DataFrame()
    
    def calculate_speed(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate speed from GPS trajectory data.
        
        Args:
            df: DataFrame with GPS points (must have latitude, longitude, timestamp)
        
        Returns:
            DataFrame with added speed column
        """
        if df.empty or 'latitude' not in df.columns or 'longitude' not in df.columns:
            return df
        
        try:
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Calculate distance between consecutive points (Haversine formula)
            def haversine_distance(lat1, lon1, lat2, lon2):
                """Calculate distance between two GPS points in meters."""
                R = 6371000  # Earth radius in meters
                
                phi1 = np.radians(lat1)
                phi2 = np.radians(lat2)
                delta_phi = np.radians(lat2 - lat1)
                delta_lambda = np.radians(lon2 - lon1)
                
                a = (np.sin(delta_phi / 2) ** 2 +
                     np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2)
                c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
                
                return R * c
            
            # Calculate distances
            distances = []
            for i in range(len(df) - 1):
                dist = haversine_distance(
                    df.iloc[i]['latitude'],
                    df.iloc[i]['longitude'],
                    df.iloc[i + 1]['latitude'],
                    df.iloc[i + 1]['longitude']
                )
                distances.append(dist)
            distances.append(0)  # Last point has no next point
            
            df['distance_meters'] = distances
            
            # Calculate time differences
            time_diffs = df['timestamp'].diff().dt.total_seconds()
            time_diffs.iloc[0] = 0
            
            # Calculate speed (m/s to km/h)
            df['speed_kmh'] = (df['distance_meters'] / time_diffs * 3.6).fillna(0)
            df['speed_kmh'] = df['speed_kmh'].replace([np.inf, -np.inf], 0)
            
            logger.info("Calculated speed from GPS trajectory")
            return df
            
        except Exception as e:
            logger.error(f"Error calculating speed: {e}")
            return df
    
    def calculate_acceleration(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate acceleration from speed data.
        
        Args:
            df: DataFrame with speed_kmh column
        
        Returns:
            DataFrame with added acceleration column
        """
        if 'speed_kmh' not in df.columns:
            df = self.calculate_speed(df)
        
        try:
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Calculate speed change
            speed_diff = df['speed_kmh'].diff()
            
            # Calculate time difference
            time_diff = df['timestamp'].diff().dt.total_seconds()
            time_diff.iloc[0] = 1  # Avoid division by zero
            
            # Calculate acceleration (km/h per second)
            df['acceleration'] = (speed_diff / time_diff).fillna(0)
            df['acceleration'] = df['acceleration'].replace([np.inf, -np.inf], 0)
            
            logger.info("Calculated acceleration from GPS data")
            return df
            
        except Exception as e:
            logger.error(f"Error calculating acceleration: {e}")
            return df
    
    def filter_by_bounds(
        self,
        df: pd.DataFrame,
        min_lat: float,
        max_lat: float,
        min_lon: float,
        max_lon: float
    ) -> pd.DataFrame:
        """
        Filter GPS points by geographic bounds.
        
        Args:
            df: DataFrame with GPS points
            min_lat: Minimum latitude
            max_lat: Maximum latitude
            min_lon: Minimum longitude
            max_lon: Maximum longitude
        
        Returns:
            Filtered DataFrame
        """
        try:
            filtered = df[
                (df['latitude'] >= min_lat) &
                (df['latitude'] <= max_lat) &
                (df['longitude'] >= min_lon) &
                (df['longitude'] <= max_lon)
            ]
            logger.info(f"Filtered to {len(filtered)} points within bounds")
            return filtered
        except Exception as e:
            logger.error(f"Error filtering by bounds: {e}")
            return df
    
    def resample_trajectory(
        self,
        df: pd.DataFrame,
        freq: str = '1min'
    ) -> pd.DataFrame:
        """
        Resample trajectory to regular time intervals.
        
        Args:
            df: DataFrame with GPS trajectory
            freq: Resampling frequency (e.g., '1min', '30s')
        
        Returns:
            Resampled DataFrame
        """
        try:
            if df.empty or 'timestamp' not in df.columns:
                return df
            
            df = df.set_index('timestamp')
            df_resampled = df.resample(freq).mean()
            df_resampled = df_resampled.reset_index()
            
            logger.info(f"Resampled trajectory to {freq} frequency")
            return df_resampled
            
        except Exception as e:
            logger.error(f"Error resampling trajectory: {e}")
            return df
    
    def detect_stops(self, df: pd.DataFrame, speed_threshold: float = 5.0, min_duration: int = 60) -> pd.DataFrame:
        """
        Detect stops in GPS trajectory (where speed is below threshold for minimum duration).
        
        Args:
            df: DataFrame with GPS trajectory and speed
            speed_threshold: Speed threshold in km/h to consider as stopped
            min_duration: Minimum duration in seconds to be considered a stop
        
        Returns:
            DataFrame with added 'is_stopped' column
        """
        try:
            if 'speed_kmh' not in df.columns:
                df = self.calculate_speed(df)
            
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Identify stops
            df['is_stopped'] = df['speed_kmh'] < speed_threshold
            
            # Calculate stop duration
            stop_groups = (df['is_stopped'] != df['is_stopped'].shift()).cumsum()
            stop_durations = df.groupby(stop_groups).apply(
                lambda x: (x['timestamp'].max() - x['timestamp'].min()).total_seconds()
                if x['is_stopped'].iloc[0] else 0
            )
            
            df['stop_duration'] = df.groupby(stop_groups)['is_stopped'].transform(
                lambda x: (x.index.max() - x.index.min()) * 60 if x.iloc[0] else 0
            )
            
            # Mark as significant stop only if duration exceeds threshold
            df['is_significant_stop'] = (df['is_stopped']) & (df['stop_duration'] >= min_duration)
            
            logger.info(f"Detected {df['is_significant_stop'].sum()} significant stops")
            return df
            
        except Exception as e:
            logger.error(f"Error detecting stops: {e}")
            return df

