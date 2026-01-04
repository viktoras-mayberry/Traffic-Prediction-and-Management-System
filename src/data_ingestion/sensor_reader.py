"""
Sensor Data Reader
Reads data from various traffic sensors (loop detectors, cameras, radar)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime, timedelta
import json
import time
from ..utils.logger import get_logger

logger = get_logger(__name__)


class SensorReader:
    """
    Reader for traffic sensor data from various sources.
    """
    
    def __init__(
        self,
        data_path: str = "data/raw/sensors",
        sensor_types: List[str] = None
    ):
        """
        Initialize sensor reader.
        
        Args:
            data_path: Path to sensor data files
            sensor_types: List of sensor types to read (loop_detector, camera, radar)
        """
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        if sensor_types is None:
            sensor_types = ["loop_detector", "camera", "radar"]
        self.sensor_types = sensor_types
    
    def read_loop_detector(
        self,
        sensor_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Read data from loop detector sensors.
        
        Args:
            sensor_id: Specific sensor ID, or None for all sensors
            start_time: Start time for data retrieval
            end_time: End time for data retrieval
        
        Returns:
            DataFrame with loop detector data
        """
        try:
            if start_time is None:
                start_time = datetime.now() - timedelta(hours=1)
            if end_time is None:
                end_time = datetime.now()
            
            # Look for sensor data files
            pattern = "loop_detector*.csv" if sensor_id is None else f"loop_detector_{sensor_id}*.csv"
            files = list(self.data_path.glob(pattern))
            
            if not files:
                logger.warning(f"No loop detector data found")
                return self._generate_sample_loop_detector_data(sensor_id, start_time, end_time)
            
            # Load and filter data
            dfs = []
            for file in files:
                df = pd.read_csv(file)
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)]
                dfs.append(df)
            
            if dfs:
                combined_df = pd.concat(dfs, ignore_index=True)
                logger.info(f"Loaded {len(combined_df)} loop detector records")
                return combined_df
            else:
                return self._generate_sample_loop_detector_data(sensor_id, start_time, end_time)
                
        except Exception as e:
            logger.error(f"Error reading loop detector data: {e}")
            return self._generate_sample_loop_detector_data(sensor_id, start_time, end_time)
    
    def read_camera_data(
        self,
        camera_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Read data from traffic cameras.
        
        Args:
            camera_id: Specific camera ID, or None for all cameras
            start_time: Start time for data retrieval
            end_time: End time for data retrieval
        
        Returns:
            DataFrame with camera data
        """
        try:
            if start_time is None:
                start_time = datetime.now() - timedelta(hours=1)
            if end_time is None:
                end_time = datetime.now()
            
            pattern = "camera*.csv" if camera_id is None else f"camera_{camera_id}*.csv"
            files = list(self.data_path.glob(pattern))
            
            if not files:
                logger.warning(f"No camera data found")
                return self._generate_sample_camera_data(camera_id, start_time, end_time)
            
            dfs = []
            for file in files:
                df = pd.read_csv(file)
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)]
                dfs.append(df)
            
            if dfs:
                combined_df = pd.concat(dfs, ignore_index=True)
                logger.info(f"Loaded {len(combined_df)} camera records")
                return combined_df
            else:
                return self._generate_sample_camera_data(camera_id, start_time, end_time)
                
        except Exception as e:
            logger.error(f"Error reading camera data: {e}")
            return self._generate_sample_camera_data(camera_id, start_time, end_time)
    
    def read_radar_data(
        self,
        radar_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Read data from radar sensors.
        
        Args:
            radar_id: Specific radar ID, or None for all radars
            start_time: Start time for data retrieval
            end_time: End time for data retrieval
        
        Returns:
            DataFrame with radar data
        """
        try:
            if start_time is None:
                start_time = datetime.now() - timedelta(hours=1)
            if end_time is None:
                end_time = datetime.now()
            
            pattern = "radar*.csv" if radar_id is None else f"radar_{radar_id}*.csv"
            files = list(self.data_path.glob(pattern))
            
            if not files:
                logger.warning(f"No radar data found")
                return self._generate_sample_radar_data(radar_id, start_time, end_time)
            
            dfs = []
            for file in files:
                df = pd.read_csv(file)
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)]
                dfs.append(df)
            
            if dfs:
                combined_df = pd.concat(dfs, ignore_index=True)
                logger.info(f"Loaded {len(combined_df)} radar records")
                return combined_df
            else:
                return self._generate_sample_radar_data(radar_id, start_time, end_time)
                
        except Exception as e:
            logger.error(f"Error reading radar data: {e}")
            return self._generate_sample_radar_data(radar_id, start_time, end_time)
    
    def read_all_sensors(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Read data from all sensor types.
        
        Args:
            start_time: Start time for data retrieval
            end_time: End time for data retrieval
        
        Returns:
            Dictionary mapping sensor types to DataFrames
        """
        results = {}
        
        if "loop_detector" in self.sensor_types:
            results["loop_detector"] = self.read_loop_detector(start_time=start_time, end_time=end_time)
        
        if "camera" in self.sensor_types:
            results["camera"] = self.read_camera_data(start_time=start_time, end_time=end_time)
        
        if "radar" in self.sensor_types:
            results["radar"] = self.read_radar_data(start_time=start_time, end_time=end_time)
        
        return results
    
    def _generate_sample_loop_detector_data(
        self,
        sensor_id: Optional[str],
        start_time: datetime,
        end_time: datetime
    ) -> pd.DataFrame:
        """Generate sample loop detector data for testing."""
        time_range = pd.date_range(start_time, end_time, freq='1min')
        n_samples = len(time_range)
        
        data = {
            'timestamp': time_range,
            'sensor_id': sensor_id or 'LD001',
            'vehicle_count': np.random.poisson(10, n_samples),
            'average_speed': np.random.normal(50, 10, n_samples).clip(0, 100),
            'occupancy': np.random.uniform(0, 100, n_samples),
            'lane': np.random.choice([1, 2, 3, 4], n_samples)
        }
        
        return pd.DataFrame(data)
    
    def _generate_sample_camera_data(
        self,
        camera_id: Optional[str],
        start_time: datetime,
        end_time: datetime
    ) -> pd.DataFrame:
        """Generate sample camera data for testing."""
        time_range = pd.date_range(start_time, end_time, freq='5min')
        n_samples = len(time_range)
        
        data = {
            'timestamp': time_range,
            'camera_id': camera_id or 'CAM001',
            'vehicle_count': np.random.poisson(15, n_samples),
            'congestion_level': np.random.choice(['low', 'medium', 'high'], n_samples),
            'latitude': np.random.uniform(40.7, 40.8, n_samples),
            'longitude': np.random.uniform(-74.0, -73.9, n_samples)
        }
        
        return pd.DataFrame(data)
    
    def _generate_sample_radar_data(
        self,
        radar_id: Optional[str],
        start_time: datetime,
        end_time: datetime
    ) -> pd.DataFrame:
        """Generate sample radar data for testing."""
        time_range = pd.date_range(start_time, end_time, freq='30s')
        n_samples = len(time_range)
        
        data = {
            'timestamp': time_range,
            'radar_id': radar_id or 'RAD001',
            'speed': np.random.normal(55, 12, n_samples).clip(0, 120),
            'direction': np.random.choice(['N', 'S', 'E', 'W'], n_samples),
            'vehicle_type': np.random.choice(['car', 'truck', 'motorcycle'], n_samples, p=[0.8, 0.15, 0.05])
        }
        
        return pd.DataFrame(data)

