"""
Tests for data ingestion modules
"""

import pytest
import pandas as pd
from datetime import datetime
from src.data_ingestion.uber_movement import UberMovementClient
from src.data_ingestion.city_apis import CityAPIClient
from src.data_ingestion.sensor_reader import SensorReader
from src.data_ingestion.gps_processor import GPSProcessor


def test_uber_movement_client():
    """Test Uber Movement client initialization."""
    client = UberMovementClient()
    assert client is not None
    assert client.data_path.exists()


def test_city_api_client():
    """Test City API client initialization."""
    client = CityAPIClient(city="nyc")
    assert client is not None
    assert client.city == "nyc"


def test_sensor_reader():
    """Test sensor reader."""
    reader = SensorReader()
    assert reader is not None
    
    # Test reading loop detector data
    start_time = datetime.now()
    end_time = datetime.now()
    data = reader.read_loop_detector(start_time=start_time, end_time=end_time)
    assert isinstance(data, pd.DataFrame)


def test_gps_processor():
    """Test GPS processor."""
    processor = GPSProcessor()
    assert processor is not None
    
    # Create sample GPS data
    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=10, freq='1min'),
        'latitude': [40.7128] * 10,
        'longitude': [-74.0060] * 10
    })
    
    # Test speed calculation
    df_with_speed = processor.calculate_speed(df)
    assert 'speed_kmh' in df_with_speed.columns

