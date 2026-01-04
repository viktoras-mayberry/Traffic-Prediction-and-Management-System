"""
Uber Movement API Integration
Downloads and processes anonymized travel time data from Uber Movement
"""

import os
import requests
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime, timedelta
import time
from ..utils.logger import get_logger

logger = get_logger(__name__)


class UberMovementClient:
    """
    Client for interacting with Uber Movement API and datasets.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        data_path: str = "data/raw/uber_movement",
        api_url: str = "https://movement.uber.com"
    ):
        """
        Initialize Uber Movement client.
        
        Args:
            api_key: API key for Uber Movement (if available)
            data_path: Local path to store downloaded data
            api_url: Base URL for Uber Movement API
        """
        self.api_key = api_key or os.getenv("UBER_MOVEMENT_API_KEY", "")
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.api_url = api_url
        self.session = requests.Session()
        
        if self.api_key:
            self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})
    
    def get_available_cities(self) -> List[Dict]:
        """
        Get list of available cities in Uber Movement dataset.
        
        Returns:
            List of city information dictionaries
        """
        try:
            # Note: Uber Movement API structure may vary
            # This is a placeholder for the actual API endpoint
            response = self.session.get(f"{self.api_url}/cities")
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Could not fetch cities: {response.status_code}")
                # Return default cities if API is not accessible
                return [
                    {"id": "nyc", "name": "New York City"},
                    {"id": "la", "name": "Los Angeles"},
                    {"id": "chicago", "name": "Chicago"}
                ]
        except Exception as e:
            logger.error(f"Error fetching cities: {e}")
            return []
    
    def download_travel_times(
        self,
        city_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Optional[pd.DataFrame]:
        """
        Download travel time data for a specific city.
        
        Args:
            city_id: City identifier (e.g., 'nyc', 'la')
            start_date: Start date for data download
            end_date: End date for data download
        
        Returns:
            DataFrame with travel time data or None if download fails
        """
        try:
            if start_date is None:
                start_date = datetime.now() - timedelta(days=30)
            if end_date is None:
                end_date = datetime.now()
            
            logger.info(f"Downloading travel times for {city_id} from {start_date} to {end_date}")
            
            # Construct API request
            params = {
                "city": city_id,
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d")
            }
            
            # Note: Actual Uber Movement API endpoints may differ
            # This is a template implementation
            response = self.session.get(
                f"{self.api_url}/travel-times",
                params=params,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                df = pd.DataFrame(data)
                
                # Save to local storage
                filename = f"{city_id}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
                filepath = self.data_path / filename
                df.to_csv(filepath, index=False)
                logger.info(f"Saved data to {filepath}")
                
                return df
            else:
                logger.warning(f"Failed to download data: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error downloading travel times: {e}")
            return None
    
    def load_local_data(self, city_id: str, days: int = 30) -> pd.DataFrame:
        """
        Load previously downloaded data from local storage.
        
        Args:
            city_id: City identifier
            days: Number of days of data to load
        
        Returns:
            DataFrame with travel time data
        """
        try:
            # Find all CSV files for this city
            pattern = f"{city_id}_*.csv"
            files = list(self.data_path.glob(pattern))
            
            if not files:
                logger.warning(f"No local data found for {city_id}")
                return pd.DataFrame()
            
            # Load and concatenate all files
            dfs = []
            for file in sorted(files)[-days:]:  # Get most recent files
                df = pd.read_csv(file)
                dfs.append(df)
            
            if dfs:
                combined_df = pd.concat(dfs, ignore_index=True)
                logger.info(f"Loaded {len(combined_df)} records for {city_id}")
                return combined_df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error loading local data: {e}")
            return pd.DataFrame()
    
    def process_travel_times(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process and clean travel time data.
        
        Args:
            df: Raw travel time DataFrame
        
        Returns:
            Processed DataFrame
        """
        if df.empty:
            return df
        
        try:
            # Standardize column names
            if 'datetime' in df.columns:
                df['timestamp'] = pd.to_datetime(df['datetime'])
            elif 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Ensure numeric columns
            numeric_cols = ['travel_time', 'distance', 'speed']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove invalid records
            df = df.dropna(subset=['timestamp'])
            
            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            logger.info(f"Processed {len(df)} travel time records")
            return df
            
        except Exception as e:
            logger.error(f"Error processing travel times: {e}")
            return df
    
    def get_latest_data(self, city_id: str) -> pd.DataFrame:
        """
        Get the latest available data for a city (download or load from cache).
        
        Args:
            city_id: City identifier
        
        Returns:
            DataFrame with latest travel time data
        """
        # Try to load from local storage first
        df = self.load_local_data(city_id, days=7)
        
        # If no local data or data is stale, try to download
        if df.empty or self._is_data_stale(df):
            logger.info(f"Local data is stale or missing, attempting download for {city_id}")
            new_df = self.download_travel_times(city_id)
            if new_df is not None and not new_df.empty:
                return self.process_travel_times(new_df)
        
        return self.process_travel_times(df)
    
    def _is_data_stale(self, df: pd.DataFrame, max_age_hours: int = 24) -> bool:
        """
        Check if data is stale.
        
        Args:
            df: DataFrame to check
            max_age_hours: Maximum age in hours before data is considered stale
        
        Returns:
            True if data is stale
        """
        if df.empty or 'timestamp' not in df.columns:
            return True
        
        latest_timestamp = pd.to_datetime(df['timestamp']).max()
        age = (datetime.now() - latest_timestamp).total_seconds() / 3600
        
        return age > max_age_hours

