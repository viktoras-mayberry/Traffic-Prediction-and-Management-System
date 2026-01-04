"""
City Government API Integration
Integrates with open data portals from various cities
"""

import os
import requests
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime, timedelta
import json
from ..utils.logger import get_logger

logger = get_logger(__name__)


class CityAPIClient:
    """
    Client for interacting with city government open data APIs.
    """
    
    def __init__(
        self,
        city: str = "nyc",
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        data_path: str = "data/raw/city_apis"
    ):
        """
        Initialize city API client.
        
        Args:
            city: City identifier (nyc, la, chicago)
            api_key: API key for city API
            api_url: Base URL for city API
            data_path: Local path to store downloaded data
        """
        self.city = city.lower()
        self.data_path = Path(data_path) / self.city
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        # City-specific configuration
        city_configs = {
            "nyc": {
                "url": "https://data.cityofnewyork.us",
                "api_key_env": "NYC_API_KEY"
            },
            "la": {
                "url": "https://data.lacity.org",
                "api_key_env": "LA_API_KEY"
            },
            "chicago": {
                "url": "https://data.cityofchicago.org",
                "api_key_env": "CHICAGO_API_KEY"
            }
        }
        
        config = city_configs.get(self.city, city_configs["nyc"])
        self.api_url = api_url or config["url"]
        self.api_key = api_key or os.getenv(config["api_key_env"], "")
        
        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({"X-App-Token": self.api_key})
    
    def get_traffic_sensors(self, limit: int = 1000) -> pd.DataFrame:
        """
        Get traffic sensor data from city API.
        
        Args:
            limit: Maximum number of records to retrieve
        
        Returns:
            DataFrame with sensor data
        """
        try:
            # NYC example: Traffic Speed dataset
            if self.city == "nyc":
                dataset_id = "i4wp-t4c5"  # NYC Traffic Speed dataset
                endpoint = f"{self.api_url}/resource/{dataset_id}.json"
            elif self.city == "la":
                # LA traffic data endpoint
                endpoint = f"{self.api_url}/resource/traffic.json"
            elif self.city == "chicago":
                # Chicago traffic data endpoint
                endpoint = f"{self.api_url}/resource/traffic.json"
            else:
                logger.warning(f"Unknown city: {self.city}")
                return pd.DataFrame()
            
            params = {"$limit": limit}
            response = self.session.get(endpoint, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                df = pd.DataFrame(data)
                
                # Save to local storage
                filename = f"sensors_{datetime.now().strftime('%Y%m%d')}.csv"
                filepath = self.data_path / filename
                df.to_csv(filepath, index=False)
                logger.info(f"Saved {len(df)} sensor records to {filepath}")
                
                return df
            else:
                logger.warning(f"Failed to fetch sensors: {response.status_code}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error fetching traffic sensors: {e}")
            return pd.DataFrame()
    
    def get_traffic_incidents(self, hours: int = 24) -> pd.DataFrame:
        """
        Get traffic incidents data.
        
        Args:
            hours: Number of hours of incidents to retrieve
        
        Returns:
            DataFrame with incident data
        """
        try:
            if self.city == "nyc":
                # NYC 311 service requests or traffic incidents
                dataset_id = "erm2-nwe9"  # NYC 311 Service Requests
                endpoint = f"{self.api_url}/resource/{dataset_id}.json"
                
                # Filter for recent traffic-related incidents
                start_time = (datetime.now() - timedelta(hours=hours)).isoformat()
                params = {
                    "$limit": 1000,
                    "$where": f"created_date >= '{start_time}'"
                }
            else:
                # Generic endpoint for other cities
                endpoint = f"{self.api_url}/resource/traffic_incidents.json"
                params = {"$limit": 1000}
            
            response = self.session.get(endpoint, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                df = pd.DataFrame(data)
                
                if not df.empty:
                    filename = f"incidents_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    filepath = self.data_path / filename
                    df.to_csv(filepath, index=False)
                    logger.info(f"Saved {len(df)} incident records")
                
                return df
            else:
                logger.warning(f"Failed to fetch incidents: {response.status_code}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error fetching traffic incidents: {e}")
            return pd.DataFrame()
    
    def get_road_network(self) -> pd.DataFrame:
        """
        Get road network data (streets, intersections, etc.).
        
        Returns:
            DataFrame with road network data
        """
        try:
            if self.city == "nyc":
                # NYC Street Centerline dataset
                dataset_id = "8k85-6kxv"
                endpoint = f"{self.api_url}/resource/{dataset_id}.json"
            else:
                endpoint = f"{self.api_url}/resource/roads.json"
            
            params = {"$limit": 5000}
            response = self.session.get(endpoint, params=params, timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                df = pd.DataFrame(data)
                
                filename = f"road_network_{datetime.now().strftime('%Y%m%d')}.csv"
                filepath = self.data_path / filename
                df.to_csv(filepath, index=False)
                logger.info(f"Saved {len(df)} road network records")
                
                return df
            else:
                logger.warning(f"Failed to fetch road network: {response.status_code}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error fetching road network: {e}")
            return pd.DataFrame()
    
    def search_datasets(self, query: str) -> List[Dict]:
        """
        Search for available datasets in the city's open data portal.
        
        Args:
            query: Search query
        
        Returns:
            List of dataset information dictionaries
        """
        try:
            # Socrata API search endpoint
            endpoint = f"{self.api_url}/api/catalog/v1"
            params = {"q": query, "limit": 20}
            
            response = self.session.get(endpoint, params=params, timeout=30)
            
            if response.status_code == 200:
                results = response.json()
                return results.get("results", [])
            else:
                logger.warning(f"Search failed: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error searching datasets: {e}")
            return []
    
    def process_sensor_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process and clean sensor data.
        
        Args:
            df: Raw sensor DataFrame
        
        Returns:
            Processed DataFrame
        """
        if df.empty:
            return df
        
        try:
            # Standardize timestamp column
            timestamp_cols = ['timestamp', 'datetime', 'created_date', 'date']
            for col in timestamp_cols:
                if col in df.columns:
                    df['timestamp'] = pd.to_datetime(df[col])
                    break
            
            # Ensure numeric columns
            numeric_cols = ['speed', 'volume', 'occupancy', 'latitude', 'longitude']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove invalid records
            df = df.dropna(subset=['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            logger.info(f"Processed {len(df)} sensor records")
            return df
            
        except Exception as e:
            logger.error(f"Error processing sensor data: {e}")
            return df

