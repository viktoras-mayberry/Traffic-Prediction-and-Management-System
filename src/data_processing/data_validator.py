"""
Data Validation and Quality Checks
Validates incoming data for quality and consistency
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from ..utils.logger import get_logger

logger = get_logger(__name__)


class DataValidator:
    """
    Validates data quality and consistency.
    """
    
    def __init__(
        self,
        outlier_method: str = "iqr",
        outlier_threshold: float = 3.0
    ):
        """
        Initialize data validator.
        
        Args:
            outlier_method: Method for outlier detection (iqr, zscore)
            outlier_threshold: Threshold for outlier detection
        """
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
    
    def validate_traffic_data(
        self,
        df: pd.DataFrame,
        required_columns: List[str] = None
    ) -> Tuple[bool, List[str]]:
        """
        Validate traffic data DataFrame.
        
        Args:
            df: DataFrame to validate
            required_columns: List of required column names
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        if df.empty:
            errors.append("DataFrame is empty")
            return False, errors
        
        # Check required columns
        if required_columns:
            missing_cols = set(required_columns) - set(df.columns)
            if missing_cols:
                errors.append(f"Missing required columns: {missing_cols}")
        
        # Check for timestamp column
        if 'timestamp' not in df.columns:
            errors.append("Missing 'timestamp' column")
        else:
            # Validate timestamp format
            try:
                pd.to_datetime(df['timestamp'])
            except Exception as e:
                errors.append(f"Invalid timestamp format: {e}")
        
        # Check for missing values in critical columns
        critical_cols = ['timestamp']
        for col in critical_cols:
            if col in df.columns and df[col].isna().any():
                errors.append(f"Missing values in critical column: {col}")
        
        # Check data types
        numeric_cols = ['speed', 'volume', 'occupancy', 'vehicle_count']
        for col in numeric_cols:
            if col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    errors.append(f"Column {col} should be numeric")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def detect_outliers(
        self,
        df: pd.DataFrame,
        columns: List[str] = None
    ) -> pd.DataFrame:
        """
        Detect outliers in specified columns.
        
        Args:
            df: DataFrame to check
            columns: List of column names to check (if None, checks all numeric columns)
        
        Returns:
            DataFrame with added 'is_outlier' column
        """
        if df.empty:
            return df
        
        df = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        outlier_mask = pd.Series([False] * len(df), index=df.index)
        
        for col in columns:
            if col not in df.columns:
                continue
            
            if self.outlier_method == "iqr":
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                col_outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
                outlier_mask = outlier_mask | col_outliers
                
            elif self.outlier_method == "zscore":
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                col_outliers = z_scores > self.outlier_threshold
                outlier_mask = outlier_mask | col_outliers
        
        df['is_outlier'] = outlier_mask
        
        if outlier_mask.any():
            logger.warning(f"Detected {outlier_mask.sum()} outlier records")
        
        return df
    
    def remove_outliers(
        self,
        df: pd.DataFrame,
        columns: List[str] = None
    ) -> pd.DataFrame:
        """
        Remove outliers from DataFrame.
        
        Args:
            df: DataFrame to clean
            columns: List of column names to check
        
        Returns:
            DataFrame with outliers removed
        """
        df_with_outliers = self.detect_outliers(df, columns)
        df_cleaned = df_with_outliers[~df_with_outliers['is_outlier']].copy()
        df_cleaned = df_cleaned.drop(columns=['is_outlier'], errors='ignore')
        
        removed_count = len(df) - len(df_cleaned)
        if removed_count > 0:
            logger.info(f"Removed {removed_count} outlier records")
        
        return df_cleaned
    
    def validate_timestamp_range(
        self,
        df: pd.DataFrame,
        min_timestamp: Optional[datetime] = None,
        max_timestamp: Optional[datetime] = None
    ) -> Tuple[bool, List[str]]:
        """
        Validate that timestamps are within expected range.
        
        Args:
            df: DataFrame with timestamp column
            min_timestamp: Minimum allowed timestamp
            max_timestamp: Maximum allowed timestamp
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        if 'timestamp' not in df.columns:
            return False, ["Missing timestamp column"]
        
        timestamps = pd.to_datetime(df['timestamp'])
        
        if min_timestamp:
            invalid_min = timestamps < min_timestamp
            if invalid_min.any():
                errors.append(f"{invalid_min.sum()} timestamps before minimum: {min_timestamp}")
        
        if max_timestamp:
            invalid_max = timestamps > max_timestamp
            if invalid_max.any():
                errors.append(f"{invalid_max.sum()} timestamps after maximum: {max_timestamp}")
        
        # Check for future timestamps (more than 1 hour in future)
        future_threshold = datetime.now() + pd.Timedelta(hours=1)
        future_timestamps = timestamps > future_threshold
        if future_timestamps.any():
            errors.append(f"{future_timestamps.sum()} timestamps in the future")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def validate_geographic_bounds(
        self,
        df: pd.DataFrame,
        min_lat: float = -90,
        max_lat: float = 90,
        min_lon: float = -180,
        max_lon: float = 180
    ) -> Tuple[bool, List[str]]:
        """
        Validate geographic coordinates are within bounds.
        
        Args:
            df: DataFrame with latitude and longitude columns
            min_lat: Minimum latitude
            max_lat: Maximum latitude
            min_lon: Minimum longitude
            max_lon: Maximum longitude
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        if 'latitude' not in df.columns or 'longitude' not in df.columns:
            return True, []  # Not an error if columns don't exist
        
        invalid_lat = (df['latitude'] < min_lat) | (df['latitude'] > max_lat)
        invalid_lon = (df['longitude'] < min_lon) | (df['longitude'] > max_lon)
        
        if invalid_lat.any():
            errors.append(f"{invalid_lat.sum()} invalid latitude values")
        
        if invalid_lon.any():
            errors.append(f"{invalid_lon.sum()} invalid longitude values")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def get_data_quality_report(self, df: pd.DataFrame) -> Dict:
        """
        Generate a comprehensive data quality report.
        
        Args:
            df: DataFrame to analyze
        
        Returns:
            Dictionary with quality metrics
        """
        report = {
            "total_records": len(df),
            "total_columns": len(df.columns),
            "missing_values": {},
            "data_types": {},
            "numeric_stats": {},
            "timestamp_info": {}
        }
        
        # Missing values
        for col in df.columns:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                report["missing_values"][col] = {
                    "count": int(missing_count),
                    "percentage": float(missing_count / len(df) * 100)
                }
        
        # Data types
        for col in df.columns:
            report["data_types"][col] = str(df[col].dtype)
        
        # Numeric statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            report["numeric_stats"][col] = {
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "median": float(df[col].median())
            }
        
        # Timestamp information
        if 'timestamp' in df.columns:
            timestamps = pd.to_datetime(df['timestamp'])
            report["timestamp_info"] = {
                "earliest": str(timestamps.min()),
                "latest": str(timestamps.max()),
                "span_hours": float((timestamps.max() - timestamps.min()).total_seconds() / 3600)
            }
        
        return report

