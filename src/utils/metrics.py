"""
Evaluation Metrics for Traffic Prediction Models
"""

import numpy as np
from typing import Dict, Union, List
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: List[str] = None
) -> Dict[str, float]:
    """
    Calculate evaluation metrics for predictions.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        metrics: List of metrics to calculate. Options: mae, rmse, mape, r2
    
    Returns:
        Dictionary of metric names and values
    """
    if metrics is None:
        metrics = ['mae', 'rmse', 'mape', 'r2']
    
    results = {}
    
    # Mean Absolute Error
    if 'mae' in metrics:
        results['mae'] = mean_absolute_error(y_true, y_pred)
    
    # Root Mean Squared Error
    if 'rmse' in metrics:
        results['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Mean Absolute Percentage Error
    if 'mape' in metrics:
        # Avoid division by zero
        mask = y_true != 0
        if mask.sum() > 0:
            results['mape'] = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            results['mape'] = np.inf
    
    # R-squared
    if 'r2' in metrics:
        results['r2'] = r2_score(y_true, y_pred)
    
    # Mean Squared Error
    if 'mse' in metrics:
        results['mse'] = mean_squared_error(y_true, y_pred)
    
    return results


class MetricsCalculator:
    """
    Class for calculating and tracking metrics over time.
    """
    
    def __init__(self):
        self.metrics_history: Dict[str, List[float]] = {}
    
    def update(self, y_true: np.ndarray, y_pred: np.ndarray, metrics: List[str] = None):
        """
        Update metrics with new predictions.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            metrics: List of metrics to calculate
        """
        current_metrics = calculate_metrics(y_true, y_pred, metrics)
        
        for metric_name, value in current_metrics.items():
            if metric_name not in self.metrics_history:
                self.metrics_history[metric_name] = []
            self.metrics_history[metric_name].append(value)
    
    def get_average_metrics(self) -> Dict[str, float]:
        """
        Get average metrics across all updates.
        
        Returns:
            Dictionary of average metric values
        """
        return {
            metric_name: np.mean(values)
            for metric_name, values in self.metrics_history.items()
        }
    
    def get_latest_metrics(self) -> Dict[str, float]:
        """
        Get the most recent metrics.
        
        Returns:
            Dictionary of latest metric values
        """
        return {
            metric_name: values[-1]
            for metric_name, values in self.metrics_history.items()
        }
    
    def reset(self):
        """Reset all metrics history."""
        self.metrics_history = {}
    
    def get_summary(self) -> str:
        """
        Get a formatted summary of metrics.
        
        Returns:
            Formatted string summary
        """
        avg_metrics = self.get_average_metrics()
        summary_lines = ["Metrics Summary:"]
        summary_lines.append("-" * 50)
        
        for metric_name, value in avg_metrics.items():
            if metric_name == 'mape':
                summary_lines.append(f"{metric_name.upper()}: {value:.2f}%")
            else:
                summary_lines.append(f"{metric_name.upper()}: {value:.4f}")
        
        return "\n".join(summary_lines)

