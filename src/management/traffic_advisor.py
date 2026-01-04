"""
Traffic Advisor
Real-time traffic advice and recommendations
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
from ..utils.logger import get_logger

logger = get_logger(__name__)


class TrafficAdvisor:
    """
    Provides real-time traffic advice and recommendations.
    """
    
    def __init__(self):
        """Initialize traffic advisor."""
        pass
    
    def get_traffic_alerts(
        self,
        congestion_data: List[Dict],
        severity_threshold: str = "high"
    ) -> List[Dict]:
        """
        Get traffic alerts based on congestion data.
        
        Args:
            congestion_data: List of congestion data dictionaries
            severity_threshold: Minimum severity level for alerts
        
        Returns:
            List of alert dictionaries
        """
        severity_levels = {'none': 0, 'low': 1, 'medium': 2, 'high': 3, 'severe': 4}
        threshold_level = severity_levels.get(severity_threshold.lower(), 2)
        
        alerts = []
        for data in congestion_data:
            severity = data.get('severity', 'none')
            severity_level = severity_levels.get(severity.lower(), 0)
            
            if severity_level >= threshold_level:
                alert = {
                    'type': 'congestion',
                    'severity': severity,
                    'location': {
                        'latitude': data.get('latitude', 0),
                        'longitude': data.get('longitude', 0),
                        'location_id': data.get('location_id', 'unknown')
                    },
                    'message': self._generate_alert_message(data),
                    'timestamp': datetime.now().isoformat(),
                    'recommendation': self._get_recommendation(data)
                }
                alerts.append(alert)
        
        logger.info(f"Generated {len(alerts)} traffic alerts")
        return alerts
    
    def _generate_alert_message(self, data: Dict) -> str:
        """Generate alert message from congestion data."""
        severity = data.get('severity', 'unknown')
        location_id = data.get('location_id', 'location')
        
        messages = {
            'severe': f"Severe congestion detected at {location_id}. Expect significant delays.",
            'high': f"High congestion at {location_id}. Consider alternative routes.",
            'medium': f"Moderate congestion at {location_id}. Minor delays expected.",
            'low': f"Light congestion at {location_id}.",
            'none': f"No congestion at {location_id}."
        }
        
        return messages.get(severity, f"Traffic condition at {location_id}.")
    
    def _get_recommendation(self, data: Dict) -> str:
        """Get recommendation based on congestion data."""
        severity = data.get('severity', 'unknown')
        
        recommendations = {
            'severe': "Avoid this route. Use alternative path if possible.",
            'high': "Consider alternative route to save time.",
            'medium': "Expect some delays. Alternative route may be faster.",
            'low': "Minor delays possible. Route is acceptable.",
            'none': "No action needed. Route is clear."
        }
        
        return recommendations.get(severity, "Monitor traffic conditions.")
    
    def get_route_recommendations(
        self,
        origin: tuple,
        destination: tuple,
        current_conditions: Dict,
        preferences: Optional[Dict] = None
    ) -> Dict:
        """
        Get route recommendations based on current conditions.
        
        Args:
            origin: (latitude, longitude) of origin
            destination: (latitude, longitude) of destination
            current_conditions: Current traffic conditions
            preferences: User preferences (fastest, shortest, avoid_tolls, etc.)
        
        Returns:
            Dictionary with route recommendations
        """
        if preferences is None:
            preferences = {'priority': 'fastest'}
        
        recommendations = {
            'origin': origin,
            'destination': destination,
            'timestamp': datetime.now().isoformat(),
            'preferences': preferences,
            'recommendations': []
        }
        
        # Generate recommendations based on conditions
        if current_conditions.get('congestion_level', 'none') in ['high', 'severe']:
            recommendations['recommendations'].append({
                'type': 'alternative_route',
                'message': 'Primary route has high congestion. Alternative routes available.',
                'priority': 'high'
            })
        
        if preferences.get('priority') == 'fastest':
            recommendations['recommendations'].append({
                'type': 'route_optimization',
                'message': 'Optimizing route for fastest travel time.',
                'priority': 'medium'
            })
        
        logger.info("Generated route recommendations")
        return recommendations
    
    def get_travel_advisory(
        self,
        route_info: Dict,
        weather_data: Optional[Dict] = None
    ) -> Dict:
        """
        Get comprehensive travel advisory for a route.
        
        Args:
            route_info: Route information dictionary
            weather_data: Optional weather data
        
        Returns:
            Dictionary with travel advisory
        """
        advisory = {
            'route_id': route_info.get('route_id', 'unknown'),
            'timestamp': datetime.now().isoformat(),
            'estimated_travel_time': route_info.get('travel_time_minutes', 0),
            'distance_km': route_info.get('distance_km', 0),
            'traffic_conditions': route_info.get('traffic_conditions', 'unknown'),
            'alerts': [],
            'recommendations': []
        }
        
        # Add traffic alerts
        if route_info.get('congestion_level') in ['high', 'severe']:
            advisory['alerts'].append({
                'type': 'congestion',
                'message': 'Heavy traffic expected on this route.'
            })
        
        # Add weather alerts
        if weather_data:
            if weather_data.get('precipitation', 0) > 0:
                advisory['alerts'].append({
                    'type': 'weather',
                    'message': f"Rain expected. Reduce speed and increase following distance."
                })
            
            if weather_data.get('visibility', 10) < 5:
                advisory['alerts'].append({
                    'type': 'weather',
                    'message': 'Low visibility conditions. Drive with caution.'
                })
        
        # Add recommendations
        if route_info.get('estimated_travel_time', 0) > 60:
            advisory['recommendations'].append({
                'type': 'planning',
                'message': 'Long journey expected. Plan for rest stops.'
            })
        
        logger.info("Generated travel advisory")
        return advisory

