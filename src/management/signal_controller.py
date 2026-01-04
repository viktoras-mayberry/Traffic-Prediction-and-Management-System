"""
Signal Controller
Dynamic traffic signal timing optimization
"""

import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from ..utils.logger import get_logger

logger = get_logger(__name__)


class SignalController:
    """
    Traffic signal control and optimization.
    """
    
    def __init__(
        self,
        min_green_time: int = 10,
        max_green_time: int = 120,
        default_cycle_time: int = 90
    ):
        """
        Initialize signal controller.
        
        Args:
            min_green_time: Minimum green time in seconds
            max_green_time: Maximum green time in seconds
            default_cycle_time: Default cycle time in seconds
        """
        self.min_green_time = min_green_time
        self.max_green_time = max_green_time
        self.default_cycle_time = default_cycle_time
    
    def optimize_signal_timing(
        self,
        intersection_id: str,
        traffic_flows: Dict[str, float],
        current_timing: Optional[Dict] = None
    ) -> Dict:
        """
        Optimize traffic signal timing for an intersection.
        
        Args:
            intersection_id: Intersection identifier
            traffic_flows: Dictionary mapping approach directions to flow rates (vehicles/hour)
            current_timing: Current signal timing configuration
        
        Returns:
            Dictionary with optimized timing recommendations
        """
        try:
            # Calculate total flow
            total_flow = sum(traffic_flows.values())
            
            if total_flow == 0:
                # No traffic, use default timing
                return self._get_default_timing(intersection_id)
            
            # Calculate green time allocation based on flow ratios
            green_times = {}
            for approach, flow in traffic_flows.items():
                flow_ratio = flow / total_flow
                green_time = max(
                    self.min_green_time,
                    min(
                        self.max_green_time,
                        int(self.default_cycle_time * flow_ratio)
                    )
                )
                green_times[approach] = green_time
            
            # Ensure minimum green times
            for approach in green_times:
                green_times[approach] = max(green_times[approach], self.min_green_time)
            
            # Calculate cycle time
            cycle_time = sum(green_times.values()) + len(green_times) * 3  # 3 seconds yellow/red
            
            # Calculate yellow and red times
            yellow_time = 3
            red_times = {}
            for approach in green_times:
                red_times[approach] = cycle_time - green_times[approach] - yellow_time
            
            result = {
                'intersection_id': intersection_id,
                'timestamp': datetime.now().isoformat(),
                'cycle_time': cycle_time,
                'timing': {
                    approach: {
                        'green': green_times[approach],
                        'yellow': yellow_time,
                        'red': red_times[approach]
                    }
                    for approach in green_times
                },
                'traffic_flows': traffic_flows,
                'recommendation': 'optimize' if current_timing else 'implement'
            }
            
            logger.info(f"Optimized signal timing for intersection {intersection_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error optimizing signal timing: {e}")
            raise
    
    def _get_default_timing(self, intersection_id: str) -> Dict:
        """Get default signal timing."""
        return {
            'intersection_id': intersection_id,
            'timestamp': datetime.now().isoformat(),
            'cycle_time': self.default_cycle_time,
            'timing': {
                'north': {'green': 30, 'yellow': 3, 'red': 57},
                'south': {'green': 30, 'yellow': 3, 'red': 57},
                'east': {'green': 30, 'yellow': 3, 'red': 57},
                'west': {'green': 30, 'yellow': 3, 'red': 57}
            },
            'recommendation': 'maintain'
        }
    
    def recommend_signal_adjustments(
        self,
        intersection_data: List[Dict]
    ) -> List[Dict]:
        """
        Recommend signal adjustments for multiple intersections.
        
        Args:
            intersection_data: List of intersection data dictionaries
        
        Returns:
            List of timing recommendations
        """
        recommendations = []
        
        for intersection in intersection_data:
            intersection_id = intersection.get('id')
            traffic_flows = intersection.get('traffic_flows', {})
            current_timing = intersection.get('current_timing')
            
            recommendation = self.optimize_signal_timing(
                intersection_id,
                traffic_flows,
                current_timing
            )
            recommendations.append(recommendation)
        
        logger.info(f"Generated {len(recommendations)} signal timing recommendations")
        return recommendations
    
    def calculate_delay_reduction(
        self,
        current_timing: Dict,
        optimized_timing: Dict
    ) -> Dict:
        """
        Calculate expected delay reduction from signal optimization.
        
        Args:
            current_timing: Current signal timing
            optimized_timing: Optimized signal timing
        
        Returns:
            Dictionary with delay reduction metrics
        """
        try:
            # Simplified delay calculation
            # In practice, this would use more sophisticated models
            
            current_cycle = current_timing.get('cycle_time', self.default_cycle_time)
            optimized_cycle = optimized_timing.get('cycle_time', self.default_cycle_time)
            
            cycle_reduction = current_cycle - optimized_cycle
            cycle_reduction_pct = (cycle_reduction / current_cycle * 100) if current_cycle > 0 else 0
            
            # Estimate delay reduction (simplified)
            # Average delay per vehicle = cycle_time / 2 (simplified)
            current_avg_delay = current_cycle / 2
            optimized_avg_delay = optimized_cycle / 2
            delay_reduction = current_avg_delay - optimized_avg_delay
            delay_reduction_pct = (delay_reduction / current_avg_delay * 100) if current_avg_delay > 0 else 0
            
            result = {
                'current_cycle_time': current_cycle,
                'optimized_cycle_time': optimized_cycle,
                'cycle_reduction_seconds': cycle_reduction,
                'cycle_reduction_percent': cycle_reduction_pct,
                'current_avg_delay': current_avg_delay,
                'optimized_avg_delay': optimized_avg_delay,
                'delay_reduction_seconds': delay_reduction,
                'delay_reduction_percent': delay_reduction_pct
            }
            
            logger.info(f"Estimated delay reduction: {delay_reduction_pct:.2f}%")
            return result
            
        except Exception as e:
            logger.error(f"Error calculating delay reduction: {e}")
            return {}

