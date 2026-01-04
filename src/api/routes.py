"""
API Routes
FastAPI route handlers
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List
import pandas as pd
import numpy as np
from datetime import datetime

from .schemas import (
    PredictionRequest, PredictionResponse,
    RouteRequest, RouteResponse,
    CongestionRequest, CongestionResponse,
    SignalRequest, SignalResponse,
    HealthResponse
)
from ..prediction.traffic_predictor import TrafficPredictor
from ..prediction.congestion_detector import CongestionDetector
from ..management.route_optimizer import RouteOptimizer
from ..management.signal_controller import SignalController
from ..utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["traffic"])

# Initialize services (in production, these would be dependency injected)
traffic_predictor = None
congestion_detector = CongestionDetector()
route_optimizer = RouteOptimizer()
signal_controller = SignalController()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0"
    )


@router.post("/predictions", response_model=PredictionResponse)
async def get_predictions(request: PredictionRequest):
    """
    Get traffic predictions.
    
    Args:
        request: Prediction request
    
    Returns:
        Prediction response
    """
    try:
        global traffic_predictor
        if traffic_predictor is None:
            # Initialize predictor (in production, load from config)
            traffic_predictor = TrafficPredictor(model_type="lstm")
        
        # Create sample data for prediction
        # In production, this would fetch actual data
        data = pd.DataFrame({
            'timestamp': [datetime.now()],
            'speed': [50.0],
            'vehicle_count': [100],
            'occupancy': [60.0]
        })
        
        prediction = traffic_predictor.predict(data, horizon_minutes=request.horizon_minutes)
        
        return PredictionResponse(**prediction)
        
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/routes/optimize", response_model=RouteResponse)
async def optimize_route(request: RouteRequest):
    """
    Get optimized route.
    
    Args:
        request: Route optimization request
    
    Returns:
        Route response
    """
    try:
        route_optimizer.algorithm = request.algorithm
        
        # Build sample graph (in production, load from road network data)
        nodes = [
            {'id': 0, 'latitude': request.origin[0], 'longitude': request.origin[1]},
            {'id': 1, 'latitude': request.destination[0], 'longitude': request.destination[1]}
        ]
        edges = [(0, 1, 1.0)]  # Simple edge
        
        route_optimizer.build_graph(nodes, edges)
        
        result = route_optimizer.find_optimal_route(
            request.origin,
            request.destination
        )
        
        if not result.get('success'):
            raise HTTPException(status_code=404, detail=result.get('message', 'Route not found'))
        
        return RouteResponse(**result)
        
    except Exception as e:
        logger.error(f"Error in route optimization: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/congestion", response_model=CongestionResponse)
async def get_congestion(
    location_ids: Optional[List[str]] = Query(None),
    bounding_box: Optional[str] = Query(None)
):
    """
    Get congestion predictions.
    
    Args:
        location_ids: Optional list of location IDs
        bounding_box: Optional bounding box (min_lat,min_lon,max_lat,max_lon)
    
    Returns:
        Congestion response
    """
    try:
        # Create sample data
        data = pd.DataFrame({
            'location_id': ['LOC001', 'LOC002', 'LOC003'],
            'latitude': [40.7128, 40.7130, 40.7125],
            'longitude': [-74.0060, -74.0058, -74.0062],
            'speed': [30.0, 25.0, 35.0],
            'speed_limit': [50.0, 50.0, 50.0],
            'occupancy': [85.0, 90.0, 70.0]
        })
        
        # Detect congestion
        congested_data = congestion_detector.detect_congestion(data)
        hotspots = congestion_detector.get_congestion_hotspots(congested_data, top_n=10)
        
        return CongestionResponse(
            timestamp=datetime.now().isoformat(),
            hotspots=hotspots,
            summary={
                'total_locations': len(data),
                'congested_count': len(hotspots),
                'severe_count': sum(1 for h in hotspots if h.get('severity') == 'severe')
            }
        )
        
    except Exception as e:
        logger.error(f"Error in congestion detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/signals/recommend", response_model=SignalResponse)
async def recommend_signal_timing(request: SignalRequest):
    """
    Get signal control recommendations.
    
    Args:
        request: Signal control request
    
    Returns:
        Signal timing response
    """
    try:
        result = signal_controller.optimize_signal_timing(
            request.intersection_id,
            request.traffic_flows
        )
        
        return SignalResponse(**result)
        
    except Exception as e:
        logger.error(f"Error in signal control: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/predictions/travel-time")
async def predict_travel_time(
    origin_lat: float = Query(..., ge=-90, le=90),
    origin_lon: float = Query(..., ge=-180, le=180),
    dest_lat: float = Query(..., ge=-90, le=90),
    dest_lon: float = Query(..., ge=-180, le=180)
):
    """
    Predict travel time between two points.
    
    Args:
        origin_lat: Origin latitude
        origin_lon: Origin longitude
        dest_lat: Destination latitude
        dest_lon: Destination longitude
    
    Returns:
        Travel time prediction
    """
    try:
        global traffic_predictor
        if traffic_predictor is None:
            traffic_predictor = TrafficPredictor(model_type="lstm")
        
        # Create sample traffic data
        data = pd.DataFrame({
            'timestamp': [datetime.now()],
            'speed': [50.0],
            'vehicle_count': [100],
            'occupancy': [60.0]
        })
        
        result = traffic_predictor.predict_travel_time(
            (origin_lat, origin_lon),
            (dest_lat, dest_lon),
            data
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error in travel time prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

