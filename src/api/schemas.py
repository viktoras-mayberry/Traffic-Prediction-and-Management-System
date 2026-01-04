"""
API Schemas
Pydantic models for API request/response validation
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Tuple
from datetime import datetime


class PredictionRequest(BaseModel):
    """Request schema for traffic prediction."""
    location_id: Optional[str] = None
    latitude: Optional[float] = Field(None, ge=-90, le=90)
    longitude: Optional[float] = Field(None, ge=-180, le=180)
    horizon_minutes: int = Field(15, ge=1, le=120)
    features: Optional[Dict] = None


class PredictionResponse(BaseModel):
    """Response schema for traffic prediction."""
    timestamp: str
    horizon_minutes: int
    predictions: Dict
    model_type: str
    confidence: Optional[float] = None


class RouteRequest(BaseModel):
    """Request schema for route optimization."""
    origin: Tuple[float, float] = Field(..., description="(latitude, longitude)")
    destination: Tuple[float, float] = Field(..., description="(latitude, longitude)")
    algorithm: Optional[str] = Field("astar", description="dijkstra, astar, bellman_ford")
    consider_traffic: bool = True
    avoid_tolls: bool = False


class RouteResponse(BaseModel):
    """Response schema for route optimization."""
    success: bool
    algorithm: str
    path: List[int]
    coordinates: List[Tuple[float, float]]
    distance_km: float
    estimated_travel_time_minutes: Optional[float] = None
    message: Optional[str] = None


class CongestionRequest(BaseModel):
    """Request schema for congestion detection."""
    location_ids: Optional[List[str]] = None
    bounding_box: Optional[Tuple[float, float, float, float]] = Field(
        None,
        description="(min_lat, min_lon, max_lat, max_lon)"
    )


class CongestionResponse(BaseModel):
    """Response schema for congestion detection."""
    timestamp: str
    hotspots: List[Dict]
    summary: Dict


class SignalRequest(BaseModel):
    """Request schema for signal control."""
    intersection_id: str
    traffic_flows: Dict[str, float] = Field(..., description="Flow rates by approach direction")


class SignalResponse(BaseModel):
    """Response schema for signal control."""
    intersection_id: str
    timestamp: str
    cycle_time: int
    timing: Dict
    recommendation: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    version: str

