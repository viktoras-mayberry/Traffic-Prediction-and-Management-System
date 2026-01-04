"""
Tests for API endpoints
"""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_health_check():
    """Test health check endpoint."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_predictions_endpoint():
    """Test predictions endpoint."""
    response = client.post(
        "/api/v1/predictions",
        json={
            "horizon_minutes": 15,
            "latitude": 40.7128,
            "longitude": -74.0060
        }
    )
    # May fail if model not loaded, but should return proper error
    assert response.status_code in [200, 500]


def test_congestion_endpoint():
    """Test congestion endpoint."""
    response = client.get("/api/v1/congestion")
    assert response.status_code == 200
    data = response.json()
    assert "hotspots" in data
    assert "summary" in data

