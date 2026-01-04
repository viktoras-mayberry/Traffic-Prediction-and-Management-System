# API Documentation

## Base URL
```
http://localhost:8000/api/v1
```

## Endpoints

### Health Check
```
GET /health
```
Returns system health status.

### Traffic Predictions
```
POST /predictions
```
Get traffic predictions for a location.

**Request Body:**
```json
{
  "location_id": "string",
  "latitude": 40.7128,
  "longitude": -74.0060,
  "horizon_minutes": 15
}
```

### Route Optimization
```
POST /routes/optimize
```
Get optimized route between two points.

**Request Body:**
```json
{
  "origin": [40.7128, -74.0060],
  "destination": [40.7580, -73.9855],
  "algorithm": "astar",
  "consider_traffic": true
}
```

### Congestion Detection
```
GET /congestion
```
Get congestion hotspots and predictions.

**Query Parameters:**
- `location_ids`: Optional list of location IDs
- `bounding_box`: Optional bounding box

### Signal Control
```
POST /signals/recommend
```
Get traffic signal timing recommendations.

**Request Body:**
```json
{
  "intersection_id": "INT001",
  "traffic_flows": {
    "north": 100.0,
    "south": 120.0,
    "east": 80.0,
    "west": 90.0
  }
}
```

### Travel Time Prediction
```
GET /predictions/travel-time
```
Predict travel time between two points.

**Query Parameters:**
- `origin_lat`: Origin latitude
- `origin_lon`: Origin longitude
- `dest_lat`: Destination latitude
- `dest_lon`: Destination longitude

