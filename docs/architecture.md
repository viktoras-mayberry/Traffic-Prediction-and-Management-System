# System Architecture

## Overview

The Traffic Prediction and Management System follows a modular, scalable architecture designed for real-time processing and prediction.

## Architecture Layers

### 1. Data Ingestion Layer
- **Uber Movement Integration**: Downloads anonymized travel time data
- **City APIs**: Integrates with municipal open data portals
- **Sensor Readers**: Processes data from loop detectors, cameras, and radar
- **GPS Processing**: Handles GPS trajectory data

### 2. Data Processing Layer
- **Apache Spark Streaming**: Real-time data processing pipeline
- **Feature Engineering**: Extracts temporal, spatial, and traffic features
- **Data Validation**: Ensures data quality and consistency

### 3. Machine Learning Layer
- **LSTM Model**: Time-series traffic prediction
- **CNN-LSTM Hybrid**: Spatial-temporal pattern recognition
- **Graph Neural Network**: Road network topology modeling
- **Model Trainer**: Unified training interface

### 4. Prediction Engine
- **Traffic Predictor**: Main prediction service
- **Congestion Detector**: Identifies and predicts congestion

### 5. Management & Optimization
- **Route Optimizer**: Dijkstra, A* algorithms
- **Signal Controller**: Dynamic traffic signal timing
- **Traffic Advisor**: Real-time recommendations

### 6. API Layer
- **REST API**: FastAPI-based endpoints
- **WebSocket Support**: Real-time updates
- **Authentication & Rate Limiting**: Security features

## Data Flow

```
Data Sources → Ingestion → Spark Streaming → Feature Engineering
                                                      ↓
Storage ← API Layer ← Management ← Prediction ← ML Models
```

## Technology Stack

- **Python 3.9+**: Core language
- **TensorFlow/Keras**: Deep learning
- **Apache Spark**: Big data processing
- **FastAPI**: REST API
- **PostgreSQL/TimescaleDB**: Time-series storage
- **Redis**: Caching
- **Docker**: Containerization

