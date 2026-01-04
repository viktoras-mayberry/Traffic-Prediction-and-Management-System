"""
FastAPI Main Application
Traffic Prediction and Management System API
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import yaml
from pathlib import Path

from .routes import router
from ..utils.logger import setup_logger, get_logger

# Setup logging
logger = setup_logger("traffic_api")

# Create FastAPI app
app = FastAPI(
    title="Traffic Prediction and Management System API",
    description="API for traffic prediction, route optimization, and traffic management",
    version="1.0.0"
)

# Load configuration
config_path = Path("config/config.yaml")
cors_origins = ["http://localhost:3000", "http://localhost:8080"]

if config_path.exists():
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            if 'api' in config and 'cors_origins' in config['api']:
                cors_origins = config['api']['cors_origins']
    except Exception as e:
        logger.warning(f"Could not load API config: {e}")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(router)


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


@app.on_event("startup")
async def startup_event():
    """Startup event handler."""
    logger.info("Traffic Prediction API starting up...")


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler."""
    logger.info("Traffic Prediction API shutting down...")


if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

