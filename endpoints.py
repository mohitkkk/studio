"""
Additional API endpoints registration module.

This module provides a function to register additional API endpoints 
to the main FastAPI application instance.
"""

import logging
from fastapi import FastAPI, HTTPException

logger = logging.getLogger(__name__)

def register_endpoints(app: FastAPI) -> None:
    """
    Register additional API endpoints with the FastAPI application.
    
    Args:
        app: The FastAPI application instance
    """
    logger.info("Registering additional API endpoints")
    
    @app.get("/api/status", summary="API Status")
    async def api_status():
        """Returns the current status of the API."""
        return {
            "status": "ok",
            "version": "1.0",
            "api_ready": True
        }
    
    @app.get("/api/check", summary="API Health Check")
    async def api_check():
        """Returns a simple health check response."""
        return {"status": "healthy"}
    
    # Add more custom endpoints here if needed in the future
    
    logger.info("Additional API endpoints registered successfully")
