from fastapi import FastAPI, APIRouter, HTTPException, Body
from pydantic import BaseModel
from typing import List, Optional
import logging
import time
import os
import json

# Setup logger
logger = logging.getLogger(__name__)

# Models for file selection webhook
class FileSelectionRequest(BaseModel):
    session_id: str
    selected_files: List[str]
    timestamp: Optional[float] = None

# Router for additional endpoints
file_router = APIRouter()

@file_router.post("/file-selection", summary="Update file selection for a session")
async def update_file_selection(request: FileSelectionRequest = Body(...)):
    """
    Webhook endpoint that receives file selection updates.
    This allows the backend to be notified when files are selected in the UI.
    """
    try:
        # Add received timestamp if not provided
        if not request.timestamp:
            request.timestamp = time.time()
            
        # Log the file selection event
        logger.info(f"File selection webhook received for session {request.session_id}: {len(request.selected_files)} files")
        logger.debug(f"Selected files: {request.selected_files}")
        
        # Here you could add logic to:
        # 1. Pre-load or cache file content/embeddings
        # 2. Update session metadata
        # 3. Track file usage statistics
        
        # For now, just acknowledge receipt
        return {
            "success": True,
            "message": f"Received selection of {len(request.selected_files)} files",
            "session_id": request.session_id,
            "timestamp": request.timestamp
        }
    except Exception as e:
        logger.error(f"Error processing file selection webhook: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process file selection: {str(e)}")

def register_endpoints(app: FastAPI):
    """
    Register all endpoints from this module with the main app.
    """
    app.include_router(file_router)
    logger.info("Additional endpoints registered")
