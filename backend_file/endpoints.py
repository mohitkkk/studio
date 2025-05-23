import os
import json
import logging
import traceback
import asyncio
import re
import time
import shutil
import html
import uuid
import cv2
import numpy as np
from datetime import datetime
from contextlib import asynccontextmanager
from typing import List, Dict, Tuple, Optional, Any

import torch
from openai import OpenAI
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Body, Path, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask
import chromadb
from chromadb.utils import embedding_functions
import fitz
import uvicorn
import requests
from pathlib import Path
from tqdm import tqdm

from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage

from doc_processing import common_utils
from doc_processing import textual_pipeline
from doc_processing import visual_pipeline

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "DEBUG").upper(),
    filename="chatbot.log", filemode='a',
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.info("--- Main API Logging Initialized ---")

from doc_processing.common_utils import (
    check_and_install_dependencies,
    setup_device,
    check_ollama_server,
    init_ollama_client,
    initialize_vault_directory,
    get_vault_files,
    add_file_to_vault,
    update_file_metadata,
    get_specific_doc_metadata,
    parse_file_query,
    remove_doc_from_metadata,
    generate_no_information_response,
    track_response_quality,
    clean_response_language
)

system_message = """
You are an AI assistant that answers questions based *strictly and solely* on the information contained within the provided document context.
DO NOT use any information from your general training data or outside knowledge. Your response MUST be grounded ONLY in the text provided in the context snippets.

**VERY IMPORTANT RULE:** DO NOT include any references to the original documents in your answer. This means NO document titles, NO file names, NO page numbers (e.g., "Page 5"), NO block numbers, and NO section numbers (e.g., "Section 2.1"). Integrate the information seamlessly into your response.

If the answer to the user's question cannot be found *within the provided document context*, state clearly and concisely that you cannot find the relevant information in the documents. DO NOT guess or make up information.

Format your response using standard Markdown syntax. Use **bold** for key terms, and bullet points for lists where appropriate.
Ensure the response is easy to read and directly answers the user's question based *only* on the context.
"""

CONFIG = {
    "vault_directory": "vault_files/",
    "vault_metadata": "vault_metadata.json",
    "processed_docs_subdir": "processed_docs",
    "vector_store_path": "vector_store",
    "vector_collection_name": "document_chunks",
    "ollama_model": "llama3:latest",
    "ollama_embedding_model": "mxbai-embed-large:latest",
    "log_file": "chatbot.log",
    "log_level": "DEBUG",
    "image_dpi": 300,
    "image_format": "png",
    "layout_analysis_file": "layout_analysis.json",
    "ocr_results_file": "ocr_results.json",
    "text_chunk_output_dir": "extracted_text_chunks",
    "create_plain_text_audit": True,
    "embedding_batch_size": 16,
    "visual_context_results": 15,
    "textual_max_chunk_size": 800,
    "textual_chunk_overlap": 0,
    "textual_top_k_per_doc": 5,
    "textual_similarity_threshold": 0.45,
    "textual_context_results_limit": 15,
    "heading_keywords": ["section", "chapter", "part", "introduction", "overview", "summary", "conclusion", "references", "appendix", "table of contents", "index", "glossary", "discussion", "results", "methodology", "procedure", "specifications", "requirements", "features", "instructions"],
    "textual_heading_keywords": ["section", "chapter", "introduction", "conclusion"],
    "layout_row_y_alignment_abs_px": 10,
    "layout_row_y_alignment_ratio_of_height": 0.5,
    "layout_cell_min_x_gap_px": 12,
    "layout_cell_merge_max_x_gap_px": 6,
    "layout_new_column_x_start_offset_ratio": 0.25,
    "layout_min_text_len_for_block": 1,
    "layout_min_block_width_px": 1,
    "layout_min_block_height_px": 1,
    "layout_max_block_height_ratio_page": 0.25,
    "layout_avg_char_width_fallback_px": 7,
    "visual_context_results": 15, # Number of chunks passed to the LLM
    "chroma_retrieval_multiplier": 5,
    "debug_mode": True,
    "debug_target_doc_id": "1747374358_2fb6b580",
    "response_tracking_dir": "response_tracking",
    "tesseract_lang": "eng",
    "min_char_confidence_tess": -1,
    "min_block_avg_confidence_tess": 60,
    "tesseract_ocr_psm": "3",
    "tesseract_timeout": 120,
    "image_detection_min_area_ratio": 0.02,
    "image_detection_max_text_overlap_ratio": 0.2,
    "text_crop_margin": 2,
    "image_crop_margin": 5,
    "jpeg_quality": 90,
    "png_compression": 3,
    "perform_ocr_on_diagrams": False,
    "diagram_ocr_psm": 11,
    "min_chunk_length_for_indexing": 5,
    "enable_deskewing": True,
    "enable_binarization_visual": True,
    "fallback_page_width": 1000,
    "fallback_page_height": 1400,

    # Update YOLO parameters for better text detection
    "yolov8_layout_model_path": 'vault_files/model/yolov8l-seg.pt',  # Ensure this path is correct
    "yolov8_score_threshold": 0.3,  # Reduced from 0.5 to catch more text regions
    "layoutparser_text_categories": ["text", "title", "list", "caption", "paragraph"],  # Added "paragraph"
    "chunk_vertical_proximity_threshold_px": 20,  # Increased from 15 for better text grouping
    "min_chunk_text_length": 5,  # Reduced from 10 to catch shorter text blocks
    
    # Add debug settings for visual processing
    "debug_layout_visualization": True,
    "debug_layout_output_subdir": "debug_layout",
    "save_all_processing_steps": True,
}

current_handlers = logging.root.handlers[:]
for handler in current_handlers: logging.root.removeHandler(handler)
logging.basicConfig(
    level=getattr(logging, CONFIG["log_level"].upper(), logging.INFO),
    filename=CONFIG["log_file"],
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__) # Re-get the logger after basicConfig is called
logger.info("--- Main API Logging Re-initialized with CONFIG Settings ---")

device, has_gpu = common_utils.setup_device(CONFIG) # Pass CONFIG to setup_device
CONFIG["use_gpu"] = has_gpu # Update CONFIG with detected device info
CONFIG["device"] = str(device)

DEPENDENCY_STATUS = common_utils.check_and_install_dependencies(CONFIG) # Pass CONFIG to check_and_install_dependencies
# Check for PyMuPDF availability status after dependency check
PYMUPDF_AVAILABLE = DEPENDENCY_STATUS.get("fitz", False)

ollama_client: Optional[OpenAI] = None
chroma_client: Optional[chromadb.ClientAPI] = None
chroma_collection: Optional[chromadb.api.models.Collection.Collection] = None
ollama_ef: Optional[embedding_functions.OllamaEmbeddingFunction] = None

async def download_yolo_model(config_dict: Dict) -> bool:
    """
    Downloads the YOLOv8 model if it doesn't exist at the specified path.
    Returns True if successful or if model already exists, False otherwise.
    """
    model_path = config_dict.get("yolov8_layout_model_path", "")
    if not model_path:
        logger.error("Missing YOLOv8 model path in configuration")
        return False
        
    # Convert relative path to absolute path within vault directory
    abs_model_path = os.path.join(os.path.abspath(config_dict["vault_directory"]), model_path)
    model_dir = os.path.dirname(abs_model_path)
    
    # Check if model already exists
    if os.path.exists(abs_model_path) and os.path.getsize(abs_model_path) > 1000000:  # >1MB file exists
        logger.info(f"YOLOv8 model already exists at: {abs_model_path}")
        return True
    
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Model doesn't exist, download it
    model_url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-seg.pt"
    logger.info(f"Downloading YOLOv8 model from {model_url} to {abs_model_path}")
    
    try:
        # Download with progress tracking
        print(f"Downloading YOLOv8 model... This may take a few minutes.")
        response = requests.get(model_url, stream=True)
        
        if response.status_code != 200:
            logger.error(f"Failed to download model: HTTP status {response.status_code}")
            return False
            
        # Get content length if available
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 KB
        
        with open(abs_model_path, 'wb') as f, tqdm(
            desc="Downloading YOLOv8 model",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for data in response.iter_content(block_size):
                f.write(data)
                progress_bar.update(len(data))
        
        # Verify download
        if os.path.exists(abs_model_path) and os.path.getsize(abs_model_path) > 1000000:
            logger.info(f"Successfully downloaded YOLOv8 model to {abs_model_path}")
            print(f"✅ YOLOv8 model downloaded successfully!")
            return True
        else:
            logger.error(f"Downloaded model file is too small or corrupted")
            return False
            
    except Exception as e:
        logger.error(f"Error downloading YOLOv8 model: {str(e)}", exc_info=True)
        # Remove partial download if it exists
        if os.path.exists(abs_model_path):
            try:
                os.remove(abs_model_path)
            except:
                pass
        return False

@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    global ollama_client, chroma_client, chroma_collection, ollama_ef
    logger.info("--- Lifespan Startup: Initiated ---")

    common_utils.initialize_vault_directory(CONFIG)
    logger.info("Vault directory initialized via common_utils.")

    ollama_client = common_utils.init_ollama_client(CONFIG, DEPENDENCY_STATUS)
    if ollama_client: logger.info("Ollama client (OpenAI compatible) initialized successfully via lifespan.")
    else: logger.error("CRITICAL: Failed to initialize Ollama client via lifespan.")

    try:
        logger.info("Lifespan: Initializing ChromaDB client...")
        base_vault_dir = os.path.abspath(CONFIG["vault_directory"])
        vector_store_subdir = CONFIG["vector_store_path"]
        chroma_client_path = os.path.join(base_vault_dir, vector_store_subdir)
        os.makedirs(chroma_client_path, exist_ok=True)
        
        chroma_client = chromadb.PersistentClient(path=chroma_client_path)
        logger.info(f"Lifespan: ChromaDB PersistentClient OK. Version: {chroma_client.get_version()}")

        ollama_ef_url = os.environ.get("OLLAMA_HOST", "http://localhost:11434") + "/api/embeddings"
        ollama_ef = embedding_functions.OllamaEmbeddingFunction(
            url=ollama_ef_url, model_name=CONFIG["ollama_embedding_model"]
        )
        logger.info(f"Lifespan: Ollama embedding function for Chroma OK (URL: {ollama_ef_url}).")

        collection_name = CONFIG["vector_collection_name"]
        chroma_collection = chroma_client.get_or_create_collection(
            name=collection_name, embedding_function=ollama_ef
        )
        logger.info(f"Lifespan: ChromaDB collection '{chroma_collection.name}' (ID: {chroma_collection.id}) obtained/created. Count: {chroma_collection.count()}")
        if chroma_collection is None: logger.error("CRITICAL LIFESPAN ERROR: chroma_collection is None!")
        else: logger.info("SUCCESS LIFESPAN: chroma_collection assigned.")
    except Exception as e_chroma_ls:
        logger.error(f"CRITICAL LIFESPAN ERROR during ChromaDB setup: {e_chroma_ls}", exc_info=True)
        chroma_client = ollama_ef = chroma_collection = None
    
    # Download YOLO model if not present
    await download_yolo_model(CONFIG)
    
    logger.info("--- Lifespan Startup: Yielding to Application ---")
    yield
    logger.info("--- Lifespan Shutdown: Initiated ---")
    logger.info("--- Lifespan Shutdown: Complete ---")

# --- Session Manager for HTTP API ---
class SessionManager:
    def __init__(self):
        self.client_file_selections: Dict[str, List[str]] = {}
        self.client_embeddings: Dict[str, Dict[str, torch.Tensor]] = {}
        self.client_content: Dict[str, Dict[str, List[str]]] = {}
        self.client_memory: Dict[str, ConversationBufferMemory] = {}
        logger.info("SessionManager initialized for HTTP API")

    # Add compatibility method for code that still expects WebSocket functionality
    async def send_json(self, client_id: str, data: dict):
        """
        Dummy method to maintain compatibility with WebSocket code.
        In HTTP-only mode, this just logs the message that would have been sent.
        """
        message_type = data.get("type", "unknown")
        logger.debug(f"[HTTP Mode] Would have sent to client {client_id}: message type={message_type}")
        # For status messages, we'll log them at info level
        if message_type == "status":
            logger.info(f"Status update for client {client_id}: {data.get('message', '')}")

    def set_client_files(self, client_id: str, files: List[str]):
        self.client_file_selections[client_id] = files

    def get_client_files(self, client_id: str) -> List[str]:
        return self.client_file_selections.get(client_id, [])

    def set_client_embeddings(self, client_id: str, embeddings: Dict[str, torch.Tensor]):
        self.client_embeddings[client_id] = embeddings

    def get_client_embeddings(self, client_id: str) -> Dict[str, torch.Tensor]:
        return self.client_embeddings.get(client_id, {})

    def set_client_content(self, client_id: str, content: Dict[str, List[str]]):
        self.client_content[client_id] = content

    def get_client_content(self, client_id: str) -> Dict[str, List[str]]:
        return self.client_content.get(client_id, {})

    def get_client_memory(self, client_id: str) -> Optional[ConversationBufferMemory]:
        if client_id not in self.client_memory:
            self.client_memory[client_id] = ConversationBufferMemory(memory_key="history", return_messages=True)
        return self.client_memory.get(client_id)

    def cleanup_session_data(self, client_id: str):
        self.client_file_selections.pop(client_id, None)
        self.client_embeddings.pop(client_id, None)
        self.client_content.pop(client_id, None)
        self.client_memory.pop(client_id, None)
        logger.info(f"Cleaned all session data for client ID: {client_id}")

    def cleanup_file_data(self, filename_is_doc_id: str):
        for client_id in list(self.client_file_selections.keys()):
            if filename_is_doc_id in self.client_file_selections.get(client_id, []):
                self.client_file_selections[client_id].remove(filename_is_doc_id)
            self.client_embeddings.get(client_id, {}).pop(filename_is_doc_id, None)
            self.client_content.get(client_id, {}).pop(filename_is_doc_id, None)

# Create session manager instance
manager = SessionManager()

# --- FastAPI App with CORS ---
app = FastAPI(
    title="Chatbot API",
    description="REST API for Document RAG Chatbot",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# --- Pydantic Models ---
class ChatRequest(BaseModel):
    message: str
    session_id: str = Field(default_factory=lambda: f"session_{uuid.uuid4().hex}")
    selected_files: List[str] = []
    format: str = "html"
    
class ChatResponse(BaseModel):
    message: str
    context: List[Dict] = []
    session_id: str
    status: str = "success"

class FileSelectionRequest(BaseModel):
    files: List[str]
    session_id: str = Field(default_factory=lambda: f"session_{uuid.uuid4().hex}")

class ProcessRequest(BaseModel):
    filename: str

class StatusResponse(BaseModel):
    status: str
    message: str

# --- Core Logic Functions ---
async def _process_chat_logic(
    session_id: str,
    user_input: str,
    selected_files: List[str] = None
) -> Tuple[str, List[Dict], str]:
    """Core logic for processing a chat message, reusable by HTTP APIs"""
    logger.info(f"Processing chat for session {session_id}: '{user_input[:50]}...'")
    
    # Use selected files from request or from session
    session_selection = selected_files or manager.get_client_files(session_id) 
    if not session_selection:
        logger.warning(f"No documents selected for session {session_id}")
        all_meta_for_no_info = common_utils.get_vault_files(CONFIG)
        msg = common_utils.generate_no_information_response(CONFIG, user_input, [], all_meta_for_no_info)
        return msg, [], "no_docs_selected"

    # Parse query to see if specific files are requested
    all_meta = common_utils.get_vault_files(CONFIG)
    cleaned_query, query_targets = common_utils.parse_file_query(CONFIG, user_input, all_meta, session_selection)
    final_targets = query_targets if query_targets else session_selection

    if not final_targets:
        logger.warning(f"Query parsing resulted in no targets for session {session_id}")
        msg = common_utils.generate_no_information_response(CONFIG, cleaned_query, [], all_meta)
        return msg, [], "no_targets_parsed"

    logger.info(f"Chat targets for '{cleaned_query[:50]}...': {final_targets}")

    # Check readiness of targeted documents
    visual_ready, textual_ready, not_ready = [], [], []
    session_text_emb = manager.get_client_embeddings(session_id)
    session_text_cont = manager.get_client_content(session_id)

    for doc_id in final_targets:
        meta = common_utils.get_specific_doc_metadata(CONFIG, doc_id)
        if not meta:
            logger.warning(f"Metadata not found for targeted doc ID {doc_id}")
            not_ready.append(f"{doc_id}(NoMeta)")
            continue

        p_type = meta.get("pipeline_type", "visual")

        # Check if document is ready based on type
        if p_type == "visual" and meta.get("indexing_complete", False) and meta.get("pipeline_step") in ["indexing_complete", "indexing_complete_no_chunks"]:
            visual_ready.append(doc_id)
        elif p_type == "textual" and doc_id in session_text_cont and doc_id in session_text_emb:
            textual_ready.append(doc_id)
        else:
            # Document not ready, add to not_ready list with status
            status_detail = meta.get('pipeline_step', 'Unknown')
            if p_type == 'visual' and not meta.get('indexing_complete'):
                status_detail = meta.get('pipeline_step', 'Indexing Incomplete')
            if p_type == 'textual' and (doc_id not in session_text_cont or doc_id not in session_text_emb):
                status_detail = 'Content/Embeddings Not Loaded'
            not_ready.append(f"{meta.get('original_filename', doc_id)}({status_detail})")

    # If no documents are ready, return appropriate message
    if not visual_ready and not textual_ready:
        logger.warning(f"No documents ready for chat for session {session_id}. Not ready: {not_ready}")
        msg_parts = ["<p>The documents are not yet ready for chat.</p>"]
        if not_ready:
            msg_parts.append("<p>Document status:</p><ul>")
            for item in not_ready:
                msg_parts.append(f"<li>{html.escape(item)}</li>")
            msg_parts.append("</ul>")
        final_unready_msg = "".join(msg_parts)
        return final_unready_msg, [], "no_docs_ready"

    # Process visual documents if available
    if visual_ready:
        if not chroma_collection:
            logger.error("ChromaDB collection not available, cannot run visual chat")
            return "Internal server error: Document index not initialized for visual chat.", [], "chroma_error"

        try:
            logger.info(f"Processing visual documents for session {session_id}")
            raw_resp_text, ctx_used = await visual_pipeline.ollama_chat_visual_async(
                config_dict=CONFIG,
                user_input=cleaned_query,
                selected_doc_ids=visual_ready,
                client_id=session_id,
                ollama_client=ollama_client,
                system_message=system_message,
                manager=manager,
                common_utils_module=common_utils,
                chroma_collection_obj=chroma_collection
            )
            logger.info(f"Visual chat pipeline returned for session {session_id}")
            return raw_resp_text, ctx_used, "success_visual"

        except Exception as e:
            logger.error(f"Visual pipeline failed for session {session_id}: {e}", exc_info=True)
            error_msg = f"Failed to process visual documents: {str(e)[:100]}"
            return error_msg, [], "visual_pipeline_failed"

    # Process textual documents as fallback
    elif textual_ready:
        try:
            logger.info(f"Processing textual documents for session {session_id}")
            raw_resp_text, ctx_used = await textual_pipeline.ollama_chat_textual_async(
                config_dict=CONFIG,
                query=cleaned_query,
                doc_ids=textual_ready,
                session_embeddings=session_text_emb,
                session_content=session_text_cont,
                client_id=session_id,
                ollama_client=ollama_client,
                manager=manager,
                common_utils_module=common_utils,
                system_message_str=system_message
            )
            logger.info(f"Textual chat pipeline returned for session {session_id}")
            return raw_resp_text, ctx_used, "success_textual"

        except Exception as e:
            logger.error(f"Textual pipeline failed for session {session_id}: {e}", exc_info=True)
            error_msg = f"Failed to process textual documents: {str(e)[:100]}"
            return error_msg, [], "textual_pipeline_failed"

    # Logic error fallback
    logger.error(f"Logic error: Neither visual nor textual documents were ready after readiness check")
    return "Internal server error: Logic error in chat handler.", [], "logic_error"

async def _trigger_doc_processing_chain(
    doc_id: str,
    session_id: Optional[str] = None
) -> Tuple[bool, str]:
    """Triggers document processing for HTTP API requests"""
    
    logger.info(f"Triggering processing for doc_id: {doc_id}")
    
    # Fetch metadata
    meta = common_utils.get_specific_doc_metadata(CONFIG, doc_id)
    if not meta:
        logger.error(f"Metadata not found for doc ID {doc_id}")
        return False, "processing_chain_failed_no_meta"

    display_name = meta.get("original_filename", doc_id)
    pipeline_type = meta.get("pipeline_type", "visual")

    # Check if already processed
    is_visual_indexed = pipeline_type == "visual" and meta.get("indexing_complete", False) and meta.get("pipeline_step") in ["indexing_complete", "indexing_complete_no_chunks"]
    is_textual_prepared = pipeline_type == "textual" and meta.get("pipeline_step") == "textual_file_prepared"

    if is_visual_indexed or is_textual_prepared:
        logger.info(f"Doc {doc_id} is already processed. Skipping.")
        return True, meta.get("pipeline_step", "unknown")

    # Process based on pipeline type
    current_step = meta.get("pipeline_step", "unknown")
    
    try:
        if pipeline_type == "visual":
            # If the document is in pending state, first extract the images
            if current_step == "visual_images_pending":
                # Extract images from PDF first
                logger.info(f"Document {doc_id} is in pending state. Extracting images first.")
                processed_doc_path = os.path.join(CONFIG["vault_directory"], 
                                                 CONFIG["processed_docs_subdir"], 
                                                 doc_id)
                pages_dir = os.path.join(processed_doc_path, "pages")
                
                # Check if the file actually exists in pages directory
                if not os.path.exists(pages_dir) or not os.listdir(pages_dir):
                    logger.error(f"Pages directory empty or missing for {doc_id}: {pages_dir}")
                    common_utils.update_file_metadata(CONFIG, doc_id, metadata_extra={
                        "pipeline_step": "visual_images_extraction_failed"
                    })
                    return False, "images_extraction_failed"
                
                # Mark as extracted to continue the pipeline
                common_utils.update_file_metadata(CONFIG, doc_id, metadata_extra={
                    "pipeline_step": "visual_images_extracted"
                })
                # Update current step for the next phase
                current_step = "visual_images_extracted"
            
            # Run layout analysis if needed
            if current_step in ["visual_images_extracted"]:
                logger.info(f"Running layout analysis for {doc_id}")
                try:
                    # Layout analysis
                    layout_success_path = await visual_pipeline.perform_layout_analysis(doc_id, CONFIG, common_utils)
                    if not layout_success_path:
                        logger.error(f"Layout analysis failed for {doc_id}")
                        return False, "layout_analysis_failed"
                        
                    # Update metadata and current step
                    meta = common_utils.get_specific_doc_metadata(CONFIG, doc_id) or meta
                    current_step = meta.get("pipeline_step", "unknown")
                    logger.info(f"Layout analysis complete for {doc_id}, new step: {current_step}")
                except Exception as e:
                    logger.error(f"Error during layout analysis for {doc_id}: {e}", exc_info=True)
                    common_utils.update_file_metadata(CONFIG, doc_id, metadata_extra={
                        "pipeline_step": f"layout_analysis_failed_{e.__class__.__name__}"
                    })
                    return False, f"layout_analysis_failed_{e.__class__.__name__}"
                
            # Run OCR extraction if needed
            if current_step in ["layout_analysis_complete"]:
                logger.info(f"Running OCR extraction for {doc_id}")
                try:
                    ocr_success_path = await visual_pipeline.perform_ocr_extraction(doc_id, CONFIG, common_utils)
                    if not ocr_success_path:
                        logger.error(f"OCR extraction failed for {doc_id}")
                        return False, "ocr_extraction_failed"
                        
                    # Update metadata and current step
                    meta = common_utils.get_specific_doc_metadata(CONFIG, doc_id) or meta
                    current_step = meta.get("pipeline_step", "unknown")
                    logger.info(f"OCR extraction complete for {doc_id}, new step: {current_step}")
                except Exception as e:
                    logger.error(f"Error during OCR extraction for {doc_id}: {e}", exc_info=True)
                    common_utils.update_file_metadata(CONFIG, doc_id, metadata_extra={
                        "pipeline_step": f"ocr_extraction_failed_{e.__class__.__name__}"
                    })
                    return False, f"ocr_extraction_failed_{e.__class__.__name__}"
                
            # Run indexing if needed
            if current_step in ["ocr_extraction_complete"]:
                logger.info(f"Running indexing for {doc_id}")
                # Check for ChromaDB
                if not chroma_collection:
                    logger.error(f"ChromaDB collection not available for {doc_id}")
                    common_utils.update_file_metadata(CONFIG, doc_id, metadata_extra={"pipeline_step": "indexing_failed_no_chroma_client"})
                    return False, "indexing_failed_no_chroma_client"
                
                try:
                    # Indexing
                    index_success = await visual_pipeline.index_ocr_data(doc_id, CONFIG, chroma_collection, common_utils)
                    if not index_success:
                        logger.error(f"Indexing failed for {doc_id}")
                        return False, "indexing_failed"
                        
                    # Update metadata and current step
                    meta = common_utils.get_specific_doc_metadata(CONFIG, doc_id) or meta
                    current_step = meta.get("pipeline_step", "unknown")
                    logger.info(f"Indexing complete for {doc_id}, new step: {current_step}")
                except Exception as e:
                    logger.error(f"Error during indexing for {doc_id}: {e}", exc_info=True)
                    common_utils.update_file_metadata(CONFIG, doc_id, metadata_extra={
                        "pipeline_step": f"indexing_failed_{e.__class__.__name__}"
                    })
                    return False, f"indexing_failed_{e.__class__.__name__}"
            
            # Check if indexing is complete
            meta = common_utils.get_specific_doc_metadata(CONFIG, doc_id) or meta
            if meta.get("indexing_complete", False):
                logger.info(f"Visual processing successful for {doc_id}")
                return True, current_step
            else:
                logger.error(f"Visual processing incomplete for {doc_id}")
                return False, current_step
                
        elif pipeline_type == "textual":
            # For textual, we just need to check if file is prepared
            if current_step == "textual_file_prepared":
                logger.info(f"Textual document already prepared for {doc_id}")
                return True, current_step
            else:
                logger.error(f"Textual document not prepared for {doc_id}")
                return False, current_step
        else:
            # Unknown pipeline type
            error_msg = f"Unknown pipeline type '{pipeline_type}' for doc ID {doc_id}"
            logger.error(error_msg)
            return False, "unknown_pipeline_type"
            
    except Exception as e:
        logger.error(f"Error in document processing chain for {doc_id}: {e}", exc_info=True)
        return False, f"processing_error_{e.__class__.__name__}"

# --- HTTP API Endpoints ---

@app.get("/", summary="API Information")
async def get_api_info():
    """Root endpoint providing API information"""
    return {
        "status": "ok",
        "version": "1.0.0",
        "endpoints": ["/", "/chat", "/files", "/file-selection", "/upload", "/process", "/delete/{doc_id}"]
    }

@app.post("/chat", summary="Process Chat Message", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Process a chat message and return a response"""
    try:
        logger.info(f"Chat request from client {request.session_id}")
        
        # Validate input
        if not request.message.strip():
            return ChatResponse(
                message="I need a question or input to respond to. Please provide a message.",
                context=[],
                session_id=request.session_id,
                status="error"
            )
            
        # Update file selection if provided in request
        if request.selected_files:
            manager.set_client_files(request.session_id, request.selected_files)
            
        # Process the chat request
        try:
            raw_resp_text, ctx_used, status = await asyncio.wait_for(
                _process_chat_logic(
                    session_id=request.session_id,
                    user_input=request.message,
                    selected_files=request.selected_files
                ),
                timeout=CONFIG.get("chat_timeout", 60.0)
            )
            
            # Clean response
            processed_response = common_utils.clean_response_language(CONFIG, raw_resp_text)
            
            # Save to memory
            memory = manager.get_client_memory(request.session_id)
            if memory:
                memory.save_context({"input": request.message}, {"output": processed_response})
                
            return ChatResponse(
                message=processed_response,
                context=ctx_used,
                session_id=request.session_id,
                status="success" if status.startswith("success") else status
            )
            
        except asyncio.TimeoutError:
            logger.error(f"Chat processing timed out for client {request.session_id}")
            return ChatResponse(
                message=f"Chat processing timed out after {CONFIG.get('chat_timeout', 60.0)} seconds. Please try again.",
                context=[],
                session_id=request.session_id,
                status="timeout"
            )
            
    except Exception as e:
        logger.error(f"Error in chat handling for client {request.session_id}: {str(e)}", exc_info=True)
        return ChatResponse(
            message="Sorry, I encountered an error processing your request. Please try again.",
            context=[],
            session_id=request.session_id,
            status="error"
        )

@app.post("/file-selection", summary="Update Selected Files")
async def file_selection_endpoint(request: FileSelectionRequest):
    """Update the file selection for a session"""
    try:
        session_id = request.session_id
        selected_files = request.files
        
        # Update session file selection
        manager.set_client_files(session_id, selected_files)
        
        # Clear existing content and embeddings for this session
        manager.set_client_content(session_id, {})
        manager.set_client_embeddings(session_id, {})
        
        logger.info(f"Updated file selection for session {session_id}: {selected_files}")
        
        # Process documents in background
        processing_tasks = []
        for doc_id in selected_files:
            processing_tasks.append(
                _trigger_doc_processing_chain(doc_id, session_id)
            )
        
        # Start background task for processing immediately
        if processing_tasks:
            # Create a single task that processes all documents sequentially
            async def process_all_documents():
                for doc_id in selected_files:
                    success, step = await _trigger_doc_processing_chain(doc_id, session_id)
                    logger.info(f"Document {doc_id} processing result: success={success}, step={step}")
            
            # Start the processing task
            asyncio.create_task(process_all_documents())
            
        return {
            "status": "updated",
            "session_id": session_id,
            "selected_files": selected_files,
            "message": "Files selected and processing started"
        }
        
    except Exception as e:
        logger.error(f"Error updating file selection: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to update file selection: {str(e)}")

@app.get("/files", summary="List Available Files")
async def get_files_endpoint():
    """Get a list of all available files in the vault"""
    common_utils.initialize_vault_directory(CONFIG)
    files = common_utils.get_vault_files(CONFIG)
    return {"files": files}

@app.post("/upload", summary="Upload a File")
async def upload_file_endpoint(
    file: UploadFile = File(...),
    user_description: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None), # session_id is kept for potential future use or logging
    pipeline_preference: str = Form("visual")
):
    """Upload a file for processing"""
    temp_file_path = None
    original_filename = file.filename or "unknown_file"
    logger.info(f"Upload request for: '{original_filename}', Preferred Pipeline: '{pipeline_preference}'")
    
    try:
        safe_original_filename = os.path.basename(original_filename)
        file_ext = os.path.splitext(safe_original_filename)[1].lower().lstrip('.')
        if not file_ext:
            file_ext = "unknown"
            
        # Validate pipeline preference
        if pipeline_preference not in ["visual", "textual"]:
            logger.warning(f"Invalid pipeline preference '{pipeline_preference}' for '{original_filename}'. Defaulting to 'visual'.")
            pipeline_preference = "visual"

        # File type validation based on pipeline
        if pipeline_preference == "visual" and file_ext not in ["pdf", "png", "jpg", "jpeg", "tiff"]:
            detail_msg = f"Visual pipeline for '.{file_ext}' not supported. Supported: pdf, png, jpg, jpeg, tiff."
            logger.warning(f"Upload failed for '{original_filename}': {detail_msg}")
            raise HTTPException(status_code=400, detail=detail_msg)
        
        # Add more specific validation for textual pipeline if needed, e.g.
        # if pipeline_preference == "textual" and file_ext in ["exe", "zip"]:
        #     raise HTTPException(status_code=400, detail=f"Textual pipeline does not support '.{file_ext}' files.")

        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        temp_file_path = os.path.join(temp_dir, f"temp_{uuid.uuid4().hex}_{safe_original_filename}")
        
        bytes_written = 0
        try:
            with open(temp_file_path, "wb") as temp_f:
                while content_chunk := await file.read(8 * 1024 * 1024): # Read in 8MB chunks
                    temp_f.write(content_chunk)
                    bytes_written += len(content_chunk)
        except Exception as e_write_temp:
            logger.error(f"Error writing temp file for '{original_filename}': {e_write_temp}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Server error saving temporary file: {str(e_write_temp)[:100]}")

        if bytes_written == 0:
            if temp_file_path and os.path.exists(temp_file_path): os.remove(temp_file_path)
            logger.warning(f"Upload failed for '{original_filename}': Uploaded file is empty.")
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        
        logger.info(f"Saved temporary file {temp_file_path} for {original_filename}, size: {bytes_written} bytes")

        # PDF validation for visual pipeline (if PyMuPDF is available)
        if pipeline_preference == "visual" and file_ext == "pdf":
            if not PYMUPDF_AVAILABLE:
                logger.error("PyMuPDF (fitz) is not available, but visual PDF upload was attempted.")
                raise HTTPException(status_code=501, detail="Server library PyMuPDF not available for visual PDF processing.")
            try:
                pdf_doc = fitz.open(temp_file_path)
                if not pdf_doc.is_pdf: # type: ignore
                    pdf_doc.close()
                    raise ValueError("File is not a valid PDF.")
                if pdf_doc.page_count == 0:
                    pdf_doc.close()
                    raise ValueError("PDF contains 0 pages.")
                if pdf_doc.is_encrypted and not pdf_doc.authenticate(""): # type: ignore
                    pdf_doc.close()
                    raise ValueError("PDF is encrypted and cannot be processed.")
                pdf_doc.close()
            except Exception as pdf_val_err:
                if temp_file_path and os.path.exists(temp_file_path): os.remove(temp_file_path)
                logger.warning(f"PDF validation failed for '{original_filename}': {pdf_val_err}")
                raise HTTPException(status_code=400, detail=f"Invalid or unsupported PDF: {str(pdf_val_err)}")
            
        # Orchestrate file processing
        try:
            success, doc_id, msg = await orchestrate_file_processing(
                CONFIG, temp_file_path, safe_original_filename, file_ext,
                user_description, pipeline_preference
            )
            
            if success and doc_id:
                final_meta = common_utils.get_specific_doc_metadata(CONFIG, doc_id) or {}
                if ollama_client: # Trigger AI metadata generation if successful and client available
                    asyncio.create_task(generate_ai_metadata_and_update(CONFIG, doc_id, ollama_client, common_utils))
                    
                return JSONResponse(status_code=200, content={
                    "message": f"'{original_filename}' upload successful. Initial processing as '{pipeline_preference}' initiated. {msg}",
                    "doc_id": doc_id,
                    "metadata": final_meta
                })
            else:
                logger.error(f"File processing orchestration failed for '{original_filename}': {msg}")
                raise HTTPException(status_code=500, detail=f"Server error during processing: {msg}")
        except Exception as e_orch:
            logger.error(f"Exception in orchestrate_file_processing for '{original_filename}': {str(e_orch)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error during file processing orchestration: {str(e_orch)}")
            
    except HTTPException:
        raise # Re-raise HTTPException directly
    except Exception as e_main_upload:
        logger.error(f"Unexpected error uploading file '{original_filename}': {str(e_main_upload)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected error during upload: {str(e_main_upload)[:100]}")
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logger.debug(f"Removed temporary file: {temp_file_path}")
            except Exception as e_rem_final:
                logger.error(f"Error removing temp file {temp_file_path} in finally block: {e_rem_final}")

async def orchestrate_file_processing(
    config_dict: Dict,
    temp_file_path: str,
    original_filename: str,
    file_extension: str,
    user_description: Optional[str],
    pipeline_preference: str
) -> Tuple[bool, Optional[str], str]:
    """Orchestrates the initial file processing after upload"""
    
    logger.info(f"Orchestrating processing for '{original_filename}' (ext: '{file_extension}') as '{pipeline_preference}'")
    doc_id = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
    logger.info(f"Generated doc_id: {doc_id} for '{original_filename}'")

    metadata_tags = [file_extension.lower()]
    processed_path_for_meta = ""
    initial_pipeline_step_status = ""
    
    if pipeline_preference == "visual":
        metadata_tags.append("visual_pipeline")
        processed_path_for_meta = os.path.join(config_dict["processed_docs_subdir"], doc_id)
        initial_pipeline_step_status = "visual_images_pending"
    elif pipeline_preference == "textual":
        metadata_tags.append("textual_pipeline")
        # Anticipate a common pattern for textual processed files, or use a placeholder
        processed_path_for_meta = f"{doc_id}_textual_content.txt" # Example placeholder
        initial_pipeline_step_status = "textual_file_pending"
    else:
        # This should be caught by upload endpoint, but defensive check
        return False, doc_id, f"Invalid pipeline preference: {pipeline_preference}"
    
    metadata_extra = {
        "original_filename": original_filename,
        "doc_type_extension": file_extension,
        "pipeline_type": pipeline_preference,
        "user_provided_description": user_description,
        "page_count": 0, # Will be updated after processing
        "processed_path": processed_path_for_meta,
        "pipeline_step": initial_pipeline_step_status
    }
    effective_description = user_description if user_description else f"{pipeline_preference.capitalize()} doc: {original_filename}"
    
    if not common_utils.add_file_to_vault(config_dict, doc_id, effective_description, metadata_tags, metadata_extra):
        logger.error(f"Failed to save initial document metadata for '{original_filename}' (ID: {doc_id}).")
        return False, doc_id, "Failed to save initial document metadata."
    
    initial_step_success = False
    msg_pipeline_step = ""
    page_count_val = 0
    
    if pipeline_preference == "visual":
        logger.info(f"Visual pipeline: Starting image extraction for doc_id: {doc_id}")
        processed_doc_base_visual_full_path = os.path.join(config_dict["vault_directory"], processed_path_for_meta)
        output_image_dir_visual = os.path.join(processed_doc_base_visual_full_path, "pages")
        target_img_format = config_dict.get('image_format', 'png').lower().lstrip('.')

        try:
            os.makedirs(output_image_dir_visual, exist_ok=True)
        except OSError as e_mkdir:
            msg_pipeline_step = f"Server error creating visual directory structure: {e_mkdir}"
            logger.error(msg_pipeline_step, exc_info=True)
            common_utils.update_file_metadata(config_dict, doc_id, metadata_extra={"pipeline_step": "visual_image_extraction_failed_mkdir"})
            return False, doc_id, msg_pipeline_step

        if file_extension == "pdf":
            if not PYMUPDF_AVAILABLE: # Should have been checked by upload endpoint too
                msg_pipeline_step = "PyMuPDF not available for visual PDF processing."
                common_utils.update_file_metadata(config_dict, doc_id, metadata_extra={"pipeline_step": "visual_image_extraction_failed_pymupdf_missing"})
                return False, doc_id, msg_pipeline_step
            try:
                pdf_doc_obj = fitz.open(temp_file_path)
                page_count_val = pdf_doc_obj.page_count
                extracted_pages_count = 0
                for i in range(page_count_val):
                    page = pdf_doc_obj.load_page(i)
                    dpi = config_dict.get("image_dpi", 300)
                    scale = dpi / 72.0
                    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False) # type: ignore
                    img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n) # type: ignore
                    
                    img_to_save = img_array
                    if pix.n == 4 and target_img_format != 'png': # BGRA to BGR if not saving as PNG (which supports alpha)
                        img_to_save = cv2.cvtColor(img_array, cv2.COLOR_BGRA2BGR)
                    elif pix.n == 1: # Grayscale to BGR
                        img_to_save = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
                    
                    output_page_path = os.path.join(output_image_dir_visual, f"page_{i+1:04d}.{target_img_format}")
                    save_params = []
                    if target_img_format in ['jpg', 'jpeg']:
                        save_params = [cv2.IMWRITE_JPEG_QUALITY, config_dict.get("jpeg_quality", 90)]
                    elif target_img_format == 'png':
                        save_params = [cv2.IMWRITE_PNG_COMPRESSION, config_dict.get("png_compression", 3)]
                    
                    if cv2.imwrite(output_page_path, img_to_save, save_params):
                        extracted_pages_count += 1
                    else:
                        logger.warning(f"Failed to save page {i+1} for {doc_id} as {target_img_format}")
                pdf_doc_obj.close()
                
                if extracted_pages_count > 0:
                    initial_step_success = True
                    msg_pipeline_step = f"Extracted {extracted_pages_count} pages from PDF as .{target_img_format}."
                    common_utils.update_file_metadata(config_dict, doc_id, metadata_extra={"pipeline_step": "visual_images_extracted", "page_count": extracted_pages_count, "extracted_pages": extracted_pages_count})
                else:
                    msg_pipeline_step = f"Failed to extract any pages from {page_count_val}-page PDF."
                    common_utils.update_file_metadata(config_dict, doc_id, metadata_extra={"pipeline_step": "visual_images_extraction_failed_zero_pages", "page_count": page_count_val})
            except Exception as e_pdf_proc:
                msg_pipeline_step = f"Visual PDF processing error: {e_pdf_proc}"
                logger.error(msg_pipeline_step, exc_info=True)
                common_utils.update_file_metadata(config_dict, doc_id, metadata_extra={"pipeline_step": f"visual_images_extraction_failed_{e_pdf_proc.__class__.__name__}"})

        elif file_extension in ["png", "jpg", "jpeg", "tiff"]:
            page_count_val = 1
            try:
                target_image_path = os.path.join(output_image_dir_visual, f"page_0001.{target_img_format}")
                img_array = cv2.imread(temp_file_path, cv2.IMREAD_UNCHANGED)
                if img_array is None: raise ValueError("Failed to read image with OpenCV.")
                
                img_to_save = img_array
                if len(img_array.shape) == 2: # Grayscale
                    img_to_save = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
                elif img_array.shape[2] == 4 and target_img_format != 'png': # BGRA
                    img_to_save = cv2.cvtColor(img_array, cv2.COLOR_BGRA2BGR)

                save_params = []
                if target_img_format in ['jpg', 'jpeg']:
                    save_params = [cv2.IMWRITE_JPEG_QUALITY, config_dict.get("jpeg_quality", 90)]
                elif target_img_format == 'png':
                    save_params = [cv2.IMWRITE_PNG_COMPRESSION, config_dict.get("png_compression", 3)]

                if cv2.imwrite(target_image_path, img_to_save, save_params):
                    initial_step_success = True
                    msg_pipeline_step = f"Visual image processed and saved as .{target_img_format}."
                    common_utils.update_file_metadata(config_dict, doc_id, metadata_extra={"pipeline_step": "visual_images_extracted", "page_count": page_count_val})
                else:
                    raise RuntimeError(f"Failed to save image to {target_image_path}")
            except Exception as e_img_proc:
                msg_pipeline_step = f"Visual image processing error: {e_img_proc}"
                logger.error(msg_pipeline_step, exc_info=True)
                common_utils.update_file_metadata(config_dict, doc_id, metadata_extra={"pipeline_step": f"visual_images_extraction_failed_{e_img_proc.__class__.__name__}"})
        else:
            msg_pipeline_step = f"Unsupported file type '{file_extension}' for visual pipeline."
            common_utils.update_file_metadata(config_dict, doc_id, metadata_extra={"pipeline_step": "visual_images_extraction_failed_unsupported_type"})

    elif pipeline_preference == "textual":
        logger.info(f"Textual pipeline: Preparing document {doc_id}")
        try:
            # textual_pipeline.prepare_textual_document should handle various types (txt, pdf, docx)
            # and save the extracted text to a file in the vault, returning its name (relative to vault).
            processed_text_filename_rel_vault = await asyncio.to_thread(
                textual_pipeline.prepare_textual_document,
                config_dict,
                temp_file_path, # Path to the temporary uploaded file
                doc_id,
                original_filename
            )
            if processed_text_filename_rel_vault:
                initial_step_success = True
                msg_pipeline_step = f"Textual document content extracted to '{processed_text_filename_rel_vault}'.",
                # Update metadata with the actual path of the processed textual file and page count (if available from prepare_textual_document)
                # For now, assume page_count is 1 for textual unless prepare_textual_document provides it.
                meta_update_textual = {"pipeline_step": "textual_file_prepared", "processed_path": processed_text_filename_rel_vault, "page_count": 1}
                # If prepare_textual_document can return page_count, update it here.
                # e.g., if it returns (filename, page_count_from_textual_prep):
                # meta_update_textual["page_count"] = page_count_from_textual_prep
                common_utils.update_file_metadata(config_dict, doc_id, metadata_extra=meta_update_textual)
            else:
                msg_pipeline_step = f"Textual pipeline failed to prepare document '{original_filename}'."
                logger.error(msg_pipeline_step + f" (Doc ID: {doc_id})")
                common_utils.update_file_metadata(config_dict, doc_id, metadata_extra={"pipeline_step": "textual_preparation_failed"})
        except Exception as e_textual_prep:
            msg_pipeline_step = f"Textual pipeline preparation error: {e_textual_prep}"
            logger.error(msg_pipeline_step, exc_info=True)
            common_utils.update_file_metadata(config_dict, doc_id, metadata_extra={"pipeline_step": f"textual_preparation_failed_{e_textual_prep.__class__.__name__}"})
    
    if initial_step_success:
        logger.info(f"Initial processing successful for doc_id '{doc_id}' ('{original_filename}') as '{pipeline_preference}'. Msg: {msg_pipeline_step}")
        return True, doc_id, msg_pipeline_step
    else:
        logger.error(f"Initial processing failed for doc_id '{doc_id}' ('{original_filename}') as '{pipeline_preference}'. Msg: {msg_pipeline_step}")
        return False, doc_id, msg_pipeline_step

async def generate_ai_metadata_and_update(
    config_dict: Dict,
    doc_id: str,
    current_ollama_client: Optional[OpenAI],
    common_utils_ref: Any # Assuming common_utils_ref is the common_utils module
):
    """Generate AI metadata for a document and update it."""
    if not current_ollama_client:
        logger.warning(f"AI Meta: Ollama client not available for doc_id {doc_id}. Skipping.")
        return
        
    logger.info(f"AI Meta Task: Starting for doc_id: {doc_id}")
    doc_meta = common_utils_ref.get_specific_doc_metadata(config_dict, doc_id)
    if not doc_meta:
        logger.error(f"AI Meta: No metadata found for doc_id {doc_id}.")
        return

    content_sample = ""
    pipeline_type = doc_meta.get("pipeline_type")
    original_filename = doc_meta.get("original_filename", doc_id)

    if pipeline_type == "textual":
        processed_textual_file_rel_path = doc_meta.get("processed_path")
        if processed_textual_file_rel_path:
            # processed_path is relative to vault_directory
            full_path = os.path.join(config_dict["vault_directory"], processed_textual_file_rel_path)
            if os.path.exists(full_path):
                try:
                    with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                        content_sample = f.read(8000) # Read up to 8000 characters for summary
                    logger.info(f"AI Meta: Read content sample from '{full_path}' for textual doc '{original_filename}'.")
                except Exception as e_read_text:
                    logger.error(f"AI Meta: Error reading textual file '{full_path}': {e_read_text}")
            else:
                logger.warning(f"AI Meta: Processed textual file '{full_path}' not found for doc '{original_filename}'.")
        else:
            logger.warning(f"AI Meta: No 'processed_path' found in metadata for textual doc '{original_filename}'.")
    elif pipeline_type == "visual":
        # For visual, AI description from content is typically harder at upload time without OCR results.
        # This could be enhanced later to use OCR results if available.
        # For now, we might rely more on user_provided_description or filename.
        # Or, if OCR results are available (e.g., ocr_results.json), extract text from there.
        # This example keeps it simple for visual, focusing on textual content sample.
        logger.info(f"AI Meta: Visual doc '{original_filename}'. AI description from content sample not implemented for visual at this stage. User description will be prioritized.")
        # If you want to attempt to get some text from visual, you'd need to access OCR results here.
        # For example, load `ocr_results.json` for this doc_id and concatenate some text.
        # This is a placeholder for more advanced visual content sampling:
        # content_sample = common_utils_ref.get_visual_content_sample(config_dict, doc_id) # Hypothetical function

    if content_sample.strip():
        try:
            # Assuming common_utils_ref.generate_file_metadata can take content_sample
            ai_desc, ai_tags = await common_utils_ref.generate_file_metadata(
                config_dict, content_sample, current_ollama_client, original_filename
            )
            
            if ai_desc or ai_tags:
                current_description = doc_meta.get("description", "") # Get existing full description
                user_provided_desc = doc_meta.get("user_provided_description", "")

                final_desc = user_provided_desc # Prioritize user description
                if not final_desc and ai_desc: # If no user desc, use AI desc
                    final_desc = ai_desc
                elif final_desc and ai_desc and final_desc == f"{pipeline_type.capitalize()} doc: {original_filename}":
                    # If current description is the generic default, prefer AI description
                    final_desc = ai_desc
                
                # Combine tags: existing + AI, ensuring uniqueness and sorted
                existing_tags = doc_meta.get("tags", [])
                new_ai_tags = ai_tags if isinstance(ai_tags, list) else []
                combined_tags = sorted(list(set(existing_tags + new_ai_tags)))
                
                update_payload = {
                    "ai_generated_description_snippet": ai_desc, # Store the raw AI desc snippet
                    "ai_generated_tags": new_ai_tags, # Store the raw AI tags
                    "ai_metadata_generated_at": datetime.now().isoformat()
                }
                # Update the main description and tags in metadata
                common_utils_ref.update_file_metadata(config_dict, doc_id, description=final_desc, tags=combined_tags, metadata_extra=update_payload)
                logger.info(f"AI Meta: Updated metadata for '{original_filename}' (ID: {doc_id}) with AI insights.")
            else:
                logger.warning(f"AI Meta: No new description or tags generated by AI for '{original_filename}'.")
        except Exception as e_ai_gen:
            logger.error(f"AI Meta: Error during AI metadata generation for '{original_filename}': {e_ai_gen}", exc_info=True)
    elif pipeline_type == "textual": # Only log if textual and no content sample
        logger.warning(f"AI Meta: Content sample for textual doc '{original_filename}' was empty. Skipping AI metadata generation.")
    
    logger.info(f"AI Meta task finished for '{original_filename}' (ID: {doc_id}).")

# --- Main Execution Block ---
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    SERVER_HOST = os.environ.get("HOST", "0.0.0.0")
    SERVER_PORT = int(os.environ.get("PORT", 3000))  # Using port 3000 to match frontend
    RELOAD_DEV_MODE = os.environ.get("RELOAD", "true").lower() == "true"
    
    print("--- Starting Chatbot API Server ---")
    print(f"Host: {SERVER_HOST}, Port: {SERVER_PORT}, Reload: {RELOAD_DEV_MODE}")
    print(f"Vault: {os.path.abspath(CONFIG['vault_directory'])}")
    print(f"Log: {os.path.abspath(CONFIG['log_file'])} ({CONFIG['log_level']})")
    print(f"GPU: {has_gpu} ({device})")
    
    # Check if YOLO model exists and download if not
    yolo_model_path = os.path.join(os.path.abspath(CONFIG["vault_directory"]), CONFIG["yolov8_layout_model_path"])
    if not os.path.exists(yolo_model_path):
        print(f"⚠️ YOLO model not found at: {yolo_model_path}. Attempting to download...")
        asyncio.run(download_yolo_model(CONFIG))
    else:
        print(f"✓ YOLO model found at: {yolo_model_path}")
    
    if not PYMUPDF_AVAILABLE:
        print("⚠️ PyMuPDF (fitz) not available. Visual PDF processing WILL FAIL.")
    
    uvicorn.run(
        "endpoints:app",
        host=SERVER_HOST,
        port=SERVER_PORT,
        reload=RELOAD_DEV_MODE,
        reload_dirs=["./"] if RELOAD_DEV_MODE else None,
        reload_includes=["*.py"] if RELOAD_DEV_MODE else None,
        reload_excludes=["*_client*.py", "temp_uploads/*", "response_tracking/*", "vault_files/*", "*.log", "vector_store/*", "__pycache__/*"] if RELOAD_DEV_MODE else None,
        log_level=CONFIG["log_level"].lower(),
        timeout_keep_alive=75
    )