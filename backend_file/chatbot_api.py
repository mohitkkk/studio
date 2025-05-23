import os
import json
import logging
import traceb
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
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File, UploadFile, Form, HTTPException, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.websockets import WebSocketState
from ultrO # <--- IMPORT YOLO
import chromadb
from chromadb.utils import embedding_functions
import fitz
import uvicorn

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

Format your response using standard Markdown syntax. Use bold for key terms, and bullet points for lists where appropriate.
Ensure the response is easy to read and directly answers the user's question based *only* on the context.
"""

# ... (existing imports and system_message definition) ...

CONFIG = {
    # --- General Vault and Core Settings (Keep) ---
    "vault_directory": "vault_files/",
    "vault_metadata": "vault_metadata.json",
    "processed_docs_subdir": "processed_docs",
    "vector_store_path": "vector_store",
    "vector_collection_name": "document_chunks",
    "ollama_model": "llama3:latest",
    "ollama_embedding_model": "mxbai-embed-large:latest",
    "log_file": "chatbot.log",
    "log_level": "DEBUG",
    "image_dpi": 300, # Still used for initial PDF to image conversion
    "image_format": "png", # Still used for initial PDF to image conversion and saving crops

    # --- Output File Names (Keep) ---
    "layout_analysis_file": "layout_analysis.json", # Will now store YOLOv8 regions
    "ocr_results_file": "ocr_results.json",       # Will now store OCR text from YOLOv8 regions & image/table info

    # --- Audit Settings (Keep) ---
    "text_chunk_output_dir": "extracted_text_chunks", # Directory for text chunk audit files
    "create_plain_text_audit": True, # Toggle for creating text chunk audit files

    "embedding_batch_size": 16, # Still used for ChromaDB batching

    # --- Retrieval Tuning (Keep) ---
    "visual_context_results": 15, # Number of chunks passed to the LLM
    "chroma_retrieval_multiplier": 5, # Multiplier for initial Chroma retrieval quantity

    # --- Textual Pipeline Settings (Keep as they are separate) ---
    "textual_max_chunk_size": 800,
    "textual_chunk_overlap": 0,
    "textual_top_k_per_doc": 5,
    "textual_similarity_threshold": 0.45,
    "textual_context_results_limit": 15,
    "heading_keywords": ["section", "chapter", "part", "introduction", "overview", "summary", "conclusion", "references", "appendix", "table of contents", "index", "glossary", "discussion", "results", "methodology", "procedure", "specifications", "requirements", "features", "instructions"],
    "textual_heading_keywords": ["section", "chapter", "introduction", "conclusion"],

    "yolov8_layout_model_path": 'vault_files/model/yolov8x-seg.pt', # <--- ADD THIS NEW KEY
    # Label map: Maps YOLOv8 output class IDs (integers) to the category names (strings) they represent.
    # This MUST match the training of the specific YOLOv8-seg model you are using (PubLayNet standard labels).
    "yolov8_label_map": {0: "text", 1: "title", 2: "list", 3:"table", 4:"figure"}, # <--- ADD THIS NEW KEY
    "yolov8_score_threshold": 0.5, # Minimum confidence score for a detected region to be kept (YOLOv8 'conf' parameter in predict)
    "layoutparser_text_categories": ["text", "title", "list", "caption"], # Which YOLOv8 categories should be OCR'd and treated as text items
    "layoutparser_image_categories": ["figure"], # Which YOLOv8 categories should be saved as image files
    "layoutparser_table_categories": ["table"], # Which YOLOv8 categories should be saved as image files (or processed with table tools)
    "chunk_vertical_proximity_threshold_px": 15, # Max vertical pixel gap between text regions to potentially merge them
    "chunk_vertical_proximity_threshold_ratio_of_height": 0.5, # Max vertical gap as % of taller region height to potentially merge
    "chunk_horizontal_overlap_ratio": 0.5, # Minimum horizontal overlap ratio to consider regions in the same conceptual column/flow for merging
    # Categories that, when encountered, typically indicate the start of a new distinct chunk (e.g., title, caption)
    "always_new_chunk_categories": ["title", "caption"], # Which YOLOv8 categories should always start a new chunk
    # Categories that can be merged with adjacent items of the same or other mergeable categories
    "merge_categories": ["text", "list"], # Which YOLOv8 categories can be merged based on proximity/alignment
    "min_chunk_text_length": 10, # Minimum character length for a final chunk to be created by the chunking function

    # --- Tesseract Settings (Still used for OCR on Crops in Step 4) (Keep) ---
    "tesseract_lang": "eng", # Language for Tesseract OCR on cropped regions
    "tesseract_ocr_crop_psm": 6, # PSM to use for Tesseract OCR *on cropped text regions*. PSM 6 or 7 are often good for crops.
    "tesseract_timeout_block": 30, # Timeout for individual block-level OCR calls

    # --- Image Processing & Saving Settings (Keep) ---
    "text_crop_margin": 2, # Margin for cropping text regions before OCR
    "image_crop_margin": 5, # Margin for cropping image/table regions before saving
    "jpeg_quality": 90, # Quality for saving JPEG crops
    "png_compression": 3, # Compression level for saving PNG crops
    "perform_ocr_on_diagrams": True, # Toggle OCR on image crops (using diagram_ocr_psm)
    "diagram_ocr_psm": 11, # PSM for OCR on diagram crops (e.g., PSM 11 for sparse text in diagrams)

    "min_chunk_length_for_indexing": 5, # Minimum character length for a chunk to be sent to the vector index (final filter in Step 5)

    # --- Preprocessing Settings (Keep) ---
    "enable_deskewing": True, # Still used in preprocess_image_opencv before passing to YOLOv8
    "enable_binarization_visual": True, # Still used in preprocess_image_opencv before passing to YOLOv

    # --- Fallback & Debug Settings (Keep) ---
    "fallback_page_width": 1000, # Used if image dimensions can't be read
    "fallback_page_height": 1400,
    "debug_layout_visualization": True, # Toggle YOLOv8 debug image visualization
    "debug_layout_output_subdir": "debug_layout", # Directory for YOLOv8 debug images
    "debug_mode": True, # General debug toggle
    "debug_target_doc_id": "1747374358_2fb6b580", # Specific doc for debug logging
    "response_tracking_dir": "response_tracking", # Directory for response tracking files

    # --- Keep existing dependency/device info (Used by YOLOv8) ---
    "use_gpu": False, # Keep based on your system check result
    "device": 'cpu', # Keep based on your system check result, will be passed to YOLOv8 predict implicitly/explicitly
    "chat_timeout": 60.0 # Keep chat timeout
}

current_handlers = logger.handlers[:]
for handler in current_handlers: logger.removeHandler(handler)
logging.basicConfig(
    level=getattr(logging, CONFIG["log_level"].upper(), logging.INFO),
    filename=CONFIG["log_file"],
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.info("--- Main API Logging Re-initialized with CONFIG Settings ---")

device, has_gpu = common_utils.setup_device(CONFIG)
CONFIG["use_gpu"] = has_gpu
CONFIG["device"] = device

DEPENDENCY_STATUS = common_utils.check_and_install_dependencies(CONFIG)
PYMUPDF_AVAILABLE = DEPENDENCY_STATUS.get("fitz", False)

ollama_client: Optional[OpenAI] = None
chroma_client: Optional[chromadb.ClientAPI] = None
chroma_collection: Optional[chromadb.api.models.Collection.Collection] = None
ollama_ef: Optional[embedding_functions.OllamaEmbeddingFunction] = None

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
    
    logger.info("--- Lifespan Startup: Yielding to Application ---")
    yield
    logger.info("--- Lifespan Shutdown: Initiated ---")
    logger.info("--- Lifespan Shutdown: Complete ---")

app = FastAPI(
    title="Chatbot API",
    description="Toggle-based Visual and Textual RAG Chatbot API",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.client_file_selections: Dict[str, List[str]] = {}
        self.client_embeddings: Dict[str, Dict[str, torch.Tensor]] = {}
        self.client_content: Dict[str, Dict[str, List[str]]] = {}
        self.client_memory: Dict[str, ConversationBufferMemory] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.client_file_selections[client_id] = []
        self.client_embeddings[client_id] = {}
        self.client_content[client_id] = {}
        self.client_memory[client_id] = ConversationBufferMemory(memory_key="history", return_messages=True)
        logger.info(f"Client connected & session initialized: {client_id}")

    def disconnect(self, client_id: str):
        self.active_connections.pop(client_id, None)
        self.client_file_selections.pop(client_id, None)
        self.client_embeddings.pop(client_id, None)
        self.client_content.pop(client_id, None)
        self.client_memory.pop(client_id, None)
        logger.info(f"Client disconnected & session cleaned: {client_id}")

    async def send_json(self, client_id: str, data: dict):
        websocket = self.active_connections.get(client_id)
        if websocket and websocket.client_state == WebSocketState.CONNECTED:
            try:
                await websocket.send_json(data)
            except Exception as e:
                logger.error(f"Error sending JSON to {client_id}: {e}")
                self.disconnect(client_id)
    
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
        return self.client_memory.get(client_id)
    def cleanup_file_data(self, filename_is_doc_id: str):
        for client_id in list(self.active_connections.keys()):
            if filename_is_doc_id in self.client_file_selections.get(client_id, []):
                self.client_file_selections[client_id].remove(filename_is_doc_id)
            self.client_embeddings.get(client_id, {}).pop(filename_is_doc_id, None)
            self.client_content.get(client_id, {}).pop(filename_is_doc_id, None)

manager = ConnectionManager()

class FileSelection(BaseModel):
    files: List[str]
    client_id: Optional[str] = None
class ChatMessageModel(BaseModel):
    message: str
    client_id: Optional[str] = None

async def orchestrate_file_processing(
    config_dict: Dict,
    temp_file_path: str,
    original_filename: str,
    file_extension: str,
    user_description: Optional[str],
    pipeline_preference: str
) -> Tuple[bool, Optional[str], str]:
    logger.info(f"Orchestrating processing for '{original_filename}' (ext: '.{file_extension}') as '{pipeline_preference}' pipeline.")
    doc_id = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
    logger.info(f"Generated doc_id: {doc_id} for '{original_filename}'")

    metadata_tags = [file_extension.lower()]
    processed_path_for_meta = ""
    initial_step_success = False
    msg_pipeline_step = ""
    page_count_val = 0 # Initialize page count

    if pipeline_preference == "visual":
        metadata_tags.append("visual_pipeline")
        processed_doc_base_visual = os.path.join(config_dict["processed_docs_subdir"], doc_id)
        processed_path_for_meta = processed_doc_base_visual # Store the base dir for visual
        initial_pipeline_step_status = "visual_images_pending"
    elif pipeline_preference == "textual":
        metadata_tags.append("textual_pipeline")
        # For textual, we don't know the final filename yet, but can anticipate it or leave blank
        # Let's just store doc_id for now, will update later
        processed_path_for_meta = f"{doc_id}_textual.txt" # Placeholder filename
        initial_pipeline_step_status = "textual_file_pending"
    else:
        # This validation is also done in upload_file_http, but keep here for safety
        return False, doc_id, f"Invalid pipeline preference: {pipeline_preference}"

    # --- Save initial metadata ---
    metadata_extra = {
        "original_filename": original_filename,
        "doc_type_extension": file_extension,
        "pipeline_type": pipeline_preference,
        "user_provided_description": user_description,
        "page_count": page_count_val, # Initial page count (may update after processing)
        "processed_path": processed_path_for_meta, # Initial guess/placeholder
        "pipeline_step": initial_pipeline_step_status # Initial status
    }
    effective_description = user_description if user_description else f"{pipeline_preference.capitalize()} doc: {original_filename}"

    # Add file to vault, including initial metadata
    if not common_utils.add_file_to_vault(config_dict, doc_id, effective_description, metadata_tags, metadata_extra):
        logger.error(f"Failed to save initial document metadata for '{original_filename}' (ID: {doc_id}).")
        return False, doc_id, "Failed to save initial document metadata."

    # --- Perform Initial Pipeline Step ---
    if pipeline_preference == "visual":
        logger.info(f"Visual pipeline: Starting image extraction for doc_id: {doc_id}")
        processed_doc_base_visual_full_path = os.path.join(config_dict["vault_directory"], processed_doc_base_visual)
        output_image_dir_visual = os.path.join(processed_doc_base_visual_full_path, "pages")

        # Determine the target image format from config, default to png
        target_img_format = config_dict.get('image_format', 'png').lower().lstrip('.')
        if target_img_format not in ['png', 'jpg', 'jpeg', 'tiff']: # Add other formats if supported by cv2.imwrite
             logger.warning(f"Unsupported target image format '{target_img_format}' configured. Defaulting to png.")
             target_img_format = 'png'


        try:
            os.makedirs(processed_doc_base_visual_full_path, exist_ok=True)
            os.makedirs(output_image_dir_visual, exist_ok=True)
        except OSError as e_mkdir:
            msg_pipeline_step = f"Server error creating visual directory structure: {e_mkdir}"
            logger.error(msg_pipeline_step, exc_info=True)
            common_utils.update_file_metadata(config_dict, doc_id, metadata_extra={"pipeline_step": "visual_image_extraction_failed_mkdir"})
            return False, doc_id, msg_pipeline_step

        if file_extension == "pdf":
            if not PYMUPDF_AVAILABLE:
                msg_pipeline_step = "PyMuPDF not available for visual PDF processing."
                logger.error(msg_pipeline_step)
                common_utils.update_file_metadata(config_dict, doc_id, metadata_extra={"pipeline_step": "visual_image_extraction_failed_pymupdf_missing"})
                return False, doc_id, msg_pipeline_step

            try:
                pdf_doc_obj = fitz.open(temp_file_path)
                page_count_val = pdf_doc_obj.page_count
                if page_count_val == 0:
                    pdf_doc_obj.close()
                    raise ValueError("PDF has 0 pages")

                extracted_pages_count = 0
                for i in range(page_count_val):
                    page = pdf_doc_obj.load_page(i)
                    try:
                        dpi = config_dict.get("image_dpi", 300)
                        scale = dpi / 72.0
                        # Use get_pixmap with options for image format and quality if needed
                        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
                        
                        # Define output path using the TARGET format
                        output_page_path = os.path.join(output_image_dir_visual, f"page_{i+1:04d}.{target_img_format}")

                        # Save with appropriate parameters if needed (e.g. JPEG quality, PNG compression)
                        save_params = []
                        if target_img_format in ['jpg', 'jpeg']:
                             save_params = [cv2.IMWRITE_JPEG_QUALITY, config_dict.get("jpeg_quality", 90)]
                        elif target_img_format == 'png':
                             save_params = [cv2.IMWRITE_PNG_COMPRESSION, config_dict.get("png_compression", 3)]

                        # Convert pixmap to numpy array (BGRA -> BGR or Gray) and save using cv2.imwrite
                        # This ensures consistent output format and allows using cv2 save options
                        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                        if pix.n == 4: # Convert BGRA to BGR if it has alpha
                             img_array = cv2.cvtColor(img_array, cv2.COLOR_BGRA2BGR)
                        elif pix.n == 1: # Grayscale
                             img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR) # Convert to BGR for consistent saving

                        if cv2.imwrite(output_page_path, img_array, save_params):
                            extracted_pages_count += 1
                        else:
                            logger.error(f"Failed to save processed image for page {i+1} of {doc_id} to {output_page_path}.")


                    except Exception as e_page_pixmap:
                        logger.error(f"Error processing or saving page {i+1} of {doc_id}: {e_page_pixmap}", exc_info=True)
                        # Continue with other pages

                pdf_doc_obj.close()
                
                # Check if at least one page was successfully extracted and saved
                if extracted_pages_count > 0:
                    initial_step_success = True
                    msg_pipeline_step = f"Extracted and saved {extracted_pages_count} visual pages as .{target_img_format}."
                    logger.info(msg_pipeline_step + f" (Doc ID: {doc_id})")
                    # Update metadata *after* success, using the actual number of pages extracted
                    common_utils.update_file_metadata(config_dict, doc_id, metadata_extra={"pipeline_step": "visual_images_extracted", "page_count": extracted_pages_count})
                else:
                    # 0 pages extracted, or all page extractions failed
                    msg_pipeline_step = "Visual PDF processing failed: 0 pages extracted or error during all page processing."
                    logger.error(msg_pipeline_step + f" (Doc ID: {doc_id})")
                    initial_step_success = False
                    common_utils.update_file_metadata(config_dict, doc_id, metadata_extra={"pipeline_step": "visual_image_extraction_failed_zero_pages"})

            except Exception as e_pdf:
                msg_pipeline_step = f"Visual PDF processing error: {e_pdf}"
                logger.error(msg_pipeline_step, exc_info=True)
                initial_step_success = False # Explicitly mark failure
                common_utils.update_file_metadata(config_dict, doc_id, metadata_extra={"pipeline_step": "visual_image_extraction_failed_exception"})


        elif file_extension in ["png", "jpg", "jpeg", "tiff"]: # Added tiff if you might handle it
            page_count_val = 1
            try:
                # Ensure the target directory exists
                os.makedirs(output_image_dir_visual, exist_ok=True)
                # Define the target path for the single image, using the configured format
                target_image_path = os.path.join(output_image_dir_visual, f"page_0001.{target_img_format}")

                # Use OpenCV to read and save the image in the target format
                img_array = cv2.imread(temp_file_path, cv2.IMREAD_UNCHANGED) # Read with unchanged flags to handle alpha
                if img_array is None:
                     raise ValueError(f"Failed to read image file with OpenCV: {temp_file_path}")

                # Convert to BGR if grayscale or includes alpha for consistent saving
                if len(img_array.shape) == 2: # Grayscale
                     img_array_to_save = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
                elif img_array.shape[2] == 4: # BGRA
                     img_array_to_save = cv2.cvtColor(img_array, cv2.COLOR_BGRA2BGR)
                else: # BGR
                     img_array_to_save = img_array

                # Save with appropriate parameters
                save_params = []
                if target_img_format in ['jpg', 'jpeg']:
                     save_params = [cv2.IMWRITE_JPEG_QUALITY, config_dict.get("jpeg_quality", 90)]
                elif target_img_format == 'png':
                     save_params = [cv2.IMWRITE_PNG_COMPRESSION, config_dict.get("png_compression", 3)]
                # Add TIFF or other format parameters here if needed

                if cv2.imwrite(target_image_path, img_array_to_save, save_params):
                    initial_step_success = True
                    msg_pipeline_step = f"Visual image processed and saved as .{target_img_format}."
                    logger.info(msg_pipeline_step + f" (Doc ID: {doc_id})")
                    # Update metadata *after* success
                    common_utils.update_file_metadata(config_dict, doc_id, metadata_extra={"pipeline_step": "visual_images_extracted", "page_count": page_count_val})
                else:
                     raise RuntimeError(f"Failed to save image to {target_image_path} using OpenCV.")


            except Exception as e_img_proc:
                msg_pipeline_step = f"Visual image processing error: {e_img_proc}"
                logger.error(msg_pipeline_step, exc_info=True)
                initial_step_success = False # Explicitly mark failure
                common_utils.update_file_metadata(config_dict, doc_id, metadata_extra={"pipeline_step": "visual_image_extraction_failed_processing"})


        else:
            msg_pipeline_step = f"Unsupported file type '{file_extension}' for visual pipeline image extraction."
            logger.warning(msg_pipeline_step)
            initial_step_success = False # Explicitly mark failure
            common_utils.update_file_metadata(config_dict, doc_id, metadata_extra={"pipeline_step": "visual_image_extraction_failed_unsupported_type"})

    elif pipeline_preference == "textual":
        logger.info(f"Textual pipeline: Starting text preparation for doc_id: {doc_id}")
        try:
            processed_text_filename_vault = await asyncio.to_thread(
                textual_pipeline.prepare_textual_document,
                config_dict,
                temp_file_path,
                doc_id,
                original_filename
            )
            if processed_text_filename_vault:
                initial_step_success = True
                msg_pipeline_step = f"Textual document processed and saved as '{processed_text_filename_vault}'."
                logger.info(msg_pipeline_step + f" (Doc ID: {doc_id})")
                # Update metadata *after* success, including the actual processed path
                common_utils.update_file_metadata(config_dict, doc_id, metadata_extra={"pipeline_step": "textual_file_prepared", "page_count": 1, "processed_path": processed_text_filename_vault})
            else:
                # textual_pipeline.prepare_textual_document should log the error internally
                msg_pipeline_step = f"Textual pipeline failed to prepare document '{original_filename}'."
                logger.error(msg_pipeline_step + f" (Doc ID: {doc_id})")
                initial_step_success = False # Explicitly mark failure
                # Update metadata to reflect failure
                common_utils.update_file_metadata(config_dict, doc_id, metadata_extra={"pipeline_step": "textual_preparation_failed"})

        except Exception as e_textual_prep:
            msg_pipeline_step = f"Textual pipeline preparation encountered an exception: {e_textual_prep}"
            logger.error(msg_pipeline_step + f" (Doc ID: {doc_id})", exc_info=True)
            initial_step_success = False # Explicitly mark failure
            common_utils.update_file_metadata(config_dict, doc_id, metadata_extra={"pipeline_step": f"textual_preparation_failed_exception_{e_textual_prep.__class__.__name__}"})


    # --- Return based on the success of the initial step ---
    if initial_step_success:
        logger.info(f"Initial processing step successful for doc_id '{doc_id}' ('{original_filename}') as '{pipeline_preference}'. Msg: {msg_pipeline_step}")
        return True, doc_id, msg_pipeline_step
    else:
        logger.error(f"Initial processing step failed for doc_id '{doc_id}' ('{original_filename}') as '{pipeline_preference}'. Msg: {msg_pipeline_step}")
        # The metadata should have already been updated with a failure step inside the pipeline blocks
        return False, doc_id, f"Initial processing step failed: {msg_pipeline_step}"

async def generate_ai_metadata_and_update(
    config_dict: Dict,
    doc_id: str,
    current_ollama_client: Optional[OpenAI],
    common_utils_ref: Any
):
    if not current_ollama_client:
        logger.warning(f"AI Meta: Ollama client NA for doc_id {doc_id}. Skipping.")
        return
        
    logger.info(f"AI Meta Task: Starting for doc_id: {doc_id}")
    doc_meta = common_utils_ref.get_specific_doc_metadata(config_dict, doc_id)
    if not doc_meta:
        logger.error(f"AI Meta: No metadata for {doc_id}.")
        return

    content_sample = ""
    pipeline_type = doc_meta.get("pipeline_type", "visual")

    if pipeline_type == "textual":
        processed_textual_file = doc_meta.get("processed_path")
        if processed_textual_file:
            full_path = os.path.join(config_dict["vault_directory"], processed_textual_file)
            if os.path.exists(full_path):
                try:
                    with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                        content_sample = f.read(15000)
                    logger.info(f"AI Meta: Read sample from '{full_path}' for doc '{doc_id}'.")
                except Exception as e:
                    logger.error(f"AI Meta: Error reading '{full_path}': {e}")
            else:
                logger.warning(f"AI Meta: Processed textual file '{full_path}' not found for doc '{doc_id}'.")
        else:
            logger.warning(f"AI Meta: No 'processed_path' for textual doc '{doc_id}'.")
    elif pipeline_type == "visual":
        logger.info(f"AI Meta: Visual doc '{doc_id}'. AI desc from content not at upload. Using user desc.")

    if content_sample.strip():
        ai_desc, ai_tags = await common_utils_ref.generate_file_metadata(config_dict, content_sample, current_ollama_client)
        if ai_desc or ai_tags:
            final_desc = doc_meta.get("user_provided_description") or ""
            if ai_desc and (not final_desc or len(ai_desc) > 10):
                final_desc = ai_desc
            combined_tags = sorted(list(set(doc_meta.get("tags", []) + (ai_tags if isinstance(ai_tags, list) else []))))
            common_utils_ref.update_file_metadata(config_dict, doc_id, description=final_desc, tags=combined_tags,
                                               metadata_extra={"ai_metadata_generated_at": datetime.now().isoformat()})
            logger.info(f"AI Meta: Updated metadata for {doc_id} with AI insights.")
        else:
            logger.warning(f"AI Meta: No desc or tags from AI for {doc_id}.")
    elif pipeline_type == "textual":
        logger.warning(f"AI Meta: Content sample for textual doc {doc_id} empty.")
    logger.info(f"AI Meta task finished for {doc_id}.")

@app.post("/upload", summary="Upload File with Pipeline Preference")
async def upload_file_http(
    file: UploadFile = File(...),
    user_description: Optional[str] = Form(None),
    client_id: Optional[str] = Form(None),
    pipeline_preference: str = Form("visual") # Default to visual
):
    temp_file_path = None
    original_filename = file.filename or "unknown_file"
    logger.info(f"Upload request for: '{original_filename}' from client: {client_id}, Preferred Pipeline: '{pipeline_preference}'")
    try:
        safe_original_filename = os.path.basename(original_filename)
        file_ext_from_name = os.path.splitext(safe_original_filename)[1].lower().lstrip('.')
        if not file_ext_from_name:
            file_ext_from_name = "unknown"

        # Basic validation for pipeline preference and file type support
        if pipeline_preference not in ["visual", "textual"]:
            logger.warning(f"Invalid pipeline preference '{pipeline_preference}' received for '{original_filename}'. Defaulting to 'visual'.")
            pipeline_preference = "visual" # Default to visual if invalid
        if pipeline_preference == "visual" and file_ext_from_name not in ["pdf", "png", "jpg", "jpeg"]:
            detail_msg = f"Visual pipeline for '.{file_ext_from_name}' not supported. Supported types: pdf, png, jpg, jpeg."
            logger.warning(f"Upload failed for '{original_filename}': {detail_msg}")
            raise HTTPException(status_code=400, detail=detail_msg)
        if pipeline_preference == "textual" and file_ext_from_name in ["png", "jpg", "jpeg"]:
            # Textual pipeline *can* potentially handle images via OCR, but the current
            # `prepare_textual_document` might not. Let's restrict explicitly for now based on current `textual_pipeline`.
             # NOTE: If textual_pipeline is updated to handle image OCR, this check can be relaxed.
            detail_msg = f"Textual pipeline for '.{file_ext_from_name}' is not explicitly supported by the current implementation. Supported types often include pdf, txt, docx etc. but this depends on the 'textual_pipeline' prepare function."
            logger.warning(f"Upload failed for '{original_filename}': {detail_msg}")
            # Maybe allow with a warning, or reject? Let's reject to avoid later failures.
            raise HTTPException(status_code=400, detail=detail_msg)


        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        temp_file_path = os.path.join(temp_dir, f"temp_{uuid.uuid4().hex}_{safe_original_filename}")
        bytes_written = 0
        # Read the file in chunks
        try:
            with open(temp_file_path, "wb") as temp_f:
                while content_chunk := await file.read(8 * 1024 * 1024): # Read in 8MB chunks
                    temp_f.write(content_chunk)
                    bytes_written += len(content_chunk)
        except Exception as e_write_temp:
             logger.error(f"Error writing temp file for '{original_filename}': {e_write_temp}", exc_info=True)
             raise HTTPException(status_code=500, detail=f"Server error saving temporary file: {str(e_write_temp)[:100]}")

        if bytes_written == 0:
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path) # Clean up empty temp file
            logger.warning(f"Upload failed for '{original_filename}': Uploaded file is empty.")
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        logger.info(f"Temp file '{temp_file_path}' ({bytes_written} bytes) saved for '{original_filename}'")

        # Additional validation for PDF for visual pipeline
        if pipeline_preference == "visual" and file_ext_from_name == "pdf":
            if not PYMUPDF_AVAILABLE:
                logger.error("PyMuPDF is not available, but visual PDF upload was attempted.")
                raise HTTPException(status_code=501, detail="PyMuPDF library not available for visual PDF processing.")
            try:
                # Open PDF with fitz to validate
                doc = fitz.open(temp_file_path)
                if not doc.is_pdf:
                    doc.close()
                    raise ValueError("File is not recognized as a PDF.")
                if doc.page_count == 0:
                     doc.close()
                     raise ValueError("PDF contains 0 pages.")
                if doc.is_encrypted and not doc.authenticate(""): # Attempt authentication with empty password first
                    doc.close()
                    # Add a specific check or prompt if password is required? For now, fail.
                    raise ValueError("PDF is encrypted and requires a password.")
                doc.close() # Close the PDF object
            except Exception as pdf_val_err:
                # Clean up temp file on validation failure
                if temp_file_path and os.path.exists(temp_file_path):
                    try: os.remove(temp_file_path)
                    except Exception as e_rem_val: logger.error(f"Error removing temp file {temp_file_path} after PDF validation error: {e_rem_val}")
                logger.warning(f"PDF validation failed for '{original_filename}': {pdf_val_err}")
                raise HTTPException(status_code=400, detail=f"Invalid or unsupported PDF format: {pdf_val_err}")

        # --- Orchestrate the initial processing step ---
        # orchestrate_file_processing now returns True/False based on the success of its initial step.
        success, processed_doc_id, msg_proc = await orchestrate_file_processing(
            CONFIG, temp_file_path, safe_original_filename, file_ext_from_name,
            user_description, pipeline_preference
        )

        # --- Check the result of the orchestration ---
        if success:
             # If orchestration returned True, the initial step (image extraction or text prep) succeeded.
             logger.info(f"'{original_filename}' (ID: {processed_doc_id}) upload and initial processing as '{pipeline_preference}' OK. Msg: {msg_proc}")

             # Fetch the latest metadata to include in the success response and for broadcasting
             final_meta = common_utils.get_specific_doc_metadata(CONFIG, processed_doc_id) or {}

             # *** START NEW NOTIFICATION LOGIC ***
             # Get the updated list of all vault files *after* the new file has been added
             updated_vault_files_after_upload = common_utils.get_vault_files(CONFIG)
             logger.info(f"Broadcasting updated file list ({len(updated_vault_files_after_upload)} files) to all active WebSocket clients after upload of {processed_doc_id}.")
             # Iterate through all active connections and send the updated list
             for conn_client_id_notify in list(manager.active_connections.keys()):
                 try:
                     # Use await manager.send_json for async send
                     await manager.send_json(conn_client_id_notify, {"type": "available_files", "files": updated_vault_files_after_upload})
                 except Exception as e_notify_upload:
                     # Log any error during notification but don't fail the upload response
                     logger.error(f"Error notifying client {conn_client_id_notify} about upload of {processed_doc_id}: {e_notify_upload}")
             logger.info(f"Finished broadcasting file list update.")
             # *** END NEW NOTIFICATION LOGIC ***


             # Trigger AI metadata generation as a background task if successful
             # It's best to trigger this *after* sending the initial success response and file list update
             # so the user sees the file appear quickly, even if AI metadata takes longer.
             if ollama_client and processed_doc_id:
                 asyncio.create_task(generate_ai_metadata_and_update(CONFIG, processed_doc_id, ollama_client, common_utils))
                 logger.info(f"Started AI metadata generation task for {processed_doc_id}.")


             return JSONResponse(status_code=200, content={
                 # Indicate upload is successful and processing initiated/ongoing
                 "message": f"'{original_filename}' (ID: {processed_doc_id}) upload successful. Initial processing as '{pipeline_preference}' complete. Further processing will occur when file is selected. {msg_proc}",
                 "doc_id": processed_doc_id,
                 "metadata": final_meta
             })
        else:
             # If orchestration returned False, the initial step failed.
             logger.error(f"'{original_filename}' (ID: {processed_doc_id if processed_doc_id else 'N/A'}) upload failed during initial processing stage. Msg: {msg_proc}")
             # Re-raise as HTTP 500, including the specific failure message from orchestration
             raise HTTPException(status_code=500, detail=f"Server error during initial processing: {msg_proc}")

    except HTTPException as http_e:
        # HTTPExceptions are raised directly and don't need extra logging here
        raise http_e
    except Exception as e_upload:
        logger.error(f"Upload endpoint encountered an unexpected error for '{original_filename}': {e_upload}", exc_info=True)
        # Catch any other unexpected errors and return a generic 500
        raise HTTPException(status_code=500, detail=f"An unexpected server error occurred during upload: {str(e_upload)[:100]}")
    finally:
        # Ensure the temporary file is removed, even if errors occurred
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logger.debug(f"Removed temporary file: {temp_file_path}")
            except Exception as e_rem_final:
                logger.error(f"Error removing temp file {temp_file_path} in finally block: {e_rem_final}")

@app.get("/files", summary="List Vault Files")
async def get_files_http_endpoint():
    common_utils.initialize_vault_directory(CONFIG)
    files = common_utils.get_vault_files(CONFIG)
    return {"files": files}

@app.delete("/delete/{doc_id_param}", summary="Delete File by ID")
async def delete_file_http_endpoint(doc_id_param: str, client_id: Optional[str] = Form(None)):
    global CONFIG, manager, chroma_collection

    logger.info(f"HTTP Delete request received for doc_id: {doc_id_param}")

    doc_meta = common_utils.get_specific_doc_metadata(CONFIG, doc_id_param)
    original_filename_display = doc_id_param
    pipeline_type_for_deletion = "unknown"
    if doc_meta:
        original_filename_display = doc_meta.get("original_filename", doc_id_param)
        pipeline_type_for_deletion = doc_meta.get("pipeline_type", "visual")

    item_deleted_from_disk = False
    item_type_deleted_on_disk = "unknown"
    deletion_messages = []

    if pipeline_type_for_deletion == "visual" or pipeline_type_for_deletion == "unknown":
        visual_path_to_delete = os.path.join(CONFIG["vault_directory"], CONFIG["processed_docs_subdir"], doc_id_param)
        if os.path.isdir(visual_path_to_delete):
            try:
                shutil.rmtree(visual_path_to_delete)
                item_deleted_from_disk = True
                item_type_deleted_on_disk = "processed visual document directory"
                logger.info(f"Deleted visual document directory: {visual_path_to_delete}")
                deletion_messages.append(f"Visual data directory removed for '{original_filename_display}'.")
            except Exception as e_rm_visual:
                logger.error(f"Error deleting visual directory '{visual_path_to_delete}': {e_rm_visual}", exc_info=True)
                deletion_messages.append(f"Error removing visual files: {e_rm_visual}")
        elif pipeline_type_for_deletion == "visual":
            logger.warning(f"Visual document directory not found for deletion: {visual_path_to_delete}")

    if pipeline_type_for_deletion == "textual" or (pipeline_type_for_deletion == "unknown" and not item_deleted_from_disk):
        textual_filename_to_delete = doc_meta.get("processed_path", f"{doc_id_param}_textual.txt") if doc_meta else f"{doc_id_param}_textual.txt"
        textual_path_to_delete = os.path.join(CONFIG["vault_directory"], textual_filename_to_delete)
        if os.path.exists(textual_path_to_delete) and os.path.isfile(textual_path_to_delete):
            try:
                os.remove(textual_path_to_delete)
                item_deleted_from_disk = True
                item_type_deleted_on_disk = "processed textual file" if item_type_deleted_on_disk == "unknown" else item_type_deleted_on_disk + " & textual file"
                logger.info(f"Deleted textual processed file: {textual_path_to_delete}")
                deletion_messages.append(f"Textual processed file removed for '{original_filename_display}'.")
            except Exception as e_rm_textual:
                logger.error(f"Error deleting textual file '{textual_path_to_delete}': {e_rm_textual}", exc_info=True)
                deletion_messages.append(f"Error removing textual file: {e_rm_textual}")
        elif pipeline_type_for_deletion == "textual":
            logger.warning(f"Textual processed file not found for deletion: {textual_path_to_delete}")

    if (doc_meta and doc_meta.get("pipeline_type") == "visual" and doc_meta.get("indexing_complete")) or pipeline_type_for_deletion == "visual":
        if chroma_collection:
            try:
                logger.info(f"Attempting to delete entries for doc_id '{doc_id_param}' from ChromaDB collection '{chroma_collection.name}'.")
                chroma_collection.delete(where={"doc_id": doc_id_param})
                logger.info(f"Submitted delete request to ChromaDB for doc_id '{doc_id_param}'.")
                deletion_messages.append(f"Vector index entries removed for '{original_filename_display}'.")
            except Exception as e_chroma_del:
                logger.error(f"Error deleting from ChromaDB for doc_id '{doc_id_param}': {e_chroma_del}", exc_info=True)
                deletion_messages.append(f"Error removing vector index entries: {e_chroma_del}")
        else:
            logger.warning(f"Chroma collection not available, cannot delete entries for {doc_id_param}.")
            deletion_messages.append(f"Vector index not available for cleanup of '{original_filename_display}'.")

    metadata_entry_removed = common_utils.remove_doc_from_metadata(CONFIG, doc_id_param)
    if metadata_entry_removed:
        deletion_messages.append(f"Metadata entry removed for '{original_filename_display}'.")
    else:
        if not item_deleted_from_disk:
            logger.warning(f"Doc ID '{doc_id_param}' not found on disk and also not found in metadata for removal.")
            raise HTTPException(status_code=404, detail=f"Document ID '{doc_id_param}' not found anywhere to delete.")
        else:
            logger.info(f"Doc ID '{doc_id_param}' deleted from disk, but was not found in metadata (or already removed).")
            deletion_messages.append(f"Metadata entry for '{original_filename_display}' was not found (possibly already removed).")

    manager.cleanup_file_data(doc_id_param)
    logger.info(f"Cleaned client session data for deleted doc_id: {doc_id_param}")

    updated_vault_files_after_delete = common_utils.get_vault_files(CONFIG)
    for conn_client_id_notify in list(manager.active_connections.keys()):
        try:
            await manager.send_json(conn_client_id_notify, {"type": "available_files", "files": updated_vault_files_after_delete})
            if conn_client_id_notify == client_id:
                await manager.send_json(client_id, {
                    "type": "file_deleted",
                    "filename": doc_id_param,
                    "message": f"Document '{original_filename_display}' (ID: {doc_id_param}) deletion processed."
                })
        except Exception as e_notify_del:
            logger.error(f"Error notifying client {conn_client_id_notify} about deletion of {doc_id_param}: {e_notify_del}")
    
    final_status_message = f"Deletion process for '{original_filename_display}' (ID: {doc_id_param}) completed. Details: {' '.join(deletion_messages)}"
    logger.info(final_status_message)
    return JSONResponse(status_code=200, content={"success": True, "message": final_status_message})

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """Handles incoming WebSocket connections and message routing."""
    global manager, ollama_client, CONFIG, common_utils, textual_pipeline, visual_pipeline, system_message, chroma_collection

    await manager.connect(websocket, client_id)
    logger.info(f"Client {client_id} connected.")

    try:
        await manager.send_json(client_id, {"type": "status", "message": "Connected."})
        available_files_list = common_utils.get_vault_files(CONFIG)
        await manager.send_json(client_id, {"type": "available_files", "files": available_files_list})
        logger.info(f"Sent {len(available_files_list)} available files to client {client_id}.")

        while True:
            message_data = await websocket.receive_json()
            message_type = message_data.get("type")
            logger.debug(f"WS Received from {client_id}: Type='{message_type}', Data='{str(message_data)[:100]}...'")

            if message_type == "chat":
                await handle_chat_message(
                    websocket,
                    message_data,
                    client_id,
                    manager,
                    ollama_client,
                    CONFIG,
                    common_utils,
                    textual_pipeline,
                    visual_pipeline,
                    system_message,
                    chroma_collection  # FIX: Added chroma_collection
                )
                logger.debug(f"Dispatched chat message for {client_id}")

            elif message_type == "select_files":
                await handle_file_selection(
                    websocket,
                    message_data,
                    client_id,
                    manager,
                    CONFIG,
                    chroma_collection,
                    common_utils,
                    textual_pipeline,
                    visual_pipeline
                )
                logger.debug(f"Dispatched select_files message for {client_id}")

            elif message_type == "delete_file":
                logger.warning(f"Delete file message received from {client_id}, handler not yet implemented or called.")
                await manager.send_json(client_id, {"type": "status", "message": "Delete file message received, handler not yet implemented."})

            elif message_type == "ping":
                await manager.send_json(client_id, {"type": "pong"})
                logger.debug(f"Sent pong response for JSON ping from {client_id}")

            else:
                logger.warning(f"Unknown message type from {client_id}: {message_type}. Message data: {message_data}")
                safe_error_message = html.escape(f"Unknown message type received: {message_type}")
                await manager.send_json(client_id, {"type": "error", "message": safe_error_message})

    except WebSocketDisconnect as e:
        logger.info(f"Client {client_id} disconnected with code: {e.code}, reason: {e.reason}")
        manager.disconnect(client_id)

    except Exception as e:
        logger.error(f"Unhandled exception in WebSocket connection for {client_id}: {e}", exc_info=True)
        try:
            safe_error_message = html.escape(f"Internal server error occurred: {str(e)}")
            await manager.send_json(client_id, {"type": "error", "message": safe_error_message})
        except Exception:
            pass
        manager.disconnect(client_id)

async def process_websocket_message(data_str: str, client_id: str):
    try:
        message_data = json.loads(data_str)
        msg_type = message_data.get("type", "")
        logger.debug(f"WS Processing type '{msg_type}' for {client_id}")
        if msg_type == "chat":
            await handle_chat_message(message_data, client_id)
        elif msg_type == "select_files":
            await handle_file_selection(message_data, client_id)
        elif msg_type == "ping":
            await manager.send_json(client_id, {"type": "pong"})
        else:
            await manager.send_json(client_id, {"type": "error", "message": f"Unknown WS msg type: {msg_type}"})
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON from {client_id}: {data_str[:100]}", exc_info=True)
        await manager.send_json(client_id, {"type": "error", "message": "Invalid JSON format received."})
    except Exception as e_proc_ws:
        logger.error(f"Error processing WS message from {client_id}: {e_proc_ws}", exc_info=True)
        await manager.send_json(client_id, {"type": "error", "message": f"Server error processing message: {e_proc_ws}"})

async def handle_file_selection(
    websocket: WebSocket,
    message_data: dict,
    client_id: str,
    manager: Any,
    config_dict: Dict,
    chroma_collection: Any, # Correctly passed
    common_utils_module: Any, # Correctly passed
    textual_pipeline_module: Any,
    visual_pipeline_module: Any # Contains the updated functions
):
    try:
        requested_doc_ids = message_data.get("files", [])
        logger.info(f"WS: Client {client_id} selected docs: {requested_doc_ids}")

        if not isinstance(requested_doc_ids, list):
            await manager.send_json(client_id, {"type": "error", "message": "Invalid file selection format."})
            logger.warning(f"Client {client_id} sent invalid file selection format: {message_data}")
            return

        all_meta = common_utils_module.get_vault_files(config_dict)
        valid_selection = [doc_id for doc_id in requested_doc_ids if isinstance(doc_id, str) and any(m.get("filename") == doc_id for m in all_meta)]

        manager.set_client_files(client_id, valid_selection)
        # Note: You might want to clear embeddings/content only for *removed* files, not the whole selection
        # But clearing seems safer to avoid stale data from previously selected files.
        manager.set_client_content(client_id, {})
        manager.set_client_embeddings(client_id, {})
        logger.info(f"Client {client_id} session data reset for new selection. Valid IDs: {valid_selection}")

        if not valid_selection:
            await manager.send_json(client_id, {"type": "status", "message": "Selection cleared."})
            logger.info(f"Client {client_id} cleared selection.")
            return

        await manager.send_json(client_id, {"type": "status", "message": f"Processing {len(valid_selection)} selected document(s)..."})

        visual_ready_c = 0
        textual_ready_c = 0
        failed_names = []

        for doc_id in valid_selection:
            meta = common_utils_module.get_specific_doc_metadata(config_dict, doc_id)
            # Re-fetch meta inside the loop to get the latest status after each step
            meta = common_utils_module.get_specific_doc_metadata(config_dict, doc_id)
            if not meta:
                logger.warning(f"Metadata not found for selected doc ID {doc_id}. Cannot process.")
                failed_names.append(f"{doc_id}(NoMeta)")
                continue

            display_name = meta.get("original_filename", doc_id)
            pipeline_type = meta.get("pipeline_type", "visual") # Default to visual if type is missing

            logger.info(f"Processing '{display_name}' (ID: {doc_id}) via '{pipeline_type}' pipeline for client {client_id}.")
            await manager.send_json(client_id, {"type": "status", "message": f"Checking processing status for '{display_name}'..."})

            processed_ok_flag = False
            current_step = meta.get("pipeline_step", "unknown")

            # --- Check status and trigger processing steps for Visual Pipeline ---
            if pipeline_type == "visual":
                # Visual requires indexing_complete and pipeline_step='indexing_complete' or 'indexing_complete_no_chunks'
                is_indexed = meta.get("indexing_complete", False)
                is_indexed_status = current_step in ["indexing_complete", "indexing_complete_no_chunks"]
                requires_processing = not (is_indexed and is_indexed_status)

                if requires_processing:
                    logger.debug(f"Visual pipeline for {doc_id} requires processing. Current step: {current_step}")
                    try:
                        # Trigger steps sequentially if not completed
                        # Note: Each step updates metadata and returns success/failure indicator (path or None, True/False)
                        # The step functions themselves handle logging and basic metadata status updates on success/failure

                        # Step 3: Layout Analysis (using LayoutParser)
                        if current_step not in ["layout_analysis_complete", "ocr_extraction_complete", "indexing_complete", "indexing_complete_no_chunks"]:
                             layout_success_path = await visual_pipeline_module.perform_layout_analysis(doc_id, config_dict, common_utils_module)
                             if not layout_success_path:
                                 # Error logged inside, metadata status updated inside
                                 raise Exception("Layout analysis failed.")
                             # Re-fetch meta to get updated status
                             meta = common_utils_module.get_specific_doc_metadata(config_dict, doc_id) or meta
                             current_step = meta.get("pipeline_step", "unknown")
                             await manager.send_json(client_id, {"type": "status", "message": f"Layout analysis done for '{display_name}'."})


                        # Step 4: OCR Extraction (using LayoutParser regions)
                        if current_step not in ["ocr_extraction_complete", "indexing_complete", "indexing_complete_no_chunks"]:
                             # This step needs to read the layout analysis file
                             ocr_success_path = await visual_pipeline_module.perform_ocr_extraction(doc_id, config_dict, common_utils_module)
                             if not ocr_success_path:
                                 # Error logged inside, metadata status updated inside
                                 raise Exception("OCR extraction failed.")
                             # Re-fetch meta
                             meta = common_utils_module.get_specific_doc_metadata(config_dict, doc_id) or meta
                             current_step = meta.get("pipeline_step", "unknown")
                             await manager.send_json(client_id, {"type": "status", "message": f"OCR extraction done for '{display_name}'."})


                        # Step 5: Indexing (using LayoutParser-based chunks)
                        if current_step not in ["indexing_complete", "indexing_complete_no_chunks"]:
                             # This step needs to read the ocr_results file and call the chunking function
                             # It also needs chroma_collection
                             index_success = await visual_pipeline_module.index_ocr_data(doc_id, config_dict, chroma_collection, common_utils_module)
                             if not index_success:
                                 # Error logged inside, metadata status updated inside
                                 raise Exception("Indexing failed.")
                             # Re-fetch meta
                             meta = common_utils_module.get_specific_doc_metadata(config_dict, doc_id) or meta
                             current_step = meta.get("pipeline_step", "unknown")
                             await manager.send_json(client_id, {"type": "status", "message": f"Indexing done for '{display_name}'."})


                        # Final check after attempting all steps
                        if meta.get("indexing_complete"):
                            processed_ok_flag = True
                            logger.info(f"Visual pipeline processing steps successful or already complete for {doc_id}")
                        else:
                             # This could happen if indexing_complete somehow wasn't set true despite index_success being true
                             # Or if an error occurred in a previous step but wasn't caught/re-raised properly.
                             # Let's treat it as a failure here.
                             logger.error(f"Visual processing for {doc_id} finished steps but indexing_complete is still False. Current step: {current_step}")
                             raise Exception(f"Visual processing finished but indexing status incomplete ({current_step}).")


                    except Exception as e_visual_proc:
                        logger.error(f"Error during visual pipeline processing chain for {doc_id}: {e_visual_proc}", exc_info=True)
                        failed_names.append(f"{display_name}(VisualProcFail: {str(e_visual_proc)[:50]})")
                        # Metadata update for the failure step is handled within the step functions
                        # A final catch-all update might be useful if status wasn't set correctly
                        meta_after_fail = common_utils_module.get_specific_doc_metadata(config_dict, doc_id) or {}
                        if not meta_after_fail.get("pipeline_step", "").startswith("visual_failed"):
                            common_utils_module.update_file_metadata(config_dict, doc_id, metadata_extra={"pipeline_step": f"visual_failed_chain_{e_visual_proc.__class__.__name__}"})


                else:
                    # Document was already indexed according to metadata
                    processed_ok_flag = True
                    logger.info(f"Visual document {doc_id} already indexed. Ready for chat.")


                if processed_ok_flag:
                    visual_ready_c += 1


            # --- Check status and trigger processing steps for Textual Pipeline ---
            elif pipeline_type == "textual":
                # Textual requires content loaded into session manager and embeddings generated
                is_content_loaded = doc_id in manager.get_client_content(client_id)
                is_embeddings_generated = doc_id in manager.get_client_embeddings(client_id)
                requires_processing = not (is_content_loaded and is_embeddings_generated)

                if requires_processing:
                    logger.debug(f"Textual pipeline for {doc_id} requires session loading. Current step: {current_step}")
                    try:
                         # This part seems correct for loading/embedding textual data into session
                        content = textual_pipeline_module.read_vault_content_textual(config_dict, [doc_id], all_meta)
                        if not content or doc_id not in content:
                            # textual_pipeline_module.read_vault_content_textual should log failure
                            raise Exception("Failed to read textual content.")

                        embeddings = await textual_pipeline_module.generate_vault_embeddings_textual(
                            config_dict,
                            content,
                            client_id,
                            manager,
                            common_utils_module,
                            ollama_client # Assuming ollama_client is available
                        )
                        if not embeddings or doc_id not in embeddings:
                             # textual_pipeline_module.generate_vault_embeddings_textual should log failure
                            raise Exception("Failed to generate embeddings.")

                        # Update manager session data
                        current_content = manager.get_client_content(client_id)
                        current_content.update(content)
                        manager.set_client_content(client_id, current_content)

                        current_embeddings = manager.get_client_embeddings(client_id)
                        current_embeddings.update(embeddings)
                        manager.set_client_embeddings(client_id, current_embeddings)

                        textual_ready_c += 1
                        processed_ok_flag = True
                        # Update metadata to indicate session data loaded status (optional, can just rely on session manager state)
                        common_utils_module.update_file_metadata(config_dict, doc_id, metadata_extra={"pipeline_step": "textual_data_loaded_for_session"})
                        logger.info(f"Textual document {doc_id} loaded and embeddings generated for session {client_id}.")

                    except Exception as e_textual_proc:
                        logger.error(f"Error during textual session loading/embedding for doc_id {doc_id}: {e_textual_proc}", exc_info=True)
                        failed_names.append(f"{display_name}(TextProcFail: {str(e_textual_proc)[:50]})")
                        # Metadata update for the failure step might be needed here if not in textual_pipeline functions
                        meta_after_fail = common_utils_module.get_specific_doc_metadata(config_dict, doc_id) or {}
                        if not meta_after_fail.get("pipeline_step", "").startswith("textual_failed"):
                            common_utils_module.update_file_metadata(config_dict, doc_id, metadata_extra={"pipeline_step": f"textual_failed_session_{e_textual_proc.__class__.__name__}"})

                else:
                    # Document session data was already loaded
                    processed_ok_flag = True
                    textual_ready_c += 1
                    logger.info(f"Textual document {doc_id} session data already loaded. Ready for chat.")

            else:
                # This case should ideally not happen if pipeline_preference is validated on upload
                logger.error(f"Unknown pipeline type '{pipeline_type}' for doc ID {doc_id} during selection handling.")
                failed_names.append(f"{display_name}(UnknownPipeline)")


            # Send status update for the specific doc
            await manager.send_json(client_id, {"type": "status", "message": f"'{display_name}' processing check complete (Ready: {processed_ok_flag})."})


        final_parts = []
        if visual_ready_c > 0:
            final_parts.append(f"{visual_ready_c} visual")
        if textual_ready_c > 0:
            final_parts.append(f"{textual_ready_c} textual")

        final_msg_text = f"Selected documents processed: {', '.join(final_parts) if final_parts else 'None'} ready."
        if failed_names:
            final_msg_text += f" Issues with: {'; '.join(failed_names)}." # Use semicolon for clarity

        await manager.send_json(client_id, {"type": "status", "message": final_msg_text})
        logger.info(f"WS: File selection processing for {client_id} finished. {final_msg_text}")

    except Exception as e_select_main:
        logger.error(f"WS: Unhandled exception in handle_file_selection for {client_id}: {e_select_main}", exc_info=True)
        error_message = f"An unexpected server error occurred during file selection: {str(e_select_main)}"
        safe_error_message = html.escape(error_message)
        await manager.send_json(client_id, {
            "type": "error",
            "message": safe_error_message,
            "client_id": client_id
        })

async def handle_chat_message(
    websocket: WebSocket,
    message_data: dict,
    client_id: str,
    manager: Any, # ConnectionManager instance
    ollama_client: Optional[OpenAI], # Ollama client instance
    CONFIG: Dict, # Main configuration dictionary
    common_utils: Any, # common_utils module reference
    textual_pipeline: Any, # textual_pipeline module reference
    visual_pipeline: Any, # visual_pipeline module reference (contains ollama_chat_visual_async)
    system_message: str, # The strict system message string
    chroma_collection: Any # ChromaDB collection object
):
    """
    Handles incoming chat messages from WebSocket clients.
    Determines which documents are targeted and ready, selects pipeline (visual/textual),
    calls the appropriate chat orchestrator, and sends the response back to the client.
    """
    try:
        user_message = message_data.get("message", "").strip()
        if not user_message:
            await manager.send_json(client_id, {"type": "error", "message": "Empty message."})
            logger.warning(f"Client {client_id} sent empty chat message.")
            return

        logger.info(f"WS Chat from {client_id}: '{user_message[:50]}...'")
        await manager.send_json(client_id, {"type": "status", "message": "Processing query..."})

        # Get the list of documents currently selected by this client
        session_selection = manager.get_client_files(client_id)
        if not session_selection:
            logger.info(f"No documents selected for client {client_id}. Sending no-info response.")
            # Generate a response indicating no docs are selected
            all_meta_for_no_info = common_utils.get_vault_files(CONFIG)
            msg = common_utils.generate_no_information_response(CONFIG, user_message, [], all_meta_for_no_info)
            await manager.send_json(client_id, {"type": "chat_response", "message": msg, "context": []})
            return

        # Parse query to see if specific files are requested, otherwise use session selection
        all_meta = common_utils.get_vault_files(CONFIG)
        cleaned_query, query_targets = common_utils.parse_file_query(CONFIG, user_message, all_meta, session_selection)
        final_targets = query_targets if query_targets else session_selection # Use parsed targets or selected files

        if not final_targets:
            logger.info(f"Query parsing resulted in no targets for client {client_id}. Sending no-info response.")
            # Generate a response indicating no relevant docs were found or targeted
            msg = common_utils.generate_no_information_response(CONFIG, cleaned_query, [], all_meta)
            await manager.send_json(client_id, {"type": "chat_response", "message": msg, "context": []})
            return

        logger.info(f"Chat targets for '{cleaned_query[:50]}...': {final_targets}")

        # Check readiness of targeted documents
        visual_ready, textual_ready, not_ready = [], [], []
        session_text_emb = manager.get_client_embeddings(client_id)
        session_text_cont = manager.get_client_content(client_id)

        for doc_id in final_targets:
            meta = common_utils.get_specific_doc_metadata(CONFIG, doc_id)
            if not meta:
                logger.warning(f"Metadata not found for targeted doc ID {doc_id}")
                not_ready.append(f"{doc_id}(NoMeta)")
                continue

            p_type = meta.get("pipeline_type", "visual") # Default to visual if type is missing

            # A visual document is ready if its metadata indicates indexing is complete.
            if p_type == "visual" and meta.get("indexing_complete", False) and meta.get("pipeline_step") in ["indexing_complete", "indexing_complete_no_chunks"]:
                 visual_ready.append(doc_id)
            # A textual document is ready if its content and embeddings are loaded into the session manager.
            elif p_type == "textual" and doc_id in session_text_cont and doc_id in session_text_emb:
                textual_ready.append(doc_id)
            else:
                # Document is targeted but not ready. Log status details.
                logger.warning(f"Targeted doc {doc_id} ('{meta.get('original_filename', doc_id)}') not ready. Pipeline type: {p_type}, Indexing Complete: {meta.get('indexing_complete',False)}, Pipeline Step: {meta.get('pipeline_step','UnknownStep')}")
                status_detail = meta.get('pipeline_step', 'Unknown')
                if p_type == 'visual' and not meta.get('indexing_complete'):
                    status_detail = meta.get('pipeline_step', 'Indexing Incomplete') # More specific if visual indexing failed/incomplete
                if p_type == 'textual' and (doc_id not in session_text_cont or doc_id not in session_text_emb):
                    status_detail = 'Content/Embeddings Not Loaded' # More specific if textual session loading failed
                not_ready.append(f"{meta.get('original_filename', doc_id)}({status_detail})")


        # If no documents are ready (either visual or textual), send a message indicating this.
        if not visual_ready and not textual_ready:
            logger.warning(f"No targeted documents were ready for chat for client {client_id}. Not ready: {not_ready}")
            msg_parts = ["<p>The targeted document(s) are not yet ready for chat.</p>"]
            if not_ready:
                msg_parts.append("<p>Not ready status:</p><ul>")
                for item in not_ready:
                    msg_parts.append(f"<li>{html.escape(item)}</li>")
                msg_parts.append("</ul>")
            msg_parts.append("<p>Please ensure files are fully processed/loaded before querying them.</p>")
            final_unready_msg = "".join(msg_parts)
            await manager.send_json(client_id, {"type": "chat_response", "message": final_unready_msg, "context": []})
            return # Exit the handler if no documents are ready

        # --- Determine which pipeline to use and call the appropriate orchestrator ---
        # Prioritize visual if any visual documents are ready.
        raw_resp_text = ""
        ctx_used: List[Dict] = [] # List of context items returned by the orchestrator

        # Check if ChromaDB is available if the Visual pipeline is the one to be used.
        if visual_ready: # If any visual docs are ready, we will attempt the visual pipeline
            if not chroma_collection:
                 logger.error("ChromaDB collection not available, cannot run visual chat.")
                 await manager.send_json(client_id, {"type": "error", "message": "Document index not initialized for visual chat."})
                 return # Exit the handler if visual is needed but Chroma is down

            if textual_ready: # Log if textual documents are ready but being ignored for visual
                logger.warning(f"Visual targets ({visual_ready}) available; textual targets ({textual_ready}) ignored in favor of visual pipeline.")

            try:
                # --- CALL THE VISUAL CHAT ORCHESTRATOR ---
                logger.info(f"Calling visual pipeline chat orchestrator for {client_id} with {len(visual_ready)} documents.")
                raw_resp_text, ctx_used = await asyncio.wait_for(
                    visual_pipeline.ollama_chat_visual_async(
                        config_dict=CONFIG, # Pass the main config dictionary
                        user_input=cleaned_query, # Pass the cleaned user query
                        selected_doc_ids=visual_ready, # Pass ONLY the list of READY visual doc IDs
                        client_id=client_id, # Pass the client ID
                        ollama_client=ollama_client, # Pass the Ollama client instance
                        system_message=system_message, # Pass the strict global system message string
                        manager=manager, # Pass the connection manager instance
                        common_utils_module=common_utils, # Pass the common_utils module reference
                        chroma_collection_obj=chroma_collection, # Pass the ChromaDB collection object
                        # page_filter=None # Optional parameter, include if you implement page-specific queries
                    ),
                    timeout=CONFIG.get("chat_timeout", 60.0) # Use config for timeout, default 60s
                )
                logger.info(f"Visual chat pipeline returned for client {client_id}.")
            except asyncio.TimeoutError:
                logger.error(f"Visual pipeline timed out for client {client_id}.")
                # Send timeout error message to client
                await manager.send_json(client_id, {"type": "error", "message": f"Query timed out ({CONFIG.get('chat_timeout', 60.0)}s) while processing visual documents. Please try again."})
                return # Exit the handler after timeout

            except Exception as e:
                logger.error(f"Visual pipeline failed for {client_id}: {e}", exc_info=True)
                # Send failure error message to client
                await manager.send_json(client_id, {"type": "error", "message": f"Failed to process visual documents: {str(e)[:100]}. Check document indexing status for selected files."})
                return # Exit the handler after visual failure

        # --- Fallback to Textual if no visual documents were ready ---
        elif textual_ready: # This block is executed ONLY if visual_ready is empty but textual_ready is not
            try:
                # --- CALL THE TEXTUAL CHAT ORCHESTRATOR ---
                logger.info(f"Calling textual pipeline chat orchestrator for {client_id} with {len(textual_ready)} documents.")
                logger.debug(f"Textual pipeline args: config_dict={CONFIG}, query={cleaned_query}, doc_ids={textual_ready}, session_embeddings=..., session_content=..., client_id={client_id}, system_message=...")
                raw_resp_text, ctx_used = await asyncio.wait_for(
                    textual_pipeline.ollama_chat_textual_async(
                        config_dict=CONFIG, # Pass the config
                        query=cleaned_query, # Pass the user query
                        doc_ids=textual_ready, # Pass ONLY the list of READY textual doc IDs
                        session_embeddings=session_text_emb, # Session data for textual
                        session_content=session_text_cont, # Session data for textual
                        client_id=client_id, # Pass the client ID
                        ollama_client=ollama_client, # Pass the Ollama client
                        manager=manager, # Pass manager
                        common_utils_module=common_utils, # Pass common_utils
                        system_message_str=system_message # Pass the strict system message string
                    ),
                    timeout=CONFIG.get("chat_timeout", 60.0) # Use config for timeout, default 60s
                )
                logger.info(f"Textual chat pipeline returned for client {client_id}.")
            except asyncio.TimeoutError:
                logger.error(f"Textual pipeline timed out for client {client_id}.")
                # Send timeout error message to client
                await manager.send_json(client_id, {"type": "error", "message": f"Query timed out ({CONFIG.get('chat_timeout', 60.0)}s) while processing textual documents. Please try again."})
                return # Exit the handler after timeout

            except Exception as e:
                logger.error(f"Textual pipeline failed for {client_id}: {e}", exc_info=True)
                # Send failure error message to client
                await manager.send_json(client_id, {"type": "error", "message": f"Failed to process textual documents: {str(e)[:100]}"})
                return # Exit the handler after textual failure

        # --- Handle case where neither pipeline ran (should be caught by initial check, but safety) ---
        else:
            # This else block should only be reached if both visual_ready and textual_ready were empty.
            # This scenario should have been handled by the check near the start of the function.
            logger.error(f"Logic error: Neither visual nor textual documents were ready after readiness check for client {client_id}. This state should have triggered an early return.")
            raw_resp_text = "<p>Server routing error: No ready documents found after readiness check.</p>"
            ctx_used = [] # Ensure ctx_used is empty list on error
            # Send error message to client
            await manager.send_json(client_id, {
                 "type": "error",
                 "message": raw_resp_text,
                 "client_id": client_id
            })
            return # Exit the handler


        # --- Post-processing and Sending Response (This section executes ONLY if one of the pipelines returned successfully) ---

        # The chat orchestrator functions (ollama_chat_visual_async and ollama_chat_textual_async)
        # are now responsible for deciding if they found relevant info and structuring the response.
        # If they return an empty ctx_used, it implies no relevant chunks were found or used,
        # and the raw_resp_text should reflect this (based on the strict system message).

        if not ctx_used:
             logger.warning(f"No document context was returned by the pipeline for query '{cleaned_query[:50]}...' in docs {final_targets}. The LLM is expected to handle this.")
             # The raw_resp_text should contain the LLM's response indicating no info found.
             # We just ensure ctx_used is an empty list for the client response.
             ctx_used = [] # Ensure ctx_used is an empty list

        # Ensure raw_resp_text is not None before passing to clean_response_language
        processed_html_response = common_utils.clean_response_language(CONFIG, raw_resp_text if raw_resp_text is not None else "An error occurred.")
        logger.debug(f"Cleaned response for client {client_id}: {processed_html_response[:100]}...")

        # Ensure ctx_used is a list before sending
        await manager.send_json(client_id, {
            "type": "chat_response",
            "message": processed_html_response,
            "context": ctx_used if isinstance(ctx_used, list) else [] # Ensure it's a list
        })
        logger.info(f"Final chat response sent for '{cleaned_query[:50]}...' to client {client_id}. Context items sent: {len(ctx_used) if isinstance(ctx_used, list) else 'N/A'}.")

    except Exception as e:
        # This catch block handles unexpected exceptions *outside* of the pipeline calls themselves
        logger.error(f"Unhandled exception in handle_chat_message for {client_id}: {e}", exc_info=True)
        safe_error_message = html.escape(f"Unexpected server error in chat handler: {str(e)[:100]}")
        await manager.send_json(client_id, {
            "type": "error",
            "message": safe_error_message,
            "client_id": client_id
        })

async def ollama_chat_visual_async(
    user_input: str,
    selected_doc_ids: List[str],
    client_id: str
) -> Tuple[str, List[Dict]]:
    global ollama_client, manager, CONFIG, common_utils, visual_pipeline, chroma_collection
    logger.info(f"--- Visual Chat Orchestrator for {client_id} --- Query: '{user_input[:50]}' Docs: {selected_doc_ids}")
    if not ollama_client:
        return "LLM client not ready.", []
    if not selected_doc_ids:
        return "No visual documents selected for this query.", []
    memory = manager.get_client_memory(client_id)
    chat_history_lc_msgs = []
    if memory:
        try:
            loaded_vars = memory.load_memory_variables({})
            chat_history_lc_msgs = loaded_vars.get(memory.memory_key, [])
        except Exception:
            pass
    formatted_history = [{"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.content} for m in chat_history_lc_msgs if hasattr(m, 'content')]
    await manager.send_json(client_id, {"type": "status", "message": f"Searching {len(selected_doc_ids)} visual document(s)..."})
    all_visual_context = []
    for doc_id_visual in selected_doc_ids:
        context_for_doc = await visual_pipeline.get_visual_context_chroma(
            doc_id=doc_id_visual,
            query=user_input,
            config_dict=CONFIG,
            chroma_collection_obj=chroma_collection,
            top_k=max(1, CONFIG.get("visual_context_results", 15) // len(selected_doc_ids))
        )
        all_visual_context.extend(context_for_doc)
    all_visual_context.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    relevant_ocr_chunks = all_visual_context[:CONFIG.get("visual_context_results", 15)]
    setattr(ollama_chat_visual_async, 'last_context', relevant_ocr_chunks)

    if not relevant_ocr_chunks:
        all_vault_meta = common_utils.get_vault_files(CONFIG)
        no_info_html = common_utils.generate_no_information_response(CONFIG, user_input, selected_doc_ids, all_vault_meta)
        if memory:
            memory.save_context({"input": user_input}, {"output": no_info_html})
        return no_info_html, []
    
    context_str_for_llm = ""
    grouped_by_doc_page_prompt = {}
    for ctx_chunk_prompt in relevant_ocr_chunks:
        meta_prompt = ctx_chunk_prompt.get("metadata", {})
        doc_name_prompt = meta_prompt.get('original_filename', meta_prompt.get('doc_id', 'Unknown'))
        page_num_prompt = meta_prompt.get('page_number', 'N/A')
        key_prompt = (doc_name_prompt, page_num_prompt)
        grouped_by_doc_page_prompt.setdefault(key_prompt, []).append(ctx_chunk_prompt['content'])
    context_parts = []
    for (doc_name_p, page_num_p), texts_p in sorted(grouped_by_doc_page_prompt.items()):
        page_ref_p = f"(Page {page_num_p})" if page_num_p != 'N/A' and page_num_p is not None else ""
        page_header = f"--- Context from Document: {doc_name_p} {page_ref_p} ---"
        page_content = "\n\n".join(texts_p)
        context_parts.append(f"{page_header}\n{page_content}\n--- End Context from {doc_name_p} {page_ref_p} ---")
    context_str_for_llm = "\n\n".join(context_parts)

    system_prompt_visual_payload = f"""You are a meticulous Information Retrieval Assistant... (Your full detailed visual prompt from previous response) ...
    --- Conversation History ---
    {json.dumps(formatted_history)}
    --- Provided Context from Files (OCR Results) ---
    {context_str_for_llm if context_str_for_llm else "No relevant context provided."}
    --- End Provided Context ---
    User Query: "{user_input}"
    Answer based *only* on the context and history, following all rules.
    Assistant Response:"""
    
    messages_for_api = [{"role": "system", "content": system_prompt_visual_payload}]
    await manager.send_json(client_id, {"type": "status", "message": "Generating visual response..."})
    try:
        llm_response_obj = await asyncio.to_thread(
            lambda: ollama_client.chat.completions.create(
                model=CONFIG["ollama_model"],
                messages=messages_for_api,
                temperature=0.05,
                top_p=0.7
            )
        )
        raw_response = llm_response_obj.choices[0].message.content.strip()
        final_response_html = common_utils.clean_response_language(CONFIG, raw_response)
        if memory:
            memory.save_context({"input": user_input}, {"output": final_response_html})
        common_utils.track_response_quality(CONFIG, user_input, final_response_html, relevant_ocr_chunks, client_id)
        return final_response_html, relevant_ocr_chunks
    except Exception as e_llm_visual:
        logger.error(f"LLM call error in visual chat: {e_llm_visual}", exc_info=True)
        return f"Error from Language Model (Visual): {str(e_llm_visual)[:100]}", []

ollama_chat_visual_async.last_context = []

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    SERVER_HOST = os.environ.get("HOST", "0.0.0.0")
    SERVER_PORT = int(os.environ.get("PORT", 8000))
    RELOAD_DEV_MODE = os.environ.get("RELOAD", "true").lower() == "true"
    
    print("--- Starting Chatbot API Server ---")
    print(f"Host: {SERVER_HOST}, Port: {SERVER_PORT}, Reload: {RELOAD_DEV_MODE}")
    print(f"Vault: {os.path.abspath(CONFIG['vault_directory'])}")
    print(f"Log: {os.path.abspath(CONFIG['log_file'])} ({CONFIG['log_level']})")
    print(f"GPU: {has_gpu} ({device})")
    
    if not PYMUPDF_AVAILABLE:
        print("⚠️ PyMuPDF (fitz) not available. Visual PDF processing WILL FAIL.")
    
    uvicorn.run(
        "chatbot_api:app",
        host=SERVER_HOST,
        port=SERVER_PORT,
        reload=RELOAD_DEV_MODE,
        reload_dirs=["./"] if RELOAD_DEV_MODE else None,
        reload_includes=["*.py"] if RELOAD_DEV_MODE else None,
        reload_excludes=["*_client*.py", "temp_uploads/*", "response_tracking/*", "vault_files/*", "*.log", "vector_store/*", "__pycache__/*"] if RELOAD_DEV_MODE else None,
        log_level=CONFIG["log_level"].lower(),
        ws_ping_interval=60,
        ws_ping_timeout=120,
        timeout_keep_alive=75
    )         
        # Limit text to reasonable size
        text = text[:8000]  # First 8K chars
        
        # Create prompt for summary
        prompt = f"""Generate a concise summary of the following document content in 2-3 sentences.
        
Content:
{text}

Summary:"""
        
        # Call Ollama API - Fix: Use ollama.chat() instead of ollama.chat.completions.create()
        response = await asyncio.to_thread(
            lambda: ollama.chat(
                model=CONFIG["ollama_model"],
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": 0.3  # Lower temperature for factual summary
                }
            )
        )
        
        # Extract summary from response - Fix: Update response structure access
        summary = response["message"]["content"].strip()
        return summary
    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}")
        logger.error(traceback.format_exc())
        return "Summary generation failed. The document was processed but no summary is available."

# Modified function to ensure document grounding
def verify_document_grounding(response_text: str, relevant_chunks: List[Dict]) -> Tuple[bool, float, str]:
    """
    Verifies that the response is grounded in the document content.
    Returns a tuple of (is_grounded, grounding_score, modified_response)
    """
    if not relevant_chunks or not response_text:
        return False, 0.0, response_text
    
    # Combine all chunk content into a single document
    document_content = " ".join([chunk["content"] for chunk in relevant_chunks])
    
    # Calculate grounding score based on n-gram overlap
    # This is a simple implementation - more sophisticated methods could be used
    response_words = set(response_text.lower().split())
    document_words = set(document_content.lower().split())
    
    # Calculate word overlap
    if not response_words:
        return False, 0.0, response_text
        
    common_words = response_words.intersection(document_words)
    grounding_score = len(common_words) / len(response_words)
    
    # If grounding score is too low, add a disclaimer and include direct quotes
    if grounding_score < 0.7:  # Threshold for acceptable grounding
        # Find the most relevant chunk based on word overlap
        best_chunk = max(relevant_chunks, key=lambda x: len(set(x["content"].lower().split()).intersection(response_words)))
        
        # Add a disclaimer and the relevant chunk as a direct quote
        modified_response = (
            f"{response_text}\n\n"
            f"<div class='document-quote'><p><i>I've supplemented the answer with this direct content from the document:</i></p>"
            f"<blockquote>{best_chunk['content']}</blockquote></div>"
        )
        
        return False, grounding_score, modified_response
    
    return True, grounding_score, response_text