import os
import json
import logging
import traceback
import torch
import ollama
import sys
import re
import time
import asyncio
import uvicorn
from typing import List, Dict, Tuple, Optional, Set
from openai import OpenAI
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Body # Removed WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field # Removed WebSocketState
from typing import Optional as PydanticOptional
from datetime import datetime
import hashlib
import random
import requests

# Import the configuration
from config import OLLAMA_CONFIG, get_full_config, SERVER_CONFIG

# Replace the hardcoded CONFIG with the imported configuration
CONFIG = get_full_config()

# Create FastAPI application instance
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Modify this to restrict origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define Pydantic models for request and response
class ChatRequest(BaseModel):
    message: str
    selected_files: List[str] = []
    client_id: str = ""
    # Optional parameters with defaults
    ollama_model: PydanticOptional[str] = None
    ollama_embedding_model: PydanticOptional[str] = None
    top_k_per_file: PydanticOptional[int] = None
    similarity_threshold: PydanticOptional[float] = None
    temperature: PydanticOptional[float] = None
    top_p: PydanticOptional[float] = None
    system_prompt: PydanticOptional[str] = None
    # Response formatting options
    clean_response: PydanticOptional[bool] = None
    remove_hedging: PydanticOptional[bool] = None
    remove_references: PydanticOptional[bool] = None
    remove_disclaimers: PydanticOptional[bool] = None
    ensure_html_structure: PydanticOptional[bool] = None
    custom_hedging_patterns: PydanticOptional[List[str]] = None
    custom_reference_patterns: PydanticOptional[List[str]] = None
    custom_disclaimer_patterns: PydanticOptional[List[str]] = None
    html_tags_to_fix: PydanticOptional[List[str]] = None
    # No information response options
    no_info_title: PydanticOptional[str] = None
    no_info_message: PydanticOptional[str] = None
    include_suggestions: PydanticOptional[bool] = None
    custom_suggestions: PydanticOptional[List[str]] = None
    no_info_html_format: PydanticOptional[bool] = None

class ChatResponse(BaseModel):
    message: str
    context: List[Dict] = []
    client_id: str = ""

# Define chat history structure for storage
class ChatHistory(BaseModel):
    chat_id: str
    chat_name: str
    messages: List[Dict[str, str]] = []
    selected_files: List[str] = []
    created_at: str
    updated_at: str
    ai_named: bool = False  # Track if AI has already named this chat

class ChatListItem(BaseModel):
    chat_id: str
    chat_name: str
    last_message: str
    message_count: int
    updated_at: str
    selected_files: List[str]

# +++ LangChain Integration: Imports +++
# from langchain.memory import ConversationBufferMemory # Memory will be handled differently or by client
from langchain.schema import HumanMessage, AIMessage
# +++++++++++++++++++++++++++++++++++++

# --- Configuration --- (Should be loaded from a file or env vars in production)
CONFIG = {
    "vault_file": "vault.txt", # Legacy, less used now
    "log_file": "chatbot.log",
    "log_level": "DEBUG", # Changed default to DEBUG as per original code
    "answer_prefix": "",  # No prefix needed for API
    "response_templates": {
        "error": "I encountered an error while processing your question. Please try again."
    },
    "ollama_model": "llama3",
    "ollama_embedding_model": "mxbai-embed-large",
    "vault_directory": "vault_files/",
    "vault_metadata": "vault_metadata.json",
    "chat_history_directory": "chat_histories/", # Added chat history directory
}

# --- Setup logging FIRST ---
# Ensure basicConfig is called only once at the very beginning.
logging.basicConfig(
    level=getattr(logging, CONFIG["log_level"].upper(), logging.INFO), # Use INFO as default if level is invalid
    filename=CONFIG["log_file"],
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s' # Corrected format string
)
logger = logging.getLogger(__name__)
logger.info("--- Logging initialized ---")

# --- Try importing pypdf AFTER logging is set up ---
try:
    from pypdf import PdfReader
    logger.info("pypdf imported successfully.")
except ImportError:
    logger.error("pypdf not found. PDF processing will fail. Please install it using: pip install pypdf")
    PdfReader = None # Set to None if import fails

# ANSI escape codes for colors - kept for potential direct script runs/debugging
PINK = '\033[95m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'

# Define common section/heading keywords
heading_keywords = [
    "section", "chapter", "part", "introduction", "overview", "summary",
    "conclusion", "references", "appendix", "table of contents", "index",
    "glossary", "discussion", "results", "methodology", "procedure",
    "specifications", "requirements", "features", "instructions"
]

# Check for GPU availability
def setup_device():
    """Initializes and returns the appropriate device for tensor operations."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_info = f"GPU detected: {torch.cuda.get_device_name(0)}"
        logging.info(f"Using GPU acceleration: {gpu_info}")
        print(f"✅ {gpu_info}")

        # Configure PyTorch to use GPU - updated to avoid deprecation warning
        torch.set_default_dtype(torch.float32)
        torch.set_default_device('cuda')

        # Return device info for embedding operations
        return device, True
    else:
        logging.info("No GPU detected, using CPU only")
        print("⚠️ No GPU detected, using CPU. Embeddings will be slower.")
        return torch.device("cpu"), False

# Call this function at startup
device, has_gpu = setup_device()

# Add this parameter to the CONFIG dictionary
CONFIG["use_gpu"] = has_gpu
CONFIG["device"] = device

# --- Dependency and Connectivity Checks ---
def check_dependencies() -> Dict[str, bool]:
    """Checks if all required dependencies are available."""
    status = {
        "torch": False,
        "ollama": False,
        "fastapi": False,
        "pypdf": False,
        "requests": False,
        # +++ LangChain Integration: Check +++
        "langchain": False
        # +++++++++++++++++++++++++++++++++++
    }

    # Check torch
    try:
        import torch
        status["torch"] = True
        logger.info("PyTorch available: ✓")
    except ImportError:
        logger.critical("PyTorch not available. This will affect embedding generation.")
        print("❌ ERROR: PyTorch not found. Install with: pip install torch")
    # Check ollama
    try:
        import ollama
        status["ollama"] = True
        logger.info("Ollama Python client available: ✓")
    except ImportError:
        logger.critical("Ollama Python client not found. This will affect embedding generation.")
        print("❌ ERROR: Ollama client not found. Install with: pip install ollama")
    # Check FastAPI
    try:
        import fastapi
        status["fastapi"] = True
        logger.info("FastAPI available: ✓")
    except ImportError:
        logger.critical("FastAPI not available. The server cannot run.")
        print("❌ ERROR: FastAPI not found. Install with: pip install fastapi uvicorn")
    # Check pypdf (optional)
    try:
        from pypdf import PdfReader
        status["pypdf"] = True
        logger.info("pypdf available: ✓")
    except ImportError:
        logger.warning("pypdf not available. PDF processing will be disabled.")
        print("⚠️ WARNING: pypdf not found. PDF support will be disabled. Install with: pip install pypdf")
    # Check requests
    try:
        import requests
        status["requests"] = True
        logger.info("Requests available: ✓")
    except ImportError:
        logger.critical("Requests not available. API connectivity will be affected.")
        print("❌ ERROR: Requests not found. Install with: pip install requests")

    # +++ LangChain Integration: Check +++
    try:
        import langchain
        status["langchain"] = True
        logger.info("LangChain available: ✓")
    except ImportError:
        logger.critical("LangChain not available. Conversation memory will not work.")
        print("❌ ERROR: LangChain not found. Install with: pip install langchain")
    # +++++++++++++++++++++++++++++++++++

    return status

# Add this function for Ollama connectivity testing
def check_ollama_server() -> Tuple[bool, str, List[str]]:
    """Checks Ollama server connectivity and available models."""
    ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    logger.info(f"Checking Ollama server at {ollama_host}")

    try:
        # Use direct API call instead of ollama client for consistent behavior
        response = requests.get(f"{ollama_host}/api/tags", timeout=5)
        if response.status_code != 200:
            error_msg = f"Failed to get models: HTTP {response.status_code}"
            logger.error(error_msg)
            return False, error_msg, []

        models_data = response.json()
        
        if models_data and "models" in models_data:
            available_models = [model.get("name", "") for model in models_data["models"]]
            model_base_names = set()
            
            for model_name in available_models:
                base_name = model_name.split(':')[0] if ':' in model_name else model_name
                if base_name:
                    model_base_names.add(base_name)
                    
            if available_models:
                logger.info(f"Available models: {', '.join(available_models)}")
                return True, "Connected with models", available_models
            else:
                logger.warning("No models available on Ollama server")
                return True, "Connected but no models available", []
        else:
            logger.warning("Unexpected model list format from Ollama server")
            return True, "Connected (unexpected model list format)", []

    except Exception as e:
        error_msg = f"Error checking Ollama server: {str(e)}"
        logger.error(error_msg)
        return False, error_msg, []

# Initialize Ollama API client
def init_ollama_client():
    """Initializes the OpenAI-compatible client to interact with the local Ollama server."""
    try:
        # Check if OpenAI client is importable
        import openai
        from openai import OpenAI
    except ImportError as e:
        logger.critical(f"Missing required module for Ollama client: {str(e)}")
        print(f"❌ ERROR: Cannot initialize Ollama client - missing module: {str(e)}")
        return None

    # Check if server is running
    server_available, message, available_models = check_ollama_server()
    if not server_available:
        logger.critical(f"Ollama server not available: {message}")
        print(f"❌ ERROR: Ollama server not available - {message}")
        print("ℹ️ Make sure Ollama is running with: ollama serve")
        return None
        
    # Now try to initialize the client with better error handling
    try:
        # Try using the OpenAI client with V1 endpoint
        client = OpenAI(
            base_url='http://localhost:11434/v1',
            api_key='ollama' # Required by OpenAI client, but Ollama ignores it
        )
        
        # Check if the configured models are available
        if available_models:
            # Check if the configured models are available using base names
            missing_models = []
            ollama_base_model = CONFIG["ollama_model"].split(':')[0]
            embed_base_model = CONFIG["ollama_embedding_model"].split(':')[0]
            
            # Create set of base model names for easier checking
            model_base_names = set()
            for model_name in available_models:
                base_name = model_name.split(':')[0] if ':' in model_name else model_name
                if base_name:
                    model_base_names.add(base_name)
            
            if ollama_base_model not in model_base_names:
                missing_models.append(CONFIG["ollama_model"])
            if embed_base_model not in model_base_names:
                missing_models.append(CONFIG["ollama_embedding_model"])
                
            if missing_models:
                logger.warning(f"Configured models not found: {', '.join(missing_models)}")
                print(f"⚠️ WARNING: Configured models not found: {', '.join(missing_models)}")
                print(f"   Available models: {', '.join(available_models)}")
                print(f"   Please run: ollama pull {' '.join(missing_models)}")
            else:
                logger.info(f"All required Ollama models are available")
                print(f"✅ Required Ollama models available: {CONFIG['ollama_model']}, {CONFIG['ollama_embedding_model']}")
        else:
            logger.warning("No models available on Ollama server")
            print("⚠️ WARNING: No models available on Ollama server")
            
        logger.info("Ollama API client initialized successfully")
        return client
    except Exception as e:
        logger.error(f"Failed to initialize Ollama API client: {str(e)}")
        logger.error(traceback.format_exc())
        print(f"❌ ERROR: Failed to initialize Ollama client - {str(e)}")
        return None

# Initialize vault directory
def initialize_vault_directory():
    """Creates the vault directory structure if it doesn't exist."""
    # Create vault directory if it doesn't exist
    os.makedirs(CONFIG["vault_directory"], exist_ok=True)
    metadata_path = os.path.join(CONFIG["vault_directory"], CONFIG["vault_metadata"])
    if not os.path.exists(metadata_path):
        try:
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump({"files": []}, f, indent=2) # Add indent for readability
            logger.info("Created vault metadata file")
        except IOError as e:
            logger.error(f"Failed to create vault metadata file {metadata_path}: {e}")

# Initialize chat history directory
def initialize_chat_history_directory():
    """Creates the chat history directory if it doesn't exist."""
    os.makedirs(CONFIG["chat_history_directory"], exist_ok=True)
    logger.info("Chat history directory initialized")

# Initialize the global Ollama client
ollama_client = init_ollama_client()

# Function to read vault content
def read_vault_content(selected_files: List[str] = None) -> Dict[str, List[str]]:
    """Reads pre-processed content chunks from selected vault files."""
    content_by_file = {}
    if not selected_files:
        logger.warning("read_vault_content called with no selected files.")
        return content_by_file # Return empty if no files selected

    # Read content from each selected file
    for filename in selected_files:
        # Basic check to prevent directory traversal
        if ".." in filename or filename.startswith("/"):
             logger.warning(f"Skipping potentially unsafe filename: {filename}")
             continue

        file_path = os.path.join(CONFIG["vault_directory"], filename)
        if not os.path.exists(file_path):
            logging.warning(f"Selected file not found, skipping: {filename}")
            continue
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                # Read lines, assuming each line is a pre-chunked piece of text
                chunks = [line for line in f.read().splitlines() if line.strip()] # Simple split by line
                if chunks:
                    content_by_file[filename] = chunks
                    logging.debug(f"Read {len(chunks)} pre-processed chunks from {filename}") # Changed to debug
                else:
                    logging.warning(f"No content chunks found in file: {filename}")
        except Exception as e:
            logging.error(f"Error reading file {filename}: {e}")

    return content_by_file

# Generate embeddings for vault content using Ollama
async def generate_vault_embeddings(content_by_file: Dict[str, List[str]]) -> Dict[str, torch.Tensor]: # Removed client_id
    """Generates embeddings for each file's content chunks with GPU acceleration if available."""
    embeddings_by_file = {}
    total_files = len(content_by_file)
    processed_files = 0
    device = CONFIG.get("device", torch.device("cpu")) # Use configured device

    for filename, chunks in content_by_file.items():
        processed_files += 1
        if not chunks:
            logger.info(f"No content chunks in {filename}, skipping embeddings generation.")
            continue

        logger.info(f"Generating embeddings for {len(chunks)} chunks in {filename} ({processed_files}/{total_files})...")
        # status_message_start = f"Generating embeddings for {filename} ({processed_files}/{total_files})..." # Removed status message
        # if client_id: await manager.send_json(client_id, {"type": "status", "message": status_message_start}) # Removed manager call

        file_embeddings = []
        count = 0
        total_chunks = len(chunks)
        start_time = time.time()
        try:
            # Process chunks in batches to avoid too many concurrent tasks
            batch_size = 10  # Process 10 chunks at a time
            for batch_start in range(0, len(chunks), batch_size):
                batch_end = min(batch_start + batch_size, len(chunks))
                batch_chunks = chunks[batch_start:batch_end]

                # Create tasks for each chunk in batch
                embedding_tasks = []
                for chunk in batch_chunks:
                    if chunk.strip():
                        # Run embedding in thread pool to avoid blocking
                        embedding_tasks.append(
                            asyncio.to_thread(
                                lambda text=chunk.strip(): ollama.embeddings(
                                    model=CONFIG["ollama_embedding_model"],
                                    prompt=text
                                )
                            )
                        )

                # Run all batch tasks concurrently and gather results
                batch_results = await asyncio.gather(*embedding_tasks, return_exceptions=True)

                # Process batch results
                for result in batch_results:
                    if isinstance(result, Exception):
                        logger.error(f"Error generating embedding: {str(result)}")
                        continue # Skip failed chunks

                    if "embedding" in result:
                        file_embeddings.append(result["embedding"])
                        count += 1
                    else:
                        logger.warning(f"Ollama embedding response missing 'embedding' key: {result}")


                # Update progress after each batch
                if count % 50 == 0 or count == total_chunks:
                    elapsed_time = time.time() - start_time
                    progress = f"({count}/{total_chunks} chunks, {elapsed_time:.1f}s)"
                    logger.info(f"...processed {progress} for {filename}.")
                    # status_message_progress = f"{status_message_start} {progress}" # Removed status message
                    # if client_id: # Removed manager call
                    #     await manager.send_json(client_id, {"type": "status", "message": status_message_progress})

            # Create tensor from embeddings with device awareness
            if file_embeddings:
                try:
                    # Ensure embeddings are valid before tensor creation
                    if all(isinstance(e, list) for e in file_embeddings):
                        embeddings_tensor = torch.tensor(file_embeddings, dtype=torch.float32)
                        # Move to appropriate device (GPU if available)
                        embeddings_tensor = embeddings_tensor.to(device)
                        embeddings_by_file[filename] = embeddings_tensor
                        logger.info(f"Generated tensor of shape {embeddings_by_file[filename].shape} for {filename} on device {device}")
                    else:
                         logger.error(f"Invalid embedding data type for tensor creation in {filename}.")

                except Exception as e:
                    logger.error(f"Error converting embeddings to tensor for {filename}: {str(e)}")
                    logger.error(traceback.format_exc()) # Log full traceback for tensor errors
            else:
                logger.warning(f"No embeddings were generated for {filename} (all chunks might have failed or were empty).")

        except Exception as e:
             logger.error(f"Unexpected error during embedding generation for {filename}: {str(e)}")
             logger.error(traceback.format_exc())
             # Continue to the next file

    logger.info("Finished generating embeddings for all selected files.")
    # if client_id: await manager.send_json(client_id, {"type": "status", "message": "Embedding generation complete."}) # Removed manager call
    return embeddings_by_file


# Text processing functions
def preprocess_document(text: str) -> str:
    """Preprocess document text to improve chunking quality"""
    # Fix section numbers running into text (like "5.2LED Information")
    text = re.sub(r'(\d+\.\d+)(\w)', r'\1 \2', text)

    # Add space after periods followed immediately by uppercase letters
    text = re.sub(r'\.([A-Z])', r'. \1', text)

    # Ensure section headers stand out (like V3 exactly)
    text = re.sub(r'(\d+(\.\d+)*)\s+([A-Za-z])', r'\n\1 \3', text)  # Preserves section headers
    # Normalize whitespace but preserve paragraph breaks
    text = re.sub(r'\s*\n\s*\n\s*', '\n\n', text)
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

def chunk_text(text: str, max_chunk_size: int = 800, overlap: int = 0, paragraph_separator: str = "\n\n") -> List[str]:
    """
    Splits text into chunks, preserving semantic structure where possible.
    
    Args:
        text: The text to chunk
        max_chunk_size: Maximum size of each chunk (default: 800)
        overlap: Number of characters to overlap between chunks (default: 0)
        paragraph_separator: String used to separate paragraphs (default: "\n\n")
    
    Returns:
        List of text chunks
    """
    # First, try to split by double newlines (paragraph boundaries) or the specified separator
    paragraphs = re.split(r'\n\s*\n' if paragraph_separator == "\n\n" else re.escape(paragraph_separator), text)
    chunks = []
    current_chunk = ""

    # Process each paragraph - exactly like V3
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        # Check if adding this paragraph exceeds the max size
        if len(current_chunk) + len(paragraph) + 2 <= max_chunk_size:
            # Add paragraph to current chunk
            if current_chunk:
                current_chunk += "\n\n" + paragraph
            else:
                current_chunk = paragraph
        else:
            # If current chunk is not empty, add it to chunks
            if current_chunk:
                chunks.append(current_chunk)

            # If paragraph itself is larger than max size, we need to split it
            if len(paragraph) > max_chunk_size:
                # Try to split by sentence boundaries first
                sentences = re.split(r'(?<=[.!?])\s*', paragraph)
                current_chunk = ""

                for sentence in sentences:
                    if not sentence.strip():
                        continue

                    if len(current_chunk) + len(sentence) + 1 <= max_chunk_size:
                        if current_chunk:
                            current_chunk += " " + sentence
                        else:
                            current_chunk = sentence
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)

                        # If single sentence exceeds max size, hard split it
                        if len(sentence) > max_chunk_size:
                            # Attempt to split on word boundaries when possible
                            words = sentence.split()
                            temp_chunk = ""

                            for word in words:
                                if len(temp_chunk) + len(word) + 1 <= max_chunk_size:
                                    if temp_chunk:
                                        temp_chunk += " " + word
                                    else:
                                        temp_chunk = word
                                else:
                                    chunks.append(temp_chunk)
                                    temp_chunk = word

                            if temp_chunk:
                                current_chunk = temp_chunk
                            else:
                                current_chunk = ""
                        else:
                            current_chunk = sentence
            else:
                # Start fresh with this paragraph
                current_chunk = paragraph

    # Add the last chunk if not empty
    if current_chunk:
        chunks.append(current_chunk)

    # Verify no empty chunks
    chunks = [c.strip() for c in chunks if c.strip()]

    return chunks


# --- File Processing Logic ---
def extract_text_from_pdf(file_path: str) -> str:
    """Extract text content from a PDF file."""
    if not PdfReader:
        logger.error("PDF processing requires 'pypdf'. Please install it.")
        return ""

    try:
        reader = PdfReader(file_path)
        text = ""

        for i, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + " "  # Add space between pages
            except Exception as page_e:
                logger.warning(f"Could not extract text from page {i+1} of PDF: {page_e}")

        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF {file_path}: {str(e)}")
        return ""

async def process_uploaded_file(temp_file_path, file_type, uploaded_filename, description=None):
    """Process the uploaded file to match V3's file processing EXACTLY."""
    try:
        # Initialize tags variable at the function start
        tags = []

        # Ensure vault directory exists
        os.makedirs(CONFIG["vault_directory"], exist_ok=True)

        # Read content based on file type
        if (file_type.lower() == "pdf" and PdfReader):
            content = extract_text_from_pdf(temp_file_path)
            if not content:
                logger.error(f"Failed to extract text from PDF: {uploaded_filename}")
                return False, None
            logger.info(f"Successfully extracted text from PDF: {uploaded_filename}")
        elif file_type.lower() in ["txt", "md", "json", "py", "js", "html", "css", "csv"]: # Handle common text types
            try:
                with open(temp_file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                logger.info(f"Successfully read text file content: {len(content)} bytes")
            except Exception as e:
                logger.error(f"Error reading text file {uploaded_filename}: {str(e)}")
                return False, None
        else:
             logger.warning(f"Unsupported file type '{file_type}' for direct text reading: {uploaded_filename}. Treating as generic.")
             # Fallback: try reading as binary and decoding with ignore
             try:
                with open(temp_file_path, "rb") as f:
                    raw_content = f.read()
                content = raw_content.decode("utf-8", errors="ignore")
                logger.info(f"Read unsupported file type as text (with potential decoding errors): {len(content)} bytes")
             except Exception as e:
                logger.error(f"Error reading generic file {uploaded_filename}: {str(e)}")
                return False, None


        # Generate a unique filename
        safe_filename = f"{int(time.time())}_{re.sub(r'[^\w\.-]', '_', uploaded_filename)}"
        output_path = os.path.join(CONFIG["vault_directory"], safe_filename)

        # Log the raw content length for debugging
        logger.debug(f"Raw content length before preprocessing: {len(content)} chars")

        # Preprocess and chunk the text EXACTLY like V3
        processed_text = preprocess_document(content)
        logger.debug(f"Processed text length: {len(processed_text)} chars")

        # Use exactly the same chunk_text function as V3 with same parameters
        chunks = chunk_text(processed_text, max_chunk_size=800, overlap=0)  # V3 uses 800 as max size
        logger.info(f"Created {len(chunks)} chunks for {safe_filename}")

        # Save the chunked text exactly like V3 does
        with open(output_path, "w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(chunk + "\n")

        # Default description if none provided (AI generation happens later if needed)
        if description is None:
            description = f"Uploaded {file_type.upper()} file: {uploaded_filename}"

        # Add file to metadata (AI generation happens in the /upload endpoint)
        if not add_file_to_vault(safe_filename, description, tags):  # Pass initial empty tags
            logger.error(f"Failed to update metadata for {safe_filename}")
            return False, None

        # Note: Summary generation is triggered from the /upload endpoint AFTER AI metadata generation

        return True, safe_filename
    except Exception as e:
        logger.error(f"Error processing uploaded file: {str(e)}")
        logger.error(traceback.format_exc())
        return False, None


async def generate_file_metadata(file_content: str, client: OpenAI) -> Tuple[Optional[str], List[str]]:
    """Uses the AI model to generate a description and keywords for a file based on its content."""
    if not client:
        logger.warning("Ollama client not available for metadata generation.")
        return None, []
    try:
        # Truncate content if too long to avoid token limits (match V3 exactly)
        sample_content = file_content[:10000] if len(file_content) > 10000 else file_content
        if not sample_content.strip():
            logger.warning("File content is empty or whitespace, cannot generate metadata.")
            return "File is empty or contains only whitespace.", []

        # Use EXACTLY the same prompt as Chatbot_V3.py
        prompt = f"""Analyze the following document content and provide:
1. A concise description (1-2 sentences) summarizing what this document is about.
2. 3-7 relevant keywords or tags (comma-separated).

Content:
{sample_content}

Respond ONLY in this exact format, with no extra text before or after:
DESCRIPTION: ** [your generated description]
KEYWORDS: ** [keyword1, keyword2, keyword3, ...]"""

        # Use system message exactly like V3
        response = await asyncio.to_thread( # Use thread to avoid blocking API call
            lambda: client.chat.completions.create(
                model=CONFIG["ollama_model"],
                messages=[{"role": "system", "content": prompt}],
                temperature=0.1 # Low temperature for factual description
            )
        )

        result = response.choices[0].message.content
        logger.info(f"AI metadata generation response: {result[:100]}...")

        # Parse the response using exact pattern from V3
        description = None # Default to None if not found
        keywords = []

        desc_match = re.search(r'DESCRIPTION:\s*\*\*\s*(.*?)(?:\nKEYWORDS:|\Z)', result, re.DOTALL | re.IGNORECASE)
        if desc_match:
            description = desc_match.group(1).strip()
            # Handle potential leading/trailing noise if KEYWORDS wasn't perfectly matched
            description = description.split('KEYWORDS:')[0].strip()


        keywords_match = re.search(r'KEYWORDS:\s*\*\*\s*(.*?)(?:\n|$)', result, re.DOTALL | re.IGNORECASE)
        if keywords_match:
            keywords_text = keywords_match.group(1).strip()
            keywords = [k.strip() for k in keywords_text.split(',') if k.strip()]

        if not description and not keywords:
             logger.warning(f"Could not parse DESCRIPTION or KEYWORDS from AI response: {result}")
             return f"AI failed to parse description.", []


        return description, keywords
    except Exception as e:
        logger.error(f"Error generating file metadata: {str(e)}")
        logger.error(traceback.format_exc())
        return f"Error during AI metadata generation: {str(e)}", []


async def generate_summary_on_upload(filename: str): # Removed client_id
    """Generates a summary for a newly uploaded file."""
    try:
        if not ollama_client:
            logger.warning(f"Cannot generate summary for {filename}, Ollama client not available.")
            return None

        # Get file content
        content_by_file = read_vault_content([filename])
        if not content_by_file or filename not in content_by_file:
            logger.error(f"Failed to read content for summary: {filename}")
            return None

        # Check if content is meaningful
        if not any(chunk.strip() for chunk in content_by_file[filename]):
            logger.warning(f"Skipping summary generation for {filename}, content is empty or whitespace.")
            update_file_metadata(filename, description="File content appears empty.")
            return None

        # Generate embeddings (needed for context selection, even for summary)
        embeddings_by_file = await generate_vault_embeddings(content_by_file) # Removed client_id
        if not embeddings_by_file:
            logger.warning(f"Failed to generate embeddings for {filename}, cannot generate summary.")
            return None

        # Generate summary
        logger.info(f"Generating automatic summary for uploaded file: {filename}")

        # Use a simplified client ID if none provided, append timestamp
        summary_client_id = f"auto_summary_{filename}_{int(time.time())}" # Use filename for more specific ID

        # Trigger summary generation
        # Note: We pass the client_id so status updates can be sent if a user is connected
        summary = await ollama_chat_multifile_async(
            user_input="FORCE_SUMMARY_GENERATION", # Use the special trigger
            selected_files=[filename],
            embeddings_by_file=embeddings_by_file,
            content_by_file=content_by_file,
            client_id=summary_client_id, 
            force_summary=True # Explicitly force summary mode
        )

        # Store the summary in file metadata if generated
        if summary and "error" not in summary.lower() and "i cannot" not in summary.lower():
            # Extract a short description from the summary
            short_desc = ""
            # Try finding the first heading
            match = re.search(r'<h[2-6]>(.*?)</h[2-6]>', summary, re.IGNORECASE | re.DOTALL)
            if match:
                short_desc = match.group(1).strip()
                # Clean potential residual HTML within the heading
                short_desc = re.sub(r'<.*?>', '', short_desc)
                short_desc = short_desc[:250] + ("..." if len(short_desc) > 250 else "") 


            if not short_desc:
                # Fall back to first 200 chars without HTML
                clean_summary = re.sub(r'<.*?>', '', summary).strip()
                # Take first few sentences or up to 250 chars
                sentences = re.split(r'(?<=[.!?])\s+', clean_summary)
                short_desc = ""
                char_count = 0
                for s in sentences:
                    if char_count + len(s) < 250:
                        short_desc += s + " "
                        char_count += len(s) + 1
                    else:
                        break
                short_desc = short_desc.strip()
                if len(clean_summary) > len(short_desc):
                    short_desc += "..."


            if not short_desc: # Ultimate fallback
                 short_desc = f"Summary generated for {filename}"

            # Update metadata with summary info, keep existing tags if possible
            existing_meta = get_vault_files()
            current_tags = []
            for f_meta in existing_meta:
                if f_meta.get("filename") == filename:
                    current_tags = f_meta.get("tags", [])
                    break
            new_tags = sorted(list(set(current_tags + ["auto-summarized"])))


            update_file_metadata(
                filename,
                description=f"AUTO-SUMMARY: {short_desc}",
                tags=new_tags
            )

            # Store full summary (optional - maybe in a separate file or DB later)
            # For now, we just update the description.
            logger.info(f"Auto-summary generated and description updated for {filename}")

            # If client_id was provided and they are connected, send the summary
            # This part is removed as manager and WebSocket connections are gone.
            # if client_id and client_id in manager.active_connections:
            #     await manager.send_json(client_id, {
            #         "type": "file_summary",
            #         "filename": filename,
            #         "summary": summary # Send the full HTML summary
            #     })

            return summary # Return the generated summary text
        else:
            logger.warning(f"Auto-summary generation for {filename} did not produce a valid result. Summary was: '{summary[:100]}...'")
            update_file_metadata(filename, description="Summary generation failed or yielded no result.")
            return None

    except Exception as e:
        logger.error(f"Error generating auto-summary for {filename}: {str(e)}")
        logger.error(traceback.format_exc())
        return None


# Duplicated function, ensure only one remains or they are distinct. Assuming this is a duplicate to be removed or was an alternative version.
# async def generate_summary_on_upload(filename: str, client_id: str = None):
#     """Generates a summary for a newly uploaded file."""
#     try:
#         # Get file content
#         content_by_file = read_vault_content([filename])
#         if not content_by_file:
#             logger.error(f"Failed to read content for summary: {filename}")
#             return
            
#         # Generate embeddings
#         embeddings_by_file = await generate_vault_embeddings(content_by_file)
        
#         # Generate summary
#         logger.info(f"Generating automatic summary for uploaded file: {filename}")
        
#         # Use a simplified client ID if none provided
#         summary_client_id = client_id or f"auto_summary_{int(time.time())}"
        
#         # Force summary generation with standard settings
#         summary = await ollama_chat_multifile_async(
#             user_input="Generate a comprehensive summary of the document",
#             selected_files=[filename],
#             embeddings_by_file=embeddings_by_file,
#             content_by_file=content_by_file,
#             client_id=summary_client_id,
#             force_summary=True  # Force summary mode
#         )
        
#         # Store the summary in file metadata
#         if summary:
#             # Extract a short description from the summary (first 200 chars or first heading content)
#             short_desc = ""
#             if "<h2>" in summary:
#                 match = re.search(r'<h2>(.*?)</h2>', summary)
#                 if match:
#                     short_desc = match.group(1)
            
#             if not short_desc:
#                 # Fall back to first 200 chars without HTML
#                 clean_summary = re.sub(r'<.*?>', '', summary).strip()
#                 short_desc = clean_summary[:200] + ("..." if len(clean_summary) > 200 else "")
            
#             # Update metadata with summary data
#             update_file_metadata(
#                 filename, 
#                 description=f"AUTO-SUMMARY: {short_desc}",
#                 tags=["auto-summarized"]  # Add a tag to indicate auto-summarization
#             )
            
#             # Store full summary in a separate metadata field or file if needed
#             # This would require extending your metadata structure
            
#             logger.info(f"Auto-summary generated for {filename}")
            
#             # If client_id was provided, send the summary to the client
#             if client_id and client_id in manager.active_connections: # This part would be removed
#                 await manager.send_json(client_id, {
#                     "type": "file_summary",
#                     "filename": filename,
#                     "summary": summary
#                 })
                
#         return summary
#     except Exception as e:
#         logger.error(f"Error generating auto-summary for {filename}: {str(e)}")
#         logger.error(traceback.format_exc())
#         return None

# --- End File Processing ---

async def generate_document_questions(document_content: str, filename: str, client: OpenAI) -> List[Dict]:
    """Generates suggested questions for a document using the AI model."""
    if not client:
        logger.warning("Ollama client not available for question generation.")
        return []
    
    try:
        # Truncate content if too long to avoid token limits
        sample_content = document_content[:8000] if len(document_content) > 8000 else document_content
        if not sample_content.strip():
            logger.warning(f"File content is empty or whitespace for {filename}, cannot generate questions.")
            return []

        # Create a prompt for question generation
        prompt = f"""Based on the following document content, generate 5 relevant and specific questions that a user might ask about this document.
        
Document: {filename}
Content:
{sample_content}

Generate 5 questions in JSON format as follows:
[
  {{"question": "Question 1 about specific content?"}},
  {{"question": "Question 2 about another specific detail?"}},
  ...
]

Important: Make questions specific to the actual document content, not generic.
Only return the JSON array, with no additional text."""

        # Call the Ollama API
        response = await asyncio.to_thread(
            lambda: client.chat.completions.create(
                model=CONFIG["ollama_model"],
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates relevant questions about documents."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7  # Slightly higher temperature for creative questions
            )
        )

        result = response.choices[0].message.content
        logger.info(f"AI question generation response: {result[:100]}...")

        # Try to parse the JSON response
        try:
            # Look for a JSON array in the response
            json_match = re.search(r'\[\s*\{.*\}\s*\]', result, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                questions = json.loads(json_str)
                # Ensure proper format of questions
                validated_questions = []
                for item in questions:
                    if isinstance(item, dict) and "question" in item:
                        validated_questions.append({"question": item["question"]})
                return validated_questions
            else:
                # Try to parse the whole response as JSON
                questions = json.loads(result)
                validated_questions = []
                for item in questions:
                    if isinstance(item, dict) and "question" in item:
                        validated_questions.append({"question": item["question"]})
                return validated_questions
        except json.JSONDecodeError:
            # If JSON parsing fails, try to extract questions with regex
            questions = []
            question_patterns = [
                r'"question":\s*"([^"]*?)"',  # Match "question": "text" with non-greedy matching
                r'\d+\.\s+([^?]+\?)',  # Match numbered lists with question marks: 1. Question text?
                r'\d+\)\s+([^?]+\?)',  # Match numbered lists with parentheses: 1) Question text?
                r'"([^"]+\?)"',  # Match quoted questions with question marks
                r'[\n\r]\s*([^"\n\r][^?\n\r]*\?)'  # Match standalone questions with question marks
            ]
            
            for pattern in question_patterns:
                matches = re.findall(pattern, result, re.DOTALL)
                if matches:
                    for match in matches[:5]:  # Limit to 5 questions
                        question_text = match.strip()
                        if question_text and len(question_text) > 10:  # Basic validation for question length
                            questions.append({"question": question_text})
                    if questions:  # Only break if we found valid questions
                        break  # Stop if we found questions with this pattern
            
            if questions:
                return questions
            
            # Last resort: split by newlines and look for question marks
            lines = result.split('\n')
            for line in lines:
                line = line.strip()
                if '?' in line and len(line) > 15:  # Basic validation
                    # Extract just the question part
                    question_part = re.sub(r'^\d+[\.\)]\s*', '', line)  # Remove leading numbers
                    question_part = re.sub(r'^["\']*|["\']*$', '', question_part)  # Remove quotes
                    if question_part and "question" not in question_part.lower()[:15]:  # Avoid matching JSON fragments
                        questions.append({"question": question_part})
            
            return questions[:5]  # Return up to 5 questions
    except Exception as e:
        logger.error(f"Error generating questions for {filename}: {str(e)}")
        logger.error(traceback.format_exc())
        return []

# File management functions
def get_vault_files() -> List[Dict]:
    """Returns metadata for all available files in the vault."""
    try:
        metadata_path = os.path.join(CONFIG["vault_directory"], CONFIG["vault_metadata"])
        if not os.path.exists(metadata_path):
            logger.warning(f"Metadata file not found: {metadata_path}")
            return []

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        if not isinstance(metadata, dict) or "files" not in metadata:
            logger.warning(f"Invalid metadata format in {metadata_path}")
            # Attempt to fix by creating a valid structure
            metadata = {"files": []}
            try:
                with open(metadata_path, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=2)
                logger.info(f"Created valid metadata structure in {metadata_path}")
            except IOError as e_fix:
                logger.error(f"Failed to fix metadata file {metadata_path}: {e_fix}")
                return []


        # Filter for files that actually exist on disk
        files_with_metadata = []
        valid_files_in_metadata = []
        needs_update = False
        for file_entry in metadata.get("files", []):
            if not isinstance(file_entry, dict) or "filename" not in file_entry:
                logger.warning(f"Skipping invalid entry in metadata: {file_entry}")
                needs_update = True
                continue

            filename = file_entry.get("filename")
            if not filename: # Skip entries with empty filenames
                 logger.warning(f"Skipping metadata entry with empty filename.")
                 needs_update = True
                 continue

            file_path = os.path.join(CONFIG["vault_directory"], filename)

            if os.path.exists(file_path):
                # Ensure basic fields exist
                file_entry.setdefault("description", f"File: {filename}")
                file_entry.setdefault("tags", [])
                file_entry.setdefault("added_date", time.strftime("%Y-%m-%d %H:%M:%S"))
                file_entry.setdefault("updated_date", file_entry["added_date"])
                files_with_metadata.append(file_entry)
                valid_files_in_metadata.append(file_entry) # Keep track of valid ones separately
            else:
                logger.warning(f"File in metadata not found on disk, removing entry: {filename}")
                needs_update = True # Mark metadata for update

        # If we detected missing files or invalid entries, update the metadata file
        if needs_update:
            logger.info("Updating metadata file to remove missing files/invalid entries...")
            updated_metadata = {"files": valid_files_in_metadata}
            try:
                with open(metadata_path, "w", encoding="utf-8") as f:
                    json.dump(updated_metadata, f, indent=2)
                logger.info("Metadata file updated successfully.")
            except IOError as e_write:
                logger.error(f"Failed to write updated metadata file {metadata_path}: {e_write}")
                # Return the list based on disk check, even if metadata update failed
                return files_with_metadata


        # Sort files by added date descending (newest first)
        files_with_metadata.sort(key=lambda x: x.get("added_date", "1970-01-01 00:00:00"), reverse=True)

        return files_with_metadata

    except json.JSONDecodeError:
        logger.error(f"Error parsing metadata file: {metadata_path}. Attempting to reset.")
        try:
             with open(metadata_path, "w", encoding="utf-8") as f:
                 json.dump({"files": []}, f, indent=2)
             logger.info("Metadata file was corrupted and has been reset.")
             return []
        except IOError as e_reset:
             logger.error(f"Failed to reset corrupted metadata file {metadata_path}: {e_reset}")
             return []
    except Exception as e:
        logger.error(f"Error reading vault files: {str(e)}")
        logger.error(traceback.format_exc())
        return []


def add_file_to_vault(filename: str, description: str, tags: List[str] = None, suggested_questions: List[Dict] = None) -> bool:
    """Adds or updates a file entry in the vault metadata."""
    try:
        metadata_path = os.path.join(CONFIG["vault_directory"], CONFIG["vault_metadata"])
        if not os.path.exists(metadata_path):
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump({"files": []}, f, indent=2)
            logger.info(f"Created metadata file: {metadata_path}")

        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except json.JSONDecodeError:
            logger.warning(f"Metadata file {metadata_path} was corrupted. Resetting.")
            metadata = {"files": []}

        if not isinstance(metadata, dict) or "files" not in metadata:
            logger.warning("Invalid metadata format detected. Resetting structure.")
            metadata = {"files": []}

        # Standardize tags: lowercase, unique, strip whitespace
        processed_tags = sorted(list(set([t.lower().strip() for t in tags if t and isinstance(t, str) and t.strip()]))) if tags else []

        # Check if file already exists to update it, otherwise add new
        found_index = -1
        current_files = metadata.get("files", [])
        for i, existing_file in enumerate(current_files):
            if isinstance(existing_file, dict) and existing_file.get("filename") == filename:
                found_index = i
                break

        now_time = time.strftime("%Y-%m-%d %H:%M:%S")

        file_entry = {
            "filename": filename,
            "description": description or f"File: {filename}", # Ensure description isn't empty
            "tags": processed_tags,
            "added_date": now_time, # Default to now for new entries
            "updated_date": now_time, # Always set update time
            "suggested_questions": suggested_questions or [] # Store the AI-generated questions
        }
        
        if found_index != -1:
             # Update existing entry, preserve original added_date
             original_added_date = current_files[found_index].get("added_date", now_time)
             # Merge tags: Keep existing, add new, ensure uniqueness
             existing_tags = current_files[found_index].get("tags", [])
             merged_tags = sorted(list(set(existing_tags + processed_tags)))

             file_entry["added_date"] = original_added_date
             file_entry["tags"] = merged_tags # Use merged tags for updates
             
             # Only update description if a new one was provided
             if description:
                 file_entry["description"] = description
             else:
                 file_entry["description"] = current_files[found_index].get("description", f"File: {filename}")
                 
             # Preserve existing questions if none provided
             if not suggested_questions and "suggested_questions" in current_files[found_index]:
                 file_entry["suggested_questions"] = current_files[found_index]["suggested_questions"]

             metadata["files"][found_index] = file_entry
             logger.info(f"Updated metadata for existing file: {filename}")
        else:
            # Add new entry
            metadata.setdefault("files", []).append(file_entry)
            logger.info(f"Added new file to metadata: {filename}")
        
        # Save updated metadata
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        
        return True

    except Exception as e:
        logger.error(f"Error adding/updating file '{filename}' in vault metadata: {str(e)}")
        logger.error(traceback.format_exc()) # Log full traceback for debugging
        return False


def update_file_metadata(filename: str, description: Optional[str] = None, 
                        tags: Optional[List[str]] = None, 
                        suggested_questions: Optional[List[Dict]] = None) -> bool:
    """Updates metadata for an existing file. Merges tags if provided."""
    try:
        metadata_path = os.path.join(CONFIG["vault_directory"], CONFIG["vault_metadata"])
        if not os.path.exists(metadata_path):
             logger.error(f"Metadata file not found: {metadata_path}")
             return False # Cannot update if file doesn't exist

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        if not isinstance(metadata, dict) or "files" not in metadata:
             logger.error(f"Invalid metadata format in {metadata_path}")
             return False

        # Find the file in metadata
        file_found = False
        for i, file_entry in enumerate(metadata.get("files", [])):
             if isinstance(file_entry, dict) and file_entry.get("filename") == filename:
                 # Update existing entry
                 if description is not None: # Allow empty string description
                     file_entry["description"] = description
                     
                 if tags is not None: # Check if tags list was provided (even if empty)
                     # Standardize new tags
                     processed_new_tags = sorted(list(set([t.lower().strip() for t in tags if t and isinstance(t, str) and t.strip()])))
                     # Merge with existing tags
                     existing_tags = file_entry.get("tags", [])
                     file_entry["tags"] = sorted(list(set(existing_tags + processed_new_tags)))
                 
                 # Update suggested questions if provided
                 if suggested_questions is not None:
                     file_entry["suggested_questions"] = suggested_questions

                 # Update modification time
                 file_entry["updated_date"] = time.strftime("%Y-%m-%d %H:%M:%S")
                 metadata["files"][i] = file_entry # Update the entry in the list
                 file_found = True
                 logger.info(f"Updated metadata fields for {filename}")
                 break # Found the file, no need to continue loop

        if not file_found:
             logger.warning(f"Tried to update metadata for '{filename}', but it was not found in the metadata file.")
             # Optionally, add it if description or tags were provided?
             # For now, let's just return False if not found.
             return False

        # Save updated metadata
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Metadata file saved successfully after updating {filename}.")
        return True

    except json.JSONDecodeError:
         logger.error(f"Error parsing metadata file during update: {metadata_path}")
         return False
    except Exception as e:
        logger.error(f"Error updating metadata for {filename}: {str(e)}")
        logger.error(traceback.format_exc())
        return False


# Query parsing and context retrieval
def parse_file_query(query: str, available_files: List[str], selected_files: List[str] = None) -> Tuple[str, List[str]]:
    """
    Smarter query parsing that considers already selected files and different phrasings.
    Defaults to searching all selected files if query is ambiguous regarding targets.

    Args:
        query: The user's query.
        available_files: List of all available filenames in the vault.
        selected_files: List of currently selected filenames by the user.

    Returns:
        cleaned_query: The query string with file specifiers removed.
        target_files: List of target filenames identified from the query or context.
    """
    if selected_files is None:
        selected_files = []

    if not available_files:
        logger.debug("parse_file_query: No available files in vault.")
        return query, []

    original_query = query.strip()
    query_lower = original_query.lower()
    # Create a mapping of lowercase filenames to original case for matching
    available_files_lower = {f.lower(): f for f in available_files}

    # --- Patterns to Extract Filename and Query ---

    # Pattern 1: Keywords at start, then file(s), then separator, then query
    # e.g., "in file X about Y", "compare file A and B on Z"
    pattern_keyword_file_query = r"(?i)(?:in|from|using|search|check|within|compare|contrast|difference between)\s+(?:file|document|doc)[s]?\s+((?:[a-zA-Z0-9_\.\-]+(?:(?:[,]?\s+|\s+and\s+)\s*[a-zA-Z0-9_\.\-])*))\s*(?:[:\-,\s]|on|about|regarding|with\srespect\sto)\s*(.+)"

    # Pattern 2: Simple "file X : query" or "doc X query" structure
    pattern_file_colon_query = r"(?i)(?:file|document|doc)[s]?\s+([a-zA-Z0-9_\.\-]+)\s*[:\-,\s]\s*(.+)"

    # Pattern 3: Query first, then keyword, then filename at the end
    # e.g., "what is X in file Y", "tell me about Z from doc A", "navigation in manual B?"
    # Made the space after the optional "file/doc" also optional using \s*
    pattern_query_keyword_file = r"(.+?)\s+(?:in|from|about)\s+(?:the\s+)?(?:file|document|doc)?\s*([a-zA-Z0-9_\.\-]+)$"

    cleaned_query = original_query # Default if no pattern modifies it
    explicit_target_files = []

    # --- Try Matching Patterns in Order ---

    # Try Pattern 1
    match = re.match(pattern_keyword_file_query, original_query)
    if match:
        # This check is slightly wrong, should check groups count *before* accessing
        # Corrected logic: Check which pattern matched based on successful match object
        file_refs_raw = match.group(1).strip()
        # Group index for query is 2 in this pattern
        cleaned_query = match.group(2).strip()
        logger.debug(f"Parser matched Pattern 1: Files='{file_refs_raw}', Query='{cleaned_query}'")
        potential_refs = re.split(r'[\s,]+(?:and\s+)?', file_refs_raw)
        for ref in potential_refs:
            ref_lower = ref.strip().lower()
            if ref_lower in available_files_lower:
                explicit_target_files.append(available_files_lower[ref_lower])
        if explicit_target_files:
             logger.info(f"Explicit targets from Pattern 1: {explicit_target_files}")
             return cleaned_query, explicit_target_files
        else:
             logger.warning(f"Pattern 1 matched but filenames '{file_refs_raw}' not found in available files.")
             cleaned_query = original_query
             explicit_target_files = []

    # Try Pattern 2 (if Pattern 1 failed or found no valid files)
    if not explicit_target_files:
        match = re.match(pattern_file_colon_query, original_query)
        if match:
            file_refs_raw = match.group(1).strip()
             # Group index for query is 2 in this pattern
            cleaned_query = match.group(2).strip()
            logger.debug(f"Parser matched Pattern 2: File='{file_refs_raw}', Query='{cleaned_query}'")
            ref_lower = file_refs_raw.lower()
            if ref_lower in available_files_lower:
                explicit_target_files.append(available_files_lower[ref_lower])
                logger.info(f"Explicit targets from Pattern 2: {explicit_target_files}")
                return cleaned_query, explicit_target_files
            else:
                logger.warning(f"Pattern 2 matched but filename '{file_refs_raw}' not found in available files.")
                cleaned_query = original_query
                explicit_target_files = []

    # Try Pattern 3 (if previous failed or found no valid files)
    if not explicit_target_files:
        match = re.match(pattern_query_keyword_file, original_query)
        if match:
            # Group 1 is the query, Group 2 is the filename
            cleaned_query = match.group(1).strip()
            file_refs_raw = match.group(2).strip()
            logger.debug(f"Parser matched Pattern 3: Query='{cleaned_query}', File='{file_refs_raw}'")
            ref_lower = file_refs_raw.lower()
            if ref_lower in available_files_lower:
                explicit_target_files.append(available_files_lower[ref_lower])
                logger.info(f"Explicit targets from Pattern 3: {explicit_target_files}")
                return cleaned_query, explicit_target_files
            else:
                 logger.warning(f"Pattern 3 matched but filename '{file_refs_raw}' not found in available files.")
                 # Reset query for default logic
                 cleaned_query = original_query
                 explicit_target_files = []

    # --- Default Logic (No explicit files successfully parsed) ---
    # This section runs if none of the patterns above resulted in finding valid, available files.
    logger.debug("No specific file pattern matched or valid filenames not found. Applying default logic based on selection.")

    # Detect general queries that should likely apply to all selected files
    general_query_patterns = [
        r"\b(summarize|summary|overview|recap)\b",
        r"\b(compare|contrast|difference|differences)\b",
        r"\bwhat(?:'s| is| are) (?:in|inside|contained in)\b",
        r"\b(tell me about|describe|explain|list)\b.+\b(all|every|each|both)\b",
        r"^(compare|contrast|summarize)(?:\s+.*)?$", # Starts with these verbs
    ]
    is_general_query = any(re.search(pattern, query_lower) for pattern in general_query_patterns)

    if is_general_query:
        # If it's a general query, target all currently selected files
        logger.info(f"Default logic: General query detected, targeting all selected files: {selected_files}")
        return original_query, selected_files # Return original query text
    elif len(selected_files) == 1:
        # If only one file is selected, assume the query applies to it implicitly
        logger.info(f"Default logic: Single file selected, using it as implicit target: {selected_files[0]}")
        return original_query, selected_files # Target the single selected file
    elif len(selected_files) > 1:
        # If multiple files selected, query isn't general, and no specific file parsed,
        # default to searching ALL selected files. Ambiguity check later will handle clarification.
        logger.info(f"Default logic: Multiple files selected ({len(selected_files)}) but not general/specific. Targeting ALL selected files by default.")
        return original_query, selected_files
    else: # No files selected
        logger.info("Default logic: No files selected. Returning empty target list.")
        return original_query, []


    # Log the input and output queries for debugging
    logger.info(f"Original query: '{query}'")
    logger.info(f"After parsing: cleaned='{cleaned_query}', targets={target_files}")

    # Default return if no specific pattern matched or context applied
    # If selected_files exist and it's not a general query, target_files should be selected_files
    if selected_files and not is_general_query:
        logger.info(f"Defaulting to selected files: {selected_files}")
        return query, selected_files
    elif selected_files and is_general_query:
        logger.info(f"Defaulting to selected files for general query: {selected_files}")
        return query, selected_files
    else:
        # If no files selected or other cases, return query and empty targets
        logger.info("No specific file targets identified, returning original query and empty targets.")
        return query, []


# --- Response cleaning and formatting ---
def clean_response_language(
    response: str,
    remove_hedging: bool = True,
    remove_references: bool = True,
    remove_disclaimers: bool = True,
    ensure_html_structure: bool = True,
    custom_hedging_patterns: Optional[List[str]] = None,
    custom_reference_patterns: Optional[List[str]] = None,
    custom_disclaimer_patterns: Optional[List[str]] = None,
    html_tags_to_fix: Optional[List[str]] = None
) -> str:
    """Cleans the LLM response by removing hedging language, references, and disclaimers."""
    
    cleaned_response = response
    
    # Remove hedging language patterns
    if remove_hedging:
        hedging_patterns = custom_hedging_patterns or [
            r"I think ", r"I believe ", r"It appears that ", r"It seems like ",
            r"possibly ", r"probably ", r"maybe ", r"perhaps ",
            r"I'm not entirely sure, but ", r"As far as I can tell, ",
            r"Based on my understanding, ", r"To the best of my knowledge, "
        ]
        for pattern in hedging_patterns:
            cleaned_response = re.sub(pattern, "", cleaned_response, flags=re.IGNORECASE)
    
    # Remove references to the document
    if remove_references:
        reference_patterns = custom_reference_patterns or [
            r"According to the document, ", r"As mentioned in the document, ",
            r"As stated in the document, ", r"The document mentions that ",
            r"Based on the provided context, ", r"From the information provided, ",
            r"As per the document, ", r"In the document, it says that "
        ]
        for pattern in reference_patterns:
            cleaned_response = re.sub(pattern, "", cleaned_response, flags=re.IGNORECASE)
    
    # Remove disclaimers
    if remove_disclaimers:
        disclaimer_patterns = custom_disclaimer_patterns or [
            r"I'm not an expert[^.]*\.", r"I'm just an AI[^.]*\.",
            r"Please consult a professional[^.]*\.", r"I don't have access to[^.]*\.",
            r"Keep in mind that[^.]*\.", r"Please note that[^.]*\.",
            r"It's important to note that[^.]*\.", r"I should note that[^.]*\."
        ]
        for pattern in disclaimer_patterns:
            cleaned_response = re.sub(pattern, "", cleaned_response, flags=re.IGNORECASE)
    
    # Ensure proper HTML structure if enabled
    if ensure_html_structure:
        tags_to_check = html_tags_to_fix or ["p", "div", "span", "h1", "h2", "h3", "h4", "ul", "ol", "li", "pre", "code"]
        
        # Add HTML wrapper if none exists and response is plain text
        if not re.search(r'<\w+>', cleaned_response):
            cleaned_response = f"<p>{cleaned_response}</p>"
        
        # Fix common unclosed tags
        for tag in tags_to_check:
            # Count opening and closing tags
            open_tags = len(re.findall(f'<{tag}[^>]*>', cleaned_response, re.IGNORECASE))
            close_tags = len(re.findall(f'</{tag}>', cleaned_response, re.IGNORECASE))
            
            # Add missing closing tags
            for _ in range(open_tags - close_tags):
                cleaned_response += f"</{tag}>"
    
    # Strip leading/trailing whitespace
    cleaned_response = cleaned_response.strip()
    
    return cleaned_response

# --- Response generation for no information cases ---
def generate_no_information_response(
    query: str,
    selected_files: List[str],
    title: str = "Information Not Found",
    custom_message: Optional[str] = None,
    include_suggestions: bool = True,
    custom_suggestions: Optional[List[str]] = None,
    html_format: bool = True
) -> str:
    """Generates a structured response when no relevant information is found."""
    
    # Default message if none provided
    if custom_message is None:
        message = f"I couldn't find specific information about that in the selected document(s)."
    else:
        message = custom_message
        
    # Generate suggestions if requested
    suggestions_html = ""
    if include_suggestions:
        suggestion_list = custom_suggestions or [
            f"Try rephrasing your question with more specific terms",
            f"Check if you've selected the right document(s)",
            f"Try a more general question about the document's topic",
            f"Ask about the main topics covered in the document"
        ]
        
        if html_format:
            suggestions_html = "<ul>\n" + "\n".join([f"<li>{s}</li>" for s in suggestion_list]) + "\n</ul>"
        else:
            suggestions_html = "\n- " + "\n- ".join(suggestion_list)
    
    # Format the final response
    if html_format:
        response = (
            f"<div class='no-info-response'>\n"
            f"<h3>{title}</h3>\n"
            f"<p>{message}</p>\n"
            f"{suggestions_html}\n"
            f"</div>"
        )
    else:
        response = (
            f"{title}\n\n"
            f"{message}\n\n"
            f"{suggestions_html}"
        )
        
    return response

# --- Context Retrieval (using the more refined version) ---
async def get_relevant_context_multifile(
    query: str,
    embeddings_by_file: Dict[str, torch.Tensor],
    content_by_file: Dict[str, List[str]],
    target_files: List[str] = None,
    top_k_per_file: int = 7,
    similarity_threshold: float = 0.5,
    force_summary: bool = False
) -> List[Dict]:
    """Finds relevant context from multiple files with improved heading detection."""
    results = []
    device = CONFIG.get("device", torch.device("cpu"))

    # Determine which files to search
    # If target_files is explicitly provided (e.g., from parse_file_query), use it.
    # Otherwise, default to searching all files for which we have embeddings.
    files_to_search = target_files if target_files is not None else list(embeddings_by_file.keys())

    if not files_to_search:
         logger.warning("get_relevant_context called with no files to search.")
         return []

    logger.info(f"Context search targets ({len(files_to_search)} files): {files_to_search}")


    # Use the passed force_summary flag instead of recalculating
    is_summary_query = force_summary

    # Log exactly what was detected for easier debugging
    if is_summary_query:
        logger.info(f"Summary mode ACTIVE for context retrieval. Query: '{query}'")
        logger.info(f"Will provide representative content from {len(files_to_search)} target files.")
    else:
         logger.info(f"Standard context retrieval mode. Query: '{query}'")

    # --- Summary Handling ---
    if is_summary_query and files_to_search:
        logger.info(f"Executing summary context retrieval for files: {files_to_search}")
        for filename in files_to_search:
            if filename not in content_by_file or not content_by_file[filename]:
                logger.warning(f"No content found for {filename} in summary request, skipping.")
                continue

            chunks = content_by_file[filename]
            total_chunks = len(chunks)
            logger.info(f"File {filename} has {total_chunks} total chunks for summary sampling.")

            # SPECIAL HANDLING FOR VERY SMALL FILES
            if total_chunks <= 5: # Increased threshold slightly
                # For tiny files, just use all available chunks with high scores
                logger.info(f"Small file detected: {filename} with only {total_chunks} chunks. Using all chunks.")
                for idx in range(total_chunks):
                    results.append({
                        "content": chunks[idx].strip(),
                        "score": 1.0 - (0.01 * idx),  # Slightly descending scores
                        "filename": filename,
                        "position": f"{idx+1}/{total_chunks}" # Add position info
                    })
                continue # Move to next file

            # Regular sampling logic for larger files continues...

            # For summaries, we want representative chunks: start, middle, end
            sample_indices = []

            # Always take the first 2 chunks (likely title, intro)
            sample_indices.extend([0, 1])

            # Add chunks around 25%, 50%, 75% marks
            if total_chunks > 10: # Only sample middle if reasonably large
                sample_indices.extend([
                    max(2, total_chunks // 4),      # ~25% position (ensure not overlapping with start)
                    max(sample_indices[-1] + 1, total_chunks // 2),      # ~50% position
                ])

            if total_chunks > 20: # Add 75% for longer docs
                sample_indices.append(max(sample_indices[-1] + 1, total_chunks * 3 // 4)) # ~75% position

            # Always add the last chunk
            sample_indices.append(total_chunks - 1)

            # Ensure indices are unique, sorted, and within bounds
            unique_indices = sorted(list(set([idx for idx in sample_indices if 0 <= idx < total_chunks])))

            logger.info(f"Selected {len(unique_indices)} representative indices for summary of {filename}: {unique_indices}")

            # Add selected chunks to results with scores prioritizing start/end
            for i, idx in enumerate(unique_indices):
                # Assign score based on position (higher score = earlier in prompt)
                position_score = 1.0
                if idx == 0: position_score = 1.0
                elif idx == 1: position_score = 0.99
                elif idx == total_chunks - 1: position_score = 0.98
                else: position_score = 0.97 - (i * 0.01) # Middle chunks slightly lower

                # Check for empty content before adding
                content = chunks[idx].strip()
                if not content:
                    logger.debug(f"Skipping empty chunk at position {idx+1}/{total_chunks} for summary of {filename}")
                    continue

                results.append({
                    "content": content,
                    "score": position_score,
                    "filename": filename,
                    "position": f"{idx+1}/{total_chunks}" # Add position info
                })

        if results:
            # Sort by filename first, then by position-based score (descending) for coherence
            results.sort(key=lambda x: (x["filename"], -x["score"]))

            logger.info(f"Returning {len(results)} chunks for summary request, sorted by file and position score.")
            return results # Return early for summary requests

    # --- Standard Query Handling ---
    if not files_to_search: # Should be caught earlier, but double-check
         logger.warning("Standard context retrieval called with no files to search.")
         return []

    # --- Standard Query Handling ---
    # Get query embedding
    try:
        # Run embedding generation in a thread pool to avoid blocking
        logger.debug(f"Generating embedding for query: '{query[:100]}...'")
        response = await asyncio.to_thread(
            lambda: ollama.embeddings(model=CONFIG["ollama_embedding_model"], prompt=query)
        )
        query_embedding = response["embedding"]
        # Move query tensor to the correct device
        query_tensor = torch.tensor(query_embedding, dtype=torch.float32, device=device).unsqueeze(0)
        logger.debug(f"Query embedding generated, shape: {query_tensor.shape}, device: {query_tensor.device}")
    except Exception as e:
        logger.error(f"Error generating query embedding: {str(e)}")
        return [] # Cannot proceed without query embedding

    # Process each targeted file
    all_file_results = []
    for filename in files_to_search:
        if filename not in embeddings_by_file or filename not in content_by_file:
            logger.warning(f"Missing embeddings or content for targeted file {filename}, skipping.")
            continue

        file_content = content_by_file[filename]
        file_embeddings = embeddings_by_file[filename] # Should already be on the correct device

        if not file_content or file_embeddings is None or file_embeddings.shape[0] == 0:
            logger.warning(f"Empty content or embeddings for targeted file {filename}, skipping.")
            continue

        # Ensure embeddings are on the same device as the query tensor
        if file_embeddings.device != device:
             logger.warning(f"Embeddings for {filename} are on {file_embeddings.device}, moving to {device}")
             try:
                 file_embeddings = file_embeddings.to(device)
             except Exception as device_error:
                 logger.error(f"Error moving tensor to device {device}: {str(device_error)}")
                 # Continue with original device if transfer fails
                 logger.warning(f"Continuing with original device: {file_embeddings.device}")


        # Calculate cosine similarity
        try:
            logger.debug(f"Calculating similarity for {filename}. Query shape: {query_tensor.shape}, Embeddings shape: {file_embeddings.shape}")
            cos_scores = torch.cosine_similarity(query_tensor, file_embeddings, dim=1)
            logger.debug(f"Calculated {len(cos_scores)} scores for {filename}.")
        except Exception as e:
            logger.error(f"Error calculating similarity for {filename}: {e}")
            logger.error(f"Query tensor device: {query_tensor.device}, dtype: {query_tensor.dtype}")
            logger.error(f"File embeddings device: {file_embeddings.device}, dtype: {file_embeddings.dtype}")
            continue


        # Get top k results for this file - like V3
        effective_top_k = min(top_k_per_file, len(cos_scores))
        if effective_top_k <= 0:
            logger.debug(f"No valid scores or top_k=0 for {filename}, skipping.")
            continue

        try:
            top_results = torch.topk(cos_scores, k=effective_top_k)
            top_indices = top_results.indices.tolist()
            top_scores = top_results.values.tolist()
            logger.debug(f"Top {effective_top_k} results for {filename}: Scores {top_scores}")
        except Exception as e:
            logger.error(f"Error getting topk results for {filename}: {e}")
            continue


        # Add results from this file meeting threshold
        for idx, score in zip(top_indices, top_scores):
            if idx < len(file_content) and score >= similarity_threshold:
                content = file_content[idx].strip()
                if not content: continue # Skip empty chunks

                # Heading detection and context expansion (minor improvement: check length)
                is_heading_like = (
                    (re.match(r'^\s*[0-9]+\.', content) and len(content) < 100) or # Starts with number. and short
                    (re.match(r'(?i)^(?:' + '|'.join(heading_keywords) + r')\b', content) and len(content) < 100) or # Starts with keyword and short
                    (len(content.split()) < 10 and content.isupper()) # All caps and short
                )

                context_enhanced_content = content
                if is_heading_like:
                    # Add previous chunk if available and not already added
                    if idx > 0:
                         prev_content = file_content[idx - 1].strip()
                         if prev_content:
                             context_enhanced_content = prev_content + "\n\n" + context_enhanced_content

                    # Add next chunk if available and not already added
                    if idx + 1 < len(file_content):
                         next_content = file_content[idx + 1].strip()
                         if next_content:
                              context_enhanced_content = context_enhanced_content + "\n\n" + next_content

                all_file_results.append({
                    "content": context_enhanced_content,
                    "score": float(score),
                    "filename": filename,
                    "original_index": idx # Keep track of original chunk index if needed
                })
            # else: logger.debug(f"Chunk {idx} score {score:.3f} below threshold {similarity_threshold}")


    # Sort all results from all files by score
    all_file_results.sort(key=lambda x: x["score"], reverse=True)

    # Deduplicate results based on content (simple exact match deduplication)
    unique_content_seen = set()
    final_results = []
    for r in all_file_results:
        content_key = r["content"] # Use the potentially expanded content for deduplication
        if content_key not in unique_content_seen:
            unique_content_seen.add(content_key)
            final_results.append(r)
            if len(final_results) >= 15: # Limit total context chunks sent to LLM
                 logger.info(f"Reached max context limit (15 unique chunks).")
                 break
        # else: logger.debug(f"Skipping duplicate content chunk from {r['filename']}")

    logger.info(f"Returning {len(final_results)} unique relevant chunks for standard query '{query[:50]}...'.")
    return final_results


async def ollama_chat_multifile_async(
    user_input: str,
    selected_files: List[str],
    embeddings_by_file: Dict[str, torch.Tensor],
    content_by_file: Dict[str, List[str]],
    client_id: str,
    target_files: Optional[List[str]] = None, 
    force_summary: bool = False,
    # Custom parameters for LLM and RAG
    ollama_model_override: Optional[str] = None,
    ollama_embedding_model_override: Optional[str] = None,
    top_k_per_file_override: Optional[int] = None,
    similarity_threshold_override: Optional[float] = None,
    temperature_override: Optional[float] = None,
    top_p_override: Optional[float] = None,
    system_prompt_override: Optional[str] = None,
    # Response formatting options
    clean_response_override: Optional[bool] = None,
    remove_hedging: Optional[bool] = None,
    remove_references: Optional[bool] = None,
    remove_disclaimers: Optional[bool] = None,
    ensure_html_structure: Optional[bool] = None,
    custom_hedging_patterns: Optional[List[str]] = None,
    custom_reference_patterns: Optional[List[str]] = None,
    custom_disclaimer_patterns: Optional[List[str]] = None,
    html_tags_to_fix: Optional[List[str]] = None,
    # No information response options
    no_info_title: Optional[str] = None,
    no_info_message: Optional[str] = None,
    include_suggestions: Optional[bool] = None,
    custom_suggestions: Optional[List[str]] = None,
    no_info_html_format: Optional[bool] = None
    # history: Optional[List[Dict[str, str]]] = None
) -> str:
    """
    Handles the chat interaction using Ollama with customizable parameters.
    """
    logger.info(f"--- Chat Request ---")
    logger.info(f"User Input: {user_input[:100]}...") # Log first 100 chars of user input
    logger.info(f"Selected Files: {selected_files}")
    logger.info(f"Target Files: {target_files}")
    logger.info(f"Force Summary: {force_summary}")
    logger.info(f"Client ID: {client_id}")

    # --- Parameter Overrides ---
    # Use overrides if provided, else fall back to defaults
    ollama_model = ollama_model_override or CONFIG["ollama_model"]
    embedding_model = ollama_embedding_model_override or CONFIG["ollama_embedding_model"]
    top_k_per_file = top_k_per_file_override or CONFIG.get("top_k_per_file", 5)
    similarity_threshold = similarity_threshold_override or CONFIG.get("similarity_threshold", 0.5)
    temperature = temperature_override or CONFIG.get("temperature", 0.7)
    top_p = top_p_override or CONFIG.get("top_p", 1.0)
    system_prompt = system_prompt_override or CONFIG.get("system_prompt", "")

    logger.info(f"Using Ollama Model: {ollama_model}")
    logger.info(f"Using Embedding Model: {embedding_model}")
    logger.info(f"Top K per File: {top_k_per_file}")
    logger.info(f"Similarity Threshold: {similarity_threshold}")
    logger.info(f"Temperature: {temperature}")
    logger.info(f"Top P: {top_p}")
    logger.info(f"System Prompt: {system_prompt[:50]}...") # Log first 50 chars of system prompt

    # --- Content Filtering: Check Selected Files First ---







    # Prioritize files explicitly selected by the user
    files_with_content = {f: content_by_file[f] for f in selected_files if f in content_by_file}



    logger.info(f"Files with content from selection: {list(files_with_content.keys())}")

    # If no files from selection have content, fall back to all available files
    if not files_with_content:
        logger.warning("No content in selected files, falling back to all available files.")
        files_with_content = content_by_file

    # --- Context Retrieval: Attempt to Find Relevant Context First ---
    logger.info(f"Attempting to retrieve relevant context for query: '{user_input[:50]}...'")
    relevant_context = await get_relevant_context_multifile(
        query=user_input,
        embeddings_by_file=embeddings_by_file,
        content_by_file=files_with_content,
        target_files=target_files,
        top_k_per_file=top_k_per_file,
        similarity_threshold=similarity_threshold,
        force_summary=force_summary
    )

    logger.info(f"Relevant context retrieved: {len(relevant_context)} chunks found.")

    # Check if this is a meta-instruction (system command or special query)
    is_meta_instruction = False
    # Check if the query contains any meta instruction patterns
    meta_instruction_patterns = [
        r'FORCE_SUMMARY_GENERATION',
        r'^\/\w+',  # Commands starting with slash like /help
        r'^![\w]+',  # Commands starting with ! like !clear
        r'(?i)^(help|system|settings|clear|reset)',  # Common system command words
    ]
    is_meta_instruction = any(re.search(pattern, user_input.strip()) for pattern in meta_instruction_patterns)
    
    # Also handle the cleaned_query and query_target_files variables
    cleaned_query = user_input
    query_target_files = target_files if target_files else selected_files
    
    # --- Handle Case: No Context Found (After Search Attempted) ---
    if not relevant_context and not is_meta_instruction:
         logger.info("No relevant context found in targeted files for the query.")
         no_info_response = generate_no_information_response(
             query=cleaned_query, 
             selected_files=query_target_files,
             title=no_info_title or "Information Not Found",
             custom_message=no_info_message,
             include_suggestions=include_suggestions if include_suggestions is not None else True,
             custom_suggestions=custom_suggestions,
             html_format=no_info_html_format if no_info_html_format is not None else True
         )
         logger.info(f"No information response generated: {no_info_response[:100]}...")
         return no_info_response

    # Chat history functions moved to global scope to avoid duplication
    
    # --- Prepare Messages for Ollama API ---
        # For meta-instructions, we might not use the usual user/system message format
    if is_meta_instruction:
            messages = [{"role": "user", "content": user_input}]
    else:
        # Default to standard message format
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]

    logger.info(f"Message format for Ollama API: {[m['role'] + ': ' + m['content'][:50] + '...' for m in messages]}") # Log roles and first 50 chars

    # --- Call Ollama API ---
    try:
        response = await asyncio.to_thread(
            lambda: ollama.chat.completions.create(  # Fixed: Use ollama.chat instead of ollama.Chat
                model=ollama_model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=150, # Limit response length
                n=1, # Number of responses
                stream=False, # Disable streaming for simplicity
                stop=None # No specific stop sequence
            )
        )

        # --- Process Response ---
        raw_response = response.choices[0].message.content.strip()
        logger.info(f"Raw LLM Response: {raw_response[:200]}...")
        
        # Only clean if explicitly requested or defaulting to True
        should_clean = clean_response_override if clean_response_override is not None else True
        if should_clean:
            assistant_response = clean_response_language(
                response=raw_response,
                remove_hedging=remove_hedging if remove_hedging is not None else True,
                remove_references=remove_references if remove_references is not None else True,
                remove_disclaimers=remove_disclaimers if remove_disclaimers is not None else True,
                ensure_html_structure=ensure_html_structure if ensure_html_structure is not None else True,
                custom_hedging_patterns=custom_hedging_patterns,
                custom_reference_patterns=custom_reference_patterns,
                custom_disclaimer_patterns=custom_disclaimer_patterns,
                html_tags_to_fix=html_tags_to_fix
            )
        else:
            assistant_response = raw_response

        logger.info(f"Final Response: {assistant_response[:200]}...")
        return assistant_response

    except Exception as e:
        logger.error(f"Error in Ollama API call: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error processing request with Ollama API: {str(e)}")

# --- HTTP API Endpoints ---

@app.post("/chat", summary="Process Chat Message", response_model=ChatResponse)
async def http_chat(request: ChatRequest):
    """
    Handles a chat message via REST API.
    Allows customization of RAG, LLM parameters, and response formatting.
    """
    try:
        logger.info(f"--- HTTP Chat Request ---")
        logger.info(f"Request Data: {request.model_dump()}")  # Updated from dict() to model_dump()

        # 1. Validate and parse input
        query = request.message.strip()
        if not query:
            raise HTTPException(status_code=400, detail="Query message cannot be empty.")
        
        # Generate or use client ID
        client_id = request.client_id
        if not client_id:
            client_id = f"chat_{hashlib.md5(str(time.time()).encode()).hexdigest()[:10]}"

        # 2. Check file access and read content
        files_with_content = read_vault_content(request.selected_files)
        if not files_with_content:
            raise HTTPException(status_code=404, detail="No valid content found in the selected files.")
        
        # 3. Load existing chat history or create initial chat name
        chat_history = load_chat_history(client_id)
        if not chat_history:
            # Initial chat name based on selected files and their types
            if request.selected_files:
                # Get base filename without timestamp
                initial_filename = os.path.splitext(request.selected_files[0])[0]
                initial_filename = re.sub(r'^\d+_', '', initial_filename)  # Remove timestamp prefix
                initial_filename = re.sub(r'[_\-]', ' ', initial_filename).title()  # Replace underscores
                
                # Get file extensions to add to name
                file_extensions = []
                for file in request.selected_files:
                    ext = os.path.splitext(file)[1].lower().replace('.', '')
                    if ext and ext not in file_extensions:
                        file_extensions.append(ext)
                
                # Create name with file type info
                if file_extensions:
                    file_types_str = f" ({', '.join(file_extensions)})"
                    # Make sure the total length stays reasonable
                    max_name_length = 25 - len(file_types_str)
                    if len(initial_filename) > max_name_length:
                        initial_filename = initial_filename[:max_name_length] + "..."
                    initial_chat_name = initial_filename + file_types_str
                else:
                    # Fallback if no extensions found
                    if len(initial_filename) > 27:
                        initial_filename = initial_filename[:27] + "..."
                    initial_chat_name = initial_filename
            else:
                initial_chat_name = "New Chat"
        else:
            initial_chat_name = chat_history.get("chat_name", "New Chat")
        
        # 4. Save user message to history
        user_message = {
            "role": "user",
            "content": query,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        updated_history = save_chat_history(client_id, initial_chat_name, user_message, request.selected_files)

        # Generate embeddings for the file content
        embeddings_by_file = await generate_vault_embeddings(files_with_content)
        
        # 5. Process request and generate response
        response_text = await ollama_chat_multifile_async(
            user_input=request.message,
            selected_files=list(files_with_content.keys()),
            embeddings_by_file=embeddings_by_file,
            content_by_file=files_with_content,
            client_id=client_id,
            # All the optional parameters
            ollama_model_override=request.ollama_model,
            ollama_embedding_model_override=request.ollama_embedding_model,
            top_k_per_file_override=request.top_k_per_file,
            similarity_threshold_override=request.similarity_threshold,
            temperature_override=request.temperature,
            top_p_override=request.top_p,
            system_prompt_override=request.system_prompt,
            clean_response_override=request.clean_response,
            remove_hedging=request.remove_hedging,
            remove_references=request.remove_references,
            remove_disclaimers=request.remove_disclaimers,
            ensure_html_structure=request.ensure_html_structure,
            custom_hedging_patterns=request.custom_hedging_patterns,
            custom_reference_patterns=request.custom_reference_patterns,
            custom_disclaimer_patterns=request.custom_disclaimer_patterns,
            html_tags_to_fix=request.html_tags_to_fix,
            no_info_title=request.no_info_title,
            no_info_message=request.no_info_message,
            include_suggestions=request.include_suggestions,
            custom_suggestions=request.custom_suggestions,
            no_info_html_format=request.no_info_html_format
        )
        
        # 6. Save response to chat history
        assistant_message = {
            "role": "assistant",
            "content": response_text,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        updated_history = save_chat_history(client_id, initial_chat_name, assistant_message, request.selected_files)
        
        # 7. Check if we should generate an AI-powered name (after 2 messages, 1 user + 1 assistant)
        if updated_history and len(updated_history["messages"]) >= 4 and not updated_history.get("ai_named", False):
            logger.info(f"Generating AI name for chat {client_id} after {len(updated_history['messages'])} messages")
            await generate_chat_name(client_id, updated_history["messages"], OLLAMA_CONFIG)
        
        # 8. Return response with updated chat_id
        return ChatResponse(
            message=response_text,
            context=[],  # Context is not being returned in this version
            client_id=client_id
        )

    except Exception as e:
        logger.error(f"Error generating chat name for {client_id}: {str(e)}")
        logger.error(traceback.format_exc())
        return False

# --- Chat History Functions ---
def load_chat_history(chat_id: str) -> dict:
    """Loads chat history from file based on chat ID."""
    try:
        chat_file = os.path.join(CONFIG["chat_history_directory"], f"{chat_id}.json")
        if not os.path.exists(chat_file):
            return None
                
        with open(chat_file, "r", encoding="utf-8") as f:
            history = json.load(f)
        return history
    except Exception as e:
        logger.error(f"Error loading chat history {chat_id}: {str(e)}")
        return None
    
def save_chat_history(chat_id: str, chat_name: str, message: dict, selected_files: List[str]) -> dict:
    """Saves a message to the chat history."""
    try:
        # Create history directory if it doesn't exist
        os.makedirs(CONFIG["chat_history_directory"], exist_ok=True)
            
        # Determine history file path
        chat_file = os.path.join(CONFIG["chat_history_directory"], f"{chat_id}.json")
            
        # Load existing history or create new
        if os.path.exists(chat_file):
            with open(chat_file, "r", encoding="utf-8") as f:
                history = json.load(f)
        else:
            # Create new history with timestamp
            now = time.strftime("%Y-%m-%d %H:%M:%S")
            history = {
                "chat_id": chat_id,
                "chat_name": chat_name,
                "messages": [],
                "selected_files": selected_files,
                "created_at": now,
                "updated_at": now,
                "ai_named": False
            }
            
        # Add the new message
        if message:
            history["messages"].append(message)
            
        # Update files and timestamp
        history["selected_files"] = selected_files
        history["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            
        # Save the updated history
        with open(chat_file, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
                
        return history
    except Exception as e:
        logger.error(f"Error saving chat history for {chat_id}: {str(e)}")
        logger.error(traceback.format_exc())
        return None

async def generate_chat_name(chat_id: str, messages: List[dict], client: OpenAI) -> bool:
    """Generate an AI-based name for the chat based on conversation history."""
    try:
        if not client or not messages or len(messages) < 2:
            logger.warning("Cannot generate chat name: insufficient data")
            return False
                
        # Extract the conversation for context
        conversation_text = ""
        for i, msg in enumerate(messages[:4]):  # Use up to first 4 messages
            role = msg.get("role", "")
            content = msg.get("content", "")
            if content and role:
                conversation_text += f"{role.upper()}: {content[:100]}...\n"
            
        # Prepare prompt for AI
        prompt = f"""Based on this conversation, generate a short, descriptive title (max 5 words):
    
        {conversation_text}
            
        TITLE: """
    
        # Call API to generate title
        try:
            response = await asyncio.to_thread(
                lambda: client.chat.completions.create(  # Use chat instead of Chat
                    model=CONFIG["ollama_model"],
                    messages=[{"role": "system", "content": prompt}],
                    temperature=0.7,
                    max_tokens=10
                )
            )
            
            # Extract title from response
            generated_title = response.choices[0].message.content.strip()
        except Exception as api_error:
            logger.error(f"Error generating title via API: {str(api_error)}")
            # Provide a fallback title instead of failing
            generated_title = f"Chat {chat_id[-6:]}; {time.strftime('%Y-%m-%d')}"
        logger.info(f"Generated title: {generated_title}")
            
        # Clean up title (remove quotes and other formatting)
        clean_title = re.sub(r'^["\'`]|["\'`]$', '', generated_title)
        clean_title = clean_title.replace("TITLE:", "").strip()
            
        if len(clean_title) > 30:
            clean_title = clean_title[:30] + "..."
                
        # Update the chat history with the new title
        history = load_chat_history(chat_id)
        if history:
            history["chat_name"] = clean_title
            history["ai_named"] = True
                
            # --- HTTP API Endpoints ---
            history["ai_named"] = True
            
            # Save the updated history
            chat_file = os.path.join(CONFIG["chat_history_directory"], f"{chat_id}.json")
            with open(chat_file, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)
                
            logger.info(f"Updated chat {chat_id} with AI-generated name: {clean_title}")
            return True
        return False
    except Exception as e:
        logger.error(f"Error generating chat name for {chat_id}: {str(e)}")
        logger.error(traceback.format_exc())
        # Return False instead of raising an exception to prevent 500 errors
        return False

# Initialize necessary directories and clients at startup
def initialize_application():
    """Performs all necessary initialization at application startup."""
    logger.info("Initializing application...")
    
    # Check dependencies first
    dependency_status = check_dependencies()
    if not all(dependency_status.values()):
        logger.warning("Some dependencies are missing, functionality may be limited.")
        print("⚠️ Some dependencies are missing. See log for details.")
    
    # Initialize directories
    initialize_vault_directory()
    initialize_chat_history_directory()
    
    # Verify Ollama client and model availability
    global ollama_client
    if ollama_client is None:
        logger.critical("Failed to initialize Ollama client. API endpoints requiring LLM will fail.")
        print("❌ Critical: Failed to initialize Ollama client")

    # Log initialization completion
    logger.info("Application initialization complete.")
    return True

# Add route handlers for common URLs that are returning 404s
@app.get("/", summary="API Information")
async def root():
    """Returns basic information about the API."""
    return {
        "name": "Document Chatbot API",
        "version": "1.0",
        "status": "running",
        "endpoints": [
            {"path": "/chat", "method": "POST", "description": "Process chat message"},
            {"path": "/files", "method": "GET", "description": "Get list of available files"},
            {"path": "/upload", "method": "POST", "description": "Upload file to vault"},
        ]
    }

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Handle favicon requests to prevent 404 errors."""
    return JSONResponse(status_code=204, content={})  # No content response

# Add endpoint to get list of files
@app.get("/files", summary="Get Available Files")
async def get_files(tag: str = None):
    """
    Returns the list of available files in the vault.
    
    Parameters:
    - tag: Optional filter to get files with a specific tag
    """
    try:
        # Get all files from the vault
        files = get_vault_files()
        
        if not files:
            return {"files": [], "count": 0, "message": "No files found in vault"}
        
        # Filter by tag if provided
        if tag:
            tag = tag.lower().strip()
            filtered_files = []
            for file in files:
                if "tags" in file and tag in [t.lower() for t in file.get("tags", [])]:
                    filtered_files.append(file)
            files = filtered_files
            
        # Format the response
        result = []
        for file in files:
            # Extract relevant fields and ensure they exist
            file_info = {
                "filename": file.get("filename", ""),
                "description": file.get("description", "No description"),
                "tags": file.get("tags", []),
                "added_date": file.get("added_date", ""),
                "updated_date": file.get("updated_date", ""),
            }
            
            # Add suggested questions if available
            if "suggested_questions" in file and file["suggested_questions"]:
                file_info["suggested_questions"] = file["suggested_questions"]
                
            result.append(file_info)
            
        return {
            "files": result,
            "count": len(result),
            "filtered_by_tag": tag if tag else None
        }
        
    except Exception as e:
        logger.error(f"Error retrieving files: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error retrieving files: {str(e)}")

# Add a function to check for and generate missing questions for all files
async def ensure_files_have_questions():
    """
    Checks all files in the vault and generates suggested questions for those without them.
    This function helps ensure all documents have question suggestions for users.
    """
    if not ollama_client:
        logger.warning("Cannot generate questions - Ollama client not available")
        return {"success": False, "message": "Ollama client not available"}
    
    try:
        # Get all files from the vault
        files = get_vault_files()
        if not files:
            logger.info("No files found in vault to check for questions")
            return {"success": True, "message": "No files found", "processed": 0}
        
        files_to_process = []
        # Identify files without questions or with empty question lists
        for file in files:
            filename = file.get("filename")
            if not filename:
                continue
                
            has_questions = (
                "suggested_questions" in file and 
                isinstance(file["suggested_questions"], list) and 
                len(file["suggested_questions"]) > 0
            )
            
            if not has_questions:
                files_to_process.append(filename)
        
        if not files_to_process:
            logger.info("All files already have suggested questions")
            return {"success": True, "message": "All files have questions", "processed": 0}
            
        logger.info(f"Found {len(files_to_process)} files without suggested questions")
        
        # Process each file that needs questions
        processed_count = 0
        for filename in files_to_process:
            try:
                # Read the file content
                content_by_file = read_vault_content([filename])
                if not content_by_file or filename not in content_by_file:
                    logger.warning(f"Could not read content for {filename}, skipping question generation")
                    continue
                
                # Join chunks to get representative content for question generation
                content = "\n\n".join(content_by_file[filename][:20])  # Use first 20 chunks for context
                if not content.strip():
                    logger.warning(f"File {filename} appears empty, skipping question generation")
                    continue
                
                # Generate questions for the file
                logger.info(f"Generating suggested questions for {filename}")
                questions = await generate_document_questions(content, filename, ollama_client)
                
                if questions:
                    # Update the file metadata with the generated questions
                    updated = update_file_metadata(filename, suggested_questions=questions)
                    if updated:
                        logger.info(f"Generated {len(questions)} questions for {filename}")
                        processed_count += 1
                    else:
                        logger.warning(f"Failed to update metadata with generated questions for {filename}")
                else:
                    logger.warning(f"No questions generated for {filename}")
                
                # Add a small delay between files to avoid overwhelming the LLM
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error processing file {filename}: {str(e)}")
                logger.error(traceback.format_exc())
                # Continue with next file
        
        return {
            "success": True,
            "message": f"Generated questions for {processed_count} files",
            "processed": processed_count,
            "total_checked": len(files_to_process)
        }
                
    except Exception as e:
        logger.error(f"Error in ensure_files_have_questions: {str(e)}")
        logger.error(traceback.format_exc())
        return {"success": False, "message": f"Error: {str(e)}"}

# Add an endpoint to trigger question generation for all files
@app.post("/generate-questions", summary="Generate Suggested Questions")
async def generate_questions_endpoint(regenerate_all: bool = False):
    """
    Checks all files and generates suggested questions for those without any.
    
    Parameters:
    - regenerate_all: If true, regenerates questions for all files, even those that already have questions
    """
    try:
        if regenerate_all:
            # If regenerating all, first clear existing questions
            files = get_vault_files()
            for file in files:
                filename = file.get("filename")
                if filename:
                    update_file_metadata(filename, suggested_questions=[])
            
            logger.info("Cleared existing questions for all files")
            
        # Generate questions
        result = await ensure_files_have_questions()
        return result
    except Exception as e:
        logger.error(f"Error in generate_questions endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error generating questions: {str(e)}")

# Run initialization
initialize_application()

# Add background task to ensure files have questions after startup
@app.on_event("startup")
async def startup_event():
    try:
        # Start question generation in a background task
        asyncio.create_task(ensure_files_have_questions())
        logger.info("Started background task to check and generate questions for files")
    except Exception as e:
        logger.error(f"Error starting question generation task: {str(e)}")

# Start server using uvicorn
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting API server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
