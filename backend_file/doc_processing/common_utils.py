import os
import json
import logging
import traceback
import sys # For check_and_install_dependencies, sys.exit, sys.executable
import re
import time
import markdown
import html
import hashlib # For track_response_quality
from datetime import datetime # For track_response_quality
from typing import List, Dict, Tuple, Optional, Any # Ensure Any is imported if used for type hints

# Third-party imports that these functions might need
import torch # For setup_device
import ollama # For check_ollama_server
import requests # For check_ollama_server, init_ollama_client
from openai import OpenAI # For init_ollama_client

logger = logging.getLogger(__name__) # Will be 'doc_processing.common_utils'

# --- System / Startup Utilities ---

def setup_device(config_dict: Dict) -> Tuple[torch.device, bool]: # Accept config_dict
    """Checks for GPU and sets up PyTorch device."""
    # This function was already fairly self-contained, just ensure it uses passed config if needed.
    # For now, it doesn't directly use config_dict, but good practice to pass it.
    if torch.cuda.is_available():
        device_val = torch.device("cuda")
        gpu_info = f"GPU detected: {torch.cuda.get_device_name(0)}"
        logging.info(f"Using GPU acceleration: {gpu_info}") # Use module's logger or logging.info
        print(f"✅ {gpu_info}")
        torch.set_default_tensor_type('torch.cuda.FloatTensor') # This is a global setting
        return device_val, True
    else:
        logging.info("No GPU detected, using CPU only")
        print("⚠️ No GPU detected, using CPU. Embeddings will be slower.")
        return torch.device("cpu"), False

def check_and_install_dependencies(config_dict: Dict) -> Dict[str, bool]: # Accept config_dict (though not used by current func body)
    """Checks (and offers to install) required dependencies."""
    # This function was already fairly self-contained.
    # It uses logger, sys, os, subprocess which are standard or imported here.
    dependency_status: Dict[str, bool] = {}
    # This list could potentially come from config_dict if you want to make it configurable
    required_packages = [
        {"name": "PySide6", "import_name": "PySide6", "critical": False},
        {"name": "websockets", "import_name": "websockets", "critical": True},
        {"name": "requests", "import_name": "requests", "critical": True}, 
        {"name": "Pillow", "import_name": "PIL", "critical": True}, # PIL is Pillow's import name
        {"name": "fastapi", "import_name": "fastapi", "critical": True},
        {"name": "uvicorn", "import_name": "uvicorn", "critical": True},
        {"name": "torch", "import_name": "torch", "critical": True},
        {"name": "langchain", "import_name": "langchain", "critical": True},
        {"name": "opencv-python", "import_name": "cv2", "critical": True},
        {"name": "pytesseract", "import_name": "pytesseract", "critical": True},
        {"name": "pandas", "import_name": "pandas", "critical": True},
        {"name": "chromadb", "import_name": "chromadb", "critical": True},
        {"name": "PyMuPDF", "import_name": "fitz", "critical": True},
        {"name": "openai", "import_name": "openai", "critical": True},
        {"name": "pypdf", "import_name": "pypdf", "critical": False}
    ]
    logger.info("--- common_utils: Checking Dependencies ---")
    all_ok = True
    for package in required_packages:
        package_display_name = package["name"]
        import_name = package["import_name"]
        try:
            __import__(import_name)
            dependency_status[import_name] = True
            logger.info(f"{package_display_name} ({import_name}) available: ✓")
        except ImportError:
            dependency_status[import_name] = False
            logger.warning(f"{package_display_name} ({import_name}) not found.")
            print(f"⚠️ Missing dependency: {package_display_name} (needed for import '{import_name}')")
            if package["critical"]:
                all_ok = False

    if not all_ok:
        missing_critical = [p["name"] for p in required_packages if p["critical"] and not dependency_status.get(p["import_name"])]
        print(f"\n❌ Critical dependencies missing: {', '.join(missing_critical)}")
        # Automated install prompt (be cautious with this in shared utils if used programmatically elsewhere)
        # For a server startup script, it's reasonable.
        user_input = input("Would you like to attempt to install missing critical dependencies now? (y/n): ")
        if user_input.lower() == 'y':
            import subprocess # Keep import local to where it's used
            for package in required_packages:
                if package["critical"] and not dependency_status.get(package["import_name"]):
                    print(f"Installing {package['name']}...")
                    try:
                        subprocess.check_call([sys.executable, "-m", "pip", "install", package["name"]])
                        logger.info(f"{package['name']} installed successfully.")
                        print(f"✅ {package['name']} installed successfully")
                        dependency_status[package["import_name"]] = True
                    except subprocess.CalledProcessError:
                        logger.error(f"Failed to install {package['name']}.")
                        print(f"❌ Failed to install {package['name']}")
            print("\nDependency installation attempt completed. Please restart the application if prompted or if issues persist.")
            all_ok_after_install = all(dependency_status.get(p["import_name"]) for p in required_packages if p["critical"])
            if not all_ok_after_install:
                 print("❌ Some critical dependencies could not be installed. Application may not function correctly. Please install them manually and restart.")
                 sys.exit(1) # Critical failure
            else:
                 print("✅ All critical dependencies seem to be installed. Restarting application...")
                 os.execl(sys.executable, sys.executable, *sys.argv) # Force restart
        else:
            print("Cannot continue without critical dependencies. Exiting.")
            sys.exit(1) # Critical failure
    else:
        logger.info("All critical dependencies seem to be available.")
        print("✅ All critical dependencies available.")
    logger.info("--- common_utils: Dependency Check Complete ---")
    return dependency_status


import os
import json
import logging
import requests # Ensure requests is imported
import ollama # Keep for the initial attempt
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)

def check_ollama_server(config_dict: Dict) -> Tuple[bool, str, List[str]]:
    ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    logger.info(f"common_utils: Checking Ollama server at {ollama_host}")
    
    available_models_full: List[str] = []
    connection_successful = False
    api_message = ""

    # --- Attempt 1: Using ollama.Client().list() ---
    try:
        logger.debug("Attempting to list models via ollama.Client().list()")
        client = ollama.Client(host=ollama_host)
        models_response = client.list()
        
        if models_response and 'models' in models_response:
            connection_successful = True # At least connected to the client endpoint
            available_models_full = [model.get('name', '') for model in models_response['models'] if model.get('name')]
            if available_models_full:
                logger.info(f"ollama.Client().list() returned: {available_models_full}")
            else:
                logger.warning("ollama.Client().list() returned an empty list of models. Will try /api/tags fallback.")
        else:
            connection_successful = True # Connected but unexpected format
            logger.warning("ollama.Client().list() returned unexpected format. Will try /api/tags fallback.")
            api_message = "Connected (unexpected model list format from client.list())."

    except requests.exceptions.ConnectionError as conn_err:
         error_msg = f"Connection Error (ollama.Client): Could not connect to Ollama at {ollama_host}. Details: {conn_err}"
         logger.error(error_msg)
         return False, error_msg, [] # Hard fail on connection error
    except Exception as e_client_list:
        logger.warning(f"Error using ollama.Client().list(): {e_client_list}. Will try /api/tags fallback.")
        connection_successful = True # Assume connected if error is not ConnectionError
        api_message = f"Error with client.list(): {e_client_list}."

    # --- Attempt 2: Fallback to direct /api/tags HTTP GET request if Attempt 1 yielded no models ---
    if not available_models_full and connection_successful: # Only try fallback if client.list() didn't populate models but connection was okay
        logger.info(f"Fallback: Attempting to list models via direct HTTP GET to {ollama_host}/api/tags")
        try:
            response = requests.get(f"{ollama_host}/api/tags", timeout=10) # Added timeout
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            models_data_tags = response.json()
            
            if models_data_tags and 'models' in models_data_tags:
                available_models_full = [model.get('name', '') for model in models_data_tags['models'] if model.get('name')]
                if available_models_full:
                    logger.info(f"Fallback /api/tags successful. Models: {available_models_full}")
                    api_message = "Connected (models fetched via /api/tags)."
                else:
                    logger.warning("Fallback /api/tags also returned an empty list of models.")
                    api_message = "Connected but no models found via /api/tags either."
            else:
                logger.warning("Fallback /api/tags returned unexpected JSON format.")
                api_message = "Connected (unexpected format from /api/tags)."
        except requests.exceptions.RequestException as e_req:
            error_msg = f"HTTP Request Error (fallback /api/tags): Could not connect or query Ollama at {ollama_host}/api/tags. Details: {e_req}"
            logger.error(error_msg)
            # If client.list() also failed badly, this might be a full connection issue
            return False, error_msg, [] # Return False if fallback also fails to connect/query
        except json.JSONDecodeError as e_json:
            error_msg = f"JSON Decode Error (fallback /api/tags): Could not parse response from {ollama_host}/api/tags. Details: {e_json}"
            logger.error(error_msg)
            api_message = "Connected (JSON error from /api/tags)." # Still connected, but can't list
        except Exception as e_fallback:
            error_msg = f"Unexpected Error (fallback /api/tags): {e_fallback}"
            logger.error(error_msg, exc_info=True)
            api_message = f"Connected (Error with /api/tags: {e_fallback})."

    # Now, evaluate based on available_models_full, regardless of how it was populated
    if not connection_successful and not available_models_full: # If initial client.list() had hard connection error and fallback didn't run or also failed
        return False, api_message or "Failed to connect or list models from Ollama.", []

    logger.info(f"DEBUG common_utils.check_ollama_server: Final 'available_models_full' before base name check: {available_models_full}")
    
    if not available_models_full: # If still no models after both attempts
        warn_msg = api_message or "Connected but no models available on Ollama server after all attempts."
        logger.warning(warn_msg)
        return True, warn_msg, []

    available_models_base = {model_name.split(':')[0] for model_name in available_models_full}
    logger.info(f"DEBUG common_utils.check_ollama_server: Derived 'available_models_base': {available_models_base}")
    
    missing_models_base = []
    required_model_chat_base = config_dict["ollama_model"].split(':')[0]
    required_model_embed_base = config_dict["ollama_embedding_model"].split(':')[0]
    
    if required_model_chat_base not in available_models_base:
        missing_models_base.append(required_model_chat_base)
    if required_model_embed_base not in available_models_base:
        missing_models_base.append(required_model_embed_base)
    
    if missing_models_base:
        pull_cmd = f"ollama pull {' '.join(missing_models_base)}"
        final_warn_msg = (
            f"Configured models (base names) not found: {', '.join(missing_models_base)}\n"
            f"Available models on server (full names): {', '.join(available_models_full)}\n"
            f"Suggestion: Run `{pull_cmd}` on the server where Ollama is running."
        )
        logger.warning(final_warn_msg)
        return True, final_warn_msg, available_models_full
    else:
        success_msg = "Connected with all required models (base names verified)."
        logger.info(success_msg)
        return True, success_msg, available_models_full


def init_ollama_client(config_dict: Dict, dependency_status_dict: Dict) -> Optional[OpenAI]:
    # Uses config_dict for model names, dependency_status_dict for 'openai' lib check
    if not dependency_status_dict.get("openai", False):
        logger.critical("Missing 'openai' library, required for Ollama client compatibility layer.")
        print("❌ ERROR: Cannot initialize Ollama client - missing 'openai' library. Install with: pip install openai")
        return None

    # Call check_ollama_server (which now takes config_dict)
    server_available, message, server_models_full_names = check_ollama_server(config_dict)
    if not server_available:
        logger.critical(f"Ollama server not available: {message}")
        print(f"❌ ERROR: Ollama server not available - {message}")
        print("ℹ️ Make sure Ollama is running (e.g., 'ollama serve') and accessible.")
        return None
    
    try:
        # The base_url should point to the OpenAI-compatible endpoint of Ollama
        ollama_host_for_openai_client = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        client = OpenAI(
            base_url=f'{ollama_host_for_openai_client}/v1', # Standard v1 endpoint
            api_key='ollama' # Required by OpenAI client, Ollama ignores it
        )

        try:
            client.models.list() # Simple test call, may or may not work depending on Ollama's strictness
            logger.info("OpenAI-compatible client successfully listed models (or call didn't error).")
        except Exception as list_models_e:
            logger.warning(f"OpenAI-compatible client test call (models.list) resulted in: {list_models_e}. Proceeding, but check Ollama compatibility if issues arise.")

        # Model availability already checked by check_ollama_server
        if server_models_full_names is not None: # Check if model list was successfully retrieved
            required_chat_model = config_dict["ollama_model"]
            required_embed_model = config_dict["ollama_embedding_model"]
            
            missing_full_names = []
            if required_chat_model not in server_models_full_names:
                missing_full_names.append(required_chat_model)
            if required_embed_model not in server_models_full_names:
                missing_full_names.append(required_embed_model)

            if missing_full_names:
                logger.warning(f"Configured models (full names) not found on server: {', '.join(missing_full_names)}. Available: {', '.join(server_models_full_names)}")
                print(f"⚠️ WARNING: Models {', '.join(missing_full_names)} not found on Ollama. Pull them with 'ollama pull ...'")
                # Decide if this is critical enough to prevent client init. For now, allow init but warn.
            else:
                logger.info(f"All required Ollama models ({required_chat_model}, {required_embed_model}) are available on the server.")
                print(f"✅ Required Ollama models available: {required_chat_model}, {required_embed_model}")
        
        logger.info("Ollama API client (OpenAI-compatible) initialized successfully.")
        return client
        
    except Exception as e:
        logger.error(f"Failed to initialize Ollama API client (OpenAI-compatible): {e}", exc_info=True)
        print(f"❌ ERROR: Failed to initialize Ollama client - {str(e)}")
        return None

def initialize_vault_directory(config_dict: Dict):

    os.makedirs(config_dict["vault_directory"], exist_ok=True)
    subdirs_to_create = [
        config_dict["processed_docs_subdir"], # For visual pipeline outputs
        config_dict.get("review_backup_subdir", "review_backups") # Optional
    ]
    if config_dict.get("create_plain_text_audit", False): # For visual pipeline text chunks
        subdirs_to_create.append(config_dict.get("text_chunk_output_dir", "extracted_text_chunks"))
    
    for subdir_key in subdirs_to_create:
        # subdir_key might be the actual name or a key in config_dict
        # Assuming it's the actual name for now as per your CONFIG structure
        subdir_path_to_create = os.path.join(config_dict["vault_directory"], subdir_key)
        os.makedirs(subdir_path_to_create, exist_ok=True)
        logger.debug(f"Ensured subdirectory exists: {subdir_path_to_create}")
    
    metadata_file_path = os.path.join(config_dict["vault_directory"], config_dict["vault_metadata"])
    if not os.path.exists(metadata_file_path):
        try:
            with open(metadata_file_path, "w", encoding="utf-8") as meta_f:
                json.dump({"files": []}, meta_f, indent=2)
            logger.info(f"Created empty vault metadata file: {metadata_file_path}")
        except IOError as e_io_meta:
            logger.error(f"Failed to create vault metadata file {metadata_file_path}: {e_io_meta}")


def get_vault_files(config_dict: Dict) -> List[Dict]:
   
    try:
        metadata_path = os.path.join(config_dict["vault_directory"], config_dict["vault_metadata"])
        if not os.path.exists(metadata_path):
            logger.warning(f"Metadata file not found: {metadata_path}. Returning empty list.")
            return []
        with open(metadata_path, "r", encoding="utf-8") as f: metadata_content = json.load(f)

        if not isinstance(metadata_content, dict) or "files" not in metadata_content:
            logger.warning(f"Invalid metadata format in {metadata_path}. Resetting structure and returning empty.")
            # Optionally attempt to fix it here if desired
            return []
        
        files_with_meta = []
        valid_entries_for_rewrite = []
        metadata_needs_update = False

        for file_entry_data in metadata_content.get("files", []):
            if not isinstance(file_entry_data, dict) or "filename" not in file_entry_data:
                logger.warning(f"Skipping invalid entry in metadata: {file_entry_data}")
                metadata_needs_update = True; continue
            
            doc_id_from_meta = file_entry_data.get("filename")
            if not doc_id_from_meta:
                 logger.warning(f"Skipping metadata entry with empty filename/doc_id.")
                 metadata_needs_update = True; continue
            
            pipeline_type_from_meta = file_entry_data.get("pipeline_type", "visual") # Default to visual if missing

            path_exists = False
            if pipeline_type_from_meta == "visual":
                # Path to the directory for this processed visual document
                item_path_on_disk = os.path.join(config_dict["vault_directory"], config_dict["processed_docs_subdir"], doc_id_from_meta)
                path_exists = os.path.isdir(item_path_on_disk)
            elif pipeline_type_from_meta == "textual":
                # Path to the processed textual file (e.g., doc_id.txt)
                # The 'processed_path' in metadata might store this name, or construct it
                textual_file_name = file_entry_data.get("processed_path", f"{doc_id_from_meta}.txt")
                item_path_on_disk = os.path.join(config_dict["vault_directory"], textual_file_name)
                path_exists = os.path.exists(item_path_on_disk)
            else: # Unknown pipeline type, try checking both as a fallback
                visual_path = os.path.join(config_dict["vault_directory"], config_dict["processed_docs_subdir"], doc_id_from_meta)
                textual_path = os.path.join(config_dict["vault_directory"], f"{doc_id_from_meta}.txt") # Assuming .txt for legacy
                path_exists = os.path.isdir(visual_path) or os.path.exists(textual_path)
                item_path_on_disk = visual_path if os.path.isdir(visual_path) else textual_path


            if path_exists:
            
                # Populate default fields for display consistency
                file_entry_data.setdefault("description", f"Doc ID: {doc_id_from_meta}")
                file_entry_data.setdefault("tags", [])
                file_entry_data.setdefault("added_date", datetime.fromtimestamp(os.path.getctime(item_path_on_disk) if os.path.exists(item_path_on_disk) else time.time()).strftime("%Y-%m-%d %H:%M:%S"))
                file_entry_data.setdefault("updated_date", datetime.fromtimestamp(os.path.getmtime(item_path_on_disk) if os.path.exists(item_path_on_disk) else time.time()).strftime("%Y-%m-%d %H:%M:%S"))
                file_entry_data.setdefault("original_filename", doc_id_from_meta) # Fallback
                file_entry_data['display_name'] = file_entry_data.get('original_filename', doc_id_from_meta)
                
                files_with_meta.append(file_entry_data)
                valid_entries_for_rewrite.append(file_entry_data)
            else:
                logger.warning(f"Item for metadata entry '{doc_id_from_meta}' (type: {pipeline_type_from_meta}) not found at expected path: {item_path_on_disk}. Removing entry.")
                metadata_needs_update = True
        
        if metadata_needs_update:
            # ... (Logic to rewrite metadata file with only valid_entries_for_rewrite) ...
            logger.info("Updating metadata file due to missing items or invalid entries.")
            try:
                with open(metadata_path, "w", encoding="utf-8") as f_update_meta:
                    json.dump({"files": valid_entries_for_rewrite}, f_update_meta, indent=2)
                logger.info("Metadata file updated successfully.")
            except IOError as e_rewrite_meta:
                logger.error(f"Failed to write updated metadata file {metadata_path}: {e_rewrite_meta}")

        files_with_meta.sort(key=lambda x_sort: x_sort.get("added_date", "1970-01-01 00:00:00"), reverse=True)
        return files_with_meta
        
    except json.JSONDecodeError: # ... (handle corrupted metadata as before) ...
        logger.error(f"Error parsing metadata file: {metadata_path}. Attempting to reset.", exc_info=True)
        try:
             with open(metadata_path, "w", encoding="utf-8") as f_reset_meta: json.dump({"files": []}, f_reset_meta, indent=2)
             logger.info("Metadata file was corrupted and has been reset.")
        except IOError as e_io_reset: logger.error(f"Failed to reset corrupted metadata file {metadata_path}: {e_io_reset}", exc_info=True)
        return []
    except Exception as e_get_files:
        logger.error(f"Error reading vault files: {str(e_get_files)}", exc_info=True)
        return []


def add_file_to_vault(config_dict: Dict, filename_is_doc_id: str, description: str, tags: List[str] = None, metadata_extra: Dict = None) -> bool:

    if metadata_extra is None: metadata_extra = {}
    doc_id_to_save = filename_is_doc_id # filename_is_doc_id is the unique doc_id
    try:
        metadata_path = os.path.join(config_dict["vault_directory"], config_dict["vault_metadata"])
    
        if not os.path.exists(metadata_path): # Initialize if doesn't exist
            with open(metadata_path, "w", encoding="utf-8") as f_create: json.dump({"files": []}, f_create, indent=2)
        
        current_metadata = {"files": []} # Default
        try:
            with open(metadata_path, "r", encoding="utf-8") as f_read: current_metadata = json.load(f_read)
            if not isinstance(current_metadata, dict) or "files" not in current_metadata:
                current_metadata = {"files": []} # Reset if invalid format
        except json.JSONDecodeError: 
            logger.warning(f"Metadata file {metadata_path} corrupted, re-initializing.")
            current_metadata = {"files": []}

        processed_tags_list = sorted(list(set([t.lower().strip() for t in tags if t and isinstance(t, str) and t.strip()]))) if tags else []
        
        found_idx = -1
        for idx, entry in enumerate(current_metadata.get("files", [])):
            if isinstance(entry, dict) and entry.get("filename") == doc_id_to_save:
                found_idx = idx; break
        
        now_timestamp_str = datetime.now().isoformat() # Use ISO format for better sorting later

        # Base entry fields from direct parameters
        new_file_entry_data = {
            "filename": doc_id_to_save,
            "description": description or f"Document ID: {doc_id_to_save}",
            "tags": processed_tags_list,
            "updated_date": now_timestamp_str
        }
        # Merge in any additional metadata provided
        if metadata_extra: new_file_entry_data.update(metadata_extra)


        if found_idx != -1: # Update existing entry
            existing_entry_data = current_metadata["files"][found_idx]
            # Preserve original added_date
            new_file_entry_data["added_date"] = existing_entry_data.get("added_date", now_timestamp_str) 
            # Merge tags
            combined_tags = sorted(list(set(existing_entry_data.get("tags", []) + processed_tags_list)))
            if metadata_extra and "tags" in metadata_extra: # If metadata_extra also has tags, merge them too
                 combined_tags = sorted(list(set(combined_tags + metadata_extra["tags"])))
            new_file_entry_data["tags"] = combined_tags
            
            current_metadata["files"][found_idx].update(new_file_entry_data) # Update existing dict
            logger.info(f"Updated metadata for existing document ID: {doc_id_to_save}")
        else: # Add new entry
            new_file_entry_data["added_date"] = now_timestamp_str # Set added_date for new entries
            current_metadata.setdefault("files", []).append(new_file_entry_data)
            logger.info(f"Added new document ID to metadata: {doc_id_to_save}")
        
        with open(metadata_path, "w", encoding="utf-8") as f_write_meta: 
            json.dump(current_metadata, f_write_meta, indent=2)
        return True

    except Exception as e_add_vault:
        logger.error(f"Error adding/updating metadata for doc ID '{doc_id_to_save}': {e_add_vault}", exc_info=True)
        return False


def update_file_metadata(config_dict: Dict, filename_is_doc_id: str, description: Optional[str] = None, tags: Optional[List[str]] = None, metadata_extra: Optional[Dict] = None) -> bool:
    """
    Updates a document entry in the vault_metadata.json file by filename (doc_id).
    Applies description, tags, and arbitrary extra metadata fields.
    Returns True if an entry was found and updated, False otherwise.
    """
    doc_id_to_update = filename_is_doc_id
    logger.debug(f"Attempting to update metadata for doc ID: {doc_id_to_update} with extra: {metadata_extra}") # Added debug log
    try:
        metadata_path = os.path.join(config_dict["vault_directory"], config_dict["vault_metadata"])
        if not os.path.exists(metadata_path):
            logger.error(f"Metadata file {metadata_path} not found for update of {doc_id_to_update}.")
            return False

        # Read existing metadata
        with open(metadata_path, "r", encoding="utf-8") as f_read_meta:
            metadata_content = json.load(f_read_meta)

        if not isinstance(metadata_content, dict) or "files" not in metadata_content or not isinstance(metadata_content["files"], list):
            logger.error(f"Invalid metadata format in {metadata_path} for update of {doc_id_to_update}.")
            return False

        entry_found_and_updated = False
        # Iterate through entries to find the one matching the doc_id
        for i_entry, file_data_entry in enumerate(metadata_content["files"]): # Iterate directly over list

            if isinstance(file_data_entry, dict) and file_data_entry.get("filename") == doc_id_to_update:
                # Found the entry, apply updates
                if description is not None:
                    file_data_entry["description"] = description
                if tags is not None:
                    # Process new tags and merge with existing ones
                    processed_new_tags_list = sorted(list(set([t.lower().strip() for t in tags if t and isinstance(t, str) and t.strip()])))
                    current_tags_list = file_data_entry.get("tags", []) # Ensure default empty list if no tags key
                    # Ensure current_tags_list is actually a list before combining
                    if not isinstance(current_tags_list, list):
                         logger.warning(f"Tags for doc {doc_id_to_update} in metadata are not a list. Resetting.")
                         current_tags_list = []

                    file_data_entry["tags"] = sorted(list(set(current_tags_list + processed_new_tags_list)))

                # --- CRITICAL FIX: Check if metadata_extra is a dict before updating ---
                if metadata_extra is not None and isinstance(metadata_extra, dict):
                    # Update the dictionary entry with key-value pairs from metadata_extra
                    file_data_entry.update(metadata_extra) # <--- This is the line that failed previously
                elif metadata_extra is not None: # Log if metadata_extra was provided but wasn't a dictionary
                    logger.warning(f"metadata_extra provided for doc {doc_id_to_update} was not a dict (type: {type(metadata_extra)}). Skipping update with extra metadata.")
                # --- END CRITICAL FIX ---

                # Update the 'updated_date' for this entry
                file_data_entry["updated_date"] = datetime.now().isoformat()

                # The list element is updated by modifying file_data_entry directly
                entry_found_and_updated = True
                logger.info(f"Updated metadata fields for doc ID {doc_id_to_update}")
                break # Exit loop once the entry is found and updated

        if not entry_found_and_updated:
            logger.warning(f"Attempted to update metadata for doc ID '{doc_id_to_update}', but it was not found in the file. No entry updated.")
            return False # Entry was not found to be updated

        with open(metadata_path, "r+", encoding="utf-8") as f_write_updated_meta:
             f_write_updated_meta.seek(0)
             json.dump(metadata_content, f_write_updated_meta, indent=2)
             f_write_updated_meta.truncate() # Cut off remaining old content if new content is smaller

        logger.info(f"Metadata file saved successfully after updating {doc_id_to_update}.")
        return True

    except json.JSONDecodeError:
        logger.error(f"Error parsing metadata file during update for {doc_id_to_update}: {metadata_path}", exc_info=True)
        return False
    except Exception as e_update_meta:
        logger.error(f"Error updating metadata for doc ID {doc_id_to_update}: {e_update_meta}", exc_info=True)
        return False


def get_specific_doc_metadata(config_dict: Dict, doc_id_to_find: str) -> Optional[Dict]:
    # ... (Your existing _get_doc_metadata_from_file renamed, using config_dict and doc_id_to_find) ...
    metadata_path = os.path.join(config_dict["vault_directory"], config_dict["vault_metadata"])
    try:
        if not os.path.exists(metadata_path):
            logger.warning(f"Metadata file {metadata_path} not found when getting specific doc meta for {doc_id_to_find}.")
            return None
        with open(metadata_path, "r", encoding="utf-8") as f_read_meta_spec:
            metadata_store_content = json.load(f_read_meta_spec)
        
        for file_entry_item in metadata_store_content.get("files", []):
            if isinstance(file_entry_item, dict) and file_entry_item.get("filename") == doc_id_to_find:
                # Ensure all expected default fields for safety before returning
                file_entry_item.setdefault("description", f"Document ID: {doc_id_to_find}")
                file_entry_item.setdefault("tags", [])
                file_entry_item.setdefault("added_date", "Unknown")
                file_entry_item.setdefault("updated_date", "Unknown")
                file_entry_item.setdefault("original_filename", doc_id_to_find)
                file_entry_item['display_name'] = file_entry_item.get('original_filename', doc_id_to_find)
                return file_entry_item
        
        logger.warning(f"Metadata for doc_id '{doc_id_to_find}' not found in {metadata_path}.")
        return None
        
    except json.JSONDecodeError:
        logger.error(f"Metadata file {metadata_path} is corrupted (get_specific_doc_metadata).", exc_info=True)
        return None
    except Exception as e_get_spec_meta:
        logger.error(f"Error reading metadata for {doc_id_to_find}: {e_get_spec_meta}", exc_info=True)
        return None


# --- Text Utilities (from text_utils.py proposal) ---

def parse_file_query(config_dict: Dict, query: str, available_files_meta: List[Dict], selected_doc_ids: List[str] = None) -> Tuple[str, List[str]]:
    selected_doc_ids_list = selected_doc_ids if selected_doc_ids is not None else [] # Ensure list
    
    if not available_files_meta:
        logger.debug("common_utils.parse_file_query: No available_files_meta provided.")
        return query, []

    original_query_str = query.strip()
    # Create maps for efficient lookup
    display_name_to_id_map_lower = {meta.get("display_name", meta.get("filename")).lower(): meta.get("filename") 
                                    for meta in available_files_meta if meta.get("filename")}
    doc_id_to_id_map_lower = {meta.get("filename").lower(): meta.get("filename") 
                              for meta in available_files_meta if meta.get("filename")}

    file_ref_regex_pattern = r"[a-zA-Z0-9_\.\-\(\)]+" # Added parentheses for filenames like "file (1).pdf"
  
    query_patterns_for_files = [
        # e.g., "in file X and file Y about Z", "search doc A for B"
        rf"(?i)(?:in|from|using|search|check|within|compare|contrast|difference between)\s+(?:file|document|doc)[s]?\s+((?:{file_ref_regex_pattern}(?:(?:[,]?\s+|\s+and\s+)\s*{file_ref_regex_pattern})*))\s*(?:[:\-,\s]|on|about|regarding|with\srespect\sto)\s*(.+)",
        # e.g., "file X : query Y", "doc A - find B"
        rf"(?i)(?:file|document|doc)[s]?\s+({file_ref_regex_pattern})\s*[:\-,\s]\s*(.+)",
        # e.g., "what is X in file Y", "find A from doc B"
        rf"(.+?)\s+(?:in|from|about)\s+(?:the\s+)?(?:file|document|doc)?\s*({file_ref_regex_pattern})\s*\??$" # Optional question mark at end
    ]

    def _resolve_reference_to_id(ref_str: str) -> Optional[str]:
        ref_lower_str = ref_str.strip().lower()
        return display_name_to_id_map_lower.get(ref_lower_str) or doc_id_to_id_map_lower.get(ref_lower_str)

    for i_pattern, pat_str in enumerate(query_patterns_for_files):
        match_obj = re.match(pat_str, original_query_str)
        if match_obj:
            groups_matched = match_obj.groups()
            # Determine which group is file_refs and which is the actual query based on pattern structure
            raw_file_refs_group_idx = 0 if i_pattern in [0, 1] else 1 
            actual_query_group_idx = 1 if i_pattern in [0, 1] else 0

            raw_file_refs_str = groups_matched[raw_file_refs_group_idx].strip()
            cleaned_query_str_candidate = groups_matched[actual_query_group_idx].strip()
            
            # Split multiple file references (e.g., "file A, file B and fileC")
            individual_potential_refs = re.split(r'[\s,]+(?:and\s+)?', raw_file_refs_str)
            resolved_target_doc_ids = [resolved_id 
                                       for ref_token_str in individual_potential_refs 
                                       if (resolved_id := _resolve_reference_to_id(ref_token_str))]

            if resolved_target_doc_ids: # If any refs were successfully resolved to actual doc_ids
                logger.info(f"Query parser (Pattern {i_pattern+1}): Explicit file refs='{raw_file_refs_str}', Cleaned query='{cleaned_query_str_candidate}'. Resolved to Doc IDs: {resolved_target_doc_ids}")
                return cleaned_query_str_candidate, resolved_target_doc_ids
            else: # Pattern matched, but the referenced "filenames" weren't found in available_files_meta
                logger.warning(f"Query parser (Pattern {i_pattern+1}) matched refs '{raw_file_refs_str}', but these were not found in available documents.")
                # Don't return yet, let default logic handle it or try other patterns.
    
    # --- Default Logic if no explicit files are parsed or resolved ---
    logger.debug("Query parser: No explicit file references parsed or resolved. Applying default target logic.")
    # General queries that often apply to all selected context
    general_query_keywords = [r"\b(summarize|summary|overview|recap|compare|contrast|difference)\b"]
    is_general_type_query = any(re.search(kw_pat, original_query_str, re.IGNORECASE) for kw_pat in general_query_keywords)

    if is_general_type_query and selected_doc_ids_list:
        logger.info(f"Query parser (Default): General query type, using all {len(selected_doc_ids_list)} currently selected doc IDs.")
        return original_query_str, selected_doc_ids_list # Use original query, target all selected
    
    if len(selected_doc_ids_list) == 1:
        logger.info(f"Query parser (Default): Single document selected ('{selected_doc_ids_list[0]}'), implicitly targeting it.")
        return original_query_str, selected_doc_ids_list
        
    if selected_doc_ids_list: # Multiple selected, but not clearly general/specific to one
        logger.info(f"Query parser (Default): Multiple ({len(selected_doc_ids_list)}) documents selected, but query isn't specific. Targeting all selected by default.")
        return original_query_str, selected_doc_ids_list
    
    logger.info("Query parser (Default): No documents selected by user, and no specific files parsed from query. Returning original query with no target files.")
    return original_query_str, []

def remove_doc_from_metadata(config_dict: Dict, doc_id_to_remove: str) -> bool:
    """
    Removes a document entry from the vault_metadata.json file.
    Returns True if an entry was found and removed, False otherwise.
    """
    metadata_path = os.path.join(config_dict["vault_directory"], config_dict["vault_metadata"])
    if not os.path.exists(metadata_path):
        logger.warning(f"Metadata file not found at {metadata_path}. Cannot remove entry for {doc_id_to_remove}.")
        return False # Or True if we consider "not there" as "removed" for idempotency

    try:
        with open(metadata_path, "r+", encoding="utf-8") as f_meta_rw:
            current_metadata = json.load(f_meta_rw)
            if not isinstance(current_metadata, dict) or "files" not in current_metadata:
                logger.error(f"Invalid metadata format in {metadata_path}. Cannot remove entry.")
                return False # Invalid format, can't proceed

            original_file_count = len(current_metadata.get("files", []))
            
            # Filter out the document to be removed
            updated_files_list = [
                entry for entry in current_metadata.get("files", [])
                if not (isinstance(entry, dict) and entry.get("filename") == doc_id_to_remove)
            ]
            
            if len(updated_files_list) < original_file_count:
                current_metadata["files"] = updated_files_list
                f_meta_rw.seek(0)       # Go to the beginning of the file
                f_meta_rw.truncate()    # Clear the file content
                json.dump(current_metadata, f_meta_rw, indent=2) # Write the new content
                logger.info(f"Successfully removed metadata entry for doc_id: {doc_id_to_remove}")
                return True
            else:
                logger.warning(f"Doc_id '{doc_id_to_remove}' not found in metadata. No entry removed.")
                return False # Entry was not found to be removed
                
    except json.JSONDecodeError:
        logger.error(f"Metadata file {metadata_path} is corrupted. Cannot remove entry for {doc_id_to_remove}.", exc_info=True)
        return False
    except Exception as e_meta_remove:
        logger.error(f"Error removing entry for {doc_id_to_remove} from metadata: {e_meta_remove}", exc_info=True)
        return False


def clean_response_language(config_dict: Dict, response: str) -> str:
    """
    Cleans up conversational filler and converts Markdown to HTML.
    Args:
        config_dict: The configuration dictionary.
        response: The raw text response from the LLM (expected to be Markdown).
    Returns:
        The cleaned and HTML-formatted response string.
    """
    if not response:
        return ""

    cleaned = response.strip()

    meta_commentary_patterns = [
        # ... (your existing list of regex patterns) ...
        r"Based on the provided context, I found relevant information that addresses your query\.",
        r"Please note that my response is strictly based on the given context and follows the rules outlined\.",
        r"The snippet mentions the LE Chat solution for the problem statement[\s\S]*?\.", # Non-greedy match
        r"This section highlights limitations related to expertise and pressure scenarios\.",
        r"This passage discuss limitations related to sharing documents or information\.",
        r"The text is unclear, but it imply difficulties in sharing knowledge or expertise\.",
        r"Please note that these findings are based solely on the given context and do not represent a comprehensive analysis of all possible limitations\.",
        r"I presented my answer in a clear, concise manner, following the rules outlined\.",
        r"The response includes:",
        r"My answer follows a logical structure, starting with an introduction to the query, followed by a summary of the relevant information, and concluding with a synthesis of the key points\.",
        r"The response is easy to read and understand, making it accessible to users\.",
        r"I found a relevant snippet that answers your query:",
        r"This snippet mentions the LE Chat solution for the problem statement", # More general
        r"Note: These sectors and their corresponding problems are mentioned in the context from Page \d+ of the document\.",
        r"The provided context presents a series of problem statements and corresponding solutions for the LE Chat system\.",
        r"These problem statements and solutions demonstrate the capabilities of LE Chat in addressing various challenges, such as knowledge loss, inaccessibility, and inconsistent guidance",
        r"However, it does provide information about.*?which be related to business objectives\.",
        r"Please note that I've only extracted relevant information from the provided context based on the conversation history and user query\.",
        r"These two points are related to business objectives, but the provided context does not contain a comprehensive list of business objectives\."
    ]
    for pattern in meta_commentary_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE | re.DOTALL).strip()

    section_number_patterns = [
        r"^\d+\.\d+\s*", # e.g., "2.1 " at start of line
        r"^\d+\s*",      # e.g., "1. " at start of line
        # Add other patterns as needed
    ]
    for pattern in section_number_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE | re.MULTILINE).strip()

    try:

        html_output = markdown.markdown(cleaned)
        logger.debug("Converted text to HTML using 'markdown' library.")

        return html_output # Return the generated HTML

    except Exception as e_markdown:
        logger.error(f"Error converting markdown to HTML: {e_markdown}. Returning raw text.", exc_info=True)
      
        return html.escape(cleaned)


def generate_no_information_response(config_dict: Dict, query: str, selected_doc_ids: List[str], all_available_files_metadata: List[Dict]) -> str:
   
    files_str_list = []
    meta_map_for_display = {m['filename']: m.get('display_name', m['filename']) for m in all_available_files_metadata}
    
    for doc_id_loop in selected_doc_ids[:3]: # Show up to 3 display names
        display_name_for_msg = meta_map_for_display.get(doc_id_loop, doc_id_loop) # Fallback to doc_id
        files_str_list.append(f"'{display_name_for_msg}'")
    
    files_str_display = ", ".join(files_str_list)
    if len(selected_doc_ids) > 3:
        files_str_display += f" and {len(selected_doc_ids) - 3} other document(s)"
    elif not selected_doc_ids:
        files_str_display = "the selected document(s)"
        
    safe_query_html = query.replace("<", "<").replace(">", ">") # Basic HTML escape
    
    response_html = f"""<h2>Information Not Found</h2>
<p>I couldn't find specific information about "<b>{safe_query_html}</b>" within {files_str_display}.</p>
<p>This could be because:</p><ul><li>The topic isn't covered in the selected document(s).</li><li>The keywords used in your query don't match the terminology in the text.</li></ul>
<p><b>Suggestions:</b></p><ul><li>Try rephrasing your question.</li><li>Ensure the relevant document(s) are selected and fully processed.</li></ul>"""
    return response_html


# --- Response Tracking Utility ---
def track_response_quality(config_dict: Dict, query: str, response: str, context: List[Dict], client_id: str):
    """
    Logs details about the AI response and the context items used to generate it.
    Records metrics, files referenced, and previews of the content and context.
    Adapts to the new context metadata structure (YOLO/LP based).

    Args:
        config_dict: The application configuration dictionary.
        query: The user's query text.
        response: The raw response text generated by the LLM.
        context: A list of context item dictionaries retrieved from the vector store.
        client_id: The ID of the client session.
    """
    try:
        tracking_dir_path = config_dict.get("response_tracking_dir", "response_tracking") # Example config key
        os.makedirs(tracking_dir_path, exist_ok=True)

        # Create a unique ID for this tracking log entry
        query_content_hash = hashlib.md5(query.encode('utf-8')).hexdigest()[:8] # Encode query before hashing
        current_timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_tracking_id = f"{current_timestamp_str}_{client_id}_{query_content_hash}"

        # --- Basic metrics ---
        # Check for common HTML elements in the response (assuming clean_response_language outputs HTML)
        has_heading_html = bool(re.search(r'<h[1-6]>', response, re.IGNORECASE))
        has_list_html = bool(re.search(r'<(ul|ol)>.*?</\1>', response, re.IGNORECASE | re.DOTALL))
        num_context_items = len(context)
        # Calculate average score of context items
        avg_context_score_val = sum(ctx.get("score", 0.0) for ctx in context) / max(1, num_context_items) if context else 0.0
        response_text_length = len(response)

        # --- Identify files used based on context metadata ---
        # Use metadata['doc_id'] and metadata['original_filename'] which are now standard
        files_referenced_in_context = sorted(list(set(
            # Get original_filename from metadata if available, fallback to doc_id
            ctx.get("metadata", {}).get("original_filename", ctx.get("metadata", {}).get("doc_id", "Unknown"))
            for ctx in context if ctx.get("metadata")
        )))

        # --- Prepare the tracking data dictionary ---
        tracking_data_dict = {
            "tracking_id": unique_tracking_id,
            "timestamp_iso": datetime.now().isoformat(),
            "client_id": client_id,
            "query_text": query,
            "metrics": {
                "has_html_heading": has_heading_html,
                "has_html_list": has_list_html,
                "context_item_count": num_context_items,
                "avg_context_score": round(avg_context_score_val, 4),
                "response_char_length": response_text_length,
            },
            "files_used_in_context": files_referenced_in_context,
            "response_text_preview": response[:500].replace('\n', ' ') + ("..." if len(response) > 500 else ""), # Increased preview size
            # --- Prepare preview for each context item, including new metadata ---
            "context_items_preview": [
                {
                    # Extract new metadata keys from the 'metadata' dictionary
                    "doc_id": item.get("metadata", {}).get("doc_id", "Unknown"),
                    "original_filename": item.get("metadata", {}).get("original_filename", "Unknown"),
                    "page_number": item.get("metadata", {}).get("page_number", "N/A"),
                    "chunk_index_on_page": item.get("metadata", {}).get("chunk_index_on_page", "N/A"),
                    "layout_category": item.get("metadata", {}).get("layout_category", "unknown"),
                    "layout_tool": item.get("metadata", {}).get("layout_tool", "unknown"),
                    # Box and source region boxes might be too verbose for simple preview, include score and content preview
                    "score": round(item.get("score", 0.0), 4),
                    "content_preview": item.get("content", "")[:200].replace('\n', ' ') + ("..." if len(item.get("content", "")) > 200 else "") # Increased preview size
                }
                for item in context # Log all context items used, not just top 3
            ]
        }

        # Write the tracking data to a JSON file
        output_log_path = os.path.join(tracking_dir_path, f"{unique_tracking_id}.json")
        try:
            with open(output_log_path, "w", encoding="utf-8") as f_track:
                json.dump(tracking_data_dict, f_track, indent=2)
            logger.info(f"Tracked response quality: id={unique_tracking_id}, score={avg_context_score_val:.2f}, length={response_text_length}, files_in_ctx={len(files_referenced_in_context)}, num_ctx_items={num_context_items}")
        except IOError as e_write:
            logger.error(f"IOError writing tracking file {output_log_path}: {e_write}", exc_info=True)
        except Exception as e_save_json:
             logger.error(f"Error serializing/saving tracking JSON for {unique_tracking_id}: {e_save_json}", exc_info=True)


    except Exception as e_track:
        logger.error(f"Unexpected error in track_response_quality function: {e_track}", exc_info=True)