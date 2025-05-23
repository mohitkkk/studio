
import os
import json
import re
import time
import logging
import traceback
from typing import List, Dict, Optional, Tuple, Any
import asyncio
import torch
import html
import random

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None
    logging.getLogger(__name__).warning("pypdf not found for textual_pipeline. Textual PDF processing will fail if attempted.")

import ollama # For direct ollama.embeddings

# Imports for type hinting the passed objects
from openai import OpenAI # For type hinting ollama_llm_client_obj
# from ..chatbot_api import ConnectionManager # If ConnectionManager was in chatbot_api and importable (Careful with circular)
# For now, using 'Any' for connection_manager_instance is fine, or define a Protocol.

from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage

logger = logging.getLogger(__name__)

# Define HEADING_KEYWORDS_TEXTUAL here or get from config_dict within get_relevant_context_textual
HEADING_KEYWORDS_TEXTUAL = [ # Example, can be expanded or moved to config
    "section", "chapter", "part", "introduction", "overview", "summary",
    "conclusion", "references", "appendix"
]


# --- Text Processing and Chunking ---
def preprocess_document_textual(config_dict: Dict, text: str) -> str:
    # ... (implementation as you provided) ...
    logger.debug(f"Textual Preprocessing: Input length {len(text)}")
    text = re.sub(r'(\d+\.\d+)(\w)', r'\1 \2', text)
    text = re.sub(r'\.([A-Z])', r'. \1', text)
    text = re.sub(r'[ \t]+\n', '\n', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    processed_text = text.strip()
    logger.debug(f"Textual Preprocessing: Output length {len(processed_text)}")
    return processed_text

def chunk_text_textual(config_dict: Dict, text: str) -> List[str]: # Removed unused max_chunk_size, overlap params for now
    max_chunk_size = config_dict.get("textual_max_chunk_size", 800)
    # overlap = config_dict.get("textual_chunk_overlap", 0) # Overlap not used in your provided logic
    # ... (implementation as you provided, using max_chunk_size from config_dict) ...
    logger.debug(f"Textual Chunking: Input length {len(text)}, max_size={max_chunk_size}")
    paragraphs = re.split(r'\n\s*\n', text) 
    chunks = []
    current_chunk_text = ""
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph: continue
        if len(current_chunk_text) + len(paragraph) + 2 <= max_chunk_size:
            if current_chunk_text: current_chunk_text += "\n\n" + paragraph
            else: current_chunk_text = paragraph
        else:
            if current_chunk_text: chunks.append(current_chunk_text)
            if len(paragraph) > max_chunk_size:
                for i in range(0, len(paragraph), max_chunk_size):
                    chunks.append(paragraph[i:i+max_chunk_size])
                current_chunk_text = "" 
            else:
                current_chunk_text = paragraph
    if current_chunk_text: chunks.append(current_chunk_text)
    final_chunks = [c.strip() for c in chunks if c.strip()]
    logger.debug(f"Textual Chunking: Produced {len(final_chunks)} chunks.")
    return final_chunks


def extract_text_from_pdf_textual(config_dict: Dict, file_path: str) -> str:
    # ... (implementation as you provided) ...
    if not PdfReader:
        logger.error("Textual PDF processing requires 'pypdf'. Please ensure it's installed.")
        return ""
    try:
        reader = PdfReader(file_path)
        text_parts = []
        for i, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                if page_text: text_parts.append(page_text)
            except Exception as page_e:
                logger.warning(f"Could not extract text from page {i+1} of PDF '{file_path}': {page_e}")
        full_text = "\n\n".join(text_parts)
        logger.info(f"Extracted {len(full_text)} chars from PDF '{file_path}' for textual pipeline.")
        return full_text
    except Exception as e:
        logger.error(f"Error extracting text from PDF {file_path} for textual pipeline: {str(e)}", exc_info=True)
        return ""


# --- Core Textual Pipeline Functions ---
def prepare_textual_document(
    config_dict: Dict, 
    source_file_path: str,
    doc_id: str,
    original_filename: str
) -> Optional[str]:
    # ... (implementation as you provided, using config_dict) ...
    logger.info(f"Preparing textual document for ID '{doc_id}' from '{original_filename}'")
    file_ext = os.path.splitext(original_filename)[1].lower()
    content = ""
    if file_ext == ".pdf":
        content = extract_text_from_pdf_textual(config_dict, source_file_path)
    elif file_ext in [".txt", ".md", ".json", ".py", ".js", ".html", ".css", ".csv", ".log", ".xml", ".ini", ".yaml", ".yml"]:
        try:
            with open(source_file_path, "r", encoding="utf-8", errors="ignore") as f: content = f.read()
            logger.info(f"Read {len(content)} chars from text file '{original_filename}'")
        except Exception as e:
            logger.error(f"Error reading text file '{original_filename}' at '{source_file_path}': {e}", exc_info=True)
            return None
    else:
        logger.warning(f"Attempting generic text read for '{original_filename}' (ext: '{file_ext}') for textual pipeline.")
        try:
            with open(source_file_path, "rb") as f_rb: raw_content = f_rb.read()
            content = raw_content.decode("utf-8", errors="ignore")
            logger.info(f"Generic text read for '{original_filename}' yielded {len(content)} chars.")
        except Exception as e_generic:
            logger.error(f"Error reading generic file '{original_filename}' as text: {e_generic}", exc_info=True)
            return None
    if not content or not content.strip():
        logger.warning(f"No text content extracted from '{original_filename}' (doc_id='{doc_id}') for textual pipeline.")
        return None
    processed_text = preprocess_document_textual(config_dict, content)
    chunks = chunk_text_textual(config_dict, processed_text) # Uses textual_max_chunk_size from config
    if not chunks:
        logger.warning(f"No chunks produced for '{original_filename}' (doc_id='{doc_id}') after textual processing.")
        return None
    processed_vault_file_name = f"{doc_id}_textual.txt"
    output_path = os.path.join(config_dict["vault_directory"], processed_vault_file_name)
    try:
        with open(output_path, "w", encoding="utf-8") as f_out:
            for chunk_item in chunks: f_out.write(chunk_item + "\n\n")
        logger.info(f"Saved {len(chunks)} processed textual chunks for doc_id '{doc_id}' to '{output_path}'")
        return processed_vault_file_name
    except Exception as e_save:
        logger.error(f"Failed to save processed textual chunks for doc_id '{doc_id}' to '{output_path}': {e_save}", exc_info=True)
        return None


def read_vault_content_textual(config_dict: Dict, selected_doc_ids: List[str], all_doc_meta: List[Dict]) -> Dict[str, List[str]]:
    # ... (implementation as you provided, using config_dict) ...
    content_by_doc_id = {}
    logger.info(f"Reading textual vault content for doc_ids: {selected_doc_ids}")
    doc_meta_map = {meta['filename']: meta for meta in all_doc_meta}
    for doc_id in selected_doc_ids:
        meta = doc_meta_map.get(doc_id)
        if not meta or meta.get("pipeline_type") != "textual":
            logger.warning(f"Skipping doc_id '{doc_id}': not found in metadata or not a textual pipeline type.")
            continue
        processed_filename = meta.get("processed_path", f"{doc_id}_textual.txt")
        file_path = os.path.join(config_dict["vault_directory"], processed_filename)
        if not os.path.exists(file_path):
            logger.warning(f"Processed textual file '{processed_filename}' not found for doc_id '{doc_id}' at '{file_path}'.")
            continue
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                file_content = f.read()
                chunks = [chunk.strip() for chunk in file_content.split("\n\n") if chunk.strip()]
                if chunks:
                    content_by_doc_id[doc_id] = chunks
                    logger.debug(f"Read {len(chunks)} textual chunks from '{file_path}' for doc_id '{doc_id}'")
                else:
                    logger.warning(f"No textual content chunks found in processed file: '{file_path}' for doc_id '{doc_id}'")
        except Exception as e:
            logger.error(f"Error reading textual processed file '{file_path}' for doc_id '{doc_id}': {e}", exc_info=True)
    return content_by_doc_id


async def generate_vault_embeddings_textual(
    config_dict: Dict,                  # Argument 1
    content: Dict[str, List[str]],      # Argument 2 (e.g., {doc_id: [chunks]})
    client_id: str,                     # Argument 3
    manager: Any,                       # Argument 4 (e.g., ConnectionManager - use Any for type safety against circular imports)
    common_utils_module: Any,           # Argument 5 (e.g., common_utils module reference - use Any)
    ollama_client: Optional[OpenAI]     # Argument 6 (the LLM client instance from init_ollama_client)
) -> Dict[str, torch.Tensor]: # type: ignore # Type ignore due to potential issues with Dict[str, torch.Tensor] if empty or mixed
    """
    Generates embeddings for textual content chunks for a given client session
    using the Ollama client (OpenAI compatible endpoint).

    Args:
        config_dict: The configuration dictionary.
        content: A dictionary mapping doc_id to a list of text chunks for that doc.
        client_id: The ID of the client session.
        manager: The ConnectionManager instance.
        common_utils_module: Reference to the common_utils module.
        ollama_client: The initialized OpenAI-compatible Ollama client.

    Returns:
        A dictionary mapping doc_id to a stacked tensor of embeddings for its chunks.
        Returns an empty dict if no content or no client. Raises Exception on failure.
    """
    # Use the device specified in the config, or fallback
    device = config_dict.get("device", torch.device("cpu"))
    if not isinstance(device, torch.device): # Fallback if not a torch.device object
        device = torch.device("cuda" if torch.cuda.is_available() and config_dict.get("use_gpu") else "cpu")
        logger.warning(f"Textual pipeline: Device in config was not a torch.device object, determined device: {device}")


    embeddings_by_doc_id: Dict[str, torch.Tensor] = {} # Initialize return dictionary
    total_docs_to_embed = len(content)

    if ollama_client is None:
        logger.error("Textual pipeline: Ollama client not available, cannot generate embeddings.")
        if client_id and manager: # Notify client if possible
            await manager.send_json(client_id, {"type": "status", "message": "Embedding failed: LLM client not available."})
        # Raise a specific exception to signal critical failure back to caller
        raise ConnectionError("LLM client is not initialized or available.") # Use a more specific error type if appropriate


    if not content:
        logger.info("Textual pipeline: No content provided for embedding generation.")
        if client_id and manager:
             await manager.send_json(client_id, {"type": "status", "message": "Embedding skipped: No textual content to process."})
        return {}


    docs_embedded_count = 0
    # Iterate through each document and its chunks
    for doc_id, chunks in content.items():
        docs_embedded_count += 1 # Increment count for status updates

        if not chunks:
            logger.info(f"Textual pipeline: No chunks found for doc_id '{doc_id}', skipping embedding generation.")
            continue # Skip this document if it has no chunks

        logger.info(f"Textual pipeline: Generating embeddings for {len(chunks)} chunks in doc_id '{doc_id}' ({docs_embedded_count}/{total_docs_to_embed})...")
        status_msg_prefix = f"Generating embeddings for '{doc_id}' ({docs_embedded_count}/{total_docs_to_embed})..."

        if client_id and manager:
            # Send status update for the current document processing
            # Using HTML escape for doc_id in case it contains problematic characters
            await manager.send_json(client_id, {"type": "status", "message": f"Generating embeddings for '{html.escape(doc_id)}' ({docs_embedded_count}/{total_docs_to_embed})..."}) # Added escape

        all_chunk_embeddings: List[List[float]] = [] # List to collect embeddings for all chunks in this doc
        batch_size = config_dict.get("embedding_batch_size", 16) # Get batch size from config
        total_chunks_in_doc = len(chunks)

        try:
            # Process chunks in batches
            for i in range(0, total_chunks_in_doc, batch_size):
                batch_of_chunks = chunks[i:i+batch_size]

                if not batch_of_chunks: continue # Skip if batch is empty

                logger.debug(f"Textual pipeline: Processing batch {i // batch_size + 1} for doc '{doc_id}' ({len(batch_of_chunks)} chunks)...")

                # --- Use ollama_client.embeddings.create ---
                # Call the OpenAI-compatible embeddings endpoint
                # This is a synchronous call in the OpenAI client, so run it in a thread.
                try:
                    embedding_response = await asyncio.to_thread(
                         # Use the ollama_client instance
                         ollama_client.embeddings.create,
                         # Arguments for client.embeddings.create:
                         model=config_dict["ollama_embedding_model"], # Use the configured embedding model
                         input=batch_of_chunks # Pass the list of strings (chunks)
                    )

                    # Process the response
                    if embedding_response and hasattr(embedding_response, 'data') and isinstance(embedding_response.data, list):
                        # Extract the embedding vectors from the response objects
                        batch_embeddings = [item.embedding for item in embedding_response.data if hasattr(item, 'embedding') and item.embedding is not None] # Ensure embedding is not None
                        all_chunk_embeddings.extend(batch_embeddings)
                        logger.debug(f"Textual pipeline: Generated {len(batch_embeddings)} embeddings for batch {i // batch_size + 1}.")
                    else:
                        logger.warning(f"Textual pipeline: Invalid embedding response format for batch {i // batch_size + 1} in doc_id '{doc_id}'. Response structure unexpected.")
                        # Decide if invalid response for a batch should fail the whole doc
                        # Raising here will stop embedding for this doc and be caught below
                        raise ValueError(f"Invalid embedding response format for batch {i//batch_size + 1}")


                except Exception as e_embedding_batch:
                    # Log batch specific error but allow loop to continue to next batch if possible
                    logger.error(f"Textual pipeline: Error generating embeddings for batch {i // batch_size + 1} for doc_id '{doc_id}': {e_embedding_batch}", exc_info=True)
                    # Decide if a batch error should fail the entire document's embedding process
                    # For now, it just logs and continues, meaning only failed batches are skipped.
                    # If you want ANY batch error to fail the doc, add 'raise' here.


            # After processing all batches for this document, check if any embeddings were generated
            if all_chunk_embeddings:
                # Convert the list of lists of floats into a torch tensor
                embeddings_tensor = torch.tensor(all_chunk_embeddings, dtype=torch.float32).to(device)
                embeddings_by_doc_id[doc_id] = embeddings_tensor
                logger.info(f"Textual pipeline: Generated stacked embedding tensor shape {embeddings_tensor.shape} for doc_id '{doc_id}' on {device}.")
            else:
                logger.warning(f"Textual pipeline: No textual embeddings were successfully generated for *any* chunk in doc_id '{doc_id}'.")
                # --- CRITICAL FIX: Raise an exception if embedding generation failed for the doc ---
                # This signals back to handle_file_selection that this document failed.
                raise RuntimeError(f"Failed to generate embeddings for doc_id '{doc_id}'")
                # --- END CRITICAL FIX ---


        except Exception as e_doc_embed:
            # Catch errors specific to processing this document (outside the batch loop or the final check)
            logger.error(f"Textual pipeline: Unhandled error during embedding generation for doc_id '{doc_id}': {e_doc_embed}", exc_info=True)
            # Re-raise the exception so it's caught by handle_file_selection's broader except block
            # which will add the document to the failed_names list.
            raise # Re-raise the caught exception


    # --- Final status update after processing all documents (controlled by handle_file_selection) ---
    logger.info(f"Textual pipeline: Finished embedding generation process for client {client_id}. Attempted {total_docs_to_embed} docs, successfully generated embeddings for {len(embeddings_by_doc_id)} docs.")
    # The final 'Processing complete' status is sent by handle_file_selection.

    return embeddings_by_doc_id # Return the dictionary {doc_id: stacked_embeddings}


async def get_relevant_context_textual(
    config_dict: Dict,                          # Argument 1 (e.g., CONFIG)
    query: str,                                 # Argument 2 (user's query)
    session_embeddings_by_doc_id: Dict[str, torch.Tensor], # Argument 3 {doc_id: tensor_of_embeddings}
    session_content_by_doc_id: Dict[str, List[str]],      # Argument 4 {doc_id: [list_of_chunks]}
    target_doc_ids: List[str],                  # Argument 5 List of doc_ids to search within
    llm_client: Optional[OpenAI],               # Argument 6 (Needed for query embedding)
    common_utils_module: Any,                   # Argument 7 (Needed for get_specific_doc_metadata)
    top_k_per_file: int = 7,
    similarity_threshold: float = 0.5,
    force_summary: bool = False                 # Argument N Flag from caller
) -> List[Dict]:
    """
    Retrieves relevant textual context chunks for a query from specified documents
    that are already loaded into the client session manager.

    Args:
        config_dict: The configuration dictionary.
        query: The user's query string.
        session_embeddings_by_doc_id: Dictionary mapping doc_id to its embeddings tensor for the session.
        session_content_by_doc_id: Dictionary mapping doc_id to its list of text chunks for the session.
        target_doc_ids: List of document IDs to search within.
        llm_client: The initialized OpenAI-compatible Ollama client (needed for query embedding).
        common_utils_module: Reference to the common_utils module (needed for metadata lookup).
        top_k_per_file: Max number of top similar chunks to get per document (for RAG).
        similarity_threshold: Minimum similarity score for chunks to be considered relevant (for RAG).
        force_summary: If True, prioritize sampling chunks across the document (summary mode).

    Returns:
        A list of dictionaries, where each dict represents a relevant context chunk
        with 'content', 'score', and 'metadata' (including 'doc_id').
        Returns an empty list if no relevant context is found or errors occur.
    """
    logger.info(f"Textual context search for query '{query[:50]}...' in doc_ids: {target_doc_ids}. Summary mode: {force_summary}")

    relevant_chunks: List[Dict] = [] # List to collect all relevant chunks from all docs

    # If no embeddings or content loaded for the session, or no target docs provided
    if not session_embeddings_by_doc_id or not session_content_by_doc_id or not target_doc_ids:
        logger.warning("Textual context retrieval skipped: No session data or target docs provided.")
        return []

    # Use the device specified in the config, or fallback
    device = config_dict.get("device", torch.device("cpu"))
    if not isinstance(device, torch.device): # Fallback if not a torch.device object
        device = torch.device("cuda" if torch.cuda.is_available() and config_dict.get("use_gpu") else "cpu")
        logger.warning(f"Textual pipeline: Device in config was not a torch.device object, determined device: {device}")


    # --- Generate embedding for the query ---
    # Requires access to the Ollama client for the embeddings model.
    query_embedding_tensor = None
    try:
        # Ensure llm_client is available before calling its methods
        if llm_client is None:
             logger.error("Textual pipeline: Ollama client not available for query embedding.")
             return [] # Cannot proceed without query embedding

        # Use asyncio.to_thread for the synchronous embedding call
        embedding_response = await asyncio.to_thread(
             llm_client.embeddings.create,
             model=config_dict.get("ollama_embedding_model", "mxbai-embed-large:latest"), # Use the configured embedding model name
             input=[query] # embeddings.create expects a list of strings
        )

        if embedding_response and hasattr(embedding_response, 'data') and isinstance(embedding_response.data, list) and embedding_response.data and hasattr(embedding_response.data[0], 'embedding'):
             # Get the embedding vector (first item in the list)
             query_embedding_list = embedding_response.data[0].embedding
             query_embedding_tensor = torch.tensor(query_embedding_list, dtype=torch.float32).to(device)
             logger.debug("Generated query embedding tensor.")
        else:
            logger.warning("Failed to generate query embedding or invalid response format.")
            return [] # Cannot proceed without query embedding

    except Exception as e_query_embed:
        logger.error(f"Error generating query embedding: {e_query_embed}", exc_info=True)
        return [] # Cannot proceed


    # --- Iterate and Search within targeted documents ---
    for doc_id in target_doc_ids:
        doc_embeddings_tensor = session_embeddings_by_doc_id.get(doc_id) # Get embeddings tensor for this doc
        doc_chunks = session_content_by_doc_id.get(doc_id) # Get chunks for this doc

        if doc_embeddings_tensor is None or doc_chunks is None or not doc_chunks or doc_embeddings_tensor.shape[0] == 0:
            logger.warning(f"Textual context search skipped for doc_id {doc_id}: Embeddings or content not loaded for session or empty.")
            continue # Skip if data not available or empty for this doc in the session

        # Ensure embeddings tensor is on the same device as query embedding
        doc_embeddings_tensor = doc_embeddings_tensor.to(device)


        try:
            # Calculate cosine similarity between the query embedding and all document chunks' embeddings
            # Cosine similarity expects shapes like (..., embedding_dim) and (..., embedding_dim)
            # Result shape depends on broadcasting.
            # Let's calculate pairwise similarity (1 query vs N chunks)
            # Unsqueeze query_embedding_tensor to make it 2D (1, embedding_dim)
            if query_embedding_tensor.ndim == 1:
                similarity_scores = torch.nn.functional.cosine_similarity(
                    query_embedding_tensor.unsqueeze(0), # Make query 2D (1, dim) for broadcasting
                    doc_embeddings_tensor,              # (num_chunks, dim)
                    dim=1 # Calculate similarity along dimension 1 (the embedding dimension)
                ) # Resulting shape is (num_chunks,)
            else:
                 logger.error(f"Unexpected query embedding tensor shape: {query_embedding_tensor.shape}. Cannot calculate similarity for doc {doc_id}.")
                 continue # Skip this document


            # --- Select chunks based on mode (Summary vs. RAG) ---
            selected_chunk_indices_for_doc = [] # List of indices for this document

            # Determine total chunks in the current document
            total_chunks_in_doc = len(doc_chunks)

            if force_summary:
                # --- CORRECTED: Build sample_indices without self-reference ---
                indices_to_consider = set()

                if total_chunks_in_doc > 0:
                    # Always include first and last (if they exist)
                    indices_to_consider.add(0)
                    if total_chunks_in_doc > 1: indices_to_consider.add(total_chunks_in_doc - 1)

                    # Include indices at quarter, half, three-quarters positions if enough chunks
                    if total_chunks_in_doc > 10:
                        # Calculate indices safely using integer division
                        indices_to_consider.add(total_chunks_in_doc // 4)
                        indices_to_consider.add(total_chunks_in_doc // 2)
                        indices_to_consider.add(total_chunks_in_doc * 3 // 4)

                    # Optionally, add a few random indices if you want some variability
                    num_random = min(3, total_chunks_in_doc - len(indices_to_consider)) # Avoid sampling more than available unique indices
                    if num_random > 0:
                       available_indices = list(set(range(total_chunks_in_doc)) - indices_to_consider) # Indices not already selected
                       if available_indices:
                           # Ensure we don't sample more than are available
                           num_random = min(num_random, len(available_indices))
                           indices_to_consider.update(random.sample(available_indices, num_random))


                # Convert the set of indices to a sorted list
                # Ensure indices are within bounds (already handled by how we build the set)
                selected_chunk_indices_for_doc = sorted(list(indices_to_consider))

                logger.debug(f"Doc {doc_id} (Summary): Sampled {len(selected_chunk_indices_for_doc)} indices out of {total_chunks_in_doc}.")

                # For summary, assign a score based on position (earlier chunks sometimes preferred for intro)
                # Or simply assign a high score like 1.0 to all included summary chunks
                # Let's use a simple index-based score: score = 1.0 - (index / max(1, total_chunks_in_doc))
                summary_chunk_scores = {idx: max(0.0, 1.0 - (idx / max(1, total_chunks_in_doc))) for idx in selected_chunk_indices_for_doc} # Ensure score >= 0

            else: # Standard RAG query
                # Filter by similarity threshold first
                threshold_indices = torch.where(similarity_scores >= similarity_threshold)[0]
                if not threshold_indices.numel(): # If no chunks meet threshold
                    logger.debug(f"No chunks met similarity threshold ({similarity_threshold:.2f}) for doc {doc_id}.")
                    continue # Skip this document

                # From the thresholded chunks, select the top K
                # Get scores and indices for chunks above the threshold
                threshold_scores = similarity_scores[threshold_indices]
                top_k_to_use = min(len(threshold_indices), top_k_per_file) # Cap at available thresholded chunks
                # Ensure top_k_to_use is at least 1 if threshold_indices.numel() > 0 to avoid error with k=0
                top_k_to_use = max(1, top_k_to_use) if threshold_indices.numel() > 0 else 0

                if top_k_to_use > 0:
                    top_scores_thresholded, top_indices_thresholded_relative = torch.topk(threshold_scores, k=top_k_to_use, largest=True)
                    # Convert relative indices back to original chunk indices
                    selected_chunk_indices_for_doc = threshold_indices[top_indices_thresholded_relative].tolist()
                    # Store similarity scores for these selected chunks
                    rag_chunk_scores = {selected_chunk_indices_for_doc[i]: top_scores_thresholded[i].item() for i in range(len(selected_chunk_indices_for_doc))}
                else:
                    # No chunks after thresholding and top_k filter
                    selected_chunk_indices_for_doc = []
                    rag_chunk_scores = {}


                logger.debug(f"Doc {doc_id} (RAG): Retrieved {len(selected_chunk_indices_for_doc)} chunks (top {top_k_to_use}, threshold {similarity_threshold:.2f})")


            # --- Collect the selected context chunks for this document ---
            doc_meta = common_utils_module.get_specific_doc_metadata(config_dict, doc_id) # Get meta again using passed module
            if doc_meta:
                 # Ensure consistent metadata fields are added to each chunk item
                 base_metadata_for_chunk = {
                     "doc_id": doc_id,
                     "original_filename": doc_meta.get("original_filename", doc_id),
                     "pipeline_type": doc_meta.get("pipeline_type", "textual"),
                     "display_name": doc_meta.get("display_name", doc_id),
                     "source": f"{doc_meta.get('display_name', doc_id)} (Textual)" # Example source string
                     # Add other relevant metadata like page_number if available in textual chunks (unlikely by default)
                 }
                 for idx in selected_chunk_indices_for_doc:
                     if 0 <= idx < len(doc_chunks): # Safety check
                         chunk_content = doc_chunks[idx].strip() # Strip content
                         if not chunk_content: continue # Skip empty chunks even if selected

                         chunk_metadata = base_metadata_for_chunk.copy()
                         # Add chunk-specific metadata like index, score, etc.
                         chunk_metadata["chunk_index"] = idx
                         # Use the appropriate score based on mode
                         if force_summary: # Use force_summary argument
                             chunk_metadata["score"] = summary_chunk_scores.get(idx, 0.0) # Use index-based score
                         else: # RAG mode
                             chunk_metadata["score"] = rag_chunk_scores.get(idx, 0.0) # Use similarity score

                         relevant_chunks.append({
                             "content": chunk_content, # Use the stripped chunk content
                             "metadata": chunk_metadata
                         })
                     # else: Index out of range warning already logged


        except Exception as e_search_doc:
            logger.error(f"Error during textual context search for doc_id '{doc_id}': {e_search_doc}", exc_info=True)
            # Decide if error in one doc search should stop the whole process - logging and skipping this doc seems more robust


    # --- Post-processing of combined context chunks ---
    # Sort all relevant chunks from all documents by score (highest first)
    relevant_chunks.sort(key=lambda x: x.get("metadata", {}).get("score", 0.0), reverse=True)

    # Apply overall limit if the sum of top_k_per_file exceeds the total desired context size
    # Also deduplicate chunks based on content
    final_context_limit = config_dict.get("textual_context_results_limit", 15)
    final_context_chunks_deduplicated: List[Dict] = []
    unique_content_tracker = set()

    for r_item_final in relevant_chunks:
        chunk_content = r_item_final.get("content", "")
        if chunk_content and chunk_content not in unique_content_tracker:
            unique_content_tracker.add(chunk_content)
            final_context_chunks_deduplicated.append(r_item_final)
            # Stop adding chunks once the overall limit is reached
            if len(final_context_chunks_deduplicated) >= final_context_limit:
                 break # Exit the loop once we have enough chunks


    logger.info(f"Returning {len(final_context_chunks_deduplicated)} unique relevant textual chunks for query '{query[:50]}...'.")

    # Return the list of final context chunks (caller will build the prompt string)
    return final_context_chunks_deduplicated

async def ollama_chat_textual_async(
    config_dict: Dict,
    query: str,
    doc_ids: List[str],
    session_embeddings: Dict[str, torch.Tensor],
    session_content: Dict[str, List[str]],
    client_id: str,
    ollama_client: Optional[OpenAI],
    manager: Any,  # ConnectionManager
    common_utils_module: Any,  # common_utils module
    system_message_str: str  # The system message string
) -> Tuple[str, List[Dict]]:
    """
    Retrieves context from loaded textual documents, builds prompt, calls LLM,
    and returns raw response text (Markdown) + context chunks used.

    Args:
        config_dict: The configuration dictionary.
        query: The user's query string.
        doc_ids: List of document IDs targeted for this chat query.
        session_embeddings: Dictionary mapping doc_id to its embeddings tensor for the session.
        session_content: Dictionary mapping doc_id to its list of text chunks for the session.
        client_id: The ID of the client session.
        ollama_client: The initialized OpenAI-compatible Ollama client.
        manager: The ConnectionManager instance (needed for status updates/memory).
        common_utils_module: Reference to the common_utils module (needed for metadata/no-info generation).
        system_message_str: The system message string for the LLM prompt.

    Returns:
        A tuple containing:
        - The raw text response from the LLM (expected to be Markdown).
        - A list of context chunk dictionaries that were used.
        Returns ("", []) if no documents were targeted or no relevant context was found by RAG.
        Returns an error message string and [] if the LLM call fails.
    """
    logger.info(f"--- ollama_chat_textual_async for client {client_id} --- Query: '{query[:50]}...'")

    if ollama_client is None:
        logger.error("LLM client not available for textual chat.")
        try:
            await manager.send_json(client_id, {"type": "status", "message": html.escape("Chat failed: LLM client not available.")})
        except Exception as e:
            logger.error(f"Failed to send status for client {client_id}: {e}", exc_info=True)
        raise ConnectionError("LLM client is not initialized or available.")

    if not doc_ids:
        logger.warning("No textual documents targeted for this specific chat query.")
        try:
            await manager.send_json(client_id, {"type": "status", "message": html.escape("Chat failed: No textual documents targeted.")})
        except Exception as e:
            logger.error(f"Failed to send status for client {client_id}: {e}", exc_info=True)
        return "", []

    # Get chat history from manager (assuming memory stores history)
    memory = manager.get_client_memory(client_id)
    chat_history_lc_msgs = []
    if memory:
        try:
            loaded_vars = await asyncio.to_thread(memory.load_memory_variables, {})
            chat_history_lc_msgs = loaded_vars.get(memory.memory_key, [])
            if not isinstance(chat_history_lc_msgs, list):
                logger.warning(f"Memory for client {client_id} returned non-list history: {type(chat_history_lc_msgs)}")
                chat_history_lc_msgs = []
            logger.info(f"Loaded {len(chat_history_lc_msgs)} history messages for textual chat (client: {client_id}).")
        except Exception as e_mem_load:
            logger.error(f"Error loading memory for textual chat (client: {client_id}): {e_mem_load}", exc_info=True)
            chat_history_lc_msgs = []

    formatted_history_for_prompt = [{"role":"user" if isinstance(m,HumanMessage) else "assistant", "content":m.content} for m in chat_history_lc_msgs if hasattr(m, 'content') and m.content is not None]

    # Retrieve relevant context chunks
    try:
        logger.debug(f"Sending status: Searching {len(doc_ids)} textual document(s) for context...")
        await manager.send_json(client_id, {"type": "status", "message": f"Searching {len(doc_ids)} textual document(s) for context..."})
    except Exception as e:
        logger.error(f"Failed to send context search status for client {client_id}: {e}", exc_info=True)

    relevant_context_chunks: List[Dict] = []
    try:
        logger.debug(f"Calling get_relevant_context_textual with {len(doc_ids)} doc_ids")
        relevant_context_chunks = await get_relevant_context_textual(
            config_dict,
            query,
            session_embeddings,
            session_content,
            doc_ids,
            ollama_client,
            common_utils_module,
            top_k_per_file=config_dict.get("textual_top_k_per_doc", 5),
            similarity_threshold=config_dict.get("textual_similarity_threshold", 0.45),
            force_summary=False
        )
        logger.debug(f"get_relevant_context_textual returned {len(relevant_context_chunks)} chunks.")

        if not relevant_context_chunks:
            logger.warning(f"No relevant textual context found by RAG for query '{query[:50]}...' in selected docs {doc_ids}.")
            try:
                await manager.send_json(client_id, {"type": "status", "message": html.escape("No relevant information found in selected documents.")})
            except Exception as e:
                logger.error(f"Failed to send no-context status for client {client_id}: {e}", exc_info=True)
            logger.info("Textual chat returning no-info condition from RAG search.")
            return "", []

    except Exception as e_get_context:
        logger.error(f"Error retrieving textual context for client {client_id}: {e_get_context}", exc_info=True)
        error_msg = f"Error retrieving context: {str(e_get_context)[:100]}"
        try:
            await manager.send_json(client_id, {"type": "error", "message": html.escape(error_msg)})
        except Exception as e:
            logger.error(f"Failed to send context error for client {client_id}: {e}", exc_info=True)
        return error_msg, []

    # Build context string for LLM
    context_str_for_llm = ""
    if relevant_context_chunks:
        context_parts = []
        grouped_context = {}
        for chunk in relevant_context_chunks:
            meta = chunk.get("metadata", {})
            doc_id = meta.get("doc_id", "Unknown")
            grouped_context.setdefault(doc_id, []).append(chunk)

        sorted_doc_ids = sorted(grouped_context.keys())
        for doc_id in sorted_doc_ids:
            chunks_for_doc = grouped_context[doc_id]
            chunks_for_doc.sort(key=lambda c: c.get('metadata', {}).get('chunk_index', 0))
            doc_meta = common_utils_module.get_specific_doc_metadata(config_dict, doc_id) or {}
            doc_display_name = html.escape(doc_meta.get('display_name', doc_id))
            context_parts.append(f"--- Context from Document: {doc_display_name} (ID: {doc_id}) ---")
            for chunk in chunks_for_doc:
                content = chunk.get('content', '')
                context_parts.append(content)
            context_parts.append(f"--- End Context from {doc_display_name} ---")
        context_str_for_llm = "\n\n".join(context_parts)
        logger.debug(f"Built context string for LLM for client {client_id}. Length: {len(context_str_for_llm)}. Preview: {context_str_for_llm[:200]}...")
    else:
        logger.debug("No final context chunks selected for building prompt string.")
        context_str_for_llm = "No relevant context provided."

    # Construct LLM messages
    messages_for_api = [
        {"role": "system", "content": system_message_str},
        *formatted_history_for_prompt,
        {"role": "user", "content": f"Context:\n{context_str_for_llm}\n\nUser Query: {query}"}
    ]

    # Call LLM
    try:
        logger.debug("Sending status: Generating textual response...")
        await manager.send_json(client_id, {"type": "status", "message": "Generating textual response..."})
    except Exception as e:
        logger.error(f"Failed to send response generation status for client {client_id}: {e}", exc_info=True)

    raw_llm_response_text = ""
    logger.info(f"Calling LLM for client {client_id}. Model: {config_dict.get('ollama_model')}. Query: '{query[:100]}...'")
    try:
        llm_api_response = await asyncio.wait_for(
            asyncio.to_thread(
                ollama_client.chat.completions.create,
                model=config_dict.get("ollama_model", "llama3:latest"),
                messages=messages_for_api,
                temperature=config_dict.get("textual_llm_temperature", 0.2),
                max_tokens=config_dict.get("textual_llm_max_tokens", 1024),
            ),
            timeout=30.0
        )
        raw_llm_response_text = llm_api_response.choices[0].message.content
        if raw_llm_response_text is not None:
            raw_llm_response_text = raw_llm_response_text.strip()
        else:
            raw_llm_response_text = ""
        logger.info(f"Received raw LLM response for client {client_id}. Preview: {raw_llm_response_text[:100]}...")

        # Save context to memory
        if memory:
            try:
                await asyncio.to_thread(memory.save_context, {"input": query}, {"output": raw_llm_response_text})
            except Exception as e_mem_save:
                logger.error(f"Error saving chat context to memory for {client_id}: {e_mem_save}", exc_info=True)

        return raw_llm_response_text, relevant_context_chunks

    except asyncio.TimeoutError:
        logger.error(f"LLM call timed out for client {client_id}. Query: '{query[:100]}...'")
        error_msg = "Query timed out while generating response. Please try again."
        try:
            await manager.send_json(client_id, {"type": "error", "message": html.escape(error_msg)})
        except Exception as e:
            logger.error(f"Failed to send timeout error for client {client_id}: {e}", exc_info=True)
        return error_msg, []
    except Exception as e_llm_call:
        logger.error(f"LLM call error in textual chat pipeline for {client_id}: {e_llm_call}", exc_info=True)
        error_msg = f"Error calling Language Model: {str(e_llm_call)[:100]}"
        try:
            await manager.send_json(client_id, {"type": "error", "message": html.escape(error_msg)})
        except Exception as e:
            logger.error(f"Failed to send LLM error for client {client_id}: {e}", exc_info=True)
        return error_msg, []
