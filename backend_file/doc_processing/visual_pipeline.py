import os
import json
import logging
import traceback
import asyncio
import re
import math
import time # For doc_id generation in process_uploaded_file if moved here
import shutil # For process_uploaded_file if moved here
import uuid # For doc_id generation in process_uploaded_file if moved here
from datetime import datetime # For index_ocr_data, perform_layout_analysis etc.

import fitz  # PyMuPDF
import cv2
import numpy as np
from ultralytics import YOLO
import pytesseract
from pytesseract import Output
import pandas as pd # Used by _get_text_blocks_tesseract if you keep it DataFrame based
import chromadb # Specifically for type hinting chroma_collection_obj
import ollama # For ollama.embeddings in get_visual_context_chroma

from typing import List, Dict, Optional, Tuple, Any 

# --- LLM and DB Libraries ---
import torch # For tensor operations (similarity, device handling)
from openai import OpenAI # <-- ADD THIS IMPORT: For type hinting and calling Ollama via OpenAI compatibility
import chromadb # For type hinting ChromaDB objects
from langchain.schema import HumanMessage, AIMessage # <-- ADD THIS IMPORT: For type hinting and checking history messages

# --- Standard Python Typing ---
from typing import List, Dict, Optional, Tuple, Any
from .common_utils import get_vault_files, update_file_metadata, get_specific_doc_metadata # Functions for metadata/vault access
import html # <-- ADD THIS IMPORT: For html.escape()

logger = logging.getLogger(__name__)

# --- HELPER FUNCTIONS (mostly internal to this module) ---

def _calculate_iou(boxA: List[int], boxB: List[int]) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    denominator = float(boxAArea + boxBArea - interArea)
    return interArea / denominator if denominator > 0 else 0.0

def _calculate_overlap_ratio(box_contour: List[int], box_text: List[int]) -> float:
    """Calculates the ratio of intersection area to the area of box_contour."""
    xA = max(box_contour[0], box_text[0])
    yA = max(box_contour[1], box_text[1])
    xB = min(box_contour[2], box_text[2])
    yB = min(box_contour[3], box_text[3])
    # Original BUG: interArea = max(0, xB - xA) * max(0, yB - yB)
    # Corrected Calculation:
    interArea = max(0, xB - xA) * max(0, yB - yA)

    boxContourArea = (box_contour[2] - box_contour[0]) * (box_contour[3] - box_contour[1])
    return interArea / float(boxContourArea) if boxContourArea > 0 else 0.0

def _deskew_image(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    try:
        gray_not = cv2.bitwise_not(gray)
        thresh = cv2.threshold(gray_not, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        coords = np.column_stack(np.where(thresh > 0))

        if len(coords) < 10: # Heuristic: need enough points for a reliable angle
            logger.debug("Deskewing: Not enough points detected, returning original image.")
            return image

        rect = cv2.minAreaRect(coords) # ((center_x, center_y), (width, height), angle)
        angle = rect[-1]

        # Normalize angle
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle # cv2.getRotationMatrix2D expects counter-clockwise angle

        if abs(angle) < 0.5: # Don't rotate for very small angles
            logger.debug(f"Deskewing: Angle {angle:.2f} too small, skipping rotation.")
            return image

        logger.info(f"Deskewing: Detected angle: {angle:.2f} degrees for rotation.")
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        # Use white for border to avoid black edges after rotation, common for documents
        border_value = (255, 255, 255) if len(image.shape) == 3 else 255 
        rotated = cv2.warpAffine(image, M, (w, h), 
                                 flags=cv2.INTER_CUBIC, 
                                 borderMode=cv2.BORDER_CONSTANT, 
                                 borderValue=border_value)
        return rotated
    except cv2.error as cv_err: # Catch OpenCV specific errors
        logger.error(f"OpenCV error during deskew calculation or rotation: {cv_err}", exc_info=True)
        return image
    except Exception as e:
         logger.error(f"Unexpected error during deskewing: {e}", exc_info=True)
         return image

def preprocess_image_opencv(image: np.ndarray, config_dict: Dict) -> Optional[np.ndarray]:
    try:
        processed_image = image.copy() # Work on a copy

        # Convert to grayscale if it's a color image
        if len(processed_image.shape) == 3 and processed_image.shape[2] == 3:
            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
        elif len(processed_image.shape) == 3 and processed_image.shape[2] == 4: # BGR-Alpha
            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGRA2GRAY)
        
        # Deskewing (optional, controlled by config)
        if config_dict.get("enable_deskewing", True):
            processed_image = _deskew_image(processed_image)

        # Noise reduction (optional)
        if config_dict.get("enable_denoising_visual", False): # Example: make it configurable
            # Parameters can also come from config_dict
            h_denoise = config_dict.get("denoising_h", 10)
            processed_image = cv2.fastNlMeansDenoising(processed_image, None, h=h_denoise, templateWindowSize=7, searchWindowSize=21)

        # Binarization (Adaptive thresholding is often good for OCR)
        if config_dict.get("enable_binarization_visual", True):
            block_size = config_dict.get("adaptive_thresh_block_size", 11) # Must be odd
            if block_size % 2 == 0: block_size +=1 # Ensure odd
            c_val = config_dict.get("adaptive_thresh_c", 5) # Constant subtracted from mean
            processed_image = cv2.adaptiveThreshold(processed_image, 255, 
                                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                cv2.THRESH_BINARY, blockSize=block_size, C=c_val)
        
        logger.debug("Image preprocessing for visual pipeline completed.")
        return processed_image
    except Exception as e:
        logger.error(f"Error during OpenCV visual preprocessing: {e}", exc_info=True)
        return None


def _get_text_blocks_tesseract(image_path_or_array: Any, config_dict: Dict, is_image_array: bool = False) -> List[Dict]:
        text_blocks = []
        try:
            # Tesseract configuration
            lang = config_dict.get('tesseract_lang', 'eng')
            psm = config_dict.get('tesseract_ocr_psm', '3') # Default to PSM 3 (auto page segmentation)
            oem = config_dict.get('tesseract_ocr_oem', '3') # Default to LSTM engine
            tess_config = f'--psm {psm} --oem {oem}'
            timeout_ocr = config_dict.get('tesseract_timeout', 90)

            input_for_tesseract = image_path_or_array

            # Perform OCR - Added robust error handling around the pytesseract call
            try:
                data_df = pytesseract.image_to_data(
                    input_for_tesseract,
                    lang=lang,
                    output_type=Output.DATAFRAME,
                    config=tess_config,
                    timeout=timeout_ocr
                )
            except pytesseract.TesseractError as tess_run_err:
                 logger.error(f"Tesseract runtime error during image_to_data: {tess_run_err}", exc_info=True)
                 return [] # Return empty list on Tesseract error
            except Exception as tess_call_e:
                 logger.error(f"Unexpected error during pytesseract.image_to_data call: {tess_call_e}", exc_info=True)
                 return [] # Return empty list on other errors

            # --- FIX START: Robust handling of DataFrame and its contents ---
            # Check if the DataFrame is None or empty immediately after the call
            if data_df is None or data_df.empty:
                logger.debug(f"Tesseract found no text data or returned empty DataFrame from '{os.path.basename(str(image_path_or_array)) if not is_image_array else 'image_array'}'")
                return []

            # Ensure required columns exist and handle potential NaNs in crucial columns
            required_cols = ['page_num', 'block_num', 'conf', 'left', 'top', 'width', 'height', 'text']
            if not all(col in data_df.columns for col in required_cols):
                missing = set(required_cols) - set(data_df.columns)
                logger.error(f"Tesseract DataFrame missing expected columns: {', '.join(missing)}")
                return []

            # Replace NaN/None in 'text' column with empty strings *before* processing
            data_df['text'] = data_df['text'].fillna('').astype(str)
            # Drop rows where crucial numerical geometry or confidence data is missing
            data_df = data_df.dropna(subset=['conf', 'left', 'top', 'width', 'height']).reset_index(drop=True)

            # Filter by confidence (conf) - Applied to the cleaned DataFrame
            # Note: The 'conf' value for block/paragraph levels can be -1. Filtering conf > -1
            # ensures we only consider actual recognized text elements (words, lines).
            data_df_filtered_conf = data_df[data_df.conf > config_dict.get('min_char_confidence_tess', -1)].copy() # Use .copy() to avoid SettingWithCopyWarning

            if data_df_filtered_conf.empty:
                logger.debug(f"Tesseract data filtered by confidence resulted in empty DataFrame for '{os.path.basename(str(image_path_or_array)) if not is_image_array else 'image_array'}'")
                return []

            # Group words into blocks based on Tesseract's block_num
            grouped_by_block = data_df_filtered_conf.groupby(['page_num', 'block_num'])

            our_block_counter = 0 # Sequential ID for blocks we define on this page
            for (page_num_tess, block_num_tess), block_words_df in grouped_by_block:
                if block_words_df.empty: continue # Should not happen after dropna, but safety

                # Recalculate bounding box for the entire Tesseract-defined block
                # This is safer than relying on Tesseract's top-level block bbox if words are filtered
                # Ensure min/max are not from an empty set
                if block_words_df.empty: continue # Should be caught by outer empty check, but inner safety

                x1 = block_words_df['left'].min()
                y1 = block_words_df['top'].min()
                x2 = (block_words_df['left'] + block_words_df['width']).max()
                y2 = (block_words_df['top'] + block_words_df['height']).max()

                # Reconstruct text for the block using the cleaned 'text' column
                block_text_str = " ".join(block_words_df['text'].tolist()).strip() # .tolist() ensures it's a list of strings/empty strings
                # --- FIX END ---
                block_text_str = re.sub(r'\s+', ' ', block_text_str) # Normalize whitespace

                # Check for invalid box dimensions after min/max
                if x1 >= x2 or y1 >= y2:
                    logger.debug(f"Skipping Tesseract block (tess_pg:{page_num_tess}, tess_blk:{block_num_tess}) due to invalid bbox [{x1},{y1},{x2},{y2}].")
                    continue

                # Average confidence of words in this block
                # Calculate mean only on valid confidence values (> -1)
                valid_confs = block_words_df[block_words_df['conf'] > -1]['conf']
                avg_block_conf = valid_confs.mean() if not valid_confs.empty else -1.0 # Default to -1 if no valid conf values

                if avg_block_conf >= config_dict.get('min_block_avg_confidence_tess', 60):
                    text_blocks.append({
                        "box": [int(x1), int(y1), int(x2), int(y2)],
                        "confidence": round(float(avg_block_conf), 2),
                        "text": block_text_str,
                        "tesseract_page_num": int(page_num_tess) if pd.notna(page_num_tess) else -1,
                        "tesseract_block_num": int(block_num_tess) if pd.notna(block_num_tess) else -1,
                        "block_num": our_block_counter # Our own sequential ID for this page
                    })
                    our_block_counter += 1
                else:
                    logger.debug(f"Skipping Tesseract block (tess_pg:{page_num_tess}, tess_blk:{block_num_tess}) due to low avg conf ({avg_block_conf:.2f}). Text: '{block_text_str[:50]}...'")

            logger.info(f"Detected {len(text_blocks)} Tesseract text blocks passing confidence from '{os.path.basename(str(image_path_or_array)) if not is_image_array else 'image_array'}'")
            return text_blocks

        except pytesseract.TesseractNotFoundError:
            logger.error("Tesseract executable not found or not in PATH.", exc_info=True)
            raise # Re-raise this critical error
        except Exception as e:
            logger.error(f"Unexpected error in _get_text_blocks_tesseract for '{os.path.basename(str(image_path_or_array))if not is_image_array else 'image_array'}': {e}", exc_info=True)
            return []


def _get_image_regions_opencv(image_path: str, text_blocks: List[Dict], config_dict: Dict) -> List[Dict]:
    # ... (Implementation as you provided, using _calculate_overlap_ratio and config_dict) ...
    # Ensure this is robust and parameters are from config_dict
    # (Using the one you provided, looks okay)
    image_regions = []
    try:
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Could not read image for image region detection: {image_path}")
            return []
        
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = img_gray.shape
        
        # Example: Invert and threshold for dark regions on light background (common for diagrams)
        # This part highly depends on the nature of your diagrams.
        # You might need different strategies for light-on-dark vs dark-on-light.
        # thresh_img = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        _, thresh_img = cv2.threshold(img_gray, config_dict.get("image_region_otsu_thresh_val", 128), 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)


        # Morphological operations to connect components of diagrams
        kernel_size = config_dict.get("image_region_morph_kernel", (5,5)) # e.g., (5,5) or (7,7)
        kernel = np.ones(kernel_size, np.uint8)
        # Closing small gaps
        closed_thresh = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, kernel, iterations=config_dict.get("image_region_close_iter", 2))
        # Removing small noise
        opened_thresh = cv2.morphologyEx(closed_thresh, cv2.MORPH_OPEN, kernel, iterations=config_dict.get("image_region_open_iter", 1))
        
        contours, _ = cv2.findContours(opened_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        logger.debug(f"Found {len(contours)} initial contours for image regions in {os.path.basename(image_path)}.")
        
        min_area_abs = config_dict.get('image_detection_min_abs_area_px', 100*100) # Min 100x100 pixels for an image
        min_area_ratio_val = config_dict.get('image_detection_min_area_ratio', 0.01) * w * h
        min_final_area = max(min_area_abs, min_area_ratio_val)

        text_block_bounding_boxes = [tb["box"] for tb in text_blocks if tb.get("box")]
        max_text_overlap_ratio_val = config_dict.get('image_detection_max_text_overlap_ratio', 0.2) # Allow up to 20% text overlap

        detected_image_regions_list = []
        for contour_item in contours:
            contour_item_area = cv2.contourArea(contour_item)
            if contour_item_area >= min_final_area:
                x_c, y_c, w_c, h_c = cv2.boundingRect(contour_item)
                # Filter by aspect ratio to avoid very thin lines being detected as images
                aspect_ratio = max(w_c, h_c) / max(1, min(w_c, h_c))
                if aspect_ratio > config_dict.get("image_region_max_aspect", 15): # Max aspect ratio e.g. 15:1
                    logger.debug(f"Filtering contour by aspect ratio {aspect_ratio:.1f}: [{x_c},{y_c},{w_c},{h_c}]")
                    continue

                contour_item_bbox = [x_c, y_c, x_c + w_c, y_c + h_c]
                
                is_predominantly_text = False
                for tb_box_coords in text_block_bounding_boxes:
                    overlap_r = _calculate_overlap_ratio(contour_item_bbox, tb_box_coords) # intersection / area_of_contour_box
                    if overlap_r > max_text_overlap_ratio_val:
                        is_predominantly_text = True
                        break 
                
                if not is_predominantly_text:
                    detected_image_regions_list.append({"box": [int(c_val) for c_val in contour_item_bbox]})
        
        logger.info(f"Detected {len(detected_image_regions_list)} image/diagram regions (passed filters) in {os.path.basename(image_path)}.")
        return detected_image_regions_list
    except Exception as e_cv_img_regions:
        logger.error(f"Error in OpenCV image region detection for {os.path.basename(image_path)}: {e_cv_img_regions}", exc_info=True)
        return []

def _ocr_text_block(image_array: np.ndarray, block_box: List[int], config_dict: Dict, margin: int) -> Optional[str]: # Pass image_array
    # ... (Your existing _ocr_text_block code)
    # Ensure it uses `logger` and `config_dict` for TESSERACT_LANG, TESSERACT_OCR_PSM.
    # `margin` is already passed.
    try:
        h_img, w_img = image_array.shape[:2] # Get dimensions from the passed image array
        x_min, y_min, x_max, y_max = block_box
        
        # Apply margin, ensuring bounds are within image dimensions
        crop_x_min = max(0, x_min - margin)
        crop_y_min = max(0, y_min - margin)
        crop_x_max = min(w_img, x_max + margin)
        crop_y_max = min(h_img, y_max + margin)

        if crop_y_max <= crop_y_min or crop_x_max <= crop_x_min:
             logger.warning(f"Invalid crop dimensions for block {block_box} after margin, skipping OCR. Image dims: {w_img}x{h_img}")
             return None
        
        cropped_image_for_ocr = image_array[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
        
        if cropped_image_for_ocr.size == 0:
             logger.warning(f"Cropped image for block {block_box} is empty, skipping OCR.")
             return None
        
        # Tesseract config string
        psm_val = config_dict.get("tesseract_ocr_psm", 6)
        lang_val = config_dict.get("tesseract_lang", "eng")
        tess_config_str = f'--psm {psm_val} -l {lang_val}'
        
        # Perform OCR
        text = pytesseract.image_to_string(cropped_image_for_ocr, config=tess_config_str, timeout=config_dict.get('tesseract_timeout_block', 30)).strip()
        
        if not text:
            # Optional: Fallback OCR attempt with different PSM if text is empty and confidence was low
            # This can be useful if initial PSM choice was poor for certain block types.
            # For now, keeping it simple.
            logger.debug(f"OCR yielded empty text for block {block_box} with PSM {psm_val}.")
            return None
            
        logger.debug(f"OCR successful for block {block_box} (PSM {psm_val}). Preview: '{text[:50]}...'")
        return text
    except pytesseract.TesseractError as tess_err: # More specific Tesseract errors
        logger.error(f"Tesseract OCR runtime error for box {block_box}: {tess_err}", exc_info=False) # exc_info=False to reduce noise for common tess errors
        return None
    except Exception as e:
        logger.error(f"Unexpected error during block OCR for box {block_box}: {e}", exc_info=True)
        return None


def _extract_and_save_image_region(full_page_image_path: str, region_box: List[int], output_path: str, config_dict: Dict) -> bool:
        margin = config_dict.get("image_crop_margin", 5)
        try:
            # Read image, keeping channels and potentially alpha
            page_image = cv2.imread(full_page_image_path, cv2.IMREAD_UNCHANGED)
            if page_image is None:
                logger.error(f"Failed to load page image for cropping region: {full_page_image_path}")
                return False

            h_img, w_img = page_image.shape[:2]
            x_min, y_min, x_max, y_max = region_box

            # Validate region bounding box
            if x_min >= x_max or y_min >= y_max or x_max < 0 or y_max < 0 or x_min > w_img or y_min > h_img or x_min < 0 or y_min < 0: # Added explicit check for negative coords
                 logger.warning(f"Invalid/out-of-bounds region bounding box {region_box}. Image dims: {w_img}x{h_img}. Skipping extraction.")
                 return False

            # Apply margin, ensuring bounds are within image dimensions
            crop_x_min = max(0, x_min - margin)
            crop_y_min = max(0, y_min - margin)
            crop_x_max = min(w_img, x_max + margin)
            crop_y_max = min(h_img, y_max + margin)

            if crop_y_max <= crop_y_min or crop_x_max <= crop_x_min:
                logger.warning(f"Invalid crop dimensions [{crop_x_min}:{crop_x_max}, {crop_y_min}:{crop_y_max}] for image region {region_box} in {os.path.basename(full_page_image_path)}, skipping save.")
                return False

            # Original BUG: cropped_region_img = page_image[crop_y_min:crop_y_max, crop_x_min:crop_y_max]
            # Corrected slicing:
            cropped_region_img = page_image[crop_y_min:crop_y_max, crop_x_min:crop_x_max]

            if cropped_region_img.size == 0:
                logger.warning(f"Cropped image region for {region_box} in {os.path.basename(full_page_image_path)} is empty ({cropped_region_img.shape}). Skipping save.")
                return False

            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Save with quality parameters if JPEG, or compression if PNG
            img_format = os.path.splitext(output_path)[1].lower()
            save_params = []
            if img_format in ['.jpg', '.jpeg']:
                # Check if cropped image has alpha channel before saving as JPEG (JPEG doesn't support alpha)
                if len(cropped_region_img.shape) == 3 and cropped_region_img.shape[2] == 4:
                     # Convert BGRA to BGR if it has alpha
                     cropped_region_img = cv2.cvtColor(cropped_region_img, cv2.COLOR_BGRA2BGR)
                save_params = [cv2.IMWRITE_JPEG_QUALITY, config_dict.get("jpeg_quality", 90)]
            elif img_format == '.png':
                save_params = [cv2.IMWRITE_PNG_COMPRESSION, config_dict.get("png_compression", 3)] # 0-9, 3 is default
            # Add support for other formats if needed, e.g., TIFF for lossless

            if cv2.imwrite(output_path, cropped_region_img, save_params):
                logger.debug(f"Saved cropped image region to: {output_path}")
                return True
            else:
                logger.error(f"Failed to save cropped image region to: {output_path}")
                return False
        except Exception as e:
            logger.error(f"Error extracting/saving image region {region_box} from {os.path.basename(full_page_image_path)}: {e}", exc_info=True)
            return False

# --- MAIN VISUAL PIPELINE FUNCTIONS ---

async def perform_layout_analysis(doc_id: str, config_dict: Dict, common_utils_module: Any) -> Optional[str]:
    """
    Performs layout analysis on processed page images of a document using YOLOv8 (segmentation).

    Args:
        doc_id: The document ID being processed.
        config_dict: The configuration dictionary.
        common_utils_module: Reference to the common_utils module.

    Returns:
        The path to the saved layout analysis JSON file if successful, None otherwise.
    """
    logger.info(f"Starting layout analysis (visual pipeline Step 3) for document ID: {doc_id} using YOLOv8 segmentation")

    # Define file paths based on config_dict
    processed_doc_base_path = os.path.join(config_dict["vault_directory"], config_dict["processed_docs_subdir"], doc_id)
    page_image_dir = os.path.join(processed_doc_base_path, "pages")
    output_json_path = os.path.join(processed_doc_base_path, config_dict["layout_analysis_file"])

    # Check if the directory containing page images exists
    if not os.path.isdir(page_image_dir):
        logger.error(f"Page image directory not found for layout analysis: {page_image_dir} (Doc ID: {doc_id})")
        try:
            common_utils_module.update_file_metadata(config_dict, doc_id, metadata_extra={"pipeline_step": "layout_analysis_failed_no_images"})
        except Exception as e_meta_update:
            logger.error(f"Failed to update metadata for {doc_id} after layout analysis failure (no images): {e_meta_update}", exc_info=True)
        return None

    # Check if layout analysis already completed based on the output file existence
    if os.path.exists(output_json_path):
        logger.info(f"Layout analysis result already exists for {doc_id} at {output_json_path}. Skipping.")
        return output_json_path

    # --- Start Main Try Block for Layout Analysis Process ---
    try:
        # Get list of page image files
        page_files = sorted([f for f in os.listdir(page_image_dir) if f.lower().endswith(f".{config_dict.get('image_format', 'png')}")])
        if not page_files:
            logger.warning(f"No processed page images (e.g., *.png) found in {page_image_dir} for {doc_id}. Cannot perform layout analysis.")
            try:
                common_utils_module.update_file_metadata(config_dict, doc_id, metadata_extra={"pipeline_step": "layout_analysis_failed_no_images"})
            except Exception as e_meta_update:
                logger.error(f"Failed to update metadata for {doc_id} after layout analysis failure (no images): {e_meta_update}", exc_info=True)
            return None

        analysis_results_dict = {} # Dictionary to store analysis results for each page

        # --- Initialize YOLOv8 Layout Model ---
        logger.info("Initializing YOLOv8 layout model...")
        layout_model = None # Initialize model variable
        try:
            # Get model path from config_dict
            yolo_model_path = config_dict.get('yolov8_layout_model_path') # Use the new key
            if not yolo_model_path or not os.path.exists(yolo_model_path):
                 error_msg = f"YOLOv8 layout model path not configured or file not found: {yolo_model_path}"
                 logger.error(error_msg)
                 try:
                     common_utils_module.update_file_metadata(config_dict, doc_id, metadata_extra={"pipeline_step": "layout_analysis_failed_yolo_model_missing"})
                 except Exception as e_meta_update:
                     logger.error(f"Failed to update metadata for {doc_id} after YOLO model missing error: {e_meta_update}", exc_info=True)
                 return None

            # Instantiate YOLO model - it handles device selection implicitly but can be controlled
            # Pass the general device setting from CONFIG for explicit control
            yolo_device = config_dict.get('device', 'cpu')
            layout_model = YOLO(yolo_model_path) # Instantiate YOLO model
            layout_model.to(yolo_device) # Explicitly move model to device
            logger.info(f"YOLOv8 layout model loaded from: {yolo_model_path}, moved to device: {yolo_device}")

            # Get label map and score threshold from config
            yolo_label_map = config_dict.get('yolov8_label_map', {0: "text", 1: "title", 2: "list", 3:"table", 4:"figure"})
            yolo_score_threshold = config_dict.get('yolov8_score_threshold', 0.5)

        except Exception as e_yolo_init:
             logger.error(f"Failed to initialize YOLOv8 layout model: {e_yolo_init}", exc_info=True)
             try:
                 common_utils_module.update_file_metadata(config_dict, doc_id, metadata_extra={"pipeline_step": f"layout_analysis_failed_yolo_init_{e_yolo_init.__class__.__name__}"})
             except Exception as e_meta_update:
                 logger.error(f"Failed to update metadata for {doc_id} after YOLO init failure: {e_meta_update}", exc_info=True)
             return None

        # Ensure model was successfully initialized before proceeding
        if layout_model is None:
             logger.error("YOLOv8 layout model initialization failed, cannot proceed with layout analysis.")
             # Metadata update should have been handled in the except block
             return None
        # --- END YOLOv8 Model Initialization ---


        # --- Inner Function for analyzing a single page ---
        # This function is updated to use YOLOv8 predict and process its output
        # Removed LayoutParser type hint since we are no longer using LP objects directly
        async def _analyze_single_page_layout(page_filename_param: str, layout_model_instance: YOLO, yolo_label_map: Dict[int, str], yolo_score_threshold: float):
            """
            Analyzes layout for a single page image using YOLOv8.
            """
            page_num_match_obj = re.search(r'page_(\d+)', page_filename_param)
            if not page_num_match_obj:
                logger.warning(f"Could not parse page number from filename '{page_filename_param}', skipping layout analysis for this page.")
                page_key_for_results = f"page_unknown_{uuid.uuid4().hex[:4]}"
                analysis_results_dict[page_key_for_results] = {
                     "status": "skipped_filename_parse_error",
                     "source_image": page_filename_param,
                     "regions": [] # Store detected regions here
                }
                return None

            page_num_from_filename = int(page_num_match_obj.group(1))
            page_key_for_results = f"page_{page_num_from_filename}"
            full_image_path_for_analysis = os.path.join(page_image_dir, page_filename_param)

            logger.info(f"Analyzing layout of {page_filename_param} (Page key: {page_key_for_results}) using YOLOv8...")

            # Initialize list for detected regions for this page
            detected_regions_yolo = []
            # page_width, page_height will be obtained from image later


            try:
                # Read original image (color/unchanged for YOLOv8 prediction and visualization)
                # YOLOv8 predict can take file path or numpy array. Passing the path is often simplest.
                # Read with cv2 first to get dimensions and for visualization later
                page_image_cv2_original = cv2.imread(full_image_path_for_analysis, cv2.IMREAD_UNCHANGED)
                if page_image_cv2_original is None:
                     logger.error(f"Failed to read image for YOLOv8 prediction: {full_image_path_for_analysis}. Skipping page layout.")
                     analysis_results_dict[page_key_for_results] = {
                         "status": "skipped_image_read_error",
                         "source_image": page_filename_param,
                         "regions": []
                     }
                     return None

                # Get dimensions from the original image (needed for clamping boxes)
                if len(page_image_cv2_original.shape) == 3:
                    page_height, page_width, _ = page_image_cv2_original.shape
                else: # Grayscale or single channel
                    page_height, page_width = page_image_cv2_original.shape[:2]

                # --- Perform Layout Analysis using YOLOv8 ---
                # YOLOv8 predict is blocking, use asyncio.to_thread
                # Pass the image as a numpy array (YOLO can handle BGR from cv2) or path
                # Passing the numpy array:
                yolo_results_list = await asyncio.to_thread(
                    layout_model_instance.predict,
                    page_image_cv2_original, # Pass image as numpy array (cv2 reads BGR)
                    conf=yolo_score_threshold, # Use configured score threshold
                    # iou=0.45, # Optional: IoU threshold for NMS
                    # classes=None, # Optional: List of class IDs to filter
                    verbose=False # Suppress verbose YOLO output for each prediction
                )

                # Process detected regions from YOLOv8 results
                if yolo_results_list: # predict returns a list of Results objects (one per image)
                     yolo_results = yolo_results_list[0] # Get the Results object for our single image

                     if hasattr(yolo_results, 'boxes') and yolo_results.boxes is not None:
                          # yolo_results.boxes contains the detection bounding boxes, classes, and scores
                          # It's a tensor, convert to numpy for iteration
                          # xyxy format is (x1, y1, x2, y2)
                          boxes = yolo_results.boxes.xyxy.cpu().numpy()
                          classes = yolo_results.boxes.cls.cpu().numpy() # Class IDs are integers
                          scores = yolo_results.boxes.conf.cpu().numpy() # Scores are floats

                          # Optionally, get masks if needed later (though not used in current OCR/chunking logic)
                          # if hasattr(yolo_results, 'masks') and yolo_results.masks is not None:
                          #     masks = yolo_results.masks.data.cpu().numpy() # Boolean masks [N, H, W]

                          for i_det in range(len(boxes)):
                               box = boxes[i_det].tolist() # Convert numpy array box [x1, y1, x2, y2] to list
                               class_id = int(classes[i_det])
                               score = float(scores[i_det])

                               # Map class ID to category name using your configured map
                               category = yolo_label_map.get(class_id, f"unknown_class_{class_id}")

                               # Ensure box coordinates are within image bounds before storing (YOLO might predict slightly outside)
                               x1, y1, x2, y2 = [int(c) for c in box]
                               x1 = max(0, x1)
                               y1 = max(0, y1)
                               x2 = min(page_width, x2)
                               y2 = min(page_height, y2)

                               # Basic validation after clamping
                               if x2 <= x1 or y2 <= y1:
                                   logger.warning(f"Skipping invalid/zero-size YOLO region {category} after clamping on page {page_key_for_results}: [{x1},{y1},{x2},{y2}]")
                                   continue

                               detected_regions_yolo.append({
                                   "box": [x1, y1, x2, y2],
                                   "category": category,
                                   "score": round(score, 4),
                                   "layout_source": "YOLOv8", # Indicate source of detection
                                   "page_key": page_key_for_results # Add page key for easier lookup later
                                   # Add mask data here if needed later: "mask": masks[i_det].tolist() if 'masks' in locals() else None
                               })
                     else:
                          logger.debug(f"YOLOv8 predict returned results object but no 'boxes' detected for {page_filename_param}. No regions detected.")

                logger.info(f"YOLOv8 detected {len(detected_regions_yolo)} regions on {page_filename_param} (Page key: {page_key_for_results}).")


                # --- DEBUG Layout Visualization (Adapted for YOLOv8 results) ---
                # This part draws the boxes detected by YOLOv8
                if config_dict.get("debug_layout_visualization", False) and page_image_cv2_original is not None:
                    try:
                       debug_layout_dir = os.path.join(processed_doc_base_path, config_dict.get("debug_layout_output_subdir", "debug_layout"))
                       os.makedirs(debug_layout_dir, exist_ok=True)
                       debug_output_filename = f"layout_debug_{doc_id}_{page_key_for_results}.{config_dict.get('image_format', 'png')}"
                       debug_output_path = os.path.join(debug_layout_dir, debug_output_filename)

                       # Ensure image is in BGR color format for drawing (cv2 drawing expects BGR)
                       if len(page_image_cv2_original.shape) == 2: # Grayscale
                           page_image_display = cv2.cvtColor(page_image_cv2_original, cv2.COLOR_GRAY2BGR)
                       elif len(page_image_cv2_original.shape) == 3 and page_image_cv2_original.shape[2] == 4: # BGRA
                           page_image_display = cv2.cvtColor(page_image_cv2_original, cv2.COLOR_BGRA2BGR)
                       else: # Already BGR
                           page_image_display = page_image_cv2_original.copy() # Work on a copy

                       h_disp, w_disp = page_image_display.shape[:2]

                       # Define colors for different categories (example) - Can be moved to config
                       # Using distinct colors for clarity in debug visualization
                       category_colors = {
                           "text": (0, 0, 255),    # Red
                           "title": (0, 255, 0),   # Green
                           "list": (255, 0, 0),    # Blue
                           "table": (255, 255, 0), # Cyan
                           "figure": (255, 0, 255) # Magenta
                           # Add more as per your model's labels
                       }
                       default_color = (128, 128, 128) # Gray for unknown categories

                       # Sort regions by vertical position for consistent drawing/labeling order
                       sorted_regions_for_viz = sorted(detected_regions_yolo, key=lambda r: r.get('box', [0,0,0,0])[1])

                       for i_viz, region_data in enumerate(sorted_regions_for_viz): # Iterate over sorted YOLOv8 results
                           region_box = region_data.get("box")
                           region_category = region_data.get("category", "unknown")
                           region_score = region_data.get("score", -1.0)

                           if region_box and len(region_box) == 4:
                               x1, y1, x2, y2 = [int(c) for c in region_box]

                               # Basic validation (should be okay after clamping, but safety)
                               if x2 <= x1 or y2 <= y1 or x1 >= w_disp or y1 >= h_disp or x2 <= 0 or y2 <= 0:
                                    logger.warning(f"Skipping drawing invalid/out-of-bounds box {region_box} for YOLO region '{region_category}' during visualization on page {page_key_for_results}.")
                                    continue

                               # Draw the rectangle
                               color = category_colors.get(region_category, default_color)
                               thickness = 2
                               cv2.rectangle(page_image_display, (x1, y1), (x2, y2), color, thickness)

                               # Add label (optional): Category Score
                               label_text = f"YOLO{i_viz} {region_category} {region_score:.2f}" # Include an index for clarity
                               font = cv2.FONT_HERSHEY_SIMPLEX
                               font_scale = 0.5
                               font_thickness = 1
                               text_color = (255 - color[0], 255 - color[1], 255 - color[2]) # Invert color for better visibility
                               bg_color = (255, 255, 255) # White background

                               # Get text size
                               (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)

                               # Determine text position (try inside top-left corner with padding)
                               text_x = x1 + 5
                               text_y = y1 + text_h + 5 # Small padding inside

                               # Ensure text doesn't go off edges
                               if text_x + text_w > w_disp: text_x = w_disp - text_w - 5
                               if text_y - text_h - baseline < 0: text_y = y2 + text_h + 5 # Fallback below box if no space inside top

                               text_x = max(0, text_x)
                               # Ensure y is below the text height from the top edge if placed above
                               text_y = max(text_h + baseline, text_y)
                               # If fallback below box, ensure it's within height bounds
                               if text_y > h_disp: text_y = h_disp - 5


                               # Draw background rectangle for text (slightly larger than text)
                               bg_x1 = text_x - 2
                               bg_y1 = text_y - text_h - baseline - 2
                               bg_x2 = text_x + text_w + 2
                               bg_y2 = text_y + baseline + 2

                               # Clamp background box to image bounds
                               bg_x1 = max(0, bg_x1)
                               bg_y1 = max(0, bg_y1)
                               bg_x2 = min(w_disp, bg_x2)
                               bg_y2 = min(h_disp, bg_y2)

                               if bg_x2 > bg_x1 and bg_y2 > bg_y1:
                                   cv2.rectangle(page_image_display, (bg_x1, bg_y1), (bg_x2, bg_y2), bg_color, -1) # White background filled rectangle
                                   cv2.putText(page_image_display, label_text, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
                               else:
                                   logger.warning(f"Skipping drawing label background/text for YOLO region '{region_category}' during visualization on page {page_key_for_results} due to invalid drawing box coordinates or size.")

                       # Save the debug image
                       save_success = cv2.imwrite(debug_output_path, page_image_display)
                       if save_success:
                           logger.debug(f"Saved YOLOv8 debug image for page {page_key_for_results} to {debug_output_path}")
                       else:
                           logger.error(f"Failed to save YOLOv8 debug image for page {page_key_for_results} to {debug_output_path}")

                    except Exception as e_debug_viz:
                        logger.error(f"Error during YOLOv8 visualization for page {page_key_for_results}: {e_debug_viz}", exc_info=True)
                # --- END DEBUG Layout Visualization ---


                # Store YOLOv8 results in the analysis_results_dict
                # The subsequent OCR step will read this and process the regions.
                analysis_results_dict[page_key_for_results] = {
                    "regions": detected_regions_yolo, # Store all detected regions from YOLOv8
                    "source_image": page_filename_param,
                    "page_dimensions": {"width": page_width, "height": page_height},
                    # Status indicates if YOLO ran successfully and found regions
                    "status": "success" if detected_regions_yolo else "yolov8_no_regions_detected", # <--- Update status string
                    "analyzed_at": datetime.now().isoformat()
                }
                logger.info(f"Completed YOLOv8 analysis for {page_filename_param}. Status: {analysis_results_dict[page_key_for_results]['status']}")

                return page_key_for_results # Indicate success (YOLO ran) for this page


            except Exception as page_analysis_e:
                logger.error(f"Failed to analyze layout using YOLOv8 for {page_filename_param}: {page_analysis_e}", exc_info=True)
                # Ensure analysis_results_dict has an entry for this page key even on failure
                analysis_results_dict[page_key_for_results] = {
                     "error": f"Layout analysis failed (YOLOv8): {str(page_analysis_e)}", # <--- Update error message
                     # Include partial results if available (might be empty)
                     "regions": detected_regions_yolo if 'detected_regions_yolo' in locals() else [], # Use YOLO regions
                     "source_image": page_filename_param, "status": "failed",
                     "analyzed_at": datetime.now().isoformat()
                }
                return None # Indicate failure

        # --- End of Inner Function _analyze_single_page_layout ---


        # Run analysis for all pages concurrently
        logger.info(f"Starting concurrent layout analysis for {len(page_files)} pages...")
        # Pass the initialized layout_model instance (YOLO model) and the relevant config to each task
        page_analysis_tasks = [_analyze_single_page_layout(pf, layout_model, yolo_label_map, yolo_score_threshold) for pf in page_files] # <--- Pass needed config
        # Use return_exceptions=True to ensure all tasks are awaited even if some fail
        page_results = await asyncio.gather(*page_analysis_tasks, return_exceptions=True)

        # Log any exceptions caught by gather
        for i, res in enumerate(page_results):
             if isinstance(res, Exception):
                 page_filename = page_files[i] if i < len(page_files) else "Unknown file"
                 logger.error(f"Exception returned from _analyze_single_page_layout task for {page_filename}: {res}", exc_info=False)
             # If res is None, it means the inner function handled and logged its error and returned None.

        logger.info(f"Finished concurrent layout analysis for {doc_id}. Total pages processed: {len(page_files)}, Successful page tasks: {len([r for r in page_results if r is not None and not isinstance(r, Exception)])}.")

        # Sort final results by page number (derived from page_key)
        # Use keys directly from analysis_results_dict to ensure we include pages that failed/skipped within the inner function
        sorted_page_keys_final = sorted(analysis_results_dict.keys(),
                                       key=lambda pk_str: int(pk_str.split('_')[1]) if pk_str.startswith('page_') and pk_str.split('_')[1].isdigit() else float('inf'))

        # Reconstruct the dictionary based on sorted keys
        final_layout_data_to_save = {key_s: analysis_results_dict[key_s] for key_s in sorted_page_keys_final}

        if not final_layout_data_to_save:
             logger.error(f"No layout analysis results generated for {doc_id} across all pages.")
             try:
                common_utils_module.update_file_metadata(config_dict, doc_id, metadata_extra={"pipeline_step": "layout_analysis_failed_no_page_results"})
             except Exception as e_meta_update:
                logger.error(f"Failed to update metadata for {doc_id} after layout analysis failure (no page results): {e_meta_update}", exc_info=True)
             return None

        # Ensure at least one page processed successfully (status != 'failed', 'skipped_...')
        # We consider 'success', 'yolov8_no_regions_detected' as "processed" statuses for this check
        # Check for 'failed' explicitly. Also check for skipped statuses caused by issues before LP detection.
        processed_pages_count = sum(1 for page_data in final_layout_data_to_save.values() if page_data.get("status") not in ["failed", "skipped_filename_parse_error", "skipped_image_read_error"])


        if processed_pages_count == 0:
             logger.error(f"No pages successfully processed during layout analysis for {doc_id}. All pages failed/skipped.")
             try:
                 common_utils_module.update_file_metadata(config_dict, doc_id, metadata_extra={"pipeline_step": "layout_analysis_failed_all_pages_failed"})
             except Exception as e_meta_update:
                logger.error(f"Failed to update metadata for {doc_id} after layout analysis failure (all pages failed): {e_meta_update}", exc_info=True)
             return None


        # Save final layout analysis JSON file
        try:
            os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
            with open(output_json_path, 'w', encoding='utf-8') as f_out:
                json.dump(final_layout_data_to_save, f_out, indent=2)
            logger.info(f"Layout analysis results saved to: {output_json_path} for doc_id {doc_id}")

            # The caller (handle_file_selection) updates the main document metadata based on this function's return
            return output_json_path # Return the path on success

        except Exception as save_e:
             logger.error(f"Error saving layout analysis JSON for {doc_id} to {output_json_path}: {save_e}", exc_info=True)
             try:
                common_utils_module.update_file_metadata(config_dict, doc_id, metadata_extra={"pipeline_step": "layout_analysis_failed_save_error"})
             except Exception as e_meta_update:
                logger.error(f"Failed to update metadata for {doc_id} after layout analysis save failure: {e_meta_update}", exc_info=True)
             return None

    # --- End Main Try Block ---
    except Exception as e_layout_main:
        logger.error(f"An unexpected error occurred during layout analysis for {doc_id}: {e_layout_main}", exc_info=True)
        try:
            # Attempt to update metadata with a generic failure status if it wasn't already set
            meta_after_fail = common_utils_module.get_specific_doc_metadata(config_dict, doc_id) or {}
            if not meta_after_fail.get("pipeline_step", "").startswith("layout_analysis_failed"):
                 common_utils_module.update_file_metadata(config_dict, doc_id, metadata_extra={"pipeline_step": f"layout_analysis_failed_main_{e_layout_main.__class__.__name__}"})
            else:
                 logger.debug(f"Metadata already updated for layout failure of {doc_id}. Current step: {meta_after_fail.get('pipeline_step')}")
        except Exception as e_meta_update:
            logger.error(f"Failed to update metadata for {doc_id} after main layout analysis failure: {e_meta_update}", exc_info=True)
        return None
    
async def perform_ocr_extraction(
        doc_id: str,
        config_dict: Dict,
        common_utils_module: Any
    ) -> Optional[str]:
        """
        Performs OCR text extraction and image region saving based on YOLOv8 analysis results.
        """
        logger.info(f"Starting OCR & Image Extraction (visual pipeline Step 4) for document ID: {doc_id}")

        # Define file paths based on config_dict
        processed_doc_base_path = os.path.join(config_dict["vault_directory"], config_dict["processed_docs_subdir"], doc_id)
        layout_json_path = os.path.join(processed_doc_base_path, config_dict["layout_analysis_file"])
        page_image_dir = os.path.join(processed_doc_base_path, "pages") # Where page_X.png files are
        output_ocr_json_path = os.path.join(processed_doc_base_path, config_dict["ocr_results_file"])
        output_extracted_images_dir = os.path.join(processed_doc_base_path, config_dict.get("extracted_images_subdir", "extracted_images"))

        # --- Check Prerequisites ---
        if not os.path.exists(layout_json_path):
            logger.error(f"Layout analysis file not found for OCR step: {layout_json_path} (Doc ID: {doc_id}). Run layout analysis first.")
            try:
                common_utils_module.update_file_metadata(config_dict, doc_id, metadata_extra={"pipeline_step": "ocr_extraction_failed_no_layout"})
            except Exception as e_meta_update:
                logger.error(f"Failed to update metadata for {doc_id} after OCR failure (no layout): {e_meta_update}", exc_info=True)
            return None

        if os.path.exists(output_ocr_json_path):
             logger.info(f"OCR results already exist for {doc_id} at {output_ocr_json_path}. Skipping.")
             return output_ocr_json_path

        # --- Start Main Try Block for OCR Extraction Process ---
        try:
            # Load layout analysis data (format with 'regions' from YOLOv8)
            try:
                with open(layout_json_path, 'r', encoding='utf-8') as f_layout:
                    layout_data_per_page = json.load(f_layout)
                # The expected structure is { "page_1": { "regions": [...] }, "page_2": {...} }
                if not isinstance(layout_data_per_page, dict) or not layout_data_per_page:
                    logger.error(f"Layout data in {layout_json_path} is empty or invalid for doc_id {doc_id}.")
                    try:
                        common_utils_module.update_file_metadata(config_dict, doc_id, metadata_extra={"pipeline_step": "ocr_extraction_failed_invalid_layout_data"})
                    except Exception as e_meta_update:
                        logger.error(f"Failed to update metadata for {doc_id} after OCR failure (invalid layout): {e_meta_update}", exc_info=True)
                    return None
            except Exception as e_load_layout:
                logger.error(f"Failed to load/parse layout data from {layout_json_path} for doc_id {doc_id}: {e_load_layout}", exc_info=True)
                try:
                    common_utils_module.update_file_metadata(config_dict, doc_id, metadata_extra={"pipeline_step": "ocr_extraction_failed_layout_read_error"})
                except Exception as e_meta_update:
                    logger.error(f"Failed to update metadata for {doc_id} after OCR failure (layout read error): {e_meta_update}", exc_info=True)
                return None


            ocr_results_final = {"doc_id": doc_id, "pages": {}}
            os.makedirs(output_extracted_images_dir, exist_ok=True)

            logger.info(f"Performing OCR and image extraction based on layout results for {len(layout_data_per_page)} pages for {doc_id}...")
            # Sort page keys numerically
            page_keys_from_layout = sorted(layout_data_per_page.keys(), key=lambda page_k_str: int(page_k_str.split('_')[1]) if page_k_str.startswith('page_') and page_k_str.split('_')[1].isdigit() else float('inf'))

            # Define which categories are considered text for OCR (using categories from YOLOv8_label_map)
            text_categories = config_dict.get("layoutparser_text_categories", ["text", "title", "list"]) # Keep the same config key, but it refers to YOLO labels now
            image_categories = config_dict.get("layoutparser_image_categories", ["figure"])
            table_categories = config_dict.get("layoutparser_table_categories", ["table"])


            for page_key_val in page_keys_from_layout:
                page_layout_info = layout_data_per_page.get(page_key_val)
                page_extracted_items_list = [] # Combine text and image results here

                ocr_results_final["pages"][page_key_val] = {
                    "status": "processing", # Initial status for this page
                    "extracted_items": page_extracted_items_list, # List reference
                    "source_image": page_layout_info.get("source_image", "unknown_source_image_for_ocr") if page_layout_info else "unknown_source_image_for_ocr",
                    "page_dimensions": page_layout_info.get("page_dimensions", {"width": config_dict.get("fallback_page_width", 1000), "height": config_dict.get("fallback_page_height", 1400)}) if page_layout_info else {"width": config_dict.get("fallback_page_width", 1000), "height": config_dict.get("fallback_page_height", 1400)}
                }

                # Check layout analysis status for this page before processing its regions
                # Check for both LayoutParser and YOLOv8 no regions statuses
                if not page_layout_info or page_layout_info.get("status") in ["failed", "skipped_filename_parse_error", "skipped_image_read_error", "skipped_preprocessing_failed", "layout_parser_no_regions_detected", "skipped_layout_error", "skipped_no_regions_detected", "skipped_image_missing", "skipped_image_load_error", "yolov8_no_regions_detected"]:
                    logger.warning(f"Skipping OCR for page {page_key_val} (Doc ID: {doc_id}) due to layout analysis status: {page_layout_info.get('status', 'N/A') if page_layout_info else 'Missing Info'}")
                    ocr_results_final["pages"][page_key_val]["status"] = "skipped_layout_error" # Set specific skip status
                    continue
                
                # Get the list of regions detected by YOLOv8 from the layout data
                detected_regions_for_page = page_layout_info.get("regions", [])
                if not detected_regions_for_page:
                    logger.warning(f"No regions found in layout data for page {page_key_val}, doc {doc_id}. Skipping OCR.")
                    ocr_results_final["pages"][page_key_val]["status"] = "skipped_no_regions_found_in_layout" # More specific skip status
                    continue


                source_image_filename_on_disk = page_layout_info["source_image"]
                full_page_image_path_on_disk = os.path.join(page_image_dir, source_image_filename_on_disk)

                # Check if the source page image file exists
                if not os.path.exists(full_page_image_path_on_disk):
                    logger.error(f"Source page image not found for OCR/Extraction: {full_page_image_path_on_disk} (Page: {page_key_val}, Doc ID: {doc_id})")
                    ocr_results_final["pages"][page_key_val]["status"] = "skipped_image_missing" # Set specific skip status
                    continue

                # Load page image (original channels) for cropping
                page_image_cv2_original = cv2.imread(full_page_image_path_on_disk, cv2.IMREAD_UNCHANGED)

                if page_image_cv2_original is None:
                    logger.error(f"Failed to load image for OCR/Extraction: {full_page_image_path_on_disk} (Page: {page_key_val}, Doc ID: {doc_id})")
                    ocr_results_final["pages"][page_key_val]["status"] = "skipped_image_load_error" # Set specific skip status
                    continue

                h_img, w_img = page_image_cv2_original.shape[:2]

                logger.info(f"Processing OCR/Extraction for {page_key_val} ('{source_image_filename_on_disk}')...")

                # Sort regions by vertical position for more natural processing order
                # This sorting is good practice before processing, regardless of source (LP or YOLO)
                sorted_regions_for_ocr = sorted(detected_regions_for_page, key=lambda r: r.get('box', [0,0,0,0])[1])


                async def _process_single_region(i_region: int, region_data: Dict):
                     """Processes a single detected region (OCR for text, save for image/table)."""
                     region_box = region_data.get("box")
                     region_category = region_data.get("category", "unknown")
                     region_score = region_data.get("score", -1.0)
                     layout_source = region_data.get("layout_source", "unknown") # Get layout source (YOLOv8 or LayoutParser)

                     # Check if the box is valid
                     if not region_box or len(region_box) != 4 or region_box[0] >= region_box[2] or region_box[1] >= region_box[3]:
                          logger.warning(f"Region {i_region} ({region_category}) on page {page_key_val} has invalid/missing 'box' data {region_box}. Skipping.")
                          return None

                     x_min, y_min, x_max, y_max = region_box

                     # --- Process based on Category ---
                     if region_category in text_categories:
                         # Process as Text Region (perform OCR on the crop)
                         # Use text crop margin
                         margin = config_dict.get("text_crop_margin", 2)
                         crop_x_min = max(0, x_min - margin)
                         crop_y_min = max(0, y_min - margin)
                         crop_x_max = min(w_img, x_max + margin)
                         crop_y_max = min(h_img, y_max + margin)

                         if crop_y_max <= crop_y_min or crop_x_max <= crop_x_min:
                              logger.warning(f"Invalid crop dimensions [{crop_x_min}:{crop_x_max}, {crop_y_min}:{crop_y_max}] for text region {i_region} ({region_category}) on page {page_key_val}, skipping OCR.")
                              return None

                         cropped_image_for_ocr = page_image_cv2_original[crop_y_min:crop_y_max, crop_x_min:crop_x_max]

                         if cropped_image_for_ocr.size == 0:
                              logger.warning(f"Cropped image for text region {i_region} ({region_category}) on page {page_key_val} is empty, skipping OCR.")
                              return None

                         # Need to preprocess the cropped image for OCR (e.g., grayscale, binarize)
                         # Pass config_dict to preprocess_image_opencv
                         processed_cropped_img = preprocess_image_opencv(cropped_image_for_ocr, config_dict)

                         if processed_cropped_img is None:
                             logger.warning(f"Preprocessing failed for text crop of region {i_region} ({region_category}) on page {page_key_val}. Skipping OCR.")
                             return None

                         # Tesseract config string for this specific crop
                         # Use PSM 6 (Assume a single block of text) or 7 (Single text line) for crops
                         # Config key for crop PSM is tesseract_ocr_crop_psm
                         crop_psm_val = config_dict.get("tesseract_ocr_crop_psm", 6)
                         lang_val = config_dict.get("tesseract_lang", "eng")
                         tess_config_str_crop = f'--psm {crop_psm_val} -l {lang_val}'
                         tess_timeout_block = config_dict.get('tesseract_timeout_block', 30)

                         try:
                             # Use asyncio.to_thread for blocking Tesseract image_to_string call
                             extracted_text_content = await asyncio.to_thread(
                                 pytesseract.image_to_string, processed_cropped_img, config=tess_config_str_crop, timeout=tess_timeout_block
                             )
                             extracted_text_content = extracted_text_content.strip()

                             if not extracted_text_content:
                                 logger.debug(f"OCR yielded empty text for region {i_region} ({region_category}) on page {page_key_val} with PSM {crop_psm_val}.")

                             # Return the extracted text item dictionary
                             return {
                                 "type": "text", # Indicates this is a text item
                                 "category": region_category, # Original category from layout tool
                                 "text": extracted_text_content,
                                 "box": region_box, # Store original layout box
                                 "score": region_score, # Store layout tool's confidence score
                                 "layout_source": layout_source, # Store which tool detected the layout
                                 "page_key": page_key_val, # Add page info
                                 "region_index_on_page": i_region # Index based on the sorted list of regions on this page
                             }
                         except pytesseract.TesseractError as tess_err:
                             logger.error(f"Tesseract OCR runtime error for region {i_region} ({region_category}) on page {page_key_val} (PSM {crop_psm_val}): {tess_err}", exc_info=False) # exc_info=False for common errors
                             return None # Indicate failure to get text
                         except Exception as e_ocr_crop:
                             logger.error(f"Unexpected error during OCR for region {i_region} ({region_category}) on page {page_key_val}: {e_ocr_crop}", exc_info=True)
                             return None # Indicate failure

                     elif region_category in image_categories or region_category in table_categories:
                         # Process as Image/Table Region (save the crop)
                         # Use image crop margin
                         image_save_margin = config_dict.get("image_crop_margin", 5)
                         # Recalculate crop box with image margin
                         img_crop_x_min = max(0, x_min - image_save_margin)
                         img_crop_y_min = max(0, y_min - image_save_margin)
                         img_crop_x_max = min(w_img, x_max + image_save_margin)
                         img_crop_y_max = min(h_img, y_max + image_save_margin)

                         if img_crop_y_max <= img_crop_y_min or img_crop_x_max <= img_crop_x_min:
                             logger.warning(f"Invalid crop dimensions for saving image region {i_region} ({region_category}) on page {page_key_val}. Skipping save.")
                             return None

                         cropped_image_to_save = page_image_cv2_original[img_crop_y_min:img_crop_y_max, img_crop_x_min:img_crop_x_max]

                         if cropped_image_to_save.size == 0:
                            logger.warning(f"Cropped image for saving region {i_region} ({region_category}) on page {page_key_val} is empty, skipping save.")
                            return None

                         # Generate a unique filename for the extracted image
                         # Use a safe version of page_key_val and category in the filename
                         sane_page_key_val = re.sub(r'\W+', '_', str(page_key_val))
                         sane_category = re.sub(r'\W+', '_', str(region_category))
                         extracted_img_filename = f"{doc_id}_{sane_page_key_val}_{sane_category}_{i_region+1}.{config_dict.get('image_format', 'png')}"
                         extracted_img_output_path = os.path.join(output_extracted_images_dir, extracted_img_filename)

                         # Call the helper function to extract and save the image region (or save directly)
                         # Saving directly using cv2.imwrite is fine and avoids another helper call
                         os.makedirs(os.path.dirname(extracted_img_output_path), exist_ok=True)
                         img_format = os.path.splitext(extracted_img_output_path)[1].lower()
                         save_params = []
                         if img_format in ['.jpg', '.jpeg']:
                              # Convert BGRA to BGR if it has alpha (JPEG doesn't support alpha)
                              if len(cropped_image_to_save.shape) == 3 and cropped_image_to_save.shape[2] == 4:
                                   cropped_image_to_save = cv2.cvtColor(cropped_image_to_save, cv2.COLOR_BGRA2BGR)
                              save_params = [cv2.IMWRITE_JPEG_QUALITY, config_dict.get("jpeg_quality", 90)]
                         elif img_format == '.png':
                             save_params = [cv2.IMWRITE_PNG_COMPRESSION, config_dict.get("png_compression", 3)]
                         # Add TIFF or other format parameters here if needed

                         try:
                             # Use asyncio.to_thread for blocking cv2.imwrite
                             saved_successfully = await asyncio.to_thread(cv2.imwrite, extracted_img_output_path, cropped_image_to_save, save_params)

                             if saved_successfully:
                                 logger.debug(f"Saved cropped region ({region_category}) to: {extracted_img_output_path}")
                                 # Store relative path for JSON
                                 relative_image_path = os.path.join(config_dict.get("extracted_images_subdir", "extracted_images"), extracted_img_filename)

                                 image_item_entry = {
                                     "type": "image" if region_category in image_categories else "table", # Type: image or table
                                     "category": region_category, # Original category from layout tool
                                     "filename": extracted_img_filename, # Filename of the saved crop
                                     "box": region_box, # Store original layout box
                                     "path": relative_image_path, # Store relative path
                                     "score": region_score, # Store layout tool's confidence score
                                     "layout_source": layout_source, # Store which tool detected the layout
                                     "page_key": page_key_val, # Add page info
                                     "region_index_on_page": i_region # Index based on the sorted list of regions on this page
                                 }

                                 # Optional: OCR on diagrams/images if configured (using diagram_ocr_psm)
                                 # This OCR is for text *within* the image/diagram
                                 if config_dict.get("perform_ocr_on_diagrams", False) and region_category in image_categories:
                                     # Load the newly saved (cropped) image for OCR
                                     cropped_diagram_img_cv2_gray = cv2.imread(extracted_img_output_path, cv2.IMREAD_GRAYSCALE)
                                     if cropped_diagram_img_cv2_gray is not None:
                                          diagram_crop_h, diagram_crop_w = cropped_diagram_img_cv2_gray.shape[:2]
                                          if diagram_crop_w > 0 and diagram_crop_h > 0:
                                               diagram_psm = config_dict.get("diagram_ocr_psm", 11) # Use config for diagram PSM
                                               lang_val_diagram = config_dict.get("tesseract_lang", "eng") # Use general language config
                                               try:
                                                   # Use asyncio.to_thread for blocking pytesseract call
                                                   diagram_text_content = await asyncio.to_thread(
                                                        pytesseract.image_to_string, cropped_diagram_img_cv2_gray,
                                                        config=f'--psm {diagram_psm} -l {lang_val_diagram}', # Use specific diagram config
                                                        timeout=config_dict.get('tesseract_timeout_block', 30) # Use block timeout
                                                   )
                                                   if diagram_text_content and diagram_text_content.strip():
                                                       image_item_entry["ocr_text"] = diagram_text_content.strip() # Add extracted text
                                                       logger.debug(f"OCR successful for diagram {extracted_img_filename}. Preview: '{image_item_entry['ocr_text'][:50]}...'")
                                               except pytesseract.TesseractError as e_ocr_diagram_tess:
                                                   logger.warning(f"Tesseract OCR runtime error for diagram {extracted_img_filename} (PSM {diagram_psm}): {e_ocr_diagram_tess}", exc_info=False)
                                               except Exception as e_ocr_diagram:
                                                    logger.error(f"Error during OCR for diagram {extracted_img_filename} (region {region_box}): {e_ocr_diagram}", exc_info=True)
                                          else:
                                               logger.warning(f"Cropped diagram image {extracted_img_filename} has zero dimensions for OCR.")
                                     else:
                                         logger.warning(f"Could not load cropped diagram {extracted_img_output_path} for OCR.")

                                 return image_item_entry # Return the image item dictionary
                             else:
                                  logger.error(f"Failed to save cropped region ({region_category}) to: {extracted_img_output_path}")
                                  return None # Indicate save failed

                         except Exception as e_extract_save:
                              logger.error(f"Error during extraction/saving of region {i_region} ({region_category}) on page {page_key_val}, doc {doc_id}: {e_extract_save}", exc_info=True)
                              return None
                     else:
                         # Handle other categories if necessary, or ignore them
                         logger.debug(f"Skipping region {i_region} with unhandled category '{region_category}' on page {page_key_val}.")
                         return None


                # Run processing for all sorted detected regions concurrently
                logger.info(f"Processing {len(sorted_regions_for_ocr)} regions on page {page_key_val} ('{source_image_filename_on_disk}')...")
                region_processing_tasks = [_process_single_region(i_r, region_data) for i_r, region_data in enumerate(sorted_regions_for_ocr)]
                # Use return_exceptions=True to ensure all tasks are awaited even if some fail
                results_from_tasks = await asyncio.gather(*region_processing_tasks, return_exceptions=True)

                # Add successfully processed items (text or image) to the list for this page
                page_extracted_items_list.extend([res for res in results_from_tasks if res is not None])
                logger.debug(f"Page {page_key_val} (Doc {doc_id}): Successfully processed/extracted {len(page_extracted_items_list)} items from {len(detected_regions_for_page)} detected regions.")


                # --- Update page status after processing all regions ---
                # A page is successful if we managed to extract *any* item (text or image)
                if page_extracted_items_list:
                    ocr_results_final["pages"][page_key_val]["status"] = "success"
                    logger.info(f"Page {page_key_val} of doc {doc_id}: OCR & Extraction successful. Extracted items: {len(page_extracted_items_list)}")
                else:
                    # If no items were extracted, it could be because no regions were detected or processing failed for all.
                    # Check if layout detection had found regions first.
                    initial_layout_status = page_layout_info.get('status', 'unknown')
                    if initial_layout_status in ["success", "layout_parser_no_regions_detected", "yolov8_no_regions_detected"]:
                         # Layout detected regions (or reported 0), but extraction failed for all.
                         ocr_results_final["pages"][page_key_val]["status"] = "no_content_extracted"
                         logger.warning(f"Page {page_key_val} of doc {doc_id}: Layout found regions (or reported 0), but no content was extracted.")
                    else:
                         # Layout itself failed or skipped for this page.
                         ocr_results_final["pages"][page_key_val]["status"] = "skipped_layout_error" # Re-use or define new skip status if needed
                         logger.warning(f"Page {page_key_val} of doc {doc_id}: Skipped extraction due to prior layout status: {initial_layout_status}")


            # --- Save final OCR results JSON ---
            # This saving logic remains the same, as it saves the ocr_results_final structure
            # which now contains pages with a list of 'extracted_items'.
            if not ocr_results_final.get("pages"):
                 logger.error(f"No OCR results structure created for {doc_id} across all pages.")
                 try:
                     common_utils_module.update_file_metadata(config_dict, doc_id, metadata_extra={"pipeline_step": "ocr_extraction_failed_no_page_structure"})
                 except Exception as e_meta_update:
                    logger.error(f"Failed to update metadata for {doc_id} after OCR failure (no page structure): {e_meta_update}", exc_info=True)
                 return None

            # Ensure at least one page has status 'success' or 'no_content_extracted' before considering the overall step semi-successful (enough to proceed to indexing)
            # Indexing can handle pages with no content if they are marked correctly.
            # Only outright 'failed' or 'skipped' pages should prevent proceeding to indexing.
            processable_pages_count = sum(1 for page_data in ocr_results_final["pages"].values() if page_data.get("status") not in ["failed", "skipped_layout_error", "skipped_no_regions_found_in_layout", "skipped_image_missing", "skipped_image_load_error"])

            if processable_pages_count == 0:
                 logger.error(f"No pages successfully processed or extracted content during OCR extraction for {doc_id}. All pages failed/skipped.")
                 try:
                     common_utils_module.update_file_metadata(config_dict, doc_id, metadata_extra={"pipeline_step": "ocr_extraction_failed_all_pages_failed"})
                 except Exception as e_meta_update:
                    logger.error(f"Failed to update metadata for {doc_id} after OCR failure (all pages failed): {e_meta_update}", exc_info=True)
                 return None

            try:
                os.makedirs(os.path.dirname(output_ocr_json_path), exist_ok=True)
                with open(output_ocr_json_path, 'w', encoding='utf-8') as f_ocr_out:
                    json.dump(ocr_results_final, f_ocr_out, indent=2)
                logger.info(f"OCR & Extraction results for doc_id {doc_id} saved to: {output_ocr_json_path}")

                # Return the path on success (even if some pages failed, as long as some were processable)
                # The overall pipeline step can be marked based on this return value.
                return output_ocr_json_path

            except Exception as e_save_ocr:
                logger.error(f"Failed to save OCR results JSON for {doc_id} to {output_ocr_json_path}: {e_save_ocr}", exc_info=True)
                try:
                    common_utils_module.update_file_metadata(config_dict, doc_id, metadata_extra={"pipeline_step": "ocr_extraction_failed_save_error"})
                except Exception as e_meta_update:
                    logger.error(f"Failed to update metadata for {doc_id} after OCR save failure: {e_meta_update}", exc_info=True)
                return None

        # --- End Main Try Block ---
        except Exception as e_ocr_main:
            logger.error(f"An unexpected error occurred during OCR extraction for {doc_id}: {e_ocr_main}", exc_info=True)
            # Note: ocr_results_final might not be defined here if error happens very early
            extracted_count = 0
            if 'ocr_results_final' in locals():
                 extracted_count = sum(len(p.get('extracted_items', [])) for p in ocr_results_final.get('pages', {}).values())

            try:
                common_utils_module.update_file_metadata(config_dict, doc_id, metadata_extra={
                    "pipeline_step": f"ocr_extraction_failed_main_{e_ocr_main.__class__.__name__}",
                    "extracted_item_count": extracted_count # Report items processed before failure
                    })
            except Exception as e_meta_update:
                logger.error(f"Failed to update metadata for {doc_id} after main OCR failure: {e_meta_update}", exc_info=True)
            return None

def group_lp_regions_into_chunks(
    # Input is now a list of dictionaries representing extracted text regions
    extracted_text_items: List[Dict],
    page_width: int,
    page_height: int,
    # --- Chunking Parameters (adapt names/meaning) ---
    # How close vertically two text regions need to be to be considered part of the same chunk
    chunk_vertical_proximity_threshold_px: int = 15, # Max vertical gap in pixels
    chunk_vertical_proximity_threshold_ratio_of_height: float = 0.5, # Max vertical gap as % of taller region height
    # How much horizontal overlap is needed to consider regions in the same column/flow
    chunk_horizontal_overlap_ratio: float = 0.5, # e.g., 50% overlap or alignment
    # How to handle different categories
    always_new_chunk_categories: List[str] = ["title", "caption"], # Categories that usually start a new chunk
    merge_categories: List[str] = ["text", "list"], # Categories that can be merged with adjacent similar categories
    # Filter small chunks after creation
    min_chunk_text_length: int = 10, # Minimum text length for a final chunk
    # Add page info parameters (will be constant for all items in the list)
    page_key: str = "page_unknown",
    doc_id: str = "unknown_doc"
) -> List[Dict]:
    if not extracted_text_items:
        return []

    logger.debug(f"Chunking {len(extracted_text_items)} extracted text items for {page_key} (Doc: {doc_id})...")

    # 1. Sort text items spatially (top-to-bottom, left-to-right)
    # Use 'box' y1 for primary sort, then x1
    sorted_items = sorted(extracted_text_items, key=lambda item: (item.get("box", [0,0,0,0])[1], item.get("box", [0,0,0,0])[0]))

    final_chunks: List[Dict] = []
    current_chunk_items: List[Dict] = []
    chunk_counter = 0 # Index chunks per page


    # Helper to calculate vertical gap between the bottom of the last item in the current chunk
    # and the top of the next item.
    def calculate_vertical_gap(last_item_box, next_item_box):
        if not last_item_box or not next_item_box or len(last_item_box) != 4 or len(next_item_box) != 4:
            return float('inf') # Invalid boxes
        return next_item_box[1] - last_item_box[3] # y_next_top - y_last_bottom

    # Helper to check vertical proximity based on thresholds
    def is_vertically_proximate(last_item, next_item):
         last_box = last_item.get("box")
         next_box = next_item.get("box")
         if not last_box or not next_box or len(last_box) != 4 or len(next_box) != 4:
             return False

         v_gap = calculate_vertical_gap(last_box, next_box)
         if v_gap < 0: return True # Overlap means proximity

         # Calculate height-based tolerance using the taller of the two items
         last_height = last_box[3] - last_box[1]
         next_height = next_box[3] - next_box[1]
         height_tolerance = max(last_height, next_height) * chunk_vertical_proximity_threshold_ratio_of_height

         # Check if gap is within absolute or height-based tolerance
         return v_gap <= chunk_vertical_proximity_threshold_px or v_gap <= height_tolerance


    # Helper to check horizontal alignment/overlap (e.g., in the same column)
    def is_horizontally_aligned(last_item, next_item):
        last_box = last_item.get("box")
        next_box = next_item.get("box")
        if not last_box or not next_box or len(last_box) != 4 or len(next_box) != 4:
            return False

        # Calculate horizontal overlap
        overlap_x1 = max(last_box[0], next_box[0])
        overlap_x2 = min(last_box[2], next_box[2])
        overlap_width = max(0, overlap_x2 - overlap_x1)

        # Check if overlap is significant compared to either box's width
        last_width = last_box[2] - last_box[0]
        next_width = next_box[2] - next_box[0]

        min_width = min(last_width, next_width) if min(last_width, next_width) > 0 else 1 # Avoid division by zero

        return overlap_width / min_width >= chunk_horizontal_overlap_ratio


    # 2. Iterate and group items into chunks
    for i, current_item in enumerate(sorted_items):
        if not current_item.get("text", "").strip():
             logger.debug(f"Skipping empty extracted text item at index {i} on {page_key}.")
             continue # Skip empty items

        # Start a new chunk if it's the first item, if it's a category that always starts a new chunk,
        # or if it's not sufficiently close/aligned with the end of the current chunk.
        if not current_chunk_items:
            current_chunk_items.append(current_item)
            logger.debug(f"Started new chunk with item {i} ({current_item.get('category')}).")
        else:
            last_item_in_chunk = current_chunk_items[-1]

            # Rule 1: Always start a new chunk for specific categories (e.g., titles, captions)
            if current_item.get("category") in always_new_chunk_categories:
                 logger.debug(f"Item {i} ({current_item.get('category')}) starts a new chunk (always_new_chunk_categories rule).")
                 # Finalize the current chunk before starting a new one
                 if current_chunk_items:
                     # Process and add the finished chunk
                     # Calculate combined box, join text, create metadata
                     combined_text = " ".join([item.get("text", "") for item in current_chunk_items]).strip()
                     combined_box = [
                         min(item.get("box", [0,0,0,0])[0] for item in current_chunk_items),
                         min(item.get("box", [0,0,0,0])[1] for item in current_chunk_items),
                         max(item.get("box", [0,0,0,0])[2] for item in current_chunk_items),
                         max(item.get("box", [0,0,0,0])[3] for item in current_chunk_items)
                     ]
                     # Determine representative category (e.g., category of the first item)
                     representative_category = current_chunk_items[0].get("category", "text")
                     source_regions_info = [{"box": item.get("box"), "category": item.get("category"), "index": item.get("region_index_on_page")} for item in current_chunk_items]


                     # Filter by minimum chunk length before adding
                     if len(combined_text) >= min_chunk_text_length:
                         final_chunks.append({
                            "text": combined_text,
                            "box": combined_box,
                            "page_key": page_key,
                            "page_chunk_index": chunk_counter, # Sequential index
                            "category": representative_category,
                            "source_regions": source_regions_info, # Link back to LP regions
                            "source_page_filename": sorted_items[0].get("source_image", "N/A") # Add source image if available in items
                         })
                         logger.debug(f"Finalized chunk {chunk_counter} (length {len(combined_text)}, category {representative_category}).")
                         chunk_counter += 1
                     else:
                         logger.debug(f"Filtered small chunk (length {len(combined_text)} < {min_chunk_text_length}). Category: {representative_category}.")

                 # Start the new chunk with the current item
                 current_chunk_items = [current_item]

            # Rule 2: Merge if vertically proximate AND horizontally aligned AND categories can be merged
            elif is_vertically_proximate(last_item_in_chunk, current_item) and \
                 is_horizontally_aligned(last_item_in_chunk, current_item) and \
                 last_item_in_chunk.get("category") in merge_categories and \
                 current_item.get("category") in merge_categories: # Only merge if both are in merge categories
                 # Add to current chunk
                 current_chunk_items.append(current_item)
                 # logger.debug(f"Merged item {i} ({current_item.get('category')}) into current chunk.")

            else:
                # Rule 3: Otherwise, finalize the current chunk and start a new one with the current item
                logger.debug(f"Item {i} ({current_item.get('category')}) starts a new chunk (spatial/category break).")
                if current_chunk_items:
                    # Process and add the finished chunk
                    combined_text = " ".join([item.get("text", "") for item in current_chunk_items]).strip()
                    combined_box = [
                        min(item.get("box", [0,0,0,0])[0] for item in current_chunk_items),
                        min(item.get("box", [0,0,0,0])[1] for item in current_chunk_items),
                        max(item.get("box", [0,0,0,0])[2] for item in current_chunk_items),
                        max(item.get("box", [0,0,0,0])[3] for item in current_chunk_items)
                    ]
                    representative_category = current_chunk_items[0].get("category", "text")
                    source_regions_info = [{"box": item.get("box"), "category": item.get("category"), "index": item.get("region_index_on_page")} for item in current_chunk_items]

                    if len(combined_text) >= min_chunk_text_length:
                        final_chunks.append({
                           "text": combined_text,
                           "box": combined_box,
                           "page_key": page_key,
                           "page_chunk_index": chunk_counter,
                           "category": representative_category,
                           "source_regions": source_regions_info,
                            "source_page_filename": sorted_items[0].get("source_image", "N/A")
                        })
                        logger.debug(f"Finalized chunk {chunk_counter} (length {len(combined_text)}, category {representative_category}).")
                        chunk_counter += 1
                    else:
                        logger.debug(f"Filtered small chunk (length {len(combined_text)} < {min_chunk_text_length}). Category: {representative_category}.")

                # Start the new chunk with the current item
                current_chunk_items = [current_item]

    # 3. Finalize the last chunk after the loop
    if current_chunk_items:
        combined_text = " ".join([item.get("text", "") for item in current_chunk_items]).strip()
        combined_box = [
            min(item.get("box", [0,0,0,0])[0] for item in current_chunk_items),
            min(item.get("box", [0,0,0,0])[1] for item in current_chunk_items),
            max(item.get("box", [0,0,0,0])[2] for item in current_chunk_items),
            max(item.get("box", [0,0,0,0])[3] for item in current_chunk_items)
        ]
        representative_category = current_chunk_items[0].get("category", "text")
        source_regions_info = [{"box": item.get("box"), "category": item.get("category"), "index": item.get("region_index_on_page")} for item in current_chunk_items]


        if len(combined_text) >= min_chunk_text_length:
            final_chunks.append({
               "text": combined_text,
               "box": combined_box,
               "page_key": page_key,
               "page_chunk_index": chunk_counter,
               "category": representative_category,
               "source_regions": source_regions_info,
                "source_page_filename": sorted_items[0].get("source_image", "N/A")
            })
            logger.debug(f"Finalized last chunk {chunk_counter} (length {len(combined_text)}, category {representative_category}).")
        else:
            logger.debug(f"Filtered small final chunk (length {len(combined_text)} < {min_chunk_text_length}). Category: {representative_category}.")


    logger.info(f"Finished chunking for {page_key} (Doc: {doc_id}). Created {len(final_chunks)} chunks.")
    return final_chunks

async def index_ocr_data(
        doc_id: str,
        config_dict: Dict,
        chroma_collection_obj: chromadb.api.models.Collection.Collection,
        common_utils_module: Any
    ) -> bool:
        """
        Indexes OCR data (structured chunks from LayoutParser regions) for a visual document into ChromaDB.

        Args:
            doc_id: The document ID being processed.
            config_dict: The configuration dictionary.
            chroma_collection_obj: The ChromaDB collection object.
            common_utils_module: Reference to the common_utils module.

        Returns:
            True if indexing completes successfully with at least one chunk indexed, False otherwise.
        """
        logger.info(f"--- Starting index_ocr_data for doc_id: {doc_id} (Visual Pipeline Step 5) ---")

        if chroma_collection_obj is None:
            logger.error(f"Cannot index doc_id '{doc_id}': ChromaDB collection object not provided.")
            try:
                common_utils_module.update_file_metadata(config_dict, doc_id, metadata_extra={"pipeline_step": "indexing_failed_no_chroma_client"})
            except Exception as e_meta_update:
                logger.error(f"Failed to update metadata for {doc_id} after indexing failure (no client): {e_meta_update}", exc_info=True)
            return False


        processed_doc_base_path = os.path.join(config_dict["vault_directory"], config_dict["processed_docs_subdir"], doc_id)
        ocr_json_path = os.path.join(processed_doc_base_path, config_dict["ocr_results_file"])
        layout_json_path = os.path.join(processed_doc_base_path, config_dict["layout_analysis_file"]) # Needed for page dimensions and LP metadata

        if not os.path.exists(ocr_json_path):
            logger.error(f"OCR results file not found for {doc_id} at {ocr_json_path}. Cannot index.")
            try:
                common_utils_module.update_file_metadata(config_dict, doc_id, metadata_extra={"pipeline_step": "indexing_failed_no_ocr_results"})
            except Exception as e_meta_update:
                logger.error(f"Failed to update metadata for {doc_id} after indexing failure (no OCR file): {e_meta_update}", exc_info=True)
            return False

        try:
            # Load OCR data (new format with 'pages' -> 'extracted_items')
            try:
                with open(ocr_json_path, 'r', encoding='utf-8') as f_ocr:
                    ocr_data_content = json.load(f_ocr)
                # Validate new format
                if not isinstance(ocr_data_content, dict) or ocr_data_content.get("doc_id") != doc_id or "pages" not in ocr_data_content:
                    logger.error(f"Invalid OCR data format or mismatched doc_id for '{doc_id}' in file {ocr_json_path}.")
                    try:
                        common_utils_module.update_file_metadata(config_dict, doc_id, metadata_extra={"pipeline_step": "indexing_failed_invalid_ocr_data"})
                    except Exception as e_meta_update:
                        logger.error(f"Failed to update metadata for {doc_id} after indexing failure (invalid OCR data): {e_meta_update}", exc_info=True)
                    return False
            except Exception as e_load_ocr:
                logger.error(f"Failed to load/parse OCR data from {ocr_json_path} for doc_id {doc_id}: {e_load_ocr}", exc_info=True)
                try:
                    common_utils_module.update_file_metadata(config_dict, doc_id, metadata_extra={"pipeline_step": "indexing_failed_ocr_read_error"})
                except Exception as e_meta_update:
                    logger.error(f"Failed to update metadata for {doc_id} after indexing failure (OCR read error): {e_meta_update}", exc_info=True)
                return False


            # Load layout data for page dimensions (optional, but good for metadata)
            page_dimensions = {}
            # Also load layout data to potentially get original LP categories if needed for chunking logic metadata
            layout_data_per_page_full = {}
            if os.path.exists(layout_json_path):
                try:
                    with open(layout_json_path, 'r', encoding='utf-8') as f_layout:
                        layout_data_per_page_full = json.load(f_layout)
                    # Extract dimensions from layout data if available
                    for p_key, p_layout_data in layout_data_per_page_full.items():
                         if p_layout_data and p_layout_data.get("page_dimensions"):
                             page_dimensions[p_key] = p_layout_data["page_dimensions"]
                except Exception as e_layout_dims:
                    logger.warning(f"Could not load layout data for page dimensions for {doc_id}: {e_layout_dims}", exc_info=True)
            # Fallback for pages not in layout data or missing dimensions
            fallback_dims = {"width": config_dict.get("fallback_page_width", 1000), "height": config_dict.get("fallback_page_height", 1400)}


            text_chunk_audit_dir = None
            if config_dict.get("create_plain_text_audit", False):
                text_chunk_audit_dir = os.path.join(processed_doc_base_path, config_dict.get("text_chunk_output_dir", "extracted_text_chunks"))
                os.makedirs(text_chunk_audit_dir, exist_ok=True)

            # Get original filename from vault metadata
            all_doc_meta_list = common_utils_module.get_vault_files(config_dict)
            doc_specific_meta_info = next((m for m in all_doc_meta_list if m.get("filename") == doc_id), {})
            original_doc_filename = doc_specific_meta_info.get("original_filename", doc_id)

            chunks_for_chroma_upsert = {"ids": [], "documents": [], "metadatas": []}
            total_chunks_actually_indexed = 0
            total_chunks_filtered_pre_index = 0
            pages_with_content_for_chunking = 0 # Track pages with text items for chunking


            ocr_page_keys = sorted(
                ocr_data_content.get("pages", {}).keys(),
                key=lambda page_k_str: int(page_k_str.split('_')[1]) if page_k_str.startswith('page_') and page_k_str.split('_')[1].isdigit() else float('inf')
            )
            logger.info(f"Processing {len(ocr_page_keys)} pages from OCR data for '{doc_id}' (Original: '{original_doc_filename}') for chunking and indexing...")

            # --- Aggregate chunks from all pages ---
            all_final_chunks: List[Dict] = []

            for page_key_str_loop in ocr_page_keys:
                page_data_from_ocr = ocr_data_content["pages"].get(page_key_str_loop)

                # Only process pages that successfully extracted items (status 'success' or 'no_content_extracted' indicates processing ran)
                if not page_data_from_ocr or page_data_from_ocr.get("status") in ["failed", "skipped_filename_parse_error", "skipped_image_read_error", "skipped_preprocessing_failed", "skipped_layout_error", "skipped_no_regions_detected", "skipped_image_missing", "skipped_image_load_error"]:
                     logger.warning(f"Skipping chunking for page {page_key_str_loop} of doc '{doc_id}' due to OCR/Extraction status: {page_data_from_ocr.get('status', 'N/A')}")
                     continue

                # Get only the text items from the extracted_items list for this page
                text_items_for_chunking = [
                    item for item in page_data_from_ocr.get("extracted_items", [])
                    if item.get("type") == "text" and item.get("text", "").strip() # Filter for type 'text' and non-empty text content
                ]

                if not text_items_for_chunking:
                     logger.debug(f"No non-empty text items extracted for page {page_key_str_loop}. Skipping chunking for this page.")
                     continue # No text items to chunk on this page

                pages_with_content_for_chunking += 1 # This page has text content that can be chunked

                current_pg_dims = page_dimensions.get(page_key_str_loop, fallback_dims)

                # --- Call the adapted chunking function ---
                # Pass the list of extracted text items for this page to the chunking function
                try:
                     # Your chunking function needs access to the page_key and doc_id for logging/metadata
                     # Assuming group_lp_regions_into_chunks is the function defined/imported elsewhere
                     page_chunks = group_lp_regions_into_chunks(
                         extracted_text_items=text_items_for_chunking, # Pass the list of dictionaries {type, category, text, box, score, page_key, region_index_on_page}
                         page_width=current_pg_dims.get("width", -1), # Pass page dimensions for spatial rules
                         page_height=current_pg_dims.get("height", -1),
                         # Pass relevant chunking config parameters
                         chunk_vertical_proximity_threshold_px=config_dict.get("chunk_vertical_proximity_threshold_px", 15),
                         chunk_vertical_proximity_threshold_ratio_of_height=config_dict.get("chunk_vertical_proximity_threshold_ratio_of_height", 0.5),
                         chunk_horizontal_overlap_ratio=config_dict.get("chunk_horizontal_overlap_ratio", 0.5),
                         always_new_chunk_categories=config_dict.get("always_new_chunk_categories", ["title", "caption"]),
                         merge_categories=config_dict.get("merge_categories", ["text", "list"]),
                         min_chunk_text_length=config_dict.get("min_chunk_text_length", config_dict.get("min_chunk_length_for_indexing", 5)), # Use min_chunk_text_length first, fallback to indexing minimum
                         # Pass page/doc identifiers
                         page_key=page_key_str_loop,
                         doc_id=doc_id
                     )
                     all_final_chunks.extend(page_chunks) # Add chunks from this page to the overall list
                     logger.debug(f"Page {page_key_str_loop}: Chunking resulted in {len(page_chunks)} final chunks.")
                except Exception as e_chunking:
                     logger.error(f"Error during chunking for page {page_key_str_loop} of doc {doc_id}: {e_chunking}", exc_info=True)
                     # Log the error but continue to the next page if one page's chunking fails
                     pass


            # --- Prepare chunks from all_final_chunks for Chroma upsert ---
            logger.info(f"Preparing {len(all_final_chunks)} chunks for ChromaDB upsert...")

            if not all_final_chunks:
                 logger.warning(f"No final chunks generated after chunking process for {doc_id}.")
                 # Proceed to update metadata as complete but empty
                 try:
                    # Get counts of image/table regions from ocr_data_content for metadata even if no text chunks
                    image_region_count = sum(len([item for item in p.get('extracted_items',[]) if item.get('type')=='image']) for p in ocr_data_content.get('pages', {}).values())
                    table_region_count = sum(len([item for item in p.get('extracted_items',[]) if item.get('type')=='table']) for p in ocr_data_content.get('pages', {}).values())

                    common_utils_module.update_file_metadata(config_dict, doc_id, metadata_extra={
                        "indexing_complete": True, # Consider it 'complete' with 0 chunks
                        "indexed_timestamp": datetime.now().isoformat(),
                        "pipeline_step": "indexing_complete_no_chunks",
                        "indexed_chunk_count": 0,
                        "layout_tool": "LayoutParser", # Still note the tool used
                        "image_region_count": image_region_count,
                        "table_region_count": table_region_count,
                    })
                 except Exception as e_meta_update:
                    logger.error(f"Failed to update metadata for {doc_id} after 0 chunks indexed: {e_meta_update}", exc_info=True)
                 return True # Return True as the process finished, even if no chunks resulted


            for chunk_iter_data in all_final_chunks:
                chunk_text_content = chunk_iter_data.get("text", "").strip()
                chunk_bbox = chunk_iter_data.get("box")
                chunk_page_key = chunk_iter_data.get("page_key") # Should come from the chunking function
                chunk_category = chunk_iter_data.get("category", "unknown") # Should come from the chunking function
                chunk_index_on_page = chunk_iter_data.get("page_chunk_index") # Index assigned during chunking
                source_regions = chunk_iter_data.get("source_regions", []) # List of original LP region info

                # Create a unique and stable ID for the chunk
                # Use doc_id, page_key, and chunk index within the page
                # Example: f"{doc_id}_page_1_chunk5"
                if not chunk_page_key or chunk_index_on_page is None:
                     logger.error(f"Chunk missing essential metadata (page_key or page_chunk_index). Skipping chunk: {chunk_iter_data.get('text', '')[:50]}...")
                     total_chunks_filtered_pre_index += 1
                     continue # Skip this chunk if critical metadata is missing

                # Ensure chunk_page_key is safe for forming an ID
                sane_chunk_page_key = re.sub(r'\W+', '_', str(chunk_page_key))
                unique_chunk_id_chroma = f"{doc_id}_{sane_chunk_page_key}_chunk{chunk_index_on_page}"[:250] # Max length 250 in ChromaDB


                # Filter short/empty chunks BEFORE adding to the upsert list
                # This filtering is also done in the chunking function, but good to double-check
                min_len_for_chunk_indexing = config_dict.get("min_chunk_length_for_indexing", 5)
                if not chunk_text_content or len(chunk_text_content) < min_len_for_chunk_indexing:
                    logger.debug(f"Skipping short/empty chunk (len {len(chunk_text_content)} < {min_len_for_chunk_indexing}) {unique_chunk_id_chroma}: '{chunk_text_content[:30]}...'")
                    total_chunks_filtered_pre_index += 1
                    continue

                # Get page number from the page_key included by the chunking function
                chunk_page_num_match = re.search(r'page_(\d+)', chunk_page_key)
                chunk_page_number_meta = int(chunk_page_num_match.group(1)) if chunk_page_num_match else 'N/A'

                # Find the correct page dimensions for this chunk's page
                chunk_pg_dims = page_dimensions.get(chunk_page_key, fallback_dims)

                # Prepare source region boxes as a list of JSON strings for metadata
                # Ensure box coordinates are ints before dumping to JSON
                source_region_boxes_json = [json.dumps([int(c) for c in b.get("box")]) for b in source_regions if b.get("box") and len(b.get("box"))==4] if source_regions else None


                metadata_for_chroma_chunk = {
                    "doc_id": doc_id,
                    "original_filename": original_doc_filename,
                    "page_number": chunk_page_number_meta,
                    "chunk_index_on_page": chunk_index_on_page, # Index assigned during chunking
                    "layout_category": chunk_category, # Category derived from LayoutParser
                    "bounding_box": json.dumps(chunk_bbox) if chunk_bbox and len(chunk_bbox)==4 else None, # Ensure box is valid and JSON string
                    "text_length": len(chunk_text_content),
                    "indexed_at": datetime.now().isoformat(),
                    "layout_tool": "LayoutParser", # Add tool used for indexing
                    # Add dimensions for potential future use in retrieval
                    "page_width": chunk_pg_dims.get("width", -1),
                    "page_height": chunk_pg_dims.get("height", -1),
                    "source_regions_boxes": source_region_boxes_json, # Store boxes of original LP regions if needed
                    "source_regions_categories": [b.get("category") for b in source_regions if b.get("category")] if source_regions else None # Store categories of source regions
                }
                # Clean up metadata - remove None values, convert any remaining non-JSON-serializable types if necessary
                metadata_for_chroma_chunk = {k: v for k, v in metadata_for_chroma_chunk.items() if v is not None}
                # Ensure all metadata values are JSON serializable (ChromaDB requires this) - Redundant check, but safer
                try:
                    # Perform a test dump to catch non-serializable values
                    json.dumps(metadata_for_chroma_chunk)
                except TypeError as e_meta_json:
                     logger.error(f"Metadata for chunk {unique_chunk_id_chroma} is not JSON serializable: {e_meta_json}. Metadata: {metadata_for_chroma_chunk}", exc_info=True)
                     # Attempt to fix common non-serializable types (e.g., numpy types)
                     # This conversion is not foolproof but handles basic numpy types
                     metadata_for_chroma_chunk = {
                         k: (float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else str(v) if v is not None else None)
                         for k, v in metadata_for_chroma_chunk.items()
                     }
                     # Remove None values that might have resulted from conversion
                     metadata_for_chroma_chunk = {k: v for k, v in metadata_for_chroma_chunk.items() if v is not None}
                     try:
                         json.dumps(metadata_for_chroma_chunk)
                     except TypeError as e_meta_json_retry:
                          logger.error(f"Metadata for chunk {unique_chunk_id_chroma} still not JSON serializable after retry: {e_meta_json_retry}. Skipping chunk.", exc_info=True)
                          total_chunks_filtered_pre_index += 1 # Count as filtered
                          continue # Skip adding this chunk if metadata can't be serialized


                if config_dict.get("debug_mode", False) and doc_id == config_dict.get("debug_target_doc_id"):
                    logger.info(f"DEBUG Indexing Visual Chunk for {doc_id}, Page {chunk_page_number_meta}, Chunk Index {chunk_index_on_page}: ID={unique_chunk_id_chroma}, Meta={json.dumps(metadata_for_chroma_chunk)[:200]}..., Text='{chunk_text_content[:150]}...'")


                chunks_for_chroma_upsert["ids"].append(unique_chunk_id_chroma)
                chunks_for_chroma_upsert["documents"].append(chunk_text_content)
                chunks_for_chroma_upsert["metadatas"].append(metadata_for_chroma_chunk)
                total_chunks_actually_indexed += 1

                # Write audit file if enabled
                if text_chunk_audit_dir:
                    try:
                        # Use a more stable filename based on doc_id, page, and chunk index
                        sane_audit_fn = f"{doc_id}_{chunk_page_key}_chunk{chunk_index_on_page}.txt"
                        sane_audit_fn = re.sub(r'[<>:"/\\|?*]', '_', sane_audit_fn) # Sanitize filename
                        audit_f_path = os.path.join(text_chunk_audit_dir, sane_audit_fn)
                        with open(audit_f_path, "w", encoding="utf-8") as audit_f_handle:
                            audit_f_handle.write(f"Chunk ID: {unique_chunk_id_chroma}\n")
                            audit_f_handle.write(f"Doc ID: {doc_id}, Page: {chunk_page_number_meta}, Chunk Index: {chunk_index_on_page}\n")
                            audit_f_handle.write(f"Layout Category: {chunk_category}\n")
                            audit_f_handle.write(f"Box: {metadata_for_chroma_chunk.get('bounding_box', 'N/A')}\n")
                            audit_f_handle.write(f"Source Regions: {metadata_for_chroma_chunk.get('source_regions_boxes')}\n") # Use the JSON list from metadata
                            audit_f_handle.write(f"Source Categories: {metadata_for_chroma_chunk.get('source_regions_categories')}\n")
                            audit_f_handle.write("---\n")
                            audit_f_handle.write(chunk_text_content)
                    except Exception as e_audit_save:
                        logger.warning(f"Failed to write audit chunk {unique_chunk_id_chroma}: {e_audit_save}", exc_info=True)


                # Perform batch upsert
                batch_size_chroma = config_dict.get("embedding_batch_size", 16)
                if len(chunks_for_chroma_upsert["ids"]) >= batch_size_chroma:
                    try:
                        if config_dict.get("debug_mode", False) and doc_id == config_dict.get("debug_target_doc_id"):
                            logger.info(f"DEBUG Visual Batch Upserting {len(chunks_for_chroma_upsert['ids'])} items for {doc_id}...")
                        # ChromaDB operations are blocking, use asyncio.to_thread
                        await asyncio.to_thread(
                            lambda: chroma_collection_obj.upsert(
                                ids=chunks_for_chroma_upsert["ids"],
                                documents=chunks_for_chroma_upsert["documents"],
                                metadatas=chunks_for_chroma_upsert["metadatas"]
                            )
                        )
                        chunks_for_chroma_upsert = {"ids": [], "documents": [], "metadatas": []} # Reset batch
                    except Exception as e_chroma_upsert_batch:
                        logger.error(f"Error upserting visual batch to ChromaDB for {doc_id}: {e_chroma_upsert_batch}", exc_info=True)
                        # Attempt to save what was processed so far in metadata before failing? Or just fail.
                        # For now, just fail the whole indexing process for this doc.
                        try:
                            common_utils_module.update_file_metadata(config_dict, doc_id, metadata_extra={
                                "indexing_complete": False,
                                "pipeline_step": f"indexing_failed_batch_upsert_{e_chroma_upsert_batch.__class__.__name__}",
                                "indexed_chunk_count": total_chunks_actually_indexed # Report chunks processed before failure
                            })
                        except Exception as e_meta_fail:
                            logger.error(f"Error updating metadata to mark batch upsert failure for {doc_id}: {e_meta_fail}", exc_info=True)
                        return False # Indicate failure

            # Upsert any remaining chunks in the last batch
            if chunks_for_chroma_upsert["ids"]:
                try:
                    if config_dict.get("debug_mode", False) and doc_id == config_dict.get("debug_target_doc_id"):
                        logger.info(f"DEBUG Visual Final Batch Upserting {len(chunks_for_chroma_upsert['ids'])} items for {doc_id}...")
                    await asyncio.to_thread(
                        lambda: chroma_collection_obj.upsert(
                            ids=chunks_for_chroma_upsert["ids"],
                            documents=chunks_for_chroma_upsert["documents"],
                            metadatas=chunks_for_chroma_upsert["metadatas"]
                        )
                    )
                except Exception as e_chroma_upsert_final:
                    logger.error(f"Error upserting visual final batch to ChromaDB for {doc_id}: {e_chroma_upsert_final}", exc_info=True)
                    try:
                        common_utils_module.update_file_metadata(config_dict, doc_id, metadata_extra={
                            "indexing_complete": False,
                            "pipeline_step": f"indexing_failed_final_upsert_{e_chroma_upsert_final.__class__.__name__}",
                            "indexed_chunk_count": total_chunks_actually_indexed # Report chunks processed before failure
                        })
                    except Exception as e_meta_fail:
                        logger.error(f"Error updating metadata to mark final upsert failure for {doc_id}: {e_meta_fail}", exc_info=True)
                    return False # Indicate failure


            logger.info(f"Finished visual indexing processing for {doc_id}. Chunks generated: {len(all_final_chunks)}, Indexed: {total_chunks_actually_indexed} chunks, Filtered: {total_chunks_filtered_pre_index} chunks.")

            # Verify ChromaDB contents - optional but good practice
            try: # <--- This is the try block that needed an except/finally
                # Use await asyncio.to_thread as ChromaDB get can be blocking
                verification_result = await asyncio.to_thread(
                    lambda: chroma_collection_obj.get(where={"doc_id": doc_id}, limit=10)
                )
                chunk_ids_in_chroma = verification_result.get("ids", [])
                chunk_count_in_chroma = len(chunk_ids_in_chroma)
                logger.info(f"Verification: Found {chunk_count_in_chroma} chunks in ChromaDB for doc_id {doc_id}.")

                # Compare expected indexed count vs actual count in Chroma (might differ if Chroma has duplicates or internal issues)
                # A simpler check: did we *attempt* to index > 0 and find 0?
                if total_chunks_actually_indexed > 0 and chunk_count_in_chroma == 0:
                    # This indicates a major issue if we thought we indexed chunks but none are found
                    logger.error(f"Major discrepancy: Attempted to index {total_chunks_actually_indexed} chunks for {doc_id} but found 0 in ChromaDB verification.")
                    try:
                        common_utils_module.update_file_metadata(config_dict, doc_id, metadata_extra={
                            "indexing_complete": False,
                            "pipeline_step": "indexing_failed_verification_mismatch",
                            "indexed_chunk_count": total_chunks_actually_indexed
                        })
                    except Exception as e_meta_fail:
                        logger.error(f"Error updating metadata after verification mismatch for {doc_id}: {e_meta_fail}", exc_info=True)
                    return False
                elif total_chunks_actually_indexed > 0 and chunk_count_in_chroma > 0:
                     # Successful indexing (at least some chunks found)
                     logger.info(f"Sample chunks for {doc_id}: {[d[:50] + '...' for d in verification_result.get('documents', [])[:min(3, len(verification_result.get('documents',[]))) ] ]}")
                     # Proceed to update metadata as complete
                     pass # Verification successful enough, proceed to final metadata update
                elif total_chunks_actually_indexed == 0:
                     # This case (no chunks generated/filtered) is handled before the upsert loop.
                     # If we are here, it means chunking generated 0 indexable chunks.
                     # Metadata should already be set to "indexing_complete_no_chunks".
                     pass # No indexing was expected, verification of 0 count is fine


            except Exception as e_verify: # <--- Added the except block for the verification try
                logger.error(f"Error verifying ChromaDB contents for {doc_id}: {e_verify}", exc_info=True)
                # Log the error but mark indexing as incomplete due to verification failure.
                indexed_count_before_fail = total_chunks_actually_indexed if 'total_chunks_actually_indexed' in locals() else 0 # Get latest count before verification error
                try:
                    common_utils_module.update_file_metadata(config_dict, doc_id, metadata_extra={
                        "indexing_complete": False, # Mark as not complete due to verification failure
                        "pipeline_step": "indexing_failed_verification",
                        "indexed_chunk_count": indexed_count_before_fail # Report chunks processed before verification failed
                    })
                except Exception as e_meta_fail:
                    logger.error(f"Error updating metadata to mark verification failure for {doc_id}: {e_meta_fail}", exc_info=True)
                # Decide whether to return True or False here. Returning False seems safer if verification failed.
                return False


            # If we reach here and total_chunks_actually_indexed > 0, indexing is considered successful.
            # The case where total_chunks_actually_indexed == 0 is handled by the check after the chunking loop.
            if total_chunks_actually_indexed > 0:
                try:
                    # Get counts of image/table regions from ocr_data_content for metadata
                    image_region_count = sum(len([item for item in ocr_data_content.get('pages', {}).get(p_key, {}).get('extracted_items',[]) if item.get('type')=='image']) for p_key in ocr_page_keys if p_key in ocr_data_content.get('pages',{}))
                    table_region_count = sum(len([item for item in ocr_data_content.get('pages', {}).get(p_key, {}).get('extracted_items',[]) if item.get('type')=='table']) for p_key in ocr_page_keys if p_key in ocr_data_content.get('pages',{}))


                    common_utils_module.update_file_metadata(config_dict, doc_id, metadata_extra={
                        "indexing_complete": True,
                        "indexed_timestamp": datetime.now().isoformat(),
                        "pipeline_step": "indexing_complete",
                        "indexed_chunk_count": total_chunks_actually_indexed,
                        # Add layout tool used metadata
                        "layout_tool": "LayoutParser",
                        "image_region_count": image_region_count,
                        "table_region_count": table_region_count,
                    })
                    logger.info(f"Metadata updated for {doc_id}: Indexing complete with {total_chunks_actually_indexed} chunks.")
                except Exception as e_meta_update:
                    logger.error(f"Failed to update metadata for {doc_id}: {e_meta_update}", exc_info=True)
                    # Metadata update failed, but indexing *did* happen.
                    # Decide if this is a critical failure. Let's return False as the state is not fully updated.
                    return False
                return True
            else:
                # This else block should technically not be reachable if the check after the chunking loop works correctly.
                # It handles the case where total_chunks_actually_indexed is 0 and the early check was missed.
                # The metadata should already be set to "indexing_complete_no_chunks".
                logger.warning(f"Indexing process finished for {doc_id} with 0 indexed chunks (fall-through case).")
                return True


        except Exception as e_index_main:
            logger.error(f"An unexpected error occurred during visual indexing for {doc_id}: {e_index_main}", exc_info=True)
            # Note: total_chunks_actually_indexed might not be defined here if error happens very early
            indexed_count_before_fail = total_chunks_actually_indexed if 'total_chunks_actually_indexed' in locals() else 0
            try:
                common_utils_module.update_file_metadata(config_dict, doc_id, metadata_extra={
                    "indexing_complete": False,
                    "pipeline_step": f"indexing_failed_main_{e_index_main.__class__.__name__}",
                    "indexed_chunk_count": indexed_count_before_fail # Include partial count before failure
                })
            except Exception as e_meta_fail:
                logger.error(f"Error updating metadata to mark main indexing failure for {doc_id}: {e_meta_fail}", exc_info=True)
            return False

async def get_visual_context_chroma(
    doc_id: str,
    query: str,
    config_dict: Dict,
    chroma_collection_obj: chromadb.api.models.Collection.Collection, # Or ChromaCollection alias
    common_utils_module: Any,
    ollama_client: Optional[OpenAI],
    top_k: int = 15,
    page_filter: Optional[int] = None
) -> List[Dict]:
    """
    Retrieves relevant visual context chunks for a query from a specific document's
    indexed chunks in ChromaDB, optionally filtered by page number.
    Adapts to new metadata structure from LayoutParser-based indexing.
    """
    logger.info(f"--- Starting get_visual_context_chroma ---")
    logger.info(f"Query: '{query[:50]}...' for Doc ID: {doc_id}")
    logger.info(f"Requested Top_k: {top_k}, Page Filter: {page_filter}")

    if chroma_collection_obj is None:
        logger.error(f"Cannot retrieve visual context for doc_id '{doc_id}': ChromaDB collection object not provided.")
        return []
    # Note: Ollama client is needed for embedding, not direct retrieval, but check is fine.
    if ollama_client is None:
         logger.error("Ollama client not available for embedding in get_visual_context_chroma.")
         return []


    context_items: List[Dict] = []
    query_embedding_list = None

    try:
        # Generate query embedding - This is a blocking LLM API call, wrap it!
        embedding_model = config_dict.get("ollama_embedding_model", "mxbai-embed-large:latest")
        logger.debug(f"Generating query embedding using model: {embedding_model}")
        try:
            embedding_response = await asyncio.to_thread(
                lambda: ollama_client.embeddings.create(
                    model=embedding_model,
                    input=[query]
                )
            )
        except Exception as e_embed:
             logger.error(f"Error generating query embedding: {e_embed}", exc_info=True)
             return []


        if embedding_response and hasattr(embedding_response, 'data') and isinstance(embedding_response.data, list) and embedding_response.data and hasattr(embedding_response.data[0], 'embedding'):
            query_embedding_list = embedding_response.data[0].embedding
            logger.debug("Generated query embedding list successfully.")
        else:
            logger.warning("Failed to generate query embedding or invalid response format from Ollama.")
            return []

        # --- Construct the where filter for ChromaDB ---
        where_filter_dict: Dict[str, Any] = {"doc_id": doc_id}

        # Add page filter to the where clause if provided
        if page_filter is not None and isinstance(page_filter, int) and page_filter > 0:
            where_filter_dict["page_number"] = page_filter # Assuming 'page_number' is your metadata key
            logger.info(f"Applying page filter for Chroma retrieval: page_number = {page_filter}")


        n_results_to_fetch = max(1, top_k)

        logger.debug(f"Querying Chroma with embedding: n_results={n_results_to_fetch}, filter={where_filter_dict}")

        # Query ChromaDB - This is a blocking call, wrap it!
        try:
            chroma_results = await asyncio.to_thread(
                lambda: chroma_collection_obj.query(
                    query_embeddings=[query_embedding_list],
                    n_results=n_results_to_fetch,
                    where=where_filter_dict, # Use the filter dictionary
                    include=['metadatas', 'documents', 'distances']
                )
            )
        except Exception as e_chroma_query:
             logger.error(f"Error querying ChromaDB: {e_chroma_query}", exc_info=True)
             return []


        logger.info(f"--- DEBUG: ChromaDB Raw Query Results for Visual Context for doc '{doc_id}' (Page filter: {page_filter}) ---")
        # Check if results are valid and contain data lists
        if (chroma_results and
            isinstance(chroma_results.get('ids'), list) and chroma_results['ids'] and isinstance(chroma_results['ids'][0], list) and
            isinstance(chroma_results.get('documents'), list) and chroma_results['documents'] and isinstance(chroma_results['documents'][0], list) and
            isinstance(chroma_results.get('metadatas'), list) and chroma_results['metadatas'] and isinstance(chroma_results['metadatas'][0], list) and
            isinstance(chroma_results.get('distances'), list) and chroma_results['distances'] and isinstance(chroma_results['distances'][0], list)):

            # Access the actual lists of results (assuming query_embeddings=[...], so results[0])
            ids_list_res = chroma_results['ids'][0]
            documents_list_res = chroma_results['documents'][0]
            metadatas_list_res = chroma_results['metadatas'][0]
            distances_list_res = chroma_results['distances'][0]

            logger.info(f"Number of chunks retrieved by Chroma for doc '{doc_id}' (Page filter: {page_filter}): {len(ids_list_res)}")

            # Process retrieved items
            for i_idx in range(len(ids_list_res)):
                 item_id = ids_list_res[i_idx]
                 # Use .get() with default [] for safety if lists are not of the same length
                 item_doc_content = documents_list_res[i_idx] if i_idx < len(documents_list_res) else ""
                 item_meta = metadatas_list_res[i_idx] if i_idx < len(metadatas_list_res) else {}
                 item_dist = distances_list_res[i_idx] if i_idx < len(distances_list_res) else 1.0 # Default distance 1 if missing

                 item_score = round(1.0 - float(item_dist), 4) # Calculate score from distance

                 # --- EXTRACT METADATA USING NEW KEYS ---
                 # Align metadata extraction with what index_ocr_data stores
                 chunk_metadata: Dict[str, Any] = {
                     "doc_id": item_meta.get('doc_id', doc_id),
                     "original_filename": item_meta.get('original_filename', doc_id), # Fallback to doc_id
                     "page_number": item_meta.get('page_number', 'N/A'), # Page number from chunking
                     "chunk_index_on_page": item_meta.get('chunk_index_on_page', 'N/A'), # Index from chunking
                     "layout_category": item_meta.get('layout_category', 'unknown'), # LayoutParser category
                     "bounding_box": item_meta.get('bounding_box', None), # Bbox of the final chunk
                     "page_width": item_meta.get('page_width', -1),
                     "page_height": item_meta.get('page_height', -1),
                     "type": "visual_chunk", # Consistent type
                     "layout_tool": item_meta.get('layout_tool', 'unknown'), # e.g., "LayoutParser"
                     # source_regions_boxes and categories are stored as JSON strings or lists in metadata
                     "source_regions_boxes": item_meta.get('source_regions_boxes', None),
                     "source_regions_categories": item_meta.get('source_regions_categories', None),
                 }

                 # Safely parse bounding box JSON string if it exists
                 if isinstance(chunk_metadata.get('bounding_box'), str):
                     try:
                         chunk_metadata['bounding_box'] = json.loads(chunk_metadata['bounding_box'])
                         # Ensure it's a list of 4 ints after parsing
                         if not isinstance(chunk_metadata['bounding_box'], list) or len(chunk_metadata['bounding_box']) != 4:
                              logger.warning(f"Parsed bounding box for chunk {item_id} is not a list of 4: {chunk_metadata['bounding_box']}")
                              chunk_metadata['bounding_box'] = None # Invalidate if not correct format
                         else:
                              chunk_metadata['bounding_box'] = [int(c) for c in chunk_metadata['bounding_box']] # Ensure ints
                     except (json.JSONDecodeError, TypeError):
                         logger.warning(f"Could not parse bounding box JSON from metadata for chunk {item_id}: {item_meta.get('bounding_box')}.")
                         chunk_metadata['bounding_box'] = None # Invalidate if parsing fails

                 # Safely parse source_regions_boxes JSON strings if they exist
                 source_region_boxes_parsed = []
                 if isinstance(chunk_metadata.get('source_regions_boxes'), list):
                     for box_json_str in chunk_metadata['source_regions_boxes']:
                         if isinstance(box_json_str, str):
                            try:
                                box_coords = json.loads(box_json_str)
                                # Ensure it's a list of 4 ints after parsing
                                if isinstance(box_coords, list) and len(box_coords) == 4:
                                    source_region_boxes_parsed.append([int(c) for c in box_coords])
                                else:
                                     logger.warning(f"Parsed source region box for chunk {item_id} is not a list of 4: {box_coords}")
                            except (json.JSONDecodeError, TypeError):
                                logger.warning(f"Could not parse source region box JSON '{box_json_str}' from metadata for chunk {item_id}.")
                         else:
                             logger.warning(f"Source region box in metadata for chunk {item_id} is not a string: {box_json_str}")
                     chunk_metadata['source_regions_boxes'] = source_region_boxes_parsed if source_region_boxes_parsed else None # Use parsed list or None
                 else:
                     chunk_metadata['source_regions_boxes'] = None # Ensure it's None if not a list


                 # source_regions_categories should already be a list of strings
                 if not isinstance(chunk_metadata.get('source_regions_categories'), list):
                     chunk_metadata['source_regions_categories'] = None # Ensure it's None if not a list


                 # Add the processed item to the context list
                 context_items.append({
                     "id": item_id,
                     "content": item_doc_content,
                     "score": item_score,
                     "distance": round(float(item_dist), 4),
                     "metadata": chunk_metadata # Use the updated metadata dictionary
                 })

            logger.info(f"Processed {len(context_items)} relevant visual chunks from ChromaDB for doc '{doc_id}' (Page filter: {page_filter}).")

        else:
            logger.info(f"ChromaDB query returned empty or invalid results structure for doc_id '{doc_id}' (Page filter: {page_filter}) matching query.")
            logger.debug(f"ChromaDB raw results structure: {chroma_results}") # Log structure for debugging

        logger.info(f"--- END DEBUG: ChromaDB Raw Query Results ---")


    except Exception as e_get_visual:
        logger.error(f"Error retrieving visual context from ChromaDB for doc '{doc_id}' (Page filter: {page_filter}): {e_get_visual}", exc_info=True)
        return []

    # --- DEBUG Logging of Retrieved Context ---
    logger.info(f"--- DEBUG: Processed Visual Context Items (before returning, page filter: {page_filter}) ---")
    if context_items:
        # Sort items by score for logging, even if the final list will be sorted differently for the LLM prompt
        sorted_log_items = sorted(context_items, key=lambda x: x.get("score", 0.0), reverse=True)
        for i_log, item_log in enumerate(sorted_log_items[:min(len(sorted_log_items), config_dict.get('visual_context_results', 15))]): # Log top N
            meta_log = item_log.get('metadata', {})
            content_preview_log = item_log.get('content', '')[:80].replace('\n', ' ') + "..."
            # Update log message to show new metadata
            logger.info(f"  Item {i_log}: Doc='{meta_log.get('original_filename', 'Unknown')}', Page={meta_log.get('page_number', 'N/A')}, ChunkIdx={meta_log.get('chunk_index_on_page', 'N/A')}, Category={meta_log.get('layout_category', 'N/A')}, Score={item_log.get('score', 0):.4f}, Distance={item_log.get('distance', -1):.4f}, Content='{content_preview_log}'")
    else:
        logger.info("  No context items to log.")
    logger.info(f"--- END DEBUG: Processed Visual Context Items ---")


    return context_items


async def ollama_chat_visual_async(
    config_dict: Dict,
    user_input: str,
    selected_doc_ids: List[str],
    client_id: str,
    ollama_client: Optional[OpenAI],
    system_message: str,
    manager: Any,
    common_utils_module: Any,
    chroma_collection_obj: chromadb.api.models.Collection.Collection,
    page_filter: Optional[int] = None
) -> Tuple[str, List[Dict]]:
    """
    Orchestrates getting visual context and generating a response using the LLM.
    """
    logger.info(f"--- Visual Chat Orchestrator for {client_id} --- Query: '{user_input[:50]}...' Docs: {selected_doc_ids}")
    logger.info(f"Visual Chat Parameters: page_filter={page_filter}, len(selected_doc_ids)={len(selected_doc_ids)}")

    # --- Pre-checks ---
    if ollama_client is None:
        logger.error("LLM client not ready in ollama_chat_visual_async.")
        return "LLM client not ready.", []
    if not selected_doc_ids:
        logger.warning("No visual documents selected for this query after readiness check.")
        return "No visual documents selected for this query.", []
    if chroma_collection_obj is None:
        logger.error("ChromaDB collection not initialized in ollama_chat_visual_async.")
        return "Document index not initialized.", []

    # --- Retrieve Chat History ---
    memory = manager.get_client_memory(client_id)
    chat_history_lc_msgs = []
    if memory:
        try:
            loaded_vars = await asyncio.to_thread(memory.load_memory_variables, {})
            chat_history_lc_msgs = loaded_vars.get(memory.memory_key, [])
        except Exception as e_mem:
            logger.error(f"Error loading memory for {client_id}: {e_mem}", exc_info=True)
    formatted_history = [{"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": str(m.content)} for m in chat_history_lc_msgs if hasattr(m, 'content')]


    await manager.send_json(client_id, {"type": "status", "message": f"Searching {len(selected_doc_ids)} visual document(s)..."})
    all_visual_context = []

    # Retrieve context concurrently from multiple documents
    context_retrieval_tasks = []
    # --- Calculate Chroma Retrieval Limit based on multiplier ---
    final_context_limit_for_llm = config_dict.get("visual_context_results", 15) # Default 15 chunks for LLM
    chroma_retrieval_multiplier = config_dict.get("chroma_retrieval_multiplier", 3) # Default multiplier is 3
    chroma_retrieval_limit_total = final_context_limit_for_llm * chroma_retrieval_multiplier # Request e.g., 45 chunks total from Chroma

    # Distribute the Chroma retrieval limit among selected documents (ensure at least 1 per doc)
    chroma_retrieval_per_doc_k = max(1, chroma_retrieval_limit_total // len(selected_doc_ids)) if chroma_retrieval_limit_total > 0 and len(selected_doc_ids) > 0 else 1

    logger.info(f"Retrieving top {chroma_retrieval_per_doc_k} chunks per document from Chroma (total {chroma_retrieval_limit_total} initially requested) for query '{user_input[:50]}...' for doc(s): {selected_doc_ids}.")
    # --- END MODIFIED CALCULATION ---


    for doc_id_visual in selected_doc_ids:
        context_retrieval_tasks.append(
             get_visual_context_chroma(
                 doc_id=doc_id_visual,
                 query=user_input,
                 config_dict=config_dict,
                 chroma_collection_obj=chroma_collection_obj,
                 common_utils_module=common_utils_module,
                 ollama_client=ollama_client,
                 top_k=chroma_retrieval_per_doc_k, # <--- Use the potentially larger retrieval limit per doc
                 page_filter=page_filter
             )
        )

    results_from_context_tasks = await asyncio.gather(*context_retrieval_tasks, return_exceptions=True)

    for res in results_from_context_tasks:
         if isinstance(res, Exception):
             logger.error(f"Exception during visual context retrieval for one document: {res}", exc_info=True)
         elif isinstance(res, list):
             all_visual_context.extend(res)
         else:
              logger.error(f"Unexpected result type from context retrieval task: {type(res)}")

    # Sort ALL retrieved chunks by score and take the TOP N (final_context_limit_for_llm) results for the LLM prompt
    all_visual_context.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    relevant_ocr_chunks_for_llm = all_visual_context[:final_context_limit_for_llm] # <--- Select the final subset for LLM prompt


    # Store the final context subset sent to the LLM for potential audit/debugging
    setattr(ollama_chat_visual_async, 'last_context', relevant_ocr_chunks_for_llm)


    # --- Handle No Relevant Context Found (after selecting the final subset) ---
    if not relevant_ocr_chunks_for_llm:
        logger.warning(f"No relevant context found (after ranking/limiting to {final_context_limit_for_llm} chunks) for query '{user_input[:50]}...' in docs {selected_doc_ids} (Page filter: {page_filter})")
        all_vault_meta = common_utils_module.get_vault_files(config_dict)
        no_info_html = common_utils_module.generate_no_information_response(config_dict, user_input, selected_doc_ids, all_vault_meta)
        no_info_html += "<p>This may be because no relevant text was found in the top retrieved chunks, or the document's content isn't suitable for this query.</p>"

        if memory:
            await asyncio.to_thread(memory.save_context, {"input": user_input}, {"output": no_info_html})

        await manager.send_json(client_id, {"type": "status", "message": "No relevant context found in selected documents."})
        return no_info_html, []


    # --- Format Context for the LLM Prompt ---
    context_str_for_llm = ""
    # Use the subset selected for the LLM (relevant_ocr_chunks_for_llm)
    if relevant_ocr_chunks_for_llm:
        context_parts = []
        # Sort chunks for the LLM prompt by doc_id, page number, then bounding box Y coordinate
        sorted_chunks_for_llm_prompt = sorted(
            relevant_ocr_chunks_for_llm,
            key=lambda x: (
                x.get('metadata', {}).get('doc_id', ''),
                int(x.get('metadata', {}).get('page_number', float('inf'))),
                x.get('metadata', {}).get('bounding_box', [0,0,0,0])[1] if x.get('metadata', {}).get('bounding_box') else 0
            )
        )


        for chunk in sorted_chunks_for_llm_prompt:
            meta = chunk.get("metadata", {})
            doc_name = meta.get('original_filename', meta.get('doc_id', 'Unknown'))
            page_num = meta.get('page_number', 'N/A')
            block_idx = meta.get('block_index_on_page', meta.get('block_num', 'N/A'))
            layout_type = meta.get('layout_type', 'unknown')
            text = chunk.get("content", "")

            page_num_display = page_num if page_num is not None and page_num != 'N/A' else 'N/A'
            block_idx_display = block_idx if block_idx is not None and block_idx != 'N/A' else 'N/A'


            if text and text.strip():
                header = f"--- Document: {doc_name}, Page: {page_num_display}, Chunk Index: {meta.get('chunk_index_on_page', 'N/A')}, Category: {meta.get('layout_category', 'unknown').replace('_', ' ').title()}, Source Tool: {meta.get('layout_tool', 'unknown')} ---"
                context_parts.append(f"{header}\n{text.strip()}\n--- End Block ---")

        context_str_for_llm = "\n\n".join(context_parts)


        if not context_str_for_llm:
             logger.warning(f"Context formatting resulted in empty string for query '{user_input[:50]}...' despite finding chunks.")
             all_vault_meta = common_utils_module.get_vault_files(config_dict)
             no_info_html = common_utils_module.generate_no_information_response(config_dict, user_input, selected_doc_ids, all_vault_meta)
             no_info_html += "<p>Relevant chunks were found, but their content could not be formatted into context for the LLM.</p>"
             if memory:
                await asyncio.to_thread(memory.save_context, {"input": user_input}, {"output": no_info_html})
             await manager.send_json(client_id, {"type": "status", "message": "Context formatting failed."})
             return no_info_html, []


        # --- Construct Messages for LLM API Call ---
        messages_for_api = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Provided Context from Files:\n{context_str_for_llm}"}
        ]
        messages_for_api.extend(formatted_history)
        messages_for_api.append({"role": "user", "content": user_input})


        logger.debug(f"Messages structure for LLM API ({len(messages_for_api)} messages), preview: {str(messages_for_api)[:300]}...")


        # --- Call the Language Model ---
        await manager.send_json(client_id, {"type": "status", "message": "Generating response..."})
        raw_response = ""
        try:
            llm_response_obj = await asyncio.to_thread(
                lambda: ollama_client.chat.completions.create(
                    model=config_dict.get("ollama_model", "llama3:latest"),
                    messages=messages_for_api,
                    temperature=config_dict.get("ollama_temperature", 0.05),
                    top_p=config_dict.get("ollama_top_p", 0.7)
                )
            )
            if not llm_response_obj or not llm_response_obj.choices or not llm_response_obj.choices[0].message or not llm_response_obj.choices[0].message.content:
                 logger.error("LLM returned an empty or invalid response object.")
                 raw_response = "The language model returned an empty response."
            else:
                 raw_response = llm_response_obj.choices[0].message.content.strip()
            logger.debug(f"Raw LLM response received for client {client_id}. Preview: '{raw_response[:100]}...'")

            # --- Post-processing and Saving ---

            if memory:
                try:
                    await asyncio.to_thread(memory.save_context, {"input": user_input}, {"output": raw_response})
                    logger.debug(f"Saved context to memory for client {client_id}.")
                except Exception as e_mem_save:
                     logger.error(f"Failed to save response to memory for {client_id}: {e_mem_save}", exc_info=True)


            processed_html_response = common_utils_module.clean_response_language(config_dict, raw_response)

            processed_html_response = re.sub(r'\s*\(Page\s+\d+\)', '', processed_html_response)
            processed_html_response = re.sub(r'\s*\[\d+\]', '', processed_html_response)
            processed_html_response = re.sub(r'\s*\(Figure\s+\d+\)', '', processed_html_response)


            logger.debug(f"Cleaned response for client {client_id}: '{processed_html_response[:100]}...'")

            try:
                # Pass the *final* context subset sent to the LLM for tracking
                common_utils_module.track_response_quality(config_dict, user_input, raw_response, relevant_ocr_chunks_for_llm, client_id)
                logger.debug(f"Tracked response quality for client {client_id}.")
            except Exception as e_track:
                logger.error(f"Failed to track response quality for {client_id}: {e_track}", exc_info=True)

            logger.info(f"Visual chat preparing SUCCESS return for {client_id}. Resp preview: '{processed_html_response[:50]}...', Context items: {len(relevant_ocr_chunks_for_llm)}")

            return processed_html_response, relevant_ocr_chunks_for_llm

        except Exception as e_llm_visual:
            logger.error(f"LLM call or post-processing error in visual chat for {client_id}: {e_llm_visual}", exc_info=True)
            error_msg_text = f"Error generating response from Language Model (Visual): {str(e_llm_visual)[:150]}."
            import traceback
            error_msg_text += f" Trace: {traceback.format_exc(limit=3).strip()}"


            if memory:
                try:
                    await asyncio.to_thread(memory.save_context, {"input": user_input}, {"output": error_msg_text})
                    logger.debug(f"Saved error message to memory for {client_id}.")
                except Exception as e_mem_save:
                     logger.error(f"Failed to save error message to memory for {client_id}: {e_mem_save}", exc_info=True)

            logger.error(f"Visual chat preparing ERROR return for {client_id}. Error message: '{error_msg_text[:50]}...'")

            return error_msg_text, []