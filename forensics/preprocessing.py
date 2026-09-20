"""
Forensics Pipeline - Preprocessing Module
==========================================
Handles image preprocessing for forensic analysis.
"""

import numpy as np
from PIL import Image
from typing import Tuple, Dict
import io
import logging

logger = logging.getLogger(__name__)


def preprocess_image(image_path: str, target_size: Tuple[int, int] = None) -> Dict:
    """
    Preprocess image for forensic analysis.
    
    Args:
        image_path: Path to input image
        target_size: Optional (width, height) for model input
        
    Returns:
        Dictionary with:
        - 'original': PIL Image (original resolution)
        - 'original_array': numpy array [H, W, 3] uint8
        - 'original_size': (width, height)
        - 'model_input': numpy array normalized for model
        - 'model_size': (width, height) of model input
    """
    # Load original
    original = Image.open(image_path).convert('RGB')
    original_size = original.size
    original_array = np.array(original)
    
    result = {
        'original': original,
        'original_array': original_array,
        'original_size': original_size,
    }
    
    if target_size is not None:
        model_input = original.resize(target_size, Image.BILINEAR)
        result['model_input'] = np.array(model_input)
        result['model_size'] = target_size
    else:
        result['model_input'] = original_array
        result['model_size'] = original_size
    
    return result


def compute_ela(image: Image.Image, quality: int = 90, scale: int = 15) -> Dict:
    """
    Compute Error Level Analysis.
    
    ELA detects regions with inconsistent compression by:
    1. Recompressing the image at a known JPEG quality
    2. Computing pixel-wise difference
    3. Scaling for visualization
    
    Args:
        image: PIL Image
        quality: JPEG recompression quality (default 90)
        scale: Visualization scale factor (default 15)
        
    Returns:
        Dictionary with:
        - 'ela_image': PIL Image (ELA visualization)
        - 'ela_array': numpy array [H, W, 3] uint8
        - 'difference_raw': raw difference [H, W, 3] float32
        - 'block_anomaly_score': float (higher = more anomalous)
    """
    # Recompress
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    recompressed = Image.open(buffer).convert('RGB')
    
    # Ensure same size
    if recompressed.size != image.size:
        recompressed = recompressed.resize(image.size, Image.BILINEAR)
    
    # Compute difference
    original_array = np.array(image).astype(np.float32)
    recompressed_array = np.array(recompressed).astype(np.float32)
    difference = np.abs(original_array - recompressed_array)
    
    # Scale for visualization
    ela_array = np.clip(difference * scale, 0, 255).astype(np.uint8)
    ela_image = Image.fromarray(ela_array)
    
    # Block-level anomaly analysis
    block_size = 16
    h, w = difference.shape[:2]
    block_means = []
    
    for y in range(0, h - block_size, block_size):
        for x in range(0, w - block_size, block_size):
            block = difference[y:y+block_size, x:x+block_size]
            block_means.append(np.mean(block))
    
    block_means = np.array(block_means)
    block_anomaly_score = float(np.std(block_means) / (np.mean(block_means) + 1e-8))
    
    return {
        'ela_image': ela_image,
        'ela_array': ela_array,
        'difference_raw': difference,
        'block_anomaly_score': block_anomaly_score,
        'ela_mean': float(np.mean(difference)),
        'ela_max': float(np.max(difference)),
    }


def normalize_for_model(image_array: np.ndarray) -> np.ndarray:
    """
    Normalize image array for model input.
    
    Args:
        image_array: [H, W, 3] uint8 or float32
        
    Returns:
        Normalized array [H, W, 3] float32
    """
    if image_array.dtype == np.uint8:
        image_array = image_array.astype(np.float32) / 255.0
    
    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    normalized = (image_array - mean) / std
    
    return normalized
