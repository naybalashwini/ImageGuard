"""
utils.py - Visualization and Utility Functions for Forensic Image Analysis

This module provides forensic-grade visualization utilities for highlighting
tampered regions and computing chain-of-custody metadata.

Forensic Rationale:
- Red overlay is used because it provides maximum contrast against most
  natural image content and is the standard color for marking anomalies
  in forensic documentation.
- Semi-transparent blending preserves underlying image details for expert review.
- SHA-256 hashing ensures file integrity verification for legal proceedings.
"""

import cv2
import numpy as np
import hashlib
from pathlib import Path
from typing import Tuple, Optional, Union
from PIL import Image


def compute_image_hash(image_path: Union[str, Path]) -> str:
    """
    Compute SHA-256 hash of an image file for chain-of-custody verification.
    
    Forensic Rationale:
    SHA-256 provides a cryptographically secure fingerprint of the original
    evidence file. This hash can be used to verify that the image has not
    been altered during the analysis process, which is critical for legal
    admissibility.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Hexadecimal string of the SHA-256 hash
    """
    sha256_hash = hashlib.sha256()
    
    with open(image_path, "rb") as f:
        # Read in chunks to handle large files efficiently
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    
    return sha256_hash.hexdigest()


def compute_hash_from_bytes(image_bytes: bytes) -> str:
    """
    Compute SHA-256 hash from image bytes (for uploaded files in Streamlit).
    
    Args:
        image_bytes: Raw bytes of the image file
        
    Returns:
        Hexadecimal string of the SHA-256 hash
    """
    return hashlib.sha256(image_bytes).hexdigest()


def create_red_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.5,
    red_color: Tuple[int, int, int] = (0, 0, 255)
) -> np.ndarray:
    """
    Create a semi-transparent red overlay on tampered regions.
    
    Forensic Rationale:
    The overlay uses alpha blending to highlight suspicious regions while
    preserving the underlying image content. This allows forensic experts
    to correlate the detected anomalies with visual features in the original
    image. The red color (BGR: 0,0,255) is chosen for maximum visibility
    and adherence to forensic documentation standards.
    
    Args:
        image: Input image as numpy array (H, W, C) in BGR format (OpenCV default)
        mask: Binary mask where 1 indicates tampered regions (H, W) or (H, W, 1)
        alpha: Transparency factor for the overlay (0.0 = invisible, 1.0 = opaque)
               Recommended: 0.4-0.6 for forensic review
        red_color: BGR color tuple for the overlay (default: pure red)
        
    Returns:
        Image with red overlay on tampered regions (same shape as input)
    """
    # Ensure image is in BGR format (OpenCV default)
    if image.ndim == 2:
        # Grayscale to BGR
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        # BGRA to BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    
    # Normalize mask to [0, 1] range and ensure 2D
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    
    mask_float = mask.astype(np.float32) / 255.0 if mask.max() > 1 else mask.astype(np.float32)
    
    # Create overlay copy
    overlay = image.copy().astype(np.float32)
    
    # Create red layer (BGR format)
    red_layer = np.zeros_like(overlay, dtype=np.float32)
    red_layer[:, :, 0] = red_color[0]  # Blue channel
    red_layer[:, :, 1] = red_color[1]  # Green channel
    red_layer[:, :, 2] = red_color[2]  # Red channel
    
    # Apply alpha blending only on masked regions
    # Formula: output = image * (1 - alpha*mask) + red * (alpha*mask)
    mask_expanded = np.expand_dims(mask_float, axis=2)
    blend_factor = alpha * mask_expanded
    
    result = overlay * (1 - blend_factor) + red_layer * blend_factor
    
    # Clip to valid range and convert back to uint8
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return result


def threshold_confidence_map(
    confidence_map: np.ndarray,
    threshold: float = 0.5,
    adaptive: bool = False,
    block_size: int = 11,
    c_value: int = 2
) -> np.ndarray:
    """
    Convert a confidence map to a binary tampering mask.
    
    Forensic Rationale:
    Thresholding converts probabilistic detector outputs into definitive
    tampered/pristine classifications. Adaptive thresholding can handle
    varying confidence levels across different image regions, which is
    useful when dealing with images that have undergone multiple types
    of manipulation.
    
    Args:
        confidence_map: Float array with values in [0, 1] indicating tampering probability
        threshold: Global threshold value (0.0-1.0) for fixed thresholding
        adaptive: If True, use adaptive thresholding instead of global
        block_size: Size of neighborhood for adaptive thresholding (must be odd)
        c_value: Constant subtracted from mean in adaptive thresholding
        
    Returns:
        Binary mask (uint8) where 255 indicates tampered, 0 indicates pristine
    """
    # Ensure confidence map is in [0, 1] range
    conf_normalized = confidence_map.copy()
    if conf_normalized.max() > 1.0:
        conf_normalized = conf_normalized / 255.0
    conf_normalized = np.clip(conf_normalized, 0, 1)
    
    # Convert to 8-bit for OpenCV operations
    conf_8bit = (conf_normalized * 255).astype(np.uint8)
    
    if adaptive:
        # Adaptive thresholding for varying confidence levels
        # Uses Gaussian-weighted sum of neighborhood values
        binary_mask = cv2.adaptiveThreshold(
            conf_8bit,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            c_value
        )
    else:
        # Global thresholding
        _, binary_mask = cv2.threshold(
            conf_8bit,
            int(threshold * 255),
            255,
            cv2.THRESH_BINARY
        )
    
    return binary_mask


def calculate_tampered_percentage(mask: np.ndarray) -> float:
    """
    Calculate the percentage of the image detected as tampered.
    
    Forensic Rationale:
    This metric provides a quantitative measure of the extent of tampering,
    which can be useful for prioritizing cases and providing expert testimony
    about the severity of image manipulation.
    
    Args:
        mask: Binary mask where non-zero values indicate tampered regions
        
    Returns:
        Percentage of image area detected as tampered (0.0-100.0)
    """
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    
    total_pixels = mask.size
    tampered_pixels = cv2.countNonZero(mask)
    
    return (tampered_pixels / total_pixels) * 100.0


def resize_image_for_processing(
    image: np.ndarray,
    max_dimension: int = 1024,
    maintain_aspect_ratio: bool = True
) -> Tuple[np.ndarray, Tuple[float, float]]:
    """
    Resize large images to fit within GPU memory constraints.
    
    Forensic Rationale:
    Large images may exceed GPU memory limits. This function provides
    controlled downsampling while maintaining aspect ratio to prevent
    distortion that could affect detection accuracy. The scale factors
    are returned to allow mapping results back to original coordinates.
    
    Args:
        image: Input image as numpy array
        max_dimension: Maximum allowed dimension (width or height)
        maintain_aspect_ratio: If True, preserve aspect ratio during resize
        
    Returns:
        Tuple of (resized_image, scale_factors) where scale_factors is (scale_x, scale_y)
    """
    height, width = image.shape[:2]
    
    if max(height, width) <= max_dimension:
        return image, (1.0, 1.0)
    
    if maintain_aspect_ratio:
        scale = max_dimension / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
    else:
        scale_x = max_dimension / width
        scale_y = max_dimension / height
        scale = min(scale_x, scale_y)
        new_width = int(width * scale)
        new_height = int(height * scale)
    
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    return resized, (scale, scale)


def load_image_to_rgb(image_input: Union[str, Path, np.ndarray, Image.Image]) -> np.ndarray:
    """
    Load an image from various sources and convert to RGB numpy array.
    
    Args:
        image_input: Can be a file path, numpy array, or PIL Image
        
    Returns:
        RGB image as numpy array (H, W, 3) with values in [0, 255]
    """
    if isinstance(image_input, (str, Path)):
        # Load from file using OpenCV (returns BGR)
        image = cv2.imread(str(image_input))
        if image is None:
            raise ValueError(f"Could not load image from {image_input}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    elif isinstance(image_input, np.ndarray):
        # Already a numpy array
        if image_input.ndim == 2:
            return cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
        elif image_input.shape[2] == 4:
            return cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB)
        elif image_input.shape[2] == 3:
            return image_input.copy()
        else:
            raise ValueError(f"Invalid image shape: {image_input.shape}")
    
    elif isinstance(image_input, Image.Image):
        # PIL Image
        return np.array(image_input.convert('RGB'))
    
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")


def save_forensic_report(
    output_path: Union[str, Path],
    original_hash: str,
    image_dimensions: Tuple[int, int],
    tampered_percentage: float,
    detection_threshold: float,
    model_name: str,
    additional_notes: Optional[str] = None
) -> None:
    """
    Save a forensic analysis report to a text file.
    
    Forensic Rationale:
    Documentation is critical for maintaining chain of custody and providing
    reproducible analysis. This report captures all essential metadata about
    the analysis performed.
    
    Args:
        output_path: Path to save the report
        original_hash: SHA-256 hash of the original image
        image_dimensions: (width, height) of the image
        tampered_percentage: Percentage of image detected as tampered
        detection_threshold: Threshold used for binary mask generation
        model_name: Name/version of the detection model used
        additional_notes: Optional notes from the analyst
    """
    from datetime import datetime
    
    report = f"""
================================================================================
                    FORENSIC IMAGE ANALYSIS REPORT
================================================================================

Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}

--------------------------------------------------------------------------------
EVIDENCE INFORMATION
--------------------------------------------------------------------------------
Original File SHA-256 Hash: {original_hash}
Image Dimensions: {image_dimensions[0]} x {image_dimensions[1]} pixels

--------------------------------------------------------------------------------
ANALYSIS PARAMETERS
--------------------------------------------------------------------------------
Detection Model: {model_name}
Confidence Threshold: {detection_threshold}

--------------------------------------------------------------------------------
FINDINGS
--------------------------------------------------------------------------------
Tampered Area Detected: {tampered_percentage:.2f}% of total image area

--------------------------------------------------------------------------------
ANALYST NOTES
--------------------------------------------------------------------------------
{additional_notes if additional_notes else "No additional notes provided."}

================================================================================
                         END OF FORENSIC REPORT
================================================================================
"""
    
    with open(output_path, 'w') as f:
        f.write(report)
