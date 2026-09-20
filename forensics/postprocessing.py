"""
Forensics Pipeline - Postprocessing Module
============================================
Handles post-processing of localization maps to generate accurate masks.

Pipeline:
1. Confidence-weighted thresholding
2. Morphological cleanup (close + open)
3. Connected component analysis
4. Small component removal (noise filtering)
5. Mask refinement
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def postprocess_localization(
    localization_map: np.ndarray,
    confidence_map: np.ndarray,
    original_size: Tuple[int, int],
    min_region_area: int = 100,
    confidence_threshold: float = 0.4,
    probability_threshold: float = 0.5,
) -> Dict:
    """
    Post-process localization map to generate accurate tamper mask.
    
    This function:
    1. Applies confidence-weighted adaptive thresholding
    2. Performs morphological cleanup
    3. Extracts connected components
    4. Filters noise (tiny isolated regions)
    5. Preserves real small manipulated regions
    6. Returns binary mask aligned to original image dimensions
    
    Args:
        localization_map: Pixel-level tamper probability [H, W] float32
        confidence_map: Reliability map [H, W] float32
        original_size: (width, height) of original image
        min_region_area: Minimum area (pixels) for a valid region
        confidence_threshold: Minimum confidence for accepting prediction
        probability_threshold: Base probability threshold
        
    Returns:
        Dictionary with:
        - 'binary_mask': [H, W] uint8 (0 or 255)
        - 'refined_mask': [H, W] uint8 with cleaned regions
        - 'components': list of component dictionaries
        - 'num_regions': number of detected regions
        - 'tampered_percentage': percentage of tampered pixels
        - 'threshold_used': actual threshold applied
    """
    h, w = localization_map.shape
    orig_w, orig_h = original_size
    
    # Ensure maps match original dimensions
    if (w, h) != original_size:
        from PIL import Image
        loc_img = Image.fromarray((localization_map * 255).astype(np.uint8))
        loc_img = loc_img.resize(original_size, Image.BILINEAR)
        localization_map = np.array(loc_img).astype(np.float32) / 255.0
        
        conf_img = Image.fromarray((confidence_map * 255).astype(np.uint8))
        conf_img = conf_img.resize(original_size, Image.BILINEAR)
        confidence_map = np.array(conf_img).astype(np.float32) / 255.0
    
    # Step 1: Confidence-weighted adaptive thresholding
    # Higher confidence areas use lower threshold (more sensitive)
    # Lower confidence areas use higher threshold (more conservative)
    adaptive_threshold = probability_threshold + (1 - confidence_map) * 0.3
    adaptive_threshold = np.clip(adaptive_threshold, probability_threshold, 0.85)
    
    # Apply threshold
    binary_mask = (localization_map > adaptive_threshold).astype(np.uint8) * 255
    
    # Step 2: Also suppress low-confidence predictions
    low_conf_mask = confidence_map < confidence_threshold
    binary_mask[low_conf_mask] = 0
    
    # Step 3: Morphological cleanup
    refined_mask = _morphological_cleanup(binary_mask)
    
    # Step 4: Connected component analysis
    components = _extract_connected_components(refined_mask)
    
    # Step 5: Filter small components (noise removal)
    # But preserve genuinely small manipulated regions
    filtered_components = []
    filtered_mask = np.zeros_like(refined_mask)
    
    for comp in components:
        if comp['area'] >= min_region_area:
            filtered_components.append(comp)
            # Add this component to filtered mask
            y_coords, x_coords = comp['coordinates']
            filtered_mask[y_coords, x_coords] = 255
        elif comp['area'] >= min_region_area // 4:
            # Keep very small regions only if they have high confidence
            region_conf = np.mean(confidence_map[comp['coordinates']])
            if region_conf > 0.6:
                filtered_components.append(comp)
                y_coords, x_coords = comp['coordinates']
                filtered_mask[y_coords, x_coords] = 255
    
    # Calculate statistics
    total_pixels = orig_h * orig_w
    tampered_pixels = np.sum(filtered_mask > 0)
    tampered_percentage = (tampered_pixels / total_pixels) * 100
    
    return {
        'binary_mask': binary_mask,
        'refined_mask': filtered_mask,
        'components': filtered_components,
        'num_regions': len(filtered_components),
        'tampered_percentage': float(tampered_percentage),
        'threshold_used': float(np.mean(adaptive_threshold)),
        'tampered_pixels': int(tampered_pixels),
        'total_pixels': int(total_pixels),
    }


def _morphological_cleanup(mask: np.ndarray) -> np.ndarray:
    """
    Apply morphological operations to clean up the binary mask.
    
    Operations:
    1. Close: fill small holes in detected regions
    2. Open: remove small noise pixels
    3. Dilate slightly: ensure connected regions
    """
    try:
        import cv2
        
        # Define structuring elements
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        # Close: fill holes
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        
        # Open: remove noise
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel_open)
        
        # Slight dilation to connect nearby regions
        cleaned = cv2.dilate(cleaned, kernel_dilate, iterations=1)
        
        return cleaned
        
    except ImportError:
        logger.warning("OpenCV not available, using numpy-based morphological ops")
        return _numpy_morphological_cleanup(mask)


def _numpy_morphological_cleanup(mask: np.ndarray) -> np.ndarray:
    """Fallback morphological cleanup using numpy."""
    from scipy import ndimage
    
    # Binary closing
    struct = ndimage.generate_binary_structure(2, 2)
    cleaned = ndimage.binary_closing(mask > 0, structure=struct, iterations=2)
    
    # Binary opening
    cleaned = ndimage.binary_opening(cleaned, structure=struct, iterations=1)
    
    return (cleaned.astype(np.uint8)) * 255


def _extract_connected_components(mask: np.ndarray) -> List[Dict]:
    """
    Extract connected components from binary mask.
    
    Returns list of component dictionaries with:
    - 'label': component ID
    - 'area': number of pixels
    - 'bbox': (x, y, w, h) bounding box
    - 'coordinates': (y_coords, x_coords) tuple of pixel coordinates
    - 'centroid': (cx, cy) center of mass
    - 'mean_probability': mean localization value in this region
    """
    try:
        import cv2
        
        # Find contours
        contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        components = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < 1:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            
            # Get pixel coordinates
            component_mask = np.zeros_like(mask)
            cv2.drawContours(component_mask, [contour], -1, 255, -1)
            coords = np.where(component_mask > 0)
            
            # Centroid
            M = cv2.moments(contour)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
            else:
                cx, cy = x + w // 2, y + h // 2
            
            components.append({
                'label': i,
                'area': int(area),
                'bbox': (int(x), int(y), int(w), int(h)),
                'coordinates': (coords[0], coords[1]),
                'centroid': (int(cx), int(cy)),
            })
        
        return components
        
    except ImportError:
        logger.warning("OpenCV not available, using scipy for connected components")
        return _scipy_connected_components(mask)


def _scipy_connected_components(mask: np.ndarray) -> List[Dict]:
    """Fallback connected component extraction using scipy."""
    from scipy import ndimage
    
    binary = mask > 0
    labeled, num_features = ndimage.label(binary)
    
    components = []
    for i in range(1, num_features + 1):
        component = labeled == i
        coords = np.where(component)
        area = int(np.sum(component))
        
        if area < 1:
            continue
        
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        
        cx = int(np.mean(coords[1]))
        cy = int(np.mean(coords[0]))
        
        components.append({
            'label': i,
            'area': area,
            'bbox': (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)),
            'coordinates': (coords[0], coords[1]),
            'centroid': (cx, cy),
        })
    
    return components


def generate_heatmap(
    localization_map: np.ndarray,
    original_size: Tuple[int, int],
    alpha: float = 0.6
) -> np.ndarray:
    """
    Generate a color heatmap from the localization map.
    
    Uses jet colormap: blue (low) -> green -> yellow -> red (high)
    
    Args:
        localization_map: [H, W] float32 in [0, 1]
        original_size: (width, height)
        alpha: overlay transparency
        
    Returns:
        RGBA heatmap image [H, W, 4] uint8
    """
    h, w = localization_map.shape
    orig_w, orig_h = original_size
    
    # Resize if needed
    if (w, h) != original_size:
        from PIL import Image
        loc_img = Image.fromarray((localization_map * 255).astype(np.uint8))
        loc_img = loc_img.resize(original_size, Image.BILINEAR)
        localization_map = np.array(loc_img).astype(np.float32) / 255.0
    
    # Create jet colormap
    heatmap = np.zeros((orig_h, orig_w, 4), dtype=np.uint8)
    
    for y in range(orig_h):
        for x in range(orig_w):
            val = localization_map[y, x]
            if val > 0.05:
                r, g, b = _jet_colormap(val)
                heatmap[y, x] = [r, g, b, int(val * 255 * alpha)]
    
    return heatmap


def _jet_colormap(value: float) -> Tuple[int, int, int]:
    """Convert a value [0, 1] to jet colormap RGB."""
    value = np.clip(value, 0, 1)
    
    if value < 0.125:
        r, g, b = 0, 0, int(128 + value * 8 * 127)
    elif value < 0.375:
        t = (value - 0.125) * 4
        r, g, b = 0, int(t * 255), 255
    elif value < 0.625:
        t = (value - 0.375) * 4
        r, g, b = int(t * 255), 255, int((1 - t) * 255)
    elif value < 0.875:
        t = (value - 0.625) * 4
        r, g, b = 255, int((1 - t) * 255), 0
    else:
        t = (value - 0.875) * 4
        r, g, b = 255, 0, int(t * 128)
    
    return int(r), int(g), int(b)


def generate_overlay(
    original_array: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.5
) -> np.ndarray:
    """
    Generate overlay of mask on original image.
    
    Args:
        original_array: [H, W, 3] uint8
        mask: [H, W] uint8 (0 or 255)
        color: overlay color (R, G, B)
        alpha: transparency
        
    Returns:
        Overlay image [H, W, 3] uint8
    """
    overlay = original_array.copy()
    mask_bool = mask > 128
    
    overlay[mask_bool] = (
        overlay[mask_bool] * (1 - alpha) + 
        np.array(color) * alpha
    ).astype(np.uint8)
    
    return overlay


def generate_contour_overlay(
    original_array: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    """
    Generate contour overlay on original image.
    
    Args:
        original_array: [H, W, 3] uint8
        mask: [H, W] uint8
        color: contour color (R, G, B)
        thickness: contour line thickness
        
    Returns:
        Image with contours [H, W, 3] uint8
    """
    result = original_array.copy()
    
    try:
        import cv2
        
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(result, contours, -1, color, thickness)
        
    except ImportError:
        # Fallback: use edge detection
        from scipy import ndimage
        edges = ndimage.sobel(mask.astype(np.float32))
        edge_mask = edges > 0.5
        result[edge_mask] = color
    
    return result
