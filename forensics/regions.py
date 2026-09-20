"""
Forensics Pipeline - Regions Module
=====================================
Handles tampered region extraction, analysis, and bounding box generation.

Key principle: Bounding boxes are ALWAYS derived from the actual pixel mask.
No manual estimation or hard-coded regions.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def extract_regions(
    components: List[Dict],
    localization_map: np.ndarray,
    confidence_map: np.ndarray,
    original_size: Tuple[int, int],
) -> List[Dict]:
    """
    Extract detailed region information from connected components.
    
    For each detected region:
    - Bounding box (from actual mask, not estimated)
    - Region confidence score
    - Area in pixels and percentage
    - Severity classification
    - Centroid coordinates
    
    Args:
        components: List of connected component dictionaries
        localization_map: [H, W] float32 tamper probability
        confidence_map: [H, W] float32 reliability map
        original_size: (width, height)
        
    Returns:
        List of region dictionaries with full analysis
    """
    orig_w, orig_h = original_size
    total_pixels = orig_w * orig_h
    
    regions = []
    
    for comp in components:
        y_coords, x_coords = comp['coordinates']
        
        if len(y_coords) == 0:
            continue
        
        # Bounding box from actual mask pixels
        x_min, x_max = int(x_coords.min()), int(x_coords.max())
        y_min, y_max = int(y_coords.min()), int(y_coords.max())
        bbox_w = x_max - x_min
        bbox_h = y_max - y_min
        
        # Region statistics
        region_area = comp['area']
        region_percentage = (region_area / total_pixels) * 100
        
        # Mean probability in this region
        region_probs = localization_map[y_coords, x_coords]
        mean_prob = float(np.mean(region_probs))
        max_prob = float(np.max(region_probs))
        
        # Mean confidence in this region
        region_confs = confidence_map[y_coords, x_coords]
        mean_conf = float(np.mean(region_confs))
        
        # Region confidence score (combines probability and model confidence)
        region_confidence = mean_prob * mean_conf
        
        # Severity classification
        severity = _classify_severity(region_confidence, region_percentage, mean_prob)
        
        # Centroid
        cx = int(np.mean(x_coords))
        cy = int(np.mean(y_coords))
        
        region = {
            'id': comp['label'],
            'bbox': {
                'x': x_min,
                'y': y_min,
                'width': bbox_w,
                'height': bbox_h,
            },
            'centroid': {'x': cx, 'y': cy},
            'area_pixels': region_area,
            'area_percentage': float(region_percentage),
            'mean_probability': mean_prob,
            'max_probability': max_prob,
            'mean_confidence': mean_conf,
            'region_confidence': float(region_confidence),
            'severity': severity,
            'description': _generate_region_description(severity, mean_prob, mean_conf, region_percentage),
        }
        
        regions.append(region)
    
    # Sort by confidence (most suspicious first)
    regions.sort(key=lambda r: r['region_confidence'], reverse=True)
    
    return regions


def _classify_severity(
    region_confidence: float,
    region_percentage: float,
    mean_probability: float
) -> str:
    """
    Classify region severity based on multiple factors.
    
    Returns: 'critical', 'high', 'medium', 'low', or 'negligible'
    """
    if region_confidence > 0.7 and mean_probability > 0.7:
        return 'critical'
    elif region_confidence > 0.5 and mean_probability > 0.5:
        return 'high'
    elif region_confidence > 0.3 and mean_probability > 0.4:
        return 'medium'
    elif region_confidence > 0.15:
        return 'low'
    else:
        return 'negligible'


def _generate_region_description(
    severity: str,
    mean_prob: float,
    mean_conf: float,
    area_pct: float
) -> str:
    """Generate human-readable description for a region."""
    
    if severity == 'critical':
        return (
            f"High-confidence manipulation detected. "
            f"Probability: {mean_prob:.1%}, Confidence: {mean_conf:.1%}, "
            f"Area: {area_pct:.2f}% of image."
        )
    elif severity == 'high':
        return (
            f"Likely manipulated region. "
            f"Probability: {mean_prob:.1%}, Confidence: {mean_conf:.1%}, "
            f"Area: {area_pct:.2f}% of image."
        )
    elif severity == 'medium':
        return (
            f"Possible manipulation detected. "
            f"Probability: {mean_prob:.1%}, Confidence: {mean_conf:.1%}, "
            f"Area: {area_pct:.2f}% of image."
        )
    elif severity == 'low':
        return (
            f"Low-confidence anomaly. May be compression artifact or noise. "
            f"Probability: {mean_prob:.1%}, Confidence: {mean_conf:.1%}, "
            f"Area: {area_pct:.2f}% of image."
        )
    else:
        return (
            f"Negligible anomaly. Likely not a manipulation. "
            f"Probability: {mean_prob:.1%}, Confidence: {mean_conf:.1%}."
        )


def compute_overall_verdict(
    regions: List[Dict],
    score: float,
    tampered_percentage: float,
    model_name: str
) -> Dict:
    """
    Compute overall forensic verdict.
    
    Args:
        regions: List of detected regions
        score: Image-level integrity score from model
        tampered_percentage: Total tampered area percentage
        model_name: Name of the model used
        
    Returns:
        Dictionary with verdict information
    """
    # Determine verdict
    if len(regions) == 0 and tampered_percentage < 1.0:
        verdict = "AUTHENTIC"
        verdict_description = "No reliable tampered region detected"
        confidence_level = "high"
    elif len(regions) == 0:
        verdict = "UNCERTAIN"
        verdict_description = "Low-level anomalies detected but no confident manipulation region"
        confidence_level = "low"
    else:
        # Check if any region has high confidence
        max_region_conf = max(r['region_confidence'] for r in regions)
        
        if max_region_conf > 0.6:
            verdict = "TAMPERED"
            verdict_description = "Manipulation detected with high confidence"
            confidence_level = "high"
        elif max_region_conf > 0.3:
            verdict = "LIKELY_TAMPERED"
            verdict_description = "Probable manipulation detected"
            confidence_level = "medium"
        else:
            verdict = "SUSPICIOUS"
            verdict_description = "Low-confidence anomalies detected. Manual review recommended."
            confidence_level = "low"
    
    # Overall confidence
    if len(regions) > 0:
        overall_confidence = float(np.mean([r['region_confidence'] for r in regions]))
    else:
        overall_confidence = 1.0 - score  # Invert: low score = high confidence it's authentic
    
    # Severity summary
    severity_counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0, 'negligible': 0}
    for r in regions:
        severity_counts[r['severity']] += 1
    
    return {
        'verdict': verdict,
        'verdict_description': verdict_description,
        'confidence_level': confidence_level,
        'overall_confidence': float(overall_confidence),
        'integrity_score': float(score),
        'num_regions': len(regions),
        'tampered_percentage': float(tampered_percentage),
        'model_used': model_name,
        'severity_summary': severity_counts,
        'warning': (
            "AI forensic detection is probabilistic. Results should be considered "
            "as supporting evidence, not absolute proof. Sophisticated forgeries "
            "may evade detection, and unusual images may trigger false positives."
        ),
    }


def format_bounding_boxes_for_visualization(
    regions: List[Dict],
    original_size: Tuple[int, int]
) -> List[Dict]:
    """
    Format bounding boxes for frontend visualization.
    
    Converts pixel coordinates to percentage-based coordinates
    for responsive display.
    
    Args:
        regions: List of region dictionaries
        original_size: (width, height)
        
    Returns:
        List of formatted bounding box dictionaries
    """
    orig_w, orig_h = original_size
    formatted = []
    
    for region in regions:
        bbox = region['bbox']
        
        formatted.append({
            'id': region['id'],
            'x_pct': (bbox['x'] / orig_w) * 100,
            'y_pct': (bbox['y'] / orig_h) * 100,
            'width_pct': (bbox['width'] / orig_w) * 100,
            'height_pct': (bbox['height'] / orig_h) * 100,
            'x_px': bbox['x'],
            'y_px': bbox['y'],
            'width_px': bbox['width'],
            'height_px': bbox['height'],
            'confidence': region['region_confidence'],
            'severity': region['severity'],
            'description': region['description'],
        })
    
    return formatted
