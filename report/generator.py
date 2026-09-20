"""
Report Generator Module
========================
Generates forensic analysis reports in text and PDF formats.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def generate_forensic_report(analysis_result: Dict) -> str:
    """
    Generate a comprehensive forensic analysis report.
    
    Args:
        analysis_result: Complete analysis result dictionary
        
    Returns:
        Formatted text report
    """
    report = []
    
    # Header
    report.append("=" * 70)
    report.append("FORENSIC ANALYSIS REPORT")
    report.append("Image Tampering Detection & Restoration System")
    report.append("=" * 70)
    report.append("")
    
    # Analysis metadata
    report.append("ANALYSIS INFORMATION")
    report.append("-" * 40)
    report.append(f"Analysis ID: {analysis_result.get('id', 'N/A')}")
    report.append(f"Timestamp: {analysis_result.get('timestamp', 'N/A')}")
    report.append(f"Filename: {analysis_result.get('filename', 'N/A')}")
    report.append(f"Image Size: {analysis_result.get('image_size', 'N/A')}")
    report.append("")
    
    # Verdict
    verdict_info = analysis_result.get('verdict', {})
    report.append("DETECTION RESULT")
    report.append("-" * 40)
    report.append(f"Verdict: {verdict_info.get('verdict', 'N/A')}")
    report.append(f"Description: {verdict_info.get('verdict_description', 'N/A')}")
    report.append(f"Confidence Level: {verdict_info.get('confidence_level', 'N/A')}")
    report.append(f"Overall Confidence: {verdict_info.get('overall_confidence', 0):.1%}")
    report.append(f"Integrity Score: {verdict_info.get('integrity_score', 0):.3f}")
    report.append(f"Number of Regions: {verdict_info.get('num_regions', 0)}")
    report.append(f"Tampered Area: {verdict_info.get('tampered_percentage', 0):.2f}%")
    report.append("")
    
    # Model information
    report.append("MODEL INFORMATION")
    report.append("-" * 40)
    report.append(f"Model Used: {verdict_info.get('model_used', 'N/A')}")
    report.append(f"Processing Time: {analysis_result.get('processing_time', 0):.2f}s")
    report.append("")
    
    # Detected regions
    regions = analysis_result.get('regions', [])
    if regions:
        report.append("DETECTED REGIONS")
        report.append("-" * 40)
        for i, region in enumerate(regions, 1):
            report.append(f"\nRegion {i}:")
            report.append(f"  Severity: {region['severity'].upper()}")
            report.append(f"  Bounding Box: x={region['bbox']['x']}, y={region['bbox']['y']}, "
                         f"w={region['bbox']['width']}, h={region['bbox']['height']}")
            report.append(f"  Centroid: ({region['centroid']['x']}, {region['centroid']['y']})")
            report.append(f"  Area: {region['area_pixels']} pixels ({region['area_percentage']:.2f}%)")
            report.append(f"  Mean Probability: {region['mean_probability']:.1%}")
            report.append(f"  Mean Confidence: {region['mean_confidence']:.1%}")
            report.append(f"  Region Confidence: {region['region_confidence']:.1%}")
            report.append(f"  Description: {region['description']}")
    else:
        report.append("DETECTED REGIONS")
        report.append("-" * 40)
        report.append("No tampered regions detected.")
    report.append("")
    
    # Forensic signals
    signals = analysis_result.get('forensic_signals', [])
    if signals:
        report.append("FORENSIC SIGNALS")
        report.append("-" * 40)
        for signal in signals:
            report.append(f"• {signal['name']} [{signal['severity'].upper()}]")
            report.append(f"  {signal['description']}")
            report.append(f"  Evidence: {signal['evidence']}")
            report.append("")
    
    # Generated outputs
    report.append("GENERATED OUTPUTS")
    report.append("-" * 40)
    outputs = [
        'original_image', 'localization_map', 'confidence_map',
        'binary_mask', 'heatmap_overlay', 'contour_overlay',
        'restored_image'
    ]
    for output in outputs:
        status = "✓ Generated" if analysis_result.get(output) else "✗ Not generated"
        report.append(f"  {output.replace('_', ' ').title()}: {status}")
    report.append("")
    
    # Severity summary
    severity_summary = verdict_info.get('severity_summary', {})
    if any(severity_summary.values()):
        report.append("SEVERITY SUMMARY")
        report.append("-" * 40)
        for severity, count in severity_summary.items():
            if count > 0:
                report.append(f"  {severity.upper()}: {count} region(s)")
        report.append("")
    
    # Warning
    report.append("IMPORTANT WARNING")
    report.append("-" * 40)
    report.append(verdict_info.get('warning', 
        "AI forensic detection is probabilistic. Results should be considered "
        "as supporting evidence, not absolute proof."
    ))
    report.append("")
    
    # Limitations
    report.append("LIMITATIONS")
    report.append("-" * 40)
    report.append("1. This analysis provides forensic evidence, not absolute proof.")
    report.append("2. The restoration is a reconstruction, not recovery of original pixels.")
    report.append("3. Sophisticated forgeries may evade detection.")
    report.append("4. Heavily compressed or unusual images may trigger false positives.")
    report.append("5. Results should be combined with other investigation methods.")
    report.append("")
    
    # Footer
    report.append("=" * 70)
    report.append(f"Report generated: {datetime.now().isoformat()}")
    report.append("System: ImageGuard - Image Tampering Detection & Restoration")
    report.append("=" * 70)
    
    return "\n".join(report)


def save_report(report_text: str, output_path: str) -> str:
    """Save report to file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    return str(path)


def generate_json_report(analysis_result: Dict) -> str:
    """Generate JSON-formatted report."""
    return json.dumps(analysis_result, indent=2, default=str)
