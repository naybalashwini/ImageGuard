"""
Report Generator

Generates professional PDF/text forensic reports from analysis results.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional


def generate_text_report(analysis_data: Dict) -> str:
    """Generate a text-based forensic report."""
    
    report = []
    report.append("=" * 70)
    report.append("FORENSIC ANALYSIS REPORT")
    report.append("Image Tampering Detection & Restoration System")
    report.append("=" * 70)
    report.append("")
    
    # Analysis metadata
    report.append("ANALYSIS INFORMATION")
    report.append("-" * 40)
    report.append(f"Analysis ID: {analysis_data.get('id', 'N/A')}")
    report.append(f"Timestamp: {analysis_data.get('timestamp', 'N/A')}")
    report.append(f"Filename: {analysis_data.get('filename', 'N/A')}")
    report.append(f"Status: {analysis_data.get('status', 'N/A')}")
    report.append("")
    
    # Detection result
    report.append("DETECTION RESULT")
    report.append("-" * 40)
    is_tampered = analysis_data.get('is_tampered', False)
    report.append(f"Verdict: {'TAMPERED - Manipulation Detected' if is_tampered else 'AUTHENTIC - No Strong Evidence of Tampering'}")
    report.append(f"Confidence: {analysis_data.get('confidence', 0):.1%}")
    report.append(f"Tampered Region: {analysis_data.get('tampered_percentage', 0):.2f}% of image area")
    report.append(f"Suspicious Regions: {analysis_data.get('num_regions', 'N/A')}")
    report.append("")
    
    # Model information
    model_info = analysis_data.get('model_info', {})
    report.append("MODEL INFORMATION")
    report.append("-" * 40)
    report.append(f"Model: {model_info.get('model', 'N/A')}")
    report.append(f"Architecture: {model_info.get('backbone', 'N/A')}")
    report.append(f"Device: {model_info.get('device', 'N/A')}")
    report.append(f"Processing Time: {analysis_data.get('processing_time', 0):.2f}s")
    report.append("")
    
    # Forensic signals
    signals = analysis_data.get('forensic_signals', [])
    report.append("FORENSIC SIGNALS")
    report.append("-" * 40)
    if signals:
        for i, signal in enumerate(signals, 1):
            report.append(f"\n{i}. {signal.get('name', 'Unknown')} [{signal.get('severity', 'unknown').upper()}]")
            report.append(f"   Description: {signal.get('description', 'N/A')}")
            report.append(f"   Evidence: {signal.get('evidence', 'N/A')}")
    else:
        report.append("No significant forensic signals detected.")
    report.append("")
    
    # Metrics
    metrics = analysis_data.get('metrics', {})
    if metrics:
        report.append("LOCALIZATION METRICS")
        report.append("-" * 40)
        for key, value in metrics.items():
            if value is not None:
                report.append(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
        report.append("")
    
    # Generated outputs
    report.append("GENERATED OUTPUTS")
    report.append("-" * 40)
    outputs = ['heatmap', 'mask', 'overlay', 'ela', 'restored']
    for output in outputs:
        path = analysis_data.get(f'{output}_path', None)
        status = "✓ Generated" if path else "✗ Not generated"
        report.append(f"  {output.capitalize()}: {status}")
    report.append("")
    
    # Limitations
    report.append("IMPORTANT LIMITATIONS")
    report.append("-" * 40)
    report.append("1. This analysis provides forensic evidence, not absolute proof.")
    report.append("2. The restoration is a reconstruction, not recovery of original pixels.")
    report.append("3. Sophisticated forgeries designed to evade detection may not be identified.")
    report.append("4. Heavily compressed or unusual images may trigger false positives.")
    report.append("5. ELA alone cannot prove tampering - it is supporting evidence only.")
    report.append("6. Results should be combined with other investigation methods.")
    report.append("")
    
    # Footer
    report.append("=" * 70)
    report.append(f"Report generated: {datetime.now().isoformat()}")
    report.append("System: Image Tampering Detection & Restoration")
    report.append("Model: ManTraNet-style CNN with ImageNet-pretrained backbone")
    report.append("=" * 70)
    
    return "\n".join(report)


def save_report(report_text: str, output_path: str):
    """Save report to file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    return str(path)


def generate_pdf_report(analysis_data: Dict, output_path: str) -> str:
    """
    Generate a PDF forensic report.
    
    Requires: pip install reportlab
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
        from reportlab.lib.colors import HexColor
        from reportlab.lib import colors
        
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72,
        )
        
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=20,
            textColor=HexColor('#10B981'),
            spaceAfter=30,
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=HexColor('#1F2937'),
            spaceAfter=12,
            spaceBefore=20,
        )
        
        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['Normal'],
            fontSize=10,
            leading=14,
        )
        
        story = []
        
        # Title
        story.append(Paragraph("Forensic Analysis Report", title_style))
        story.append(Paragraph("Image Tampering Detection & Restoration System", body_style))
        story.append(Spacer(1, 20))
        
        # Analysis Info
        story.append(Paragraph("Analysis Information", heading_style))
        story.append(Paragraph(f"Analysis ID: {analysis_data.get('id', 'N/A')}", body_style))
        story.append(Paragraph(f"Timestamp: {analysis_data.get('timestamp', 'N/A')}", body_style))
        story.append(Paragraph(f"Filename: {analysis_data.get('filename', 'N/A')}", body_style))
        story.append(Spacer(1, 10))
        
        # Detection Result
        story.append(Paragraph("Detection Result", heading_style))
        verdict = "TAMPERED" if analysis_data.get('is_tampered') else "AUTHENTIC"
        story.append(Paragraph(f"Verdict: <b>{verdict}</b>", body_style))
        story.append(Paragraph(f"Confidence: {analysis_data.get('confidence', 0):.1%}", body_style))
        story.append(Paragraph(f"Tampered Area: {analysis_data.get('tampered_percentage', 0):.2f}%", body_style))
        story.append(Spacer(1, 10))
        
        # Add images if available
        for img_type in ['heatmap', 'mask', 'overlay', 'ela', 'restored']:
            img_path = analysis_data.get(f'{img_type}_path')
            if img_path and Path(img_path).exists():
                story.append(Paragraph(f"{img_type.capitalize()}", heading_style))
                try:
                    img = Image(img_path, width=5*inch, height=3.5*inch)
                    story.append(img)
                    story.append(Spacer(1, 10))
                except Exception:
                    story.append(Paragraph(f"[Image: {img_path}]", body_style))
        
        # Limitations
        story.append(Paragraph("Limitations", heading_style))
        limitations = [
            "This analysis provides forensic evidence, not absolute proof.",
            "The restoration is a reconstruction, not recovery of original pixels.",
            "Sophisticated forgeries may evade detection.",
            "Results should be combined with other investigation methods.",
        ]
        for lim in limitations:
            story.append(Paragraph(f"• {lim}", body_style))
        
        # Build PDF
        doc.build(story)
        return output_path
        
    except ImportError:
        # Fallback to text report
        text_report = generate_text_report(analysis_data)
        txt_path = output_path.replace('.pdf', '.txt')
        save_report(text_report, txt_path)
        return txt_path
