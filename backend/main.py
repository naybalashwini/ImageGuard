"""
ImageGuard Backend API
=======================
FastAPI backend for image tampering detection and restoration.

Integrates:
- TruFor/MVSS-Net localization models
- Pixel-level post-processing pipeline
- Image restoration/inpainting
- Forensic report generation
"""

import os
import sys
import uuid
import time
import json
import logging
import sqlite3
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
UPLOAD_DIR = Path("uploads")
RESULT_DIR = Path("results")
DB_PATH = Path("forensic.db")
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# Create directories
UPLOAD_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

# Initialize FastAPI
app = FastAPI(
    title="ImageGuard - Image Tampering Detection & Restoration",
    description="AI-powered forensic analysis using TruFor for pixel-level forgery localization",
    version="2.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files for results
app.mount("/static/results", StaticFiles(directory="results"), name="results")

# Global state
localizer = None
restorer = None


def init_db():
    """Initialize SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS analyses (
            id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            original_path TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            verdict TEXT,
            confidence REAL,
            tampered_percentage REAL,
            num_regions INTEGER,
            model_used TEXT,
            result_json TEXT,
            processing_time REAL,
            status TEXT DEFAULT 'pending'
        )
    """)
    conn.commit()
    conn.close()


def load_models():
    """Load AI models at startup."""
    global localizer, restorer
    
    # Initialize localizer
    from forensics.localization import ForensicLocalizer
    localizer = ForensicLocalizer(device="auto")
    localizer.initialize()
    
    # Initialize restorer
    from restoration.inpainting import ImageRestorer
    restorer = ImageRestorer(device="auto")
    
    logger.info("Models initialized")


@app.on_event("startup")
async def startup_event():
    init_db()
    load_models()


# Pydantic models
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    version: str


class AnalysisResponse(BaseModel):
    id: str
    status: str
    message: str


# API Endpoints

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=localizer is not None and localizer.is_available(),
        model_name=localizer.get_model_name() if localizer else "Not loaded",
        version="2.0.0"
    )


@app.post("/api/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Accept an image and perform complete forensic analysis.
    
    Pipeline:
    1. Validate and save image
    2. Preprocess
    3. Run TruFor/MVSS-Net localization
    4. Post-process (morphological cleanup, connected components)
    5. Extract regions and bounding boxes
    6. Generate visualizations (heatmap, mask, overlay, contours)
    7. Restore/inpaint tampered regions
    8. Generate forensic report
    """
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Read and validate file
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large. Maximum: 20MB")
    
    # Validate it's actually an image
    try:
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(contents))
        img.verify()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid or corrupt image file")
    
    # Generate analysis ID and save file
    analysis_id = str(uuid.uuid4())
    safe_filename = f"{analysis_id}{ext}"
    file_path = UPLOAD_DIR / safe_filename
    
    with open(file_path, "wb") as f:
        f.write(contents)
    
    # Create database entry
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO analyses (id, filename, original_path, timestamp, status)
        VALUES (?, ?, ?, ?, 'processing')
    """, (analysis_id, file.filename, str(file_path), datetime.now().isoformat()))
    conn.commit()
    conn.close()
    
    # Process in background
    if background_tasks:
        background_tasks.add_task(process_full_analysis, analysis_id, str(file_path))
    else:
        # Synchronous processing
        process_full_analysis(analysis_id, str(file_path))
    
    return {"id": analysis_id, "status": "processing", "message": "Analysis started"}


def process_full_analysis(analysis_id: str, file_path: str):
    """
    Complete forensic analysis pipeline.
    
    This is the core pipeline that:
    1. Runs localization model
    2. Post-processes the localization map
    3. Extracts regions from the actual pixel mask
    4. Generates all visualizations
    5. Performs restoration
    6. Saves results
    """
    start_time = time.time()
    
    try:
        from PIL import Image
        from forensics.preprocessing import preprocess_image, compute_ela
        from forensics.postprocessing import (
            postprocess_localization,
            generate_heatmap,
            generate_overlay,
            generate_contour_overlay,
        )
        from forensics.regions import (
            extract_regions,
            compute_overall_verdict,
            format_bounding_boxes_for_visualization,
        )
        from report.generator import generate_forensic_report
        
        # Create result directory
        result_dir = RESULT_DIR / analysis_id
        result_dir.mkdir(exist_ok=True)
        
        # Step 1: Preprocess
        logger.info(f"[{analysis_id}] Step 1: Preprocessing")
        prep = preprocess_image(file_path)
        original = prep['original']
        original_array = prep['original_array']
        original_size = prep['original_size']
        
        # Step 2: ELA Analysis
        logger.info(f"[{analysis_id}] Step 2: Error Level Analysis")
        ela_result = compute_ela(original)
        ela_result['ela_image'].save(str(result_dir / 'ela.png'))
        
        # Step 3: Run localization model
        logger.info(f"[{analysis_id}] Step 3: Running localization model")
        loc_result = localizer.localize(file_path)
        localization_map = loc_result['localization_map']
        confidence_map = loc_result['confidence_map']
        score = loc_result['score']
        model_name = loc_result['model_name']
        
        # Save localization map as image
        loc_map_img = Image.fromarray((localization_map * 255).astype(np.uint8))
        loc_map_img.save(str(result_dir / 'localization_map.png'))
        
        # Save confidence map
        conf_map_img = Image.fromarray((confidence_map * 255).astype(np.uint8))
        conf_map_img.save(str(result_dir / 'confidence_map.png'))
        
        # Step 4: Post-process localization map
        logger.info(f"[{analysis_id}] Step 4: Post-processing localization")
        post_result = postprocess_localization(
            localization_map=localization_map,
            confidence_map=confidence_map,
            original_size=original_size,
            min_region_area=100,
            confidence_threshold=0.4,
            probability_threshold=0.5,
        )
        
        binary_mask = post_result['binary_mask']
        refined_mask = post_result['refined_mask']
        components = post_result['components']
        num_regions = post_result['num_regions']
        tampered_percentage = post_result['tampered_percentage']
        
        # Save masks
        Image.fromarray(binary_mask).save(str(result_dir / 'binary_mask.png'))
        Image.fromarray(refined_mask).save(str(result_dir / 'refined_mask.png'))
        
        # Step 5: Extract regions with bounding boxes
        logger.info(f"[{analysis_id}] Step 5: Extracting regions")
        regions = extract_regions(
            components=components,
            localization_map=localization_map,
            confidence_map=confidence_map,
            original_size=original_size,
        )
        
        # Step 6: Compute verdict
        logger.info(f"[{analysis_id}] Step 6: Computing verdict")
        verdict = compute_overall_verdict(
            regions=regions,
            score=score,
            tampered_percentage=tampered_percentage,
            model_name=model_name,
        )
        
        # Step 7: Generate visualizations
        logger.info(f"[{analysis_id}] Step 7: Generating visualizations")
        
        # Heatmap
        heatmap_array = generate_heatmap(localization_map, original_size)
        Image.fromarray(heatmap_array, 'RGBA').save(str(result_dir / 'heatmap.png'))
        
        # Overlay (mask on original)
        overlay_array = generate_overlay(original_array, refined_mask)
        Image.fromarray(overlay_array).save(str(result_dir / 'overlay.png'))
        
        # Contour overlay
        contour_array = generate_contour_overlay(original_array, refined_mask)
        Image.fromarray(contour_array).save(str(result_dir / 'contours.png'))
        
        # Step 8: Restoration
        logger.info(f"[{analysis_id}] Step 8: Restoring tampered regions")
        restored = restorer.restore(original, refined_mask)
        restored.save(str(result_dir / 'restored.png'))
        
        # Step 9: Format bounding boxes for frontend
        bounding_boxes = format_bounding_boxes_for_visualization(regions, original_size)
        
        # Step 10: Generate forensic signals
        forensic_signals = _generate_forensic_signals(
            ela_result, localization_map, confidence_map, regions, tampered_percentage
        )
        
        processing_time = time.time() - start_time
        
        # Compile complete result
        complete_result = {
            'id': analysis_id,
            'filename': Path(file_path).name,
            'timestamp': datetime.now().isoformat(),
            'image_size': f"{original_size[0]}x{original_size[1]}",
            'processing_time': processing_time,
            'verdict': verdict,
            'regions': regions,
            'bounding_boxes': bounding_boxes,
            'forensic_signals': forensic_signals,
            'ela_stats': {
                'mean': ela_result['ela_mean'],
                'max': ela_result['ela_max'],
                'block_anomaly_score': ela_result['block_anomaly_score'],
            },
            'outputs': {
                'original': f'/static/results/{analysis_id}/original.png',
                'localization_map': f'/static/results/{analysis_id}/localization_map.png',
                'confidence_map': f'/static/results/{analysis_id}/confidence_map.png',
                'binary_mask': f'/static/results/{analysis_id}/binary_mask.png',
                'refined_mask': f'/static/results/{analysis_id}/refined_mask.png',
                'heatmap': f'/static/results/{analysis_id}/heatmap.png',
                'overlay': f'/static/results/{analysis_id}/overlay.png',
                'contours': f'/static/results/{analysis_id}/contours.png',
                'ela': f'/static/results/{analysis_id}/ela.png',
                'restored': f'/static/results/{analysis_id}/restored.png',
            }
        }
        
        # Save original to results
        original.save(str(result_dir / 'original.png'))
        
        # Generate and save report
        report_text = generate_forensic_report(complete_result)
        with open(result_dir / 'report.txt', 'w') as f:
            f.write(report_text)
        
        # Save JSON result
        with open(result_dir / 'result.json', 'w') as f:
            json.dump(complete_result, f, indent=2, default=str)
        
        # Update database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE analyses SET 
                verdict = ?,
                confidence = ?,
                tampered_percentage = ?,
                num_regions = ?,
                model_used = ?,
                result_json = ?,
                processing_time = ?,
                status = 'complete'
            WHERE id = ?
        """, (
            verdict['verdict'],
            verdict['overall_confidence'],
            tampered_percentage,
            num_regions,
            model_name,
            json.dumps(complete_result, default=str),
            processing_time,
            analysis_id
        ))
        conn.commit()
        conn.close()
        
        logger.info(
            f"[{analysis_id}] Analysis complete in {processing_time:.2f}s. "
            f"Verdict: {verdict['verdict']}, Regions: {num_regions}"
        )
        
    except Exception as e:
        logger.error(f"[{analysis_id}] Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("UPDATE analyses SET status = 'error' WHERE id = ?", (analysis_id,))
        conn.commit()
        conn.close()


def _generate_forensic_signals(ela_result, localization_map, confidence_map, regions, tampered_pct):
    """Generate forensic signal descriptions."""
    signals = []
    
    # ELA signal
    if ela_result['block_anomaly_score'] > 0.5:
        signals.append({
            'name': 'ELA Compression Anomaly',
            'description': 'Error Level Analysis shows inconsistent compression patterns.',
            'severity': 'high' if ela_result['block_anomaly_score'] > 1.0 else 'medium',
            'evidence': f'Block anomaly score: {ela_result["block_anomaly_score"]:.3f}'
        })
    
    # Localization signal
    high_conf_pixels = np.sum(localization_map > 0.7)
    total_pixels = localization_map.size
    high_conf_ratio = high_conf_pixels / total_pixels
    
    if high_conf_ratio > 0.01:
        signals.append({
            'name': 'Localization Confidence',
            'description': f'{high_conf_ratio*100:.1f}% of pixels have high manipulation probability.',
            'severity': 'high' if high_conf_ratio > 0.05 else 'medium',
            'evidence': f'High-confidence area: {high_conf_ratio*100:.2f}%'
        })
    
    # Region-based signals
    critical_regions = [r for r in regions if r['severity'] == 'critical']
    if critical_regions:
        signals.append({
            'name': 'Critical Manipulation Regions',
            'description': f'{len(critical_regions)} region(s) with critical manipulation confidence.',
            'severity': 'critical',
            'evidence': f'Max region confidence: {max(r["region_confidence"] for r in critical_regions):.1%}'
        })
    
    if not signals:
        signals.append({
            'name': 'No Significant Anomalies',
            'description': 'Forensic analysis did not detect strong indicators of manipulation.',
            'severity': 'low',
            'evidence': 'All forensic signals within normal range'
        })
    
    return signals


@app.get("/api/history")
async def get_history():
    """Get analysis history."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, filename, timestamp, verdict, confidence, tampered_percentage, 
               num_regions, model_used, status
        FROM analyses ORDER BY timestamp DESC LIMIT 50
    """)
    rows = cursor.fetchall()
    conn.close()
    
    return [
        {
            "id": row[0],
            "filename": row[1],
            "timestamp": row[2],
            "verdict": row[3],
            "confidence": row[4],
            "tampered_percentage": row[5],
            "num_regions": row[6],
            "model_used": row[7],
            "status": row[8],
        }
        for row in rows
    ]


@app.get("/api/results/{analysis_id}")
async def get_results(analysis_id: str):
    """Get detailed analysis results."""
    result_file = RESULT_DIR / analysis_id / 'result.json'
    
    if result_file.exists():
        with open(result_file, 'r') as f:
            return json.load(f)
    
    # Try database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT result_json, status FROM analyses WHERE id = ?", (analysis_id,))
    row = cursor.fetchone()
    conn.close()
    
    if row and row[0]:
        return json.loads(row[0])
    elif row:
        return {"id": analysis_id, "status": row[1]}
    
    raise HTTPException(status_code=404, detail="Analysis not found")


@app.get("/api/download/{analysis_id}/{file_type}")
async def download_file(analysis_id: str, file_type: str):
    """Download a result file."""
    allowed_types = {
        'original', 'localization_map', 'confidence_map', 'binary_mask',
        'refined_mask', 'heatmap', 'overlay', 'contours', 'ela', 'restored', 'report'
    }
    
    if file_type not in allowed_types:
        raise HTTPException(status_code=400, detail=f"Invalid type. Allowed: {', '.join(allowed_types)}")
    
    ext = '.txt' if file_type == 'report' else '.png'
    file_path = RESULT_DIR / analysis_id / f"{file_type}{ext}"
    
    if file_path.exists():
        return FileResponse(str(file_path), filename=f"{file_type}_{analysis_id}{ext}")
    
    raise HTTPException(status_code=404, detail="File not found")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
