"""
Image Tampering Detection & Restoration System
Backend API - FastAPI

This is the production backend that wraps the AI pipeline.
Run with: uvicorn backend.main:app --host 0.0.0.0 --port 8000
"""

import os
import uuid
import sqlite3
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
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
    title="Image Tampering Detection & Restoration API",
    description="AI-powered forensic analysis for detecting and localizing image manipulations",
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
app.mount("/static", StaticFiles(directory="results"), name="static")

# Database initialization
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
            is_tampered INTEGER NOT NULL,
            confidence REAL NOT NULL,
            tampered_percentage REAL NOT NULL,
            heatmap_path TEXT,
            mask_path TEXT,
            overlay_path TEXT,
            ela_path TEXT,
            restored_path TEXT,
            model_info TEXT,
            forensic_signals TEXT,
            metrics TEXT,
            processing_time REAL,
            status TEXT DEFAULT 'pending'
        )
    """)
    conn.commit()
    conn.close()

init_db()

# Model loading (loaded once at startup)
model_instance = None

def load_model():
    """Load the AI model once at startup."""
    global model_instance
    try:
        from ai.inference.pipeline import ForensicPipeline
        model_instance = ForensicPipeline()
        model_instance.load_models()
        logger.info("AI models loaded successfully")
    except ImportError:
        logger.warning("AI pipeline not available - running in demo mode")
        model_instance = None
    except Exception as e:
        logger.error(f"Failed to load AI models: {e}")
        model_instance = None

@app.on_event("startup")
async def startup_event():
    load_model()

# Pydantic models
class AnalysisResponse(BaseModel):
    id: str
    filename: str
    timestamp: str
    is_tampered: bool
    confidence: float
    tampered_percentage: float
    status: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str

# API Endpoints

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=model_instance is not None,
        version="1.0.0"
    )

@app.post("/api/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Accept an image and perform forensic analysis.
    Returns analysis ID for retrieving results.
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
    
    # Read file
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
    
    # Generate safe filename
    analysis_id = str(uuid.uuid4())
    safe_filename = f"{analysis_id}{ext}"
    file_path = UPLOAD_DIR / safe_filename
    
    # Save file
    with open(file_path, "wb") as f:
        f.write(contents)
    
    # Create database entry
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO analyses (id, filename, original_path, timestamp, is_tampered, 
                            confidence, tampered_percentage, status)
        VALUES (?, ?, ?, ?, 0, 0.0, 0.0, 'processing')
    """, (analysis_id, file.filename, str(file_path), datetime.now().isoformat()))
    conn.commit()
    conn.close()
    
    # Process in background
    if background_tasks:
        background_tasks.add_task(process_analysis, analysis_id, str(file_path))
    else:
        # Synchronous processing (for testing)
        process_analysis(analysis_id, str(file_path))
    
    return {"id": analysis_id, "status": "processing"}

def process_analysis(analysis_id: str, file_path: str):
    """Process a single analysis."""
    start_time = time.time()
    
    try:
        if model_instance is None:
            # Demo mode - generate placeholder results
            result = generate_demo_result(file_path)
        else:
            # Real AI pipeline
            result = model_instance.analyze(file_path)
        
        processing_time = time.time() - start_time
        
        # Save results
        result_dir = RESULT_DIR / analysis_id
        result_dir.mkdir(exist_ok=True)
        
        # Save result images
        for key in ['heatmap', 'mask', 'overlay', 'ela', 'restored']:
            if key in result and result[key] is not None:
                img_path = result_dir / f"{key}.png"
                result[key].save(str(img_path))
                result[f"{key}_path"] = str(img_path)
        
        # Update database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE analyses SET 
                is_tampered = ?,
                confidence = ?,
                tampered_percentage = ?,
                heatmap_path = ?,
                mask_path = ?,
                overlay_path = ?,
                ela_path = ?,
                restored_path = ?,
                model_info = ?,
                forensic_signals = ?,
                metrics = ?,
                processing_time = ?,
                status = 'complete'
            WHERE id = ?
        """, (
            1 if result['is_tampered'] else 0,
            result['confidence'],
            result['tampered_percentage'],
            result.get('heatmap_path'),
            result.get('mask_path'),
            result.get('overlay_path'),
            result.get('ela_path'),
            result.get('restored_path'),
            json.dumps(result.get('model_info', {})),
            json.dumps(result.get('forensic_signals', [])),
            json.dumps(result.get('metrics', {})),
            processing_time,
            analysis_id
        ))
        conn.commit()
        conn.close()
        
        logger.info(f"Analysis {analysis_id} completed in {processing_time:.2f}s")
        
    except Exception as e:
        logger.error(f"Analysis {analysis_id} failed: {e}")
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("UPDATE analyses SET status = 'error' WHERE id = ?", (analysis_id,))
        conn.commit()
        conn.close()

def generate_demo_result(file_path: str) -> dict:
    """Generate demo results when AI model is not available."""
    from PIL import Image
    import numpy as np
    
    img = Image.open(file_path)
    img_array = np.array(img)
    h, w = img_array.shape[:2]
    
    # Generate simple heatmap
    heatmap = np.zeros((h, w, 4), dtype=np.uint8)
    cx, cy = w // 3, h // 3
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
    radius = min(h, w) // 5
    mask_region = dist < radius
    heatmap[mask_region, 0] = 255
    heatmap[mask_region, 3] = 180
    heatmap_img = Image.fromarray(heatmap, 'RGBA')
    
    # Generate mask
    mask = np.zeros((h, w, 4), dtype=np.uint8)
    mask[mask_region, 0] = 255
    mask[mask_region, 3] = 200
    mask_img = Image.fromarray(mask, 'RGBA')
    
    # Generate ELA (simple difference)
    ela = np.abs(img_array.astype(np.int16) - (img_array * 0.95).astype(np.int16))
    ela = (ela * 10).clip(0, 255).astype(np.uint8)
    ela_img = Image.fromarray(ela)
    
    # Generate overlay
    overlay = img_array.copy()
    overlay[mask_region] = (overlay[mask_region] * 0.5 + np.array([255, 0, 0]) * 0.5).astype(np.uint8)
    overlay_img = Image.fromarray(overlay)
    
    # Generate restored (simple blur in masked region)
    from PIL import ImageFilter
    restored = img.copy()
    blurred = img.filter(ImageFilter.GaussianBlur(radius=5))
    restored_arr = np.array(restored)
    blurred_arr = np.array(blurred)
    restored_arr[mask_region] = blurred_arr[mask_region]
    restored_img = Image.fromarray(restored_arr)
    
    tampered_pixels = np.sum(mask_region)
    total_pixels = h * w
    
    return {
        'is_tampered': True,
        'confidence': 0.72,
        'tampered_percentage': (tampered_pixels / total_pixels) * 100,
        'heatmap': heatmap_img,
        'mask': mask_img,
        'overlay': overlay_img,
        'ela': ela_img,
        'restored': restored_img,
        'model_info': {
            'model': 'Demo Mode (AI model not loaded)',
            'note': 'Install AI dependencies for real analysis'
        },
        'forensic_signals': [
            {
                'name': 'Demo Mode Active',
                'description': 'AI model not loaded. Results are placeholder.',
                'severity': 'medium',
                'evidence': 'Install torch, opencv-python, and model weights for real analysis'
            }
        ],
        'metrics': {}
    }

@app.get("/api/history")
async def get_history():
    """Get analysis history."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, filename, timestamp, is_tampered, confidence, tampered_percentage, status
        FROM analyses ORDER BY timestamp DESC LIMIT 50
    """)
    rows = cursor.fetchall()
    conn.close()
    
    return [
        {
            "id": row[0],
            "filename": row[1],
            "timestamp": row[2],
            "is_tampered": bool(row[3]),
            "confidence": row[4],
            "tampered_percentage": row[5],
            "status": row[6]
        }
        for row in rows
    ]

@app.get("/api/history/{analysis_id}")
async def get_analysis(analysis_id: str):
    """Get detailed analysis result."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM analyses WHERE id = ?", (analysis_id,))
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    columns = [desc[0] for desc in cursor.description]
    result = dict(zip(columns, row))
    
    # Convert types
    result['is_tampered'] = bool(result['is_tampered'])
    result['model_info'] = json.loads(result['model_info']) if result['model_info'] else {}
    result['forensic_signals'] = json.loads(result['forensic_signals']) if result['forensic_signals'] else []
    result['metrics'] = json.loads(result['metrics']) if result['metrics'] else {}
    
    return result

@app.get("/api/download/{analysis_id}/{file_type}")
async def download_file(analysis_id: str, file_type: str):
    """Download a result file."""
    allowed_types = {'heatmap', 'mask', 'overlay', 'ela', 'restored', 'original'}
    if file_type not in allowed_types:
        raise HTTPException(status_code=400, detail=f"Invalid file type. Allowed: {', '.join(allowed_types)}")
    
    if file_type == 'original':
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT original_path FROM analyses WHERE id = ?", (analysis_id,))
        row = cursor.fetchone()
        conn.close()
        if row:
            return FileResponse(row[0], filename=f"original_{analysis_id}")
    else:
        file_path = RESULT_DIR / analysis_id / f"{file_type}.png"
        if file_path.exists():
            return FileResponse(str(file_path), filename=f"{file_type}_{analysis_id}.png")
    
    raise HTTPException(status_code=404, detail="File not found")

@app.get("/api/report/{analysis_id}")
async def generate_report(analysis_id: str):
    """Generate a forensic report."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM analyses WHERE id = ?", (analysis_id,))
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    columns = [desc[0] for desc in cursor.description]
    result = dict(zip(columns, row))
    
    report = f"""
FORENSIC ANALYSIS REPORT
========================

Analysis ID: {result['id']}
Timestamp: {result['timestamp']}
Filename: {result['filename']}
Status: {result['status']}

DETECTION RESULT
----------------
Verdict: {'TAMPERED' if result['is_tampered'] else 'AUTHENTIC'}
Confidence: {result['confidence']:.1%}
Tampered Region: {result['tampered_percentage']:.2f}%

MODEL INFORMATION
-----------------
{json.dumps(json.loads(result['model_info']) if result['model_info'] else {}, indent=2)}

FORENSIC SIGNALS
----------------
{json.dumps(json.loads(result['forensic_signals']) if result['forensic_signals'] else [], indent=2)}

PROCESSING TIME
---------------
{result['processing_time']:.2f} seconds

LIMITATIONS
-----------
1. This analysis provides forensic evidence, not absolute proof.
2. The restoration is a reconstruction, not recovery of original pixels.
3. Sophisticated forgeries may evade detection.
4. Results should be combined with other investigation methods.
"""
    
    report_path = RESULT_DIR / analysis_id / "report.txt"
    report_path.parent.mkdir(exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(report)
    
    return FileResponse(str(report_path), filename=f"forensic_report_{analysis_id}.txt")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
