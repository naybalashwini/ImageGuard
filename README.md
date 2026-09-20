# Image Tampering Detection & Restoration System

## AI-Powered Digital Image Forensics

A complete, production-ready system for detecting, localizing, and restoring manipulated images using deep learning and classical forensic techniques.

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Problem Statement](#problem-statement)
3. [System Architecture](#system-architecture)
4. [Technology Selection](#technology-selection)
5. [Model Architecture](#model-architecture)
6. [Dataset Strategy](#dataset-strategy)
7. [Installation (Windows)](#installation-windows)
8. [Usage](#usage)
9. [API Documentation](#api-documentation)
10. [Training](#training)
11. [Evaluation](#evaluation)
12. [Docker Deployment](#docker-deployment)
13. [Testing](#testing)
14. [Limitations](#limitations)
15. [Future Scope](#future-scope)

---

## Overview

This system provides:
- **Image-level detection**: Is the image authentic or tampered?
- **Pixel-level localization**: Where exactly was the image manipulated?
- **Heatmap generation**: Visual probability map of manipulation
- **Binary mask**: Clear delineation of suspicious regions
- **Overlay visualization**: Detected regions on original image
- **ELA analysis**: Error Level Analysis for compression artifacts
- **Image restoration**: Inpainting reconstruction of tampered regions
- **Forensic reports**: Detailed documentation of findings

---

## Problem Statement

Digital image manipulation has become increasingly sophisticated with AI-powered editing tools, generative fill, and deepfake technology. This system addresses the need for automated forensic analysis that can:

1. Detect tampering on **unseen images** (generalization)
2. Provide **spatial localization** (not just classification)
3. Generate **interpretable evidence** (heatmaps, masks, ELA)
4. Attempt **restoration** of manipulated regions
5. Work as a **practical tool** for forensic investigation

---

## System Architecture

```
USER IMAGE
    ↓
PREPROCESSING (Resize, Normalize)
    ↓
FORENSIC FEATURE ANALYSIS
├── Error Level Analysis (ELA)
├── Texture Variance Analysis
├── Noise Residual Analysis
└── Color Channel Analysis
    ↓
TAMPERING DETECTION (ManTraNet-style CNN)
    ↓
TAMPERED REGION LOCALIZATION (Pixel-level probability map)
    ↓
MASK GENERATION (Adaptive thresholding + morphological ops)
    ↓
MASK REFINEMENT (Connected component analysis)
    ↓
IMAGE RESTORATION (LaMa / OpenCV inpainting)
    ↓
RESULT ANALYSIS (Metrics, confidence, signals)
    ↓
PDF REPORT
```

---

## Technology Selection

### Why Each Technology Was Chosen:

| Technology | Purpose | Why Selected |
|-----------|---------|-------------|
| **PyTorch** | Deep Learning Framework | Best ecosystem for research, excellent GPU support, dynamic graphs |
| **ResNet-50** | Encoder Backbone | ImageNet-pretrained, proven feature extraction, efficient |
| **ManTraNet Architecture** | Forgery Localization | Designed specifically for manipulation detection, spatial attention |
| **FastAPI** | Backend API | Async support, automatic docs, high performance, type safety |
| **React + Vite** | Frontend | Fast development, modern tooling, excellent DX |
| **OpenCV** | Image Processing | Industry standard, comprehensive CV functions |
| **LaMa** | Inpainting | State-of-the-art for large mask inpainting |
| **SQLite** | Database | Zero-config, perfect for local development |

### Why NOT other approaches:
- **Generic CNN classifier**: No spatial localization (only real/fake)
- **Training from scratch**: Requires massive data, poor generalization
- **GAN-based detection**: Hard to train, unstable, poor generalization
- **Transformer-only**: Computationally expensive, needs huge data
- **ELA alone**: Not sufficient for modern manipulations

---

## Model Architecture

### Forgery Localization Model (ManTraNet-style)

```
Input Image (3 × 512 × 512)
    ↓
[ResNet-50 Encoder - ImageNet Pretrained]
    ├── Stage 1: 64 channels (1/4 resolution)
    ├── Stage 2: 256 channels (1/4 resolution)
    ├── Stage 3: 512 channels (1/8 resolution)
    ├── Stage 4: 1024 channels (1/16 resolution)
    └── Stage 5: 2048 channels (1/32 resolution)
    ↓
[Attention Decoder with Skip Connections]
    ├── Decoder 4: 512 ch → Attention → Skip from Stage 4
    ├── Decoder 3: 256 ch → Attention → Skip from Stage 3
    ├── Decoder 2: 128 ch → Attention → Skip from Stage 2
    └── Decoder 1: 64 ch → Attention → Skip from Stage 1
    ↓
[Prediction Head]
    Conv2d(64→32) → ReLU → Conv2d(32→1) → Sigmoid
    ↓
Output: Manipulation Probability Map (1 × H × W)
```

**Key Design Decisions:**
1. **Transfer Learning**: ResNet-50 pretrained on ImageNet provides robust features that generalize to unseen images
2. **Spatial Attention**: Decoder attention blocks focus on manipulation artifacts
3. **Skip Connections**: Preserve fine-grained spatial details for precise localization
4. **Sigmoid Output**: Produces probability map for interpretable results

---

## Dataset Strategy

### Datasets Used

| Dataset | Manipulation Types | Images | Masks | Usage |
|---------|-------------------|--------|-------|-------|
| CASIA v2.0 | Splicing, Copy-move | ~12,000 | Yes | Training |
| Columbia | Splicing | ~1,200 | Yes | Training |
| NIST Nimble | Various | ~6,000 | Yes | Validation |
| Coverage | Copy-move | 100 pairs | Yes | Cross-dataset Test |
| DEFACTO | Modern editing | ~1,000 | Yes | Unseen Test |

### Anti-Leakage Strategy

1. **Strict separation**: No image appears in both train and test
2. **Cross-dataset evaluation**: Train on CASIA+Columbia, test on Coverage+DEFACTO
3. **Perceptual hashing**: Remove near-duplicates across splits
4. **Multiple manipulation types**: Evaluate on splicing, copy-move, object removal, AI editing

### Dataset Download

```bash
# CASIA v2.0
# Download from: https://pkorus.github.io/downloads/casia-dataset
# Place in: datasets/CASIA/

# Columbia
# Download from: http://www.cs.columbia.edu/CAVE/databases/multimedia_splicing/
# Place in: datasets/Columbia/

# Coverage
# Download from: https://github.com/ericjjj99/Coverage-dataset
# Place in: datasets/Coverage/

# DEFACTO
# Download from: https://defacto-fpi.github.io/
# Place in: datasets/DEFACTO/
```

---

## Installation (Windows)

### Prerequisites

1. **Python 3.10+**: Download from https://www.python.org/downloads/
   - During installation, check "Add Python to PATH"

2. **Git**: Download from https://git-scm.com/download/win

3. **Node.js 18+**: Download from https://nodejs.org/

4. **CUDA** (optional, for GPU): Download from https://developer.nvidia.com/cuda-downloads

### Step-by-Step Installation

```powershell
# 1. Clone the project
git clone <repository-url>
cd ImageTamperingDetection

# 2. Create Python virtual environment
python -m venv venv
venv\Scripts\activate

# 3. Install Python dependencies
pip install -r requirements.txt

# For GPU support (NVIDIA GPU with CUDA):
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. Install frontend dependencies
cd frontend
npm install
cd ..

# 5. Download model weights (optional - system works without them in demo mode)
# Place weights in: ai/models/weights/forgery_localization.pth

# 6. Setup datasets (optional - for training/evaluation)
mkdir datasets\CASIA\train\images
mkdir datasets\CASIA\train\masks
mkdir datasets\CASIA\val\images
mkdir datasets\CASIA\val\masks
# ... download and place dataset files

# 7. Start the backend
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# 8. In another terminal, start the frontend
cd frontend
npm run dev
```

### Access the Application

- Frontend: http://localhost:5173
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

---

## Usage

### Web Interface

1. Navigate to http://localhost:5173
2. Click "Analyze Image"
3. Upload or drag-and-drop an image
4. Click "Start Forensic Analysis"
5. View results: heatmap, mask, overlay, ELA, restored image
6. Download report

### API Usage

```bash
# Health check
curl http://localhost:8000/api/health

# Analyze an image
curl -X POST http://localhost:8000/api/analyze \
  -F "file=@test_image.jpg"

# Get results
curl http://localhost:8000/api/history/{id}

# Download heatmap
curl http://localhost:8000/api/download/{id}/heatmap -o heatmap.png

# Generate report
curl http://localhost:8000/api/report/{id} -o report.txt
```

---

## API Documentation

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| POST | `/api/analyze` | Upload and analyze image |
| GET | `/api/history` | Get analysis history |
| GET | `/api/history/{id}` | Get analysis details |
| GET | `/api/download/{id}/{type}` | Download result file |
| GET | `/api/report/{id}` | Generate forensic report |

### File Types for Download

- `heatmap` - Color-coded probability map
- `mask` - Binary tampering mask
- `overlay` - Detected regions on original
- `ela` - Error Level Analysis
- `restored` - Inpainted reconstruction
- `original` - Original uploaded image

---

## Training

### Train the Model

```bash
# Activate virtual environment
venv\Scripts\activate

# Train with default settings
python training/train.py --dataset_dir ./datasets --epochs 50 --batch_size 8

# Train with custom settings
python training/train.py \
    --dataset_dir ./datasets \
    --output_dir ./ai/models/weights \
    --epochs 100 \
    --batch_size 16 \
    --lr 1e-4 \
    --device cuda \
    --seed 42
```

### Training Configuration

- **Optimizer**: AdamW (weight_decay=1e-4)
- **Scheduler**: CosineAnnealingLR
- **Loss**: 0.5 × BCE + 0.5 × Dice Loss
- **Gradient Clipping**: max_norm=1.0
- **Augmentation**: Random flip, rotation, brightness, contrast

---

## Evaluation

### Run Evaluation

```bash
# Evaluate on test set
python evaluation/evaluate.py \
    --model_path ./ai/models/weights/forgery_localization.pth \
    --test_dir ./datasets/Coverage/test

# Cross-dataset evaluation
python evaluation/evaluate.py \
    --model_path ./ai/models/weights/forgery_localization.pth \
    --test_dir ./datasets/DEFACTO/test \
    --cross_dataset
```

### Metrics Computed

**Image-Level Detection:**
- Accuracy, Precision, Recall, F1, ROC-AUC

**Pixel-Level Localization:**
- IoU, Dice, Pixel Precision, Pixel Recall, Pixel F1

**Restoration Quality** (when ground truth available):
- PSNR, SSIM, LPIPS

---

## Docker Deployment

```bash
# Build and run (CPU)
docker-compose up --build

# Build and run (GPU - uncomment GPU section in docker-compose.yml)
docker-compose up --build

# Access at http://localhost:8000
```

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_api.py -v

# Run with coverage
pytest tests/ --cov=backend --cov=ai
```

---

## Limitations

1. **No absolute proof**: Forensic analysis provides evidence, not certainty
2. **Sophisticated forgeries**: Anti-forensic techniques may evade detection
3. **Restoration is approximation**: Cannot recover exact original pixels
4. **False positives**: Heavy compression or unusual content may trigger false detections
5. **Computational requirements**: Full AI pipeline requires GPU for optimal speed
6. **Training data bias**: Novel manipulation types may not be detected
7. **Resolution limits**: Very large images may need resizing

---

## Future Scope

- Vision Transformer (ViT) based forensic models
- Diffusion-based restoration
- Video forensics (temporal consistency)
- Metadata analysis (EXIF, camera fingerprint)
- Ensemble of multiple forensic models
- Real-time detection via model distillation
- Adversarial training for anti-forensic resistance

---

## Project Structure

```
ImageTamperingDetection/
│
├── frontend/               # React + Vite + Tailwind frontend
│   ├── src/
│   │   ├── components/     # Reusable UI components
│   │   ├── pages/          # Page components
│   │   ├── services/       # API services
│   │   ├── types/          # TypeScript types
│   │   └── utils/          # Utility functions
│   └── package.json
│
├── backend/                # FastAPI backend
│   └── main.py            # API endpoints
│
├── ai/                     # AI/ML pipeline
│   ├── models/
│   │   └── forgery_localization.py  # Model architecture
│   ├── inference/
│   │   └── pipeline.py    # Complete inference pipeline
│   └── restoration/
│       └── inpainting.py  # Image restoration
│
├── training/
│   └── train.py           # Training script
│
├── evaluation/
│   └── evaluate.py        # Evaluation script
│
├── datasets/              # Dataset storage
├── tests/                 # Automated tests
├── reports/               # Generated reports
│
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker configuration
├── docker-compose.yml    # Docker Compose
└── README.md             # This file
```

---

## Viva Questions & Answers

### Q1: What is image tampering?
A: Image tampering refers to deliberate modification of digital images to mislead viewers. Common types include splicing (compositing from different images), copy-move (duplicating regions), object removal/addition, and AI-based editing.

### Q2: How does the system detect tampering?
A: The system uses a ManTraNet-style CNN with ResNet-50 backbone (pretrained on ImageNet) that learns to identify manipulation artifacts at the pixel level. It's combined with classical forensic techniques (ELA, texture analysis) for robustness.

### Q3: How does localization work?
A: The encoder-decoder architecture produces a per-pixel probability map. The encoder extracts multi-scale features, and the decoder with attention mechanisms upsamples these to generate a full-resolution localization map.

### Q4: Why was this model chosen?
A: ManTraNet-style architecture provides: (1) pixel-level localization, (2) transfer learning from ImageNet for generalization, (3) proven cross-dataset performance, (4) efficient computation on consumer hardware.

### Q5: What datasets were used?
A: CASIA v2.0 and Columbia for training, NIST for validation, Coverage and DEFACTO for cross-dataset testing. All are publicly available with ground-truth masks.

### Q6: How was data leakage prevented?
A: Strict train/test separation by dataset. No image appears in both training and testing. Near-duplicate removal using perceptual hashing. Cross-dataset evaluation ensures true generalization measurement.

### Q7: How does the system handle unseen images?
A: Transfer learning from ImageNet provides generalizable features. The model learns manipulation artifacts (not dataset-specific patterns). Cross-dataset evaluation validates performance on truly unseen data.

### Q8: How was the model evaluated?
A: Image-level metrics (Accuracy, Precision, Recall, F1, ROC-AUC) and pixel-level metrics (IoU, Dice, Pixel F1). Multiple thresholds evaluated. Cross-dataset testing on completely separate datasets.

### Q9: How does restoration work?
A: After localization, the tampered region mask is used to guide inpainting. LaMa (or OpenCV fallback) fills the masked region with plausible content based on surrounding context. This is a reconstruction, not original pixel recovery.

### Q10: What are the limitations?
A: No forensic tool provides absolute proof. Sophisticated anti-forensic attacks may evade detection. Restoration is approximate. Heavy compression causes false positives. The system works best as supporting evidence in a broader investigation.

---

## License

This project is developed as an academic final-year B.Tech project.

---

## Acknowledgments

- ManTraNet: Wu et al., "ManTra-Net: Manipulation Tracing Network", ICCV 2019
- LaMa: Suvorov et al., "Resolution-robust Large Mask Inpainting with Fourier Convolutions", WACV 2022
- Dataset providers: CASIA, Columbia, Coverage, NIST, DEFACTO
