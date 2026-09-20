# ImageGuard - Image Tampering Detection & Restoration

## Complete Setup Guide (Windows)

### Prerequisites Installation

#### 1. Python 3.10+
```powershell
# Download from: https://www.python.org/downloads/
# IMPORTANT: Check "Add Python to PATH" during installation

# Verify installation
python --version
pip --version
```

#### 2. Git
```powershell
# Download from: https://git-scm.com/download/win
# Verify
git --version
```

#### 3. Node.js 18+
```powershell
# Download from: https://nodejs.org/
# Verify
node --version
npm --version
```

#### 4. CUDA (Optional - for GPU acceleration)
```powershell
# Download from: https://developer.nvidia.com/cuda-downloads
# Only needed if you have an NVIDIA GPU
```

---

### Project Setup

```powershell
# 1. Navigate to project directory
cd ImageGuard

# 2. Create Python virtual environment
python -m venv venv

# 3. Activate virtual environment
venv\Scripts\activate

# 4. Install Python dependencies
pip install -r requirements.txt

# For GPU support (NVIDIA GPU with CUDA 12.1):
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 5. Install frontend dependencies
npm install

# 6. Create necessary directories
mkdir uploads
mkdir results
mkdir models\weights\trufor
mkdir models\weights\mvss
```

---

### Model Weights Setup

#### TruFor Weights (Primary Model)
```powershell
# Download TruFor weights
# URL: https://www.grip.unina.it/download/prog/TruFor/TruFor_weights.zip
# MD5: 7bee48f3476c75616c3c5721ab256ff8

# Extract and place in:
# models/weights/trufor/trufor.pth.tar
```

#### MVSS-Net Weights (Fallback - Optional)
```powershell
# If TruFor weights are unavailable, the system will use
# classical forensic analysis as fallback.
# MVSS-Net weights can be placed in:
# models/weights/mvss/mvss_net.pth
```

**Note:** The system works WITHOUT pretrained weights using classical forensic analysis (ELA + texture + noise analysis). For best results, download TruFor weights.

---

### Running the Application

#### Start Backend (Terminal 1)
```powershell
# Activate virtual environment
venv\Scripts\activate

# Start FastAPI backend
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

#### Start Frontend (Terminal 2)
```powershell
# In a new terminal
npm run dev
```

#### Access the Application
- Frontend: http://localhost:5173
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

---

### Docker Deployment

```powershell
# CPU-only
docker-compose up --build

# GPU (uncomment GPU section in docker-compose.yml)
docker-compose up --build
```

---

### Training (Optional)

```powershell
# Setup datasets
python scripts/setup_datasets.py --create-dirs

# Download and place datasets in:
# datasets/CASIA/
# datasets/Columbia/
# datasets/Coverage/
# datasets/DEFACTO/

# Train model
python training/train.py --dataset_dir ./datasets --epochs 50

# Evaluate
python evaluation/evaluate.py --test_dir ./datasets/Coverage/test
```

---

### Testing

```powershell
# Run backend tests
pytest tests/ -v

# Test API manually
curl http://localhost:8000/api/health
```

---

### Troubleshooting

#### "Module not found" errors
```powershell
# Ensure virtual environment is activated
venv\Scripts\activate

# Reinstall dependencies
pip install -r requirements.txt
```

#### "CUDA out of memory"
```powershell
# Reduce batch size in training
# Or use CPU mode by setting DEVICE=cpu in .env
```

#### Frontend can't connect to backend
```powershell
# Ensure backend is running on port 8000
# Check CORS settings in backend/main.py
```

#### TruFor weights not loading
```powershell
# System will fall back to classical forensic analysis
# For best results, download TruFor weights from:
# https://www.grip.unina.it/download/prog/TruFor/TruFor_weights.zip
```

---

### Project Structure

```
ImageGuard/
├── backend/
│   └── main.py              # FastAPI backend with complete pipeline
├── models/
│   ├── trufor_detector.py   # TruFor localization model
│   └── mvss_detector.py     # MVSS-Net fallback model
├── forensics/
│   ├── preprocessing.py     # Image preprocessing & ELA
│   ├── localization.py      # Model-based localization
│   ├── postprocessing.py    # Mask generation & cleanup
│   └── regions.py           # Region extraction & bounding boxes
├── restoration/
│   └── inpainting.py        # Image restoration
├── report/
│   └── generator.py         # Forensic report generation
├── training/
│   └── train.py             # Model training script
├── evaluation/
│   └── evaluate.py          # Evaluation on test sets
├── src/                     # React frontend
│   ├── pages/
│   ├── components/
│   ├── services/
│   └── types/
├── requirements.txt
├── package.json
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check & model status |
| POST | `/api/analyze` | Upload and analyze image |
| GET | `/api/results/{id}` | Get analysis results |
| GET | `/api/history` | Get analysis history |
| GET | `/api/download/{id}/{type}` | Download result file |

### Result File Types

- `original` - Original uploaded image
- `localization_map` - Pixel-level tamper probability
- `confidence_map` - Model confidence/reliability
- `binary_mask` - Raw binary mask
- `refined_mask` - Cleaned mask after post-processing
- `heatmap` - Color-coded probability visualization
- `overlay` - Mask overlaid on original
- `contours` - Contour boundaries of regions
- `ela` - Error Level Analysis
- `restored` - Inpainted reconstruction
- `report` - Text forensic report
