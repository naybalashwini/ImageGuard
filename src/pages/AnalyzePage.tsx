import { useState, useCallback, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { Upload, Image, AlertCircle, Loader2, CheckCircle2, X, Server } from 'lucide-react';
import { uploadAndAnalyze, checkHealth } from '../services/analysisService';
import { AnalysisState } from '../types';

const MAX_FILE_SIZE = 20 * 1024 * 1024;
const ALLOWED_TYPES = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'];

export function AnalyzePage() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [analysisState, setAnalysisState] = useState<AnalysisState>({
    status: 'idle',
    progress: 0,
    currentStep: '',
  });
  const [resultId, setResultId] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [apiStatus, setApiStatus] = useState<'checking' | 'online' | 'offline'>('checking');
  const [modelName, setModelName] = useState<string>('');
  const fileInputRef = useRef<HTMLInputElement>(null);
  const navigate = useNavigate();

  // Check API health on mount
  useState(() => {
    checkHealth()
      .then(health => {
        setApiStatus('online');
        setModelName(health.model_name);
      })
      .catch(() => setApiStatus('offline'));
  });

  const validateFile = (f: File): string | null => {
    if (!ALLOWED_TYPES.includes(f.type)) {
      return 'Invalid file type. Supported: JPG, JPEG, PNG, WEBP';
    }
    if (f.size > MAX_FILE_SIZE) {
      return 'File too large. Maximum: 20MB';
    }
    return null;
  };

  const handleFileSelect = useCallback((f: File) => {
    setError(null);
    const validationError = validateFile(f);
    if (validationError) {
      setError(validationError);
      return;
    }
    setFile(f);
    const reader = new FileReader();
    reader.onload = (e) => setPreview(e.target?.result as string);
    reader.readAsDataURL(f);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const f = e.dataTransfer.files[0];
    if (f) handleFileSelect(f);
  }, [handleFileSelect]);

  const handleAnalyze = async () => {
    if (!file) return;
    setError(null);
    setResultId(null);
    
    try {
      const id = await uploadAndAnalyze(file, setAnalysisState);
      setResultId(id);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Analysis failed');
      setAnalysisState({ status: 'error', progress: 0, currentStep: '' });
    }
  };

  const handleReset = () => {
    setFile(null);
    setPreview(null);
    setError(null);
    setAnalysisState({ status: 'idle', progress: 0, currentStep: '' });
    setResultId(null);
  };

  const isProcessing = analysisState.status !== 'idle' && analysisState.status !== 'complete' && analysisState.status !== 'error';

  return (
    <div className="max-w-4xl mx-auto space-y-8">
      <div className="text-center">
        <h1 className="text-3xl font-bold text-white mb-2">Image Forensic Analysis</h1>
        <p className="text-gray-400">
          Upload an image for pixel-level tampering localization using TruFor
        </p>
      </div>

      {/* API Status */}
      <div className={`flex items-center gap-2 p-3 rounded-lg text-sm ${
        apiStatus === 'online' 
          ? 'bg-emerald-500/10 border border-emerald-500/20 text-emerald-400'
          : apiStatus === 'offline'
          ? 'bg-red-500/10 border border-red-500/20 text-red-400'
          : 'bg-gray-800 border border-gray-700 text-gray-400'
      }`}>
        <Server className="w-4 h-4" />
        {apiStatus === 'online' && <span>Backend connected • Model: {modelName}</span>}
        {apiStatus === 'offline' && <span>Backend offline. Start the server: <code className="bg-gray-800 px-1 rounded">uvicorn backend.main:app --port 8000</code></span>}
        {apiStatus === 'checking' && <span>Checking backend connection...</span>}
      </div>

      {/* Upload Area */}
      {!preview && (
        <div
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onClick={() => fileInputRef.current?.click()}
          className={`border-2 border-dashed rounded-2xl p-12 text-center cursor-pointer transition-all ${
            isDragging ? 'border-emerald-400 bg-emerald-500/5' : 'border-gray-700 hover:border-gray-600 bg-gray-900/50'
          }`}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="image/jpeg,image/jpg,image/png,image/webp"
            onChange={(e) => e.target.files?.[0] && handleFileSelect(e.target.files[0])}
            className="hidden"
          />
          <Upload className={`w-12 h-12 mx-auto mb-4 ${isDragging ? 'text-emerald-400' : 'text-gray-500'}`} />
          <h3 className="text-lg font-semibold text-white mb-2">
            {isDragging ? 'Drop image here' : 'Drag & drop an image'}
          </h3>
          <p className="text-sm text-gray-400 mb-4">or click to browse</p>
          <div className="flex items-center justify-center gap-4 text-xs text-gray-500">
            <span>JPG, PNG, WEBP</span><span>•</span><span>Max 20MB</span>
          </div>
        </div>
      )}

      {/* Preview */}
      {preview && (
        <div className="bg-gray-900 rounded-2xl border border-gray-800 overflow-hidden">
          <div className="p-4 flex items-center justify-between border-b border-gray-800">
            <div className="flex items-center gap-3">
              <Image className="w-5 h-5 text-emerald-400" />
              <div>
                <p className="text-sm font-medium text-white">{file?.name}</p>
                <p className="text-xs text-gray-500">{file ? (file.size / 1024 / 1024).toFixed(2) + ' MB' : ''}</p>
              </div>
            </div>
            {!isProcessing && (
              <button onClick={handleReset} className="p-2 hover:bg-gray-800 rounded-lg">
                <X className="w-4 h-4 text-gray-400" />
              </button>
            )}
          </div>
          <div className="p-4">
            <img src={preview} alt="Preview" className="max-h-96 w-full object-contain rounded-lg" />
          </div>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="flex items-center gap-3 p-4 bg-red-500/10 border border-red-500/20 rounded-xl">
          <AlertCircle className="w-5 h-5 text-red-400 shrink-0" />
          <p className="text-sm text-red-300">{error}</p>
        </div>
      )}

      {/* Progress */}
      {isProcessing && (
        <div className="bg-gray-900 rounded-2xl border border-gray-800 p-6">
          <div className="flex items-center gap-3 mb-4">
            <Loader2 className="w-5 h-5 text-emerald-400 animate-spin" />
            <span className="text-sm font-medium text-white">{analysisState.currentStep}</span>
          </div>
          <div className="w-full bg-gray-800 rounded-full h-2">
            <div
              className="h-full bg-gradient-to-r from-emerald-500 to-cyan-500 rounded-full transition-all duration-500"
              style={{ width: `${analysisState.progress}%` }}
            />
          </div>
          <p className="text-xs text-gray-500 mt-2">{analysisState.progress}% complete</p>
          
          <div className="mt-6 grid grid-cols-2 sm:grid-cols-4 gap-2">
            {['Upload', 'Localization', 'Post-processing', 'Restoration'].map((step, i) => (
              <div key={step} className={`text-center p-2 rounded-lg text-xs ${
                analysisState.progress >= (i + 1) * 25
                  ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                  : 'bg-gray-800/50 text-gray-500 border border-gray-700/50'
              }`}>
                {step}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Complete */}
      {analysisState.status === 'complete' && resultId && (
        <div className="bg-emerald-500/5 border border-emerald-500/20 rounded-2xl p-6 text-center">
          <CheckCircle2 className="w-12 h-12 text-emerald-400 mx-auto mb-4" />
          <h3 className="text-xl font-bold text-white mb-2">Analysis Complete</h3>
          <p className="text-gray-400 mb-6">View detailed forensic results with pixel-level localization.</p>
          <div className="flex items-center justify-center gap-4">
            <button
              onClick={() => navigate(`/results/${resultId}`)}
              className="px-6 py-3 bg-emerald-500 hover:bg-emerald-600 text-white font-semibold rounded-lg"
            >
              View Results
            </button>
            <button
              onClick={handleReset}
              className="px-6 py-3 bg-gray-800 hover:bg-gray-700 text-gray-300 font-semibold rounded-lg border border-gray-700"
            >
              Analyze Another
            </button>
          </div>
        </div>
      )}

      {/* Analyze Button */}
      {preview && analysisState.status === 'idle' && (
        <button
          onClick={handleAnalyze}
          disabled={apiStatus !== 'online'}
          className="w-full py-4 bg-gradient-to-r from-emerald-500 to-cyan-500 hover:from-emerald-600 hover:to-cyan-600 text-white font-bold text-lg rounded-xl transition-all shadow-lg shadow-emerald-500/20 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          🔍 Start Forensic Analysis (TruFor)
        </button>
      )}

      {/* Info */}
      <div className="bg-gray-900/50 rounded-xl border border-gray-800 p-4">
        <h4 className="text-sm font-semibold text-gray-300 mb-2">Analysis Pipeline:</h4>
        <ul className="text-xs text-gray-500 space-y-1">
          <li>• <strong>TruFor Localization:</strong> Pixel-level forgery detection using transformer-based fusion</li>
          <li>• <strong>Confidence Map:</strong> Reliability estimation to suppress false positives</li>
          <li>• <strong>Adaptive Thresholding:</strong> Confidence-weighted mask generation</li>
          <li>• <strong>Morphological Cleanup:</strong> Connected component analysis and noise removal</li>
          <li>• <strong>Region Extraction:</strong> Bounding boxes derived from actual pixel mask</li>
          <li>• <strong>ELA Analysis:</strong> Error Level Analysis for compression artifact detection</li>
          <li>• <strong>Restoration:</strong> Inpainting reconstruction of detected regions</li>
        </ul>
      </div>
    </div>
  );
}

function handleDragOver(e: React.DragEvent) {
  e.preventDefault();
}

function handleDragLeave() {}
