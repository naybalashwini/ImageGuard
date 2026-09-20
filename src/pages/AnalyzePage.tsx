import { useState, useCallback, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { Upload, Image, AlertCircle, Loader2, CheckCircle2, X } from 'lucide-react';
import { analyzeImage } from '../services/analysisService';
import { AnalysisState } from '../types';

const MAX_FILE_SIZE = 20 * 1024 * 1024; // 20MB
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
  const fileInputRef = useRef<HTMLInputElement>(null);
  const navigate = useNavigate();

  const validateFile = (f: File): string | null => {
    if (!ALLOWED_TYPES.includes(f.type)) {
      return 'Invalid file type. Supported formats: JPG, JPEG, PNG, WEBP';
    }
    if (f.size > MAX_FILE_SIZE) {
      return 'File too large. Maximum size: 20MB';
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

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback(() => {
    setIsDragging(false);
  }, []);

  const handleAnalyze = async () => {
    if (!file) return;
    
    setError(null);
    setResultId(null);
    
    try {
      const result = await analyzeImage(file, setAnalysisState);
      setResultId(result.id);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Analysis failed. Please try again.');
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
        <p className="text-gray-400">Upload an image to detect tampering, localize manipulated regions, and generate restoration.</p>
      </div>

      {/* Upload Area */}
      {!preview && (
        <div
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onClick={() => fileInputRef.current?.click()}
          className={`border-2 border-dashed rounded-2xl p-12 text-center cursor-pointer transition-all ${
            isDragging
              ? 'border-emerald-400 bg-emerald-500/5'
              : 'border-gray-700 hover:border-gray-600 bg-gray-900/50'
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
          <p className="text-sm text-gray-400 mb-4">or click to browse files</p>
          <div className="flex items-center justify-center gap-4 text-xs text-gray-500">
            <span>JPG, JPEG, PNG, WEBP</span>
            <span>•</span>
            <span>Max 20MB</span>
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
              <button onClick={handleReset} className="p-2 hover:bg-gray-800 rounded-lg transition-colors">
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
          <div className="w-full bg-gray-800 rounded-full h-2 overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-emerald-500 to-cyan-500 rounded-full transition-all duration-500"
              style={{ width: `${analysisState.progress}%` }}
            />
          </div>
          <div className="mt-3 flex items-center justify-between text-xs text-gray-500">
            <span>Processing pipeline</span>
            <span>{analysisState.progress}%</span>
          </div>
          
          {/* Pipeline steps */}
          <div className="mt-6 grid grid-cols-3 sm:grid-cols-5 gap-2">
            {[
              { name: 'Upload', threshold: 10 },
              { name: 'ELA', threshold: 35 },
              { name: 'Features', threshold: 55 },
              { name: 'Localization', threshold: 70 },
              { name: 'Restoration', threshold: 90 },
            ].map(step => (
              <div
                key={step.name}
                className={`text-center p-2 rounded-lg text-xs ${
                  analysisState.progress >= step.threshold
                    ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                    : 'bg-gray-800/50 text-gray-500 border border-gray-700/50'
                }`}
              >
                {analysisState.progress >= step.threshold ? (
                  <CheckCircle2 className="w-4 h-4 mx-auto mb-1" />
                ) : (
                  <div className="w-4 h-4 mx-auto mb-1 rounded-full border border-gray-600" />
                )}
                {step.name}
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
          <p className="text-gray-400 mb-6">View the detailed forensic analysis results.</p>
          <div className="flex items-center justify-center gap-4">
            <button
              onClick={() => navigate(`/results/${resultId}`)}
              className="px-6 py-3 bg-emerald-500 hover:bg-emerald-600 text-white font-semibold rounded-lg transition-all"
            >
              View Results
            </button>
            <button
              onClick={handleReset}
              className="px-6 py-3 bg-gray-800 hover:bg-gray-700 text-gray-300 font-semibold rounded-lg transition-all border border-gray-700"
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
          className="w-full py-4 bg-gradient-to-r from-emerald-500 to-cyan-500 hover:from-emerald-600 hover:to-cyan-600 text-white font-bold text-lg rounded-xl transition-all shadow-lg shadow-emerald-500/20"
        >
          🔍 Start Forensic Analysis
        </button>
      )}

      {/* Info */}
      <div className="bg-gray-900/50 rounded-xl border border-gray-800 p-4">
        <h4 className="text-sm font-semibold text-gray-300 mb-2">What this analysis does:</h4>
        <ul className="text-xs text-gray-500 space-y-1">
          <li>• Performs Error Level Analysis (ELA) to detect compression inconsistencies</li>
          <li>• Extracts forensic features using texture, noise, and color analysis</li>
          <li>• Generates a pixel-level tampering localization heatmap</li>
          <li>• Creates a binary mask of suspicious regions</li>
          <li>• Overlays detected regions on the original image</li>
          <li>• Attempts inpainting restoration of detected regions</li>
          <li>• Generates a forensic analysis report</li>
        </ul>
      </div>
    </div>
  );
}
