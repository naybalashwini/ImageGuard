import { AnalysisResult, AnalysisState, HistoryEntry } from '../types';

const API_BASE = import.meta.env?.VITE_API_URL || 'http://localhost:8000';

async function apiFetch(path: string, options?: RequestInit) {
  const response = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: {
      ...options?.headers,
    },
  });
  
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Request failed' }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }
  
  return response.json();
}

export async function checkHealth(): Promise<{ status: string; model_loaded: boolean; model_name: string }> {
  return apiFetch('/api/health');
}

export async function uploadAndAnalyze(
  file: File,
  onProgress: (state: AnalysisState) => void
): Promise<string> {
  // Step 1: Upload
  onProgress({ status: 'uploading', progress: 10, currentStep: 'Uploading image...' });
  
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await fetch(`${API_BASE}/api/analyze`, {
    method: 'POST',
    body: formData,
  });
  
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Upload failed' }));
    throw new Error(error.detail || 'Upload failed');
  }
  
  const { id } = await response.json();
  
  // Step 2: Poll for results
  onProgress({ status: 'processing', progress: 20, currentStep: 'Running forensic analysis...' });
  
  const maxAttempts = 120; // 2 minutes max
  let attempts = 0;
  
  while (attempts < maxAttempts) {
    await delay(1500);
    attempts++;
    
    try {
      const result = await apiFetch(`/api/results/${id}`);
      
      if (result.status === 'complete') {
        onProgress({ status: 'complete', progress: 100, currentStep: 'Analysis complete' });
        return id;
      } else if (result.status === 'error') {
        throw new Error('Analysis failed on server');
      }
      
      // Update progress
      const progress = Math.min(90, 20 + attempts * 3);
      const steps = [
        'Preprocessing image...',
        'Running Error Level Analysis...',
        'Extracting forensic features...',
        'Running TruFor localization...',
        'Post-processing localization map...',
        'Extracting tampered regions...',
        'Generating visualizations...',
        'Restoring tampered regions...',
        'Generating forensic report...',
      ];
      const stepIdx = Math.min(steps.length - 1, Math.floor((progress - 20) / 8));
      
      onProgress({ 
        status: 'processing', 
        progress, 
        currentStep: steps[stepIdx] 
      });
    } catch (e) {
      if (e instanceof Error && e.message.includes('failed on server')) {
        throw e;
      }
      // Continue polling
    }
  }
  
  throw new Error('Analysis timed out');
}

export async function getResults(id: string): Promise<AnalysisResult> {
  const data = await apiFetch(`/api/results/${id}`);
  
  // Transform backend response to frontend format
  return {
    id: data.id,
    filename: data.filename || 'unknown',
    timestamp: data.timestamp || new Date().toISOString(),
    isTampered: data.verdict?.verdict === 'TAMPERED' || data.verdict?.verdict === 'LIKELY_TAMPERED',
    confidence: data.verdict?.overall_confidence || 0,
    tamperedPercentage: data.verdict?.tampered_percentage || 0,
    heatmapUrl: data.outputs?.heatmap ? `${API_BASE}${data.outputs.heatmap}` : '',
    maskUrl: data.outputs?.refined_mask ? `${API_BASE}${data.outputs.refined_mask}` : '',
    overlayUrl: data.outputs?.overlay ? `${API_BASE}${data.outputs.overlay}` : '',
    elaUrl: data.outputs?.ela ? `${API_BASE}${data.outputs.ela}` : '',
    restoredUrl: data.outputs?.restored ? `${API_BASE}${data.outputs.restored}` : '',
    originalUrl: data.outputs?.original ? `${API_BASE}${data.outputs.original}` : '',
    confidenceMapUrl: data.outputs?.confidence_map ? `${API_BASE}${data.outputs.confidence_map}` : '',
    localizationMapUrl: data.outputs?.localization_map ? `${API_BASE}${data.outputs.localization_map}` : '',
    contoursUrl: data.outputs?.contours ? `${API_BASE}${data.outputs.contours}` : '',
    detectionDetails: {
      modelUsed: data.verdict?.model_used || 'Unknown',
      backboneArchitecture: 'TruFor (Transformer-based fusion)',
      inputResolution: data.image_size || 'Unknown',
      processingTime: data.processing_time || 0,
      numberOfSuspiciousRegions: data.verdict?.num_regions || 0,
      primaryRegionConfidence: data.verdict?.overall_confidence || 0,
    },
    forensicSignals: data.forensic_signals || [],
    verdict: data.verdict || {},
    regions: data.regions || [],
    boundingBoxes: data.bounding_boxes || [],
    metrics: {
      localizationIoU: null,
      localizationDice: null,
      pixelPrecision: null,
      pixelRecall: null,
      pixelF1: null,
      restorationPSNR: null,
      restorationSSIM: null,
    },
  };
}

export async function getHistory(): Promise<HistoryEntry[]> {
  const data = await apiFetch('/api/history');
  
  return data.map((entry: any) => ({
    id: entry.id,
    filename: entry.filename,
    timestamp: entry.timestamp,
    isTampered: entry.verdict === 'TAMPERED' || entry.verdict === 'LIKELY_TAMPERED',
    confidence: entry.confidence || 0,
    tamperedPercentage: entry.tampered_percentage || 0,
    thumbnailUrl: '', // Will be loaded from API
    verdict: entry.verdict,
    numRegions: entry.num_regions,
    modelUsed: entry.model_used,
    status: entry.status,
  }));
}

export function getDownloadUrl(analysisId: string, fileType: string): string {
  return `${API_BASE}/api/download/${analysisId}/${fileType}`;
}

function delay(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}
