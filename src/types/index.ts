export interface AnalysisResult {
  id: string;
  filename: string;
  timestamp: string;
  isTampered: boolean;
  confidence: number;
  tamperedPercentage: number;
  heatmapUrl: string;
  maskUrl: string;
  overlayUrl: string;
  elaUrl: string;
  restoredUrl: string;
  originalUrl: string;
  confidenceMapUrl: string;
  localizationMapUrl: string;
  contoursUrl: string;
  detectionDetails: DetectionDetails;
  forensicSignals: ForensicSignal[];
  verdict: VerdictInfo;
  regions: DetectedRegion[];
  boundingBoxes: BoundingBox[];
  metrics: AnalysisMetrics;
}

export interface DetectionDetails {
  modelUsed: string;
  backboneArchitecture: string;
  inputResolution: string;
  processingTime: number;
  numberOfSuspiciousRegions: number;
  primaryRegionConfidence: number;
}

export interface ForensicSignal {
  name: string;
  description: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  evidence: string;
}

export interface VerdictInfo {
  verdict: string;
  verdict_description: string;
  confidence_level: string;
  overall_confidence: number;
  integrity_score: number;
  num_regions: number;
  tampered_percentage: number;
  model_used: string;
  severity_summary: Record<string, number>;
  warning: string;
}

export interface DetectedRegion {
  id: number;
  bbox: { x: number; y: number; width: number; height: number };
  centroid: { x: number; y: number };
  area_pixels: number;
  area_percentage: number;
  mean_probability: number;
  max_probability: number;
  mean_confidence: number;
  region_confidence: number;
  severity: string;
  description: string;
}

export interface BoundingBox {
  id: number;
  x_pct: number;
  y_pct: number;
  width_pct: number;
  height_pct: number;
  x_px: number;
  y_px: number;
  width_px: number;
  height_px: number;
  confidence: number;
  severity: string;
  description: string;
}

export interface AnalysisMetrics {
  localizationIoU: number | null;
  localizationDice: number | null;
  pixelPrecision: number | null;
  pixelRecall: number | null;
  pixelF1: number | null;
  restorationPSNR: number | null;
  restorationSSIM: number | null;
}

export interface HistoryEntry {
  id: string;
  filename: string;
  timestamp: string;
  isTampered: boolean;
  confidence: number;
  tamperedPercentage: number;
  thumbnailUrl: string;
  verdict?: string;
  numRegions?: number;
  modelUsed?: string;
  status?: string;
}

export interface AnalysisState {
  status: 'idle' | 'uploading' | 'processing' | 'complete' | 'error';
  progress: number;
  currentStep: string;
}
