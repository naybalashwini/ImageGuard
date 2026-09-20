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
  detectionDetails: DetectionDetails;
  forensicSignals: ForensicSignal[];
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
}

export interface AnalysisState {
  status: 'idle' | 'uploading' | 'processing' | 'ela' | 'localization' | 'restoration' | 'complete' | 'error';
  progress: number;
  currentStep: string;
}
