import { useParams, Link } from 'react-router-dom';
import { useEffect, useState } from 'react';
import { 
  ArrowLeft, Download, AlertTriangle, CheckCircle2, XCircle, 
  Image, Map, Layers, Eye, RefreshCw, FileText, Info
} from 'lucide-react';
import { AnalysisResult } from '../types';
import { getResultById, downloadImage, downloadReport } from '../services/analysisService';

export function ResultsPage() {
  const { id } = useParams<{ id: string }>();
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [activeTab, setActiveTab] = useState<string>('heatmap');

  useEffect(() => {
    if (id) {
      const r = getResultById(id);
      setResult(r);
    }
  }, [id]);

  if (!result) {
    return (
      <div className="text-center py-16">
        <XCircle className="w-16 h-16 text-gray-600 mx-auto mb-4" />
        <h2 className="text-xl font-bold text-white mb-2">Analysis Not Found</h2>
        <p className="text-gray-400 mb-6">The requested analysis result could not be found.</p>
        <Link to="/analyze" className="px-4 py-2 bg-emerald-500 text-white rounded-lg hover:bg-emerald-600 transition-colors">
          Start New Analysis
        </Link>
      </div>
    );
  }

  const tabs = [
    { id: 'heatmap', label: 'Heatmap', icon: Map },
    { id: 'mask', label: 'Mask', icon: Layers },
    { id: 'overlay', label: 'Overlay', icon: Eye },
    { id: 'ela', label: 'ELA', icon: Image },
    { id: 'restored', label: 'Restored', icon: RefreshCw },
  ];

  const getTabImage = () => {
    switch (activeTab) {
      case 'heatmap': return result.heatmapUrl;
      case 'mask': return result.maskUrl;
      case 'overlay': return result.overlayUrl;
      case 'ela': return result.elaUrl;
      case 'restored': return result.restoredUrl;
      default: return result.heatmapUrl;
    }
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <Link to="/analyze" className="p-2 hover:bg-gray-800 rounded-lg transition-colors">
            <ArrowLeft className="w-5 h-5 text-gray-400" />
          </Link>
          <div>
            <h1 className="text-2xl font-bold text-white">Forensic Analysis Results</h1>
            <p className="text-sm text-gray-500">ID: {result.id.slice(0, 8)} • {new Date(result.timestamp).toLocaleString()}</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => downloadReport(result)}
            className="flex items-center gap-2 px-4 py-2 bg-gray-800 hover:bg-gray-700 text-gray-300 rounded-lg text-sm transition-colors border border-gray-700"
          >
            <FileText className="w-4 h-4" />
            <span className="hidden sm:inline">Report</span>
          </button>
          <button
            onClick={() => downloadImage(getTabImage(), `${activeTab}_${result.filename}`)}
            className="flex items-center gap-2 px-4 py-2 bg-emerald-500 hover:bg-emerald-600 text-white rounded-lg text-sm transition-colors"
          >
            <Download className="w-4 h-4" />
            <span className="hidden sm:inline">Download</span>
          </button>
        </div>
      </div>

      {/* Verdict */}
      <div className={`rounded-2xl border p-6 ${
        result.isTampered 
          ? 'bg-red-500/5 border-red-500/20' 
          : 'bg-emerald-500/5 border-emerald-500/20'
      }`}>
        <div className="flex items-center gap-4">
          {result.isTampered ? (
            <AlertTriangle className="w-10 h-10 text-red-400" />
          ) : (
            <CheckCircle2 className="w-10 h-10 text-emerald-400" />
          )}
          <div>
            <h2 className={`text-xl font-bold ${result.isTampered ? 'text-red-400' : 'text-emerald-400'}`}>
              {result.isTampered ? '⚠️ Tampering Detected' : '✓ No Strong Evidence of Tampering'}
            </h2>
            <p className="text-sm text-gray-400 mt-1">
              Confidence: {(result.confidence * 100).toFixed(1)}% • 
              Suspicious area: {result.tamperedPercentage.toFixed(2)}% • 
              Regions: {result.detectionDetails.numberOfSuspiciousRegions}
            </p>
          </div>
        </div>
      </div>

      {/* Main Image Display */}
      <div className="bg-gray-900 rounded-2xl border border-gray-800 overflow-hidden">
        {/* Tabs */}
        <div className="flex border-b border-gray-800 overflow-x-auto">
          {tabs.map(tab => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-2 px-4 py-3 text-sm font-medium whitespace-nowrap transition-all border-b-2 ${
                  activeTab === tab.id
                    ? 'text-emerald-400 border-emerald-400 bg-emerald-500/5'
                    : 'text-gray-400 border-transparent hover:text-white hover:bg-gray-800/50'
                }`}
              >
                <Icon className="w-4 h-4" />
                {tab.label}
              </button>
            );
          })}
        </div>
        
        {/* Image */}
        <div className="p-4">
          <img 
            src={getTabImage()} 
            alt={activeTab}
            className="w-full max-h-[500px] object-contain rounded-lg"
          />
          <div className="mt-3 flex items-center justify-between">
            <p className="text-xs text-gray-500">
              {activeTab === 'heatmap' && 'Color-coded probability map: Blue (low) → Red (high suspicion)'}
              {activeTab === 'mask' && 'Binary mask: Red regions indicate detected tampered areas'}
              {activeTab === 'overlay' && 'Detected regions overlaid on original image (50% opacity)'}
              {activeTab === 'ela' && 'Error Level Analysis: Brighter regions indicate compression inconsistency'}
              {activeTab === 'restored' && 'Inpainting reconstruction of detected tampered regions'}
            </p>
            <button
              onClick={() => downloadImage(getTabImage(), `${activeTab}_${result.filename}`)}
              className="text-xs text-emerald-400 hover:text-emerald-300 flex items-center gap-1"
            >
              <Download className="w-3 h-3" /> Download
            </button>
          </div>
        </div>
      </div>

      {/* Side by side comparison */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-gray-900 rounded-xl border border-gray-800 p-4">
          <h3 className="text-sm font-semibold text-gray-300 mb-3 flex items-center gap-2">
            <Image className="w-4 h-4 text-gray-400" /> Original Image
          </h3>
          <img src={result.originalUrl} alt="Original" className="w-full rounded-lg object-contain max-h-64" />
        </div>
        <div className="bg-gray-900 rounded-xl border border-gray-800 p-4">
          <h3 className="text-sm font-semibold text-gray-300 mb-3 flex items-center gap-2">
            <RefreshCw className="w-4 h-4 text-gray-400" /> Restored Image
          </h3>
          <img src={result.restoredUrl} alt="Restored" className="w-full rounded-lg object-contain max-h-64" />
        </div>
      </div>

      {/* Forensic Signals */}
      <div className="bg-gray-900 rounded-2xl border border-gray-800 p-6">
        <h3 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
          <Info className="w-5 h-5 text-emerald-400" />
          Forensic Signals
        </h3>
        <div className="space-y-3">
          {result.forensicSignals.map((signal, idx) => (
            <div key={idx} className="bg-gray-800/50 rounded-xl p-4 border border-gray-700/50">
              <div className="flex items-center justify-between mb-2">
                <h4 className="text-sm font-semibold text-white">{signal.name}</h4>
                <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${
                  signal.severity === 'critical' ? 'bg-red-500/20 text-red-400' :
                  signal.severity === 'high' ? 'bg-orange-500/20 text-orange-400' :
                  signal.severity === 'medium' ? 'bg-yellow-500/20 text-yellow-400' :
                  'bg-gray-500/20 text-gray-400'
                }`}>
                  {signal.severity.toUpperCase()}
                </span>
              </div>
              <p className="text-xs text-gray-400 mb-1">{signal.description}</p>
              <p className="text-xs text-gray-500 font-mono">Evidence: {signal.evidence}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Model Information */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-gray-900 rounded-xl border border-gray-800 p-6">
          <h3 className="text-sm font-semibold text-gray-300 mb-4">Model Information</h3>
          <dl className="space-y-2 text-sm">
            <div className="flex justify-between">
              <dt className="text-gray-500">Model</dt>
              <dd className="text-gray-300 text-right text-xs max-w-[200px]">{result.detectionDetails.modelUsed}</dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-gray-500">Architecture</dt>
              <dd className="text-gray-300 text-right text-xs">{result.detectionDetails.backboneArchitecture}</dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-gray-500">Input Resolution</dt>
              <dd className="text-gray-300">{result.detectionDetails.inputResolution}</dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-gray-500">Processing Time</dt>
              <dd className="text-gray-300">{result.detectionDetails.processingTime.toFixed(2)}s</dd>
            </div>
          </dl>
        </div>
        <div className="bg-gray-900 rounded-xl border border-gray-800 p-6">
          <h3 className="text-sm font-semibold text-gray-300 mb-4">Detection Summary</h3>
          <dl className="space-y-2 text-sm">
            <div className="flex justify-between">
              <dt className="text-gray-500">Verdict</dt>
              <dd className={result.isTampered ? 'text-red-400' : 'text-emerald-400'}>
                {result.isTampered ? 'Tampered' : 'Authentic'}
              </dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-gray-500">Confidence</dt>
              <dd className="text-gray-300">{(result.confidence * 100).toFixed(1)}%</dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-gray-500">Tampered Area</dt>
              <dd className="text-gray-300">{result.tamperedPercentage.toFixed(2)}%</dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-gray-500">Suspicious Regions</dt>
              <dd className="text-gray-300">{result.detectionDetails.numberOfSuspiciousRegions}</dd>
            </div>
          </dl>
        </div>
      </div>

      {/* Limitations */}
      <div className="bg-amber-500/5 border border-amber-500/20 rounded-xl p-4">
        <div className="flex items-start gap-3">
          <AlertTriangle className="w-5 h-5 text-amber-400 mt-0.5 shrink-0" />
          <div>
            <h4 className="text-sm font-semibold text-amber-400 mb-1">Important Limitations</h4>
            <ul className="text-xs text-gray-400 space-y-1">
              <li>• This analysis provides forensic evidence, not absolute proof of tampering.</li>
              <li>• The restoration is a reconstruction attempt, not recovery of original pixels.</li>
              <li>• Sophisticated forgeries may evade detection (false negatives are possible).</li>
              <li>• Heavily compressed or unusual images may trigger false positives.</li>
              <li>• ELA alone cannot prove tampering — it is used as supporting evidence.</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
