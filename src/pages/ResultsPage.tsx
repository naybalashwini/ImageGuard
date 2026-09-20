import { useParams, Link } from 'react-router-dom';
import { useEffect, useState } from 'react';
import { 
  ArrowLeft, Download, AlertTriangle, CheckCircle2, XCircle, 
  Map, Layers, Eye, RefreshCw, FileText, Info, Target
} from 'lucide-react';
import { AnalysisResult } from '../types';
import { getResults, getDownloadUrl } from '../services/analysisService';

export function ResultsPage() {
  const { id } = useParams<{ id: string }>();
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<string>('heatmap');

  useEffect(() => {
    if (id) {
      getResults(id)
        .then(setResult)
        .catch(() => setResult(null))
        .finally(() => setLoading(false));
    }
  }, [id]);

  if (loading) {
    return (
      <div className="text-center py-16">
        <div className="animate-spin w-8 h-8 border-2 border-emerald-400 border-t-transparent rounded-full mx-auto mb-4" />
        <p className="text-gray-400">Loading results...</p>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="text-center py-16">
        <XCircle className="w-16 h-16 text-gray-600 mx-auto mb-4" />
        <h2 className="text-xl font-bold text-white mb-2">Analysis Not Found</h2>
        <Link to="/analyze" className="px-4 py-2 bg-emerald-500 text-white rounded-lg hover:bg-emerald-600">
          Start New Analysis
        </Link>
      </div>
    );
  }

  const tabs = [
    { id: 'heatmap', label: 'Heatmap', icon: Map, url: result.heatmapUrl },
    { id: 'mask', label: 'Mask', icon: Layers, url: result.maskUrl },
    { id: 'overlay', label: 'Overlay', icon: Eye, url: result.overlayUrl },
    { id: 'contours', label: 'Contours', icon: Target, url: result.contoursUrl },
    { id: 'confidence', label: 'Confidence', icon: Info, url: result.confidenceMapUrl },
    { id: 'ela', label: 'ELA', icon: Layers, url: result.elaUrl },
    { id: 'restored', label: 'Restored', icon: RefreshCw, url: result.restoredUrl },
  ];

  const activeTabData = tabs.find(t => t.id === activeTab);

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <Link to="/analyze" className="p-2 hover:bg-gray-800 rounded-lg">
            <ArrowLeft className="w-5 h-5 text-gray-400" />
          </Link>
          <div>
            <h1 className="text-2xl font-bold text-white">Forensic Analysis Results</h1>
            <p className="text-sm text-gray-500">
              ID: {result.id.slice(0, 8)} • Model: {result.detectionDetails.modelUsed}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <a
            href={getDownloadUrl(result.id, 'report')}
            className="flex items-center gap-2 px-4 py-2 bg-gray-800 hover:bg-gray-700 text-gray-300 rounded-lg text-sm border border-gray-700"
          >
            <FileText className="w-4 h-4" />
            <span className="hidden sm:inline">Report</span>
          </a>
          {activeTabData?.url && (
            <a
              href={activeTabData.url}
              download={`${activeTab}_${result.filename}`}
              className="flex items-center gap-2 px-4 py-2 bg-emerald-500 hover:bg-emerald-600 text-white rounded-lg text-sm"
            >
              <Download className="w-4 h-4" />
              <span className="hidden sm:inline">Download</span>
            </a>
          )}
        </div>
      </div>

      {/* Verdict */}
      <div className={`rounded-2xl border p-6 ${
        result.isTampered ? 'bg-red-500/5 border-red-500/20' : 'bg-emerald-500/5 border-emerald-500/20'
      }`}>
        <div className="flex items-center gap-4">
          {result.isTampered ? (
            <AlertTriangle className="w-10 h-10 text-red-400" />
          ) : (
            <CheckCircle2 className="w-10 h-10 text-emerald-400" />
          )}
          <div>
            <h2 className={`text-xl font-bold ${result.isTampered ? 'text-red-400' : 'text-emerald-400'}`}>
              {result.verdict?.verdict_description || (result.isTampered ? '⚠️ Tampering Detected' : '✓ No Strong Evidence')}
            </h2>
            <p className="text-sm text-gray-400 mt-1">
              Confidence: {(result.confidence * 100).toFixed(1)}% •
              Regions: {result.detectionDetails.numberOfSuspiciousRegions} •
              Area: {result.tamperedPercentage.toFixed(2)}%
            </p>
          </div>
        </div>
      </div>

      {/* Main Image Display */}
      <div className="bg-gray-900 rounded-2xl border border-gray-800 overflow-hidden">
        <div className="flex border-b border-gray-800 overflow-x-auto">
          {tabs.map(tab => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                disabled={!tab.url}
                className={`flex items-center gap-2 px-4 py-3 text-sm font-medium whitespace-nowrap border-b-2 transition-all ${
                  !tab.url ? 'opacity-30 cursor-not-allowed' :
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
        
        <div className="p-4">
          {activeTabData?.url ? (
            <img 
              src={activeTabData.url} 
              alt={activeTab}
              className="w-full max-h-[500px] object-contain rounded-lg"
            />
          ) : (
            <div className="text-center py-12 text-gray-500">No visualization available</div>
          )}
          <p className="text-xs text-gray-500 mt-3">
            {activeTab === 'heatmap' && 'Color-coded probability map from TruFor: Blue (low) → Red (high suspicion)'}
            {activeTab === 'mask' && 'Binary mask after morphological cleanup and connected component filtering'}
            {activeTab === 'overlay' && 'Detected regions overlaid on original image'}
            {activeTab === 'contours' && 'Contour boundaries of detected tampered regions'}
            {activeTab === 'confidence' && 'Confidence/reliability map from TruFor'}
            {activeTab === 'ela' && 'Error Level Analysis: compression artifact visualization'}
            {activeTab === 'restored' && 'Inpainting reconstruction of detected regions'}
          </p>
        </div>
      </div>

      {/* Detected Regions */}
      {result.regions && result.regions.length > 0 && (
        <div className="bg-gray-900 rounded-2xl border border-gray-800 p-6">
          <h3 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
            <Target className="w-5 h-5 text-emerald-400" />
            Detected Regions ({result.regions.length})
          </h3>
          <div className="space-y-3">
            {result.regions.map((region, idx) => (
              <div key={idx} className="bg-gray-800/50 rounded-xl p-4 border border-gray-700/50">
                <div className="flex items-center justify-between mb-2">
                  <h4 className="text-sm font-semibold text-white">Region {idx + 1}</h4>
                  <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${
                    region.severity === 'critical' ? 'bg-red-500/20 text-red-400' :
                    region.severity === 'high' ? 'bg-orange-500/20 text-orange-400' :
                    region.severity === 'medium' ? 'bg-yellow-500/20 text-yellow-400' :
                    'bg-gray-500/20 text-gray-400'
                  }`}>
                    {region.severity.toUpperCase()}
                  </span>
                </div>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
                  <div>
                    <span className="text-gray-500">Confidence:</span>
                    <span className="text-white ml-1">{(region.region_confidence * 100).toFixed(1)}%</span>
                  </div>
                  <div>
                    <span className="text-gray-500">Probability:</span>
                    <span className="text-white ml-1">{(region.mean_probability * 100).toFixed(1)}%</span>
                  </div>
                  <div>
                    <span className="text-gray-500">Area:</span>
                    <span className="text-white ml-1">{region.area_percentage.toFixed(2)}%</span>
                  </div>
                  <div>
                    <span className="text-gray-500">BBox:</span>
                    <span className="text-white ml-1">{region.bbox.width}×{region.bbox.height}px</span>
                  </div>
                </div>
                <p className="text-xs text-gray-400 mt-2">{region.description}</p>
              </div>
            ))}
          </div>
        </div>
      )}

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
                <span className={`text-xs px-2 py-0.5 rounded-full ${
                  signal.severity === 'critical' ? 'bg-red-500/20 text-red-400' :
                  signal.severity === 'high' ? 'bg-orange-500/20 text-orange-400' :
                  signal.severity === 'medium' ? 'bg-yellow-500/20 text-yellow-400' :
                  'bg-gray-500/20 text-gray-400'
                }`}>
                  {signal.severity.toUpperCase()}
                </span>
              </div>
              <p className="text-xs text-gray-400">{signal.description}</p>
              <p className="text-xs text-gray-500 font-mono mt-1">Evidence: {signal.evidence}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Side by side */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-gray-900 rounded-xl border border-gray-800 p-4">
          <h3 className="text-sm font-semibold text-gray-300 mb-3">Original Image</h3>
          {result.originalUrl && (
            <img src={result.originalUrl} alt="Original" className="w-full rounded-lg object-contain max-h-64" />
          )}
        </div>
        <div className="bg-gray-900 rounded-xl border border-gray-800 p-4">
          <h3 className="text-sm font-semibold text-gray-300 mb-3">Restored Image</h3>
          {result.restoredUrl && (
            <img src={result.restoredUrl} alt="Restored" className="w-full rounded-lg object-contain max-h-64" />
          )}
        </div>
      </div>

      {/* Warning */}
      <div className="bg-amber-500/5 border border-amber-500/20 rounded-xl p-4">
        <div className="flex items-start gap-3">
          <AlertTriangle className="w-5 h-5 text-amber-400 mt-0.5 shrink-0" />
          <div>
            <h4 className="text-sm font-semibold text-amber-400 mb-1">Important</h4>
            <p className="text-xs text-gray-400">
              {result.verdict?.warning || 'AI forensic detection is probabilistic. Results are supporting evidence, not absolute proof.'}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
