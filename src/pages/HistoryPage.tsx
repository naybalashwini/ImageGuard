import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { Clock, AlertTriangle, CheckCircle2, ArrowRight, Loader2 } from 'lucide-react';
import { HistoryEntry } from '../types';
import { getHistory } from '../services/analysisService';

export function HistoryPage() {
  const [entries, setEntries] = useState<HistoryEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getHistory()
      .then(setEntries)
      .catch(err => setError(err.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="text-center py-16">
        <Loader2 className="w-8 h-8 text-emerald-400 animate-spin mx-auto mb-4" />
        <p className="text-gray-400">Loading history...</p>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold text-white">Analysis History</h1>
        <p className="text-gray-400 mt-1">Previous forensic analysis results</p>
      </div>

      {error && (
        <div className="p-4 bg-red-500/10 border border-red-500/20 rounded-xl text-sm text-red-300">
          Could not connect to backend: {error}
        </div>
      )}

      {entries.length === 0 && !error ? (
        <div className="text-center py-16 bg-gray-900 rounded-2xl border border-gray-800">
          <Clock className="w-16 h-16 text-gray-600 mx-auto mb-4" />
          <h2 className="text-xl font-bold text-white mb-2">No Analysis History</h2>
          <p className="text-gray-400 mb-6">Upload and analyze an image to see results here.</p>
          <Link
            to="/analyze"
            className="px-6 py-3 bg-emerald-500 hover:bg-emerald-600 text-white font-semibold rounded-lg"
          >
            Start Analysis
          </Link>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {entries.map(entry => (
            <Link
              key={entry.id}
              to={`/results/${entry.id}`}
              className="bg-gray-900 rounded-xl border border-gray-800 overflow-hidden hover:border-gray-700 transition-all group"
            >
              <div className="aspect-video bg-gray-800 relative overflow-hidden flex items-center justify-center">
                <div className="text-gray-600 text-sm">Analysis #{entry.id.slice(0, 8)}</div>
                <div className={`absolute top-2 right-2 px-2 py-1 rounded-full text-xs font-medium ${
                  entry.isTampered
                    ? 'bg-red-500/80 text-white'
                    : 'bg-emerald-500/80 text-white'
                }`}>
                  {entry.verdict || (entry.isTampered ? '⚠️ Tampered' : '✓ Clean')}
                </div>
              </div>
              <div className="p-4">
                <h3 className="text-sm font-medium text-white truncate">{entry.filename}</h3>
                <p className="text-xs text-gray-500 mt-1">
                  {new Date(entry.timestamp).toLocaleString()}
                </p>
                <div className="flex items-center justify-between mt-3">
                  <div className="flex items-center gap-2">
                    {entry.isTampered ? (
                      <AlertTriangle className="w-4 h-4 text-red-400" />
                    ) : (
                      <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                    )}
                    <span className="text-xs text-gray-400">
                      {entry.numRegions || 0} regions • {(entry.confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                  <ArrowRight className="w-4 h-4 text-gray-500 group-hover:text-emerald-400 transition-colors" />
                </div>
              </div>
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}
