import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { Clock, Trash2, AlertTriangle, CheckCircle2, ArrowRight } from 'lucide-react';
import { HistoryEntry } from '../types';
import { getHistoryEntries, clearHistory } from '../services/analysisService';

export function HistoryPage() {
  const [entries, setEntries] = useState<HistoryEntry[]>([]);

  useEffect(() => {
    setEntries(getHistoryEntries());
  }, []);

  const handleClear = () => {
    if (window.confirm('Clear all analysis history? This cannot be undone.')) {
      clearHistory();
      setEntries([]);
    }
  };

  return (
    <div className="space-y-8">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white">Analysis History</h1>
          <p className="text-gray-400 mt-1">Previous forensic analysis results</p>
        </div>
        {entries.length > 0 && (
          <button
            onClick={handleClear}
            className="flex items-center gap-2 px-4 py-2 bg-red-500/10 hover:bg-red-500/20 text-red-400 rounded-lg text-sm transition-colors border border-red-500/20"
          >
            <Trash2 className="w-4 h-4" />
            Clear All
          </button>
        )}
      </div>

      {entries.length === 0 ? (
        <div className="text-center py-16 bg-gray-900 rounded-2xl border border-gray-800">
          <Clock className="w-16 h-16 text-gray-600 mx-auto mb-4" />
          <h2 className="text-xl font-bold text-white mb-2">No Analysis History</h2>
          <p className="text-gray-400 mb-6">Upload and analyze an image to see results here.</p>
          <Link
            to="/analyze"
            className="px-6 py-3 bg-emerald-500 hover:bg-emerald-600 text-white font-semibold rounded-lg transition-all"
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
              <div className="aspect-video bg-gray-800 relative overflow-hidden">
                <img
                  src={entry.thumbnailUrl}
                  alt={entry.filename}
                  className="w-full h-full object-cover"
                />
                <div className={`absolute top-2 right-2 px-2 py-1 rounded-full text-xs font-medium ${
                  entry.isTampered
                    ? 'bg-red-500/80 text-white'
                    : 'bg-emerald-500/80 text-white'
                }`}>
                  {entry.isTampered ? '⚠️ Tampered' : '✓ Clean'}
                </div>
              </div>
              <div className="p-4">
                <h3 className="text-sm font-medium text-white truncate">{entry.filename}</h3>
                <p className="text-xs text-gray-500 mt-1">
                  {new Date(entry.timestamp).toLocaleDateString()} {new Date(entry.timestamp).toLocaleTimeString()}
                </p>
                <div className="flex items-center justify-between mt-3">
                  <div className="flex items-center gap-2">
                    {entry.isTampered ? (
                      <AlertTriangle className="w-4 h-4 text-red-400" />
                    ) : (
                      <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                    )}
                    <span className="text-xs text-gray-400">
                      {(entry.confidence * 100).toFixed(0)}% confidence
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
