import { Link } from 'react-router-dom';
import { Shield, Zap, Eye, RefreshCw, FileText, AlertTriangle } from 'lucide-react';

export function HomePage() {
  return (
    <div className="space-y-16">
      {/* Hero Section */}
      <section className="text-center py-12">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-sm mb-6">
          <Shield className="w-4 h-4" />
          AI-Powered Image Forensics
        </div>
        <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold text-white mb-6 leading-tight">
          Image Tampering<br />
          <span className="text-transparent bg-clip-text bg-gradient-to-r from-emerald-400 to-cyan-400">
            Detection & Restoration
          </span>
        </h1>
        <p className="text-lg text-gray-400 max-w-2xl mx-auto mb-8">
          Advanced AI forensic system that detects manipulated images, localizes tampered regions,
          generates pixel-level masks, and performs inpainting restoration.
        </p>
        <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
          <Link
            to="/analyze"
            className="px-6 py-3 bg-emerald-500 hover:bg-emerald-600 text-white font-semibold rounded-lg transition-all shadow-lg shadow-emerald-500/20"
          >
            Analyze Image
          </Link>
          <Link
            to="/about"
            className="px-6 py-3 bg-gray-800 hover:bg-gray-700 text-gray-300 font-semibold rounded-lg transition-all border border-gray-700"
          >
            Learn More
          </Link>
        </div>
      </section>

      {/* Pipeline Visualization */}
      <section className="bg-gray-900 rounded-2xl border border-gray-800 p-8">
        <h2 className="text-2xl font-bold text-white text-center mb-8">Analysis Pipeline</h2>
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4">
          {[
            { step: '1', title: 'Upload', desc: 'Image input & validation' },
            { step: '2', title: 'Preprocessing', desc: 'Resize & normalize' },
            { step: '3', title: 'ELA', desc: 'Error Level Analysis' },
            { step: '4', title: 'Detection', desc: 'AI model inference' },
            { step: '5', title: 'Localization', desc: 'Heatmap & mask' },
            { step: '6', title: 'Overlay', desc: 'Region visualization' },
            { step: '7', title: 'Restoration', desc: 'Inpainting reconstruction' },
            { step: '8', title: 'Metrics', desc: 'Quality assessment' },
            { step: '9', title: 'Report', desc: 'Forensic documentation' },
          ].map((item) => (
            <div key={item.step} className="bg-gray-800/50 rounded-xl p-4 border border-gray-700/50 text-center">
              <div className="w-8 h-8 rounded-full bg-emerald-500/20 text-emerald-400 flex items-center justify-center text-sm font-bold mx-auto mb-2">
                {item.step}
              </div>
              <h3 className="text-sm font-semibold text-white">{item.title}</h3>
              <p className="text-xs text-gray-500 mt-1">{item.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Features */}
      <section className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {[
          {
            icon: Eye,
            title: 'Tampering Localization',
            desc: 'Pixel-level detection of manipulated regions using deep learning with spatial attention mechanisms.',
            color: 'emerald',
          },
          {
            icon: Zap,
            title: 'Error Level Analysis',
            desc: 'Classical forensic technique that identifies regions with inconsistent compression artifacts.',
            color: 'amber',
          },
          {
            icon: RefreshCw,
            title: 'Image Restoration',
            desc: 'Inpainting reconstruction of detected tampered regions using learned image priors.',
            color: 'cyan',
          },
          {
            icon: Shield,
            title: 'Multi-Signal Analysis',
            desc: 'Combines texture analysis, noise patterns, color consistency, and deep features for robust detection.',
            color: 'purple',
          },
          {
            icon: FileText,
            title: 'Forensic Reports',
            desc: 'Detailed documentation of analysis results, forensic evidence, and model confidence scores.',
            color: 'blue',
          },
          {
            icon: AlertTriangle,
            title: 'Generalization Focus',
            desc: 'Designed to detect manipulations on unseen images through proper train/test separation and cross-dataset evaluation.',
            color: 'red',
          },
        ].map((feature) => {
          const Icon = feature.icon;
          return (
            <div key={feature.title} className="bg-gray-900 rounded-xl border border-gray-800 p-6 hover:border-gray-700 transition-all">
              <div className={`w-10 h-10 rounded-lg bg-${feature.color}-500/10 flex items-center justify-center mb-4`}>
                <Icon className={`w-5 h-5 text-${feature.color}-400`} />
              </div>
              <h3 className="text-lg font-semibold text-white mb-2">{feature.title}</h3>
              <p className="text-sm text-gray-400">{feature.desc}</p>
            </div>
          );
        })}
      </section>

      {/* Technology Stack */}
      <section className="bg-gray-900 rounded-2xl border border-gray-800 p-8">
        <h2 className="text-2xl font-bold text-white text-center mb-8">Technology Stack</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <div>
            <h3 className="text-lg font-semibold text-emerald-400 mb-4">AI/ML Pipeline</h3>
            <ul className="space-y-2 text-sm text-gray-400">
              <li>• <span className="text-white">Localization Model:</span> ManTraNet-style CNN (Encoder-Decoder)</li>
              <li>• <span className="text-white">Backbone:</span> ResNet-50 pretrained on ImageNet</li>
              <li>• <span className="text-white">Framework:</span> PyTorch with transfer learning</li>
              <li>• <span className="text-white">ELA Engine:</span> JPEG recompression + pixel difference</li>
              <li>• <span className="text-white">Restoration:</span> LaMa / OpenCV inpainting</li>
              <li>• <span className="text-white">Evaluation:</span> IoU, Dice, F1, PSNR, SSIM</li>
            </ul>
          </div>
          <div>
            <h3 className="text-lg font-semibold text-cyan-400 mb-4">Application Stack</h3>
            <ul className="space-y-2 text-sm text-gray-400">
              <li>• <span className="text-white">Frontend:</span> React + Vite + Tailwind CSS</li>
              <li>• <span className="text-white">Backend:</span> FastAPI (Python)</li>
              <li>• <span className="text-white">Database:</span> SQLite / PostgreSQL</li>
              <li>• <span className="text-white">Deployment:</span> Docker + Docker Compose</li>
              <li>• <span className="text-white">GPU Support:</span> CUDA / cuDNN when available</li>
              <li>• <span className="text-white">Report:</span> PDF generation with forensic details</li>
            </ul>
          </div>
        </div>
      </section>

      {/* Disclaimer */}
      <section className="bg-amber-500/5 border border-amber-500/20 rounded-xl p-6">
        <div className="flex items-start gap-3">
          <AlertTriangle className="w-5 h-5 text-amber-400 mt-0.5 shrink-0" />
          <div>
            <h3 className="text-sm font-semibold text-amber-400 mb-1">Important Disclaimer</h3>
            <p className="text-sm text-gray-400">
              This system provides forensic analysis as supporting evidence. Results are based on statistical analysis 
              and AI model predictions. The system may produce false positives or miss sophisticated forgeries. 
              No single tool can definitively prove or disprove image authenticity. Always consider results 
              alongside other evidence and expert analysis.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
}
