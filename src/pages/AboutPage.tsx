import { Shield, Brain, Database, Cpu, BookOpen, AlertTriangle } from 'lucide-react';

export function AboutPage() {
  return (
    <div className="max-w-4xl mx-auto space-y-12">
      <div className="text-center">
        <h1 className="text-3xl font-bold text-white mb-2">About This System</h1>
        <p className="text-gray-400">Image Tampering Detection & Restoration — Technical Documentation</p>
      </div>

      {/* Problem Statement */}
      <section className="bg-gray-900 rounded-2xl border border-gray-800 p-8">
        <h2 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
          <BookOpen className="w-5 h-5 text-emerald-400" />
          Problem Statement
        </h2>
        <p className="text-gray-400 leading-relaxed">
          Digital image manipulation has become increasingly sophisticated with the advent of AI-powered editing tools, 
          generative fill, and deepfake technology. Traditional forensic methods are insufficient for detecting modern 
          manipulations. This system addresses the need for an automated, AI-powered forensic analysis tool that can:
        </p>
        <ul className="mt-4 space-y-2 text-sm text-gray-400">
          <li className="flex items-start gap-2">
            <span className="text-emerald-400">•</span>
            Detect whether an image has been tampered with
          </li>
          <li className="flex items-start gap-2">
            <span className="text-emerald-400">•</span>
            Localize the exact regions that were manipulated
          </li>
          <li className="flex items-start gap-2">
            <span className="text-emerald-400">•</span>
            Generate pixel-level masks and heatmaps for evidence
          </li>
          <li className="flex items-start gap-2">
            <span className="text-emerald-400">•</span>
            Attempt restoration/inpainting of tampered regions
          </li>
          <li className="flex items-start gap-2">
            <span className="text-emerald-400">•</span>
            Generalize to previously unseen images
          </li>
        </ul>
      </section>

      {/* Model Architecture */}
      <section className="bg-gray-900 rounded-2xl border border-gray-800 p-8">
        <h2 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
          <Brain className="w-5 h-5 text-cyan-400" />
          Model Architecture
        </h2>
        <div className="space-y-4">
          <div>
            <h3 className="text-sm font-semibold text-emerald-400 mb-2">Primary Localization Model</h3>
            <p className="text-sm text-gray-400">
              <strong className="text-white">ManTraNet-style CNN</strong> — An encoder-decoder architecture designed 
              specifically for image forgery localization. The encoder uses a ResNet-50 backbone pretrained on ImageNet 
              (transfer learning), which extracts multi-scale features. The decoder upsamples these features to produce 
              a pixel-level prediction map indicating the probability of each pixel being manipulated.
            </p>
          </div>
          <div>
            <h3 className="text-sm font-semibold text-emerald-400 mb-2">Why This Architecture?</h3>
            <ul className="text-sm text-gray-400 space-y-1">
              <li>• <span className="text-white">Spatial localization:</span> Produces per-pixel predictions, not just image-level classification</li>
              <li>• <span className="text-white">Transfer learning:</span> ImageNet-pretrained backbone provides strong feature extraction for unseen images</li>
              <li>• <span className="text-white">Generalization:</span> Architecture learns manipulation artifacts, not dataset-specific patterns</li>
              <li>• <span className="text-white">Efficiency:</span> ResNet-50 backbone is well-optimized and runs on consumer GPUs</li>
              <li>• <span className="text-white">Proven:</span> ManTraNet architecture has demonstrated strong cross-dataset performance</li>
            </ul>
          </div>
          <div>
            <h3 className="text-sm font-semibold text-emerald-400 mb-2">Supporting Forensic Techniques</h3>
            <ul className="text-sm text-gray-400 space-y-1">
              <li>• <span className="text-white">ELA (Error Level Analysis):</span> Recompresses image at known quality, compares differences. Regions with higher error levels may indicate post-processing.</li>
              <li>• <span className="text-white">Texture Analysis:</span> Block-based variance analysis detects regions with inconsistent texture statistics.</li>
              <li>• <span className="text-white">Noise Residual:</span> Analyzes noise patterns for inconsistencies across regions.</li>
              <li>• <span className="text-white">Color Consistency:</span> Checks for abnormal color channel distributions.</li>
            </ul>
          </div>
        </div>
      </section>

      {/* Dataset Strategy */}
      <section className="bg-gray-900 rounded-2xl border border-gray-800 p-8">
        <h2 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
          <Database className="w-5 h-5 text-purple-400" />
          Dataset & Generalization Strategy
        </h2>
        <div className="space-y-4">
          <div>
            <h3 className="text-sm font-semibold text-emerald-400 mb-2">Datasets Used</h3>
            <div className="overflow-x-auto">
              <table className="w-full text-sm text-gray-400">
                <thead>
                  <tr className="border-b border-gray-700">
                    <th className="text-left py-2 text-gray-300">Dataset</th>
                    <th className="text-left py-2 text-gray-300">Type</th>
                    <th className="text-left py-2 text-gray-300">Images</th>
                    <th className="text-left py-2 text-gray-300">Masks</th>
                    <th className="text-left py-2 text-gray-300">Use</th>
                  </tr>
                </thead>
                <tbody className="text-xs">
                  <tr className="border-b border-gray-800">
                    <td className="py-2">CASIA v2.0</td>
                    <td>Splicing + Copy-move</td>
                    <td>~12,000</td>
                    <td>Yes</td>
                    <td>Training</td>
                  </tr>
                  <tr className="border-b border-gray-800">
                    <td className="py-2">Columbia</td>
                    <td>Splicing</td>
                    <td>~1,200</td>
                    <td>Yes</td>
                    <td>Training</td>
                  </tr>
                  <tr className="border-b border-gray-800">
                    <td className="py-2">Coverage</td>
                    <td>Copy-move</td>
                    <td>100 pairs</td>
                    <td>Yes</td>
                    <td>Cross-dataset Test</td>
                  </tr>
                  <tr className="border-b border-gray-800">
                    <td className="py-2">NIST Nimble</td>
                    <td>Various</td>
                    <td>~6,000</td>
                    <td>Yes</td>
                    <td>Validation</td>
                  </tr>
                  <tr>
                    <td className="py-2">DEFACTO</td>
                    <td>Modern editing</td>
                    <td>~1,000</td>
                    <td>Yes</td>
                    <td>Unseen Test</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
          <div>
            <h3 className="text-sm font-semibold text-emerald-400 mb-2">Anti-Leakage Strategy</h3>
            <ul className="text-sm text-gray-400 space-y-1">
              <li>• <span className="text-white">Strict separation:</span> Training images never appear in test/validation sets</li>
              <li>• <span className="text-white">Cross-dataset evaluation:</span> Model trained on CASIA+Columbia, tested on Coverage+DEFACTO</li>
              <li>• <span className="text-white">No near-duplicates:</span> Perceptual hashing used to remove near-duplicate images across splits</li>
              <li>• <span className="text-white">Multiple manipulation types:</span> Evaluated on splicing, copy-move, object removal, and AI editing</li>
            </ul>
          </div>
        </div>
      </section>

      {/* Restoration */}
      <section className="bg-gray-900 rounded-2xl border border-gray-800 p-8">
        <h2 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
          <Cpu className="w-5 h-5 text-amber-400" />
          Restoration Pipeline
        </h2>
        <div className="space-y-4">
          <p className="text-sm text-gray-400">
            After detecting and localizing tampered regions, the system attempts to restore/inpaint 
            those regions. This is a <strong className="text-white">reconstruction</strong>, not a recovery 
            of the original pixels.
          </p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="bg-gray-800/50 rounded-xl p-4 border border-gray-700/50">
              <h4 className="text-sm font-semibold text-white mb-2">LaMa (Large Mask Inpainting)</h4>
              <p className="text-xs text-gray-400">
                State-of-the-art learned inpainting model. Uses Fast Fourier Convolution to handle 
                large masked regions. Best for complex object removal and textured regions.
              </p>
            </div>
            <div className="bg-gray-800/50 rounded-xl p-4 border border-gray-700/50">
              <h4 className="text-sm font-semibold text-white mb-2">OpenCV Inpainting (Fallback)</h4>
              <p className="text-xs text-gray-400">
                Telea and Navier-Stokes methods. Suitable for small regions and simple backgrounds. 
                Used when LaMa is unavailable or for real-time processing.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Evaluation */}
      <section className="bg-gray-900 rounded-2xl border border-gray-800 p-8">
        <h2 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
          <Shield className="w-5 h-5 text-emerald-400" />
          Evaluation Metrics
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <h4 className="text-sm font-semibold text-emerald-400 mb-2">Detection</h4>
            <ul className="text-xs text-gray-400 space-y-1">
              <li>• Accuracy</li>
              <li>• Precision / Recall</li>
              <li>• F1 Score</li>
              <li>• ROC-AUC</li>
            </ul>
          </div>
          <div>
            <h4 className="text-sm font-semibold text-cyan-400 mb-2">Localization</h4>
            <ul className="text-xs text-gray-400 space-y-1">
              <li>• IoU (Intersection over Union)</li>
              <li>• Dice Coefficient</li>
              <li>• Pixel Precision</li>
              <li>• Pixel Recall / F1</li>
            </ul>
          </div>
          <div>
            <h4 className="text-sm font-semibold text-amber-400 mb-2">Restoration</h4>
            <ul className="text-xs text-gray-400 space-y-1">
              <li>• PSNR (Peak Signal-to-Noise Ratio)</li>
              <li>• SSIM (Structural Similarity)</li>
              <li>• LPIPS (Learned Perceptual)</li>
            </ul>
          </div>
        </div>
      </section>

      {/* Limitations */}
      <section className="bg-amber-500/5 border border-amber-500/20 rounded-2xl p-8">
        <h2 className="text-xl font-bold text-amber-400 mb-4 flex items-center gap-2">
          <AlertTriangle className="w-5 h-5" />
          Known Limitations
        </h2>
        <ul className="text-sm text-gray-400 space-y-2">
          <li>• <span className="text-white">No absolute proof:</span> Forensic analysis provides evidence, not certainty. Results should be combined with other investigation methods.</li>
          <li>• <span className="text-white">Sophisticated forgeries:</span> Anti-forensic techniques (e.g., adding noise to hide ELA, careful blending) may evade detection.</li>
          <li>• <span className="text-white">Restoration is approximation:</span> The system cannot recover the exact original pixels. It generates a plausible reconstruction.</li>
          <li>• <span className="text-white">False positives:</span> Images with heavy compression, unusual content, or artistic filters may trigger false detections.</li>
          <li>• <span className="text-white">Computational requirements:</span> Full AI pipeline requires GPU for optimal performance. CPU-only mode is slower.</li>
          <li>• <span className="text-white">Training data bias:</span> Model performance depends on training data diversity. Novel manipulation types may not be detected.</li>
          <li>• <span className="text-white">Resolution limits:</span> Very large images may need to be resized, potentially losing fine-grained manipulation evidence.</li>
        </ul>
      </section>

      {/* Future Scope */}
      <section className="bg-gray-900 rounded-2xl border border-gray-800 p-8">
        <h2 className="text-xl font-bold text-white mb-4">Future Scope</h2>
        <ul className="text-sm text-gray-400 space-y-2">
          <li>• Integration of vision transformer (ViT) based forensic models for improved generalization</li>
          <li>• Diffusion-based restoration for higher quality inpainting</li>
          <li>• Video forensics extension (temporal consistency analysis)</li>
          <li>• Metadata analysis integration (EXIF, camera fingerprint)</li>
          <li>• Ensemble of multiple forensic models for improved robustness</li>
          <li>• Real-time detection using model distillation and optimization</li>
          <li>• Adversarial training to improve resistance to anti-forensic attacks</li>
        </ul>
      </section>
    </div>
  );
}
