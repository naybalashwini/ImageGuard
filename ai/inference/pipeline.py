"""
Forensic Inference Pipeline

This module implements the complete inference pipeline for image tampering detection.
It combines:
1. Preprocessing
2. Error Level Analysis (ELA)
3. Deep learning-based forgery localization
4. Post-processing (mask generation, refinement)
5. Image restoration/inpainting
"""

import os
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
from PIL import Image

logger = logging.getLogger(__name__)


class ForensicPipeline:
    """
    Complete forensic analysis pipeline.
    
    Architecture:
    - Localization: ManTraNet-style CNN (ResNet-50 encoder + attention decoder)
    - ELA: Classical JPEG recompression analysis
    - Restoration: LaMa (Large Mask Inpainting) or OpenCV fallback
    
    The model uses transfer learning from ImageNet-pretrained ResNet-50,
    which provides strong feature extraction that generalizes to unseen images.
    """
    
    def __init__(self, model_dir: str = "ai/models/weights", device: str = "auto"):
        self.model_dir = Path(model_dir)
        self.device = self._select_device(device)
        self.localization_model = None
        self.restoration_model = None
        self._models_loaded = False
    
    def _select_device(self, device: str) -> str:
        """Select compute device."""
        if device == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    logger.info("Using CUDA GPU")
                    return "cuda"
            except ImportError:
                pass
            logger.info("Using CPU (install torch with CUDA for GPU support)")
            return "cpu"
        return device
    
    def load_models(self):
        """Load all required models."""
        self._load_localization_model()
        self._load_restoration_model()
        self._models_loaded = True
        logger.info("All models loaded successfully")
    
    def _load_localization_model(self):
        """Load the forgery localization model."""
        try:
            from ai.models.forgery_localization import ForgeryLocalizationModel
            self.localization_model = ForgeryLocalizationModel(
                model_dir=str(self.model_dir),
                device=self.device
            )
            self.localization_model.load()
        except Exception as e:
            logger.warning(f"Could not load localization model: {e}")
            logger.info("Falling back to classical forensic analysis")
            self.localization_model = None
    
    def _load_restoration_model(self):
        """Load the image restoration model."""
        try:
            from ai.restoration.inpainting import InpaintingModel
            self.restoration_model = InpaintingModel(device=self.device)
            self.restoration_model.load()
        except Exception as e:
            logger.warning(f"Could not load restoration model: {e}")
            logger.info("Will use OpenCV inpainting as fallback")
            self.restoration_model = None
    
    def analyze(self, image_path: str) -> Dict:
        """
        Complete forensic analysis pipeline.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Dictionary containing all analysis results
        """
        logger.info(f"Starting analysis of: {image_path}")
        
        # Load image
        original = Image.open(image_path).convert('RGB')
        original_array = np.array(original)
        
        # Step 1: Preprocessing
        preprocessed = self._preprocess(original)
        
        # Step 2: Error Level Analysis
        ela_result = self._compute_ela(original)
        
        # Step 3: Forensic Feature Extraction & Localization
        if self.localization_model is not None:
            localization_map = self.localization_model.predict(preprocessed)
        else:
            # Fallback: classical forensic analysis
            localization_map = self._classical_forensic_analysis(original_array, ela_result)
        
        # Step 4: Post-processing
        heatmap, mask, overlay = self._postprocess(localization_map, original_array)
        
        # Step 5: Restoration
        restored = self._restore(original, mask)
        
        # Calculate metrics
        tampered_pixels = np.sum(mask[:, :, 3] > 128)
        total_pixels = mask.shape[0] * mask.shape[1]
        tampered_percentage = (tampered_pixels / total_pixels) * 100
        
        # Determine verdict
        is_tampered = tampered_percentage > 2.0  # More than 2% suspicious area
        
        # Confidence based on localization map intensity
        confidence = float(np.mean(localization_map[localization_map > 0.3])) if np.any(localization_map > 0.3) else 0.1
        confidence = min(0.95, max(0.1, confidence))
        
        # Generate forensic signals
        forensic_signals = self._generate_forensic_signals(
            original_array, ela_result, localization_map, tampered_percentage
        )
        
        return {
            'is_tampered': is_tampered,
            'confidence': confidence,
            'tampered_percentage': tampered_percentage,
            'heatmap': Image.fromarray(heatmap),
            'mask': Image.fromarray(mask),
            'overlay': Image.fromarray(overlay),
            'ela': ela_result['ela_image'],
            'restored': restored,
            'model_info': {
                'model': 'ManTraNet-style CNN' if self.localization_model else 'Classical Forensic Analysis',
                'backbone': 'ResNet-50 (ImageNet pretrained)' if self.localization_model else 'N/A',
                'device': self.device,
            },
            'forensic_signals': forensic_signals,
            'metrics': {},
        }
    
    def _preprocess(self, image: Image.Image) -> np.ndarray:
        """Preprocess image for model input."""
        # Resize to model input size while maintaining aspect ratio
        target_size = 512
        w, h = image.size
        scale = target_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        resized = image.resize((new_w, new_h), Image.BILINEAR)
        array = np.array(resized).astype(np.float32) / 255.0
        
        # Normalize with ImageNet statistics
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        normalized = (array - mean) / std
        
        return normalized
    
    def _compute_ela(self, image: Image.Image) -> Dict:
        """
        Compute Error Level Analysis.
        
        ELA works by:
        1. Saving the image at a known JPEG quality (e.g., 90)
        2. Computing the pixel-wise difference between original and recompressed
        3. Scaling the difference for visualization
        
        Regions with higher error levels may indicate post-processing.
        """
        import io
        
        # Recompress at quality 90
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=90)
        buffer.seek(0)
        recompressed = Image.open(buffer).convert('RGB')
        
        # Compute difference
        original_array = np.array(image).astype(np.float32)
        recompressed_array = np.array(recompressed).astype(np.float32)
        
        # Resize if needed (JPEG compression might change size slightly)
        if original_array.shape != recompressed_array.shape:
            recompressed = recompressed.resize(image.size, Image.BILINEAR)
            recompressed_array = np.array(recompressed).astype(np.float32)
        
        difference = np.abs(original_array - recompressed_array)
        
        # Scale for visualization
        scale = 15
        ela_scaled = np.clip(difference * scale, 0, 255).astype(np.uint8)
        ela_image = Image.fromarray(ela_scaled)
        
        # Calculate ELA statistics
        ela_mean = np.mean(difference)
        ela_std = np.std(difference)
        ela_max = np.max(difference)
        
        # Block-level analysis for anomaly detection
        block_size = 16
        h, w = difference.shape[:2]
        block_means = []
        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block = difference[y:y+block_size, x:x+block_size]
                block_means.append(np.mean(block))
        
        block_means = np.array(block_means)
        block_anomaly_score = np.std(block_means) / (np.mean(block_means) + 1e-8)
        
        return {
            'ela_image': ela_image,
            'ela_mean': float(ela_mean),
            'ela_std': float(ela_std),
            'ela_max': float(ela_max),
            'block_anomaly_score': float(block_anomaly_score),
            'difference_raw': difference,
        }
    
    def _classical_forensic_analysis(
        self, 
        image_array: np.ndarray, 
        ela_result: Dict
    ) -> np.ndarray:
        """
        Classical forensic analysis when deep learning model is unavailable.
        
        Combines:
        1. ELA anomaly detection
        2. Local texture variance analysis
        3. Noise pattern inconsistency
        4. Color channel analysis
        
        Returns a localization probability map [0, 1].
        """
        h, w = image_array.shape[:2]
        gray = np.mean(image_array, axis=2)
        
        # 1. ELA-based localization
        ela_diff = ela_result['difference_raw']
        ela_gray = np.mean(ela_diff, axis=2) if ela_diff.ndim == 3 else ela_diff
        
        # Resize ELA to match image if needed
        if ela_gray.shape != (h, w):
            ela_img = Image.fromarray((ela_gray * 255 / (ela_gray.max() + 1e-8)).astype(np.uint8))
            ela_img = ela_img.resize((w, h), Image.BILINEAR)
            ela_gray = np.array(ela_img).astype(np.float32) / 255.0
        
        ela_map = ela_gray / (ela_gray.max() + 1e-8)
        
        # 2. Texture variance analysis (block-based)
        block_size = 32
        variance_map = np.zeros((h, w), dtype=np.float32)
        
        for y in range(0, h - block_size, block_size // 2):
            for x in range(0, w - block_size, block_size // 2):
                block = gray[y:y+block_size, x:x+block_size]
                var = np.var(block)
                variance_map[y:y+block_size, x:x+block_size] = np.maximum(
                    variance_map[y:y+block_size, x:x+block_size], var
                )
        
        # Normalize variance map
        var_mean = np.mean(variance_map[variance_map > 0]) if np.any(variance_map > 0) else 1
        var_std = np.std(variance_map[variance_map > 0]) if np.any(variance_map > 0) else 1
        texture_map = np.clip((variance_map - var_mean) / (var_std + 1e-8), 0, 3) / 3
        
        # 3. Noise residual analysis
        # Estimate noise using Laplacian
        from scipy.ndimage import laplace
        noise_residual = np.abs(laplace(gray))
        noise_map = noise_residual / (noise_residual.max() + 1e-8)
        
        # 4. Combine signals
        # Weight based on reliability
        localization_map = (
            0.4 * ela_map +
            0.3 * texture_map +
            0.3 * noise_map
        )
        
        # Normalize to [0, 1]
        localization_map = localization_map / (localization_map.max() + 1e-8)
        
        return localization_map
    
    def _postprocess(
        self, 
        localization_map: np.ndarray, 
        original_array: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Post-process localization map to generate heatmap, mask, and overlay.
        
        Steps:
        1. Generate heatmap (color-coded probability)
        2. Generate binary mask (threshold + morphological operations)
        3. Generate overlay (mask on original image)
        """
        h, w = localization_map.shape
        
        # 1. Heatmap
        heatmap = np.zeros((h, w, 4), dtype=np.uint8)
        
        # Map probability to color (jet colormap approximation)
        for y in range(h):
            for x in range(w):
                val = localization_map[y, x]
                if val > 0.1:
                    # Hot colormap: blue -> cyan -> green -> yellow -> red
                    if val < 0.25:
                        r, g, b = 0, int(val * 4 * 255), 255
                    elif val < 0.5:
                        r, g, b = 0, 255, int((1 - (val - 0.25) * 4) * 255)
                    elif val < 0.75:
                        r, g, b = int((val - 0.5) * 4 * 255), 255, 0
                    else:
                        r, g, b = 255, int((1 - (val - 0.75) * 4) * 255), 0
                    
                    heatmap[y, x] = [r, g, b, int(val * 200)]
        
        # 2. Binary mask
        # Adaptive thresholding
        threshold = np.mean(localization_map) + 1.5 * np.std(localization_map)
        threshold = max(0.3, min(0.7, threshold))
        
        binary = (localization_map > threshold).astype(np.uint8) * 255
        
        # Morphological operations to clean up mask
        try:
            import cv2
            kernel = np.ones((5, 5), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        except ImportError:
            pass  # Skip morphological ops if cv2 not available
        
        mask = np.zeros((h, w, 4), dtype=np.uint8)
        mask[binary > 128] = [255, 50, 50, 200]
        
        # 3. Overlay
        overlay = original_array.copy()
        mask_region = binary > 128
        overlay[mask_region] = (
            overlay[mask_region] * 0.5 + np.array([255, 0, 0]) * 0.5
        ).astype(np.uint8)
        
        return heatmap, mask, overlay
    
    def _restore(self, original: Image.Image, mask: np.ndarray) -> Image.Image:
        """
        Restore/inpaint the tampered regions.
        
        Uses LaMa if available, otherwise falls back to OpenCV inpainting.
        """
        original_array = np.array(original)
        mask_binary = (mask[:, :, 3] > 128).astype(np.uint8) * 255
        
        if self.restoration_model is not None:
            # Use LaMa
            restored = self.restoration_model.inpaint(original_array, mask_binary)
            return Image.fromarray(restored)
        
        # Fallback: OpenCV inpainting
        try:
            import cv2
            # Use Telea's method for inpainting
            restored = cv2.inpaint(
                cv2.cvtColor(original_array, cv2.COLOR_RGB2BGR),
                mask_binary,
            inpaintRadius=5,
                flags=cv2.INPAINT_TELEA
            )
            restored = cv2.cvtColor(restored, cv2.COLOR_BGR2RGB)
            return Image.fromarray(restored)
        except ImportError:
            # Final fallback: simple blur-based inpainting
            from PIL import ImageFilter
            blurred = original.filter(ImageFilter.GaussianBlur(radius=10))
            result = original.copy()
            result_array = np.array(result)
            blurred_array = np.array(blurred)
            mask_region = mask[:, :, 3] > 128
            result_array[mask_region] = blurred_array[mask_region]
            return Image.fromarray(result_array)
    
    def _generate_forensic_signals(
        self,
        image_array: np.ndarray,
        ela_result: Dict,
        localization_map: np.ndarray,
        tampered_percentage: float
    ) -> list:
        """Generate forensic signal descriptions."""
        signals = []
        
        # ELA signal
        if ela_result['block_anomaly_score'] > 0.5:
            signals.append({
                'name': 'ELA Compression Anomaly',
                'description': 'Error Level Analysis shows inconsistent compression patterns across image regions, suggesting possible post-processing or splicing.',
                'severity': 'high' if ela_result['block_anomaly_score'] > 1.0 else 'medium',
                'evidence': f'Block anomaly score: {ela_result["block_anomaly_score"]:.3f}'
            })
        
        # Localization signal
        high_confidence_pixels = np.sum(localization_map > 0.7)
        total_pixels = localization_map.size
        high_conf_ratio = high_confidence_pixels / total_pixels
        
        if high_conf_ratio > 0.01:
            signals.append({
                'name': 'Localization Confidence',
                'description': f'{high_conf_ratio*100:.1f}% of pixels have high manipulation probability (>0.7).',
                'severity': 'high' if high_conf_ratio > 0.05 else 'medium',
                'evidence': f'High-confidence area: {high_conf_ratio*100:.2f}% of image'
            })
        
        # Texture signal
        gray = np.mean(image_array, axis=2)
        block_size = 32
        h, w = gray.shape
        block_vars = []
        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block_vars.append(np.var(gray[y:y+block_size, x:x+block_size]))
        
        if len(block_vars) > 0:
            cv = np.std(block_vars) / (np.mean(block_vars) + 1e-8)
            if cv > 0.8:
                signals.append({
                    'name': 'Texture Inconsistency',
                    'description': 'Significant variation in local texture statistics suggests possible region-level manipulation.',
                    'severity': 'medium',
                    'evidence': f'Texture variance coefficient: {cv:.3f}'
                })
        
        # Color channel analysis
        r_var = np.var(image_array[:, :, 0])
        g_var = np.var(image_array[:, :, 1])
        b_var = np.var(image_array[:, :, 2])
        max_var = max(r_var, g_var, b_var)
        min_var = min(r_var, g_var, b_var)
        
        if min_var > 0 and max_var / min_var > 3:
            signals.append({
                'name': 'Color Channel Imbalance',
                'description': 'Unusual variance ratio between color channels may indicate selective editing.',
                'severity': 'low',
                'evidence': f'Channel variance ratio: {max_var/min_var:.2f}'
            })
        
        # Overall assessment
        if tampered_percentage < 1.0:
            signals.append({
                'name': 'Low Manipulation Evidence',
                'description': 'Very small suspicious area detected. May be noise or compression artifact rather than manipulation.',
                'severity': 'low',
                'evidence': f'Tampered area: {tampered_percentage:.2f}%'
            })
        
        if not signals:
            signals.append({
                'name': 'No Significant Anomalies',
                'description': 'Forensic analysis did not detect strong indicators of manipulation.',
                'severity': 'low',
                'evidence': 'All forensic signals within normal range'
            })
        
        return signals
