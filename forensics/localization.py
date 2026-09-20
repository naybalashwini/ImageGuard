"""
Forensics Pipeline - Localization Module
=========================================
Handles pixel-level forgery localization using TruFor/MVSS-Net.
"""

import numpy as np
from PIL import Image
from typing import Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)


class ForensicLocalizer:
    """
    Pixel-level forgery localization engine.
    
    Uses TruFor as primary detector and MVSS-Net as fallback.
    Produces:
    - Localization map: pixel-level tamper probability
    - Confidence map: prediction reliability
    - Integrity score: image-level forgery score
    """
    
    def __init__(self, device: str = "auto"):
        self.device = device
        self.primary_detector = None
        self.fallback_detector = None
        self.active_detector = None
        self._initialized = False
    
    def initialize(self) -> bool:
        """
        Initialize detectors.
        
        Tries to load TruFor first, then MVSS-Net as fallback.
        
        Returns:
            True if at least one detector is available.
        """
        # Try TruFor (primary)
        try:
            from models.trufor_detector import TruForDetector
            self.primary_detector = TruForDetector(device=self.device)
            if self.primary_detector.load():
                self.active_detector = self.primary_detector
                logger.info("Using TruFor as primary localization model")
                self._initialized = True
                return True
            else:
                logger.warning("TruFor not available, trying MVSS-Net...")
        except Exception as e:
            logger.warning(f"TruFor initialization failed: {e}")
        
        # Try MVSS-Net (fallback)
        try:
            from models.mvss_detector import MVSSDetector
            self.fallback_detector = MVSSDetector(device=self.device)
            if self.fallback_detector.load():
                self.active_detector = self.fallback_detector
                logger.info("Using MVSS-Net as fallback localization model")
                self._initialized = True
                return True
            else:
                logger.warning("MVSS-Net not available, using classical analysis")
        except Exception as e:
            logger.warning(f"MVSS-Net initialization failed: {e}")
        
        # Use classical forensic analysis as final fallback
        logger.info("Using classical forensic analysis (no deep learning model)")
        self._initialized = True
        return True
    
    def is_available(self) -> bool:
        """Check if any detector is available."""
        return self._initialized
    
    def get_model_name(self) -> str:
        """Get the name of the active detector."""
        if self.active_detector is not None:
            return getattr(self.active_detector, 'model_name', 'Unknown')
        return "Classical Forensic Analysis"
    
    def localize(self, image_path: str) -> Dict[str, Any]:
        """
        Perform pixel-level forgery localization.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Dictionary with:
            - 'localization_map': pixel-level tamper probability [H, W] float32
            - 'confidence_map': reliability map [H, W] float32
            - 'score': image-level integrity score float
            - 'original_size': (width, height) of original image
            - 'model_name': name of the model used
        """
        if not self._initialized:
            raise RuntimeError("Localizer not initialized. Call initialize() first.")
        
        # Use active detector if available
        if self.active_detector is not None:
            try:
                result = self.active_detector.predict(image_path)
                return result
            except Exception as e:
                logger.error(f"Detection failed: {e}")
                logger.info("Falling back to classical analysis")
        
        # Classical forensic analysis fallback
        return self._classical_localization(image_path)
    
    def _classical_localization(self, image_path: str) -> Dict[str, Any]:
        """
        Classical forensic localization using multiple signals.
        
        Combines:
        1. ELA (Error Level Analysis)
        2. Texture variance analysis
        3. Noise residual analysis
        4. Color channel consistency
        """
        from forensics.preprocessing import preprocess_image, compute_ela
        
        # Preprocess
        prep = preprocess_image(image_path)
        original = prep['original']
        original_array = prep['original_array']
        original_size = prep['original_size']
        
        # ELA
        ela_result = compute_ela(original)
        ela_diff = ela_result['difference_raw']
        ela_gray = np.mean(ela_diff, axis=2)
        
        # Normalize ELA map
        ela_map = ela_gray / (ela_gray.max() + 1e-8)
        
        # Texture variance analysis
        gray = np.mean(original_array, axis=2).astype(np.float32)
        h, w = gray.shape
        block_size = 32
        variance_map = np.zeros((h, w), dtype=np.float32)
        
        for y in range(0, h - block_size, block_size // 2):
            for x in range(0, w - block_size, block_size // 2):
                block = gray[y:y+block_size, x:x+block_size]
                var = np.var(block)
                variance_map[y:y+block_size, x:x+block_size] = np.maximum(
                    variance_map[y:y+block_size, x:x+block_size], var
                )
        
        # Normalize texture map
        var_mean = np.mean(variance_map[variance_map > 0]) if np.any(variance_map > 0) else 1
        var_std = np.std(variance_map[variance_map > 0]) if np.any(variance_map > 0) else 1
        texture_map = np.clip((variance_map - var_mean) / (var_std + 1e-8), 0, 3) / 3
        
        # Noise residual analysis (Laplacian)
        try:
            from scipy.ndimage import laplace
            noise_residual = np.abs(laplace(gray))
            noise_map = noise_residual / (noise_residual.max() + 1e-8)
        except ImportError:
            # Simple gradient-based noise estimation
            grad_x = np.gradient(gray, axis=1)
            grad_y = np.gradient(gray, axis=0)
            noise_residual = np.sqrt(grad_x**2 + grad_y**2)
            noise_map = noise_residual / (noise_residual.max() + 1e-8)
        
        # Combine signals with weights
        localization_map = (
            0.4 * ela_map +
            0.3 * texture_map +
            0.3 * noise_map
        )
        
        # Normalize to [0, 1]
        localization_map = localization_map / (localization_map.max() + 1e-8)
        
        # Generate confidence map (lower confidence for edges and high-variance regions)
        confidence_map = np.ones_like(localization_map) * 0.6
        
        # Reduce confidence in high-variance areas
        confidence_map = confidence_map * (1 - 0.3 * texture_map)
        
        # Image-level score
        score = float(np.mean(localization_map[localization_map > 0.3])) if np.any(localization_map > 0.3) else 0.1
        score = min(0.95, max(0.05, score))
        
        return {
            'localization_map': localization_map,
            'confidence_map': confidence_map,
            'score': score,
            'original_size': original_size,
            'model_name': 'Classical Forensic Analysis (ELA + Texture + Noise)',
        }
