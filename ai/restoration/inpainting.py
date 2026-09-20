"""
Image Restoration / Inpainting Module

Provides inpainting for detected tampered regions.
Supports:
1. LaMa (Large Mask Inpainting) - for complex regions
2. OpenCV inpainting - fallback for simple regions

The restoration is a RECONSTRUCTION, not recovery of original pixels.
"""

import logging
import numpy as np
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class InpaintingModel:
    """
    Image inpainting model for restoration of tampered regions.
    
    Primary: LaMa (Large Mask Inpainting)
    Fallback: OpenCV Telea / Navier-Stokes
    
    LaMa uses Fast Fourier Convolutions which are particularly effective
    for large masked regions because they can reason about the global
    image structure.
    """
    
    def __init__(self, model_path: str = "ai/models/weights/lama.pth", device: str = "cpu"):
        self.model_path = Path(model_path)
        self.device = device
        self.model = None
        self._use_lama = False
    
    def load(self):
        """Load the inpainting model."""
        if self.model_path.exists():
            try:
                self._load_lama()
                self._use_lama = True
                logger.info("LaMa inpainting model loaded")
                return
            except Exception as e:
                logger.warning(f"Could not load LaMa: {e}")
        
        logger.info("Using OpenCV inpainting as fallback")
        self._use_lama = False
    
    def _load_lama(self):
        """Load LaMa model."""
        import torch
        # LaMa model loading would go here
        # This requires the simple-lama-inpainting package or custom implementation
        # For now, we use OpenCV as the reliable fallback
        raise NotImplementedError("LaMa requires additional setup. Using OpenCV fallback.")
    
    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Inpaint masked regions of the image.
        
        Args:
            image: RGB image [H, W, 3] uint8
            mask: Binary mask [H, W] uint8 (255 = inpaint, 0 = keep)
            
        Returns:
            Inpainted image [H, W, 3] uint8
        """
        if self._use_lama and self.model is not None:
            return self._inpaint_lama(image, mask)
        
        return self._inpaint_opencv(image, mask)
    
    def _inpaint_lama(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Inpaint using LaMa model."""
        import torch
        
        self.model.eval()
        
        # Normalize
        img_tensor = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0) / 255.0
        
        # Concatenate image and mask
        input_tensor = torch.cat([img_tensor, mask_tensor], dim=0).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)
        
        result = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        result = (result * 255).clip(0, 255).astype(np.uint8)
        
        return result
    
    def _inpaint_opencv(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Inpaint using OpenCV.
        
        Uses Telea's method (fast marching) which works well for
        moderate-sized regions. For very large regions, the quality
        may be limited.
        """
        try:
            import cv2
            
            # Convert RGB to BGR for OpenCV
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Ensure mask is single channel uint8
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            mask = mask.astype(np.uint8)
            
            # Apply inpainting with Telea's method
            result_bgr = cv2.inpaint(
                image_bgr,
                mask,
                inpaintRadius=7,
                flags=cv2.INPAINT_TELEA
            )
            
            # Convert back to RGB
            result = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
            
            return result
            
        except ImportError:
            logger.warning("OpenCV not available. Using simple blur-based inpainting.")
            return self._inpaint_simple(image, mask)
    
    def _inpaint_simple(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Simple fallback inpainting using surrounding pixel averaging.
        
        This is a basic method that fills masked regions with the average
        of surrounding non-masked pixels. Quality is limited but functional.
        """
        result = image.copy()
        
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        
        mask_bool = mask > 128
        h, w = image.shape[:2]
        
        # For each masked pixel, average surrounding non-masked pixels
        search_radius = 20
        
        for y in range(h):
            for x in range(w):
                if mask_bool[y, x]:
                    # Search for non-masked neighbors
                    y_min = max(0, y - search_radius)
                    y_max = min(h, y + search_radius)
                    x_min = max(0, x - search_radius)
                    x_max = min(w, x + search_radius)
                    
                    region = image[y_min:y_max, x_min:x_max]
                    region_mask = ~mask_bool[y_min:y_max, x_min:x_max]
                    
                    if np.any(region_mask):
                        avg_color = np.mean(region[region_mask], axis=0)
                        result[y, x] = avg_color.astype(np.uint8)
        
        # Apply slight Gaussian blur for smoothness
        try:
            import cv2
            result = cv2.GaussianBlur(result, (5, 5), 0)
            # Restore non-masked regions
            result[~mask_bool] = image[~mask_bool]
        except ImportError:
            pass
        
        return result
