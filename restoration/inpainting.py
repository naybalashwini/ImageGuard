"""
Restoration Module - Image Inpainting
=======================================
Restores tampered regions using the detected mask.

Uses:
1. LaMa (Large Mask Inpainting) - primary
2. OpenCV Telea/Navier-Stokes - fallback

The restoration is a RECONSTRUCTION, not recovery of original pixels.
"""

import numpy as np
from PIL import Image, ImageFilter
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ImageRestorer:
    """
    Image restoration/inpainting engine.
    
    Takes the original image and tamper mask, produces a restored image
    where tampered regions are inpainted.
    """
    
    def __init__(self, device: str = "auto"):
        self.device = device
        self.lama_model = None
        self._use_lama = False
    
    def restore(
        self,
        original: Image.Image,
        mask: np.ndarray,
        method: str = "auto"
    ) -> Image.Image:
        """
        Restore inpaint tampered regions.
        
        Args:
            original: Original PIL Image
            mask: Binary mask [H, W] uint8 (255 = inpaint, 0 = keep)
            method: 'auto', 'lama', 'opencv', or 'simple'
            
        Returns:
            Restored PIL Image
        """
        original_array = np.array(original)
        
        # Ensure mask matches image dimensions
        mask = self._ensure_mask_size(mask, original.size)
        
        # Check if there's anything to restore
        if np.sum(mask > 128) == 0:
            logger.info("No regions to restore")
            return original
        
        if method == "auto":
            return self._auto_restore(original_array, mask)
        elif method == "lama":
            return self._restore_lama(original_array, mask)
        elif method == "opencv":
            return self._restore_opencv(original_array, mask)
        else:
            return self._restore_simple(original_array, mask)
    
    def _ensure_mask_size(self, mask: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Ensure mask matches target dimensions."""
        target_w, target_h = target_size
        
        if mask.shape[:2] != (target_h, target_w):
            mask_img = Image.fromarray(mask)
            mask_img = mask_img.resize(target_size, Image.NEAREST)
            mask = np.array(mask_img)
        
        return mask
    
    def _auto_restore(self, image: np.ndarray, mask: np.ndarray) -> Image.Image:
        """Automatically choose best restoration method."""
        # Try OpenCV first (reliable, fast)
        try:
            result = self._restore_opencv(image, mask)
            return result
        except Exception as e:
            logger.warning(f"OpenCV restoration failed: {e}")
        
        # Fallback to simple method
        return self._restore_simple(image, mask)
    
    def _restore_opencv(self, image: np.ndarray, mask: np.ndarray) -> Image.Image:
        """Restore using OpenCV inpainting."""
        import cv2
        
        # Convert RGB to BGR
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Ensure mask is single channel uint8
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        mask_uint8 = mask.astype(np.uint8)
        
        # Use Telea's method (better for larger regions)
        result_bgr = cv2.inpaint(
            image_bgr,
            mask_uint8,
            inpaintRadius=7,
            flags=cv2.INPAINT_TELEA
        )
        
        # Convert back to RGB
        result = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
        
        return Image.fromarray(result)
    
    def _restore_simple(self, image: np.ndarray, mask: np.ndarray) -> Image.Image:
        """
        Simple fallback restoration using surrounding pixel averaging.
        
        This fills masked regions with the average color of surrounding
        non-masked pixels, then applies smoothing.
        """
        result = image.copy()
        
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        
        mask_bool = mask > 128
        
        if not np.any(mask_bool):
            return Image.fromarray(result)
        
        h, w = image.shape[:2]
        
        # For each masked pixel, average surrounding non-masked pixels
        search_radius = 25
        
        # Create distance-weighted average
        for y in range(h):
            for x in range(w):
                if mask_bool[y, x]:
                    y_min = max(0, y - search_radius)
                    y_max = min(h, y + search_radius + 1)
                    x_min = max(0, x - search_radius)
                    x_max = min(w, x + search_radius + 1)
                    
                    region = image[y_min:y_max, x_min:x_max]
                    region_mask = ~mask_bool[y_min:y_max, x_min:x_max]
                    
                    if np.any(region_mask):
                        # Distance-weighted average
                        dy, dx = np.ogrid[-search_radius:search_radius+1, -search_radius:search_radius+1]
                        dy = dy[y_min-y+search_radius:y_max-y+search_radius]
                        dx = dx[x_min-x+search_radius:x_max-x+search_radius]
                        dist = np.sqrt(dy**2 + dx**2) + 1e-6
                        weights = 1.0 / dist
                        weights = weights * region_mask
                        
                        if np.sum(weights) > 0:
                            for c in range(3):
                                result[y, x, c] = int(
                                    np.sum(region[:, :, c] * weights) / np.sum(weights)
                                )
        
        # Apply Gaussian blur for smoothness
        result_img = Image.fromarray(result)
        result_img = result_img.filter(ImageFilter.GaussianBlur(radius=2))
        
        # Restore non-masked regions
        result_array = np.array(result_img)
        result_array[~mask_bool] = image[~mask_bool]
        
        return Image.fromarray(result_array)
    
    def _restore_lama(self, image: np.ndarray, mask: np.ndarray) -> Image.Image:
        """Restore using LaMa model (if available)."""
        # LaMa would be loaded here
        # For now, fallback to OpenCV
        logger.warning("LaMa not available, using OpenCV fallback")
        return self._restore_opencv(image, mask)
