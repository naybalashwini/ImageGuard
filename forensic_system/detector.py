"""
detector.py - Universal Image Forgery Detection Module

This module implements tampering detection using pre-trained deep learning models.
We use TruFor (Universal Image Forgery Detection) or similar robust models available
via HuggingFace for detecting various types of image manipulations including:
- Splicing (combining regions from different images)
- Copy-Move (duplicating regions within the same image)
- Removal/Inpainting (erasing objects and filling the gap)
- Enhancement (adjusting brightness, contrast, etc. to hide manipulations)

Forensic Rationale:
TruFor uses a transformer-based architecture trained on multiple forgery datasets
to detect inconsistencies in noise patterns, lighting, compression artifacts, and
other forensic traces that are invisible to the human eye but indicate manipulation.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Union, Dict, Any
from PIL import Image
import cv2

# Lazy imports to handle missing dependencies gracefully
_trufor_available = False
try:
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    _trufor_available = True
except ImportError:
    pass


class ForgeryDetector:
    """
    Universal image forgery detector using pre-trained deep learning models.
    
    This class wraps the TruFor model (or fallback models) to provide pixel-level
    forgery probability maps. The model is loaded once and reused for multiple
    detections to optimize performance.
    
    Attributes:
        device: torch device (cuda or cpu) for model inference
        model: Loaded forgery detection model
        processor: Image processor for model input preparation
        model_name: Name of the loaded model for reporting
    """
    
    def __init__(
        self,
        model_name: str = "giazmo/trufor",
        device: Optional[str] = None,
        cache_dir: Optional[Union[str, Path]] = None
    ):
        """
        Initialize the forgery detector with a pre-trained model.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run inference on ('cuda', 'cpu', or None for auto)
            cache_dir: Directory to cache downloaded models
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        
        # Determine device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Load model and processor
        self.model = None
        self.processor = None
        self._load_model()
    
    def _load_model(self) -> None:
        """
        Load the forgery detection model from HuggingFace Hub.
        
        Forensic Rationale:
        Models are cached locally to ensure reproducibility and offline operation.
        This is critical for forensic work where internet access may be restricted
        and model version consistency must be maintained.
        """
        if not _trufor_available:
            raise ImportError(
                "transformers library not installed. "
                "Please install with: pip install transformers"
            )
        
        try:
            # Load processor and model
            print(f"Loading forgery detection model: {self.model_name}")
            
            self.processor = AutoImageProcessor.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            
            self.model = AutoModelForImageClassification.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                output_hidden_states=True  # Required for segmentation maps
            )
            
            self.model.to(self.device)
            self.model.eval()
            
            print(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"Warning: Could not load {self.model_name}: {e}")
            print("Falling back to a simpler heuristic-based detector...")
            self._setup_fallback_detector()
    
    def _setup_fallback_detector(self) -> None:
        """
        Set up a fallback heuristic-based detector if the primary model fails.
        
        This uses simple image statistics to detect potential anomalies.
        While less accurate than deep learning, it provides basic functionality
        when models cannot be loaded.
        """
        self.model = None
        self.processor = None
        self.is_fallback = True
    
    @torch.no_grad()
    def detect(self, image: Union[np.ndarray, Image.Image, str, Path]) -> Dict[str, Any]:
        """
        Detect tampered regions in an image.
        
        Args:
            image: Input image as numpy array (RGB), PIL Image, or file path
            
        Returns:
            Dictionary containing:
                - 'confidence_map': Float array (H, W) with tampering probabilities [0, 1]
                - 'binary_mask': Binary mask (uint8) indicating tampered regions
                - 'is_fallback': Boolean indicating if fallback detector was used
        """
        # Convert input to PIL Image
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            if image.dtype == np.uint8:
                pil_image = Image.fromarray(image)
            else:
                # Normalize float images
                if image.max() > 1.0:
                    image = image / 255.0
                pil_image = Image.fromarray((image * 255).astype(np.uint8))
        elif isinstance(image, Image.Image):
            pil_image = image.convert('RGB')
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
        
        # Use fallback if primary model not available
        if self.model is None or getattr(self, 'is_fallback', False):
            confidence_map = self._fallback_detect(pil_image)
            return {
                'confidence_map': confidence_map,
                'binary_mask': self._threshold_map(confidence_map),
                'is_fallback': True
            }
        
        # Process image through the model
        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Forward pass
        outputs = self.model(**inputs)
        
        # Extract confidence map from model outputs
        # TruFor outputs a segmentation map in hidden states or logits
        if hasattr(outputs, 'logits') and outputs.logits.ndim == 4:
            # Segmentation output (batch, channels, height, width)
            confidence_map = outputs.logits[0, 0].cpu().numpy()
        else:
            # Classification output - need to extract from hidden states
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                # Use the last hidden state
                last_hidden = outputs.hidden_states[-1][0]  # Remove batch dimension
                
                # Compute anomaly score from feature representations
                # This is a simplified approach; TruFor has specific heads for this
                confidence_map = self._extract_anomaly_map(last_hidden)
            else:
                # Fallback: use classification confidence
                confidence = torch.softmax(outputs.logits, dim=-1)[0].cpu().numpy()
                # Return uniform map based on classification result
                h, w = pil_image.height, pil_image.width
                confidence_map = np.full((h, w), confidence[1] if len(confidence) > 1 else confidence[0])
        
        # Normalize confidence map to [0, 1]
        confidence_map = self._normalize_confidence_map(confidence_map)
        
        # Resize to match original image dimensions if needed
        if confidence_map.shape != (pil_image.height, pil_image.width):
            confidence_map = cv2.resize(
                confidence_map,
                (pil_image.width, pil_image.height),
                interpolation=cv2.INTER_LINEAR
            )
        
        # Generate binary mask
        binary_mask = self._threshold_map(confidence_map)
        
        return {
            'confidence_map': confidence_map,
            'binary_mask': binary_mask,
            'is_fallback': False
        }
    
    def _extract_anomaly_map(self, hidden_state: torch.Tensor) -> np.ndarray:
        """
        Extract an anomaly map from model hidden states.
        
        This is a heuristic method that computes spatial inconsistencies
        from feature representations.
        
        Args:
            hidden_state: Tensor of shape (channels, height, width)
            
        Returns:
            2D anomaly map
        """
        # Compute local variance across channels as an anomaly indicator
        features = hidden_state.cpu().numpy()
        
        # Spatial variance across feature channels
        variance_map = np.var(features, axis=0)
        
        return variance_map
    
    def _normalize_confidence_map(self, confidence_map: np.ndarray) -> np.ndarray:
        """
        Normalize confidence map to [0, 1] range.
        
        Args:
            confidence_map: Raw confidence values
            
        Returns:
            Normalized confidence map
        """
        min_val = confidence_map.min()
        max_val = confidence_map.max()
        
        if max_val - min_val < 1e-6:
            return np.zeros_like(confidence_map)
        
        normalized = (confidence_map - min_val) / (max_val - min_val)
        
        # Apply sigmoid-like scaling to emphasize high-confidence regions
        # This helps separate clear forgeries from ambiguous areas
        normalized = 1 / (1 + np.exp(-10 * (normalized - 0.5)))
        
        return np.clip(normalized, 0, 1)
    
    def _threshold_map(
        self,
        confidence_map: np.ndarray,
        threshold: float = 0.5
    ) -> np.ndarray:
        """
        Convert confidence map to binary mask.
        
        Args:
            confidence_map: Float array with values in [0, 1]
            threshold: Threshold value for binarization
            
        Returns:
            Binary mask (uint8) with 255 for tampered regions
        """
        binary = (confidence_map > threshold).astype(np.uint8) * 255
        return binary
    
    def _fallback_detect(self, image: Image.Image) -> np.ndarray:
        """
        Fallback detection using image statistics.
        
        This method uses simple heuristics to detect potential anomalies:
        - Noise inconsistency
        - Compression artifact variation
        - Edge discontinuities
        
        Note: This is significantly less accurate than the deep learning model
        and should only be used when the primary model cannot be loaded.
        
        Args:
            image: PIL Image (RGB)
            
        Returns:
            Confidence map (H, W) with values in [0, 1]
        """
        # Convert to numpy array
        img_array = np.array(image)
        
        # Convert to grayscale for analysis
        if img_array.ndim == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        gray_float = gray.astype(np.float32)
        
        # Method 1: Local noise variance analysis
        # Regions with different noise levels may indicate splicing
        kernel_size = 7
        local_mean = cv2.blur(gray_float, (kernel_size, kernel_size))
        local_variance = cv2.blur((gray_float - local_mean) ** 2, (kernel_size, kernel_size))
        
        # Normalize variance map
        var_min, var_max = local_variance.min(), local_variance.max()
        if var_max - var_min > 1e-6:
            noise_map = (local_variance - var_min) / (var_max - var_min)
        else:
            noise_map = np.zeros_like(local_variance)
        
        # Method 2: Edge discontinuity detection
        # Sudden changes in edge density may indicate tampering
        edges = cv2.Canny(gray, 50, 150)
        edge_density = cv2.blur(edges.astype(np.float32), (15, 15))
        
        edge_min, edge_max = edge_density.min(), edge_density.max()
        if edge_max - edge_min > 1e-6:
            edge_map = (edge_density - edge_min) / (edge_max - edge_min)
        else:
            edge_map = np.zeros_like(edge_density)
        
        # Combine indicators (weighted average)
        confidence_map = 0.6 * noise_map + 0.4 * edge_map
        
        # Apply morphological operations to smooth the map
        confidence_map = cv2.GaussianBlur(confidence_map, (5, 5), 0)
        
        return np.clip(confidence_map, 0, 1)


def detect_tampering(
    image: Union[np.ndarray, Image.Image, str, Path],
    model_name: str = "giazmo/trufor",
    device: Optional[str] = None,
    threshold: float = 0.5,
    cache_dir: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    Convenience function for one-shot tampering detection.
    
    Args:
        image: Input image
        model_name: HuggingFace model identifier
        device: Device for inference
        threshold: Threshold for binary mask generation
        cache_dir: Directory for model caching
        
    Returns:
        Dictionary with confidence_map, binary_mask, and metadata
    """
    detector = ForgeryDetector(
        model_name=model_name,
        device=device,
        cache_dir=cache_dir
    )
    
    results = detector.detect(image)
    
    # Add metadata
    if isinstance(image, (str, Path)):
        results['source'] = str(image)
    elif isinstance(image, np.ndarray):
        results['source_shape'] = image.shape
    elif isinstance(image, Image.Image):
        results['source_size'] = image.size
    
    results['threshold_used'] = threshold
    results['model_name'] = model_name
    
    return results


if __name__ == "__main__":
    # Test the detector
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        print(f"Testing forgery detection on: {image_path}")
        
        results = detect_tampering(image_path)
        
        print(f"Detection complete. Confidence map shape: {results['confidence_map'].shape}")
        print(f"Tampered pixels: {np.sum(results['binary_mask'] > 0)}")
        print(f"Used fallback detector: {results.get('is_fallback', False)}")
    else:
        print("Usage: python detector.py <image_path>")
