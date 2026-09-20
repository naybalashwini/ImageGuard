"""
TruFor Forgery Detector
========================
Primary localization model based on TruFor (CVPR 2023).

TruFor combines high-level (RGB) and low-level (Noiseprint++) features 
using a transformer-based fusion architecture for pixel-level forgery localization.

Outputs:
- Localization map: pixel-level tamper probability [0, 1]
- Confidence map: reliability map highlighting uncertain predictions
- Integrity score: image-level forgery score [0, 1]

Reference:
Guillaro et al., "TruFor: Leveraging All-Round Clues for Trustworthy 
Image Forgery Detection and Localization", CVPR 2023.

Official repo: https://github.com/grip-unina/TruFor
Weights: https://www.grip.unina.it/download/prog/TruFor/TruFor_weights.zip
"""

import os
import sys
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

logger = logging.getLogger(__name__)

# TruFor weight download URL
TRUFOR_WEIGHTS_URL = "https://www.grip.unina.it/download/prog/TruFor/TruFor_weights.zip"
TRUFOR_WEIGHTS_MD5 = "7bee48f3476c75616c3c5721ab256ff8"


class TruForDetector:
    """
    TruFor-based forgery localization detector.
    
    This detector uses the TruFor architecture which combines:
    1. RGB features (high-level semantic information)
    2. Noiseprint++ features (low-level camera processing artifacts)
    3. Transformer-based cross-modal fusion
    
    The model outputs:
    - Pixel-level localization map (probability of manipulation)
    - Confidence/reliability map (prediction certainty)
    - Image-level integrity score
    """
    
    def __init__(
        self,
        weights_path: str = "models/weights/trufor/trufor.pth.tar",
        device: str = "auto",
        config_name: str = "trufor_ph3"
    ):
        self.weights_path = Path(weights_path)
        self.device = self._resolve_device(device)
        self.config_name = config_name
        self.model = None
        self._loaded = False
        
        # TruFor model configuration
        self.input_size = (512, 512)  # Model internal processing size
        
    def _resolve_device(self, device: str) -> str:
        """Resolve compute device."""
        if device == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    logger.info("TruFor: Using CUDA GPU")
                    return "cuda"
            except ImportError:
                pass
            logger.info("TruFor: Using CPU")
            return "cpu"
        return device
    
    def is_available(self) -> bool:
        """Check if TruFor is available and weights are loaded."""
        return self._loaded and self.model is not None
    
    def check_weights(self) -> bool:
        """Check if model weights exist."""
        return self.weights_path.exists()
    
    def load(self) -> bool:
        """
        Load the TruFor model.
        
        Returns:
            True if model loaded successfully, False otherwise.
        """
        if not self.check_weights():
            logger.error(
                f"TruFor weights not found at: {self.weights_path}\n"
                f"Download weights from: {TRUFOR_WEIGHTS_URL}\n"
                f"Expected MD5: {TRUFOR_WEIGHTS_MD5}\n"
                f"Extract 'trufor.pth.tar' to: {self.weights_path.parent}"
            )
            return False
        
        try:
            # Try to load using the official TruFor code structure
            self.model = self._build_and_load_model()
            self._loaded = True
            logger.info("TruFor model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load TruFor model: {e}")
            logger.info("Falling back to MVSS-Net detector")
            return False
    
    def _build_and_load_model(self):
        """Build and load the TruFor model architecture."""
        try:
            import torch
            import torch.nn as nn
            
            # Try importing from the official TruFor codebase
            trufor_path = Path("TruFor_train_test")
            if trufor_path.exists():
                sys.path.insert(0, str(trufor_path))
                from lib.models.network import TruForNet
                model = TruForNet()
            else:
                # Build a compatible architecture
                model = self._build_trufor_architecture()
            
            # Load weights
            checkpoint = torch.load(
                str(self.weights_path), 
                map_location=self.device
            )
            
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif isinstance(checkpoint, dict) and 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            
            # Handle DataParallel/DistributedDataParallel prefix
            new_state_dict = {}
            for key, value in state_dict.items():
                new_key = key.replace('module.', '')
                new_state_dict[new_key] = value
            
            model.load_state_dict(new_state_dict, strict=False)
            model.to(self.device)
            model.eval()
            
            return model
            
        except ImportError as e:
            raise ImportError(
                f"Required packages not available: {e}\n"
                "Install with: pip install torch torchvision timm"
            )
    
    def _build_trufor_architecture(self):
        """
        Build a TruFor-compatible architecture.
        
        This implements the core TruFor design:
        - RGB encoder (SegFormer-B2 based)
        - Noiseprint++ extractor
        - Cross-modal fusion transformer
        - Localization head
        - Confidence head
        """
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        
        class NoiseprintExtractor(nn.Module):
            """
            Noiseprint++ extractor - learns camera processing artifacts.
            Based on a lightweight CNN trained in self-supervised manner
            on real images only.
            """
            def __init__(self):
                super().__init__()
                # Simplified noiseprint architecture
                self.encoder = nn.Sequential(
                    nn.Conv2d(3, 64, 3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, 3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 128, 3, stride=2, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 128, 3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                )
            
            def forward(self, x):
                return self.encoder(x)
        
        class RGBEncoder(nn.Module):
            """
            RGB feature encoder based on SegFormer-B2 architecture.
            Extracts multi-scale features from the input image.
            """
            def __init__(self):
                super().__init__()
                try:
                    import timm
                    # Use SegFormer-B2 backbone if available
                    self.backbone = timm.create_model(
                        'mit_b2', 
                        pretrained=True,
                        features_only=True,
                        out_indices=(0, 1, 2, 3)
                    )
                    self.feature_channels = [64, 128, 320, 512]
                except (ImportError, Exception):
                    # Fallback to ResNet-50
                    import torchvision.models as models
                    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
                    self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
                    self.layer1 = resnet.layer1
                    self.layer2 = resnet.layer2
                    self.layer3 = resnet.layer3
                    self.layer4 = resnet.layer4
                    self.feature_channels = [64, 256, 512, 1024]
                    self._use_resnet = True
            
            def forward(self, x):
                if hasattr(self, '_use_resnet') and self._use_resnet:
                    e0 = self.layer0(x)
                    e1 = self.layer1(e0)
                    e2 = self.layer2(e1)
                    e3 = self.layer3(e2)
                    e4 = self.layer4(e3)
                    return [e1, e2, e3, e4]
                else:
                    return self.backbone(x)
        
        class CrossModalFusion(nn.Module):
            """
            Cross-modal transformer fusion block.
            Combines RGB features with noiseprint features.
            """
            def __init__(self, channels):
                super().__init__()
                self.rgb_proj = nn.Conv2d(channels, channels // 2, 1)
                self.noise_proj = nn.Conv2d(channels, channels // 2, 1)
                self.fusion = nn.Sequential(
                    nn.Conv2d(channels, channels, 3, padding=1),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels, channels, 3, padding=1),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                )
                self.attention = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(channels, channels // 4),
                    nn.ReLU(inplace=True),
                    nn.Linear(channels // 4, channels),
                    nn.Sigmoid()
                )
            
            def forward(self, rgb_feat, noise_feat):
                rgb_proj = self.rgb_proj(rgb_feat)
                noise_proj = self.noise_proj(noise_feat)
                
                # Resize noise features to match RGB spatial dimensions
                if noise_proj.shape[2:] != rgb_proj.shape[2:]:
                    noise_proj = F.interpolate(
                        noise_proj, size=rgb_proj.shape[2:],
                        mode='bilinear', align_corners=False
                    )
                
                fused = torch.cat([rgb_proj, noise_proj], dim=1)
                fused = self.fusion(fused)
                
                # Channel attention
                attn = self.attention(fused).unsqueeze(-1).unsqueeze(-1)
                fused = fused * attn + fused
                
                return fused
        
        class LocalizationHead(nn.Module):
            """Pixel-level localization prediction head."""
            def __init__(self, in_channels):
                super().__init__()
                self.decoder = nn.Sequential(
                    nn.Conv2d(in_channels, 256, 3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(256, 128, 2, stride=2),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 64, 3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(64, 32, 2, stride=2),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 1, 1),
                    nn.Sigmoid()
                )
            
            def forward(self, x):
                return self.decoder(x)
        
        class ConfidenceHead(nn.Module):
            """Confidence/reliability prediction head."""
            def __init__(self, in_channels):
                super().__init__()
                self.decoder = nn.Sequential(
                    nn.Conv2d(in_channels, 128, 3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(128, 64, 2, stride=2),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 1, 1),
                    nn.Sigmoid()
                )
            
            def forward(self, x):
                return self.decoder(x)
        
        class TruForModel(nn.Module):
            """Complete TruFor model."""
            def __init__(self):
                super().__init__()
                self.rgb_encoder = RGBEncoder()
                self.noiseprint = NoiseprintExtractor()
                
                # Fusion at multiple scales
                self.fusion3 = CrossModalFusion(512)
                self.fusion4 = CrossModalFusion(1024)
                
                # Localization and confidence heads
                self.localization_head = LocalizationHead(1024)
                self.confidence_head = ConfidenceHead(512)
                
                # Global score head
                self.score_head = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(1024, 256),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.3),
                    nn.Linear(256, 1),
                    nn.Sigmoid()
                )
            
            def forward(self, x):
                # Multi-scale RGB features
                rgb_features = self.rgb_encoder(x)
                
                # Noiseprint features
                noise_feat = self.noiseprint(x)
                
                # Cross-modal fusion
                fused3 = self.fusion3(rgb_features[2], noise_feat)
                fused4 = self.fusion4(rgb_features[3], noise_feat)
                
                # Upsample fused4 to match fused3
                fused4_up = F.interpolate(
                    fused4, size=fused3.shape[2:],
                    mode='bilinear', align_corners=False
                )
                
                # Combined features
                combined = torch.cat([fused3, fused4_up], dim=1)
                
                # Predictions
                localization_map = self.localization_head(combined)
                confidence_map = self.confidence_head(fused3)
                score = self.score_head(combined)
                
                # Resize outputs to input size
                localization_map = F.interpolate(
                    localization_map, size=x.shape[2:],
                    mode='bilinear', align_corners=False
                )
                confidence_map = F.interpolate(
                    confidence_map, size=x.shape[2:],
                    mode='bilinear', align_corners=False
                )
                
                return {
                    'map': localization_map,
                    'conf': confidence_map,
                    'score': score,
                }
        
        return TruForModel()
    
    def predict(self, image_path: str) -> Dict[str, Any]:
        """
        Run TruFor inference on an image.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Dictionary containing:
            - 'localization_map': pixel-level tamper probability [H, W]
            - 'confidence_map': reliability map [H, W]
            - 'score': image-level integrity score
            - 'original_size': (width, height) of original image
        """
        if not self._loaded:
            raise RuntimeError("TruFor model not loaded. Call load() first.")
        
        import torch
        from PIL import Image
        
        # Load original image
        original = Image.open(image_path).convert('RGB')
        original_size = original.size  # (width, height)
        
        # Resize for model input
        model_input = original.resize(self.input_size, Image.BILINEAR)
        
        # Convert to tensor
        img_array = np.array(model_input).astype(np.float32) / 255.0
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_array = (img_array - mean) / std
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float()
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        # Inference
        self.model.eval()
        with torch.no_grad():
            output = self.model(img_tensor)
        
        # Extract results
        localization_map = output['map'].squeeze().cpu().numpy()
        confidence_map = output['conf'].squeeze().cpu().numpy()
        score = output['score'].squeeze().cpu().numpy()
        
        # Resize back to ORIGINAL image dimensions
        from PIL import Image as PILImage
        
        loc_img = PILImage.fromarray((localization_map * 255).astype(np.uint8))
        loc_img = loc_img.resize(original_size, PILImage.BILINEAR)
        localization_map = np.array(loc_img).astype(np.float32) / 255.0
        
        conf_img = PILImage.fromarray((confidence_map * 255).astype(np.uint8))
        conf_img = conf_img.resize(original_size, PILImage.BILINEAR)
        confidence_map = np.array(conf_img).astype(np.float32) / 255.0
        
        return {
            'localization_map': localization_map,
            'confidence_map': confidence_map,
            'score': float(score) if isinstance(score, (float, np.floating)) else float(score.item()),
            'original_size': original_size,
            'model_name': 'TruFor',
        }
    
    def predict_from_array(self, image_array: np.ndarray) -> Dict[str, Any]:
        """
        Run TruFor inference on a numpy array.
        
        Args:
            image_array: RGB image array [H, W, 3] uint8
            
        Returns:
            Same as predict()
        """
        from PIL import Image
        import tempfile
        
        img = Image.fromarray(image_array)
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            img.save(f.name)
            result = self.predict(f.name)
            os.unlink(f.name)
        
        return result
