"""
MVSS-Net Forgery Detector (Fallback)
=====================================
Secondary localization model based on MVSS-Net.

MVSS-Net (Multi-view Stroke Suppression Network) uses edge-guided
multi-scale feature extraction for forgery localization.

Reference:
Dong et al., "Multi-view Stroke Suppression Network for Image Forgery Localization"

Official repo: https://github.com/dong03/MVSS-Net
"""

import os
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)


class MVSSDetector:
    """
    MVSS-Net based forgery localization detector.
    
    This is the fallback detector when TruFor is unavailable.
    It uses edge-guided multi-scale feature extraction.
    """
    
    def __init__(
        self,
        weights_path: str = "models/weights/mvss/mvss_net.pth",
        device: str = "auto"
    ):
        self.weights_path = Path(weights_path)
        self.device = self._resolve_device(device)
        self.model = None
        self._loaded = False
        self.input_size = (512, 512)
    
    def _resolve_device(self, device: str) -> str:
        if device == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda"
            except ImportError:
                pass
            return "cpu"
        return device
    
    def is_available(self) -> bool:
        return self._loaded and self.model is not None
    
    def check_weights(self) -> bool:
        return self.weights_path.exists()
    
    def load(self) -> bool:
        """Load the MVSS-Net model."""
        if not self.check_weights():
            logger.warning(
                f"MVSS-Net weights not found at: {self.weights_path}\n"
                "Using classical forensic analysis as fallback."
            )
            return False
        
        try:
            self.model = self._build_model()
            
            checkpoint = __import__('torch').load(
                str(self.weights_path),
                map_location=self.device
            )
            
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model' in checkpoint:
                    state_dict = checkpoint['model']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            # Clean keys
            new_state_dict = {}
            for k, v in state_dict.items():
                new_key = k.replace('module.', '')
                new_state_dict[new_key] = v
            
            self.model.load_state_dict(new_state_dict, strict=False)
            self.model.to(self.device)
            self.model.eval()
            self._loaded = True
            
            logger.info("MVSS-Net model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load MVSS-Net: {e}")
            return False
    
    def _build_model(self):
        """Build MVSS-Net architecture."""
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        
        class EdgeBranch(nn.Module):
            """Edge detection branch for boundary-aware features."""
            def __init__(self, in_channels=3):
                super().__init__()
                self.edge_conv = nn.Sequential(
                    nn.Conv2d(in_channels, 32, 3, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, 3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                )
            
            def forward(self, x):
                return self.edge_conv(x)
        
        class MVSSNet(nn.Module):
            """MVSS-Net architecture for forgery localization."""
            def __init__(self):
                super().__init__()
                import torchvision.models as models
                
                # RGB encoder (ResNet-50 backbone)
                resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
                self.encoder1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
                self.encoder2 = resnet.layer1
                self.encoder3 = resnet.layer2
                self.encoder4 = resnet.layer3
                self.encoder5 = resnet.layer4
                
                # Edge branch
                self.edge_branch = EdgeBranch(3)
                
                # Decoder
                self.decoder4 = nn.Sequential(
                    nn.ConvTranspose2d(2048 + 64, 512, 2, stride=2),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(512, 512, 3, padding=1),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=True),
                )
                self.decoder3 = nn.Sequential(
                    nn.ConvTranspose2d(512 + 1024, 256, 2, stride=2),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, 256, 3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                )
                self.decoder2 = nn.Sequential(
                    nn.ConvTranspose2d(256 + 512, 128, 2, stride=2),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 64, 3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                )
                self.decoder1 = nn.Sequential(
                    nn.ConvTranspose2d(64 + 256, 64, 2, stride=2),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 32, 3, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                )
                
                # Final prediction
                self.final = nn.Sequential(
                    nn.Conv2d(32, 1, 1),
                    nn.Sigmoid()
                )
            
            def forward(self, x):
                # Encoder
                e1 = self.encoder1(x)
                e2 = self.encoder2(e1)
                e3 = self.encoder3(e2)
                e4 = self.encoder4(e3)
                e5 = self.encoder5(e4)
                
                # Edge features
                edge_feat = self.edge_branch(x)
                
                # Decoder with skip connections
                d4 = self.decoder4(torch.cat([e5, F.interpolate(edge_feat, size=e5.shape[2:], mode='bilinear')], dim=1))
                d3 = self.decoder3(torch.cat([d4, e4], dim=1))
                d2 = self.decoder2(torch.cat([d3, e3], dim=1))
                d1 = self.decoder1(torch.cat([d2, e2], dim=1))
                
                # Upsample to input size
                d1 = F.interpolate(d1, size=x.shape[2:], mode='bilinear', align_corners=False)
                
                # Final prediction
                output = self.final(d1)
                
                return {
                    'map': output,
                    'conf': torch.ones_like(output) * 0.7,  # MVSS doesn't have confidence
                    'score': output.mean(dim=[2, 3]),
                }
        
        return MVSSNet()
    
    def predict(self, image_path: str) -> Dict[str, Any]:
        """Run MVSS-Net inference."""
        if not self._loaded:
            raise RuntimeError("MVSS-Net model not loaded.")
        
        import torch
        from PIL import Image
        
        original = Image.open(image_path).convert('RGB')
        original_size = original.size
        
        model_input = original.resize(self.input_size, Image.BILINEAR)
        img_array = np.array(model_input).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_array = (img_array - mean) / std
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float()
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            output = self.model(img_tensor)
        
        localization_map = output['map'].squeeze().cpu().numpy()
        confidence_map = output['conf'].squeeze().cpu().numpy()
        score = output['score'].squeeze().cpu().numpy()
        
        # Resize to original dimensions
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
            'model_name': 'MVSS-Net',
        }
