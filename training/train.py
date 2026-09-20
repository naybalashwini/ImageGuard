"""
Training Script for Forgery Localization Model

This script trains the ManTraNet-style forgery localization model
using transfer learning from ImageNet-pretrained ResNet-50.

Usage:
    python training/train.py --dataset_dir ./datasets --epochs 50 --batch_size 8

Dataset Structure:
    datasets/
    ├── CASIA/
    │   ├── train/
    │   │   ├── images/
    │   │   └── masks/
    │   └── val/
    │       ├── images/
    │       └── masks/
    └── Columbia/
        ├── train/
        │   ├── images/
        │   └── masks/
        └── val/
            ├── images/
            └── masks/

IMPORTANT: Test sets are NEVER used during training.
"""

import os
import sys
import json
import time
import random
import logging
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


class ForgeryDataset:
    """
    Dataset for forgery localization training.
    
    Each sample consists of:
    - Original/tampered image
    - Ground truth binary mask
    
    Supports multiple datasets with proper train/val/test separation.
    """
    
    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        transform=None,
        augment: bool = False
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.augment = augment
        
        # Find all image-mask pairs
        self.samples = self._find_pairs()
        logger.info(f"Found {len(self.samples)} image-mask pairs in {image_dir}")
    
    def _find_pairs(self) -> List[Tuple[str, str]]:
        """Find matching image-mask pairs."""
        pairs = []
        
        if not self.image_dir.exists():
            logger.warning(f"Image directory not found: {self.image_dir}")
            return pairs
        
        for img_path in self.image_dir.iterdir():
            if img_path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.tif', '.bmp'}:
                # Look for corresponding mask
                mask_name = img_path.stem + '_mask' + img_path.suffix
                mask_path = self.mask_dir / mask_name
                
                if not mask_path.exists():
                    # Try without _mask suffix
                    mask_path = self.mask_dir / img_path.with_suffix('.png').name
                
                if mask_path.exists():
                    pairs.append((str(img_path), str(mask_path)))
        
        return pairs
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        from PIL import Image
        
        img_path, mask_path = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Load mask
        mask = Image.open(mask_path).convert('L')
        
        # Resize to same dimensions
        target_size = 512
        image = image.resize((target_size, target_size), Image.BILINEAR)
        mask = mask.resize((target_size, target_size), Image.NEAREST)
        
        # Augmentation
        if self.augment:
            image, mask = self._augment(image, mask)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Convert mask to tensor
        mask_array = np.array(mask).astype(np.float32) / 255.0
        mask_array = (mask_array > 0.5).astype(np.float32)
        
        return image, mask_array
    
    def _augment(self, image, mask):
        """Apply data augmentation."""
        import random
        
        # Random horizontal flip
        if random.random() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        
        # Random vertical flip
        if random.random() > 0.5:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
        
        # Random rotation (small angles)
        if random.random() > 0.5:
            angle = random.uniform(-15, 15)
            image = image.rotate(angle, Image.BILINEAR)
            mask = mask.rotate(angle, Image.NEAREST)
        
        # Random brightness/contrast
        if random.random() > 0.5:
            from PIL import ImageEnhance
            factor = random.uniform(0.8, 1.2)
            image = ImageEnhance.Brightness(image).enhance(factor)
        
        if random.random() > 0.5:
            from PIL import ImageEnhance
            factor = random.uniform(0.8, 1.2)
            image = ImageEnhance.Contrast(image).enhance(factor)
        
        return image, mask


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    import torch
    
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch_idx, (images, masks) in enumerate(dataloader):
        images = images.to(device)
        masks = masks.unsqueeze(1).to(device)  # [B, 1, H, W]
        
        optimizer.zero_grad()
        
        # Forward
        outputs = model(images)
        
        # Loss (weighted BCE + Dice)
        bce_loss = criterion(outputs, masks)
        dice_loss = dice_loss_fn(outputs, masks)
        loss = 0.5 * bce_loss + 0.5 * dice_loss
        
        # Backward
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if batch_idx % 10 == 0:
            logger.info(
                f"Epoch {epoch} [{batch_idx}/{len(dataloader)}] "
                f"Loss: {loss.item():.4f} (BCE: {bce_loss.item():.4f}, Dice: {dice_loss.item():.4f})"
            )
    
    return total_loss / max(num_batches, 1)


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    import torch
    
    model.eval()
    total_loss = 0
    num_batches = 0
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.unsqueeze(1).to(device)
            
            outputs = model(images)
            
            bce_loss = criterion(outputs, masks)
            dice_loss = dice_loss_fn(outputs, masks)
            loss = 0.5 * bce_loss + 0.5 * dice_loss
            
            total_loss += loss.item()
            num_batches += 1
            
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(masks.cpu().numpy())
    
    avg_loss = total_loss / max(num_batches, 1)
    
    # Calculate metrics
    preds = np.concatenate(all_preds).flatten()
    targets = np.concatenate(all_targets).flatten()
    
    # Binary metrics at threshold 0.5
    binary_preds = (preds > 0.5).astype(np.float32)
    
    # IoU
    intersection = np.sum(binary_preds * targets)
    union = np.sum(binary_preds) + np.sum(targets) - intersection
    iou = intersection / (union + 1e-8)
    
    # Dice
    dice = 2 * intersection / (np.sum(binary_preds) + np.sum(targets) + 1e-8)
    
    # Pixel precision/recall
    tp = np.sum(binary_preds * targets)
    fp = np.sum(binary_preds * (1 - targets))
    fn = np.sum((1 - binary_preds) * targets)
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    metrics = {
        'loss': avg_loss,
        'iou': float(iou),
        'dice': float(dice),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
    }
    
    return avg_loss, metrics


def dice_loss_fn(predictions, targets, smooth=1e-6):
    """Dice loss function."""
    import torch
    
    predictions = predictions.view(-1)
    targets = targets.view(-1)
    
    intersection = torch.sum(predictions * targets)
    dice = (2. * intersection + smooth) / (
        torch.sum(predictions) + torch.sum(targets) + smooth
    )
    
    return 1 - dice


def main():
    parser = argparse.ArgumentParser(description='Train Forgery Localization Model')
    parser.add_argument('--dataset_dir', type=str, default='./datasets',
                       help='Root directory of datasets')
    parser.add_argument('--output_dir', type=str, default='./ai/models/weights',
                       help='Output directory for model weights')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (auto/cpu/cuda)')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Import torch
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, ConcatDataset
        import torchvision.transforms as transforms
    except ImportError:
        logger.error("PyTorch not installed. Install with: pip install torch torchvision")
        sys.exit(1)
    
    # Device selection
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    
    # Load datasets
    dataset_dir = Path(args.dataset_dir)
    
    train_datasets = []
    val_datasets = []
    
    # Look for datasets in the directory
    for ds_name in ['CASIA', 'Columbia', 'NIST', 'Coverage']:
        ds_path = dataset_dir / ds_name
        
        if ds_path.exists():
            train_path = ds_path / 'train'
            val_path = ds_path / 'val'
            
            if train_path.exists():
                train_ds = ForgeryDataset(
                    image_dir=str(train_path / 'images'),
                    mask_dir=str(train_path / 'masks'),
                    transform=train_transform,
                    augment=True
                )
                if len(train_ds) > 0:
                    train_datasets.append(train_ds)
                    logger.info(f"Loaded {len(train_ds)} training samples from {ds_name}")
            
            if val_path.exists():
                val_ds = ForgeryDataset(
                    image_dir=str(val_path / 'images'),
                    mask_dir=str(val_path / 'masks'),
                    transform=val_transform,
                    augment=False
                )
                if len(val_ds) > 0:
                    val_datasets.append(val_ds)
                    logger.info(f"Loaded {len(val_ds)} validation samples from {ds_name}")
    
    if not train_datasets:
        logger.error("No training data found. Please download and setup datasets.")
        logger.info("See README.md for dataset download instructions.")
        sys.exit(1)
    
    # Combine datasets
    train_dataset = ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]
    val_dataset = ConcatDataset(val_datasets) if len(val_datasets) > 1 else (val_datasets[0] if val_datasets else None)
    
    logger.info(f"Total training samples: {len(train_dataset)}")
    if val_dataset:
        logger.info(f"Total validation samples: {len(val_dataset)}")
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
    
    # Build model
    from ai.models.forgery_localization import ForgeryLocalizationModel
    model_wrapper = ForgeryLocalizationModel(device=str(device))
    model = model_wrapper._build_model()
    model.to(device)
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    best_val_loss = float('inf')
    best_val_dice = 0
    training_log = []
    
    logger.info(f"Starting training for {args.epochs} epochs")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Validate
        val_loss = None
        val_metrics = None
        if val_loader:
            val_loss, val_metrics = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        epoch_time = time.time() - start_time
        
        # Log
        log_entry = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_metrics': val_metrics,
            'lr': optimizer.param_groups[0]['lr'],
            'time': epoch_time,
        }
        training_log.append(log_entry)
        
        logger.info(
            f"Epoch {epoch}/{args.epochs} - "
            f"Train Loss: {train_loss:.4f} - "
            f"Val Loss: {val_loss:.4f if val_loss else 'N/A'} - "
            f"Val Dice: {val_metrics['dice']:.4f if val_metrics else 'N/A'} - "
            f"LR: {optimizer.param_groups[0]['lr']:.6f} - "
            f"Time: {epoch_time:.1f}s"
        )
        
        # Save best model
        if val_loss is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), str(output_dir / 'best_model.pth'))
            logger.info(f"Saved best model (val_loss: {best_val_loss:.4f})")
        
        if val_metrics and val_metrics['dice'] > best_val_dice:
            best_val_dice = val_metrics['dice']
            torch.save(model.state_dict(), str(output_dir / 'best_dice_model.pth'))
        
        # Save checkpoint every 5 epochs
        if epoch % 5 == 0:
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_metrics': val_metrics,
                },
                str(output_dir / f'checkpoint_epoch_{epoch}.pth')
            )
    
    # Save final model
    torch.save(model.state_dict(), str(output_dir / 'forgery_localization.pth'))
    
    # Save training log
    with open(output_dir / 'training_log.json', 'w') as f:
        json.dump(training_log, f, indent=2)
    
    # Save training config
    config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'seed': args.seed,
        'device': str(device),
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset) if val_dataset else 0,
        'datasets_used': [ds_name for ds_name in ['CASIA', 'Columbia', 'NIST', 'Coverage'] 
                        if (dataset_dir / ds_name).exists()],
        'best_val_loss': best_val_loss,
        'best_val_dice': best_val_dice,
        'timestamp': datetime.now().isoformat(),
    }
    
    with open(output_dir / 'training_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info("Training complete!")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Best validation Dice: {best_val_dice:.4f}")
    logger.info(f"Model saved to: {output_dir / 'forgery_localization.pth'}")


if __name__ == '__main__':
    main()
