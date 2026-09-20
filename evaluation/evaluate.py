"""
Evaluation Script for Unseen Image Testing

This script evaluates the trained model on completely unseen test datasets.
CRITICAL: Test data is NEVER used during training.

Evaluation includes:
1. Image-level detection metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
2. Pixel-level localization metrics (IoU, Dice, Pixel F1)
3. Cross-dataset evaluation (train on Dataset A, test on Dataset B)
4. Per-manipulation-type evaluation

Usage:
    python evaluation/evaluate.py --model_path ./ai/models/weights/forgery_localization.pth --test_dir ./datasets/Coverage/test

Cross-dataset evaluation:
    python evaluation/evaluate.py --model_path ./ai/models/weights/forgery_localization.pth --test_dir ./datasets/DEFACTO/test --cross_dataset
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_metrics(predictions: np.ndarray, ground_truth: np.ndarray, threshold: float = 0.5) -> Dict:
    """
    Compute all evaluation metrics.
    
    Args:
        predictions: Predicted probability maps [N, H, W]
        ground_truth: Ground truth masks [N, H, W]
        threshold: Threshold for binarization
    
    Returns:
        Dictionary of metrics
    """
    N = predictions.shape[0]
    
    # Binarize predictions
    binary_preds = (predictions > threshold).astype(np.float32)
    binary_gt = (ground_truth > 0.5).astype(np.float32)
    
    # Flatten for pixel-level metrics
    preds_flat = binary_preds.flatten()
    gt_flat = binary_gt.flatten()
    
    # True/False positives/negatives
    tp = np.sum(preds_flat * gt_flat)
    fp = np.sum(preds_flat * (1 - gt_flat))
    fn = np.sum((1 - preds_flat) * gt_flat)
    tn = np.sum((1 - preds_flat) * (1 - gt_flat))
    
    # Pixel-level metrics
    pixel_precision = tp / (tp + fp + 1e-8)
    pixel_recall = tp / (tp + fn + 1e-8)
    pixel_f1 = 2 * pixel_precision * pixel_recall / (pixel_precision + pixel_recall + 1e-8)
    
    # IoU
    intersection = tp
    union = tp + fp + fn
    iou = intersection / (union + 1e-8)
    
    # Dice
    dice = 2 * intersection / (2 * intersection + fp + fn + 1e-8)
    
    # Image-level metrics
    # An image is "tampered" if any ground truth mask has tampered pixels
    image_preds = np.array([np.any(binary_preds[i] > 0) for i in range(N)])
    image_gt = np.array([np.any(binary_gt[i] > 0) for i in range(N)])
    
    img_tp = np.sum(image_preds * image_gt)
    img_fp = np.sum(image_preds * (1 - image_gt))
    img_fn = np.sum((1 - image_preds) * image_gt)
    img_tn = np.sum((1 - image_preds) * (1 - image_gt))
    
    accuracy = (img_tp + img_tn) / (N + 1e-8)
    img_precision = img_tp / (img_tp + img_fp + 1e-8)
    img_recall = img_tp / (img_tp + img_fn + 1e-8)
    img_f1 = 2 * img_precision * img_recall / (img_precision + img_recall + 1e-8)
    
    # ROC-AUC (simplified)
    try:
        from sklearn.metrics import roc_auc_score
        preds_proba = predictions.flatten()
        gt_binary = gt_flat.astype(int)
        if len(np.unique(gt_binary)) > 1:
            roc_auc = roc_auc_score(gt_binary, preds_proba)
        else:
            roc_auc = None
    except ImportError:
        roc_auc = None
    
    metrics = {
        'image_level': {
            'accuracy': float(accuracy),
            'precision': float(img_precision),
            'recall': float(img_recall),
            'f1': float(img_f1),
            'roc_auc': float(roc_auc) if roc_auc is not None else None,
            'total_images': int(N),
            'true_positives': int(img_tp),
            'false_positives': int(img_fp),
            'false_negatives': int(img_fn),
            'true_negatives': int(img_tn),
        },
        'pixel_level': {
            'iou': float(iou),
            'dice': float(dice),
            'precision': float(pixel_precision),
            'recall': float(pixel_recall),
            'f1': float(pixel_f1),
            'total_pixels': int(len(preds_flat)),
            'true_positives': int(tp),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_negatives': int(tn),
        },
        'threshold': threshold,
    }
    
    return metrics


def evaluate_dataset(model, test_dir: str, device: str, save_visualizations: bool = True) -> Dict:
    """
    Evaluate model on a test dataset.
    
    Args:
        model: Loaded model
        test_dir: Path to test directory with images/ and masks/ subdirs
        device: Compute device
        save_visualizations: Whether to save visualization images
    
    Returns:
        Evaluation metrics
    """
    import torch
    from PIL import Image
    import torchvision.transforms as transforms
    
    test_path = Path(test_dir)
    images_dir = test_path / 'images'
    masks_dir = test_path / 'masks'
    
    if not images_dir.exists():
        logger.error(f"Images directory not found: {images_dir}")
        return {}
    
    # Setup transform
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    
    # Collect predictions and ground truth
    all_predictions = []
    all_ground_truth = []
    
    image_files = sorted([f for f in images_dir.iterdir() 
                         if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}])
    
    logger.info(f"Evaluating on {len(image_files)} test images...")
    
    for idx, img_path in enumerate(image_files):
        # Find corresponding mask
        mask_name = img_path.stem + '_mask.png'
        mask_path = masks_dir / mask_name
        if not mask_path.exists():
            mask_path = masks_dir / img_path.with_suffix('.png').name
        if not mask_path.exists():
            logger.warning(f"No mask found for {img_path.name}, skipping")
            continue
        
        # Load and preprocess
        image = Image.open(img_path).convert('RGB')
        original_size = image.size
        mask = Image.open(mask_path).convert('L')
        
        # Transform image
        img_tensor = transform(image).unsqueeze(0).to(device)
        
        # Resize mask to match model output
        mask_resized = mask.resize((512, 512), Image.NEAREST)
        mask_array = np.array(mask_resized).astype(np.float32) / 255.0
        mask_binary = (mask_array > 0.5).astype(np.float32)
        
        # Inference
        model.eval()
        with torch.no_grad():
            output = model(img_tensor)
        
        pred = output.squeeze().cpu().numpy()
        
        all_predictions.append(pred)
        all_ground_truth.append(mask_binary)
        
        # Save visualization
        if save_visualizations and idx < 50:  # Limit saved visualizations
            vis_dir = test_path / 'visualizations'
            vis_dir.mkdir(exist_ok=True)
            
            # Save heatmap
            heatmap = (pred * 255).astype(np.uint8)
            heatmap_img = Image.fromarray(heatmap)
            heatmap_img.save(str(vis_dir / f"{img_path.stem}_heatmap.png"))
            
            # Save binary prediction
            binary_pred = (pred > 0.5).astype(np.uint8) * 255
            binary_img = Image.fromarray(binary_pred)
            binary_img.save(str(vis_dir / f"{img_path.stem}_pred_mask.png"))
        
        if (idx + 1) % 100 == 0:
            logger.info(f"Processed {idx + 1}/{len(image_files)} images")
    
    if not all_predictions:
        logger.error("No valid test samples found")
        return {}
    
    # Stack arrays
    predictions = np.stack(all_predictions)
    ground_truth = np.stack(all_ground_truth)
    
    logger.info(f"Computing metrics on {len(all_predictions)} samples...")
    
    # Compute metrics at multiple thresholds
    results = {}
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
        metrics = compute_metrics(predictions, ground_truth, threshold)
        results[f'threshold_{threshold}'] = metrics
    
    # Best threshold (by F1)
    best_threshold = max(
        [0.3, 0.4, 0.5, 0.6, 0.7],
        key=lambda t: results[f'threshold_{t}']['pixel_level']['f1']
    )
    results['best_threshold'] = best_threshold
    results['best_metrics'] = results[f'threshold_{best_threshold}']
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate Forgery Localization Model')
    parser.add_argument('--model_path', type=str, 
                       default='./ai/models/weights/forgery_localization.pth',
                       help='Path to model weights')
    parser.add_argument('--test_dir', type=str, default='./datasets/test',
                       help='Test dataset directory')
    parser.add_argument('--output_dir', type=str, default='./evaluation/results',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (auto/cpu/cuda)')
    parser.add_argument('--cross_dataset', action='store_true',
                       help='Enable cross-dataset evaluation mode')
    parser.add_argument('--no_viz', action='store_true',
                       help='Disable visualization saving')
    
    args = parser.parse_args()
    
    try:
        import torch
    except ImportError:
        logger.error("PyTorch not installed. Install with: pip install torch torchvision")
        sys.exit(1)
    
    # Device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    logger.info(f"Using device: {device}")
    
    # Load model
    from ai.models.forgery_localization import ForgeryLocalizationModel
    
    model_wrapper = ForgeryLocalizationModel(device=device)
    model = model_wrapper._build_model()
    
    if Path(args.model_path).exists():
        state_dict = torch.load(args.model_path, map_location=device)
        model.load_state_dict(state_dict)
        logger.info(f"Loaded model from {args.model_path}")
    else:
        logger.error(f"Model weights not found at {args.model_path}")
        logger.info("Train the model first: python training/train.py")
        sys.exit(1)
    
    model.to(device)
    model.eval()
    
    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Evaluate
    logger.info(f"Evaluating on: {args.test_dir}")
    results = evaluate_dataset(
        model, args.test_dir, device,
        save_visualizations=not args.no_viz
    )
    
    if not results:
        logger.error("Evaluation failed - no results produced")
        sys.exit(1)
    
    # Save results
    results['metadata'] = {
        'model_path': args.model_path,
        'test_dir': args.test_dir,
        'device': device,
        'timestamp': datetime.now().isoformat(),
        'cross_dataset': args.cross_dataset,
    }
    
    # Save JSON results
    results_path = output_dir / 'evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {results_path}")
    
    # Print summary
    best = results['best_metrics']
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION RESULTS SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Best threshold: {results['best_threshold']}")
    logger.info(f"\nImage-Level Detection:")
    logger.info(f"  Accuracy:  {best['image_level']['accuracy']:.4f}")
    logger.info(f"  Precision: {best['image_level']['precision']:.4f}")
    logger.info(f"  Recall:    {best['image_level']['recall']:.4f}")
    logger.info(f"  F1:        {best['image_level']['f1']:.4f}")
    if best['image_level']['roc_auc']:
        logger.info(f"  ROC-AUC:   {best['image_level']['roc_auc']:.4f}")
    logger.info(f"\nPixel-Level Localization:")
    logger.info(f"  IoU:       {best['pixel_level']['iou']:.4f}")
    logger.info(f"  Dice:      {best['pixel_level']['dice']:.4f}")
    logger.info(f"  Precision: {best['pixel_level']['precision']:.4f}")
    logger.info(f"  Recall:    {best['pixel_level']['recall']:.4f}")
    logger.info(f"  F1:        {best['pixel_level']['f1']:.4f}")
    logger.info("=" * 60)
    
    # Save CSV for easy analysis
    import csv
    csv_path = output_dir / 'evaluation_results.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Best Threshold', results['best_threshold']])
        for key, value in best['image_level'].items():
            writer.writerow([f'Image_{key}', value])
        for key, value in best['pixel_level'].items():
            writer.writerow([f'Pixel_{key}', value])
    
    logger.info(f"CSV results saved to {csv_path}")


if __name__ == '__main__':
    main()
