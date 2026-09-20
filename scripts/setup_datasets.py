"""
Dataset Setup Script

Downloads and organizes datasets for training and evaluation.
Run this before training the model.

Usage:
    python scripts/setup_datasets.py
"""

import os
import sys
import json
import shutil
import hashlib
from pathlib import Path
from typing import Dict, List


DATASETS = {
    'CASIA': {
        'description': 'CASIA Image Splicing Detection Evaluation Dataset v2.0',
        'url': 'https://pkorus.github.io/downloads/casia-dataset',
        'manipulation_types': ['splicing', 'copy-move'],
        'num_images': '~12,000',
        'has_masks': True,
        'usage': 'Training + Validation',
        'notes': 'Requires manual download from the website. Academic use only.',
    },
    'Columbia': {
        'description': 'Columbia Uncompressed Image Splicing Detection Dataset',
        'url': 'http://www.cs.columbia.edu/CAVE/databases/multimedia_splicing/',
        'manipulation_types': ['splicing'],
        'num_images': '~1,200',
        'has_masks': True,
        'usage': 'Training',
        'notes': 'Contains both authentic and spliced images with ground truth.',
    },
    'Coverage': {
        'description': 'Coverage Dataset for Copy-Move Forgery Detection',
        'url': 'https://github.com/ericjjj99/Coverage-dataset',
        'manipulation_types': ['copy-move'],
        'num_images': '100 pairs',
        'has_masks': True,
        'usage': 'Cross-dataset Testing',
        'notes': '100 original images with copy-move forgeries and ground truth masks.',
    },
    'DEFACTO': {
        'description': 'DEFACTO Dataset for Deepfake Detection',
        'url': 'https://defacto-fpi.github.io/',
        'manipulation_types': ['modern_editing', 'ai_generated', 'deepfake'],
        'num_images': '~1,000',
        'has_masks': True,
        'usage': 'Unseen Test Set',
        'notes': 'Modern manipulation types including AI-based editing.',
    },
    'NIST': {
        'description': 'NIST Nimble Challenge Media Forensics Dataset',
        'url': 'https://www.nist.gov/itl/iad/image-group/nimble-challenge-2017-evaluation',
        'manipulation_types': ['splicing', 'copy-move', 'enhancement'],
        'num_images': '~6,000',
        'has_masks': True,
        'usage': 'Validation',
        'notes': 'Government dataset with diverse manipulation types.',
    },
}


def create_directory_structure(base_dir: str):
    """Create the required directory structure for all datasets."""
    base = Path(base_dir)
    
    for ds_name in DATASETS:
        for split in ['train', 'val', 'test']:
            (base / ds_name / split / 'images').mkdir(parents=True, exist_ok=True)
            (base / ds_name / split / 'masks').mkdir(parents=True, exist_ok=True)
    
    print(f"✓ Created directory structure in {base_dir}")


def print_dataset_info():
    """Print information about all datasets."""
    print("\n" + "=" * 70)
    print("DATASET INFORMATION")
    print("=" * 70)
    
    for name, info in DATASETS.items():
        print(f"\n{name}")
        print("-" * 40)
        print(f"  Description: {info['description']}")
        print(f"  URL: {info['url']}")
        print(f"  Manipulation Types: {', '.join(info['manipulation_types'])}")
        print(f"  Number of Images: {info['num_images']}")
        print(f"  Has Ground Truth Masks: {'Yes' if info['has_masks'] else 'No'}")
        print(f"  Usage: {info['usage']}")
        print(f"  Notes: {info['notes']}")
    
    print("\n" + "=" * 70)
    print("RECOMMENDED SPLIT STRATEGY")
    print("=" * 70)
    print("""
    TRAINING:
      - CASIA v2.0 (train split)
      - Columbia (train split)
    
    VALIDATION:
      - CASIA v2.0 (val split)
      - NIST (val split)
    
    TESTING (Unseen):
      - Coverage (test split) - Cross-dataset evaluation
      - DEFACTO (test split) - Modern manipulation evaluation
    
    IMPORTANT:
      - NEVER use test data during training
      - Use perceptual hashing to remove near-duplicates across splits
      - Document all preprocessing steps for reproducibility
    """)


def verify_dataset(base_dir: str, dataset_name: str) -> Dict:
    """Verify a dataset is properly set up."""
    base = Path(base_dir) / dataset_name
    result = {
        'exists': base.exists(),
        'train_images': 0,
        'train_masks': 0,
        'val_images': 0,
        'val_masks': 0,
        'test_images': 0,
        'test_masks': 0,
    }
    
    if not base.exists():
        return result
    
    for split in ['train', 'val', 'test']:
        img_dir = base / split / 'images'
        mask_dir = base / split / 'masks'
        
        if img_dir.exists():
            result[f'{split}_images'] = len(list(img_dir.iterdir()))
        if mask_dir.exists():
            result[f'{split}_masks'] = len(list(mask_dir.iterdir()))
    
    return result


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Setup datasets for training')
    parser.add_argument('--base_dir', type=str, default='./datasets',
                       help='Base directory for datasets')
    parser.add_argument('--info', action='store_true',
                       help='Print dataset information')
    parser.add_argument('--verify', type=str, default=None,
                       help='Verify a specific dataset')
    parser.add_argument('--create-dirs', action='store_true',
                       help='Create directory structure')
    
    args = parser.parse_args()
    
    if args.info:
        print_dataset_info()
        return
    
    if args.verify:
        result = verify_dataset(args.base_dir, args.verify)
        print(f"\nDataset: {args.verify}")
        print(f"  Directory exists: {result['exists']}")
        print(f"  Train images: {result['train_images']}")
        print(f"  Train masks: {result['train_masks']}")
        print(f"  Val images: {result['val_images']}")
        print(f"  Val masks: {result['val_masks']}")
        print(f"  Test images: {result['test_images']}")
        print(f"  Test masks: {result['test_masks']}")
        return
    
    if args.create_dirs:
        create_directory_structure(args.base_dir)
        return
    
    # Default: show info and create dirs
    print_dataset_info()
    create_directory_structure(args.base_dir)
    
    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("""
    1. Download datasets from the URLs listed above
    2. Extract and place images/masks in the appropriate directories
    3. Verify setup: python scripts/setup_datasets.py --verify CASIA
    4. Train model: python training/train.py --dataset_dir ./datasets
    5. Evaluate: python evaluation/evaluate.py --test_dir ./datasets/Coverage/test
    """)


if __name__ == '__main__':
    main()
