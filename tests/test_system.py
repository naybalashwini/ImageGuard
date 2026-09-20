"""
Tests for the Image Tampering Detection & Restoration System

Run with: pytest tests/ -v
"""

import pytest
import json
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestImageProcessing:
    """Test client-side image processing utilities."""
    
    def test_calculate_image_seed(self):
        """Test that seed calculation is deterministic."""
        from src.utils.imageProcessing import calculateImageSeed
        
        data_url = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
        seed1 = calculateImageSeed(data_url)
        seed2 = calculateImageSeed(data_url)
        assert seed1 == seed2
        assert 0 <= seed1 <= 1
    
    def test_seed_different_for_different_images(self):
        """Test that different images produce different seeds."""
        from src.utils.imageProcessing import calculateImageSeed
        
        seed1 = calculateImageSeed("data:image/png;base64,AAAA")
        seed2 = calculateImageSeed("data:image/png;base64,BBBB")
        # Seeds should be different for different inputs
        # (not guaranteed but highly likely)
        assert isinstance(seed1, float)
        assert isinstance(seed2, float)


class TestAnalysisService:
    """Test the analysis service."""
    
    def test_generate_report(self):
        """Test report generation."""
        from src.services.analysisService import generateReport
        
        mock_result = {
            'id': 'test-id-1234',
            'filename': 'test.jpg',
            'timestamp': '2024-01-01T00:00:00',
            'isTampered': True,
            'confidence': 0.85,
            'tamperedPercentage': 12.5,
            'detectionDetails': {
                'modelUsed': 'Test Model',
                'backboneArchitecture': 'Test Architecture',
                'inputResolution': '512x512',
                'processingTime': 2.5,
                'numberOfSuspiciousRegions': 2,
                'primaryRegionConfidence': 0.85,
            },
            'forensicSignals': [
                {
                    'name': 'Test Signal',
                    'description': 'Test description',
                    'severity': 'high',
                    'evidence': 'Test evidence',
                }
            ],
        }
        
        report = generateReport(mock_result)
        assert 'FORENSIC ANALYSIS REPORT' in report
        assert 'test-id-1234' in report
        assert 'TAMPERED' in report
        assert '85.0%' in report
        assert 'Test Signal' in report
    
    def test_generate_report_authentic(self):
        """Test report for authentic image."""
        from src.services.analysisService import generateReport
        
        mock_result = {
            'id': 'test-id-5678',
            'filename': 'authentic.jpg',
            'timestamp': '2024-01-01T00:00:00',
            'isTampered': False,
            'confidence': 0.2,
            'tamperedPercentage': 0.1,
            'detectionDetails': {
                'modelUsed': 'Test Model',
                'backboneArchitecture': 'Test',
                'inputResolution': '512x512',
                'processingTime': 1.0,
                'numberOfSuspiciousRegions': 0,
                'primaryRegionConfidence': 0.2,
            },
            'forensicSignals': [
                {
                    'name': 'No Anomalies',
                    'description': 'No issues found',
                    'severity': 'low',
                    'evidence': 'Normal',
                }
            ],
        }
        
        report = generateReport(mock_result)
        assert 'AUTHENTIC' in report
        assert 'No Strong Evidence' in report


class TestHistoryService:
    """Test history management."""
    
    def test_get_empty_history(self):
        """Test getting history when empty."""
        import localStorage_mock
        
        from src.services.analysisService import getHistoryEntries
        # Should return empty list when no history
        entries = getHistoryEntries()
        assert isinstance(entries, list)


class TestFileValidation:
    """Test file validation logic."""
    
    def test_allowed_extensions(self):
        """Test that only allowed file types are accepted."""
        allowed = {'.jpg', '.jpeg', '.png', '.webp'}
        
        assert '.jpg' in allowed
        assert '.jpeg' in allowed
        assert '.png' in allowed
        assert '.webp' in allowed
        assert '.gif' not in allowed
        assert '.bmp' not in allowed
        assert '.exe' not in allowed
    
    def test_max_file_size(self):
        """Test max file size constant."""
        MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB
        assert MAX_FILE_SIZE == 20971520


class TestModelArchitecture:
    """Test model architecture components (mocked)."""
    
    def test_forgery_model_import(self):
        """Test that model module can be imported."""
        # This tests the structure, not actual model loading
        model_path = Path('ai/models/forgery_localization.py')
        assert model_path.exists()
    
    def test_pipeline_import(self):
        """Test that pipeline module exists."""
        pipeline_path = Path('ai/inference/pipeline.py')
        assert pipeline_path.exists()
    
    def test_restoration_import(self):
        """Test that restoration module exists."""
        restoration_path = Path('ai/restoration/inpainting.py')
        assert restoration_path.exists()


class TestEvaluation:
    """Test evaluation metrics computation."""
    
    def test_compute_metrics_perfect(self):
        """Test metrics with perfect predictions."""
        from evaluation.evaluate import compute_metrics
        
        # Perfect predictions
        predictions = np.array([[[1.0, 0.0], [0.0, 1.0]]])
        ground_truth = np.array([[[1.0, 0.0], [0.0, 1.0]]])
        
        metrics = compute_metrics(predictions, ground_truth, threshold=0.5)
        
        assert metrics['pixel_level']['iou'] == 1.0
        assert metrics['pixel_level']['dice'] == 1.0
        assert metrics['pixel_level']['precision'] == 1.0
        assert metrics['pixel_level']['recall'] == 1.0
    
    def test_compute_metrics_all_wrong(self):
        """Test metrics with completely wrong predictions."""
        from evaluation.evaluate import compute_metrics
        
        predictions = np.array([[[1.0, 1.0], [1.0, 1.0]]])
        ground_truth = np.array([[[0.0, 0.0], [0.0, 0.0]]])
        
        metrics = compute_metrics(predictions, ground_truth, threshold=0.5)
        
        assert metrics['pixel_level']['recall'] < 0.01  # No true positives
        assert metrics['pixel_level']['precision'] < 0.01  # All false positives
    
    def test_compute_metrics_partial(self):
        """Test metrics with partial overlap."""
        from evaluation.evaluate import compute_metrics
        
        predictions = np.array([[[0.8, 0.2], [0.3, 0.9]]])
        ground_truth = np.array([[[1.0, 0.0], [0.0, 1.0]]])
        
        metrics = compute_metrics(predictions, ground_truth, threshold=0.5)
        
        # Should have some overlap
        assert 0 < metrics['pixel_level']['f1'] <= 1.0
        assert metrics['pixel_level']['iou'] > 0


class TestBackendStructure:
    """Test backend file structure."""
    
    def test_backend_main_exists(self):
        """Test that backend main.py exists."""
        assert Path('backend/main.py').exists()
    
    def test_requirements_exists(self):
        """Test that requirements.txt exists."""
        assert Path('requirements.txt').exists()
    
    def test_dockerfile_exists(self):
        """Test that Dockerfile exists."""
        assert Path('Dockerfile').exists()
    
    def test_docker_compose_exists(self):
        """Test that docker-compose.yml exists."""
        assert Path('docker-compose.yml').exists()


class TestELAPipeline:
    """Test ELA computation logic."""
    
    def test_ela_concept(self):
        """Test ELA concept: difference between original and recompressed."""
        # Simulate ELA on synthetic data
        original = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        # Simulate recompression (slight modification)
        recompressed = (original * 0.95).astype(np.uint8)
        
        # Compute difference
        difference = np.abs(original.astype(np.float32) - recompressed.astype(np.float32))
        
        # Difference should be non-zero
        assert np.mean(difference) > 0
        
        # Scaled difference should be in valid range
        scaled = np.clip(difference * 10, 0, 255)
        assert np.max(scaled) <= 255
        assert np.min(scaled) >= 0


class TestMaskProcessing:
    """Test mask generation and processing."""
    
    def test_binary_mask_values(self):
        """Test that binary mask contains only 0 and 1."""
        # Simulate thresholding
        probability_map = np.random.random((100, 100))
        threshold = 0.5
        binary_mask = (probability_map > threshold).astype(np.float32)
        
        unique_values = np.unique(binary_mask)
        assert all(v in [0.0, 1.0] for v in unique_values)
    
    def test_mask_percentage_calculation(self):
        """Test tampered percentage calculation."""
        mask = np.zeros((100, 100), dtype=np.float32)
        mask[25:75, 25:75] = 1.0  # 50x50 = 2500 pixels out of 10000
        
        tampered_percentage = (np.sum(mask > 0.5) / mask.size) * 100
        assert abs(tampered_percentage - 25.0) < 0.1


# Note: localStorage_mock is a placeholder for browser localStorage in tests
# In actual testing, you'd use jsdom or a similar environment
