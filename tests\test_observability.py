"""
Unit tests for observability map computation.
"""

import pytest
import torch
import numpy as np
from core.observability_math import generate_observability_map


class TestObservabilityMap:
    """Test cases for observability map generation."""
    
    def test_basic_computation(self):
        """Test basic observability map computation."""
        gt = torch.rand(1, 3, 64, 64)
        restored = [torch.rand(1, 3, 64, 64) for _ in range(3)]
        
        obs_map, stats = generate_observability_map(gt, restored)
        
        assert obs_map.shape == (1, 1, 64, 64)
        assert torch.all(obs_map >= 0) and torch.all(obs_map <= 1)
        assert 'error_map' in stats
        assert 'variance_map' in stats
    
    def test_perfect_restoration(self):
        """Test with perfect restoration (should give high observability)."""
        gt = torch.ones(1, 3, 32, 32) * 0.5
        restored = [gt.clone() for _ in range(3)]
        
        obs_map, stats = generate_observability_map(gt, restored)
        
        # Perfect restoration should have high observability
        assert obs_map.mean() > 0.9
        assert stats['mean_error'] < 1e-6
        assert stats['mean_variance'] < 1e-6
    
    def test_completely_wrong(self):
        """Test with completely wrong restoration."""
        gt = torch.ones(1, 3, 32, 32) * 0.5
        restored = [torch.zeros(1, 3, 32, 32) for _ in range(3)]
        
        obs_map, stats = generate_observability_map(gt, restored)
        
        # Wrong restoration should have low observability
        assert obs_map.mean() < 0.5
        assert stats['mean_error'] > 0.1
    
    def test_high_variance(self):
        """Test with high variance across models."""
        gt = torch.ones(1, 3, 32, 32) * 0.5
        restored = [
            torch.zeros(1, 3, 32, 32),
            torch.ones(1, 3, 32, 32),
            torch.ones(1, 3, 32, 32) * 0.5
        ]
        
        obs_map, stats = generate_observability_map(gt, restored)
        
        # High variance should reduce observability
        assert stats['mean_variance'] > 0.1
    
    def test_empty_restored_list(self):
        """Test that empty restored list raises error."""
        gt = torch.rand(1, 3, 32, 32)
        
        with pytest.raises(ValueError):
            generate_observability_map(gt, [])
    
    def test_invalid_gt_dimensions(self):
        """Test that invalid GT dimensions raise error."""
        gt = torch.rand(3, 32, 32)  # Missing batch dimension
        restored = [torch.rand(1, 3, 32, 32)]
        
        with pytest.raises(ValueError):
            generate_observability_map(gt, restored)
    
    def test_different_sizes(self):
        """Test handling of different image sizes."""
        gt = torch.rand(1, 3, 64, 64)
        restored = [
            torch.rand(1, 3, 32, 32),
            torch.rand(1, 3, 128, 128)
        ]
        
        obs_map, stats = generate_observability_map(gt, restored)
        
        # Should resize all to GT size
        assert obs_map.shape == (1, 1, 64, 64)


class TestObservabilityMetrics:
    """Test cases for observability metrics."""
    
    def test_stats_dict_contents(self):
        """Test that stats dict contains all expected keys."""
        gt = torch.rand(1, 3, 32, 32)
        restored = [torch.rand(1, 3, 32, 32) for _ in range(3)]
        
        obs_map, stats = generate_observability_map(gt, restored)
        
        expected_keys = [
            'error_map', 'variance_map', 'consensus_mean',
            'num_models', 'alpha', 'beta',
            'mean_error', 'mean_variance', 'mean_observability'
        ]
        
        for key in expected_keys:
            assert key in stats
    
    def test_parameter_effects(self):
        """Test that alpha and beta parameters affect output."""
        gt = torch.rand(1, 3, 32, 32)
        restored = [torch.rand(1, 3, 32, 32) for _ in range(3)]
        
        obs_map1, _ = generate_observability_map(gt, restored, alpha=1.0, beta=0.5)
        obs_map2, _ = generate_observability_map(gt, restored, alpha=2.0, beta=0.5)
        
        # Higher alpha should generally give lower observability
        assert obs_map2.mean() <= obs_map1.mean()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
