"""
Test the validation module for empirical replication checks.

Tests that:
1. SIZE variable validation works correctly
2. LQP/UQP correlation validation works
3. Asymmetry premium sign validation works
4. MKT_VOL coefficient validation works
5. DOWN_ASY sign validation works
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.python.validation import (
    ReplicationValidator,
    ValidationError,
    validate_empirical_results
)


class TestSizeValidation:
    """Test SIZE variable validation."""
    
    def test_correct_size_passes(self):
        """SIZE ~5.0 (log millions) should pass."""
        validator = ReplicationValidator(strict=False)
        data = pd.DataFrame({'SIZE': np.random.normal(5.0, 1.0, 1000)})
        assert validator.validate_size_variable(data) == True
    
    def test_wrong_size_fails(self):
        """SIZE ~14.0 (log dollars/thousands) should fail."""
        validator = ReplicationValidator(strict=False)
        data = pd.DataFrame({'SIZE': np.random.normal(14.0, 2.0, 1000)})
        assert validator.validate_size_variable(data) == False
    
    def test_missing_column_fails(self):
        """Missing SIZE column should fail."""
        validator = ReplicationValidator(strict=False)
        data = pd.DataFrame({'OTHER': np.random.normal(5.0, 1.0, 1000)})
        assert validator.validate_size_variable(data) == False


class TestLQPUQPCorrelation:
    """Test LQP/UQP correlation validation."""
    
    def test_correct_correlation_passes(self):
        """Correlation ~-0.08 should pass."""
        validator = ReplicationValidator(strict=False)
        np.random.seed(42)
        n = 1000
        
        # Generate nearly independent LQP and UQP
        lqp = np.random.normal(0.15, 0.05, n)
        uqp = 0.15 + 0.02 * np.random.randn(n)  # Very weak relation
        
        data = pd.DataFrame({'LQP': lqp, 'UQP': uqp})
        # Correlation should be close to 0 (within -0.15 to 0.05)
        result = validator.validate_lqp_uqp_correlation(data)
        # This test depends on random data, check result is boolean-like
        assert result is True or result == True or bool(result) == True
    
    def test_too_negative_correlation_fails(self):
        """Correlation ~-0.5 should fail (too negative)."""
        validator = ReplicationValidator(strict=False)
        np.random.seed(42)
        n = 1000
        
        # Generate strongly negatively correlated LQP and UQP
        lqp = np.random.normal(0.15, 0.05, n)
        uqp = 0.30 - lqp + 0.01 * np.random.randn(n)  # Strong negative
        
        data = pd.DataFrame({'LQP': lqp, 'UQP': uqp})
        assert validator.validate_lqp_uqp_correlation(data) == False


class TestAsymmetryPremium:
    """Test asymmetry premium validation."""
    
    def test_positive_premium_passes(self):
        """Positive and significant premium should pass."""
        validator = ReplicationValidator(strict=False)
        np.random.seed(42)
        # Need higher mean and lower std to ensure significance
        premium = pd.Series(np.random.normal(0.006, 0.015, 120))  # ~0.6% mean
        assert validator.validate_asymmetry_premium(premium) == True
    
    def test_negative_premium_fails(self):
        """Negative premium should fail."""
        validator = ReplicationValidator(strict=False)
        np.random.seed(42)
        premium = pd.Series(np.random.normal(-0.004, 0.015, 120))
        # With this setup, mean is negative and significant
        result = validator.validate_asymmetry_premium(premium)
        # Premium being negative means the first check (premium > 0) fails
        assert result == False


class TestMKTVOLCoefficient:
    """Test MKT_VOL coefficient validation."""
    
    def test_negative_coef_passes(self):
        """Negative MKT_VOL coefficient should pass."""
        validator = ReplicationValidator(strict=False)
        assert validator.validate_mkt_vol_coefficient(-0.124, -2.5) == True
    
    def test_positive_coef_fails(self):
        """Positive MKT_VOL coefficient should fail."""
        validator = ReplicationValidator(strict=False)
        assert validator.validate_mkt_vol_coefficient(0.05, 1.2) == False


class TestDOWNASYSign:
    """Test DOWN_ASY sign validation."""
    
    def test_correct_positive_sign(self):
        """LQP > UQP should give positive DOWN_ASY."""
        validator = ReplicationValidator(strict=False)
        # LQP > UQP => crash-prone => DOWN_ASY should be positive
        assert validator.validate_down_asy_sign(
            lqp=0.20, uqp=0.15, down_asy=0.05
        ) == True
    
    def test_correct_negative_sign(self):
        """LQP < UQP should give negative DOWN_ASY."""
        validator = ReplicationValidator(strict=False)
        # LQP < UQP => boom-prone => DOWN_ASY should be negative
        assert validator.validate_down_asy_sign(
            lqp=0.12, uqp=0.18, down_asy=-0.03
        ) == True
    
    def test_wrong_positive_sign(self):
        """LQP < UQP with positive DOWN_ASY should fail."""
        validator = ReplicationValidator(strict=False)
        # LQP < UQP but DOWN_ASY is positive => wrong sign
        assert validator.validate_down_asy_sign(
            lqp=0.12, uqp=0.18, down_asy=0.03
        ) == False
    
    def test_wrong_negative_sign(self):
        """LQP > UQP with negative DOWN_ASY should fail."""
        validator = ReplicationValidator(strict=False)
        # LQP > UQP but DOWN_ASY is negative => wrong sign
        assert validator.validate_down_asy_sign(
            lqp=0.20, uqp=0.15, down_asy=-0.05
        ) == False


class TestSortingValidation:
    """Test sorting direction validation."""
    
    def test_correct_sorting_passes(self):
        """Monotonically increasing DOWN_ASY by decile should pass."""
        validator = ReplicationValidator(strict=False)
        data = pd.DataFrame({
            'DOWN_ASY': list(range(1, 11)),  # Increasing
            'DECILE': list(range(1, 11))
        })
        assert validator.validate_sorting_direction(data) == True
    
    def test_wrong_sorting_fails(self):
        """Decreasing DOWN_ASY by decile should fail."""
        validator = ReplicationValidator(strict=False)
        data = pd.DataFrame({
            'DOWN_ASY': list(range(10, 0, -1)),  # Decreasing
            'DECILE': list(range(1, 11))
        })
        assert validator.validate_sorting_direction(data) == False


class TestStrictMode:
    """Test strict validation mode."""
    
    def test_strict_raises_on_failure(self):
        """Strict mode should raise ValidationError on failure."""
        validator = ReplicationValidator(strict=True)
        wrong_size = pd.DataFrame({'SIZE': np.random.normal(14.0, 2.0, 1000)})
        
        with pytest.raises(ValidationError):
            validator.validate_size_variable(wrong_size)
    
    def test_non_strict_warns_but_continues(self):
        """Non-strict mode should warn but not raise."""
        validator = ReplicationValidator(strict=False)
        wrong_size = pd.DataFrame({'SIZE': np.random.normal(14.0, 2.0, 1000)})
        
        # Should not raise
        result = validator.validate_size_variable(wrong_size)
        assert result == False  # But should return False


class TestFullValidation:
    """Test complete validation suite."""
    
    def test_full_validation_with_good_data(self):
        """Full validation with correct data should pass."""
        np.random.seed(42)
        
        # Create correct data
        table3_data = pd.DataFrame({
            'LQP': np.random.normal(0.15, 0.05, 1000),
            'UQP': 0.15 + 0.02 * np.random.randn(1000)
        })
        
        table4_data = pd.DataFrame({
            'SIZE': np.random.normal(5.0, 1.0, 1000),
            'DOWN_ASY': list(range(1, 11)) * 100,
            'DECILE': list(range(1, 11)) * 100
        })
        
        validator = ReplicationValidator(strict=False)
        results = validator.run_full_validation(
            table3_data=table3_data,
            table4_data=table4_data
        )
        
        assert isinstance(results, dict)
        assert 'all_passed' in results
        assert 'results' in results


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
