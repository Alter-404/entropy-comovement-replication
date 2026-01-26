"""
Test Suite for Phase 5: Advanced Characteristics and Final Reporting

Tests:
1. Firm characteristics computation accuracy
2. Fama-MacBeth correlation methodology
3. Summary statistics generation
4. Time-series regression functionality
5. Integration with Phase 4 outputs
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent / 'src' / 'python'))

from characteristics import CharacteristicsEngine


class TestCharacteristics:
    """Tests for firm characteristics computation."""
    
    def test_beta_computation(self):
        """Test CAPM beta calculation."""
        # Create synthetic data with known beta
        np.random.seed(42)
        n = 252
        
        # Market returns
        r_m = np.random.randn(n) * 0.02
        
        # Stock returns with beta = 1.5
        true_beta = 1.5
        r_i = 0.001 + true_beta * r_m + np.random.randn(n) * 0.01
        
        # Compute beta
        engine = CharacteristicsEngine()
        beta, ivol = engine.compute_beta_and_ivol(r_i, r_m)
        
        # Check beta accuracy
        assert abs(beta - true_beta) < 0.1, f"Beta={beta:.2f}, expected ~{true_beta}"
        assert ivol > 0, "IVOL should be positive"
        assert ivol < 0.02, "IVOL should be reasonable"
        
        print(f"✓ Beta test passed: computed={beta:.3f}, true={true_beta}")
    
    def test_downside_beta(self):
        """Test downside beta computation."""
        np.random.seed(42)
        n = 252
        
        # Market returns
        r_m = np.random.randn(n) * 0.02
        
        # Stock more sensitive during downturns
        r_i = np.zeros(n)
        for t in range(n):
            if r_m[t] < 0:
                r_i[t] = 2.0 * r_m[t]  # Stronger reaction downside
            else:
                r_i[t] = 1.0 * r_m[t]  # Normal reaction upside
        
        r_i += np.random.randn(n) * 0.005
        
        # Compute
        engine = CharacteristicsEngine()
        beta_down = engine.compute_downside_beta(r_i, r_m)
        beta, _ = engine.compute_beta_and_ivol(r_i, r_m)
        
        # Downside beta should be larger
        assert beta_down > beta, f"Downside beta ({beta_down:.2f}) should exceed regular beta ({beta:.2f})"
        
        print(f"✓ Downside beta test passed: β={beta:.2f}, β⁻={beta_down:.2f}")
    
    def test_coskewness(self):
        """Test coskewness calculation."""
        np.random.seed(42)
        n = 500
        
        # Market returns
        r_m = np.random.randn(n) * 0.02
        
        # Stock with negative coskewness (crash risk)
        r_i = np.zeros(n)
        for t in range(n):
            if r_m[t] < -0.02:  # Market crash
                r_i[t] = -0.05  # Stock crashes too
            else:
                r_i[t] = 0.8 * r_m[t]
        
        r_i += np.random.randn(n) * 0.01
        
        # Compute
        engine = CharacteristicsEngine()
        coskew = engine.compute_coskewness(r_i, r_m)
        
        # Should be negative (crash risk)
        assert coskew < 0, f"Coskewness should be negative for crash-prone stock: {coskew:.3f}"
        
        print(f"✓ Coskewness test passed: {coskew:.3f}")
    
    def test_illiquidity(self):
        """Test Amihud illiquidity measure."""
        np.random.seed(42)
        n = 252
        
        # Returns
        returns = np.random.randn(n) * 0.02
        
        # Volume (larger volume → lower illiquidity)
        volume_high = np.ones(n) * 1e6
        volume_low = np.ones(n) * 1e4
        
        # Price
        price = np.ones(n) * 50.0
        
        # Compute
        engine = CharacteristicsEngine()
        illiq_high_vol = engine.compute_illiquidity(returns, volume_high, price)
        illiq_low_vol = engine.compute_illiquidity(returns, volume_low, price)
        
        # Low volume should have higher illiquidity
        assert illiq_low_vol > illiq_high_vol, \
            f"Low volume illiq ({illiq_low_vol:.2f}) should exceed high volume ({illiq_high_vol:.2f})"
        
        print(f"✓ Illiquidity test passed: high_vol={illiq_high_vol:.2f}, low_vol={illiq_low_vol:.2f}")
    
    def test_rolling_window_computation(self):
        """Test full rolling window characteristic computation."""
        np.random.seed(42)
        
        # Create synthetic dataset
        n_stocks = 5
        n_days = 500
        dates = pd.date_range('2020-01-01', periods=n_days, freq='D')
        
        data = []
        for stock_id in range(1, n_stocks + 1):
            for date in dates:
                data.append({
                    'PERMNO': stock_id,
                    'DATE': date,
                    'RET': np.random.randn() * 0.02,
                    'VOLUME': 1e5 + np.random.rand() * 1e5,
                    'PRICE': 50 + np.random.randn() * 10
                })
        
        stock_df = pd.DataFrame(data)
        
        # Market returns
        mkt_data = []
        for date in dates:
            mkt_data.append({
                'DATE': date,
                'MKT_RF': np.random.randn() * 0.015
            })
        mkt_df = pd.DataFrame(mkt_data)
        
        # Compute characteristics
        engine = CharacteristicsEngine(window_size=252, min_observations=100)
        chars = engine.compute_all_characteristics(stock_df, mkt_df)
        
        # Checks
        assert len(chars) > 0, "Should produce some characteristics"
        assert 'PERMNO' in chars.columns, "Should have PERMNO"
        assert 'DATE' in chars.columns, "Should have DATE"
        assert 'BETA' in chars.columns, "Should have BETA"
        assert 'IVOL' in chars.columns, "Should have IVOL"
        
        # All characteristics should be finite where non-NaN
        for col in ['BETA', 'IVOL', 'DOWNSIDE_BETA', 'COSKEW']:
            if col in chars.columns:
                valid_vals = chars[col].dropna()
                assert all(np.isfinite(valid_vals)), f"{col} should have finite values"
        
        print(f"✓ Rolling window test passed: {len(chars)} observations computed")


class TestFamaMacBeth:
    """Tests for Fama-MacBeth methodology."""
    
    def test_correlation_computation(self):
        """Test cross-sectional correlation averaging."""
        from scripts.replicate_table3 import fama_macbeth_correlation
        
        # Create panel data with known correlation
        np.random.seed(42)
        n_stocks = 100
        n_months = 60
        
        data_list = []
        for month in range(n_months):
            for stock in range(n_stocks):
                # Create correlated variables (ρ ≈ 0.7)
                x = np.random.randn()
                y = 0.7 * x + np.random.randn() * np.sqrt(1 - 0.7**2)
                
                data_list.append({
                    'DATE': pd.Timestamp('2020-01-01') + pd.DateOffset(months=month),
                    'STOCK': stock,
                    'VAR1': x,
                    'VAR2': y
                })
        
        data = pd.DataFrame(data_list)
        
        # Compute FM correlation
        corr, t_stat, n = fama_macbeth_correlation(data, 'VAR1', 'VAR2')
        
        # Check results
        assert abs(corr - 0.7) < 0.1, f"Correlation {corr:.2f} should be near 0.7"
        assert abs(t_stat) > 2.0, f"t-statistic {t_stat:.2f} should be significant"
        assert n == n_months, f"Should use all {n_months} months"
        
        print(f"✓ Fama-MacBeth test passed: ρ={corr:.3f}, t={t_stat:.2f}")


class TestSummaryStatistics:
    """Tests for summary statistics (Table 4)."""
    
    def test_decile_sorting(self):
        """Test decile sorting produces monotonic patterns."""
        from scripts.replicate_table4 import generate_table4
        
        # Create synthetic data with monotonic relationship
        np.random.seed(42)
        n_stocks = 1000
        n_months = 12
        
        asy_data = []
        char_data = []
        
        for month in range(n_months):
            date = pd.Timestamp('2020-01-01') + pd.DateOffset(months=month)
            
            for stock in range(n_stocks):
                # DOWN_ASY
                asy = np.random.randn()
                
                # Characteristic correlated with DOWN_ASY
                char = 0.5 * asy + np.random.randn() * 0.5
                
                asy_data.append({
                    'PERMNO': stock,
                    'DATE': date,
                    'DOWN_ASY': asy
                })
                
                char_data.append({
                    'PERMNO': stock,
                    'DATE': date,
                    'BETA': 1.0 + char,
                    'IVOL': 0.02 + abs(char) * 0.01,
                    'DOWNSIDE_BETA': 1.1 + char,
                    'COSKEW': char,
                    'COKURT': char * 2,
                    'ILLIQ': abs(char) * 5
                })
        
        asy_df = pd.DataFrame(asy_data)
        char_df = pd.DataFrame(char_data)
        
        # Generate table
        table = generate_table4(asy_df, char_df, n_deciles=10, output_path=None)
        
        # Check DOWN_ASY is monotonic across deciles
        down_asy_vals = table.iloc[:-2]['DOWN_ASY']  # Exclude High-Low and t-stat
        diffs = down_asy_vals.diff().dropna()
        
        assert all(diffs > 0), "DOWN_ASY should increase monotonically across deciles"
        
        print(f"✓ Decile sorting test passed: monotonic pattern confirmed")


class TestTimeSeriesRegression:
    """Tests for time-series regressions (Table 6)."""
    
    def test_newey_west_se(self):
        """Test Newey-West standard errors."""
        from scripts.replicate_table6 import newey_west_standard_errors
        
        # Create synthetic regression data with autocorrelation
        np.random.seed(42)
        n = 100
        
        X = np.column_stack([
            np.ones(n),
            np.random.randn(n)
        ])
        
        # Errors with autocorrelation
        errors = np.zeros(n)
        errors[0] = np.random.randn()
        for t in range(1, n):
            errors[t] = 0.5 * errors[t-1] + np.random.randn()
        
        # Compute NW standard errors
        se_nw = newey_west_standard_errors(X, errors, n_lags=12)
        
        # Should be positive
        assert all(se_nw > 0), "Standard errors should be positive"
        assert len(se_nw) == X.shape[1], "Should have SE for each parameter"
        
        print(f"✓ Newey-West test passed: SE={se_nw}")
    
    def test_regression_r_squared(self):
        """Test regression R² computation."""
        from scripts.replicate_table6 import run_time_series_regression
        
        # Create data with known R²
        np.random.seed(42)
        n = 100
        
        x = np.random.randn(n)
        y = 2.0 + 1.5 * x + np.random.randn(n) * 0.5
        
        data_df = pd.DataFrame({
            'Y': y,
            'X': x
        })
        
        # Run regression
        result = run_time_series_regression(
            data_df['Y'],
            data_df[['X']],
            "Test Model"
        )
        
        # R² should be high (strong relationship)
        assert result['r_squared'] > 0.7, f"R²={result['r_squared']:.2f} should be > 0.7"
        
        print(f"✓ Regression test passed: R²={result['r_squared']:.3f}")


class TestIntegration:
    """Integration tests across Phase 5 components."""
    
    def test_end_to_end_workflow(self):
        """Test complete Phase 5 workflow."""
        # This would test:
        # 1. Load Phase 4 DOWN_ASY scores
        # 2. Compute characteristics
        # 3. Generate Tables 3, 4, 6
        # 4. Verify all outputs exist
        
        # Placeholder for now
        print("✓ End-to-end workflow test (placeholder)")
    
    def test_data_alignment(self):
        """Test proper alignment of DOWN_ASY and characteristics."""
        np.random.seed(42)
        
        # Create misaligned datasets
        dates1 = pd.date_range('2020-01-01', periods=10, freq='M')
        dates2 = pd.date_range('2020-02-01', periods=10, freq='M')
        
        asy_df = pd.DataFrame({
            'PERMNO': [1] * len(dates1),
            'DATE': dates1,
            'DOWN_ASY': np.random.randn(len(dates1))
        })
        
        char_df = pd.DataFrame({
            'PERMNO': [1] * len(dates2),
            'DATE': dates2,
            'BETA': np.random.randn(len(dates2))
        })
        
        # Merge
        merged = asy_df.merge(char_df, on=['PERMNO', 'DATE'], how='inner')
        
        # Should have overlap
        assert len(merged) > 0, "Should have overlapping dates"
        assert len(merged) < min(len(asy_df), len(char_df)), "Should drop non-overlapping"
        
        print(f"✓ Data alignment test passed: {len(merged)} overlapping observations")


def run_all_tests():
    """Run all Phase 5 tests."""
    print("="*80)
    print("PHASE 5 TEST SUITE")
    print("="*80)
    
    test_classes = [
        TestCharacteristics,
        TestFamaMacBeth,
        TestSummaryStatistics,
        TestTimeSeriesRegression,
        TestIntegration
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for test_class in test_classes:
        print(f"\n{test_class.__name__}:")
        print("-" * 80)
        
        test_obj = test_class()
        test_methods = [m for m in dir(test_obj) if m.startswith('test_')]
        
        for method_name in test_methods:
            total_tests += 1
            try:
                method = getattr(test_obj, method_name)
                method()
                passed_tests += 1
            except Exception as e:
                print(f"✗ {method_name} FAILED: {str(e)}")
    
    print("\n" + "="*80)
    print(f"TEST SUMMARY: {passed_tests}/{total_tests} tests passed")
    print("="*80)
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
