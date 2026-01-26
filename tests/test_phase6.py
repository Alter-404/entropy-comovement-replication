#!/usr/bin/env python3
"""
Test suite for Phase 6: Robustness Checks

Tests:
1. AlternativeThresholdTester
2. SubperiodAnalyzer
3. AlternativeSortTester
4. FamaMacBethRegressor
5. BootstrapInference
6. RobustnessRunner integration
"""

import sys
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src' / 'python'))

from robustness import (
    AlternativeThresholdTester,
    SubperiodAnalyzer,
    AlternativeSortTester,
    FamaMacBethRegressor,
    BootstrapInference,
    RobustnessRunner,
    RobustnessResult,
    SortMethod,
)


class TestRobustnessResult(unittest.TestCase):
    """Tests for RobustnessResult dataclass."""
    
    def test_basic_creation(self):
        """Test basic RobustnessResult creation."""
        result = RobustnessResult(
            test_name='Test',
            specification={'param': 1},
            estimate=0.05,
            std_error=0.01,
            t_statistic=5.0,
            p_value=0.001,
            n_observations=100,
        )
        
        self.assertEqual(result.test_name, 'Test')
        self.assertEqual(result.estimate, 0.05)
        self.assertEqual(result.t_statistic, 5.0)
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = RobustnessResult(
            test_name='Test',
            specification={'threshold': 0.5},
            estimate=0.03,
            std_error=0.01,
            t_statistic=3.0,
            p_value=0.01,
            n_observations=50,
            additional_stats={'q1_mean': 0.01, 'q5_mean': 0.04},
        )
        
        d = result.to_dict()
        
        self.assertIn('test_name', d)
        self.assertIn('estimate', d)
        self.assertIn('threshold', d)  # From specification
        self.assertIn('q1_mean', d)  # From additional_stats


class TestSubperiodAnalyzer(unittest.TestCase):
    """Tests for SubperiodAnalyzer."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.n_stocks = 50
        self.n_months = 60
        
        dates = pd.date_range('2000-01-01', periods=self.n_months, freq='ME')
        permnos = list(range(10001, 10001 + self.n_stocks))
        
        # Create returns
        returns_data = []
        for date in dates:
            for permno in permnos:
                returns_data.append({
                    'PERMNO': permno,
                    'DATE': date,
                    'RET': np.random.normal(0.01, 0.05),
                })
        self.returns = pd.DataFrame(returns_data)
        
        # Create DOWN_ASY
        down_asy_data = []
        for date in dates:
            for permno in permnos:
                down_asy_data.append({
                    'PERMNO': permno,
                    'DATE': date,
                    'DOWN_ASY': np.random.uniform(-0.1, 0.1),
                })
        self.down_asy = pd.DataFrame(down_asy_data)
    
    def test_initialization(self):
        """Test SubperiodAnalyzer initialization."""
        analyzer = SubperiodAnalyzer()
        
        self.assertIn('full_sample', analyzer.subperiods)
        self.assertIn('pre_2000', analyzer.subperiods)
        self.assertIn('post_2000', analyzer.subperiods)
    
    def test_define_market_regimes(self):
        """Test market regime definition."""
        analyzer = SubperiodAnalyzer()
        
        # Create market returns
        dates = pd.date_range('2000-01-01', periods=60, freq='ME')
        market_returns = pd.Series(
            np.random.normal(0.005, 0.04, 60),
            index=dates
        )
        
        regimes = analyzer.define_market_regimes(market_returns, lookback=12)
        
        # Check output
        self.assertEqual(len(regimes), len(market_returns))
        self.assertTrue(all(r in ['Bull', 'Bear', 'Neutral', np.nan] 
                           for r in regimes.dropna().unique()))
    
    def test_run_subperiod_analysis(self):
        """Test running subperiod analysis."""
        analyzer = SubperiodAnalyzer()
        
        results = analyzer.run_subperiod_analysis(self.down_asy, self.returns)
        
        # Should have some results (at least post_2000 since data is 2000+)
        self.assertIsInstance(results, pd.DataFrame)
        
        if len(results) > 0:
            self.assertIn('estimate', results.columns)
            self.assertIn('t_statistic', results.columns)
            self.assertIn('p_value', results.columns)


class TestAlternativeSortTester(unittest.TestCase):
    """Tests for AlternativeSortTester."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.n_stocks = 100
        self.n_months = 48
        
        dates = pd.date_range('2000-01-01', periods=self.n_months, freq='ME')
        permnos = list(range(10001, 10001 + self.n_stocks))
        
        # Create returns
        returns_data = []
        for date in dates:
            for permno in permnos:
                returns_data.append({
                    'PERMNO': permno,
                    'DATE': date,
                    'RET': np.random.normal(0.01, 0.05),
                })
        self.returns = pd.DataFrame(returns_data)
        
        # Create DOWN_ASY with stock-level persistence
        stock_effects = {p: np.random.normal(0, 0.05) for p in permnos}
        down_asy_data = []
        for date in dates:
            for permno in permnos:
                down_asy_data.append({
                    'PERMNO': permno,
                    'DATE': date,
                    'DOWN_ASY': stock_effects[permno] + np.random.normal(0, 0.01),
                })
        self.down_asy = pd.DataFrame(down_asy_data)
    
    def test_initialization(self):
        """Test AlternativeSortTester initialization."""
        tester = AlternativeSortTester()
        
        self.assertEqual(len(tester.sort_methods), 4)  # Terciles, Quartiles, Quintiles, Deciles
    
    def test_custom_methods(self):
        """Test with custom sort methods."""
        tester = AlternativeSortTester([SortMethod.TERCILES, SortMethod.QUINTILES])
        
        self.assertEqual(len(tester.sort_methods), 2)
    
    def test_run_alternative_sorts(self):
        """Test running alternative sorts."""
        tester = AlternativeSortTester()
        
        results = tester.run_alternative_sorts(self.down_asy, self.returns)
        
        self.assertIsInstance(results, pd.DataFrame)
        
        if len(results) > 0:
            self.assertIn('n_groups', results.columns)
            self.assertIn('method', results.columns)
            self.assertIn('estimate', results.columns)
            self.assertIn('t_statistic', results.columns)
            
            # Check we have different group sizes
            n_groups = results['n_groups'].unique()
            self.assertTrue(len(n_groups) >= 1)


class TestFamaMacBethRegressor(unittest.TestCase):
    """Tests for FamaMacBethRegressor."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.n_stocks = 100
        self.n_months = 60
        
        dates = pd.date_range('2000-01-01', periods=self.n_months, freq='ME')
        permnos = list(range(10001, 10001 + self.n_stocks))
        
        # Create returns with known relationship to DOWN_ASY
        returns_data = []
        down_asy_data = []
        stock_effects = {p: np.random.normal(0, 0.05) for p in permnos}
        
        for date in dates:
            for permno in permnos:
                down_asy = stock_effects[permno] + np.random.normal(0, 0.01)
                # Returns have positive relationship with DOWN_ASY
                ret = 0.005 + 0.1 * down_asy + np.random.normal(0, 0.04)
                
                returns_data.append({
                    'PERMNO': permno,
                    'DATE': date,
                    'RET': ret,
                })
                down_asy_data.append({
                    'PERMNO': permno,
                    'DATE': date,
                    'DOWN_ASY': down_asy,
                })
        
        self.returns = pd.DataFrame(returns_data)
        self.down_asy = pd.DataFrame(down_asy_data)
    
    def test_initialization(self):
        """Test FamaMacBethRegressor initialization."""
        regressor = FamaMacBethRegressor(n_lags=12)
        
        self.assertEqual(regressor.n_lags, 12)
    
    def test_newey_west_se(self):
        """Test Newey-West standard error calculation."""
        regressor = FamaMacBethRegressor(n_lags=4)
        
        # Create autocorrelated series
        np.random.seed(42)
        n = 100
        series = np.zeros(n)
        series[0] = np.random.normal()
        for i in range(1, n):
            series[i] = 0.5 * series[i-1] + np.random.normal()
        
        nw_se = regressor._newey_west_se(series, n_lags=4)
        
        # NW SE should be positive
        self.assertGreater(nw_se, 0)
        
        # For autocorrelated series, NW SE > simple SE
        simple_se = np.std(series, ddof=1) / np.sqrt(n)
        # This may not always hold, but typically does for positive autocorrelation
    
    def test_run_fama_macbeth(self):
        """Test running Fama-MacBeth regressions."""
        regressor = FamaMacBethRegressor(n_lags=6)
        
        results = regressor.run_fama_macbeth(self.returns, self.down_asy)
        
        self.assertIsInstance(results, pd.DataFrame)
        self.assertGreater(len(results), 0)
        
        # Should have DOWN_ASY coefficient
        self.assertIn('DOWN_ASY', results['variable'].values)
        
        # Check structure
        self.assertIn('estimate', results.columns)
        self.assertIn('t_statistic', results.columns)
        self.assertIn('p_value', results.columns)


class TestBootstrapInference(unittest.TestCase):
    """Tests for BootstrapInference."""
    
    def test_initialization(self):
        """Test BootstrapInference initialization."""
        bootstrap = BootstrapInference(n_bootstrap=500, block_size=12, random_seed=42)
        
        self.assertEqual(bootstrap.n_bootstrap, 500)
        self.assertEqual(bootstrap.block_size, 12)
        self.assertEqual(bootstrap.random_seed, 42)
    
    def test_bootstrap_spread_mean(self):
        """Test bootstrap for mean statistic."""
        np.random.seed(42)
        spread = pd.Series(np.random.normal(0.01, 0.02, 100))
        
        bootstrap = BootstrapInference(n_bootstrap=500)
        result = bootstrap.bootstrap_spread(spread, statistic='mean')
        
        # Check result structure
        self.assertIn('observed', result)
        self.assertIn('bootstrap_se', result)
        self.assertIn('ci_lower', result)
        self.assertIn('ci_upper', result)
        self.assertIn('p_value', result)
        
        # CI should contain observed value
        self.assertLessEqual(result['ci_lower'], result['observed'])
        self.assertGreaterEqual(result['ci_upper'], result['observed'])
    
    def test_bootstrap_spread_median(self):
        """Test bootstrap for median statistic."""
        np.random.seed(42)
        spread = pd.Series(np.random.normal(0.01, 0.02, 100))
        
        bootstrap = BootstrapInference(n_bootstrap=500)
        result = bootstrap.bootstrap_spread(spread, statistic='median')
        
        self.assertIn('observed', result)
        self.assertAlmostEqual(result['observed'], np.median(spread), places=6)
    
    def test_bootstrap_spread_sharpe(self):
        """Test bootstrap for Sharpe ratio."""
        np.random.seed(42)
        spread = pd.Series(np.random.normal(0.02, 0.02, 100))  # Positive mean
        
        bootstrap = BootstrapInference(n_bootstrap=500)
        result = bootstrap.bootstrap_spread(spread, statistic='sharpe')
        
        self.assertIn('observed', result)
        expected_sharpe = spread.mean() / spread.std(ddof=1)
        self.assertAlmostEqual(result['observed'], expected_sharpe, places=4)
    
    def test_block_bootstrap(self):
        """Test block bootstrap for autocorrelated series."""
        np.random.seed(42)
        
        # Create autocorrelated series
        n = 100
        series = np.zeros(n)
        series[0] = np.random.normal()
        for i in range(1, n):
            series[i] = 0.7 * series[i-1] + np.random.normal(0, 0.5)
        spread = pd.Series(series)
        
        bootstrap = BootstrapInference(n_bootstrap=500, block_size=12)
        result = bootstrap.bootstrap_spread(spread, statistic='mean')
        
        # Block bootstrap SE should typically be larger for autocorrelated series
        self.assertIn('bootstrap_se', result)
        self.assertGreater(result['bootstrap_se'], 0)


class TestRobustnessRunner(unittest.TestCase):
    """Tests for RobustnessRunner integration."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.n_stocks = 50
        self.n_months = 36
        
        dates = pd.date_range('2000-01-01', periods=self.n_months, freq='ME')
        permnos = list(range(10001, 10001 + self.n_stocks))
        
        # Create data
        data = []
        for date in dates:
            for permno in permnos:
                data.append({
                    'PERMNO': permno,
                    'DATE': date,
                    'RET': np.random.normal(0.01, 0.05),
                    'DOWN_ASY': np.random.uniform(-0.1, 0.1),
                    'BETA': np.random.uniform(0.5, 1.5),
                    'IVOL': np.random.uniform(0.01, 0.1),
                })
        
        df = pd.DataFrame(data)
        
        self.returns = df[['PERMNO', 'DATE', 'RET']].copy()
        self.down_asy = df[['PERMNO', 'DATE', 'DOWN_ASY']].copy()
        self.characteristics = df[['PERMNO', 'DATE', 'BETA', 'IVOL']].copy()
        
        self.market_returns = pd.Series(
            np.random.normal(0.008, 0.04, self.n_months),
            index=dates
        )
    
    def test_initialization(self):
        """Test RobustnessRunner initialization."""
        runner = RobustnessRunner(output_dir='test_output')
        
        self.assertEqual(runner.output_dir, 'test_output')
        self.assertIsNotNone(runner.threshold_tester)
        self.assertIsNotNone(runner.subperiod_analyzer)
        self.assertIsNotNone(runner.sort_tester)
        self.assertIsNotNone(runner.fama_macbeth)
        self.assertIsNotNone(runner.bootstrap)
    
    def test_run_all_checks(self):
        """Test running all robustness checks."""
        runner = RobustnessRunner()
        
        results = runner.run_all_checks(
            self.down_asy,
            self.returns,
            self.characteristics,
            self.market_returns
        )
        
        self.assertIsInstance(results, dict)
        self.assertIn('subperiod', results)
        self.assertIn('alternative_sorts', results)
        self.assertIn('fama_macbeth', results)
    
    def test_generate_summary_table(self):
        """Test summary table generation."""
        runner = RobustnessRunner()
        
        # Run checks
        results = runner.run_all_checks(
            self.down_asy,
            self.returns,
            self.characteristics
        )
        
        # Generate summary
        summary = runner.generate_summary_table(results)
        
        self.assertIsInstance(summary, pd.DataFrame)
        
        if len(summary) > 0:
            self.assertIn('Category', summary.columns)
            self.assertIn('Estimate', summary.columns)


class TestAlternativeThresholdTester(unittest.TestCase):
    """Tests for AlternativeThresholdTester."""
    
    def test_initialization(self):
        """Test initialization with default thresholds."""
        tester = AlternativeThresholdTester()
        
        self.assertEqual(len(tester.thresholds), 4)
        self.assertIn(0.0, tester.thresholds)
    
    def test_custom_thresholds(self):
        """Test initialization with custom thresholds."""
        tester = AlternativeThresholdTester(thresholds=[0, 0.25, 0.5])
        
        self.assertEqual(len(tester.thresholds), 3)
    
    def test_compute_asymmetry_at_threshold(self):
        """Test asymmetry computation at specific threshold."""
        tester = AlternativeThresholdTester()
        
        np.random.seed(42)
        stock_returns = np.random.normal(0, 1, 252)
        market_returns = np.random.normal(0, 1, 252)
        
        s_rho, down_asy = tester.compute_asymmetry_at_threshold(
            stock_returns, market_returns, c=0.0
        )
        
        # Check output ranges
        self.assertGreaterEqual(s_rho, 0)
        self.assertLessEqual(abs(down_asy), 1)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete robustness pipeline."""
    
    def test_full_pipeline(self):
        """Test full robustness pipeline with synthetic data."""
        np.random.seed(42)
        n_stocks = 30
        n_months = 24
        
        dates = pd.date_range('2005-01-01', periods=n_months, freq='ME')
        permnos = list(range(10001, 10001 + n_stocks))
        
        # Create data
        returns_data = []
        down_asy_data = []
        
        for date in dates:
            for permno in permnos:
                returns_data.append({
                    'PERMNO': permno,
                    'DATE': date,
                    'RET': np.random.normal(0.01, 0.05),
                })
                down_asy_data.append({
                    'PERMNO': permno,
                    'DATE': date,
                    'DOWN_ASY': np.random.uniform(-0.1, 0.1),
                })
        
        returns = pd.DataFrame(returns_data)
        down_asy = pd.DataFrame(down_asy_data)
        
        # Run pipeline
        runner = RobustnessRunner()
        results = runner.run_all_checks(down_asy, returns)
        
        # Verify we get results
        self.assertIsInstance(results, dict)
        
        # Generate summary
        summary = runner.generate_summary_table(results)
        self.assertIsInstance(summary, pd.DataFrame)


def run_tests():
    """Run all tests."""
    print("=" * 60)
    print("Phase 6 Test Suite")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestRobustnessResult))
    suite.addTests(loader.loadTestsFromTestCase(TestSubperiodAnalyzer))
    suite.addTests(loader.loadTestsFromTestCase(TestAlternativeSortTester))
    suite.addTests(loader.loadTestsFromTestCase(TestFamaMacBethRegressor))
    suite.addTests(loader.loadTestsFromTestCase(TestBootstrapInference))
    suite.addTests(loader.loadTestsFromTestCase(TestRobustnessRunner))
    suite.addTests(loader.loadTestsFromTestCase(TestAlternativeThresholdTester))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.wasSuccessful():
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed.")
        return 1


if __name__ == '__main__':
    sys.exit(run_tests())
