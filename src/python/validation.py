"""
Validation Module for Empirical Replication

This module provides validation checks to ensure output tables match
the paper's benchmarks before final generation.

Reference: Jiang, Wu, Zhou (2018) - "Asymmetry in Stock Comovements: An Entropy Approach"
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple
import warnings


class ValidationError(Exception):
    """Exception raised when validation fails."""
    pass


class ReplicationValidator:
    """
    Validator for empirical replication outputs.
    
    Implements strict checks against paper benchmarks to catch
    common errors in data processing and sorting.
    """
    
    # Benchmarks from the paper
    BENCHMARKS = {
        'SIZE_MIN': 4.0,      # Minimum expected mean SIZE (log millions)
        'SIZE_MAX': 6.0,      # Maximum expected mean SIZE
        'LQP_UQP_CORR_MIN': -0.15,  # LQP/UQP correlation should not be too negative
        'LQP_UQP_CORR_MAX': 0.05,   # Upper bound for LQP/UQP correlation
        'HIGH_LOW_MIN': 0.0,   # Premium should be positive
        'MKT_VOL_COEF_MAX': 0.0,  # MKT_VOL coefficient should be negative
        'DOWN_ASY_BETA_CORR_MAX': 0.10,  # DOWN_ASY must be orthogonal to BETA
    }
    
    def __init__(self, strict: bool = True):
        """
        Initialize validator.
        
        Parameters
        ----------
        strict : bool
            If True, raise ValidationError on failures.
            If False, only warn.
        """
        self.strict = strict
        self.validation_results = {}
    
    def _report(self, check_name: str, passed: bool, message: str):
        """Report validation result."""
        self.validation_results[check_name] = {
            'passed': passed,
            'message': message
        }
        
        if passed:
            print(f"  ✓ {check_name}: PASSED - {message}")
        else:
            print(f"  ✗ {check_name}: FAILED - {message}")
            if self.strict:
                raise ValidationError(f"{check_name}: {message}")
            else:
                warnings.warn(f"Validation failed: {check_name} - {message}")
    
    def validate_size_variable(self, data: pd.DataFrame, 
                                size_col: str = 'SIZE') -> bool:
        """
        Validate SIZE variable is correctly scaled.
        
        Paper Definition: SIZE = log(Market Cap in Millions)
        Expected mean: ~4.79 for Decile 1, ~6.5 for Decile 10
        Overall mean should be around 5.0
        
        Parameters
        ----------
        data : pd.DataFrame
            Data with SIZE column
        size_col : str
            Name of SIZE column
        
        Returns
        -------
        bool
            True if validation passes
        """
        if size_col not in data.columns:
            self._report('SIZE_EXISTS', False, f"Column '{size_col}' not found")
            return False
        
        avg_size = data[size_col].mean()
        
        passed = self.BENCHMARKS['SIZE_MIN'] < avg_size < self.BENCHMARKS['SIZE_MAX']
        
        self._report(
            'SIZE_MAGNITUDE',
            passed,
            f"Mean SIZE = {avg_size:.2f} (expected: {self.BENCHMARKS['SIZE_MIN']:.1f} - {self.BENCHMARKS['SIZE_MAX']:.1f})"
        )
        
        return passed
    
    def validate_lqp_uqp_correlation(self, data: pd.DataFrame,
                                       lqp_col: str = 'LQP',
                                       uqp_col: str = 'UQP') -> bool:
        """
        Validate LQP/UQP correlation matches paper.
        
        Paper: Correlation ≈ -0.08 (slight negative)
        
        Parameters
        ----------
        data : pd.DataFrame
            Data with LQP and UQP columns
        
        Returns
        -------
        bool
            True if validation passes
        """
        if lqp_col not in data.columns or uqp_col not in data.columns:
            self._report('LQP_UQP_EXISTS', False, 
                        f"Columns '{lqp_col}' or '{uqp_col}' not found")
            return False
        
        corr = data[[lqp_col, uqp_col]].corr().iloc[0, 1]
        
        passed = self.BENCHMARKS['LQP_UQP_CORR_MIN'] < corr < self.BENCHMARKS['LQP_UQP_CORR_MAX']
        
        self._report(
            'LQP_UQP_CORRELATION',
            passed,
            f"Corr(LQP, UQP) = {corr:.3f} (expected: {self.BENCHMARKS['LQP_UQP_CORR_MIN']:.2f} to {self.BENCHMARKS['LQP_UQP_CORR_MAX']:.2f})"
        )
        
        return passed
    
    def validate_down_asy_orthogonality(self, corr_matrix: pd.DataFrame) -> bool:
        """
        Validate DOWN_ASY is orthogonal to systematic risk factors.
        
        Paper shows DOWN_ASY correlations:
        - BETA: -0.029 (essentially zero)
        - SIZE: -0.079 (small negative)
        - BM: +0.066 (small positive)
        
        The key constraint is that |Corr(DOWN_ASY, BETA)| < 0.10
        
        Parameters
        ----------
        corr_matrix : pd.DataFrame
            Correlation matrix from Table 3
        
        Returns
        -------
        bool
            True if DOWN_ASY is properly orthogonal to BETA
        """
        if 'DOWN_ASY' not in corr_matrix.columns or 'BETA' not in corr_matrix.columns:
            self._report('DOWN_ASY_ORTHOGONALITY', False, 
                        "Required columns not found in correlation matrix")
            return False
        
        beta_corr = corr_matrix.loc['DOWN_ASY', 'BETA']
        
        passed = abs(beta_corr) < self.BENCHMARKS['DOWN_ASY_BETA_CORR_MAX']
        
        self._report(
            'DOWN_ASY_ORTHOGONALITY',
            passed,
            f"|Corr(DOWN_ASY, BETA)| = {abs(beta_corr):.3f} (expected < {self.BENCHMARKS['DOWN_ASY_BETA_CORR_MAX']:.2f})"
        )
        
        if not passed:
            self._report(
                'DOWN_ASY_CONSTRUCTION',
                False,
                f"DOWN_ASY appears to be a linear risk factor, not an entropy measure. "
                f"Check that it's computed using Eq. 16: sign(LQP-UQP) * S_rho"
            )
        
        return passed
    
    def validate_asymmetry_premium(self, premium_series: pd.Series) -> bool:
        """
        Validate asymmetry premium is positive.
        
        Paper: High-Low spread ≈ +0.38% per month
        
        Parameters
        ----------
        premium_series : pd.Series
            Monthly High-Low return spread
        
        Returns
        -------
        bool
            True if validation passes
        """
        mean_premium = premium_series.mean()
        t_stat = mean_premium / (premium_series.std() / np.sqrt(len(premium_series)))
        
        passed = mean_premium > self.BENCHMARKS['HIGH_LOW_MIN']
        
        self._report(
            'ASYMMETRY_PREMIUM_POSITIVE',
            passed,
            f"Mean Premium = {mean_premium*100:.3f}% (t={t_stat:.2f}), expected > 0"
        )
        
        # Additional check for significance
        significant = abs(t_stat) > 1.96
        self._report(
            'ASYMMETRY_PREMIUM_SIGNIFICANT',
            significant,
            f"t-statistic = {t_stat:.2f}, expected |t| > 2.0 for significance"
        )
        
        return passed and significant
    
    def validate_mkt_vol_coefficient(self, coef: float, t_stat: float = None) -> bool:
        """
        Validate MKT_VOL regression coefficient is negative.
        
        Paper: MKT_VOL coefficient ≈ -0.124 (negative relationship)
        
        Parameters
        ----------
        coef : float
            Regression coefficient on MKT_VOL
        t_stat : float, optional
            t-statistic for the coefficient
        
        Returns
        -------
        bool
            True if validation passes
        """
        passed = coef < self.BENCHMARKS['MKT_VOL_COEF_MAX']
        
        msg = f"MKT_VOL coefficient = {coef:.4f}"
        if t_stat is not None:
            msg += f" (t={t_stat:.2f})"
        msg += ", expected < 0"
        
        self._report('MKT_VOL_NEGATIVE', passed, msg)
        
        return passed
    
    def validate_down_asy_sign(self, 
                                 lqp: float, 
                                 uqp: float, 
                                 down_asy: float) -> bool:
        """
        Validate DOWN_ASY sign is correct.
        
        Paper Definition: DOWN_ASY = sign(LQP - UQP) * S_rho
        
        If LQP > UQP (more crash probability), DOWN_ASY should be positive.
        If LQP < UQP (more boom probability), DOWN_ASY should be negative.
        
        Parameters
        ----------
        lqp : float
            Lower quadrant probability
        uqp : float
            Upper quadrant probability
        down_asy : float
            Computed DOWN_ASY value
        
        Returns
        -------
        bool
            True if sign is correct
        """
        expected_sign = 1 if lqp >= uqp else -1
        actual_sign = 1 if down_asy >= 0 else -1
        
        passed = expected_sign == actual_sign
        
        self._report(
            'DOWN_ASY_SIGN',
            passed,
            f"LQP={lqp:.4f}, UQP={uqp:.4f}, DOWN_ASY={down_asy:.4f}, "
            f"Expected sign: {expected_sign}, Actual: {actual_sign}"
        )
        
        return passed
    
    def validate_sorting_direction(self, 
                                     decile_data: pd.DataFrame,
                                     down_asy_col: str = 'DOWN_ASY',
                                     decile_col: str = 'DECILE') -> bool:
        """
        Validate sorting direction: Decile 10 should have highest DOWN_ASY.
        
        Parameters
        ----------
        decile_data : pd.DataFrame
            Data with DOWN_ASY and DECILE columns
        
        Returns
        -------
        bool
            True if sorting is correct
        """
        decile_means = decile_data.groupby(decile_col)[down_asy_col].mean()
        
        # Check if monotonically increasing
        is_increasing = all(
            decile_means.iloc[i] <= decile_means.iloc[i+1] 
            for i in range(len(decile_means)-1)
        )
        
        self._report(
            'SORTING_MONOTONIC',
            is_increasing,
            f"DOWN_ASY: D1={decile_means.iloc[0]:.4f}, D10={decile_means.iloc[-1]:.4f}, "
            f"should be increasing"
        )
        
        return is_increasing
    
    def run_full_validation(self, 
                             table3_data: Optional[pd.DataFrame] = None,
                             table4_data: Optional[pd.DataFrame] = None,
                             table5_premium: Optional[pd.Series] = None,
                             table6_mkt_vol_coef: Optional[float] = None) -> Dict:
        """
        Run complete validation suite.
        
        Parameters
        ----------
        table3_data : pd.DataFrame, optional
            Data for Table 3 validation
        table4_data : pd.DataFrame, optional
            Data for Table 4 validation
        table5_premium : pd.Series, optional
            Premium series for Table 5 validation
        table6_mkt_vol_coef : float, optional
            MKT_VOL coefficient for Table 6 validation
        
        Returns
        -------
        dict
            Summary of all validation results
        """
        print("="*60)
        print("RUNNING REPLICATION VALIDATION SUITE")
        print("="*60)
        
        all_passed = True
        
        if table3_data is not None:
            print("\n[Table 3 Validation]")
            if 'LQP' in table3_data.columns and 'UQP' in table3_data.columns:
                all_passed &= self.validate_lqp_uqp_correlation(table3_data)
        
        if table4_data is not None:
            print("\n[Table 4 Validation]")
            all_passed &= self.validate_size_variable(table4_data)
            if 'DOWN_ASY' in table4_data.columns and 'DECILE' in table4_data.columns:
                all_passed &= self.validate_sorting_direction(table4_data)
        
        if table5_premium is not None:
            print("\n[Table 5 Validation]")
            all_passed &= self.validate_asymmetry_premium(table5_premium)
        
        if table6_mkt_vol_coef is not None:
            print("\n[Table 6 Validation]")
            all_passed &= self.validate_mkt_vol_coefficient(table6_mkt_vol_coef)
        
        print("\n" + "="*60)
        if all_passed:
            print("✓ ALL VALIDATIONS PASSED")
        else:
            print("✗ SOME VALIDATIONS FAILED")
        print("="*60)
        
        return {
            'all_passed': all_passed,
            'results': self.validation_results
        }


def validate_empirical_results(df_table3: Optional[pd.DataFrame] = None,
                                 df_table4: Optional[pd.DataFrame] = None,
                                 df_table5: Optional[pd.DataFrame] = None,
                                 strict: bool = True) -> bool:
    """
    Convenience function for quick validation.
    
    This is the main entry point for validation before generating final tables.
    
    Parameters
    ----------
    df_table3 : pd.DataFrame, optional
        Cross-sectional correlation data
    df_table4 : pd.DataFrame, optional
        Summary statistics data with SIZE column
    df_table5 : pd.DataFrame, optional
        Returns data with 'HIGH_LOW' column
    strict : bool
        If True, raise error on failure
    
    Returns
    -------
    bool
        True if all validations pass
    
    Raises
    ------
    ValidationError
        If strict=True and validation fails
    """
    validator = ReplicationValidator(strict=strict)
    
    # Extract premium from Table 5 if available
    premium_series = None
    if df_table5 is not None and 'HIGH_LOW' in df_table5.columns:
        premium_series = df_table5['HIGH_LOW']
    
    results = validator.run_full_validation(
        table3_data=df_table3,
        table4_data=df_table4,
        table5_premium=premium_series
    )
    
    return results['all_passed']


# Unit tests for the validator
if __name__ == "__main__":
    print("Testing Validation Module\n")
    
    # Test 1: SIZE validation
    print("Test 1: SIZE variable validation")
    validator = ReplicationValidator(strict=False)
    
    # Correct SIZE (log millions, mean ~5.0)
    correct_size = pd.DataFrame({'SIZE': np.random.normal(5.0, 1.0, 1000)})
    validator.validate_size_variable(correct_size)
    
    # Wrong SIZE (log dollars, mean ~14.0)
    wrong_size = pd.DataFrame({'SIZE': np.random.normal(14.0, 2.0, 1000)})
    validator.validate_size_variable(wrong_size)
    
    # Test 2: LQP/UQP correlation
    print("\nTest 2: LQP/UQP correlation validation")
    
    # Correct correlation (~-0.08)
    n = 1000
    lqp = np.random.normal(0.15, 0.05, n)
    uqp = 0.15 + 0.02 * np.random.randn(n)  # Nearly independent
    correct_corr = pd.DataFrame({'LQP': lqp, 'UQP': uqp})
    validator.validate_lqp_uqp_correlation(correct_corr)
    
    # Wrong correlation (~-0.5, too negative)
    uqp_wrong = 0.15 - 0.5 * (lqp - 0.15) + 0.02 * np.random.randn(n)
    wrong_corr = pd.DataFrame({'LQP': lqp, 'UQP': uqp_wrong})
    validator.validate_lqp_uqp_correlation(wrong_corr)
    
    # Test 3: Premium validation
    print("\nTest 3: Asymmetry premium validation")
    
    # Correct premium (positive)
    correct_premium = pd.Series(np.random.normal(0.004, 0.02, 120))
    validator.validate_asymmetry_premium(correct_premium)
    
    # Wrong premium (negative)
    wrong_premium = pd.Series(np.random.normal(-0.002, 0.02, 120))
    validator.validate_asymmetry_premium(wrong_premium)
    
    print("\n✓ Validation module tests complete!")
