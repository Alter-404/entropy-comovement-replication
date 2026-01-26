"""
Robustness checks and sensitivity analysis for the entropy-comovement replication.

Implements:
1. Alternative threshold testing (different c values for asymmetry)
2. Subperiod analysis (pre/post 2000, bull/bear markets)
3. Alternative portfolio sorts (terciles, quartiles, deciles)
4. Fama-MacBeth cross-sectional regressions
5. Bootstrap inference for statistical robustness
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum
import warnings
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp


class SortMethod(Enum):
    """Portfolio sorting methods."""
    TERCILES = 3
    QUARTILES = 4
    QUINTILES = 5
    DECILES = 10


@dataclass
class RobustnessResult:
    """Container for robustness test results."""
    test_name: str
    specification: Dict[str, Any]
    estimate: float
    std_error: float
    t_statistic: float
    p_value: float
    n_observations: int
    additional_stats: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DataFrame creation."""
        result = {
            'test_name': self.test_name,
            'estimate': self.estimate,
            'std_error': self.std_error,
            't_statistic': self.t_statistic,
            'p_value': self.p_value,
            'n_observations': self.n_observations,
        }
        result.update(self.specification)
        if self.additional_stats:
            result.update(self.additional_stats)
        return result


class AlternativeThresholdTester:
    """
    Test asymmetry measure with alternative threshold values.
    
    The paper uses c=0 as the threshold for defining "extreme" comovements.
    This class tests sensitivity to different threshold choices.
    """
    
    def __init__(self, thresholds: Optional[List[float]] = None):
        """
        Initialize threshold tester.
        
        Parameters
        ----------
        thresholds : List[float], optional
            Threshold values to test. Default: [-0.5, 0, 0.5, 1.0]
        """
        self.thresholds = thresholds or [-0.5, 0.0, 0.5, 1.0]
    
    def compute_asymmetry_at_threshold(
        self,
        stock_returns: np.ndarray,
        market_returns: np.ndarray,
        c: float
    ) -> Tuple[float, float]:
        """
        Compute asymmetry measure at a specific threshold.
        
        This is a simplified version - in production, this would call the C++ engine.
        
        Parameters
        ----------
        stock_returns : np.ndarray
            Standardized stock returns.
        market_returns : np.ndarray
            Standardized market returns.
        c : float
            Threshold value.
            
        Returns
        -------
        Tuple[float, float]
            (s_rho, down_asy) - entropy measure and directional asymmetry.
        """
        # Standardize returns
        x = (stock_returns - np.mean(stock_returns)) / np.std(stock_returns, ddof=1)
        y = (market_returns - np.mean(market_returns)) / np.std(market_returns, ddof=1)
        
        # Count observations in each quadrant relative to threshold
        lower_quadrant = np.sum((x < -c) & (y < -c)) / len(x)
        upper_quadrant = np.sum((x > c) & (y > c)) / len(x)
        
        # Simple asymmetry measure (simplified from full entropy calculation)
        s_rho = np.abs(lower_quadrant - upper_quadrant)
        down_asy = (1 if lower_quadrant >= upper_quadrant else -1) * s_rho
        
        return s_rho, down_asy
    
    def run_threshold_sensitivity(
        self,
        down_asy_by_threshold: Dict[float, pd.DataFrame],
        returns: pd.DataFrame,
        characteristics: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Run portfolio sorts for each threshold and compare results.
        
        Parameters
        ----------
        down_asy_by_threshold : Dict[float, pd.DataFrame]
            DOWN_ASY scores computed at each threshold.
            Keys are threshold values, values are DataFrames with PERMNO, DATE, DOWN_ASY.
        returns : pd.DataFrame
            Monthly returns with PERMNO, DATE, RET columns.
        characteristics : pd.DataFrame, optional
            Firm characteristics for controls.
            
        Returns
        -------
        pd.DataFrame
            Comparison of High-Low spreads across thresholds.
        """
        results = []
        
        for threshold, down_asy in down_asy_by_threshold.items():
            # Merge with returns (lagged to avoid look-ahead bias)
            merged = returns.copy()
            merged['MONTH'] = pd.to_datetime(merged['DATE']).dt.to_period('M')
            
            down_asy_copy = down_asy.copy()
            down_asy_copy['MONTH'] = pd.to_datetime(down_asy_copy['DATE']).dt.to_period('M')
            down_asy_copy['MONTH'] = down_asy_copy['MONTH'] + 1  # Lag signal
            
            merged = merged.merge(
                down_asy_copy[['PERMNO', 'MONTH', 'DOWN_ASY']],
                on=['PERMNO', 'MONTH'],
                how='inner'
            )
            
            # Sort into quintiles
            merged['QUINTILE'] = merged.groupby('MONTH')['DOWN_ASY'].transform(
                lambda x: pd.qcut(x, 5, labels=False, duplicates='drop') + 1
            )
            
            # Compute quintile returns
            quintile_returns = merged.groupby(['MONTH', 'QUINTILE'])['RET'].mean().unstack()
            
            if 1 in quintile_returns.columns and 5 in quintile_returns.columns:
                spread = quintile_returns[5] - quintile_returns[1]
                
                mean_spread = spread.mean()
                std_spread = spread.std(ddof=1)
                t_stat = mean_spread / (std_spread / np.sqrt(len(spread)))
                p_val = 2 * (1 - self._normal_cdf(abs(t_stat)))
                
                results.append(RobustnessResult(
                    test_name='Threshold Sensitivity',
                    specification={'threshold': threshold},
                    estimate=mean_spread,
                    std_error=std_spread / np.sqrt(len(spread)),
                    t_statistic=t_stat,
                    p_value=p_val,
                    n_observations=len(spread),
                    additional_stats={
                        'q1_mean': quintile_returns[1].mean() if 1 in quintile_returns.columns else np.nan,
                        'q5_mean': quintile_returns[5].mean() if 5 in quintile_returns.columns else np.nan,
                    }
                ))
        
        return pd.DataFrame([r.to_dict() for r in results])
    
    @staticmethod
    def _normal_cdf(x: float) -> float:
        """Standard normal CDF approximation."""
        from scipy.stats import norm
        return norm.cdf(x)


class SubperiodAnalyzer:
    """
    Analyze asymmetry premium across different time periods and market conditions.
    """
    
    def __init__(self):
        """Initialize subperiod analyzer."""
        self.subperiods = {
            'full_sample': (None, None),
            'pre_2000': (None, '1999-12-31'),
            'post_2000': ('2000-01-01', None),
            'pre_crisis': (None, '2007-06-30'),
            'crisis': ('2007-07-01', '2009-03-31'),
            'post_crisis': ('2009-04-01', None),
        }
    
    def define_market_regimes(
        self,
        market_returns: pd.Series,
        lookback: int = 12
    ) -> pd.Series:
        """
        Define bull/bear market regimes based on trailing returns.
        
        Parameters
        ----------
        market_returns : pd.Series
            Monthly market returns indexed by date.
        lookback : int
            Number of months for trailing return calculation.
            
        Returns
        -------
        pd.Series
            Market regime labels ('Bull', 'Bear', 'Neutral').
        """
        # Calculate trailing 12-month returns
        trailing_return = market_returns.rolling(lookback).sum()
        
        # Define regimes based on percentiles
        regime = pd.Series(index=market_returns.index, dtype=str)
        q33 = trailing_return.quantile(0.33)
        q67 = trailing_return.quantile(0.67)
        
        regime[trailing_return <= q33] = 'Bear'
        regime[trailing_return >= q67] = 'Bull'
        regime[(trailing_return > q33) & (trailing_return < q67)] = 'Neutral'
        
        return regime
    
    def run_subperiod_analysis(
        self,
        down_asy: pd.DataFrame,
        returns: pd.DataFrame,
        market_returns: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Run portfolio analysis for each subperiod.
        
        Parameters
        ----------
        down_asy : pd.DataFrame
            DOWN_ASY scores with PERMNO, DATE, DOWN_ASY columns.
        returns : pd.DataFrame
            Monthly returns with PERMNO, DATE, RET columns.
        market_returns : pd.Series, optional
            Monthly market returns for regime definition.
            
        Returns
        -------
        pd.DataFrame
            Results for each subperiod.
        """
        results = []
        
        for period_name, (start_date, end_date) in self.subperiods.items():
            # Filter data for subperiod
            down_asy_sub = down_asy.copy()
            returns_sub = returns.copy()
            
            if start_date:
                down_asy_sub = down_asy_sub[pd.to_datetime(down_asy_sub['DATE']) >= start_date]
                returns_sub = returns_sub[pd.to_datetime(returns_sub['DATE']) >= start_date]
            
            if end_date:
                down_asy_sub = down_asy_sub[pd.to_datetime(down_asy_sub['DATE']) <= end_date]
                returns_sub = returns_sub[pd.to_datetime(returns_sub['DATE']) <= end_date]
            
            if len(down_asy_sub) < 100:
                continue
            
            # Compute spread
            spread_result = self._compute_spread(down_asy_sub, returns_sub)
            
            if spread_result is not None:
                results.append(RobustnessResult(
                    test_name='Subperiod Analysis',
                    specification={'period': period_name, 'start': start_date, 'end': end_date},
                    estimate=spread_result['mean'],
                    std_error=spread_result['se'],
                    t_statistic=spread_result['t_stat'],
                    p_value=spread_result['p_value'],
                    n_observations=spread_result['n_obs'],
                ))
        
        # Add market regime analysis if market returns provided
        if market_returns is not None:
            regime = self.define_market_regimes(market_returns)
            
            for regime_name in ['Bull', 'Bear', 'Neutral']:
                regime_dates = regime[regime == regime_name].index
                
                down_asy_sub = down_asy[pd.to_datetime(down_asy['DATE']).isin(regime_dates)]
                returns_sub = returns[pd.to_datetime(returns['DATE']).isin(regime_dates)]
                
                if len(down_asy_sub) < 50:
                    continue
                
                spread_result = self._compute_spread(down_asy_sub, returns_sub)
                
                if spread_result is not None:
                    results.append(RobustnessResult(
                        test_name='Market Regime',
                        specification={'regime': regime_name},
                        estimate=spread_result['mean'],
                        std_error=spread_result['se'],
                        t_statistic=spread_result['t_stat'],
                        p_value=spread_result['p_value'],
                        n_observations=spread_result['n_obs'],
                    ))
        
        return pd.DataFrame([r.to_dict() for r in results])
    
    def _compute_spread(
        self,
        down_asy: pd.DataFrame,
        returns: pd.DataFrame
    ) -> Optional[Dict[str, float]]:
        """Compute High-Low spread for a subset of data."""
        from scipy.stats import norm
        
        # Merge with lag
        merged = returns.copy()
        merged['MONTH'] = pd.to_datetime(merged['DATE']).dt.to_period('M')
        
        down_asy_copy = down_asy.copy()
        down_asy_copy['MONTH'] = pd.to_datetime(down_asy_copy['DATE']).dt.to_period('M')
        down_asy_copy['MONTH'] = down_asy_copy['MONTH'] + 1
        
        merged = merged.merge(
            down_asy_copy[['PERMNO', 'MONTH', 'DOWN_ASY']],
            on=['PERMNO', 'MONTH'],
            how='inner'
        )
        
        if len(merged) < 50:
            return None
        
        # Sort into quintiles
        try:
            merged['QUINTILE'] = merged.groupby('MONTH')['DOWN_ASY'].transform(
                lambda x: pd.qcut(x, 5, labels=False, duplicates='drop') + 1
                if len(x) >= 5 else np.nan
            )
        except Exception:
            return None
        
        merged = merged.dropna(subset=['QUINTILE'])
        
        # Compute quintile returns
        quintile_returns = merged.groupby(['MONTH', 'QUINTILE'])['RET'].mean().unstack()
        
        if 1 not in quintile_returns.columns or 5 not in quintile_returns.columns:
            return None
        
        spread = quintile_returns[5] - quintile_returns[1]
        spread = spread.dropna()
        
        if len(spread) < 12:
            return None
        
        mean_spread = spread.mean()
        std_spread = spread.std(ddof=1)
        se = std_spread / np.sqrt(len(spread))
        t_stat = mean_spread / se
        p_val = 2 * (1 - norm.cdf(abs(t_stat)))
        
        return {
            'mean': mean_spread,
            'se': se,
            't_stat': t_stat,
            'p_value': p_val,
            'n_obs': len(spread),
        }


class AlternativeSortTester:
    """
    Test portfolio sorts with alternative groupings (terciles, quartiles, deciles).
    """
    
    def __init__(self, sort_methods: Optional[List[SortMethod]] = None):
        """
        Initialize alternative sort tester.
        
        Parameters
        ----------
        sort_methods : List[SortMethod], optional
            Sort methods to test. Default: all methods.
        """
        self.sort_methods = sort_methods or list(SortMethod)
    
    def run_alternative_sorts(
        self,
        down_asy: pd.DataFrame,
        returns: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Run portfolio sorts with different grouping methods.
        
        Parameters
        ----------
        down_asy : pd.DataFrame
            DOWN_ASY scores with PERMNO, DATE, DOWN_ASY columns.
        returns : pd.DataFrame
            Monthly returns with PERMNO, DATE, RET columns.
            
        Returns
        -------
        pd.DataFrame
            Results for each sorting method.
        """
        from scipy.stats import norm
        results = []
        
        # Merge data with lag
        merged = returns.copy()
        merged['MONTH'] = pd.to_datetime(merged['DATE']).dt.to_period('M')
        
        down_asy_copy = down_asy.copy()
        down_asy_copy['MONTH'] = pd.to_datetime(down_asy_copy['DATE']).dt.to_period('M')
        down_asy_copy['MONTH'] = down_asy_copy['MONTH'] + 1
        
        merged = merged.merge(
            down_asy_copy[['PERMNO', 'MONTH', 'DOWN_ASY']],
            on=['PERMNO', 'MONTH'],
            how='inner'
        )
        
        for sort_method in self.sort_methods:
            n_groups = sort_method.value
            
            try:
                merged[f'GROUP_{n_groups}'] = merged.groupby('MONTH')['DOWN_ASY'].transform(
                    lambda x: pd.qcut(x, n_groups, labels=False, duplicates='drop') + 1
                    if len(x) >= n_groups else np.nan
                )
            except Exception:
                continue
            
            group_col = f'GROUP_{n_groups}'
            merged_valid = merged.dropna(subset=[group_col])
            
            # Compute group returns
            group_returns = merged_valid.groupby(['MONTH', group_col])['RET'].mean().unstack()
            
            low_group = 1
            high_group = n_groups
            
            if low_group not in group_returns.columns or high_group not in group_returns.columns:
                continue
            
            spread = group_returns[high_group] - group_returns[low_group]
            spread = spread.dropna()
            
            if len(spread) < 12:
                continue
            
            mean_spread = spread.mean()
            std_spread = spread.std(ddof=1)
            se = std_spread / np.sqrt(len(spread))
            t_stat = mean_spread / se
            p_val = 2 * (1 - norm.cdf(abs(t_stat)))
            
            results.append(RobustnessResult(
                test_name='Alternative Sort',
                specification={'n_groups': n_groups, 'method': sort_method.name},
                estimate=mean_spread,
                std_error=se,
                t_statistic=t_stat,
                p_value=p_val,
                n_observations=len(spread),
                additional_stats={
                    'low_group_mean': group_returns[low_group].mean(),
                    'high_group_mean': group_returns[high_group].mean(),
                }
            ))
        
        return pd.DataFrame([r.to_dict() for r in results])


class FamaMacBethRegressor:
    """
    Run Fama-MacBeth cross-sectional regressions.
    
    The standard approach:
    1. Each month t, run cross-sectional regression: r_{i,t+1} = γ₀ + γ₁·DOWN_ASY_{i,t} + controls + ε
    2. Time-series average of γ coefficients with Newey-West standard errors
    """
    
    def __init__(self, n_lags: int = 12):
        """
        Initialize Fama-MacBeth regressor.
        
        Parameters
        ----------
        n_lags : int
            Number of lags for Newey-West standard errors.
        """
        self.n_lags = n_lags
    
    def run_fama_macbeth(
        self,
        returns: pd.DataFrame,
        down_asy: pd.DataFrame,
        characteristics: Optional[pd.DataFrame] = None,
        control_vars: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Run Fama-MacBeth regressions.
        
        Parameters
        ----------
        returns : pd.DataFrame
            Monthly returns with PERMNO, DATE, RET columns.
        down_asy : pd.DataFrame
            DOWN_ASY scores with PERMNO, DATE, DOWN_ASY columns.
        characteristics : pd.DataFrame, optional
            Firm characteristics for controls.
        control_vars : List[str], optional
            List of control variable names to include.
            
        Returns
        -------
        pd.DataFrame
            Regression results with coefficients and t-statistics.
        """
        from scipy.stats import norm
        
        # Prepare data
        merged = returns.copy()
        merged['MONTH'] = pd.to_datetime(merged['DATE']).dt.to_period('M')
        
        down_asy_copy = down_asy.copy()
        down_asy_copy['MONTH'] = pd.to_datetime(down_asy_copy['DATE']).dt.to_period('M')
        down_asy_copy['MONTH'] = down_asy_copy['MONTH'] + 1  # Lag signal
        
        merged = merged.merge(
            down_asy_copy[['PERMNO', 'MONTH', 'DOWN_ASY']],
            on=['PERMNO', 'MONTH'],
            how='inner'
        )
        
        # Add characteristics if provided
        if characteristics is not None and control_vars:
            char_copy = characteristics.copy()
            char_copy['MONTH'] = pd.to_datetime(char_copy['DATE']).dt.to_period('M')
            char_copy['MONTH'] = char_copy['MONTH'] + 1  # Lag
            
            cols_to_merge = ['PERMNO', 'MONTH'] + control_vars
            cols_to_merge = [c for c in cols_to_merge if c in char_copy.columns]
            
            merged = merged.merge(
                char_copy[cols_to_merge],
                on=['PERMNO', 'MONTH'],
                how='inner'
            )
        
        # Define regression specifications
        specs = [
            ('DOWN_ASY Only', ['DOWN_ASY']),
        ]
        
        if control_vars:
            specs.append(('With Controls', ['DOWN_ASY'] + control_vars))
        
        results = []
        
        for spec_name, regressors in specs:
            valid_regressors = [r for r in regressors if r in merged.columns]
            
            if not valid_regressors:
                continue
            
            # Run cross-sectional regressions each month
            months = merged['MONTH'].unique()
            coefficients = {reg: [] for reg in valid_regressors}
            
            for month in months:
                month_data = merged[merged['MONTH'] == month].copy()
                
                if len(month_data) < len(valid_regressors) + 10:
                    continue
                
                # Drop missing values
                month_data = month_data.dropna(subset=['RET'] + valid_regressors)
                
                if len(month_data) < len(valid_regressors) + 10:
                    continue
                
                # Run OLS
                y = month_data['RET'].values
                X = month_data[valid_regressors].values
                X = np.column_stack([np.ones(len(y)), X])  # Add intercept
                
                try:
                    beta = np.linalg.lstsq(X, y, rcond=None)[0]
                    
                    for i, reg in enumerate(valid_regressors):
                        coefficients[reg].append(beta[i + 1])  # Skip intercept
                except Exception:
                    continue
            
            # Compute time-series averages with Newey-West SE
            for reg in valid_regressors:
                coef_series = np.array(coefficients[reg])
                
                if len(coef_series) < 24:
                    continue
                
                mean_coef = np.mean(coef_series)
                nw_se = self._newey_west_se(coef_series, self.n_lags)
                t_stat = mean_coef / nw_se if nw_se > 0 else np.nan
                p_val = 2 * (1 - norm.cdf(abs(t_stat))) if not np.isnan(t_stat) else np.nan
                
                results.append(RobustnessResult(
                    test_name='Fama-MacBeth',
                    specification={'model': spec_name, 'variable': reg},
                    estimate=mean_coef,
                    std_error=nw_se,
                    t_statistic=t_stat,
                    p_value=p_val,
                    n_observations=len(coef_series),
                ))
        
        return pd.DataFrame([r.to_dict() for r in results])
    
    def _newey_west_se(self, series: np.ndarray, n_lags: int) -> float:
        """
        Compute Newey-West standard error.
        
        Parameters
        ----------
        series : np.ndarray
            Time series of coefficient estimates.
        n_lags : int
            Number of lags.
            
        Returns
        -------
        float
            Newey-West standard error.
        """
        n = len(series)
        mean = np.mean(series)
        demeaned = series - mean
        
        # Variance
        gamma_0 = np.sum(demeaned ** 2) / n
        
        # Autocovariances with Bartlett weights
        gamma_sum = gamma_0
        for lag in range(1, min(n_lags + 1, n)):
            gamma_j = np.sum(demeaned[lag:] * demeaned[:-lag]) / n
            weight = 1 - lag / (n_lags + 1)
            gamma_sum += 2 * weight * gamma_j
        
        return np.sqrt(gamma_sum / n)


class BootstrapInference:
    """
    Bootstrap inference for robustness of statistical results.
    """
    
    def __init__(
        self,
        n_bootstrap: int = 1000,
        block_size: Optional[int] = None,
        random_seed: int = 42
    ):
        """
        Initialize bootstrap inference.
        
        Parameters
        ----------
        n_bootstrap : int
            Number of bootstrap samples.
        block_size : int, optional
            Block size for block bootstrap (to preserve autocorrelation).
        random_seed : int
            Random seed for reproducibility.
        """
        self.n_bootstrap = n_bootstrap
        self.block_size = block_size
        self.random_seed = random_seed
    
    def bootstrap_spread(
        self,
        spread_series: pd.Series,
        statistic: str = 'mean'
    ) -> Dict[str, float]:
        """
        Bootstrap confidence interval for spread statistic.
        
        Parameters
        ----------
        spread_series : pd.Series
            Time series of High-Low spreads.
        statistic : str
            Statistic to bootstrap ('mean', 'median', 'sharpe').
            
        Returns
        -------
        Dict[str, float]
            Bootstrap results including confidence intervals.
        """
        np.random.seed(self.random_seed)
        
        n = len(spread_series)
        values = spread_series.values
        
        # Compute observed statistic
        if statistic == 'mean':
            observed = np.mean(values)
            stat_func = np.mean
        elif statistic == 'median':
            observed = np.median(values)
            stat_func = np.median
        elif statistic == 'sharpe':
            observed = np.mean(values) / np.std(values, ddof=1) if np.std(values, ddof=1) > 0 else 0
            stat_func = lambda x: np.mean(x) / np.std(x, ddof=1) if np.std(x, ddof=1) > 0 else 0
        else:
            raise ValueError(f"Unknown statistic: {statistic}")
        
        # Bootstrap
        bootstrap_stats = []
        
        if self.block_size:
            # Block bootstrap
            n_blocks = int(np.ceil(n / self.block_size))
            
            for _ in range(self.n_bootstrap):
                blocks = np.random.randint(0, n - self.block_size + 1, n_blocks)
                sample = np.concatenate([values[b:b + self.block_size] for b in blocks])[:n]
                bootstrap_stats.append(stat_func(sample))
        else:
            # Standard bootstrap
            for _ in range(self.n_bootstrap):
                sample = np.random.choice(values, size=n, replace=True)
                bootstrap_stats.append(stat_func(sample))
        
        bootstrap_stats = np.array(bootstrap_stats)
        
        # Compute confidence intervals
        ci_lower = np.percentile(bootstrap_stats, 2.5)
        ci_upper = np.percentile(bootstrap_stats, 97.5)
        
        # Bootstrap standard error
        bootstrap_se = np.std(bootstrap_stats, ddof=1)
        
        # Bootstrap p-value (two-sided test of H0: statistic = 0)
        p_value = np.mean(np.abs(bootstrap_stats - observed) >= np.abs(observed))
        
        return {
            'observed': observed,
            'bootstrap_se': bootstrap_se,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'p_value': p_value,
            'n_bootstrap': self.n_bootstrap,
        }


class RobustnessRunner:
    """
    Main class to run all robustness checks.
    """
    
    def __init__(self, output_dir: str = 'outputs/tables'):
        """
        Initialize robustness runner.
        
        Parameters
        ----------
        output_dir : str
            Directory for output tables.
        """
        self.output_dir = output_dir
        self.threshold_tester = AlternativeThresholdTester()
        self.subperiod_analyzer = SubperiodAnalyzer()
        self.sort_tester = AlternativeSortTester()
        self.fama_macbeth = FamaMacBethRegressor()
        self.bootstrap = BootstrapInference()
    
    def run_all_checks(
        self,
        down_asy: pd.DataFrame,
        returns: pd.DataFrame,
        characteristics: Optional[pd.DataFrame] = None,
        market_returns: Optional[pd.Series] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Run all robustness checks.
        
        Parameters
        ----------
        down_asy : pd.DataFrame
            DOWN_ASY scores.
        returns : pd.DataFrame
            Monthly returns.
        characteristics : pd.DataFrame, optional
            Firm characteristics.
        market_returns : pd.Series, optional
            Market returns for regime analysis.
            
        Returns
        -------
        Dict[str, pd.DataFrame]
            Results from each robustness check.
        """
        results = {}
        
        print("Running robustness checks...")
        
        # 1. Subperiod analysis
        print("  - Subperiod analysis...")
        results['subperiod'] = self.subperiod_analyzer.run_subperiod_analysis(
            down_asy, returns, market_returns
        )
        
        # 2. Alternative sorts
        print("  - Alternative portfolio sorts...")
        results['alternative_sorts'] = self.sort_tester.run_alternative_sorts(
            down_asy, returns
        )
        
        # 3. Fama-MacBeth regressions
        print("  - Fama-MacBeth regressions...")
        control_vars = None
        if characteristics is not None:
            control_vars = ['BETA', 'IVOL', 'DOWNSIDE_BETA', 'COSKEW']
            control_vars = [c for c in control_vars if c in characteristics.columns]
        
        results['fama_macbeth'] = self.fama_macbeth.run_fama_macbeth(
            returns, down_asy, characteristics, control_vars
        )
        
        print("Robustness checks complete.")
        
        return results
    
    def save_results(
        self,
        results: Dict[str, pd.DataFrame],
        prefix: str = 'robustness'
    ) -> None:
        """
        Save robustness results to files.
        
        Parameters
        ----------
        results : Dict[str, pd.DataFrame]
            Results from robustness checks.
        prefix : str
            Filename prefix.
        """
        import os
        os.makedirs(self.output_dir, exist_ok=True)
        
        for name, df in results.items():
            if len(df) > 0:
                filepath = os.path.join(self.output_dir, f'{prefix}_{name}.csv')
                df.to_csv(filepath, index=False)
                print(f"Saved: {filepath}")
    
    def generate_summary_table(
        self,
        results: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Generate a summary table of all robustness results.
        
        Parameters
        ----------
        results : Dict[str, pd.DataFrame]
            Results from robustness checks.
            
        Returns
        -------
        pd.DataFrame
            Summary table.
        """
        summary_rows = []
        
        for test_name, df in results.items():
            if len(df) == 0:
                continue
            
            for _, row in df.iterrows():
                summary_rows.append({
                    'Category': test_name,
                    'Test': row.get('test_name', ''),
                    'Specification': str(row.get('specification', row.get('model', row.get('period', '')))),
                    'Estimate': row.get('estimate', np.nan),
                    't-statistic': row.get('t_statistic', np.nan),
                    'p-value': row.get('p_value', np.nan),
                    'N': row.get('n_observations', 0),
                })
        
        return pd.DataFrame(summary_rows)


# Demo function for testing
def demo_robustness():
    """Demonstrate robustness module functionality."""
    print("=" * 60)
    print("Phase 6: Robustness Checks - Demo")
    print("=" * 60)
    
    # Create synthetic data
    np.random.seed(42)
    n_stocks = 100
    n_months = 60
    
    dates = pd.date_range('2000-01-01', periods=n_months, freq='ME')
    permnos = list(range(10001, 10001 + n_stocks))
    
    # Create returns
    returns_data = []
    for date in dates:
        for permno in permnos:
            returns_data.append({
                'PERMNO': permno,
                'DATE': date,
                'RET': np.random.normal(0.01, 0.05),
            })
    returns = pd.DataFrame(returns_data)
    
    # Create DOWN_ASY scores
    down_asy_data = []
    for date in dates:
        for permno in permnos:
            down_asy_data.append({
                'PERMNO': permno,
                'DATE': date,
                'DOWN_ASY': np.random.uniform(-0.1, 0.1) + (permno - 10050) * 0.001,
            })
    down_asy = pd.DataFrame(down_asy_data)
    
    # Create characteristics
    char_data = []
    for date in dates:
        for permno in permnos:
            char_data.append({
                'PERMNO': permno,
                'DATE': date,
                'BETA': np.random.uniform(0.5, 1.5),
                'IVOL': np.random.uniform(0.01, 0.1),
                'DOWNSIDE_BETA': np.random.uniform(0.5, 2.0),
                'COSKEW': np.random.uniform(-0.5, 0.5),
            })
    characteristics = pd.DataFrame(char_data)
    
    # Create market returns
    market_returns = pd.Series(
        np.random.normal(0.008, 0.04, n_months),
        index=dates
    )
    
    # Run robustness checks
    runner = RobustnessRunner()
    results = runner.run_all_checks(
        down_asy, returns, characteristics, market_returns
    )
    
    # Print results
    print("\n1. Subperiod Analysis:")
    print("-" * 40)
    if len(results['subperiod']) > 0:
        print(results['subperiod'][['test_name', 'estimate', 't_statistic', 'n_observations']].to_string(index=False))
    else:
        print("No subperiod results available.")
    
    print("\n2. Alternative Portfolio Sorts:")
    print("-" * 40)
    if len(results['alternative_sorts']) > 0:
        print(results['alternative_sorts'][['n_groups', 'method', 'estimate', 't_statistic']].to_string(index=False))
    else:
        print("No alternative sort results available.")
    
    print("\n3. Fama-MacBeth Regressions:")
    print("-" * 40)
    if len(results['fama_macbeth']) > 0:
        fm_display = results['fama_macbeth'][['model', 'variable', 'estimate', 't_statistic']]
        print(fm_display.to_string(index=False))
    else:
        print("No Fama-MacBeth results available.")
    
    # Generate summary
    print("\n4. Summary Table:")
    print("-" * 40)
    summary = runner.generate_summary_table(results)
    if len(summary) > 0:
        print(f"Total robustness tests: {len(summary)}")
        significant = (summary['p-value'] < 0.05).sum()
        print(f"Significant at 5%: {significant}")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == '__main__':
    demo_robustness()
