"""
src/python/portfolio.py
Parallel rolling window computation of DOWN_ASY for empirical analysis.

This module implements the large-scale rolling window analysis required for Phase 4.
For each stock-month, it computes the DOWN_ASY score using the past 12 months of
daily returns, strictly following the paper's methodology.

Key Features:
- Parallel processing using multiprocessing.Pool
- Efficient rolling window generation (zero-copy views)
- Automatic caching to data/processed/down_asy_scores.parquet
- Strict look-ahead bias prevention (t-1 signal for t returns)
- Integration with Phase 1 C++ entropy engine

Mathematical Reference:
- Rolling window: Use daily returns from t-252 to t-1 to compute DOWN_ASY at time t
- Standardization: (r - mean(r)) / std(r) applied per window
- Entropy call: entropy_cpp.calculate_metrics(x_std, y_std, c=0.0)
- Signal lag: DOWN_ASY computed at t-1 predicts returns at t

Performance Note:
- Full CRSP dataset: ~3,000 stocks × 600 months = 1.8M computations
- Estimated runtime: ~10-30 hours on 8 cores (depends on CPU)
- Uses aggressive caching to enable restarts without re-computation
"""

import numpy as np
import pandas as pd
import multiprocessing as mp
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from datetime import datetime, timedelta
import sys

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Try to import the C++ entropy engine
try:
    import entropy_cpp
    HAS_CPP_ENGINE = True
except ImportError:
    print("WARNING: entropy_cpp not found. Using dummy implementation.")
    HAS_CPP_ENGINE = False
    
    # Dummy implementation for testing without C++
    class DummyEngine:
        def calculate_metrics(self, x, y, c):
            # Simple proxy: correlation-based asymmetry
            rho = np.corrcoef(x, y)[0, 1]
            return abs(rho) * 0.1, rho * 0.1
    
    class entropy_cpp:
        @staticmethod
        def EntropyEngine():
            return DummyEngine()


def get_month_end_indices(dates: pd.DatetimeIndex) -> np.ndarray:
    """
    Identify month-end indices in a datetime series.
    
    Parameters
    ----------
    dates : pd.DatetimeIndex
        Time series of dates.
    
    Returns
    -------
    np.ndarray
        Integer indices of month-end observations.
    """
    # Group by year-month and take last position of each group
    month_groups = dates.to_series().reset_index(drop=True).groupby([dates.year, dates.month])
    month_end_positions = month_groups.apply(lambda x: x.index[-1]).values
    return month_end_positions.astype(int)


def standardize_array(arr: np.ndarray) -> np.ndarray:
    """
    Standardize array to mean=0, std=1.
    
    Parameters
    ----------
    arr : np.ndarray
        Input array.
    
    Returns
    -------
    np.ndarray
        Standardized array.
    """
    if len(arr) < 2:
        return arr
    
    mean = np.mean(arr)
    std = np.std(arr, ddof=1)
    
    if std < 1e-10:  # Handle constant series
        return np.zeros_like(arr)
    
    return (arr - mean) / std


def process_stock_rolling(args: Tuple) -> List[Dict]:
    """
    Worker function to compute DOWN_ASY for a single stock's history.
    
    This function processes one stock at a time, computing rolling window
    DOWN_ASY scores for all valid month-ends in the stock's trading history.
    
    Algorithm:
    1. Identify month-end dates
    2. For each month-end:
       a. Extract 252-day lookback window (if available)
       b. Filter: Skip if fewer than 100 observations
       c. Standardize stock and market returns
       d. Call C++ entropy engine
       e. Store (PERMNO, DATE, DOWN_ASY, S_rho)
    
    Parameters
    ----------
    args : tuple
        (permno, dates, stock_returns, market_returns)
        - permno: Stock identifier
        - dates: List of dates (will be converted to DatetimeIndex)
        - stock_returns: Daily returns for the stock (list)
        - market_returns: Daily market returns (list)
    
    Returns
    -------
    List[Dict]
        List of dictionaries with keys: PERMNO, DATE, DOWN_ASY, S_rho
    """
    permno, dates_list, stock_returns_list, market_returns_list = args
    
    # Convert lists back to numpy arrays and DatetimeIndex
    dates = pd.DatetimeIndex(dates_list)
    stock_returns = np.array(stock_returns_list)
    market_returns = np.array(market_returns_list)
    
    results = []
    
    # Identify month-end indices
    try:
        month_ends = get_month_end_indices(dates)
    except Exception as e:
        print(f"Error finding month-ends for PERMNO {permno}: {e}")
        return results
    
    # Rolling window size: 252 trading days ≈ 12 months
    window_size = 252
    
    for month_end_idx in month_ends:
        # Define lookback window [t-252, t-1]
        # Note: We use t-1 to prevent look-ahead bias
        end_idx = month_end_idx
        start_idx = max(0, end_idx - window_size)
        
        # Extract window
        window_stock = stock_returns[start_idx:end_idx]
        window_mkt = market_returns[start_idx:end_idx]
        window_dates = dates[start_idx:end_idx]
        
        # Filter: Minimum 100 observations (per paper)
        if len(window_stock) < 100:
            continue
        
        # Check for valid data (no NaNs, finite values)
        if np.any(~np.isfinite(window_stock)) or np.any(~np.isfinite(window_mkt)):
            continue
        
        # Standardization (per paper, Section IV.A)
        try:
            x_std = standardize_array(window_stock)
            y_std = standardize_array(window_mkt)
        except Exception as e:
            print(f"Standardization error for PERMNO {permno} at {dates[month_end_idx]}: {e}")
            continue
        
        # Check standardization validity
        if np.all(x_std == 0) or np.all(y_std == 0):
            continue  # Skip constant series
        
        # Call C++ entropy engine
        try:
            engine = entropy_cpp.EntropyEngine()
            s_rho, down_asy = engine.calculate_metrics(x_std, y_std, c=0.0)
        except Exception as e:
            print(f"Entropy calculation error for PERMNO {permno} at {dates[month_end_idx]}: {e}")
            continue
        
        # Store result (use next month for forward-looking return)
        results.append({
            'PERMNO': permno,
            'DATE': dates[month_end_idx],
            'DOWN_ASY': down_asy,
            'S_rho': s_rho,
            'N_OBS': len(window_stock)
        })
    
    return results


class PortfolioAnalyzer:
    """
    Main class for portfolio-level analysis.
    
    This class orchestrates the rolling window computation across the entire
    CRSP dataset, manages parallelization, and handles caching.
    
    Attributes
    ----------
    data_dir : Path
        Directory containing processed data files.
    cache_dir : Path
        Directory for caching intermediate results.
    n_processes : int
        Number of parallel processes (default: all available CPUs - 1).
    
    Methods
    -------
    compute_rolling_asymmetry(stock_data, market_data, force_refresh=False)
        Compute DOWN_ASY for all stocks using parallel processing.
    load_cached_scores()
        Load previously computed DOWN_ASY scores from cache.
    sort_into_portfolios(scores, characteristic, n_bins=5)
        Sort stocks into portfolios based on a characteristic.
    compute_portfolio_returns(scores, returns, n_bins=5, equal_weighted=True)
        Calculate portfolio returns from sorts.
    """
    
    def __init__(self, 
                 data_dir: str = "data/processed",
                 cache_dir: str = "data/processed",
                 n_processes: Optional[int] = None):
        """
        Initialize the portfolio analyzer.
        
        Parameters
        ----------
        data_dir : str
            Path to directory with processed data.
        cache_dir : str
            Path to cache directory.
        n_processes : int, optional
            Number of processes for parallel execution.
            If None, uses (CPU count - 1).
        """
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        if n_processes is None:
            self.n_processes = max(1, mp.cpu_count() - 1)
        else:
            self.n_processes = n_processes
        
        self.scores_cache_path = self.cache_dir / "down_asy_scores.parquet"
    
    def compute_rolling_asymmetry(self,
                                   stock_data: pd.DataFrame,
                                   market_data: pd.Series,
                                   force_refresh: bool = False) -> pd.DataFrame:
        """
        Compute DOWN_ASY scores for all stocks using rolling windows.
        
        This is the main computational workhorse of Phase 4. It processes
        each stock in parallel, computing DOWN_ASY for every month-end
        in the stock's trading history.
        
        Parameters
        ----------
        stock_data : pd.DataFrame
            Daily stock returns with MultiIndex (PERMNO, DATE) or columns.
            Expected structure: 
            - Index: DATE
            - Columns: PERMNO (one column per stock)
            OR
            - MultiIndex: (PERMNO, DATE)
            - Column: 'RET' or 'EXRET'
        market_data : pd.Series
            Daily market returns indexed by DATE.
        force_refresh : bool
            If True, recompute even if cache exists.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with columns: PERMNO, DATE, DOWN_ASY, S_rho, N_OBS
        """
        # Check cache
        if not force_refresh and self.scores_cache_path.exists():
            print(f"Loading cached scores from {self.scores_cache_path}")
            return pd.read_parquet(self.scores_cache_path)
        
        print("Computing rolling window DOWN_ASY scores...")
        print(f"Using {self.n_processes} parallel processes")
        
        # Prepare data structure for parallel processing
        # Convert to wide format if needed
        if isinstance(stock_data.index, pd.MultiIndex):
            # MultiIndex format: pivot to wide
            stock_data_wide = stock_data.reset_index().pivot(
                index='DATE', columns='PERMNO', values='RET'
            )
        else:
            # Already wide format
            stock_data_wide = stock_data
        
        # Align market data with stock data dates
        common_dates = stock_data_wide.index.intersection(market_data.index)
        stock_data_aligned = stock_data_wide.loc[common_dates]
        market_data_aligned = market_data.loc[common_dates]
        
        # Prepare arguments for each stock
        permnos = stock_data_aligned.columns
        n_stocks = len(permnos)
        
        print(f"Processing {n_stocks} stocks...")
        
        args_list = []
        for permno in permnos:
            stock_returns = stock_data_aligned[permno].values
            
            # Skip if all NaN
            if np.all(~np.isfinite(stock_returns)):
                continue
            
            # Convert to lists for better pickling compatibility
            args_list.append((
                permno,
                stock_data_aligned.index.tolist(),  # Convert to list
                stock_returns.tolist(),  # Convert to list
                market_data_aligned.values.tolist()  # Convert to list
            ))
        
        # Parallel processing
        print(f"Starting parallel computation for {len(args_list)} valid stocks...")
        
        with mp.Pool(processes=self.n_processes) as pool:
            results_list = pool.map(process_stock_rolling, args_list)
        
        # Flatten results
        all_results = []
        for stock_results in results_list:
            all_results.extend(stock_results)
        
        # Convert to DataFrame
        if len(all_results) == 0:
            print("WARNING: No results generated!")
            return pd.DataFrame(columns=['PERMNO', 'DATE', 'DOWN_ASY', 'S_rho', 'N_OBS'])
        
        df_results = pd.DataFrame(all_results)
        
        print(f"Computed {len(df_results)} stock-month observations")
        print(f"Date range: {df_results['DATE'].min()} to {df_results['DATE'].max()}")
        print(f"Unique stocks: {df_results['PERMNO'].nunique()}")
        
        # Cache results
        print(f"Caching results to {self.scores_cache_path}")
        df_results.to_parquet(self.scores_cache_path, index=False)
        
        return df_results
    
    def load_cached_scores(self) -> pd.DataFrame:
        """
        Load previously computed DOWN_ASY scores.
        
        Returns
        -------
        pd.DataFrame
            Cached scores.
        
        Raises
        ------
        FileNotFoundError
            If cache file does not exist.
        """
        if not self.scores_cache_path.exists():
            raise FileNotFoundError(
                f"Cache not found at {self.scores_cache_path}. "
                f"Run compute_rolling_asymmetry() first."
            )
        
        return pd.read_parquet(self.scores_cache_path)
    
    def sort_into_portfolios(self,
                             scores: pd.DataFrame,
                             characteristic: str = 'DOWN_ASY',
                             n_bins: int = 5,
                             method: str = 'quantile') -> pd.DataFrame:
        """
        Sort stocks into portfolios based on a characteristic.
        
        This implements the univariate portfolio formation procedure.
        At each date, stocks are ranked and assigned to bins.
        
        Parameters
        ----------
        scores : pd.DataFrame
            DataFrame with columns: PERMNO, DATE, <characteristic>
        characteristic : str
            Column name to sort on (e.g., 'DOWN_ASY').
        n_bins : int
            Number of portfolios (bins).
        method : str
            Sorting method: 'quantile' or 'equal'.
        
        Returns
        -------
        pd.DataFrame
            Original DataFrame with added 'PORTFOLIO' column.
        """
        scores = scores.copy()
        
        def assign_portfolios(group):
            """Assign portfolio numbers within a date."""
            # Preserve DATE (groupby drops it)
            date_val = group['DATE'].iloc[0] if 'DATE' in group.columns else group.name
            
            if method == 'quantile':
                # Quantile-based (equal number of stocks per portfolio)
                group['PORTFOLIO'] = pd.qcut(
                    group[characteristic],
                    q=n_bins,
                    labels=False,
                    duplicates='drop'
                ) + 1  # 1-indexed
            else:
                # Equal-width bins
                group['PORTFOLIO'] = pd.cut(
                    group[characteristic],
                    bins=n_bins,
                    labels=False,
                    duplicates='drop'
                ) + 1
            
            # Restore DATE column
            group['DATE'] = date_val
            return group
        
        # Apply sorting within each date
        scores_sorted = scores.groupby('DATE', group_keys=False).apply(assign_portfolios)
        scores_sorted = scores_sorted.reset_index(drop=True)
        
        return scores_sorted
    
    def compute_portfolio_returns(self,
                                   portfolio_assignments: pd.DataFrame,
                                   returns: pd.DataFrame,
                                   n_bins: int = 5,
                                   equal_weighted: bool = True,
                                   lag: int = 1) -> pd.DataFrame:
        """
        Calculate portfolio returns from portfolio assignments.
        
        This implements the return calculation step. Portfolio assignments
        at time t-1 are matched with returns at time t (preventing look-ahead bias).
        
        Parameters
        ----------
        portfolio_assignments : pd.DataFrame
            Output of sort_into_portfolios() with columns:
            PERMNO, DATE, PORTFOLIO
        returns : pd.DataFrame
            Stock returns with columns: PERMNO, DATE, RET
        n_bins : int
            Number of portfolios.
        equal_weighted : bool
            If True, equal-weighted portfolios. If False, value-weighted.
        lag : int
            Number of months to lag signal (default=1 to prevent look-ahead).
        
        Returns
        -------
        pd.DataFrame
            Portfolio returns with columns: DATE, PORTFOLIO_1, ..., PORTFOLIO_N
        """
        # Lag the portfolio assignments
        portfolio_assignments = portfolio_assignments.copy()
        
        # Ensure DATE is a column (handle various pandas behaviors)
        if 'DATE' not in portfolio_assignments.columns:
            if portfolio_assignments.index.name == 'DATE':
                portfolio_assignments = portfolio_assignments.reset_index()
            elif isinstance(portfolio_assignments.index, pd.MultiIndex):
                portfolio_assignments = portfolio_assignments.reset_index()
        
        portfolio_assignments['DATE'] = portfolio_assignments['DATE'] + pd.DateOffset(months=lag)
        
        # Merge with returns
        merged = portfolio_assignments.merge(
            returns,
            on=['PERMNO', 'DATE'],
            how='inner'
        )
        
        # Calculate portfolio returns
        if equal_weighted:
            # Equal-weighted: simple average
            port_returns = merged.groupby(['DATE', 'PORTFOLIO'])['RET'].mean()
        else:
            # Value-weighted: would need market cap data
            raise NotImplementedError("Value-weighted returns require market cap data")
        
        # Pivot to wide format
        port_returns_wide = port_returns.unstack('PORTFOLIO')
        
        # Rename columns
        port_returns_wide.columns = [f'PORTFOLIO_{int(i)}' for i in port_returns_wide.columns]
        
        # Calculate High-Low spread
        if n_bins >= 2:
            port_returns_wide['HIGH_LOW'] = (
                port_returns_wide[f'PORTFOLIO_{n_bins}'] - 
                port_returns_wide['PORTFOLIO_1']
            )
        
        return port_returns_wide.reset_index()


def double_sort_portfolios(scores: pd.DataFrame,
                            control_var: str,
                            target_var: str = 'DOWN_ASY',
                            n_bins: int = 5) -> pd.DataFrame:
    """
    Perform sequential double sorts.
    
    This implements the double-sorting procedure used in Table 7.
    First sort on control variable, then sort on target variable within
    each control bin.
    
    Algorithm:
    1. At each date, sort stocks into n_bins based on control_var
    2. Within each control bin, sort into n_bins based on target_var
    3. Result: n_bins × n_bins portfolios
    
    Parameters
    ----------
    scores : pd.DataFrame
        DataFrame with columns: PERMNO, DATE, control_var, target_var
    control_var : str
        First sorting variable (e.g., 'DOWNSIDE_BETA').
    target_var : str
        Second sorting variable (e.g., 'DOWN_ASY').
    n_bins : int
        Number of bins for each sort.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with added columns: CONTROL_PORT, TARGET_PORT
    """
    scores = scores.copy()
    
    def double_sort_date(group):
        """Apply double sort within a date."""
        # First sort: control variable
        group['CONTROL_PORT'] = pd.qcut(
            group[control_var],
            q=n_bins,
            labels=False,
            duplicates='drop'
        ) + 1
        
        # Second sort: target variable within each control bin
        def sort_within_control(sub_group):
            sub_group['TARGET_PORT'] = pd.qcut(
                sub_group[target_var],
                q=n_bins,
                labels=False,
                duplicates='drop'
            ) + 1
            return sub_group
        
        group = group.groupby('CONTROL_PORT').apply(sort_within_control)
        return group
    
    # Apply double sort within each date
    scores_sorted = scores.groupby('DATE', group_keys=False).apply(double_sort_date)
    scores_sorted = scores_sorted.reset_index(drop=True)
    
    # Ensure portfolio columns exist (handle edge cases)
    if 'CONTROL_PORT' not in scores_sorted.columns:
        scores_sorted['CONTROL_PORT'] = 1
    if 'TARGET_PORT' not in scores_sorted.columns:
        scores_sorted['TARGET_PORT'] = 1
    
    return scores_sorted


if __name__ == "__main__":
    """
    Example usage and quick test.
    """
    print("=== Portfolio Module Test ===\n")
    
    # Create dummy data for testing
    np.random.seed(42)
    
    dates = pd.date_range('2020-01-01', '2022-12-31', freq='B')  # Business days
    n_stocks = 10
    permnos = [f'STOCK_{i:02d}' for i in range(1, n_stocks + 1)]
    
    # Create dummy stock returns (wide format)
    stock_returns = pd.DataFrame(
        np.random.randn(len(dates), n_stocks) * 0.02,
        index=dates,
        columns=permnos
    )
    
    # Create dummy market returns
    market_returns = pd.Series(
        np.random.randn(len(dates)) * 0.015,
        index=dates,
        name='MKT'
    )
    
    # Initialize analyzer
    analyzer = PortfolioAnalyzer(
        cache_dir='data/cache',
        n_processes=2
    )
    
    print("1. Computing rolling asymmetry scores...")
    scores = analyzer.compute_rolling_asymmetry(
        stock_returns,
        market_returns,
        force_refresh=True
    )
    
    print(f"\nScores shape: {scores.shape}")
    print(f"Columns: {scores.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(scores.head())
    
    if len(scores) > 0:
        print("\n2. Sorting into quintiles...")
        sorted_scores = analyzer.sort_into_portfolios(
            scores,
            characteristic='DOWN_ASY',
            n_bins=5
        )
        
        print(f"\nPortfolio distribution:")
        print(sorted_scores['PORTFOLIO'].value_counts().sort_index())
        
        print("\n3. Computing portfolio returns...")
        # Create dummy monthly returns for testing
        monthly_dates = pd.date_range('2020-01-31', '2022-12-31', freq='M')
        monthly_returns = []
        for permno in permnos:
            for date in monthly_dates:
                monthly_returns.append({
                    'PERMNO': permno,
                    'DATE': date,
                    'RET': np.random.randn() * 0.05
                })
        
        returns_df = pd.DataFrame(monthly_returns)
        
        port_returns = analyzer.compute_portfolio_returns(
            sorted_scores,
            returns_df,
            n_bins=5,
            equal_weighted=True
        )
        
        print(f"\nPortfolio returns shape: {port_returns.shape}")
        print(f"Columns: {port_returns.columns.tolist()}")
        print(f"\nFirst few rows:")
        print(port_returns.head())
        
        if 'HIGH_LOW' in port_returns.columns:
            hl_mean = port_returns['HIGH_LOW'].mean()
            hl_std = port_returns['HIGH_LOW'].std()
            hl_tstat = hl_mean / (hl_std / np.sqrt(len(port_returns)))
            
            print(f"\nHigh-Low Spread Statistics:")
            print(f"  Mean: {hl_mean:.4f}")
            print(f"  Std:  {hl_std:.4f}")
            print(f"  t-stat: {hl_tstat:.2f}")
    
    print("\n✓ Portfolio module test complete!")
