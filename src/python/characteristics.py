"""
Firm Characteristics Computation Module

This module implements the computation of firm characteristics used in the
empirical analysis, including:
- Idiosyncratic Volatility (IVOL)
- Coskewness (Harvey-Siddique)
- Downside Beta
- Cokurtosis
- Illiquidity (Amihud)

All calculations use rolling windows and are vectorized for efficiency.

References:
- Harvey and Siddique (2000) for Coskewness
- Amihud (2002) for Illiquidity
- Ang et al. (2006) for Downside Beta
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict
import warnings


class CharacteristicsEngine:
    """
    Compute firm characteristics from stock and market returns.
    
    All characteristics are computed using rolling windows to match
    the empirical analysis in the paper.
    """
    
    def __init__(self, window_size: int = 252, min_observations: int = 100):
        """
        Initialize the characteristics engine.
        
        Parameters
        ----------
        window_size : int
            Number of days for rolling window (default: 252 = 1 year).
        min_observations : int
            Minimum number of observations required for calculation.
        """
        self.window_size = window_size
        self.min_observations = min_observations
    
    def compute_all_characteristics(self,
                                      stock_returns: pd.DataFrame,
                                      market_returns: pd.Series,
                                      volumes: Optional[pd.DataFrame] = None,
                                      prices: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Compute all firm characteristics for all stocks.
        
        Parameters
        ----------
        stock_returns : pd.DataFrame
            Daily stock returns. Index: Date, Columns: PERMNO
        market_returns : pd.Series
            Daily market returns. Index: Date
        volumes : pd.DataFrame, optional
            Daily trading volumes. Index: Date, Columns: PERMNO
        prices : pd.DataFrame, optional
            Daily stock prices. Index: Date, Columns: PERMNO
        
        Returns
        -------
        pd.DataFrame
            Characteristics for each stock-month.
            Columns: PERMNO, DATE, IVOL, BETA, DOWNSIDE_BETA, 
                     COSKEW, COKURT, ILLIQ
        """
        results = []
        
        # Get month-end dates
        month_ends = self._get_month_ends(stock_returns.index)
        
        print(f"Computing characteristics for {len(stock_returns.columns)} stocks...")
        print(f"Window size: {self.window_size} days, Min obs: {self.min_observations}")
        
        for permno in stock_returns.columns:
            stock_ret = stock_returns[permno].dropna()
            
            if len(stock_ret) < self.min_observations:
                continue
            
            # Align with market
            common_dates = stock_ret.index.intersection(market_returns.index)
            if len(common_dates) < self.min_observations:
                continue
            
            stock_ret_aligned = stock_ret.loc[common_dates]
            market_ret_aligned = market_returns.loc[common_dates]
            
            # Compute characteristics for each month-end
            for month_end in month_ends:
                if month_end not in common_dates:
                    continue
                
                # Extract window [month_end - window_size, month_end]
                window_end_idx = common_dates.get_loc(month_end)
                window_start_idx = max(0, window_end_idx - self.window_size + 1)
                
                if window_end_idx - window_start_idx + 1 < self.min_observations:
                    continue
                
                window_dates = common_dates[window_start_idx:window_end_idx + 1]
                r_stock = stock_ret_aligned.loc[window_dates].values
                r_market = market_ret_aligned.loc[window_dates].values
                
                # Compute characteristics
                char_dict = {
                    'PERMNO': permno,
                    'DATE': month_end
                }
                
                # CAPM Beta and IVOL
                beta, ivol = self.compute_beta_and_ivol(r_stock, r_market)
                char_dict['BETA'] = beta
                char_dict['IVOL'] = ivol
                
                # Downside Beta
                downside_beta = self.compute_downside_beta(r_stock, r_market)
                char_dict['DOWNSIDE_BETA'] = downside_beta
                
                # Coskewness
                coskew = self.compute_coskewness(r_stock, r_market)
                char_dict['COSKEW'] = coskew
                
                # Cokurtosis
                cokurt = self.compute_cokurtosis(r_stock, r_market)
                char_dict['COKURT'] = cokurt
                
                # Illiquidity (if volume and price data available)
                if volumes is not None and prices is not None and permno in volumes.columns:
                    vol = volumes[permno].loc[window_dates].values if permno in volumes.columns else None
                    price = prices[permno].loc[window_dates].values if permno in prices.columns else None
                    illiq = self.compute_illiquidity(r_stock, vol, price)
                    char_dict['ILLIQ'] = illiq
                else:
                    char_dict['ILLIQ'] = np.nan
                
                results.append(char_dict)
        
        df = pd.DataFrame(results)
        print(f"Computed {len(df)} stock-month characteristic observations")
        
        return df
    
    def compute_beta_and_ivol(self, 
                               stock_returns: np.ndarray, 
                               market_returns: np.ndarray) -> Tuple[float, float]:
        """
        Compute CAPM beta and idiosyncratic volatility.
        
        CAPM Model:
            r_i = α + β * r_m + ε
        
        IVOL = std(ε)
        
        Parameters
        ----------
        stock_returns : np.ndarray
            Stock returns (T,)
        market_returns : np.ndarray
            Market returns (T,)
        
        Returns
        -------
        Tuple[float, float]
            (beta, ivol)
        """
        # Remove NaN
        mask = ~(np.isnan(stock_returns) | np.isnan(market_returns))
        r_stock = stock_returns[mask]
        r_market = market_returns[mask]
        
        if len(r_stock) < 2:
            return np.nan, np.nan
        
        # OLS: beta = Cov(r_i, r_m) / Var(r_m)
        cov_matrix = np.cov(r_stock, r_market)
        cov_stock_market = cov_matrix[0, 1]
        var_market = cov_matrix[1, 1]
        
        if var_market < 1e-10:
            return np.nan, np.nan
        
        beta = cov_stock_market / var_market
        
        # Compute residuals
        alpha = np.mean(r_stock) - beta * np.mean(r_market)
        residuals = r_stock - (alpha + beta * r_market)
        
        # IVOL = std(residuals)
        ivol = np.std(residuals, ddof=1)
        
        return beta, ivol
    
    def compute_downside_beta(self,
                               stock_returns: np.ndarray,
                               market_returns: np.ndarray) -> float:
        """
        Compute downside beta (beta conditional on market downturns).
        
        Downside Beta:
            β⁻ = Cov(r_i, r_m | r_m < μ_m) / Var(r_m | r_m < μ_m)
        
        Parameters
        ----------
        stock_returns : np.ndarray
            Stock returns (T,)
        market_returns : np.ndarray
            Market returns (T,)
        
        Returns
        -------
        float
            Downside beta
        """
        # Remove NaN
        mask = ~(np.isnan(stock_returns) | np.isnan(market_returns))
        r_stock = stock_returns[mask]
        r_market = market_returns[mask]
        
        if len(r_stock) < 2:
            return np.nan
        
        # Identify downside (market < mean)
        market_mean = np.mean(r_market)
        downside_mask = r_market < market_mean
        
        if np.sum(downside_mask) < 2:
            return np.nan
        
        r_stock_down = r_stock[downside_mask]
        r_market_down = r_market[downside_mask]
        
        # Compute conditional covariance and variance
        cov_down = np.cov(r_stock_down, r_market_down)[0, 1]
        var_market_down = np.var(r_market_down, ddof=1)
        
        if var_market_down < 1e-10:
            return np.nan
        
        downside_beta = cov_down / var_market_down
        
        return downside_beta
    
    def compute_coskewness(self,
                            stock_returns: np.ndarray,
                            market_returns: np.ndarray) -> float:
        """
        Compute coskewness (Harvey and Siddique, 2000).
        
        Coskewness:
            COSKEW = E[(r_i - μ_i) * (r_m - μ_m)²] / (σ_i * σ_m²)
        
        Measures the co-movement of stock returns with squared market returns.
        Negative coskewness indicates stock performs poorly when market crashes.
        
        Parameters
        ----------
        stock_returns : np.ndarray
            Stock returns (T,)
        market_returns : np.ndarray
            Market returns (T,)
        
        Returns
        -------
        float
            Coskewness
        """
        # Remove NaN
        mask = ~(np.isnan(stock_returns) | np.isnan(market_returns))
        r_stock = stock_returns[mask]
        r_market = market_returns[mask]
        
        if len(r_stock) < 2:
            return np.nan
        
        # Standardize
        stock_mean = np.mean(r_stock)
        market_mean = np.mean(r_market)
        stock_std = np.std(r_stock, ddof=1)
        market_std = np.std(r_market, ddof=1)
        
        if stock_std < 1e-10 or market_std < 1e-10:
            return np.nan
        
        # Compute coskewness
        stock_dev = r_stock - stock_mean
        market_dev = r_market - market_mean
        market_dev_squared = market_dev ** 2
        
        coskew = np.mean(stock_dev * market_dev_squared) / (stock_std * market_std ** 2)
        
        return coskew
    
    def compute_cokurtosis(self,
                            stock_returns: np.ndarray,
                            market_returns: np.ndarray) -> float:
        """
        Compute cokurtosis (fourth co-moment).
        
        Cokurtosis:
            COKURT = E[(r_i - μ_i) * (r_m - μ_m)³] / (σ_i * σ_m³)
        
        Measures the co-movement of stock returns with cubed market returns.
        
        Parameters
        ----------
        stock_returns : np.ndarray
            Stock returns (T,)
        market_returns : np.ndarray
            Market returns (T,)
        
        Returns
        -------
        float
            Cokurtosis
        """
        # Remove NaN
        mask = ~(np.isnan(stock_returns) | np.isnan(market_returns))
        r_stock = stock_returns[mask]
        r_market = market_returns[mask]
        
        if len(r_stock) < 2:
            return np.nan
        
        # Standardize
        stock_mean = np.mean(r_stock)
        market_mean = np.mean(r_market)
        stock_std = np.std(r_stock, ddof=1)
        market_std = np.std(r_market, ddof=1)
        
        if stock_std < 1e-10 or market_std < 1e-10:
            return np.nan
        
        # Compute cokurtosis
        stock_dev = r_stock - stock_mean
        market_dev = r_market - market_mean
        market_dev_cubed = market_dev ** 3
        
        cokurt = np.mean(stock_dev * market_dev_cubed) / (stock_std * market_std ** 3)
        
        return cokurt
    
    def compute_illiquidity(self,
                             returns: np.ndarray,
                             volumes: Optional[np.ndarray],
                             prices: Optional[np.ndarray]) -> float:
        """
        Compute Amihud (2002) illiquidity measure.
        
        Illiquidity:
            ILLIQ = (1/N) * Σ |r_t| / (VOLD_t * P_t)
        
        where VOLD is dollar volume.
        
        Higher values indicate lower liquidity.
        
        Parameters
        ----------
        returns : np.ndarray
            Daily returns (T,)
        volumes : np.ndarray, optional
            Daily trading volumes (T,)
        prices : np.ndarray, optional
            Daily prices (T,)
        
        Returns
        -------
        float
            Illiquidity measure (multiplied by 10^6 for readability)
        """
        if volumes is None or prices is None:
            return np.nan
        
        # Remove NaN
        mask = ~(np.isnan(returns) | np.isnan(volumes) | np.isnan(prices))
        mask &= (volumes > 0) & (prices > 0)
        
        if np.sum(mask) < 2:
            return np.nan
        
        r = returns[mask]
        vol = volumes[mask]
        p = prices[mask]
        
        # Dollar volume
        dollar_volume = vol * p
        
        # Illiquidity ratio
        illiq_ratios = np.abs(r) / dollar_volume
        
        # Average and scale by 10^6
        illiq = np.mean(illiq_ratios) * 1e6
        
        # Winsorize at 99th percentile (common practice)
        illiq = min(illiq, 30.0)  # Cap at 30 as in paper
        
        return illiq
    
    def _get_month_ends(self, dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
        """
        Identify month-end dates.
        
        Parameters
        ----------
        dates : pd.DatetimeIndex
            All dates in the dataset.
        
        Returns
        -------
        pd.DatetimeIndex
            Month-end dates.
        """
        df = pd.DataFrame(index=dates)
        df['month'] = df.index.to_period('M')
        month_ends = df.groupby('month').apply(lambda x: x.index.max())
        return pd.DatetimeIndex(month_ends.values)


def compute_characteristics_for_dataset(stock_returns: pd.DataFrame,
                                          market_returns: pd.Series,
                                          window_size: int = 252,
                                          min_observations: int = 100,
                                          volumes: Optional[pd.DataFrame] = None,
                                          prices: Optional[pd.DataFrame] = None,
                                          cache_path: Optional[str] = None,
                                          force_refresh: bool = False) -> pd.DataFrame:
    """
    Compute characteristics for entire dataset with caching.
    
    This is the main entry point for characteristic computation.
    
    Parameters
    ----------
    stock_returns : pd.DataFrame
        Daily stock returns. Index: Date, Columns: PERMNO
    market_returns : pd.Series
        Daily market returns. Index: Date
    window_size : int
        Rolling window size (default: 252 days)
    min_observations : int
        Minimum observations required (default: 100)
    volumes : pd.DataFrame, optional
        Trading volumes for illiquidity calculation
    prices : pd.DataFrame, optional
        Stock prices for illiquidity calculation
    cache_path : str, optional
        Path to cache file (Parquet format)
    force_refresh : bool
        If True, recompute even if cache exists
    
    Returns
    -------
    pd.DataFrame
        Firm characteristics for all stock-months
    """
    # Check cache
    if cache_path and not force_refresh:
        try:
            df = pd.read_parquet(cache_path)
            print(f"Loaded {len(df)} cached characteristic observations from {cache_path}")
            return df
        except FileNotFoundError:
            print("Cache not found, computing characteristics...")
    
    # Compute characteristics
    engine = CharacteristicsEngine(window_size=window_size, 
                                     min_observations=min_observations)
    
    df = engine.compute_all_characteristics(
        stock_returns,
        market_returns,
        volumes=volumes,
        prices=prices
    )
    
    # Cache results
    if cache_path:
        df.to_parquet(cache_path)
        print(f"Cached results to {cache_path}")
    
    return df


# Testing code
if __name__ == "__main__":
    """
    Quick test of characteristics computation.
    """
    print("=== Characteristics Module Test ===\n")
    
    # Generate dummy data
    np.random.seed(42)
    n_stocks = 10
    n_days = 500
    
    dates = pd.date_range('2020-01-01', periods=n_days, freq='D')
    
    # Market returns
    market_returns = pd.Series(
        np.random.randn(n_days) * 0.01,
        index=dates
    )
    
    # Stock returns (correlated with market)
    stock_returns_dict = {}
    for i in range(n_stocks):
        beta = 0.8 + np.random.rand() * 0.4  # Beta in [0.8, 1.2]
        idio_vol = 0.005 + np.random.rand() * 0.01  # Idiosyncratic vol
        
        stock_ret = beta * market_returns + np.random.randn(n_days) * idio_vol
        stock_returns_dict[f'STOCK_{i:02d}'] = stock_ret
    
    stock_returns = pd.DataFrame(stock_returns_dict)
    
    print(f"Data shape:")
    print(f"  Stock returns: {stock_returns.shape}")
    print(f"  Market returns: {market_returns.shape}")
    print(f"  Date range: {dates[0]} to {dates[-1]}")
    
    # Compute characteristics
    print("\nComputing characteristics...")
    engine = CharacteristicsEngine(window_size=252, min_observations=100)
    
    chars = engine.compute_all_characteristics(
        stock_returns,
        market_returns
    )
    
    print(f"\nResults:")
    print(f"  Observations: {len(chars)}")
    print(f"  Stocks: {chars['PERMNO'].nunique()}")
    print(f"  Date range: {chars['DATE'].min()} to {chars['DATE'].max()}")
    
    print("\nCharacteristic summary statistics:")
    for col in ['BETA', 'IVOL', 'DOWNSIDE_BETA', 'COSKEW', 'COKURT']:
        values = chars[col].dropna()
        if len(values) > 0:
            print(f"  {col:15s}: mean={values.mean():7.4f}, std={values.std():7.4f}, "
                  f"min={values.min():7.4f}, max={values.max():7.4f}")
    
    # Test specific calculations
    print("\n=== Testing Individual Functions ===")
    
    # Test beta and IVOL
    r_stock = np.random.randn(252) * 0.02
    r_market = np.random.randn(252) * 0.015
    beta, ivol = engine.compute_beta_and_ivol(r_stock, r_market)
    print(f"\nBeta and IVOL test:")
    print(f"  Beta: {beta:.4f}")
    print(f"  IVOL: {ivol:.4f}")
    
    # Test downside beta
    downside_beta = engine.compute_downside_beta(r_stock, r_market)
    print(f"\nDownside Beta test:")
    print(f"  Downside Beta: {downside_beta:.4f}")
    print(f"  Regular Beta: {beta:.4f}")
    print(f"  Difference: {downside_beta - beta:.4f}")
    
    # Test coskewness
    coskew = engine.compute_coskewness(r_stock, r_market)
    print(f"\nCoskewness test:")
    print(f"  Coskewness: {coskew:.4f}")
    
    # Test cokurtosis
    cokurt = engine.compute_cokurtosis(r_stock, r_market)
    print(f"\nCokurtosis test:")
    print(f"  Cokurtosis: {cokurt:.4f}")
    
    print("\n✓ Characteristics module test complete!")
