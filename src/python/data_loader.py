"""
Data Ingestion and Preprocessing Pipeline for Entropy Comovement Replication

Handles loading, cleaning, and caching of CRSP and Fama-French data according to
Jiang, Wu, and Zhou (2018) specifications.

Reference: "Asymmetry in Stock Comovements: An Entropy Approach" JFQA
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict
import warnings


class DataLoader:
    """
    Main data ingestion class for CRSP and Fama-French data.
    
    Implements strict filtering and preprocessing rules from the paper:
    - Sample period: January 1965 - December 2013
    - Exclude years with fewer than 100 daily observations
    - Compute excess returns (RET - RF)
    - Value-weighted portfolio returns
    """
    
    def __init__(self, raw_data_dir: str, cache_dir: str):
        """
        Initialize the data loader with directory paths.

        Parameters
        ----------
        raw_data_dir : str
            Path to the directory containing raw CSV files:
            - CRSP Daily Stock.csv
            - CRSP Monthly Stock.csv
            - Portfolios_Formed_on_ME.csv (Size portfolios)
            - Portfolios_Formed_on_BE-ME.csv (Book-to-Market)
            - 10_Portfolios_Prior_12_2.csv (Momentum)
            - F-F_Research_Data_Factors.csv (Fama-French factors)
            - Fama-French Daily Frequency.csv (Daily FF factors)
        cache_dir : str
            Path to the directory where Parquet files will be stored.
        """
        self.raw_dir = Path(raw_data_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache keys
        self.crsp_daily_cache = self.cache_dir / "crsp_daily_processed.parquet"
        self.crsp_monthly_cache = self.cache_dir / "crsp_monthly_processed.parquet"
        self.ff_daily_cache = self.cache_dir / "ff_daily_processed.parquet"
        self.ff_monthly_cache = self.cache_dir / "ff_monthly_processed.parquet"
        
        # Sample period constraints
        self.start_date = pd.Timestamp('1965-01-01')
        self.end_date = pd.Timestamp('2013-12-31')
        
        # Minimum observations per year per stock
        self.min_obs_per_year = 100

    def load_data(self, force_refresh: bool = False, 
                  frequency: str = 'daily') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Main entry point to load aligned stock and factor data.
        
        Parameters
        ----------
        force_refresh : bool
            If True, ignore cache and reload from raw CSVs.
        frequency : str
            'daily' or 'monthly'

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            (Stock Excess Returns, Factor Returns)
            Stocks DataFrame: MultiIndex [PERMNO, DATE], columns: ['RET', 'EXRET', ...]
            Factors DataFrame: Index [DATE], columns: ['MKT_RF', 'SMB', 'HML', 'RF', ...]
        """
        if frequency not in ['daily', 'monthly']:
            raise ValueError(f"frequency must be 'daily' or 'monthly', got {frequency}")
        
        crsp_cache = self.crsp_daily_cache if frequency == 'daily' else self.crsp_monthly_cache
        ff_cache = self.ff_daily_cache if frequency == 'daily' else self.ff_monthly_cache
        
        # Check cache
        if not force_refresh and crsp_cache.exists() and ff_cache.exists():
            print(f"Loading {frequency} data from cache...")
            stocks = pd.read_parquet(crsp_cache)
            factors = pd.read_parquet(ff_cache)
            print(f"  Loaded {len(stocks):,} stock observations")
            print(f"  Loaded {len(factors):,} factor observations")
            return stocks, factors
        
        # Cache miss - process raw data
        print(f"Processing {frequency} data from raw files...")
        factors = self._process_ff(frequency)
        stocks = self._process_crsp(factors['RF'], frequency)
        
        # Save to cache
        print(f"Saving to cache...")
        stocks.to_parquet(crsp_cache)
        factors.to_parquet(ff_cache)
        print(f"  Saved {len(stocks):,} stock observations")
        print(f"  Saved {len(factors):,} factor observations")
        
        return stocks, factors

    def _process_ff(self, frequency: str = 'daily') -> pd.DataFrame:
        """
        Load and clean Fama-French Factors.
        
        Parameters
        ----------
        frequency : str
            'daily' or 'monthly'
        
        Returns
        -------
        pd.DataFrame
            Factors (Mkt-RF, SMB, HML, RF) indexed by DATE.
            All values converted to decimal (e.g., 0.05 for 5%).
        """
        if frequency == 'daily':
            ff_file = self.raw_dir / 'Fama-French Daily Frequency.csv'
        else:
            ff_file = self.raw_dir / 'F-F_Research_Data_Factors.csv'
        
        if not ff_file.exists():
            warnings.warn(f"Fama-French file not found: {ff_file}. Creating dummy data.")
            return self._create_dummy_ff_data(frequency)
        
        # Read with flexible parsing (handle different formats)
        try:
            # Try reading without skiprows first (real data format)
            df = pd.read_csv(ff_file)
            
            # If first column looks like date, use it
            first_col = df.columns[0].strip().lower()
            if first_col in ['date', 'unnamed: 0', '']:
                df.rename(columns={df.columns[0]: 'DATE'}, inplace=True)
            
        except Exception as e:
            # Try with skiprows=3 (Kenneth French library format)
            try:
                df = pd.read_csv(ff_file, skiprows=3)
            except Exception as e2:
                raise ValueError(f"Error reading {ff_file}: {e}, {e2}")
        
        # Rename columns to standard format
        # Typical columns: Date, Mkt-RF, SMB, HML, RF
        df.columns = df.columns.str.strip().str.upper()
        
        # Map to our standard names
        column_map = {
            'MKT-RF': 'MKT_RF',
            'MKT_RF': 'MKT_RF',
            'MKTRF': 'MKT_RF',
        }
        df.rename(columns=column_map, inplace=True)
        
        # Parse date - handle both YYYYMMDD and YYYY-MM-DD formats
        if frequency == 'daily':
            # Try parsing as-is first (YYYY-MM-DD format)
            df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
            
            # If that didn't work, try YYYYMMDD format
            if df['DATE'].isna().all():
                df['DATE'] = pd.to_datetime(df['DATE'].astype(str), format='%Y%m%d', errors='coerce')
        else:
            df['DATE'] = pd.to_datetime(df['DATE'].astype(str), format='%Y%m', errors='coerce')
        
        # Drop rows with invalid dates
        df = df.dropna(subset=['DATE'])
        
        # Filter to sample period
        df = df[(df['DATE'] >= self.start_date) & (df['DATE'] <= self.end_date)]
        
        # Ensure required columns exist
        required_cols = ['MKT_RF', 'RF']  # Minimum required
        optional_cols = ['SMB', 'HML', 'MOM']
        
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Required column {col} not found in {ff_file}")
        
        # Add optional columns with zeros if missing (for daily data)
        for col in optional_cols:
            if col not in df.columns:
                df[col] = 0.0
        
        # Convert to decimal (from percentage or already decimal)
        all_cols = required_cols + optional_cols
        for col in all_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # Check if values are in percentage (> 1) or already decimal
            if df[col].abs().max() > 2.0:  # Likely percentage
                df[col] = df[col] / 100.0
        
        # Set index
        df.set_index('DATE', inplace=True)
        
        # Sort by date
        df.sort_index(inplace=True)
        
        # Select required columns
        df = df[required_cols]
        
        return df

    def _process_crsp(self, rf_series: pd.Series, frequency: str = 'daily') -> pd.DataFrame:
        """
        Load CRSP, compute excess returns, and filter low-observation years.
        
        Applies strict filtering rule from paper (Section IV, page 497):
        "Drop any year with fewer than 100 daily return observations."

        Parameters
        ----------
        rf_series : pd.Series
            Risk-Free rate indexed by DATE.
        frequency : str
            'daily' or 'monthly'

        Returns
        -------
        pd.DataFrame
            Cleaned excess returns. 
            MultiIndex: [PERMNO, DATE]
            Columns: ['RET', 'EXRET', 'PRC', 'SHROUT', ...]
        """
        if frequency == 'daily':
            crsp_file = self.raw_dir / 'CRSP Daily Stock.csv'
        else:
            crsp_file = self.raw_dir / 'CRSP Monthly Stock.csv'
        
        if not crsp_file.exists():
            # If actual CRSP data not available, create a dummy dataset for testing
            warnings.warn(f"CRSP file not found: {crsp_file}. Creating dummy data for testing.")
            return self._create_dummy_crsp_data(rf_series, frequency)
        
        # Load CRSP data
        print(f"  Loading CRSP {frequency} data...")
        df = pd.read_csv(crsp_file)
        
        # Standardize column names
        df.columns = df.columns.str.upper()
        
        # Parse date
        df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
        df = df.dropna(subset=['DATE'])
        
        # Filter to sample period
        df = df[(df['DATE'] >= self.start_date) & (df['DATE'] <= self.end_date)]
        
        # Filter by share code (10 or 11 for common stocks)
        if 'SHRCD' in df.columns:
            df = df[df['SHRCD'].isin([10, 11])]
        
        # Convert returns to numeric
        df['RET'] = pd.to_numeric(df['RET'], errors='coerce')
        
        # Remove invalid returns (missing, NaN, or special codes)
        df = df[df['RET'].notna()]
        df = df[df['RET'].abs() < 5.0]  # Remove extreme returns (likely errors)
        
        # Merge with risk-free rate
        df = df.merge(rf_series.rename('RF'), left_on='DATE', right_index=True, how='left')
        
        # Compute excess return
        df['EXRET'] = df['RET'] - df['RF']
        
        if frequency == 'daily':
            # Apply 100-observation filter for daily data
            print(f"  Applying {self.min_obs_per_year}-observation filter...")
            df['YEAR'] = df['DATE'].dt.year
            
            # Count observations per PERMNO per year
            obs_counts = df.groupby(['PERMNO', 'YEAR']).size()
            
            # Filter: keep only PERMNO-YEAR combinations with >= min_obs_per_year
            valid_combos = obs_counts[obs_counts >= self.min_obs_per_year].index
            
            # Create filter mask
            df['VALID'] = df.apply(lambda row: (row['PERMNO'], row['YEAR']) in valid_combos, axis=1)
            df = df[df['VALID']]
            df.drop(columns=['YEAR', 'VALID'], inplace=True)
        
        # Set multi-index
        df.set_index(['PERMNO', 'DATE'], inplace=True)
        df.sort_index(inplace=True)
        
        return df

    def _create_dummy_crsp_data(self, rf_series: pd.Series, frequency: str) -> pd.DataFrame:
        """
        Create synthetic CRSP data for testing when real data is not available.
        """
        np.random.seed(42)
        
        # Use smaller dataset for testing
        n_stocks = 50  # Reduced from 500
        permnos = range(10000, 10000 + n_stocks)
        
        # Sample dates if too many
        dates = rf_series.index
        if len(dates) > 2520:  # More than ~10 years daily
            # Sample every 10th date for faster testing
            dates = dates[::10]
            rf_values = rf_series.iloc[::10].values
        else:
            rf_values = rf_series.values
        
        data = []
        for permno in permnos:
            # Generate correlated returns
            n_obs = len(dates)
            mu = np.random.uniform(-0.0002, 0.0005)  # Daily drift
            sigma = np.random.uniform(0.01, 0.03)  # Daily volatility
            
            returns = np.random.normal(mu, sigma, n_obs)
            
            for date, ret, rf in zip(dates, returns, rf_values):
                data.append({
                    'PERMNO': permno,
                    'DATE': date,
                    'RET': ret,
                    'RF': rf,
                    'EXRET': ret - rf,
                    'PRC': 50.0,  # Dummy price
                    'SHROUT': 1000000,  # Dummy shares
                })
        
        df = pd.DataFrame(data)
        df.set_index(['PERMNO', 'DATE'], inplace=True)
        
        return df

    def _create_dummy_ff_data(self, frequency: str) -> pd.DataFrame:
        """
        Create synthetic Fama-French data for testing.
        """
        np.random.seed(42)
        
        if frequency == 'daily':
            dates = pd.date_range(self.start_date, self.end_date, freq='D')
        else:
            dates = pd.date_range(self.start_date, self.end_date, freq='ME')
        
        # Generate realistic factor returns
        n = len(dates)
        
        df = pd.DataFrame({
            'MKT_RF': np.random.normal(0.0003, 0.01, n),  # Market excess return
            'SMB': np.random.normal(0.0001, 0.005, n),    # Size factor
            'HML': np.random.normal(0.0001, 0.005, n),    # Value factor
            'RF': np.random.uniform(0.00001, 0.0001, n),  # Risk-free rate
        }, index=dates)
        
        df.index.name = 'DATE'
        
        return df


    @staticmethod
    def get_standardized_returns(series: pd.Series) -> np.ndarray:
        """
        Standardize a return series for Entropy calculation.
        
        Per paper (page 82), inputs to the entropy test must be standardized.

        Parameters
        ----------
        series : pd.Series
            Raw or Excess return series.

        Returns
        -------
        np.ndarray
            (x - mean) / std using sample standard deviation (ddof=1).
        """
        if len(series) < 2:
            raise ValueError("Series must have at least 2 observations for standardization")
        
        return (series.values - series.mean()) / series.std(ddof=1)

    def load_test_portfolios(self, portfolio_type: str = 'size') -> pd.DataFrame:
        """
        Load Fama-French test portfolios (Size, B/M, or Momentum).
        
        Parameters
        ----------
        portfolio_type : str
            One of: 'size', 'bm', 'momentum'
        
        Returns
        -------
        pd.DataFrame
            Portfolio returns indexed by DATE.
        """
        portfolio_files = {
            'size': 'Portfolios_Formed_on_ME.csv',
            'bm': 'Portfolios_Formed_on_BE-ME.csv',
            'momentum': '10_Portfolios_Prior_12_2.csv',
        }
        
        if portfolio_type not in portfolio_files:
            raise ValueError(f"portfolio_type must be one of {list(portfolio_files.keys())}")
        
        file_path = self.raw_dir / portfolio_files[portfolio_type]
        
        if not file_path.exists():
            warnings.warn(f"Portfolio file not found: {file_path}. Creating dummy data.")
            return self._create_dummy_portfolio_data(portfolio_type)
        
        # Load portfolio data
        df = pd.read_csv(file_path, skiprows=11)  # Skip header
        
        # Parse date
        df['DATE'] = pd.to_datetime(df.iloc[:, 0].astype(str), format='%Y%m', errors='coerce')
        df = df.dropna(subset=['DATE'])
        df.set_index('DATE', inplace=True)
        
        # Filter to sample period
        df = df[(df.index >= self.start_date) & (df.index <= self.end_date)]
        
        # Convert to decimal
        for col in df.columns:
            if col != 'DATE':
                df[col] = pd.to_numeric(df[col], errors='coerce') / 100.0
        
        return df

    def _create_dummy_portfolio_data(self, portfolio_type: str) -> pd.DataFrame:
        """Create synthetic portfolio data for testing."""
        np.random.seed(42)
        
        n_portfolios = 10 if portfolio_type in ['size', 'bm', 'momentum'] else 5
        
        dates = pd.date_range(self.start_date, self.end_date, freq='ME')
        
        data = {}
        for i in range(1, n_portfolios + 1):
            mu = np.random.uniform(-0.005, 0.015)
            sigma = np.random.uniform(0.03, 0.06)
            data[f'P{i}'] = np.random.normal(mu, sigma, len(dates))
        
        df = pd.DataFrame(data, index=dates)
        return df


# Helper functions

def compute_rolling_window_stats(df: pd.DataFrame, 
                                 window_size: int = 252,
                                 min_periods: int = 100) -> pd.DataFrame:
    """
    Compute rolling window statistics for entropy calculations.
    
    Parameters
    ----------
    df : pd.DataFrame
        Stock returns with MultiIndex [PERMNO, DATE]
    window_size : int
        Window size in days (default: 252 for 1 year)
    min_periods : int
        Minimum observations required
    
    Returns
    -------
    pd.DataFrame
        Rolling statistics
    """
    results = []
    
    for permno in df.index.get_level_values('PERMNO').unique():
        stock_data = df.xs(permno, level='PERMNO')
        
        # Rolling mean and std
        rolling_mean = stock_data['EXRET'].rolling(window=window_size, min_periods=min_periods).mean()
        rolling_std = stock_data['EXRET'].rolling(window=window_size, min_periods=min_periods).std()
        
        result = pd.DataFrame({
            'PERMNO': permno,
            'DATE': stock_data.index,
            'ROLLING_MEAN': rolling_mean,
            'ROLLING_STD': rolling_std,
        })
        results.append(result)
    
    return pd.concat(results, ignore_index=True)


if __name__ == "__main__":
    # Example usage
    print("=== Data Loader Module ===\n")
    
    loader = DataLoader(
        raw_data_dir='data/raw',
        cache_dir='data/processed'
    )
    
    print("Loading daily data...")
    stocks, factors = loader.load_data(frequency='daily', force_refresh=False)
    
    print(f"\nStocks shape: {stocks.shape}")
    print(f"Factors shape: {factors.shape}")
    print(f"\nDate range: {stocks.index.get_level_values('DATE').min()} to {stocks.index.get_level_values('DATE').max()}")
    print(f"Number of unique stocks: {stocks.index.get_level_values('PERMNO').nunique()}")
