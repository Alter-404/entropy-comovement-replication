#!/usr/bin/env python3
"""
Extension 1: Crisis Indicators
==============================

Generate binary crisis indicator variables for regime analysis.

Definitions:
    A. NBER Recession: Months within official NBER recession dates
    B. Market Crash: Months where Mkt-RF < -5%
    C. High Volatility: VIX or realized vol > 90th percentile

Output: Crisis_Flags.csv with columns [DATE, NBER, CRASH, HIGH_VOL]
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
from datetime import datetime


# NBER Recession Dates (US Business Cycle Reference Dates)
# Source: https://www.nber.org/research/data/us-business-cycle-expansions-and-contractions
NBER_RECESSIONS = [
    ('1960-04-01', '1961-02-28'),
    ('1969-12-01', '1970-11-30'),
    ('1973-11-01', '1975-03-31'),
    ('1980-01-01', '1980-07-31'),
    ('1981-07-01', '1982-11-30'),
    ('1990-07-01', '1991-03-31'),
    ('2001-03-01', '2001-11-30'),
    ('2007-12-01', '2009-06-30'),
    ('2020-02-01', '2020-04-30'),
]


def create_nber_indicator(dates: pd.DatetimeIndex) -> pd.Series:
    """
    Create NBER recession indicator.
    
    Parameters
    ----------
    dates : pd.DatetimeIndex
        Monthly dates to flag
        
    Returns
    -------
    pd.Series
        Binary indicator (1 = recession, 0 = expansion)
    """
    nber = pd.Series(0, index=dates, name='NBER')
    
    for start, end in NBER_RECESSIONS:
        start_dt = pd.Timestamp(start)
        end_dt = pd.Timestamp(end)
        mask = (dates >= start_dt) & (dates <= end_dt)
        nber.loc[mask] = 1
    
    return nber


def create_crash_indicator(mkt_rf: pd.Series, threshold: float = -0.05) -> pd.Series:
    """
    Create market crash indicator based on extreme negative returns.
    
    Parameters
    ----------
    mkt_rf : pd.Series
        Monthly market excess returns
    threshold : float
        Crash threshold (default: -5%)
        
    Returns
    -------
    pd.Series
        Binary indicator (1 = crash month, 0 = normal)
    """
    crash = (mkt_rf < threshold).astype(int)
    crash.name = 'CRASH'
    return crash


def create_volatility_indicator(
    mkt_rf: pd.Series,
    percentile: float = 90,
    window: int = 21
) -> pd.Series:
    """
    Create high volatility indicator based on realized volatility.
    
    Uses rolling 21-day (monthly) realized volatility if daily data,
    or rolling 12-month volatility for monthly data.
    
    Parameters
    ----------
    mkt_rf : pd.Series
        Market returns (daily or monthly)
    percentile : float
        Percentile threshold (default: 90th)
    window : int
        Rolling window for volatility calculation
        
    Returns
    -------
    pd.Series
        Binary indicator (1 = high vol, 0 = normal)
    """
    # Calculate rolling volatility
    vol = mkt_rf.rolling(window=window, min_periods=window//2).std() * np.sqrt(12)  # Annualized
    
    # Calculate threshold
    threshold = vol.quantile(percentile / 100)
    
    high_vol = (vol > threshold).astype(int)
    high_vol.name = 'HIGH_VOL'
    
    return high_vol


def load_market_returns(data_dir: Path) -> pd.Series:
    """Load Fama-French market returns."""
    
    # Try to load from processed cache first
    cache_file = data_dir.parent / 'cache' / 'ff_factors.parquet'
    if cache_file.exists():
        df = pd.read_parquet(cache_file)
        # Handle various column name formats
        mkt_col = None
        for col in ['MKT_RF', 'Mkt-RF', 'MKT-RF', 'mkt_rf', 'mkt-rf']:
            if col in df.columns:
                mkt_col = col
                break
        if mkt_col is not None:
            return df[mkt_col]
    
    # Try raw FF data
    ff_file = data_dir / 'F-F_Research_Data_Factors.csv'
    if ff_file.exists():
        # Try different skiprows values as format may vary
        for skiprows in [0, 3]:
            try:
                df = pd.read_csv(ff_file, skiprows=skiprows)
                
                # Check if we got the header row
                cols = list(df.columns)
                
                # Identify date column (first column usually)
                date_col = cols[0]
                df = df.rename(columns={date_col: 'date'})
                
                # Filter to numeric dates (YYYYMM format)
                df = df[df['date'].apply(lambda x: str(x).replace(' ', '').isdigit())]
                if len(df) == 0:
                    continue
                    
                df['date'] = pd.to_datetime(df['date'].astype(str).str.strip(), format='%Y%m')
                df = df.set_index('date')
                df = df.apply(pd.to_numeric, errors='coerce')
                
                # Handle various column name formats for Mkt-RF
                mkt_col = None
                for col in ['Mkt-RF', 'MKT-RF', 'MKT_RF', 'mkt-rf', 'mkt_rf']:
                    if col in df.columns:
                        mkt_col = col
                        break
                
                if mkt_col is not None:
                    # FF factors are in percentages
                    print(f"  [INFO] Loaded market returns from {ff_file} (skiprows={skiprows})")
                    return df[mkt_col] / 100
            except Exception as e:
                continue
        
        print(f"  [WARN] Could not parse market returns from {ff_file}")
        return None
    
    return None


def generate_crisis_flags(
    data_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    start_date: str = '1965-01-01',
    end_date: str = '2013-12-31',
    demo_mode: bool = False
) -> pd.DataFrame:
    """
    Generate comprehensive crisis indicator dataset.
    
    Parameters
    ----------
    data_dir : Path, optional
        Directory containing raw data
    output_dir : Path, optional
        Directory for output CSV
    start_date : str
        Sample start date
    end_date : str
        Sample end date
    demo_mode : bool
        If True, generate synthetic data
        
    Returns
    -------
    pd.DataFrame
        Crisis flags with columns [DATE, NBER, CRASH, HIGH_VOL, ANY_CRISIS]
    """
    print("\n" + "="*60)
    print("  Extension 1: Crisis Indicators")
    print("="*60)
    
    # Setup paths
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / 'data' / 'raw'
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / 'outputs' / 'tables'
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate date range
    dates = pd.date_range(start=start_date, end=end_date, freq='ME')
    print(f"\n  Sample period: {start_date} to {end_date}")
    print(f"  Total months: {len(dates)}")
    
    # Load or generate market returns
    if demo_mode:
        print("\n  [DEMO MODE] Generating synthetic market returns...")
        np.random.seed(42)
        mkt_rf = pd.Series(
            np.random.randn(len(dates)) * 0.045 + 0.006,  # Mean ~0.6%/month, vol ~4.5%
            index=dates,
            name='MKT_RF'
        )
        # Inject crisis periods
        crisis_months = [
            ('2000-03-01', '2002-10-31'),  # Dot-com
            ('2007-12-01', '2009-03-31'),  # GFC
        ]
        for start, end in crisis_months:
            mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
            mkt_rf.loc[mask] = np.random.randn(mask.sum()) * 0.08 - 0.03  # Negative bias
    else:
        mkt_rf = load_market_returns(data_dir)
        if mkt_rf is None:
            print("  [WARN] Could not load market returns, generating synthetic data")
            np.random.seed(42)
            mkt_rf = pd.Series(
                np.random.randn(len(dates)) * 0.045 + 0.006,
                index=dates,
                name='MKT_RF'
            )
    
    # Align dates - FF data uses month-start (1st), convert to month-end
    if mkt_rf is not None and len(mkt_rf) > 0:
        # Convert mkt_rf index to month-end to match dates
        mkt_rf.index = mkt_rf.index + pd.offsets.MonthEnd(0)
        # Filter to sample period and reindex
        mkt_rf = mkt_rf[mkt_rf.index >= pd.Timestamp(start_date)]
        mkt_rf = mkt_rf[mkt_rf.index <= pd.Timestamp(end_date)]
        mkt_rf = mkt_rf.reindex(dates).ffill()
    else:
        mkt_rf = pd.Series(np.nan, index=dates, name='MKT_RF')
    
    # Create indicators
    print("\n  Creating crisis indicators...")
    
    # A. NBER Recession
    nber = create_nber_indicator(dates)
    n_nber = nber.sum()
    print(f"    [A] NBER Recession months: {n_nber} ({100*n_nber/len(dates):.1f}%)")
    
    # B. Market Crash (Mkt-RF < -5%)
    crash = create_crash_indicator(mkt_rf, threshold=-0.05)
    n_crash = crash.sum()
    print(f"    [B] Market Crash months (< -5%): {n_crash} ({100*n_crash/len(dates):.1f}%)")
    
    # C. High Volatility (> 90th percentile)
    high_vol = create_volatility_indicator(mkt_rf, percentile=90, window=12)
    n_highvol = high_vol.sum()
    print(f"    [C] High Volatility months (>90th): {n_highvol} ({100*n_highvol/len(dates):.1f}%)")
    
    # Combine into DataFrame
    crisis_flags = pd.DataFrame({
        'DATE': dates,
        'NBER': nber.values,
        'CRASH': crash.values,
        'HIGH_VOL': high_vol.values,
        'MKT_RF': mkt_rf.values
    })
    
    # Add composite indicator (any crisis)
    crisis_flags['ANY_CRISIS'] = (
        (crisis_flags['NBER'] == 1) | 
        (crisis_flags['CRASH'] == 1) | 
        (crisis_flags['HIGH_VOL'] == 1)
    ).astype(int)
    
    n_any = crisis_flags['ANY_CRISIS'].sum()
    print(f"    [*] Any Crisis months: {n_any} ({100*n_any/len(dates):.1f}%)")
    
    # Save to CSV
    output_path = output_dir / 'Crisis_Flags.csv'
    crisis_flags.to_csv(output_path, index=False)
    print(f"\n  Saved: {output_path}")
    
    # Summary statistics
    print("\n  " + "-"*50)
    print("  Crisis Overlap Analysis:")
    print("  " + "-"*50)
    
    # Overlap matrix
    indicators = ['NBER', 'CRASH', 'HIGH_VOL']
    for i, ind1 in enumerate(indicators):
        for ind2 in indicators[i+1:]:
            overlap = ((crisis_flags[ind1] == 1) & (crisis_flags[ind2] == 1)).sum()
            print(f"    {ind1} & {ind2}: {overlap} months")
    
    return crisis_flags


def main():
    """Generate crisis indicators."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Crisis Indicators')
    parser.add_argument('--data-dir', type=str, help='Raw data directory')
    parser.add_argument('--output-dir', type=str, help='Output directory')
    parser.add_argument('--demo', action='store_true', help='Use demo data')
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir) if args.data_dir else None
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    crisis_flags = generate_crisis_flags(
        data_dir=data_dir,
        output_dir=output_dir,
        demo_mode=args.demo
    )
    
    print("\n  Crisis Flags Summary:")
    print(crisis_flags.describe())


if __name__ == '__main__':
    main()
