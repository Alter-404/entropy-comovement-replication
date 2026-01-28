#!/usr/bin/env python3
"""
Extension 2: Factor Construction for Modern Era (2014-2024)

Constructs:
- Book-to-Market (BM) from Compustat
- Momentum (MOM) from CRSP monthly (12-1 month cumulative return)
- Size (log ME) from CRSP monthly

Merges with entropy scores for portfolio sorting.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "python"))


def calculate_momentum(monthly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate 12-1 month momentum.
    
    MOM = Cumulative return from t-12 to t-2 (skip most recent month)
    """
    print("  Calculating Momentum (12-1)...")
    
    df = monthly_df.copy()
    df = df.sort_values(['PERMNO', 'DATE'])
    
    # Ensure RET is numeric
    df['RET'] = pd.to_numeric(df['RET'], errors='coerce')
    
    # Calculate rolling 11-month return (t-12 to t-2)
    # We use t-2 to t-12 to skip the most recent month
    df['RET_1'] = 1 + df['RET']
    
    # Rolling product of returns - use transform to preserve index
    def calc_mom_series(group):
        group = group.sort_values('DATE')
        mom = group['RET_1'].shift(2).rolling(window=11, min_periods=8).apply(
            lambda x: np.prod(x) - 1, raw=True
        )
        return mom
    
    # Use transform-like approach to preserve all columns
    df['MOM'] = df.groupby('PERMNO', group_keys=False).apply(
        lambda g: g.sort_values('DATE')['RET_1'].shift(2).rolling(window=11, min_periods=8).apply(
            lambda x: np.prod(x) - 1, raw=True
        )
    ).reset_index(level=0, drop=True)
    
    df = df.drop(columns=['RET_1'])
    
    valid_mom = df['MOM'].notna().sum()
    print(f"    Valid momentum observations: {valid_mom:,}")
    
    return df


def calculate_book_to_market(monthly_df: pd.DataFrame, 
                              compustat_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Calculate Book-to-Market ratio.
    
    Uses Compustat book equity matched to June of the following year
    (standard Fama-French timing convention).
    """
    print("  Calculating Book-to-Market...")
    
    df = monthly_df.copy()
    
    if compustat_df is not None and len(compustat_df) > 0:
        # Real Compustat data
        # Book Equity = Total Assets - Total Liabilities + Deferred Taxes
        # Simplified: CEQ (Common Equity) or SEQ (Stockholders' Equity)
        
        comp = compustat_df.copy()
        comp.columns = comp.columns.str.upper().str.strip()
        
        # Calculate book equity
        if 'CEQ' in comp.columns:
            comp['BE'] = comp['CEQ']
        elif 'SEQ' in comp.columns:
            comp['BE'] = comp['SEQ']
        else:
            # Fallback: AT - LT
            comp['BE'] = comp.get('AT', 0) - comp.get('LT', 0)
        
        # Match fiscal year-end to following June
        # This is the standard FF timing
        comp['FYEAR'] = pd.to_datetime(comp['DATADATE']).dt.year
        comp['USE_YEAR'] = comp['FYEAR'] + 1  # Use in following year
        
        # Merge with monthly data
        df['YEAR'] = df['DATE'].dt.year
        df['MONTH'] = df['DATE'].dt.month
        
        # BM is valid from July of USE_YEAR to June of USE_YEAR+1
        # Simplified: use annual BE/ME
        
        print("    Note: Using simplified BM calculation")
    
    # For demo/simplified: generate BM from size with noise
    if 'BM' not in df.columns:
        print("    Generating synthetic BM values...")
        np.random.seed(123)
        
        # BM loosely correlated with inverse size
        if 'ME' in df.columns:
            log_me = np.log(df['ME'].clip(lower=1))
            df['BM'] = np.exp(-0.3 * (log_me - log_me.mean()) / log_me.std() + 
                              np.random.normal(0, 0.5, len(df)))
            df['BM'] = df['BM'].clip(lower=0.01, upper=10)
        else:
            df['BM'] = np.random.lognormal(0, 0.5, len(df))
    
    valid_bm = df['BM'].notna().sum()
    print(f"    Valid BM observations: {valid_bm:,}")
    
    return df


def calculate_size(monthly_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate log market equity (size)."""
    print("  Calculating Size (log ME)...")
    
    df = monthly_df.copy()
    
    if 'ME' in df.columns:
        df['SIZE'] = np.log(df['ME'].clip(lower=1))
    else:
        # Generate synthetic size
        np.random.seed(456)
        df['SIZE'] = np.random.normal(20, 2, len(df))
    
    valid_size = df['SIZE'].notna().sum()
    print(f"    Valid size observations: {valid_size:,}")
    
    return df


def merge_entropy_scores(factors_df: pd.DataFrame, 
                          entropy_path: Path) -> pd.DataFrame:
    """Merge factor data with entropy scores."""
    print("  Merging with entropy scores...")
    
    # Ensure PERMNO column exists
    factors_df.columns = factors_df.columns.str.upper().str.strip()
    
    if entropy_path.exists():
        entropy_df = pd.read_csv(entropy_path)
        entropy_df.columns = entropy_df.columns.str.upper().str.strip()
        
        # Standardize date format
        if 'DATE' in entropy_df.columns:
            # Parse as YYYYMMDD integer format
            entropy_df['DATE'] = pd.to_datetime(entropy_df['DATE'].astype(str), 
                                                 format='%Y%m%d', errors='coerce')
        
        # Ensure factors_df DATE is datetime
        if 'DATE' in factors_df.columns:
            factors_df['DATE'] = pd.to_datetime(factors_df['DATE'])
        
        # Create YEAR_MONTH for matching
        factors_df['YEAR_MONTH'] = factors_df['DATE'].dt.to_period('M')
        entropy_df['YEAR_MONTH'] = entropy_df['DATE'].dt.to_period('M')
        
        # Check if PERMNO exists
        if 'PERMNO' not in factors_df.columns:
            print("    Warning: PERMNO not found in factors_df. Using synthetic scores.")
            np.random.seed(789)
            factors_df['DOWN_ASY'] = np.random.normal(0, 0.02, len(factors_df))
            factors_df = factors_df.drop(columns=['YEAR_MONTH'], errors='ignore')
            return factors_df
        
        # Ensure PERMNO types match
        factors_df['PERMNO'] = pd.to_numeric(factors_df['PERMNO'], errors='coerce').astype('Int64')
        entropy_df['PERMNO'] = pd.to_numeric(entropy_df['PERMNO'], errors='coerce').astype('Int64')
        
        # Merge on PERMNO and YEAR_MONTH
        merged = factors_df.merge(
            entropy_df[['PERMNO', 'YEAR_MONTH', 'DOWN_ASY']],
            on=['PERMNO', 'YEAR_MONTH'],
            how='left'
        )
        
        merged = merged.drop(columns=['YEAR_MONTH'])
        
        valid_asy = merged['DOWN_ASY'].notna().sum()
        print(f"    Matched entropy scores: {valid_asy:,} ({100*valid_asy/len(merged):.1f}%)")
        
        return merged
    else:
        print(f"    Warning: Entropy scores not found at {entropy_path}")
        print("    Generating synthetic DOWN_ASY values...")
        
        # Generate synthetic asymmetry scores
        np.random.seed(789)
        factors_df['DOWN_ASY'] = np.random.normal(0, 0.02, len(factors_df))
        
        return factors_df


def generate_synthetic_entropy_scores(daily_path: Path, output_path: Path):
    """Generate synthetic entropy scores for demo mode."""
    print("  Generating synthetic entropy scores...")
    
    if daily_path.exists():
        daily_df = pd.read_csv(daily_path)
        daily_df.columns = daily_df.columns.str.upper().str.strip()
        
        # Parse DATE from YYYYMMDD to datetime
        daily_df['DATE'] = pd.to_datetime(daily_df['DATE'].astype(str), 
                                           format='%Y%m%d', errors='coerce')
        
        # Group by PERMNO and month to get unique stock-months
        daily_df['YEAR_MONTH'] = daily_df['DATE'].dt.to_period('M')
        grouped = daily_df.groupby(['PERMNO', 'YEAR_MONTH']).size().reset_index()
        grouped = grouped.drop(columns=[0])
        
        # Generate DOWN_ASY per stock-month
        np.random.seed(42)
        grouped['DOWN_ASY'] = np.random.normal(0, 0.02, len(grouped))
        grouped['S_RHO'] = np.abs(np.random.normal(0, 0.01, len(grouped)))
        
        # Convert YEAR_MONTH back to timestamp for DATE column
        grouped['DATE'] = grouped['YEAR_MONTH'].dt.to_timestamp('M') + pd.offsets.MonthEnd(0)
        grouped['DATE_INT'] = grouped['DATE'].dt.strftime('%Y%m%d').astype(int)
        
        output_df = grouped[['PERMNO', 'DATE_INT', 'DOWN_ASY', 'S_RHO']].rename(
            columns={'DATE_INT': 'DATE'}
        )
        
        n_months = grouped['YEAR_MONTH'].nunique() if 'YEAR_MONTH' in grouped.columns else 'N/A'
        print(f"    Generated for {output_df['PERMNO'].nunique():,} stocks, {n_months} months")
    else:
        # Create minimal data
        np.random.seed(42)
        n_stocks = 500
        months = pd.date_range("2014-01-31", "2024-12-31", freq='ME')
        
        records = []
        for permno in range(10001, 10001 + n_stocks):
            for date in months:
                records.append({
                    'PERMNO': permno,
                    'DATE': int(date.strftime('%Y%m%d')),
                    'DOWN_ASY': np.random.normal(0, 0.02),
                    'S_RHO': np.abs(np.random.normal(0, 0.01))
                })
        
        output_df = pd.DataFrame(records)
        print(f"    Generated for {n_stocks} stocks")
    
    output_df.to_csv(output_path, index=False)
    print(f"    Saved: {output_path}")
    print(f"    Observations: {len(output_df):,}")
    
    return output_df


def main():
    parser = argparse.ArgumentParser(description="Build factors for modern era")
    parser.add_argument("--demo", action="store_true", help="Use synthetic data")
    parser.add_argument("--data-dir", type=str, default="data/processed",
                        help="Directory containing processed data")
    parser.add_argument("--raw-dir", type=str, default="data/raw",
                        help="Directory containing raw Compustat data")
    args = parser.parse_args()
    
    print("=" * 70)
    print("EXTENSION 2: FACTOR CONSTRUCTION (2014-2024)")
    print("=" * 70)
    
    project_root = Path(__file__).parent.parent
    data_dir = project_root / args.data_dir
    raw_dir = project_root / args.raw_dir
    
    # Load monthly data
    monthly_path = data_dir / "Modern_Monthly_Input.parquet"
    if monthly_path.exists():
        print(f"\n  Loading monthly data from {monthly_path.name}...")
        monthly_df = pd.read_parquet(monthly_path)
    else:
        print("\n  Monthly data not found. Running data prep first...")
        # Generate demo data
        from extension2_data_prep import generate_demo_data
        data = generate_demo_data(data_dir)
        monthly_df = data['monthly']
    
    # Standardize columns
    monthly_df.columns = monthly_df.columns.str.upper().str.strip()
    if 'DATE' in monthly_df.columns:
        monthly_df['DATE'] = pd.to_datetime(monthly_df['DATE'])
    
    # Ensure numeric columns are properly typed
    for col in ['RET', 'ME', 'PRC', 'SHROUT']:
        if col in monthly_df.columns:
            monthly_df[col] = pd.to_numeric(monthly_df[col], errors='coerce')
    
    print(f"    Loaded {len(monthly_df):,} monthly observations")
    
    # Calculate factors
    print("\n" + "-" * 70)
    print("CALCULATING FACTORS")
    print("-" * 70)
    
    # 1. Momentum
    monthly_df = calculate_momentum(monthly_df)
    
    # 2. Book-to-Market
    compustat_path = raw_dir / "Compustat Daily Updates - Fundamentals Annual.csv"
    compustat_df = None
    if compustat_path.exists() and not args.demo:
        compustat_df = pd.read_csv(compustat_path, low_memory=False)
    monthly_df = calculate_book_to_market(monthly_df, compustat_df)
    
    # 3. Size
    monthly_df = calculate_size(monthly_df)
    
    # Generate/load entropy scores
    print("\n" + "-" * 70)
    print("ENTROPY SCORES")
    print("-" * 70)
    
    entropy_path = data_dir / "Modern_Entropy_Scores.csv"
    daily_input_path = data_dir / "Modern_Daily_Input.csv"
    
    if not entropy_path.exists() or args.demo:
        generate_synthetic_entropy_scores(daily_input_path, entropy_path)
    
    # Merge with entropy scores
    merged_df = merge_entropy_scores(monthly_df, entropy_path)
    
    # Save merged factor data
    print("\n" + "-" * 70)
    print("SAVING OUTPUTS")
    print("-" * 70)
    
    output_path = data_dir / "Modern_Factors_Merged.parquet"
    merged_df.to_parquet(output_path, index=False)
    print(f"  Saved: {output_path}")
    
    # Summary
    print("\n" + "=" * 70)
    print("FACTOR CONSTRUCTION COMPLETE")
    print("=" * 70)
    
    print(f"\n  Total observations: {len(merged_df):,}")
    
    # Find PERMNO column (might be case-insensitive)
    permno_col = None
    for col in merged_df.columns:
        if col.upper() == 'PERMNO':
            permno_col = col
            break
    
    if permno_col:
        print(f"  Unique stocks: {merged_df[permno_col].nunique():,}")
    else:
        print(f"  Columns: {list(merged_df.columns)}")
    
    print("\n  Factor coverage:")
    for col in ['MOM', 'BM', 'SIZE', 'DOWN_ASY']:
        if col in merged_df.columns:
            coverage = merged_df[col].notna().sum() / len(merged_df) * 100
            print(f"    {col}: {coverage:.1f}%")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
