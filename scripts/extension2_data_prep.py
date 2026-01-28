#!/usr/bin/env python3
"""
Extension 2: Data Preparation for Modern Era (2014-2024)

Prepares CRSP daily and monthly data for out-of-sample testing of
the Asymmetry Risk Premium in the post-publication era.

Filters:
- Date range: Jan 1, 2014 - Dec 31, 2024
- Price >= $5 (exclude penny stocks)
- Share Code 10 or 11 (common US equities only)
- Strict PERMNO alignment between daily and monthly data
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "python"))


def load_crsp_daily(filepath: Path, start_date: str = "2014-01-01", 
                    end_date: str = "2024-12-31") -> pd.DataFrame:
    """Load and filter CRSP daily data."""
    print(f"  Loading CRSP Daily from {filepath.name}...")
    
    df = pd.read_csv(filepath, low_memory=False)
    
    # Standardize column names
    df.columns = df.columns.str.upper().str.strip()
    
    # Parse dates - try multiple formats
    date_col = 'DATE' if 'DATE' in df.columns else 'date'
    if date_col.upper() in df.columns:
        # Try parsing as-is first (handles YYYY-MM-DD and other formats)
        df['DATE'] = pd.to_datetime(df[date_col.upper()], errors='coerce')
        # If that failed, try YYYYMMDD format
        if df['DATE'].isna().all():
            df['DATE'] = pd.to_datetime(df[date_col.upper()], format='%Y%m%d', errors='coerce')
    
    # Filter date range
    df = df[(df['DATE'] >= start_date) & (df['DATE'] <= end_date)]
    
    # Filter share codes (10, 11 = common US equities)
    if 'SHRCD' in df.columns:
        df = df[df['SHRCD'].isin([10, 11])]
    
    # Filter price >= $5
    price_col = 'PRC' if 'PRC' in df.columns else 'PRICE'
    if price_col in df.columns:
        df[price_col] = pd.to_numeric(df[price_col], errors='coerce').abs()  # Handle negative prices (bid-ask midpoint)
        df = df[df[price_col] >= 5.0]
    
    # Ensure return column exists and is numeric
    ret_col = 'RET' if 'RET' in df.columns else 'RETX'
    if ret_col in df.columns:
        df['RET'] = pd.to_numeric(df[ret_col], errors='coerce')
    
    print(f"    Loaded {len(df):,} daily observations")
    print(f"    Date range: {df['DATE'].min()} to {df['DATE'].max()}")
    print(f"    Unique stocks: {df['PERMNO'].nunique():,}")
    
    return df


def load_crsp_monthly(filepath: Path, start_date: str = "2014-01-01",
                      end_date: str = "2024-12-31") -> pd.DataFrame:
    """Load and filter CRSP monthly data."""
    print(f"  Loading CRSP Monthly from {filepath.name}...")
    
    df = pd.read_csv(filepath, low_memory=False)
    
    # Standardize column names
    df.columns = df.columns.str.upper().str.strip()
    
    # Parse dates - try multiple formats
    date_col = 'DATE' if 'DATE' in df.columns else 'date'
    if date_col.upper() in df.columns:
        # Try parsing as-is first (handles YYYY-MM-DD and other formats)
        df['DATE'] = pd.to_datetime(df[date_col.upper()], errors='coerce')
        # If that failed, try YYYYMMDD format
        if df['DATE'].isna().all():
            df['DATE'] = pd.to_datetime(df[date_col.upper()], format='%Y%m%d', errors='coerce')
    
    # Filter date range
    df = df[(df['DATE'] >= start_date) & (df['DATE'] <= end_date)]
    
    # Filter share codes
    if 'SHRCD' in df.columns:
        df = df[df['SHRCD'].isin([10, 11])]
    
    # Filter price >= $5
    price_col = 'PRC' if 'PRC' in df.columns else 'PRICE'
    if price_col in df.columns:
        df[price_col] = pd.to_numeric(df[price_col], errors='coerce').abs()
        df = df[df[price_col] >= 5.0]
    
    # Ensure return column is numeric
    ret_col = 'RET' if 'RET' in df.columns else 'RETX'
    if ret_col in df.columns:
        df['RET'] = pd.to_numeric(df[ret_col], errors='coerce')
    
    # Calculate market equity
    if 'SHROUT' in df.columns and price_col in df.columns:
        df['SHROUT'] = pd.to_numeric(df['SHROUT'], errors='coerce')
        df['ME'] = df[price_col].abs() * df['SHROUT'] * 1000  # SHROUT in thousands
    
    print(f"    Loaded {len(df):,} monthly observations")
    print(f"    Unique stocks: {df['PERMNO'].nunique():,}")
    
    return df


def align_daily_monthly(daily_df: pd.DataFrame, monthly_df: pd.DataFrame) -> pd.DataFrame:
    """Ensure strict PERMNO alignment between daily and monthly data."""
    print("  Aligning daily and monthly data...")
    
    # Get PERMNOs present in both datasets
    daily_permnos = set(daily_df['PERMNO'].unique())
    monthly_permnos = set(monthly_df['PERMNO'].unique())
    common_permnos = daily_permnos & monthly_permnos
    
    print(f"    Daily PERMNOs: {len(daily_permnos):,}")
    print(f"    Monthly PERMNOs: {len(monthly_permnos):,}")
    print(f"    Common PERMNOs: {len(common_permnos):,}")
    
    # Filter to common PERMNOs
    daily_aligned = daily_df[daily_df['PERMNO'].isin(common_permnos)].copy()
    
    return daily_aligned


def format_for_entropy_engine(df: pd.DataFrame) -> pd.DataFrame:
    """Format data for C++ entropy engine input."""
    print("  Formatting for entropy engine...")
    
    # Required columns: PERMNO, DATE, RET
    output_cols = ['PERMNO', 'DATE', 'RET']
    
    # Ensure columns exist
    if 'RET' not in df.columns:
        raise ValueError("RET column not found in data")
    
    # Select and clean
    output = df[output_cols].copy()
    output = output.dropna(subset=['RET'])
    
    # Sort by PERMNO and DATE
    output = output.sort_values(['PERMNO', 'DATE'])
    
    # Format DATE as YYYYMMDD integer
    output['DATE'] = output['DATE'].dt.strftime('%Y%m%d').astype(int)
    
    print(f"    Final observations: {len(output):,}")
    print(f"    Final stocks: {output['PERMNO'].nunique():,}")
    
    return output


def generate_demo_data(output_dir: Path) -> dict:
    """Generate synthetic demo data for testing without real CRSP data."""
    print("  Generating synthetic demo data (2014-2024)...")
    
    np.random.seed(42)
    
    # Parameters (reduced for demo speed)
    n_stocks = 100  # Reduced from 500 for faster demo
    start_date = pd.Timestamp("2014-01-01")
    end_date = pd.Timestamp("2024-12-31")
    
    # Generate trading days
    trading_days = pd.bdate_range(start_date, end_date)
    n_days = len(trading_days)
    
    # Generate monthly dates
    monthly_dates = pd.date_range(start_date, end_date, freq='ME')
    n_months = len(monthly_dates)
    
    print(f"    Simulating {n_stocks} stocks over {n_days} trading days...")
    
    # Daily data
    daily_records = []
    for permno in range(10001, 10001 + n_stocks):
        # Random entry/exit
        entry_idx = np.random.randint(0, n_days // 4)
        exit_idx = np.random.randint(3 * n_days // 4, n_days)
        
        stock_days = trading_days[entry_idx:exit_idx]
        
        # Generate returns with some structure
        base_vol = np.random.uniform(0.01, 0.04)
        returns = np.random.normal(0.0003, base_vol, len(stock_days))
        
        # Add some market correlation
        market_component = np.random.normal(0, 0.01, len(stock_days))
        returns += market_component * np.random.uniform(0.5, 1.5)
        
        for date, ret in zip(stock_days, returns):
            daily_records.append({
                'PERMNO': permno,
                'DATE': date,
                'RET': ret
            })
    
    daily_df = pd.DataFrame(daily_records)
    
    # Monthly data with market equity
    monthly_records = []
    for permno in range(10001, 10001 + n_stocks):
        stock_months = monthly_dates[np.random.randint(0, 12):n_months - np.random.randint(0, 12)]
        
        base_me = np.random.lognormal(20, 2)  # Log-normal market equity
        
        for i, date in enumerate(stock_months):
            me = base_me * np.exp(np.random.normal(0, 0.1) * i / 12)
            monthly_records.append({
                'PERMNO': permno,
                'DATE': date,
                'ME': me,
                'RET': np.random.normal(0.01, 0.08)
            })
    
    monthly_df = pd.DataFrame(monthly_records)
    
    print(f"    Generated {len(daily_df):,} daily observations")
    print(f"    Generated {len(monthly_df):,} monthly observations")
    
    return {
        'daily': daily_df,
        'monthly': monthly_df
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare modern era data (2014-2024)")
    parser.add_argument("--demo", action="store_true", help="Use synthetic demo data")
    parser.add_argument("--raw-dir", type=str, default=None, 
                        help="Directory containing raw CRSP files")
    parser.add_argument("--output-dir", type=str, default="data/processed",
                        help="Output directory for processed files")
    args = parser.parse_args()
    
    print("=" * 70)
    print("EXTENSION 2: DATA PREPARATION (2014-2024)")
    print("=" * 70)
    
    project_root = Path(__file__).parent.parent
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine raw data directory
    if args.raw_dir:
        raw_dir = project_root / args.raw_dir
    else:
        # Check for extension subfolder first, then fall back to raw
        extension_dir = project_root / "data" / "raw" / "extension"
        if extension_dir.exists() and (extension_dir / "CRSP Daily Stock.csv").exists():
            raw_dir = extension_dir
            print(f"\n  Using extension data directory: {raw_dir}")
        else:
            raw_dir = project_root / "data" / "raw"
    
    if args.demo:
        print("\n[DEMO MODE] Generating synthetic data...")
        data = generate_demo_data(output_dir)
        daily_df = data['daily']
        monthly_df = data['monthly']
    else:
        print("\n[PRODUCTION MODE] Loading real CRSP data...")
        
        # Check for required files
        daily_path = raw_dir / "CRSP Daily Stock.csv"
        monthly_path = raw_dir / "CRSP Monthly Stock.csv"
        
        if not daily_path.exists():
            print(f"  ERROR: {daily_path} not found")
            print("  Use --demo flag for synthetic data")
            return 1
        
        if not monthly_path.exists():
            print(f"  ERROR: {monthly_path} not found")
            return 1
        
        # Load data
        daily_df = load_crsp_daily(daily_path)
        monthly_df = load_crsp_monthly(monthly_path)
        
        # Align PERMNOs
        daily_df = align_daily_monthly(daily_df, monthly_df)
    
    # Format for entropy engine
    entropy_input = format_for_entropy_engine(daily_df)
    
    # Save outputs
    print("\n  Saving outputs...")
    
    # Daily input for entropy engine
    daily_output_path = output_dir / "Modern_Daily_Input.csv"
    entropy_input.to_csv(daily_output_path, index=False)
    print(f"    Saved: {daily_output_path}")
    
    # Monthly data for factor construction
    monthly_output_path = output_dir / "Modern_Monthly_Input.parquet"
    monthly_df.to_parquet(monthly_output_path, index=False)
    print(f"    Saved: {monthly_output_path}")
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("DATA PREPARATION COMPLETE")
    print("=" * 70)
    print(f"  Daily observations: {len(entropy_input):,}")
    print(f"  Monthly observations: {len(monthly_df):,}")
    print(f"  Unique stocks: {entropy_input['PERMNO'].nunique():,}")
    print(f"  Date range: 2014-01-01 to 2024-12-31")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
