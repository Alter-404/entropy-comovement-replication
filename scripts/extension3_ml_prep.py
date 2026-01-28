#!/usr/bin/env python3
"""
Extension 3: ML Feature Engineering & Target Generation

Prepares features and targets for XGBoost prediction of future DOWN_ASY ranks.

Features (X):
- Micro: DOWN_ASY (lag 1), IVOL, TURN, SIZE, MOM, BM
- Macro: MKT_VOL (rolling 12m std of Mkt-RF), Mkt-RF (lag 1)
- Interactions: DOWN_ASY * MKT_VOL, IVOL * TURN

Target (Y):
- Cross-sectional rank of DOWN_ASY in month t+1 (normalized to [0,1])
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "python"))


def load_panel_data(data_dir: Path, demo: bool = False) -> pd.DataFrame:
    """Load the full panel dataset with stock characteristics."""
    print("  Loading panel data...")
    
    # Try multiple possible sources
    panel_path = data_dir / "Full_Panel_Data.csv"
    alt_path = data_dir / "firm_characteristics.parquet"
    
    if panel_path.exists() and not demo:
        df = pd.read_csv(panel_path)
    elif alt_path.exists() and not demo:
        df = pd.read_parquet(alt_path)
    else:
        # Generate synthetic panel data
        print("    Generating synthetic panel data...")
        df = generate_synthetic_panel()
    
    df.columns = df.columns.str.upper().str.strip()
    
    # Standardize date column
    date_col = None
    for col in ['DATE', 'YEARMONTH', 'YM']:
        if col in df.columns:
            date_col = col
            break
    
    if date_col and date_col != 'DATE':
        df = df.rename(columns={date_col: 'DATE'})
    
    if 'DATE' in df.columns:
        df['DATE'] = pd.to_datetime(df['DATE'])
    
    print(f"    Loaded {len(df):,} observations")
    print(f"    Columns: {list(df.columns)}")
    
    return df


def generate_synthetic_panel() -> pd.DataFrame:
    """Generate synthetic panel data for demo mode."""
    np.random.seed(42)
    
    n_stocks = 200
    dates = pd.date_range("1963-01-31", "2024-12-31", freq='ME')
    n_months = len(dates)
    
    records = []
    for permno in range(10001, 10001 + n_stocks):
        # Random entry/exit
        entry = np.random.randint(0, n_months // 3)
        exit_idx = np.random.randint(2 * n_months // 3, n_months)
        
        for i, date in enumerate(dates[entry:exit_idx]):
            # Generate correlated characteristics
            size = np.random.normal(20, 2)
            bm = np.random.lognormal(0, 0.5)
            mom = np.random.normal(0.1, 0.3)
            ivol = np.random.lognormal(-3, 0.5)
            turn = np.random.lognormal(-2, 0.8)
            
            # DOWN_ASY depends on characteristics + noise
            down_asy = (0.3 * ivol + 0.2 * turn - 0.1 * size / 20 + 
                        np.random.normal(0, 0.02))
            
            # Returns depend on characteristics
            ret = 0.01 + 0.5 * down_asy + np.random.normal(0, 0.08)
            
            records.append({
                'PERMNO': permno,
                'DATE': date,
                'RET': ret,
                'DOWN_ASY': down_asy,
                'IVOL': ivol,
                'TURN': turn,
                'SIZE': size,
                'MOM': mom,
                'BM': bm,
                'ME': np.exp(size)
            })
    
    return pd.DataFrame(records)


def load_market_factors(data_dir: Path, raw_dir: Path, demo: bool = False) -> pd.DataFrame:
    """Load market factors for macro features."""
    print("  Loading market factors...")
    
    # Try multiple possible sources
    factor_path = data_dir / "Market_Factors.csv"
    ff_path = raw_dir / "Fama-French Monthly Frequency.csv"
    alt_ff = raw_dir / "F-F_Research_Data_Factors.csv"
    
    df = None
    
    if factor_path.exists() and not demo:
        df = pd.read_csv(factor_path)
    elif ff_path.exists() and not demo:
        df = pd.read_csv(ff_path)
    elif alt_ff.exists() and not demo:
        # Parse FF file with special handling for first column as date
        df = pd.read_csv(alt_ff)
        # First column is unnamed date in YYYYMM format
        first_col = df.columns[0]
        df = df.rename(columns={first_col: 'DATE_RAW'})
        # Filter to valid numeric dates
        df = df[df['DATE_RAW'].apply(lambda x: str(x).replace(' ', '').isdigit())]
        df['DATE'] = pd.to_datetime(df['DATE_RAW'].astype(str).str.strip(), format='%Y%m')
        df = df.drop(columns=['DATE_RAW'])
    
    if df is None:
        # Generate synthetic factors
        print("    Generating synthetic market factors...")
        dates = pd.date_range("1963-01-31", "2024-12-31", freq='ME')
        
        np.random.seed(123)
        df = pd.DataFrame({
            'DATE': dates,
            'MKT_RF': np.random.normal(0.008, 0.045, len(dates)),
            'SMB': np.random.normal(0.002, 0.03, len(dates)),
            'HML': np.random.normal(0.003, 0.03, len(dates)),
            'RF': np.random.uniform(0.001, 0.005, len(dates))
        })
        
        # Add market crashes
        crash_periods = [
            ('1987-10-01', '1987-10-31', -0.20),
            ('2000-03-01', '2002-10-31', -0.03),
            ('2008-09-01', '2009-03-31', -0.08),
            ('2020-02-01', '2020-03-31', -0.12)
        ]
        for start, end, shock in crash_periods:
            mask = (df['DATE'] >= start) & (df['DATE'] <= end)
            df.loc[mask, 'MKT_RF'] += shock
    
    df.columns = df.columns.str.upper().str.strip()
    
    # Rename columns if needed
    rename_map = {'MKT-RF': 'MKT_RF', 'MKTRF': 'MKT_RF'}
    df = df.rename(columns=rename_map)
    
    # Standardize date
    if 'DATE' in df.columns:
        df['DATE'] = pd.to_datetime(df['DATE'])
    
    # Scale if in percentage
    for col in ['MKT_RF', 'SMB', 'HML', 'RF']:
        if col in df.columns and df[col].abs().max() > 1:
            df[col] = df[col] / 100
    
    print(f"    Loaded {len(df)} months of factors")
    
    return df


def compute_macro_features(factors_df: pd.DataFrame) -> pd.DataFrame:
    """Compute macro features: MKT_VOL and lagged Mkt-RF."""
    print("  Computing macro features...")
    
    df = factors_df.copy()
    df = df.sort_values('DATE')
    
    # Rolling 12-month market volatility
    if 'MKT_RF' in df.columns:
        df['MKT_VOL'] = df['MKT_RF'].rolling(window=12, min_periods=6).std()
        df['MKT_RF_LAG1'] = df['MKT_RF'].shift(1)
    else:
        df['MKT_VOL'] = 0.04  # Default
        df['MKT_RF_LAG1'] = 0.008
    
    print(f"    MKT_VOL range: {df['MKT_VOL'].min():.4f} to {df['MKT_VOL'].max():.4f}")
    
    return df


def compute_features_and_target(panel_df: pd.DataFrame, 
                                 factors_df: pd.DataFrame) -> pd.DataFrame:
    """Merge data and compute all features and target."""
    print("  Computing features and target...")
    
    # Merge panel with factors
    panel_df['YEAR_MONTH'] = panel_df['DATE'].dt.to_period('M')
    factors_df['YEAR_MONTH'] = factors_df['DATE'].dt.to_period('M')
    
    df = panel_df.merge(
        factors_df[['YEAR_MONTH', 'MKT_VOL', 'MKT_RF_LAG1']],
        on='YEAR_MONTH',
        how='left'
    )
    
    # Compute DOWN_ASY from LQP and UQP if not present
    if 'DOWN_ASY' not in df.columns:
        if 'LQP' in df.columns and 'UQP' in df.columns:
            # DOWN_ASY = LQP - UQP (downside asymmetry measure)
            df['DOWN_ASY'] = df['LQP'] - df['UQP']
            print(f"    Computed DOWN_ASY from LQP - UQP")
        else:
            # Generate synthetic DOWN_ASY
            np.random.seed(42)
            df['DOWN_ASY'] = np.random.normal(0, 0.02, len(df))
            print(f"    Generated synthetic DOWN_ASY")
    
    # Lag features by 1 month (predict t+1 using t)
    df = df.sort_values(['PERMNO', 'DATE'])
    
    # Lag DOWN_ASY
    df['DOWN_ASY_LAG1'] = df.groupby('PERMNO')['DOWN_ASY'].shift(1)
    
    # Compute interaction terms
    df['DOWN_ASY_X_MKT_VOL'] = df['DOWN_ASY_LAG1'] * df['MKT_VOL']
    
    if 'IVOL' in df.columns and 'TURN' in df.columns:
        df['IVOL_X_TURN'] = df['IVOL'] * df['TURN']
    else:
        df['IVOL_X_TURN'] = 0
    
    # Compute target: Cross-sectional rank of DOWN_ASY (normalized to [0,1])
    def compute_rank(group):
        if len(group) < 10:
            return pd.Series([np.nan] * len(group), index=group.index)
        ranks = group['DOWN_ASY'].rank(method='average')
        # Normalize to [0, 1]
        return (ranks - 1) / (len(group) - 1)
    
    df['TARGET_RANK'] = df.groupby('YEAR_MONTH')['DOWN_ASY'].transform(
        lambda x: (x.rank(method='average') - 1) / (len(x) - 1) if len(x) > 1 else np.nan
    )
    
    # Shift target to align: predict t+1 rank using t features
    df['TARGET_RANK_NEXT'] = df.groupby('PERMNO')['TARGET_RANK'].shift(-1)
    
    print(f"    Features computed for {len(df):,} observations")
    
    return df


def prepare_ml_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare final ML dataset with feature selection."""
    print("  Preparing final ML dataset...")
    
    # Define feature columns
    feature_cols = [
        'DOWN_ASY_LAG1',  # Past asymmetry
        'MKT_VOL',        # Market volatility
        'MKT_RF_LAG1',    # Past market return
        'DOWN_ASY_X_MKT_VOL',  # Interaction
    ]
    
    # Add optional micro features if available
    optional_cols = ['IVOL', 'TURN', 'SIZE', 'MOM', 'BM', 'IVOL_X_TURN']
    for col in optional_cols:
        if col in df.columns:
            feature_cols.append(col)
    
    # Select columns
    keep_cols = ['PERMNO', 'DATE', 'YEAR_MONTH', 'TARGET_RANK_NEXT', 'TARGET_RANK', 'RET'] + feature_cols
    keep_cols = [c for c in keep_cols if c in df.columns]
    
    ml_df = df[keep_cols].copy()
    
    # Drop rows with missing target
    before = len(ml_df)
    ml_df = ml_df.dropna(subset=['TARGET_RANK_NEXT'])
    after = len(ml_df)
    
    print(f"    Dropped {before - after:,} rows with missing target")
    print(f"    Final dataset: {after:,} observations")
    print(f"    Features: {[c for c in feature_cols if c in ml_df.columns]}")
    
    return ml_df


def main():
    parser = argparse.ArgumentParser(description="ML Feature Engineering")
    parser.add_argument("--demo", action="store_true", help="Use synthetic data")
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--raw-dir", type=str, default="data/raw")
    args = parser.parse_args()
    
    print("=" * 70)
    print("EXTENSION 3: ML FEATURE ENGINEERING")
    print("=" * 70)
    
    project_root = Path(__file__).parent.parent
    data_dir = project_root / args.data_dir
    raw_dir = project_root / args.raw_dir
    
    # Load data
    print("\n" + "-" * 70)
    print("LOADING DATA")
    print("-" * 70)
    
    panel_df = load_panel_data(data_dir, demo=args.demo)
    factors_df = load_market_factors(data_dir, raw_dir, demo=args.demo)
    
    # Compute macro features
    factors_df = compute_macro_features(factors_df)
    
    # Merge and compute features
    print("\n" + "-" * 70)
    print("FEATURE ENGINEERING")
    print("-" * 70)
    
    merged_df = compute_features_and_target(panel_df, factors_df)
    ml_df = prepare_ml_dataset(merged_df)
    
    # Save
    print("\n" + "-" * 70)
    print("SAVING OUTPUTS")
    print("-" * 70)
    
    output_path = data_dir / "ML_Features_Target.parquet"
    ml_df.to_parquet(output_path, index=False)
    print(f"  Saved: {output_path}")
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("ML DATASET SUMMARY")
    print("=" * 70)
    
    print(f"\n  Observations: {len(ml_df):,}")
    print(f"  Unique stocks: {ml_df['PERMNO'].nunique():,}")
    print(f"  Date range: {ml_df['DATE'].min()} to {ml_df['DATE'].max()}")
    
    print("\n  Feature correlations with target:")
    feature_cols = [c for c in ml_df.columns if c not in 
                    ['PERMNO', 'DATE', 'YEAR_MONTH', 'TARGET_RANK_NEXT', 'TARGET_RANK', 'RET']]
    for col in feature_cols:
        if col in ml_df.columns:
            corr = ml_df[col].corr(ml_df['TARGET_RANK_NEXT'])
            print(f"    {col}: {corr:.4f}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
