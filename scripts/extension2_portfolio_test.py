#!/usr/bin/env python3
"""
Extension 2: Portfolio Sorting & Alpha Testing (2014-2024)

Performs:
1. Monthly decile sorts on DOWN_ASY
2. Value-weighted portfolio returns
3. CAPM and Fama-French 5-Factor alpha estimation
4. Sub-period analysis (Pre-Covid vs Post-Covid)

Outputs Table 9: Modern Era Performance
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "python"))


def load_ff_factors(raw_dir: Path, demo: bool = False) -> pd.DataFrame:
    """Load Fama-French 5 factors."""
    print("  Loading Fama-French factors...")
    
    if demo:
        # Generate synthetic factors
        np.random.seed(42)
        dates = pd.date_range("2014-01-31", "2024-12-31", freq='ME')
        
        factors = pd.DataFrame({
            'DATE': dates,
            'MKT_RF': np.random.normal(0.007, 0.04, len(dates)),
            'SMB': np.random.normal(0.002, 0.03, len(dates)),
            'HML': np.random.normal(0.003, 0.03, len(dates)),
            'RMW': np.random.normal(0.003, 0.02, len(dates)),
            'CMA': np.random.normal(0.002, 0.02, len(dates)),
            'RF': np.random.uniform(0.0001, 0.004, len(dates))
        })
        print(f"    Generated {len(factors)} months of synthetic factors")
        return factors
    
    # Try to load real data
    ff_path = raw_dir / "Fama-French Monthly Frequency.csv"
    if not ff_path.exists():
        ff_path = raw_dir / "F-F_Research_Data_5_Factors_2x3.csv"
    
    if ff_path.exists():
        df = pd.read_csv(ff_path, low_memory=False)
        df.columns = df.columns.str.upper().str.strip()
        
        # Parse date
        if 'DATE' in df.columns:
            df['DATE'] = pd.to_datetime(df['DATE'].astype(str), format='%Y%m', errors='coerce')
        
        # Rename columns if needed
        rename_map = {
            'MKT-RF': 'MKT_RF',
            'MKTRF': 'MKT_RF'
        }
        df = df.rename(columns=rename_map)
        
        # Scale if in percentage
        for col in ['MKT_RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']:
            if col in df.columns and df[col].abs().max() > 1:
                df[col] = df[col] / 100
        
        # Filter to modern era
        df = df[(df['DATE'] >= '2014-01-01') & (df['DATE'] <= '2024-12-31')]
        
        print(f"    Loaded {len(df)} months of FF factors")
        return df
    
    # Fallback to demo
    print("    FF factors not found, using synthetic data")
    return load_ff_factors(raw_dir, demo=True)


def sort_portfolios(merged_df: pd.DataFrame, n_portfolios: int = 10) -> pd.DataFrame:
    """Sort stocks into deciles based on DOWN_ASY each month."""
    print(f"  Sorting into {n_portfolios} portfolios...")
    
    df = merged_df.copy()
    df.columns = df.columns.str.upper().str.strip()
    df = df.dropna(subset=['DOWN_ASY', 'RET'])
    
    if 'DATE' not in df.columns:
        raise ValueError("DATE column not found")
    
    df['DATE'] = pd.to_datetime(df['DATE'])
    df['YEAR_MONTH'] = df['DATE'].dt.to_period('M')
    
    # Assign deciles within each month using transform instead of apply
    def assign_decile_for_month(asy_values):
        if len(asy_values) < n_portfolios:
            return pd.Series([np.nan] * len(asy_values), index=asy_values.index)
        return pd.qcut(
            asy_values.rank(method='first'), 
            n_portfolios, 
            labels=range(1, n_portfolios + 1)
        )
    
    df['DECILE'] = df.groupby('YEAR_MONTH')['DOWN_ASY'].transform(assign_decile_for_month)
    df = df.dropna(subset=['DECILE'])
    df['DECILE'] = df['DECILE'].astype(int)
    
    print(f"    Assigned {len(df):,} stock-months to deciles")
    
    return df


def calculate_portfolio_returns(sorted_df: pd.DataFrame, 
                                  value_weighted: bool = True) -> pd.DataFrame:
    """Calculate portfolio returns for each decile."""
    print(f"  Calculating {'value-weighted' if value_weighted else 'equal-weighted'} returns...")
    
    df = sorted_df.copy()
    
    if value_weighted and 'ME' in df.columns:
        # Value-weighted
        df['WEIGHT'] = df.groupby(['YEAR_MONTH', 'DECILE'])['ME'].transform(
            lambda x: x / x.sum()
        )
        df['WEIGHTED_RET'] = df['RET'] * df['WEIGHT']
        
        portfolio_rets = df.groupby(['YEAR_MONTH', 'DECILE'])['WEIGHTED_RET'].sum()
    else:
        # Equal-weighted
        portfolio_rets = df.groupby(['YEAR_MONTH', 'DECILE'])['RET'].mean()
    
    # Pivot to wide format
    portfolio_rets = portfolio_rets.unstack()
    portfolio_rets.columns = [f'D{int(c)}' for c in portfolio_rets.columns]
    
    # Calculate High-Low spread
    portfolio_rets['HIGH_LOW'] = portfolio_rets['D10'] - portfolio_rets['D1']
    
    # Convert index to datetime
    portfolio_rets.index = portfolio_rets.index.to_timestamp('M')
    portfolio_rets = portfolio_rets.reset_index()
    portfolio_rets = portfolio_rets.rename(columns={'YEAR_MONTH': 'DATE'})
    
    print(f"    Calculated returns for {len(portfolio_rets)} months")
    
    return portfolio_rets


def run_regressions(portfolio_rets: pd.DataFrame, 
                    ff_factors: pd.DataFrame) -> dict:
    """Run CAPM and FF5 regressions on the High-Low spread."""
    print("  Running factor regressions...")
    
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools import add_constant
    
    # Merge with factors
    portfolio_rets['YEAR_MONTH'] = pd.to_datetime(portfolio_rets['DATE']).dt.to_period('M')
    ff_factors['YEAR_MONTH'] = pd.to_datetime(ff_factors['DATE']).dt.to_period('M')
    
    merged = portfolio_rets.merge(ff_factors, on='YEAR_MONTH', how='inner', suffixes=('', '_FF'))
    
    results = {}
    
    # NOTE: HIGH_LOW is already a long-short spread (D10 - D1), which is a self-financing
    # portfolio. It does NOT need RF subtracted - it's already an excess return by construction.
    # Subtracting RF would double-count and give incorrect (too negative) results.
    y = merged['HIGH_LOW'].values
    
    # 1. Raw return
    results['raw_return'] = {
        'mean': np.mean(y) * 100,
        'std': np.std(y) * 100,
        't_stat': np.mean(y) / (np.std(y) / np.sqrt(len(y)))
    }
    
    # 2. CAPM Alpha
    if 'MKT_RF' in merged.columns:
        X_capm = add_constant(merged['MKT_RF'].values)
        model_capm = OLS(y, X_capm).fit(cov_type='HAC', cov_kwds={'maxlags': 12})
        results['capm_alpha'] = {
            'alpha': model_capm.params[0] * 100,
            't_stat': model_capm.tvalues[0],
            'beta': model_capm.params[1],
            'r2': model_capm.rsquared
        }
    
    # 3. FF5 Alpha
    ff5_cols = ['MKT_RF', 'SMB', 'HML', 'RMW', 'CMA']
    available_cols = [c for c in ff5_cols if c in merged.columns]
    
    if len(available_cols) >= 3:
        X_ff5 = add_constant(merged[available_cols].values)
        model_ff5 = OLS(y, X_ff5).fit(cov_type='HAC', cov_kwds={'maxlags': 12})
        results['ff5_alpha'] = {
            'alpha': model_ff5.params[0] * 100,
            't_stat': model_ff5.tvalues[0],
            'betas': dict(zip(available_cols, model_ff5.params[1:])),
            'r2': model_ff5.rsquared
        }
    
    return results


def run_subperiod_analysis(portfolio_rets: pd.DataFrame,
                           ff_factors: pd.DataFrame) -> dict:
    """Run analysis for Pre-Covid and Post-Covid subperiods."""
    print("  Running sub-period analysis...")
    
    results = {}
    
    # Pre-Covid: 2014-01 to 2019-12
    pre_covid = portfolio_rets[
        (portfolio_rets['DATE'] >= '2014-01-01') & 
        (portfolio_rets['DATE'] <= '2019-12-31')
    ]
    ff_pre = ff_factors[
        (ff_factors['DATE'] >= '2014-01-01') & 
        (ff_factors['DATE'] <= '2019-12-31')
    ]
    
    if len(pre_covid) > 12:
        results['pre_covid'] = run_regressions(pre_covid.copy(), ff_pre.copy())
        print(f"    Pre-Covid: {len(pre_covid)} months")
    
    # Post-Covid: 2020-01 to 2024-12
    post_covid = portfolio_rets[
        (portfolio_rets['DATE'] >= '2020-01-01') & 
        (portfolio_rets['DATE'] <= '2024-12-31')
    ]
    ff_post = ff_factors[
        (ff_factors['DATE'] >= '2020-01-01') & 
        (ff_factors['DATE'] <= '2024-12-31')
    ]
    
    if len(post_covid) > 12:
        results['post_covid'] = run_regressions(post_covid.copy(), ff_post.copy())
        print(f"    Post-Covid: {len(post_covid)} months")
    
    return results


def format_table9(full_results: dict, subperiod_results: dict) -> pd.DataFrame:
    """Format results as Table 9."""
    
    rows = []
    
    # Full period
    rows.append({
        'Period': 'Full Sample (2014-2024)',
        'Raw Return (%)': f"{full_results['raw_return']['mean']:.2f}",
        'Raw t-stat': f"{full_results['raw_return']['t_stat']:.2f}",
        'CAPM Alpha (%)': f"{full_results.get('capm_alpha', {}).get('alpha', 0):.2f}",
        'CAPM t-stat': f"{full_results.get('capm_alpha', {}).get('t_stat', 0):.2f}",
        'FF5 Alpha (%)': f"{full_results.get('ff5_alpha', {}).get('alpha', 0):.2f}",
        'FF5 t-stat': f"{full_results.get('ff5_alpha', {}).get('t_stat', 0):.2f}"
    })
    
    # Pre-Covid
    if 'pre_covid' in subperiod_results:
        pre = subperiod_results['pre_covid']
        rows.append({
            'Period': 'Pre-Covid (2014-2019)',
            'Raw Return (%)': f"{pre['raw_return']['mean']:.2f}",
            'Raw t-stat': f"{pre['raw_return']['t_stat']:.2f}",
            'CAPM Alpha (%)': f"{pre.get('capm_alpha', {}).get('alpha', 0):.2f}",
            'CAPM t-stat': f"{pre.get('capm_alpha', {}).get('t_stat', 0):.2f}",
            'FF5 Alpha (%)': f"{pre.get('ff5_alpha', {}).get('alpha', 0):.2f}",
            'FF5 t-stat': f"{pre.get('ff5_alpha', {}).get('t_stat', 0):.2f}"
        })
    
    # Post-Covid
    if 'post_covid' in subperiod_results:
        post = subperiod_results['post_covid']
        rows.append({
            'Period': 'Post-Covid (2020-2024)',
            'Raw Return (%)': f"{post['raw_return']['mean']:.2f}",
            'Raw t-stat': f"{post['raw_return']['t_stat']:.2f}",
            'CAPM Alpha (%)': f"{post.get('capm_alpha', {}).get('alpha', 0):.2f}",
            'CAPM t-stat': f"{post.get('capm_alpha', {}).get('t_stat', 0):.2f}",
            'FF5 Alpha (%)': f"{post.get('ff5_alpha', {}).get('alpha', 0):.2f}",
            'FF5 t-stat': f"{post.get('ff5_alpha', {}).get('t_stat', 0):.2f}"
        })
    
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Portfolio sorting and alpha testing")
    parser.add_argument("--demo", action="store_true", help="Use demo data")
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--raw-dir", type=str, default="data/raw")
    parser.add_argument("--output-dir", type=str, default="outputs/tables")
    args = parser.parse_args()
    
    print("=" * 70)
    print("EXTENSION 2: PORTFOLIO TESTING (2014-2024)")
    print("=" * 70)
    
    project_root = Path(__file__).parent.parent
    data_dir = project_root / args.data_dir
    raw_dir = project_root / args.raw_dir
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load merged factor data
    merged_path = data_dir / "Modern_Factors_Merged.parquet"
    
    if merged_path.exists():
        print(f"\n  Loading merged data from {merged_path.name}...")
        merged_df = pd.read_parquet(merged_path)
    else:
        print("\n  Merged data not found. Please run extension2_build_factors.py first.")
        return 1
    
    # Load FF factors
    ff_factors = load_ff_factors(raw_dir, demo=args.demo)
    
    # Sort portfolios
    print("\n" + "-" * 70)
    print("PORTFOLIO CONSTRUCTION")
    print("-" * 70)
    
    sorted_df = sort_portfolios(merged_df)
    portfolio_rets = calculate_portfolio_returns(sorted_df, value_weighted=True)
    
    # Save portfolio returns
    port_ret_path = data_dir / "Modern_Portfolio_Returns.csv"
    portfolio_rets.to_csv(port_ret_path, index=False)
    print(f"\n  Saved portfolio returns: {port_ret_path}")
    
    # Run regressions
    print("\n" + "-" * 70)
    print("ALPHA ESTIMATION")
    print("-" * 70)
    
    full_results = run_regressions(portfolio_rets.copy(), ff_factors.copy())
    subperiod_results = run_subperiod_analysis(portfolio_rets.copy(), ff_factors.copy())
    
    # Format and save Table 9
    print("\n" + "-" * 70)
    print("TABLE 9: MODERN ERA PERFORMANCE")
    print("-" * 70)
    
    table9 = format_table9(full_results, subperiod_results)
    
    print("\n" + table9.to_string(index=False))
    
    # Save table
    table9_path = output_dir / "Table_9_Modern_Performance.csv"
    table9.to_csv(table9_path, index=False)
    print(f"\n  Saved: {table9_path}")
    
    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    
    full_raw = full_results['raw_return']['mean']
    full_t = full_results['raw_return']['t_stat']
    
    if full_t > 2.0:
        print(f"\n  The High-Low spread of {full_raw:.2f}% per month is STATISTICALLY SIGNIFICANT")
        print("  (t = {:.2f} > 2.0)".format(full_t))
        print("\n  CONCLUSION: The Asymmetry Risk Premium PERSISTS in the modern era.")
    elif full_t > 1.65:
        print(f"\n  The High-Low spread of {full_raw:.2f}% per month is MARGINALLY SIGNIFICANT")
        print("  (t = {:.2f}, between 1.65 and 2.0)".format(full_t))
        print("\n  CONCLUSION: The premium shows WEAK persistence post-publication.")
    else:
        print(f"\n  The High-Low spread of {full_raw:.2f}% per month is NOT SIGNIFICANT")
        print("  (t = {:.2f} < 1.65)".format(full_t))
        print("\n  CONCLUSION: The premium has DECAYED after publication.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
