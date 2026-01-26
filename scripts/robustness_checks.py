#!/usr/bin/env python3
"""
scripts/robustness_checks.py
Run all robustness checks for the entropy-comovement replication.

This script performs:
1. Subperiod analysis (pre/post 2000, crisis periods)
2. Market regime analysis (bull/bear markets)
3. Alternative portfolio sorts (terciles, quartiles, deciles)
4. Fama-MacBeth cross-sectional regressions
5. Bootstrap inference for confidence intervals

Output: Multiple CSV tables in outputs/tables/

Usage:
    python scripts/robustness_checks.py
    python scripts/robustness_checks.py --demo  # Quick demo with synthetic data

Author: Entropy Replication Project
"""

import argparse
import os
import sys
from pathlib import Path
import time

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src' / 'python'))

from robustness import (
    RobustnessRunner,
    SubperiodAnalyzer,
    AlternativeSortTester,
    FamaMacBethRegressor,
    BootstrapInference,
    SortMethod,
)


def load_data(data_dir: str) -> tuple:
    """
    Load data for robustness checks.
    
    Parameters
    ----------
    data_dir : str
        Path to processed data directory.
        
    Returns
    -------
    tuple
        (down_asy, returns, characteristics, market_returns, factors)
    """
    data_path = Path(data_dir)
    
    # Load DOWN_ASY scores
    down_asy_path = data_path / 'down_asy_scores.parquet'
    if down_asy_path.exists():
        down_asy = pd.read_parquet(down_asy_path)
        print(f"Loaded DOWN_ASY scores: {len(down_asy):,} observations")
    else:
        print(f"Warning: {down_asy_path} not found")
        down_asy = None
    
    # Load monthly returns
    returns_path = data_path / 'monthly_returns.parquet'
    if returns_path.exists():
        returns = pd.read_parquet(returns_path)
        print(f"Loaded monthly returns: {len(returns):,} observations")
    else:
        print(f"Warning: {returns_path} not found")
        returns = None
    
    # Load characteristics
    char_path = data_path / 'firm_characteristics.parquet'
    if char_path.exists():
        characteristics = pd.read_parquet(char_path)
        print(f"Loaded characteristics: {len(characteristics):,} observations")
    else:
        char_path = data_path / 'characteristics.parquet'
        if char_path.exists():
            characteristics = pd.read_parquet(char_path)
            print(f"Loaded characteristics: {len(characteristics):,} observations")
        else:
            print("Warning: characteristics file not found")
            characteristics = None
    
    # Load Fama-French factors
    factors = None
    factors_path = data_path / 'ff_factors.parquet'
    if factors_path.exists():
        factors = pd.read_parquet(factors_path)
        print(f"Loaded FF factors: {len(factors):,} observations")
        
        # Extract market returns
        if 'MKT_RF' in factors.columns:
            market_returns = factors.set_index('DATE')['MKT_RF']
        elif 'Mkt-RF' in factors.columns:
            market_returns = factors.set_index('DATE')['Mkt-RF']
        else:
            market_returns = None
    else:
        market_returns = None
    
    return down_asy, returns, characteristics, market_returns, factors


def run_subperiod_analysis(
    down_asy: pd.DataFrame,
    returns: pd.DataFrame,
    market_returns: pd.Series = None,
    output_dir: str = 'outputs/tables'
) -> pd.DataFrame:
    """
    Run subperiod analysis and save results.
    
    Parameters
    ----------
    down_asy : pd.DataFrame
        DOWN_ASY scores.
    returns : pd.DataFrame
        Monthly returns.
    market_returns : pd.Series, optional
        Market returns for regime analysis.
    output_dir : str
        Output directory.
        
    Returns
    -------
    pd.DataFrame
        Subperiod analysis results.
    """
    print("\n" + "=" * 60)
    print("Subperiod Analysis")
    print("=" * 60)
    
    analyzer = SubperiodAnalyzer()
    results = analyzer.run_subperiod_analysis(down_asy, returns, market_returns)
    
    if len(results) > 0:
        # Save results
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'robustness_subperiod.csv')
        results.to_csv(output_path, index=False)
        print(f"Saved: {output_path}")
        
        # Display summary
        print("\nResults:")
        print("-" * 60)
        display_cols = ['test_name', 'estimate', 't_statistic', 'p_value', 'n_observations']
        available_cols = [c for c in display_cols if c in results.columns]
        print(results[available_cols].to_string(index=False))
        
        # Highlight significant results
        sig_mask = results['p_value'] < 0.05
        print(f"\nSignificant at 5% level: {sig_mask.sum()} / {len(results)}")
    else:
        print("No subperiod results generated.")
    
    return results


def run_alternative_sorts(
    down_asy: pd.DataFrame,
    returns: pd.DataFrame,
    output_dir: str = 'outputs/tables'
) -> pd.DataFrame:
    """
    Run alternative portfolio sorts and save results.
    
    Parameters
    ----------
    down_asy : pd.DataFrame
        DOWN_ASY scores.
    returns : pd.DataFrame
        Monthly returns.
    output_dir : str
        Output directory.
        
    Returns
    -------
    pd.DataFrame
        Alternative sort results.
    """
    print("\n" + "=" * 60)
    print("Alternative Portfolio Sorts")
    print("=" * 60)
    
    tester = AlternativeSortTester()
    results = tester.run_alternative_sorts(down_asy, returns)
    
    if len(results) > 0:
        # Save results
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'robustness_alternative_sorts.csv')
        results.to_csv(output_path, index=False)
        print(f"Saved: {output_path}")
        
        # Display summary
        print("\nResults:")
        print("-" * 60)
        display_cols = ['method', 'n_groups', 'estimate', 't_statistic', 'p_value']
        available_cols = [c for c in display_cols if c in results.columns]
        print(results[available_cols].to_string(index=False))
        
        # Check consistency
        print("\nConsistency check:")
        for _, row in results.iterrows():
            sig_marker = "***" if row['p_value'] < 0.01 else "**" if row['p_value'] < 0.05 else "*" if row['p_value'] < 0.10 else ""
            print(f"  {row['method']}: spread = {row['estimate']:.4f}, t = {row['t_statistic']:.2f} {sig_marker}")
    else:
        print("No alternative sort results generated.")
    
    return results


def run_fama_macbeth(
    down_asy: pd.DataFrame,
    returns: pd.DataFrame,
    characteristics: pd.DataFrame = None,
    output_dir: str = 'outputs/tables'
) -> pd.DataFrame:
    """
    Run Fama-MacBeth regressions and save results.
    
    Parameters
    ----------
    down_asy : pd.DataFrame
        DOWN_ASY scores.
    returns : pd.DataFrame
        Monthly returns.
    characteristics : pd.DataFrame, optional
        Firm characteristics.
    output_dir : str
        Output directory.
        
    Returns
    -------
    pd.DataFrame
        Fama-MacBeth regression results.
    """
    print("\n" + "=" * 60)
    print("Fama-MacBeth Cross-Sectional Regressions")
    print("=" * 60)
    
    regressor = FamaMacBethRegressor(n_lags=12)
    
    # Define control variables
    control_vars = None
    if characteristics is not None:
        potential_controls = ['BETA', 'IVOL', 'DOWNSIDE_BETA', 'COSKEW', 'COKURT', 'ILLIQ', 'SIZE', 'BM']
        control_vars = [c for c in potential_controls if c in characteristics.columns]
        print(f"Using control variables: {control_vars}")
    
    results = regressor.run_fama_macbeth(
        returns, down_asy, characteristics, control_vars
    )
    
    if len(results) > 0:
        # Save results
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'robustness_fama_macbeth.csv')
        results.to_csv(output_path, index=False)
        print(f"Saved: {output_path}")
        
        # Display summary
        print("\nResults:")
        print("-" * 60)
        display_cols = ['model', 'variable', 'estimate', 't_statistic', 'p_value', 'n_observations']
        available_cols = [c for c in display_cols if c in results.columns]
        print(results[available_cols].to_string(index=False))
        
        # Focus on DOWN_ASY coefficient
        down_asy_results = results[results['variable'] == 'DOWN_ASY']
        if len(down_asy_results) > 0:
            print("\nDOWN_ASY coefficient across specifications:")
            for _, row in down_asy_results.iterrows():
                sig_marker = "***" if row['p_value'] < 0.01 else "**" if row['p_value'] < 0.05 else "*" if row['p_value'] < 0.10 else ""
                print(f"  {row.get('model', 'N/A')}: γ = {row['estimate']:.4f}, t = {row['t_statistic']:.2f} {sig_marker}")
    else:
        print("No Fama-MacBeth results generated.")
    
    return results


def run_bootstrap_analysis(
    down_asy: pd.DataFrame,
    returns: pd.DataFrame,
    n_bootstrap: int = 1000,
    output_dir: str = 'outputs/tables'
) -> pd.DataFrame:
    """
    Run bootstrap inference for spread statistics.
    
    Parameters
    ----------
    down_asy : pd.DataFrame
        DOWN_ASY scores.
    returns : pd.DataFrame
        Monthly returns.
    n_bootstrap : int
        Number of bootstrap samples.
    output_dir : str
        Output directory.
        
    Returns
    -------
    pd.DataFrame
        Bootstrap inference results.
    """
    print("\n" + "=" * 60)
    print("Bootstrap Inference")
    print("=" * 60)
    
    # First compute the spread series
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
    
    if len(merged) < 100:
        print("Insufficient data for bootstrap analysis.")
        return pd.DataFrame()
    
    # Sort into quintiles
    try:
        merged['QUINTILE'] = merged.groupby('MONTH')['DOWN_ASY'].transform(
            lambda x: pd.qcut(x, 5, labels=False, duplicates='drop') + 1
            if len(x) >= 5 else np.nan
        )
    except Exception as e:
        print(f"Error creating quintiles: {e}")
        return pd.DataFrame()
    
    merged = merged.dropna(subset=['QUINTILE'])
    
    # Compute quintile returns
    quintile_returns = merged.groupby(['MONTH', 'QUINTILE'])['RET'].mean().unstack()
    
    if 1 not in quintile_returns.columns or 5 not in quintile_returns.columns:
        print("Could not compute High-Low spread.")
        return pd.DataFrame()
    
    spread = quintile_returns[5] - quintile_returns[1]
    spread = spread.dropna()
    
    if len(spread) < 24:
        print("Insufficient spread observations for bootstrap.")
        return pd.DataFrame()
    
    print(f"Computing bootstrap with {n_bootstrap} samples...")
    
    # Run bootstrap for different statistics
    bootstrap = BootstrapInference(n_bootstrap=n_bootstrap, block_size=12)
    
    results = []
    for statistic in ['mean', 'median', 'sharpe']:
        boot_result = bootstrap.bootstrap_spread(spread, statistic)
        results.append({
            'statistic': statistic,
            'observed': boot_result['observed'],
            'bootstrap_se': boot_result['bootstrap_se'],
            'ci_lower_95': boot_result['ci_lower'],
            'ci_upper_95': boot_result['ci_upper'],
            'p_value': boot_result['p_value'],
            'n_bootstrap': boot_result['n_bootstrap'],
        })
    
    results_df = pd.DataFrame(results)
    
    if len(results_df) > 0:
        # Save results
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'robustness_bootstrap.csv')
        results_df.to_csv(output_path, index=False)
        print(f"Saved: {output_path}")
        
        # Display summary
        print("\nResults:")
        print("-" * 60)
        print(results_df.to_string(index=False))
        
        print("\n95% Confidence Intervals:")
        for _, row in results_df.iterrows():
            print(f"  {row['statistic'].capitalize()}: [{row['ci_lower_95']:.4f}, {row['ci_upper_95']:.4f}]")
    
    return results_df


def generate_robustness_summary(
    all_results: dict,
    output_dir: str = 'outputs/tables'
) -> pd.DataFrame:
    """
    Generate a summary table of all robustness checks.
    
    Parameters
    ----------
    all_results : dict
        Dictionary of results from each robustness check.
    output_dir : str
        Output directory.
        
    Returns
    -------
    pd.DataFrame
        Summary table.
    """
    print("\n" + "=" * 60)
    print("Robustness Summary")
    print("=" * 60)
    
    summary_rows = []
    
    # Subperiod results
    if 'subperiod' in all_results and len(all_results['subperiod']) > 0:
        for _, row in all_results['subperiod'].iterrows():
            summary_rows.append({
                'Category': 'Subperiod',
                'Test': row.get('test_name', 'Subperiod'),
                'Specification': row.get('period', row.get('regime', '')),
                'Estimate': row.get('estimate', np.nan),
                't-stat': row.get('t_statistic', np.nan),
                'p-value': row.get('p_value', np.nan),
                'N': row.get('n_observations', 0),
            })
    
    # Alternative sorts
    if 'alternative_sorts' in all_results and len(all_results['alternative_sorts']) > 0:
        for _, row in all_results['alternative_sorts'].iterrows():
            summary_rows.append({
                'Category': 'Portfolio Sort',
                'Test': row.get('method', 'Sort'),
                'Specification': f"n_groups={row.get('n_groups', '')}",
                'Estimate': row.get('estimate', np.nan),
                't-stat': row.get('t_statistic', np.nan),
                'p-value': row.get('p_value', np.nan),
                'N': row.get('n_observations', 0),
            })
    
    # Fama-MacBeth
    if 'fama_macbeth' in all_results and len(all_results['fama_macbeth']) > 0:
        down_asy_fm = all_results['fama_macbeth'][
            all_results['fama_macbeth']['variable'] == 'DOWN_ASY'
        ]
        for _, row in down_asy_fm.iterrows():
            summary_rows.append({
                'Category': 'Fama-MacBeth',
                'Test': 'DOWN_ASY',
                'Specification': row.get('model', ''),
                'Estimate': row.get('estimate', np.nan),
                't-stat': row.get('t_statistic', np.nan),
                'p-value': row.get('p_value', np.nan),
                'N': row.get('n_observations', 0),
            })
    
    # Bootstrap
    if 'bootstrap' in all_results and len(all_results['bootstrap']) > 0:
        for _, row in all_results['bootstrap'].iterrows():
            summary_rows.append({
                'Category': 'Bootstrap',
                'Test': row.get('statistic', 'Statistic').capitalize(),
                'Specification': f"n_boot={row.get('n_bootstrap', '')}",
                'Estimate': row.get('observed', np.nan),
                't-stat': np.nan,  # Bootstrap doesn't have t-stat
                'p-value': row.get('p_value', np.nan),
                'N': np.nan,
            })
    
    summary_df = pd.DataFrame(summary_rows)
    
    if len(summary_df) > 0:
        # Save summary
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'robustness_summary.csv')
        summary_df.to_csv(output_path, index=False)
        print(f"Saved: {output_path}")
        
        # Display summary
        print("\nComplete Robustness Summary:")
        print("-" * 60)
        print(summary_df.to_string(index=False))
        
        # Count significant results
        sig_mask = summary_df['p-value'] < 0.05
        print(f"\nTotal tests: {len(summary_df)}")
        print(f"Significant at 5%: {sig_mask.sum()} ({100*sig_mask.mean():.1f}%)")
        
        # Check sign consistency
        positive = (summary_df['Estimate'] > 0).sum()
        print(f"Positive estimates: {positive} ({100*positive/len(summary_df):.1f}%)")
    
    return summary_df


def run_demo():
    """Run demo with synthetic data."""
    print("=" * 60)
    print("Phase 6: Robustness Checks - Demo Mode")
    print("=" * 60)
    
    # Create synthetic data
    np.random.seed(42)
    n_stocks = 200
    n_months = 120  # 10 years
    
    dates = pd.date_range('2000-01-01', periods=n_months, freq='ME')
    permnos = list(range(10001, 10001 + n_stocks))
    
    print(f"\nGenerating synthetic data:")
    print(f"  - {n_stocks} stocks")
    print(f"  - {n_months} months")
    print(f"  - Period: {dates[0].strftime('%Y-%m')} to {dates[-1].strftime('%Y-%m')}")
    
    # Create returns with cross-sectional variation
    returns_data = []
    for date in dates:
        for permno in permnos:
            # Returns have slight positive correlation with DOWN_ASY
            base_return = np.random.normal(0.008, 0.05)
            returns_data.append({
                'PERMNO': permno,
                'DATE': date,
                'RET': base_return + 0.02 * (permno - 10100) / 100 + np.random.normal(0, 0.02),
            })
    returns = pd.DataFrame(returns_data)
    
    # Create DOWN_ASY scores with persistence
    down_asy_data = []
    stock_effects = {p: np.random.normal(0, 0.05) for p in permnos}
    for date in dates:
        for permno in permnos:
            down_asy_data.append({
                'PERMNO': permno,
                'DATE': date,
                'DOWN_ASY': stock_effects[permno] + np.random.normal(0, 0.02),
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
        np.random.normal(0.007, 0.04, n_months),
        index=dates
    )
    
    # Store results
    all_results = {}
    
    start_time = time.time()
    
    # Run all robustness checks
    print("\nRunning robustness checks...")
    
    all_results['subperiod'] = run_subperiod_analysis(
        down_asy, returns, market_returns
    )
    
    all_results['alternative_sorts'] = run_alternative_sorts(
        down_asy, returns
    )
    
    all_results['fama_macbeth'] = run_fama_macbeth(
        down_asy, returns, characteristics
    )
    
    all_results['bootstrap'] = run_bootstrap_analysis(
        down_asy, returns, n_bootstrap=500  # Fewer for demo
    )
    
    # Generate summary
    summary = generate_robustness_summary(all_results)
    
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 60)
    print(f"Demo completed in {elapsed:.2f} seconds")
    print("=" * 60)
    print("\nOutput files generated in outputs/tables/:")
    print("  - robustness_subperiod.csv")
    print("  - robustness_alternative_sorts.csv")
    print("  - robustness_fama_macbeth.csv")
    print("  - robustness_bootstrap.csv")
    print("  - robustness_summary.csv")
    
    return summary


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Run robustness checks for entropy-comovement replication.'
    )
    parser.add_argument(
        '--demo', action='store_true',
        help='Run demo with synthetic data'
    )
    parser.add_argument(
        '--data-dir', type=str, default='data/processed',
        help='Path to processed data directory'
    )
    parser.add_argument(
        '--output-dir', type=str, default='outputs/tables',
        help='Path to output directory'
    )
    parser.add_argument(
        '--n-bootstrap', type=int, default=1000,
        help='Number of bootstrap samples'
    )
    
    args = parser.parse_args()
    
    if args.demo:
        run_demo()
        return
    
    # Load real data
    print("Loading data...")
    down_asy, returns, characteristics, market_returns, factors = load_data(args.data_dir)
    
    if down_asy is None or returns is None:
        print("Error: Required data files not found.")
        print("Run with --demo to test with synthetic data.")
        sys.exit(1)
    
    # Store results
    all_results = {}
    
    start_time = time.time()
    
    # Run all robustness checks
    all_results['subperiod'] = run_subperiod_analysis(
        down_asy, returns, market_returns, args.output_dir
    )
    
    all_results['alternative_sorts'] = run_alternative_sorts(
        down_asy, returns, args.output_dir
    )
    
    all_results['fama_macbeth'] = run_fama_macbeth(
        down_asy, returns, characteristics, args.output_dir
    )
    
    all_results['bootstrap'] = run_bootstrap_analysis(
        down_asy, returns, args.n_bootstrap, args.output_dir
    )
    
    # Generate summary
    summary = generate_robustness_summary(all_results, args.output_dir)
    
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 60)
    print(f"All robustness checks completed in {elapsed:.2f} seconds")
    print("=" * 60)


if __name__ == '__main__':
    main()
