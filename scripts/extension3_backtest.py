#!/usr/bin/env python3
"""
Extension 3: Portfolio Construction & Backtest

Constructs portfolios based on ML-predicted asymmetry ranks:
- Long: Decile 10 (Predicted high asymmetry / risky)
- Short: Decile 1 (Predicted low asymmetry / safe)

Compares against standard historical sorting strategy.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "python"))


def load_predictions(data_dir: Path) -> pd.DataFrame:
    """Load ML predictions."""
    print("  Loading ML predictions...")
    
    pred_path = data_dir / "ML_Predictions.csv"
    
    if pred_path.exists():
        df = pd.read_csv(pred_path)
        df['DATE'] = pd.to_datetime(df['DATE'])
    else:
        raise FileNotFoundError(f"Predictions not found at {pred_path}")
    
    print(f"    Loaded {len(df):,} predictions")
    
    return df


def load_returns(data_dir: Path) -> pd.DataFrame:
    """Load stock returns for portfolio construction."""
    print("  Loading returns data...")
    
    # Try multiple sources
    ml_path = data_dir / "ML_Features_Target.parquet"
    panel_path = data_dir / "Full_Panel_Data.csv"
    
    if ml_path.exists():
        df = pd.read_parquet(ml_path)
    elif panel_path.exists():
        df = pd.read_csv(panel_path)
    else:
        raise FileNotFoundError("Returns data not found")
    
    df.columns = df.columns.str.upper().str.strip()
    df['DATE'] = pd.to_datetime(df['DATE'])
    
    # Check for RET column or create synthetic returns
    if 'RET' not in df.columns:
        print("    Warning: RET column not found, generating synthetic returns")
        np.random.seed(42)
        df['RET'] = np.random.normal(0.01, 0.08, len(df))
    
    # Select available columns
    keep_cols = ['PERMNO', 'DATE', 'RET']
    if 'ME' in df.columns:
        keep_cols.append('ME')
    else:
        # Create synthetic ME for equal weighting
        df['ME'] = 1.0
        keep_cols.append('ME')
    
    print(f"    Loaded {len(df):,} return observations")
    
    return df[keep_cols].drop_duplicates()


def load_standard_strategy(tables_dir: Path) -> pd.Series:
    """Load standard (historical sorting) strategy returns."""
    print("  Loading standard strategy returns...")
    
    table5_path = tables_dir / "Table_5_Returns_Alphas.csv"
    
    if table5_path.exists():
        df = pd.read_csv(table5_path)
        # Extract High-Low spread if available
        if 'High-Low' in df.columns:
            # This is a summary table, not time series
            pass
    
    # If no time series available, return None
    return None


def construct_ml_portfolios(predictions_df: pd.DataFrame, 
                            returns_df: pd.DataFrame,
                            n_portfolios: int = 10) -> pd.DataFrame:
    """Construct portfolios based on ML predictions."""
    print("  Constructing ML-based portfolios...")
    
    # Merge predictions with returns
    predictions_df['YEAR_MONTH'] = predictions_df['DATE'].dt.to_period('M')
    returns_df['YEAR_MONTH'] = returns_df['DATE'].dt.to_period('M')
    
    # Shift returns by 1 month (predict at t, realize at t+1)
    returns_df['NEXT_YEAR_MONTH'] = returns_df['YEAR_MONTH'] + 1
    
    merged = predictions_df.merge(
        returns_df[['PERMNO', 'NEXT_YEAR_MONTH', 'RET', 'ME']].rename(
            columns={'NEXT_YEAR_MONTH': 'YEAR_MONTH'}
        ),
        on=['PERMNO', 'YEAR_MONTH'],
        how='inner'
    )
    
    print(f"    Merged predictions with returns: {len(merged):,}")
    
    # Assign deciles based on predicted rank using percentile-based approach
    def assign_decile_for_month(x):
        """Assign decile ranks (1-10) based on predicted rank percentiles."""
        if len(x) < n_portfolios:
            return pd.Series([np.nan] * len(x), index=x.index)
        # Use percentile ranks to assign deciles (1-10)
        pct_rank = x.rank(pct=True)
        deciles = np.ceil(pct_rank * n_portfolios).clip(1, n_portfolios)
        return deciles
    
    merged['PRED_DECILE'] = merged.groupby('YEAR_MONTH')['PREDICTED_RANK'].transform(
        assign_decile_for_month
    )
    
    merged = merged.dropna(subset=['PRED_DECILE'])
    merged['PRED_DECILE'] = merged['PRED_DECILE'].astype(int)
    
    # Calculate value-weighted returns for each decile
    def vw_return(group):
        if 'ME' in group.columns and group['ME'].sum() > 0:
            weights = group['ME'] / group['ME'].sum()
            return (group['RET'] * weights).sum()
        return group['RET'].mean()
    
    portfolio_rets = merged.groupby(['YEAR_MONTH', 'PRED_DECILE']).apply(vw_return)
    portfolio_rets = portfolio_rets.unstack()
    portfolio_rets.columns = [f'D{int(c)}' for c in portfolio_rets.columns]
    
    # Calculate High-Low spread (ML strategy)
    if 'D10' in portfolio_rets.columns and 'D1' in portfolio_rets.columns:
        portfolio_rets['ML_HIGH_LOW'] = portfolio_rets['D10'] - portfolio_rets['D1']
    else:
        # Use available extreme deciles
        max_decile = max([int(c[1:]) for c in portfolio_rets.columns if c.startswith('D')])
        min_decile = min([int(c[1:]) for c in portfolio_rets.columns if c.startswith('D')])
        portfolio_rets['ML_HIGH_LOW'] = portfolio_rets[f'D{max_decile}'] - portfolio_rets[f'D{min_decile}']
    
    portfolio_rets = portfolio_rets.reset_index()
    portfolio_rets['DATE'] = portfolio_rets['YEAR_MONTH'].dt.to_timestamp('M')
    
    print(f"    Portfolio returns: {len(portfolio_rets)} months")
    
    return portfolio_rets


def construct_standard_portfolios(returns_df: pd.DataFrame,
                                   predictions_df: pd.DataFrame,
                                   n_portfolios: int = 10) -> pd.DataFrame:
    """Construct standard portfolios based on historical (actual) ranks."""
    print("  Constructing standard (historical) portfolios...")
    
    # Use actual ranks from predictions
    predictions_df['YEAR_MONTH'] = predictions_df['DATE'].dt.to_period('M')
    returns_df['YEAR_MONTH'] = returns_df['DATE'].dt.to_period('M')
    
    # Shift returns
    returns_df['NEXT_YEAR_MONTH'] = returns_df['YEAR_MONTH'] + 1
    
    merged = predictions_df.merge(
        returns_df[['PERMNO', 'NEXT_YEAR_MONTH', 'RET', 'ME']].rename(
            columns={'NEXT_YEAR_MONTH': 'YEAR_MONTH'}
        ),
        on=['PERMNO', 'YEAR_MONTH'],
        how='inner'
    )
    
    # Assign deciles based on ACTUAL rank (lagged) using percentile approach
    def assign_decile_for_month(x):
        """Assign decile ranks (1-10) based on actual rank percentiles."""
        if len(x) < n_portfolios:
            return pd.Series([np.nan] * len(x), index=x.index)
        pct_rank = x.rank(pct=True)
        deciles = np.ceil(pct_rank * n_portfolios).clip(1, n_portfolios)
        return deciles
    
    merged['ACTUAL_DECILE'] = merged.groupby('YEAR_MONTH')['ACTUAL_RANK'].transform(
        assign_decile_for_month
    )
    
    merged = merged.dropna(subset=['ACTUAL_DECILE'])
    merged['ACTUAL_DECILE'] = merged['ACTUAL_DECILE'].astype(int)
    
    # VW returns
    def vw_return(group):
        if 'ME' in group.columns and group['ME'].sum() > 0:
            weights = group['ME'] / group['ME'].sum()
            return (group['RET'] * weights).sum()
        return group['RET'].mean()
    
    portfolio_rets = merged.groupby(['YEAR_MONTH', 'ACTUAL_DECILE']).apply(vw_return)
    portfolio_rets = portfolio_rets.unstack()
    portfolio_rets.columns = [f'D{int(c)}' for c in portfolio_rets.columns]
    
    # High-Low spread
    if 'D10' in portfolio_rets.columns and 'D1' in portfolio_rets.columns:
        portfolio_rets['STD_HIGH_LOW'] = portfolio_rets['D10'] - portfolio_rets['D1']
    else:
        max_decile = max([int(c[1:]) for c in portfolio_rets.columns if c.startswith('D')])
        min_decile = min([int(c[1:]) for c in portfolio_rets.columns if c.startswith('D')])
        portfolio_rets['STD_HIGH_LOW'] = portfolio_rets[f'D{max_decile}'] - portfolio_rets[f'D{min_decile}']
    
    portfolio_rets = portfolio_rets.reset_index()
    portfolio_rets['DATE'] = portfolio_rets['YEAR_MONTH'].dt.to_timestamp('M')
    
    return portfolio_rets


def calculate_metrics(returns: pd.Series, benchmark: pd.Series = None) -> dict:
    """Calculate performance metrics."""
    returns = returns.dropna()
    
    # Annualization factor
    ann_factor = 12
    
    # Basic stats
    mean_ret = returns.mean() * ann_factor
    std_ret = returns.std() * np.sqrt(ann_factor)
    sharpe = mean_ret / std_ret if std_ret > 0 else 0
    
    # Cumulative return
    cum_ret = (1 + returns).prod() - 1
    
    # Max drawdown
    cum_wealth = (1 + returns).cumprod()
    running_max = cum_wealth.expanding().max()
    drawdown = cum_wealth / running_max - 1
    max_drawdown = drawdown.min()
    
    metrics = {
        'Mean Return (Ann.)': mean_ret * 100,
        'Volatility (Ann.)': std_ret * 100,
        'Sharpe Ratio': sharpe,
        'Cumulative Return': cum_ret * 100,
        'Max Drawdown': max_drawdown * 100
    }
    
    # Information ratio vs benchmark
    if benchmark is not None:
        aligned = pd.DataFrame({'ret': returns, 'bench': benchmark}).dropna()
        if len(aligned) > 12:
            active_ret = aligned['ret'] - aligned['bench']
            tracking_error = active_ret.std() * np.sqrt(ann_factor)
            ir = (active_ret.mean() * ann_factor) / tracking_error if tracking_error > 0 else 0
            hit_rate = (aligned['ret'] > aligned['bench']).mean()
            
            metrics['Information Ratio'] = ir
            metrics['Hit Rate'] = hit_rate * 100
    
    return metrics


def format_table10(ml_metrics: dict, std_metrics: dict) -> pd.DataFrame:
    """Format results as Table 10."""
    
    rows = [
        {
            'Strategy': 'ML Strategy (XGBoost)',
            **{k: f"{v:.2f}" if isinstance(v, float) else v for k, v in ml_metrics.items()}
        },
        {
            'Strategy': 'Standard Strategy (Historical)',
            **{k: f"{v:.2f}" if isinstance(v, float) else v for k, v in std_metrics.items()}
        }
    ]
    
    # Add difference row
    diff = {
        'Strategy': 'ML - Standard',
        'Mean Return (Ann.)': f"{ml_metrics['Mean Return (Ann.)'] - std_metrics['Mean Return (Ann.)']:.2f}",
        'Sharpe Ratio': f"{ml_metrics['Sharpe Ratio'] - std_metrics['Sharpe Ratio']:.2f}",
    }
    if 'Information Ratio' in ml_metrics:
        diff['Information Ratio'] = f"{ml_metrics.get('Information Ratio', 0):.2f}"
    if 'Hit Rate' in ml_metrics:
        diff['Hit Rate'] = f"{ml_metrics.get('Hit Rate', 50):.1f}"
    
    rows.append(diff)
    
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="ML Portfolio Backtest")
    parser.add_argument("--demo", action="store_true", help="Demo mode")
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--output-dir", type=str, default="outputs/tables")
    args = parser.parse_args()
    
    print("=" * 70)
    print("EXTENSION 3: ML PORTFOLIO BACKTEST")
    print("=" * 70)
    
    project_root = Path(__file__).parent.parent
    data_dir = project_root / args.data_dir
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n" + "-" * 70)
    print("LOADING DATA")
    print("-" * 70)
    
    try:
        predictions_df = load_predictions(data_dir)
    except FileNotFoundError:
        print("  Predictions not found. Running walk-forward training first...")
        import subprocess
        demo_flag = ["--demo"] if args.demo else []
        subprocess.run([sys.executable, "scripts/extension3_walk_forward.py"] + demo_flag,
                       cwd=str(project_root))
        predictions_df = load_predictions(data_dir)
    
    returns_df = load_returns(data_dir)
    
    # Construct portfolios
    print("\n" + "-" * 70)
    print("PORTFOLIO CONSTRUCTION")
    print("-" * 70)
    
    ml_portfolio = construct_ml_portfolios(predictions_df, returns_df)
    std_portfolio = construct_standard_portfolios(returns_df, predictions_df)
    
    # Merge for comparison
    comparison = ml_portfolio[['DATE', 'ML_HIGH_LOW']].merge(
        std_portfolio[['DATE', 'STD_HIGH_LOW']],
        on='DATE',
        how='inner'
    )
    
    # Save portfolio returns
    portfolio_path = data_dir / "ML_Portfolio_Returns.csv"
    comparison.to_csv(portfolio_path, index=False)
    print(f"\n  Saved portfolio returns: {portfolio_path}")
    
    # Calculate metrics
    print("\n" + "-" * 70)
    print("PERFORMANCE METRICS")
    print("-" * 70)
    
    ml_metrics = calculate_metrics(
        comparison['ML_HIGH_LOW'], 
        benchmark=comparison['STD_HIGH_LOW']
    )
    std_metrics = calculate_metrics(comparison['STD_HIGH_LOW'])
    
    print("\n  ML Strategy:")
    for k, v in ml_metrics.items():
        print(f"    {k}: {v:.2f}" if isinstance(v, float) else f"    {k}: {v}")
    
    print("\n  Standard Strategy:")
    for k, v in std_metrics.items():
        print(f"    {k}: {v:.2f}" if isinstance(v, float) else f"    {k}: {v}")
    
    # Format and save Table 10
    print("\n" + "-" * 70)
    print("TABLE 10: ML PERFORMANCE COMPARISON")
    print("-" * 70)
    
    table10 = format_table10(ml_metrics, std_metrics)
    print("\n" + table10.to_string(index=False))
    
    table10_path = output_dir / "Table_10_ML_Performance.csv"
    table10.to_csv(table10_path, index=False)
    print(f"\n  Saved: {table10_path}")
    
    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    
    sharpe_diff = ml_metrics['Sharpe Ratio'] - std_metrics['Sharpe Ratio']
    
    if sharpe_diff > 0.1:
        print("\n  The ML strategy OUTPERFORMS the standard strategy.")
        print(f"  Sharpe Ratio improvement: {sharpe_diff:.2f}")
        print("\n  CONCLUSION: ML adds value by better predicting future asymmetry.")
    elif sharpe_diff > -0.1:
        print("\n  The ML strategy performs SIMILARLY to the standard strategy.")
        print(f"  Sharpe Ratio difference: {sharpe_diff:.2f}")
        print("\n  CONCLUSION: ML adds complexity without clear alpha improvement.")
    else:
        print("\n  The ML strategy UNDERPERFORMS the standard strategy.")
        print(f"  Sharpe Ratio decline: {sharpe_diff:.2f}")
        print("\n  CONCLUSION: Historical ranks are sufficient; ML adds noise.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
