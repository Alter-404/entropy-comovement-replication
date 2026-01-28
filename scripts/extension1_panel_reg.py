#!/usr/bin/env python3
"""
Extension 1: Panel Interaction Regression
==========================================

Test if the predictive power of DOWN_ASY changes during crisis regimes.

Model: R_{i,t+1} = α + β₁*DOWN_ASY_{i,t} + β₂*Crisis_t + β₃*(DOWN_ASY × Crisis) + Controls + ε

Focus: Coefficient β₃ (Interaction Term)
    - β₃ > 0: Market prices asymmetry MORE aggressively during crises
    - β₃ < 0: Market prices asymmetry LESS aggressively during crises

Output: Table_8_PanelB_Panel.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import statsmodels.api as sm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


def load_stock_data(
    processed_dir: Path,
    demo_mode: bool = False
) -> pd.DataFrame:
    """
    Load stock-level panel data with returns and characteristics.
    
    Returns DataFrame with columns:
        PERMNO, DATE, RET, DOWN_ASY, SIZE, BM, MOM, BETA
    """
    if demo_mode:
        np.random.seed(42)
        n_stocks = 500
        n_months = 120  # 10 years
        
        dates = pd.date_range('2004-01-31', periods=n_months, freq='ME')
        
        data = []
        for permno in range(1, n_stocks + 1):
            # Generate stock-specific characteristics
            base_down_asy = np.random.uniform(-0.1, 0.2)
            base_size = np.random.uniform(4, 12)  # Log market cap
            base_bm = np.random.uniform(-1, 1)
            base_beta = np.random.uniform(0.5, 1.5)
            
            for i, date in enumerate(dates):
                # Time-varying characteristics with persistence
                down_asy = base_down_asy + np.random.randn() * 0.05
                size = base_size + np.random.randn() * 0.2
                bm = base_bm + np.random.randn() * 0.1
                mom = np.random.randn() * 0.3
                beta = base_beta + np.random.randn() * 0.1
                
                # Return: depends on DOWN_ASY (positive premium)
                ret = (
                    0.008 +  # Base return
                    0.015 * down_asy +  # DOWN_ASY premium
                    -0.002 * size +  # Size effect
                    0.003 * bm +  # Value effect
                    np.random.randn() * 0.08  # Idiosyncratic
                )
                
                data.append({
                    'PERMNO': permno,
                    'DATE': date,
                    'RET': ret,
                    'DOWN_ASY': down_asy,
                    'SIZE': size,
                    'BM': bm,
                    'MOM': mom,
                    'BETA': beta
                })
        
        return pd.DataFrame(data)
    
    # Try to load from processed data
    chars_file = processed_dir / 'firm_characteristics.parquet'
    scores_file = processed_dir / 'down_asy_scores.parquet'
    
    if chars_file.exists() and scores_file.exists():
        chars = pd.read_parquet(chars_file)
        scores = pd.read_parquet(scores_file)
        
        # Merge
        data = pd.merge(chars, scores, on=['PERMNO', 'DATE'], how='inner')
        
        if len(data) > 0:
            return data
    
    # Fallback to demo
    return load_stock_data(processed_dir, demo_mode=True)


def load_crisis_flags(tables_dir: Path) -> pd.DataFrame:
    """Load crisis indicator flags."""
    crisis_file = tables_dir / 'Crisis_Flags.csv'
    if crisis_file.exists():
        return pd.read_csv(crisis_file, parse_dates=['DATE'])
    else:
        raise FileNotFoundError(
            f"Crisis flags not found at {crisis_file}. "
            "Run extension1_crisis_indicators.py first."
        )


def fama_macbeth_regression(
    data: pd.DataFrame,
    y_col: str,
    x_cols: List[str],
    time_col: str = 'DATE'
) -> Dict:
    """
    Run Fama-MacBeth two-pass regression.
    
    Pass 1: Cross-sectional regressions for each time period
    Pass 2: Time-series means of coefficients with Newey-West errors
    """
    # First pass: cross-sectional regressions
    coef_series = {col: [] for col in x_cols}
    coef_series['const'] = []
    dates = []
    
    for date, group in data.groupby(time_col):
        if len(group) < len(x_cols) + 5:  # Need sufficient obs
            continue
        
        y = group[y_col].values
        X = sm.add_constant(group[x_cols].values)
        
        try:
            model = sm.OLS(y, X)
            results = model.fit()
            
            coef_series['const'].append(results.params[0])
            for i, col in enumerate(x_cols):
                coef_series[col].append(results.params[i + 1])
            dates.append(date)
        except:
            continue
    
    if len(dates) == 0:
        return None
    
    # Second pass: time-series statistics
    results = {'N_periods': len(dates)}
    
    for col in ['const'] + x_cols:
        series = np.array(coef_series[col])
        mean = np.mean(series)
        
        # Newey-West standard errors (simplified HAC variance)
        T = len(series)
        lag = min(12, T // 2)
        
        # Simple HAC variance
        gamma0 = np.var(series, ddof=1)
        var_nw = gamma0
        
        for k in range(1, lag + 1):
            weight = 1 - k / (lag + 1)
            if len(series) > k:
                cov_k = np.mean((series[k:] - mean) * (series[:-k] - mean))
                var_nw += 2 * weight * cov_k
        
        se_nw = np.sqrt(var_nw / T)
        t_stat = mean / se_nw if se_nw > 0 else 0
        
        results[f'{col}_coef'] = mean
        results[f'{col}_se'] = se_nw
        results[f'{col}_t'] = t_stat
    
    return results


def run_panel_regression_with_interaction(
    stock_data: pd.DataFrame,
    crisis_flags: pd.DataFrame,
    crisis_col: str,
    crisis_name: str,
    control_vars: List[str] = None
) -> Dict:
    """
    Run panel regression with crisis interaction.
    
    Model: R_{i,t+1} = α + β₁*DOWN_ASY + β₂*Crisis + β₃*(DOWN_ASY × Crisis) + Controls
    """
    if control_vars is None:
        control_vars = ['SIZE', 'BM', 'MOM', 'BETA']
    
    # Merge stock data with crisis flags
    data = pd.merge(
        stock_data,
        crisis_flags[['DATE', crisis_col]],
        on='DATE',
        how='inner'
    )
    
    if len(data) == 0:
        return None
    
    # Create interaction term
    data['DOWN_ASY_x_CRISIS'] = data['DOWN_ASY'] * data[crisis_col]
    
    # Lead returns (next month)
    data = data.sort_values(['PERMNO', 'DATE'])
    data['RET_LEAD'] = data.groupby('PERMNO')['RET'].shift(-1)
    data = data.dropna(subset=['RET_LEAD'])
    
    # Regression variables
    x_cols = ['DOWN_ASY', crisis_col, 'DOWN_ASY_x_CRISIS']
    available_controls = [c for c in control_vars if c in data.columns]
    x_cols.extend(available_controls)
    
    # Run Fama-MacBeth
    results = fama_macbeth_regression(
        data,
        y_col='RET_LEAD',
        x_cols=x_cols,
        time_col='DATE'
    )
    
    if results is None:
        return None
    
    # Extract key results
    output = {
        'Crisis Definition': crisis_name,
        'β (DOWN_ASY)': results.get('DOWN_ASY_coef', np.nan),
        'β DOWN_ASY t-stat': results.get('DOWN_ASY_t', np.nan),
        'β (Crisis)': results.get(f'{crisis_col}_coef', np.nan),
        'β Crisis t-stat': results.get(f'{crisis_col}_t', np.nan),
        'β (Interaction)': results.get('DOWN_ASY_x_CRISIS_coef', np.nan),
        'β Interaction t-stat': results.get('DOWN_ASY_x_CRISIS_t', np.nan),
        'N Periods': results.get('N_periods', 0)
    }
    
    return output


def generate_table8_panelB(
    processed_dir: Optional[Path] = None,
    tables_dir: Optional[Path] = None,
    demo_mode: bool = False
) -> pd.DataFrame:
    """
    Generate Table 8 Panel B: Panel Interaction Regressions.
    
    Parameters
    ----------
    processed_dir : Path, optional
        Directory with processed data
    tables_dir : Path, optional
        Directory for output tables
    demo_mode : bool
        If True, use synthetic data
        
    Returns
    -------
    pd.DataFrame
        Regression results table
    """
    print("\n" + "="*70)
    print("  Extension 1: Panel Interaction Regression")
    print("="*70)
    
    # Setup paths
    if processed_dir is None:
        processed_dir = Path(__file__).parent.parent / 'data' / 'processed'
    if tables_dir is None:
        tables_dir = Path(__file__).parent.parent / 'outputs' / 'tables'
    
    processed_dir = Path(processed_dir)
    tables_dir = Path(tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n  Loading data...")
    stock_data = load_stock_data(processed_dir, demo_mode=demo_mode)
    print(f"    Stock-month observations: {len(stock_data):,}")
    
    try:
        crisis_flags = load_crisis_flags(tables_dir)
    except FileNotFoundError:
        print("  [WARN] Crisis flags not found, generating...")
        from extension1_crisis_indicators import generate_crisis_flags
        crisis_flags = generate_crisis_flags(
            output_dir=tables_dir,
            demo_mode=demo_mode
        )
    
    # Ensure DATE alignment
    stock_data['DATE'] = pd.to_datetime(stock_data['DATE'])
    crisis_flags['DATE'] = pd.to_datetime(crisis_flags['DATE'])
    
    # Run regressions
    print("\n  Running panel regressions with crisis interactions...")
    results = []
    
    crisis_definitions = [
        ('NBER', 'NBER Recession'),
        ('CRASH', 'Market Crash (<-5%)'),
        ('HIGH_VOL', 'High Volatility (>90th pct)'),
        ('ANY_CRISIS', 'Any Crisis')
    ]
    
    for col, name in crisis_definitions:
        result = run_panel_regression_with_interaction(
            stock_data,
            crisis_flags,
            col,
            name
        )
        
        if result is not None:
            results.append(result)
            
            # Print summary
            beta_int = result['β (Interaction)']
            t_stat = result['β Interaction t-stat']
            
            if beta_int > 0:
                interpretation = "MORE aggressive pricing during crisis"
            else:
                interpretation = "LESS aggressive pricing during crisis"
            
            significance = "***" if abs(t_stat) > 2.58 else "**" if abs(t_stat) > 1.96 else "*" if abs(t_stat) > 1.64 else ""
            
            print(f"    {name}:")
            print(f"      β_interaction = {beta_int*100:.3f}%{significance} (t={t_stat:.2f})")
            print(f"      → {interpretation}")
    
    # Create results DataFrame
    table = pd.DataFrame(results)
    
    # Save
    output_path = tables_dir / 'Table_8_PanelB_Panel.csv'
    table.to_csv(output_path, index=False, float_format='%.4f')
    print(f"\n  Saved: {output_path}")
    
    # Summary interpretation
    print("\n" + "="*70)
    print("  INTERPRETATION SUMMARY")
    print("="*70)
    
    print("\n  Does the market price DOWN_ASY differently during crises?")
    print("  " + "-"*55)
    
    avg_interaction = np.mean([r['β (Interaction)'] for r in results])
    avg_base = np.mean([r['β (DOWN_ASY)'] for r in results])
    
    if avg_interaction > 0:
        print("  → Market prices asymmetry MORE AGGRESSIVELY during crises")
        print(f"     Base DOWN_ASY premium: {avg_base*100:.3f}%/month")
        print(f"     Additional crisis premium: {avg_interaction*100:.3f}%/month")
        print("  → Investors demand HIGHER compensation for asymmetry risk in bad times")
    else:
        print("  → Market prices asymmetry LESS AGGRESSIVELY during crises")
        print(f"     Base DOWN_ASY premium: {avg_base*100:.3f}%/month")
        print(f"     Crisis discount: {avg_interaction*100:.3f}%/month")
        print("  → DOWN_ASY premium shrinks or reverses during turbulence")
    
    return table


def main():
    """Generate Table 8 Panel B."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Panel Interaction Regression')
    parser.add_argument('--processed-dir', type=str, help='Processed data directory')
    parser.add_argument('--output-dir', type=str, help='Output directory')
    parser.add_argument('--demo', action='store_true', help='Use demo data')
    args = parser.parse_args()
    
    processed_dir = Path(args.processed_dir) if args.processed_dir else None
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    table = generate_table8_panelB(
        processed_dir=processed_dir,
        tables_dir=output_dir,
        demo_mode=args.demo
    )
    
    print("\n  Table 8 Panel B:")
    print(table.to_string(index=False))


if __name__ == '__main__':
    main()
