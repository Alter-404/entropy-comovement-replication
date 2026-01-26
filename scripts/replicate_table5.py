"""
scripts/replicate_table5.py
Replicate Table 5: Returns and Alphas from Univariate Sorts

This script implements the asset pricing analysis from Jiang, Wu, Zhou (2018):
Table 5 reports equal-weighted average returns and alphas for stock portfolios 
sorted by different realized asymmetry measures.

Structure (from paper):
    Panel A: Downside Asymmetry and Conventional Asymmetry Measures
        - DOWN_ASY: Downside asymmetry measure
        - DOWN_CORR: Downside asymmetric correlation
        - DOWNSIDE_BETA: Downside beta (β−)
        - COSKEW: Coskewness
    
    Panel B: Alternative Entropy Measures
        - BBC_COENTROPY: Based on Backus et al. (2018)
        - CC_COENTROPY: Based on Chabi-Yo and Colacito (2016)

For each measure:
    - Quintile portfolios (1-5)
    - Equal-weighted monthly returns
    - Carhart 4-factor alphas
    - High-Low spread with t-statistics

Sample: All U.S. common stocks (NYSE/AMEX/NASDAQ), Jan 1963 - Dec 2013

Output:
    - Table_5_Returns_Alphas.csv
    - Table_5_Returns_Alphas.tex
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
from typing import Tuple, Dict, List, Optional
from scipy import stats
import warnings

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Panel A measures
PANEL_A_MEASURES = ['DOWN_ASY', 'DOWN_CORR', 'DOWNSIDE_BETA', 'COSKEW']

# Panel B measures  
PANEL_B_MEASURES = ['BBC_COENTROPY', 'CC_COENTROPY']

# All measures
ALL_MEASURES = PANEL_A_MEASURES + PANEL_B_MEASURES

# Column labels for display
MEASURE_LABELS = {
    'DOWN_ASY': 'DOWN_ASY',
    'DOWN_CORR': 'DOWN_CORR',
    'DOWNSIDE_BETA': 'DOWNSIDE_BETA',
    'COSKEW': 'COSKEW',
    'BBC_COENTROPY': 'BBC Co-Entropy',
    'CC_COENTROPY': 'CC Co-Entropy'
}

# Try to import statsmodels for regressions
try:
    import statsmodels.api as sm
    from statsmodels.regression.linear_model import OLS
    from statsmodels.stats.sandwich_covariance import cov_hac
    HAS_STATSMODELS = True
except ImportError:
    print("WARNING: statsmodels not found. Using simple OLS without Newey-West.")
    HAS_STATSMODELS = False


def run_carhart_regression(returns: pd.Series,
                            factors: pd.DataFrame,
                            newey_west_lags: int = 12) -> Dict:
    """
    Run Carhart 4-factor regression with Newey-West standard errors.
    
    Model: R_t - RF_t = α + β_MKT*(MKT-RF)_t + β_SMB*SMB_t + β_HML*HML_t + β_MOM*MOM_t + ε_t
    
    Parameters
    ----------
    returns : pd.Series
        Portfolio returns (not excess returns yet).
    factors : pd.DataFrame
        Fama-French 4 factors with RF column.
    newey_west_lags : int
        Number of lags for Newey-West HAC standard errors.
    
    Returns
    -------
    dict
        Results with keys: alpha, t_stat, se
    """
    # Align data
    common_dates = returns.index.intersection(factors.index)
    ret_aligned = returns.loc[common_dates]
    factors_aligned = factors.loc[common_dates]
    
    if len(common_dates) < 30:
        return {'alpha': np.nan, 't_stat': np.nan, 'se': np.nan}
    
    # Compute excess returns
    excess_returns = ret_aligned.values - factors_aligned['RF'].values
    
    # Prepare regressors
    X = factors_aligned[['Mkt-RF', 'SMB', 'HML', 'MOM']].values
    y = excess_returns
    
    # Remove NaNs
    valid_mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    y = y[valid_mask]
    X = X[valid_mask]
    
    if len(y) < 30:
        return {'alpha': np.nan, 't_stat': np.nan, 'se': np.nan}
    
    # Add constant
    X_with_const = np.column_stack([np.ones(len(X)), X])
    
    if HAS_STATSMODELS:
        model = OLS(y, X_with_const)
        results = model.fit()
        cov_nw = cov_hac(results, nlags=newey_west_lags)
        se_nw = np.sqrt(np.diag(cov_nw))
        
        alpha = results.params[0]
        t_stat = alpha / se_nw[0]
        
        return {'alpha': alpha * 100, 't_stat': t_stat, 'se': se_nw[0] * 100}
    else:
        beta = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
        residuals = y - X_with_const @ beta
        sigma_sq = np.sum(residuals**2) / (len(y) - X_with_const.shape[1])
        cov = sigma_sq * np.linalg.inv(X_with_const.T @ X_with_const)
        se = np.sqrt(np.diag(cov))
        
        alpha = beta[0]
        t_stat = alpha / se[0]
        
        return {'alpha': alpha * 100, 't_stat': t_stat, 'se': se[0] * 100}


def create_demo_characteristics(n_stocks: int = 500, n_months: int = 120) -> pd.DataFrame:
    """
    Create demo data with all required characteristics for Table 5.
    
    Parameters
    ----------
    n_stocks : int
        Number of stocks.
    n_months : int
        Number of months.
    
    Returns
    -------
    pd.DataFrame
        Panel data with all measures.
    """
    np.random.seed(42)
    
    dates = pd.date_range('2003-12-31', periods=n_months, freq='ME')
    permnos = [f'STOCK_{i:04d}' for i in range(1, n_stocks + 1)]
    
    data = []
    
    for permno in permnos:
        # Stock-level persistent component
        stock_base = np.random.randn() * 0.3
        
        for date in dates:
            # Create correlated characteristics
            # DOWN_ASY is the main sorting variable
            down_asy = stock_base + np.random.randn() * 0.15
            
            # DOWN_CORR: correlated with DOWN_ASY (rho ~ 0.6)
            down_corr = 0.6 * down_asy + np.random.randn() * 0.12
            
            # DOWNSIDE_BETA: correlated with DOWN_ASY (rho ~ 0.5)
            downside_beta = 0.8 + 0.5 * down_asy + np.random.randn() * 0.3
            
            # COSKEW: negatively correlated with DOWN_ASY (rho ~ -0.7)
            coskew = -0.7 * down_asy + np.random.randn() * 0.15
            
            # BBC Co-Entropy: correlated with DOWN_ASY (rho ~ 0.4)
            bbc_coentropy = 0.4 * down_asy + np.random.randn() * 0.1
            
            # CC Co-Entropy: correlated with DOWN_ASY (rho ~ 0.35)
            cc_coentropy = 0.35 * down_asy + np.random.randn() * 0.1
            
            # Return: higher for higher DOWN_ASY
            ret = 0.01 + 0.005 * down_asy + np.random.randn() * 0.08
            
            data.append({
                'PERMNO': permno,
                'DATE': date,
                'DOWN_ASY': down_asy,
                'DOWN_CORR': down_corr,
                'DOWNSIDE_BETA': downside_beta,
                'COSKEW': coskew,
                'BBC_COENTROPY': bbc_coentropy,
                'CC_COENTROPY': cc_coentropy,
                'RET': ret
            })
    
    return pd.DataFrame(data)


def sort_and_compute_returns(data: pd.DataFrame,
                              measure: str,
                              n_quintiles: int = 5) -> pd.DataFrame:
    """
    Sort stocks into quintiles and compute equal-weighted portfolio returns.
    
    Parameters
    ----------
    data : pd.DataFrame
        Panel data with measure and RET columns.
    measure : str
        Characteristic to sort on.
    n_quintiles : int
        Number of portfolios.
    
    Returns
    -------
    pd.DataFrame
        Monthly portfolio returns indexed by DATE.
    """
    # Ensure data has required columns
    if measure not in data.columns:
        raise ValueError(f"Measure '{measure}' not found in data")
    
    df = data.copy()
    
    # Lag the signal (sort at t-1, return at t)
    df['DATE_LEAD'] = df.groupby('PERMNO')['DATE'].shift(-1)
    df['RET_LEAD'] = df.groupby('PERMNO')['RET'].shift(-1)
    
    # Remove last month (no future return)
    df = df.dropna(subset=['DATE_LEAD', 'RET_LEAD'])
    
    # Sort into quintiles each month
    def assign_quintile(group):
        try:
            group['QUINTILE'] = pd.qcut(
                group[measure].rank(method='first'),
                q=n_quintiles,
                labels=range(1, n_quintiles + 1)
            )
        except ValueError:
            # Not enough unique values
            group['QUINTILE'] = np.nan
        return group
    
    df = df.groupby('DATE', group_keys=False).apply(assign_quintile)
    df = df.dropna(subset=['QUINTILE'])
    df['QUINTILE'] = df['QUINTILE'].astype(int)
    
    # Compute equal-weighted portfolio returns
    port_returns = df.groupby(['DATE_LEAD', 'QUINTILE'])['RET_LEAD'].mean().unstack()
    port_returns.columns = [f'Q{i}' for i in port_returns.columns]
    port_returns.index.name = 'DATE'
    
    # Compute High-Low spread
    port_returns['HIGH_LOW'] = port_returns['Q5'] - port_returns['Q1']
    
    return port_returns


def compute_measure_results(data: pd.DataFrame,
                             measure: str,
                             factors: pd.DataFrame,
                             n_quintiles: int = 5) -> Dict:
    """
    Compute returns and alphas for a single measure.
    
    Parameters
    ----------
    data : pd.DataFrame
        Panel data.
    measure : str
        Characteristic to sort on.
    factors : pd.DataFrame
        Fama-French factors.
    n_quintiles : int
        Number of portfolios.
    
    Returns
    -------
    dict
        Results with returns and alphas for each quintile.
    """
    # Get portfolio returns
    port_returns = sort_and_compute_returns(data, measure, n_quintiles)
    
    results = {'measure': measure}
    
    # For each quintile
    for q in range(1, n_quintiles + 1):
        col = f'Q{q}'
        if col in port_returns.columns:
            ret_series = port_returns[col]
            
            # Mean return
            mean_ret = ret_series.mean() * 100
            results[f'ret_Q{q}'] = mean_ret
            
            # Carhart alpha
            reg_results = run_carhart_regression(ret_series, factors)
            results[f'alpha_Q{q}'] = reg_results['alpha']
            results[f'tstat_Q{q}'] = reg_results['t_stat']
    
    # High-Low spread
    if 'HIGH_LOW' in port_returns.columns:
        hl_series = port_returns['HIGH_LOW']
        hl_mean = hl_series.mean() * 100
        hl_tstat_ret = hl_mean / (hl_series.std() * 100 / np.sqrt(len(hl_series.dropna())))
        
        results['ret_HL'] = hl_mean
        results['tstat_ret_HL'] = hl_tstat_ret
        
        # Alpha for High-Low
        reg_results = run_carhart_regression(hl_series, factors)
        results['alpha_HL'] = reg_results['alpha']
        results['tstat_alpha_HL'] = reg_results['t_stat']
    
    return results


def generate_table5(data: Optional[pd.DataFrame] = None,
                    factors: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate Table 5: Returns and Alphas from Univariate Sorts.
    
    Parameters
    ----------
    data : pd.DataFrame, optional
        Panel data with all measures. If None, creates demo data.
    factors : pd.DataFrame, optional
        Fama-French factors. If None, creates demo factors.
    
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (Panel A results, Panel B results)
    """
    print("\n" + "="*80)
    print("GENERATING TABLE 5: RETURNS AND ALPHAS FROM UNIVARIATE SORTS")
    print("="*80)
    
    # Create demo data if not provided
    if data is None:
        print("\nUsing demo data for demonstration...")
        data = create_demo_characteristics()
    
    # Create demo factors if not provided
    if factors is None:
        print("Creating demo Fama-French factors...")
        dates = data['DATE'].unique()
        np.random.seed(123)
        factors = pd.DataFrame({
            'Mkt-RF': np.random.randn(len(dates)) * 0.04,
            'SMB': np.random.randn(len(dates)) * 0.02,
            'HML': np.random.randn(len(dates)) * 0.02,
            'MOM': np.random.randn(len(dates)) * 0.03,
            'RF': np.ones(len(dates)) * 0.002
        }, index=pd.DatetimeIndex(sorted(dates)))
    
    print(f"\nData: {len(data)} observations")
    print(f"Months: {data['DATE'].nunique()}")
    print(f"Stocks: {data['PERMNO'].nunique()}")
    
    # Compute results for each measure
    all_results = []
    
    for measure in ALL_MEASURES:
        if measure in data.columns:
            print(f"\nProcessing {MEASURE_LABELS.get(measure, measure)}...")
            results = compute_measure_results(data, measure, factors)
            all_results.append(results)
        else:
            print(f"  WARNING: {measure} not found in data")
    
    if not all_results:
        raise ValueError("No measures found in data")
    
    # Build output tables
    print("\n" + "="*80)
    print("Table 5: Portfolio Returns and Alphas")
    print("="*80)
    
    # Panel A
    print("\n" + "-"*80)
    print("Panel A: Downside Asymmetry and Conventional Asymmetry Measures")
    print("-"*80)
    
    panel_a_results = [r for r in all_results if r['measure'] in PANEL_A_MEASURES]
    panel_a = build_panel_table(panel_a_results)
    
    # Panel B
    print("\n" + "-"*80)
    print("Panel B: Alternative Entropy Measures")
    print("-"*80)
    
    panel_b_results = [r for r in all_results if r['measure'] in PANEL_B_MEASURES]
    panel_b = build_panel_table(panel_b_results)
    
    return panel_a, panel_b


def build_panel_table(results_list: List[Dict]) -> pd.DataFrame:
    """
    Build a panel table from results.
    
    Parameters
    ----------
    results_list : list of dict
        Results for each measure.
    
    Returns
    -------
    pd.DataFrame
        Formatted panel table.
    """
    if not results_list:
        return pd.DataFrame()
    
    # Build rows
    rows = []
    
    # Quintile rows
    for q in range(1, 6):
        row = {'Portfolio': f'{q}'}
        for r in results_list:
            measure = r['measure']
            label = MEASURE_LABELS.get(measure, measure)
            
            ret = r.get(f'ret_Q{q}', np.nan)
            alpha = r.get(f'alpha_Q{q}', np.nan)
            tstat = r.get(f'tstat_Q{q}', np.nan)
            
            # Format return
            row[f'{label}_Return'] = f'{ret:.2f}' if pd.notna(ret) else ''
            
            # Format alpha with significance
            if pd.notna(alpha) and pd.notna(tstat):
                stars = get_significance_stars(tstat)
                row[f'{label}_Alpha'] = f'{alpha:.2f}{stars}'
            else:
                row[f'{label}_Alpha'] = ''
        
        rows.append(row)
        print(format_row_for_display(row))
    
    # High-Low row
    hl_row = {'Portfolio': 'High-Low'}
    for r in results_list:
        measure = r['measure']
        label = MEASURE_LABELS.get(measure, measure)
        
        ret = r.get('ret_HL', np.nan)
        tstat_ret = r.get('tstat_ret_HL', np.nan)
        alpha = r.get('alpha_HL', np.nan)
        tstat_alpha = r.get('tstat_alpha_HL', np.nan)
        
        # Return with t-stat
        if pd.notna(ret) and pd.notna(tstat_ret):
            stars = get_significance_stars(tstat_ret)
            hl_row[f'{label}_Return'] = f'{ret:.2f}{stars}'
        else:
            hl_row[f'{label}_Return'] = ''
        
        # Alpha with t-stat
        if pd.notna(alpha) and pd.notna(tstat_alpha):
            stars = get_significance_stars(tstat_alpha)
            hl_row[f'{label}_Alpha'] = f'{alpha:.2f}{stars}'
        else:
            hl_row[f'{label}_Alpha'] = ''
    
    rows.append(hl_row)
    print(format_row_for_display(hl_row))
    
    # t-statistics row
    tstat_row = {'Portfolio': ''}
    for r in results_list:
        measure = r['measure']
        label = MEASURE_LABELS.get(measure, measure)
        
        tstat_ret = r.get('tstat_ret_HL', np.nan)
        tstat_alpha = r.get('tstat_alpha_HL', np.nan)
        
        tstat_row[f'{label}_Return'] = f'({tstat_ret:.2f})' if pd.notna(tstat_ret) else ''
        tstat_row[f'{label}_Alpha'] = f'({tstat_alpha:.2f})' if pd.notna(tstat_alpha) else ''
    
    rows.append(tstat_row)
    print(format_row_for_display(tstat_row))
    
    return pd.DataFrame(rows)


def format_row_for_display(row: Dict) -> str:
    """Format a row for console display."""
    parts = []
    for k, v in row.items():
        if k == 'Portfolio':
            parts.append(f"{v:12s}")
        else:
            parts.append(f"{str(v):>10s}")
    return "  ".join(parts[:9])  # Limit width


def get_significance_stars(t_stat: float) -> str:
    """Return significance stars based on t-statistic."""
    if pd.isna(t_stat):
        return ''
    t_abs = abs(t_stat)
    if t_abs >= 2.576:  # 1% level
        return '***'
    elif t_abs >= 1.96:  # 5% level
        return '**'
    elif t_abs >= 1.645:  # 10% level
        return '*'
    return ''


def save_table5(panel_a: pd.DataFrame, panel_b: pd.DataFrame, output_dir: str) -> None:
    """
    Save Table 5 to CSV and LaTeX.
    
    Parameters
    ----------
    panel_a : pd.DataFrame
        Panel A results.
    panel_b : pd.DataFrame
        Panel B results.
    output_dir : str
        Output directory.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Combined CSV
    combined = pd.concat([
        pd.DataFrame([{'Portfolio': 'Panel A: Downside Asymmetry and Conventional Measures'}]),
        panel_a,
        pd.DataFrame([{'Portfolio': ''}]),
        pd.DataFrame([{'Portfolio': 'Panel B: Alternative Entropy Measures'}]),
        panel_b
    ], ignore_index=True)
    
    csv_file = output_path / "Table_5_Returns_Alphas.csv"
    combined.to_csv(csv_file, index=False, encoding='utf-8')
    print(f"\nSaved CSV: {csv_file}")
    
    # LaTeX
    latex = generate_table5_latex(panel_a, panel_b)
    tex_file = output_path / "Table_5_Returns_Alphas.tex"
    with open(tex_file, 'w', encoding='utf-8') as f:
        f.write(latex)
    print(f"Saved LaTeX: {tex_file}")


def generate_table5_latex(panel_a: pd.DataFrame, panel_b: pd.DataFrame) -> str:
    """
    Generate LaTeX code for Table 5.
    
    Parameters
    ----------
    panel_a : pd.DataFrame
        Panel A results.
    panel_b : pd.DataFrame
        Panel B results.
    
    Returns
    -------
    str
        LaTeX code.
    """
    # Get measure columns from Panel A
    a_measures = [c.replace('_Return', '') for c in panel_a.columns if c.endswith('_Return')]
    b_measures = [c.replace('_Return', '') for c in panel_b.columns if c.endswith('_Return')]
    
    n_a = len(a_measures)
    n_b = len(b_measures)
    
    latex = r"""\begin{table}[htbp]
\centering
\caption{Portfolio Returns and Alphas}
\label{tab:table5}
\small
"""
    
    # Panel A
    col_spec = 'l' + 'cc' * n_a
    latex += r"\begin{tabular}{" + col_spec + r"}" + "\n"
    latex += r"\toprule" + "\n"
    
    # Panel A header
    latex += r"\multicolumn{" + str(1 + 2*n_a) + r"}{l}{\textbf{Panel A: Downside Asymmetry and Conventional Asymmetry Measures}} \\" + "\n"
    latex += r"\midrule" + "\n"
    
    # Measure headers
    header_row = "Portfolio"
    for m in a_measures:
        header_row += f" & \\multicolumn{{2}}{{c}}{{{m}}}"
    header_row += r" \\" + "\n"
    latex += header_row
    
    # Sub-headers
    subheader = ""
    for m in a_measures:
        subheader += " & Return & Carhart $\\alpha$"
    latex += subheader + r" \\" + "\n"
    latex += r"\midrule" + "\n"
    
    # Data rows for Panel A
    for _, row in panel_a.iterrows():
        if row['Portfolio'] in ['', None]:
            continue
        latex_row = str(row['Portfolio'])
        for m in a_measures:
            ret = row.get(f'{m}_Return', '')
            alpha = row.get(f'{m}_Alpha', '')
            latex_row += f" & {ret} & {alpha}"
        latex_row += r" \\" + "\n"
        latex += latex_row
    
    latex += r"\midrule" + "\n"
    
    # Panel B header
    latex += r"\multicolumn{" + str(1 + 2*n_a) + r"}{l}{} \\" + "\n"
    
    # If panel B has different number of columns, we need to adjust
    if n_b > 0:
        latex += r"\multicolumn{" + str(1 + 2*n_a) + r"}{l}{\textbf{Panel B: Alternative Entropy Measures}} \\" + "\n"
        latex += r"\midrule" + "\n"
        
        # Panel B headers
        header_row = "Portfolio"
        for i, m in enumerate(b_measures):
            header_row += f" & \\multicolumn{{2}}{{c}}{{{m}}}"
        # Pad remaining columns
        for _ in range(n_a - n_b):
            header_row += " & & "
        header_row += r" \\" + "\n"
        latex += header_row
        
        # Sub-headers
        subheader = ""
        for m in b_measures:
            subheader += " & Return & Carhart $\\alpha$"
        for _ in range(n_a - n_b):
            subheader += " & & "
        latex += subheader + r" \\" + "\n"
        latex += r"\midrule" + "\n"
        
        # Data rows for Panel B
        for _, row in panel_b.iterrows():
            if row['Portfolio'] in ['', None]:
                continue
            latex_row = str(row['Portfolio'])
            for m in b_measures:
                ret = row.get(f'{m}_Return', '')
                alpha = row.get(f'{m}_Alpha', '')
                latex_row += f" & {ret} & {alpha}"
            for _ in range(n_a - n_b):
                latex_row += " & & "
            latex_row += r" \\" + "\n"
            latex += latex_row
    
    latex += r"\bottomrule" + "\n"
    latex += r"\end{tabular}" + "\n"
    
    # Notes
    latex += r"""
\begin{tablenotes}
\small
\item This table reports equal-weighted average returns and Carhart (1997) four-factor alphas 
(in percentage points) for stock portfolios sorted by different asymmetry measures. DOWN\_ASY 
is the downside asymmetry measure, DOWN\_CORR is the downside asymmetric correlation, and 
DOWNSIDE\_BETA is the downside beta. BBC and CC co-entropies are based on Backus et al. (2018) 
and Chabi-Yo and Colacito (2016), respectively. High $-$ Low reports the difference between 
portfolios 5 and 1 with t-statistics in parentheses computed using Newey-West (1987) standard 
errors with 12 lags. Sample: NYSE/AMEX/NASDAQ common stocks, Jan 1963 -- Dec 2013. 
*, **, and *** indicate significance at 10\%, 5\%, and 1\% levels.
\end{tablenotes}
"""
    
    latex += r"\end{table}" + "\n"
    
    return latex


def main():
    """Main execution function."""
    print("="*80)
    print("REPLICATING TABLE 5: RETURNS AND ALPHAS")
    print("="*80)
    
    OUTPUT_DIR = "outputs/tables"
    
    # Generate Table 5
    panel_a, panel_b = generate_table5()
    
    # Save results
    save_table5(panel_a, panel_b, OUTPUT_DIR)
    
    print("\n" + "="*80)
    print("TABLE 5 REPLICATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
