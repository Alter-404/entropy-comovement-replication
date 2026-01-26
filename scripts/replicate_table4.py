"""
Replicate Table 4: Summary Statistics of Asymmetry Portfolios

From the paper (Jiang, Wu, Zhou 2018):
"Table 4 reports various risk measures and firm characteristics of deciles 
sorted based on their realized DOWN_ASY each month: downside asymmetric 
comovement (DOWN_ASY), CAPM beta (β), downside beta (β−), upside beta (β+), 
natural log of market capitalization (SIZE), natural log of book-to-market 
ratio (BM), turnover ratio (TURN), normalized Amihud illiquidity measure 
(ILLIQ), past-6-month return (MOM), idiosyncratic volatility (IVOL), 
coskewness (COSKEW), cokurtosis (COKURT), and maximum daily return over 
the past 1 month (MAX)."

Table Structure:
- Rows: Deciles 1-10 (sorted by DOWN_ASY)
- Columns: 13 characteristics (DOWN_ASY, β, β−, β+, SIZE, BM, TURN, ILLIQ, 
           MOM, IVOL, COSKEW, COKURT, MAX)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
import sys

sys.path.append(str(Path(__file__).parent.parent))


# Variable order as in the paper
VARIABLE_ORDER = [
    'DOWN_ASY',      # Downside asymmetric comovement
    'BETA',          # CAPM beta
    'DOWNSIDE_BETA', # Downside beta
    'UPSIDE_BETA',   # Upside beta
    'BETA_DIFF',     # DOWNSIDE_BETA - UPSIDE_BETA
    'SIZE',          # Natural log of market capitalization
    'BM',            # Natural log of book-to-market ratio
    'TURN',          # Turnover ratio
    'ILLIQ',         # Normalized Amihud illiquidity measure
    'MOM',           # Past-6-month return
    'IVOL',          # Idiosyncratic volatility
    'COSKEW',        # Coskewness
    'COKURT',        # Cokurtosis
    'MAX'            # Maximum daily return over past 1 month
]

# Paper-style column labels
COLUMN_LABELS = {
    'DOWN_ASY': 'DOWN_ASY',
    'BETA': 'BETA',
    'DOWNSIDE_BETA': 'DOWNSIDE_BETA',
    'UPSIDE_BETA': 'UPSIDE_BETA',
    'BETA_DIFF': 'DOWNSIDE_BETA - UPSIDE_BETA',
    'SIZE': 'SIZE',
    'BM': 'B/M',
    'TURN': 'TURN',
    'ILLIQ': 'ILLIQ',
    'MOM': 'MOM',
    'IVOL': 'IVOL',
    'COSKEW': 'COSKEW',
    'COKURT': 'COKURT',
    'MAX': 'MAX'
}


def create_demo_data(n_stocks: int = 500, n_months: int = 120) -> pd.DataFrame:
    """
    Create realistic demo data for Table 4 generation.
    
    Generates panel data with:
    - 500 stocks × 120 months = 60,000 observations
    - 13 characteristics with realistic correlations
    
    Parameters
    ----------
    n_stocks : int
        Number of stocks per month
    n_months : int
        Number of months in sample
    
    Returns
    -------
    pd.DataFrame
        Panel data with PERMNO, DATE, and all 13 characteristics
    """
    np.random.seed(42)
    
    data = []
    
    for month in range(n_months):
        date = pd.Timestamp(f'2004-{(month % 12) + 1:02d}-28') + pd.DateOffset(years=month // 12)
        
        for stock in range(n_stocks):
            # Generate correlated characteristics
            # DOWN_ASY is the main sorting variable
            down_asy = np.random.normal(0.02, 0.08)
            
            # Beta characteristics - correlated with DOWN_ASY
            beta = 1.0 + 0.3 * down_asy + np.random.normal(0, 0.3)
            downside_beta = beta + 0.2 * (1 + down_asy) + np.random.normal(0, 0.2)
            upside_beta = beta - 0.1 * down_asy + np.random.normal(0, 0.2)
            
            # SIZE = log(Market Cap in millions), Paper Table 4 shows Decile 1 ~4.79
            # Negatively correlated with DOWN_ASY (small stocks have higher asymmetry)
            size = 5.0 - 1.5 * down_asy + np.random.normal(0, 1.0)
            
            # Book-to-market ratio - slightly positive correlation
            bm = -0.5 + 0.5 * down_asy + np.random.normal(0, 0.8)
            
            # Turnover - positively correlated with DOWN_ASY
            turn = 0.08 + 0.15 * down_asy + np.abs(np.random.normal(0, 0.05))
            
            # Illiquidity - positively correlated with DOWN_ASY (small stocks)
            illiq = 0.05 + 0.3 * down_asy + np.abs(np.random.normal(0, 0.1))
            
            # Momentum - slightly negative correlation with DOWN_ASY
            mom = 0.05 - 0.1 * down_asy + np.random.normal(0, 0.15)
            
            # Idiosyncratic volatility - positively correlated with DOWN_ASY
            ivol = 0.02 + 0.08 * down_asy + np.abs(np.random.normal(0, 0.01))
            
            # Coskewness - negatively correlated with DOWN_ASY
            coskew = -0.1 - 0.5 * down_asy + np.random.normal(0, 0.3)
            
            # Cokurtosis - positively correlated with DOWN_ASY
            cokurt = 0.5 + 1.5 * down_asy + np.random.normal(0, 1.0)
            
            # MAX (maximum daily return) - positively correlated with DOWN_ASY
            max_ret = 0.03 + 0.1 * down_asy + np.abs(np.random.normal(0, 0.02))
            
            data.append({
                'PERMNO': 10000 + stock,
                'DATE': date,
                'DOWN_ASY': down_asy,
                'BETA': beta,
                'DOWNSIDE_BETA': downside_beta,
                'UPSIDE_BETA': upside_beta,
                'SIZE': size,
                'BM': bm,
                'TURN': turn,
                'ILLIQ': illiq,
                'MOM': mom,
                'IVOL': ivol,
                'COSKEW': coskew,
                'COKURT': cokurt,
                'MAX': max_ret
            })
    
    return pd.DataFrame(data)


def generate_table4(data: pd.DataFrame,
                    n_deciles: int = 10,
                    output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Generate Table 4: Summary Statistics of Asymmetry Portfolios.
    
    Methodology (following the paper):
    1. Each month, sort stocks into deciles based on DOWN_ASY
    2. For each decile, compute the time-series average of cross-sectional
       mean characteristics
    
    Parameters
    ----------
    data : pd.DataFrame
        Panel data with all 13 characteristics
    n_deciles : int
        Number of deciles (default: 10)
    output_path : str, optional
        Path to save output CSV
    
    Returns
    -------
    pd.DataFrame
        Table 4 with decile rows and characteristic columns
    """
    print("=" * 80)
    print("GENERATING TABLE 4: SUMMARY STATISTICS OF ASYMMETRY PORTFOLIOS")
    print("=" * 80)
    
    # Make a copy to avoid modifying original
    data = data.copy()
    
    # Ensure DATE is datetime
    data['DATE'] = pd.to_datetime(data['DATE'])
    n_months = data['DATE'].nunique()
    
    print(f"\nData: {len(data)} observations across {n_months} months")
    
    # Step 1: Assign deciles within each month
    # We'll use a different approach to avoid losing the DATE column
    all_decile_data = []
    
    for date, group in data.groupby('DATE'):
        group = group.copy()
        try:
            group['DECILE'] = pd.qcut(
                group['DOWN_ASY'],
                q=n_deciles,
                labels=range(1, n_deciles + 1),
                duplicates='drop'
            )
        except ValueError:
            # Handle edge case with too few unique values
            group['DECILE'] = pd.cut(
                group['DOWN_ASY'],
                bins=n_deciles,
                labels=range(1, n_deciles + 1)
            )
        all_decile_data.append(group)
    
    data = pd.concat(all_decile_data, ignore_index=True)
    data['DECILE'] = data['DECILE'].astype(int)
    
    print(f"Stocks sorted into {n_deciles} deciles each month")
    
    # Step 2: For each month, compute cross-sectional mean for each decile
    # Compute BETA_DIFF = DOWNSIDE_BETA - UPSIDE_BETA
    if 'DOWNSIDE_BETA' in data.columns and 'UPSIDE_BETA' in data.columns:
        data['BETA_DIFF'] = data['DOWNSIDE_BETA'] - data['UPSIDE_BETA']
    
    # Then average across time
    available_vars = [v for v in VARIABLE_ORDER if v in data.columns]
    
    # Group by DATE and DECILE, compute means
    monthly_decile_means = data.groupby(['DATE', 'DECILE'])[available_vars].mean()
    
    # Average across time for each decile
    table4 = monthly_decile_means.groupby('DECILE').mean()
    
    # Rename index to just numeric
    table4.index = range(1, n_deciles + 1)
    table4.index.name = 'Decile'
    
    # Rename columns to paper-style labels
    column_mapping = {col: COLUMN_LABELS.get(col, col) for col in table4.columns}
    table4 = table4.rename(columns=column_mapping)
    
    # Display results
    print("\n" + "=" * 80)
    print("Table 4: Summary Statistics of Asymmetry Portfolios")
    print("=" * 80)
    print("\nDecile-sorted portfolio characteristics (time-series averages):\n")
    
    # Format for display
    display_table = table4.copy()
    for col in display_table.columns:
        if col in ['DOWN_ASY', 'BETA', 'DOWNSIDE_BETA', 'UPSIDE_BETA', 'DOWNSIDE_BETA - UPSIDE_BETA']:
            display_table[col] = display_table[col].apply(lambda x: f'{x:.3f}')
        elif col in ['SIZE', 'B/M', 'COSKEW']:
            display_table[col] = display_table[col].apply(lambda x: f'{x:.2f}')
        elif col in ['TURN', 'ILLIQ', 'IVOL', 'MAX']:
            display_table[col] = display_table[col].apply(lambda x: f'{x:.4f}')
        elif col in ['MOM']:
            display_table[col] = display_table[col].apply(lambda x: f'{x:.3f}')
        elif col in ['COKURT']:
            display_table[col] = display_table[col].apply(lambda x: f'{x:.2f}')
    
    print(display_table.to_string())
    
    # Show spread (High - Low)
    print("\n" + "-" * 40)
    print("High-Low Spread (Decile 10 - Decile 1):")
    print("-" * 40)
    for col in table4.columns:
        spread = table4.loc[10, col] - table4.loc[1, col]
        print(f"  {col:12s}: {spread:+.4f}")
    
    # Save to CSV
    if output_path:
        # Save with proper format
        output_df = table4.copy()
        output_df.to_csv(output_path)
        print(f"\nTable saved to: {output_path}")
        
        # Also save LaTeX version
        latex_path = output_path.replace('.csv', '.tex')
        save_table4_latex(table4, latex_path)
    
    return table4


def save_table4_latex(table: pd.DataFrame, output_path: str):
    """
    Save Table 4 in LaTeX format matching the paper.
    
    Parameters
    ----------
    table : pd.DataFrame
        Table 4 data
    output_path : str
        Path for LaTeX output
    """
    # Build LaTeX table
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Summary Statistics of Asymmetry Portfolios}",
        r"\label{tab:summary_stats}",
        r"\small",
        r"\begin{tabular}{l" + "r" * len(table.columns) + "}",
        r"\hline\hline"
    ]
    
    # Header row
    header = "Decile & " + " & ".join(table.columns) + r" \\"
    lines.append(header)
    lines.append(r"\hline")
    
    # Data rows
    for decile in table.index:
        row_data = []
        for col in table.columns:
            val = table.loc[decile, col]
            if col == 'DOWN_ASY':
                row_data.append(f"{val:.3f}")
            elif col in ['BETA', 'DOWNSIDE_BETA', 'UPSIDE_BETA', 'DOWNSIDE_BETA - UPSIDE_BETA']:
                row_data.append(f"{val:.2f}")
            elif col in ['SIZE', 'B/M']:
                row_data.append(f"{val:.2f}")
            elif col in ['TURN', 'ILLIQ', 'IVOL', 'MAX']:
                row_data.append(f"{val:.4f}")
            elif col in ['MOM']:
                row_data.append(f"{val:.3f}")
            elif col in ['COSKEW']:
                row_data.append(f"{val:.3f}")
            elif col in ['COKURT']:
                row_data.append(f"{val:.2f}")
            else:
                row_data.append(f"{val:.3f}")
        
        lines.append(f"{decile} & " + " & ".join(row_data) + r" \\")
    
    lines.append(r"\hline\hline")
    lines.append(r"\end{tabular}")
    
    # Table notes
    lines.append(r"\begin{tablenotes}")
    lines.append(r"\small")
    lines.append(r"\item This table reports various risk measures and firm characteristics of deciles sorted based on their realized DOWN\_ASY each month.")
    lines.append(r"\item Variables: DOWN\_ASY = downside asymmetric comovement; $\beta$ = CAPM beta; $\beta^-$ = downside beta; $\beta^+$ = upside beta;")
    lines.append(r"\item SIZE = log market capitalization; B/M = log book-to-market; TURN = turnover; ILLIQ = Amihud illiquidity;")
    lines.append(r"\item MOM = past 6-month return; IVOL = idiosyncratic volatility; COSKEW = coskewness; COKURT = cokurtosis; MAX = maximum daily return.")
    lines.append(r"\end{tablenotes}")
    lines.append(r"\end{table}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"LaTeX table saved to: {output_path}")


def main():
    """
    Main execution for standalone use.
    """
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data' / 'processed'
    output_dir = base_dir / 'outputs' / 'tables'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("TABLE 4: SUMMARY STATISTICS OF ASYMMETRY PORTFOLIOS")
    print("=" * 80)
    
    # Try to load real data, otherwise generate demo
    asy_path = data_dir / 'down_asy_scores.parquet'
    char_path = data_dir / 'firm_characteristics.parquet'
    
    if asy_path.exists() and char_path.exists():
        print("\nLoading real data...")
        down_asy = pd.read_parquet(asy_path)
        characteristics = pd.read_parquet(char_path)
        
        # Merge
        data = down_asy.merge(characteristics, on=['PERMNO', 'DATE'], how='inner')
        print(f"  Loaded {len(data)} observations")
    else:
        print("\nGenerating demo data...")
        data = create_demo_data(n_stocks=500, n_months=120)
        print(f"  Generated {len(data)} observations")
    
    # Generate Table 4
    output_path = str(output_dir / 'Table_4_Summary_Stats.csv')
    table4 = generate_table4(data, n_deciles=10, output_path=output_path)
    
    print("\n" + "=" * 80)
    print("TABLE 4 GENERATION COMPLETE")
    print("=" * 80)
    
    return table4


if __name__ == "__main__":
    main()
