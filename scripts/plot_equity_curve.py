#!/usr/bin/env python3
"""
scripts/plot_equity_curve.py
Generate Figure 5: Cumulative Wealth (Equity Curve)

Shows the growth of $1 invested in:
- Low Asymmetry portfolio (Blue): Smooth upward trajectory
- High Asymmetry portfolio (Red): Sharp drops during crashes

Key insight: The "Low Asymmetry" portfolio is safer during market crashes,
while "High Asymmetry" portfolio earns a premium for bearing crash risk.

Author: Entropy Replication Project
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_timeseries_data() -> pd.DataFrame:
    """Load time-series returns data."""
    data_file = Path(__file__).resolve().parents[1] / "outputs" / "tables" / "Time_Series_Returns.csv"
    
    if not data_file.exists():
        print("Time-series data not found. Generating...")
        from generate_timeseries_data import create_demo_timeseries
        df = create_demo_timeseries()
        df.to_csv(data_file, index=False, date_format='%Y-%m')
    else:
        df = pd.read_csv(data_file, parse_dates=['Date'])
    
    return df


def compute_cumulative_wealth(returns: pd.Series, initial: float = 1.0) -> pd.Series:
    """
    Compute cumulative wealth from return series.
    
    Parameters
    ----------
    returns : pd.Series
        Monthly returns (decimal, not percentage).
    initial : float
        Initial investment value.
    
    Returns
    -------
    pd.Series
        Cumulative wealth series.
    """
    return initial * (1 + returns).cumprod()


def plot_equity_curve(df: pd.DataFrame, output_file: str, log_scale: bool = True) -> None:
    """
    Generate Figure 5: Cumulative Wealth Plot.
    
    Parameters
    ----------
    df : pd.DataFrame
        Time-series data with Date, Ret_Low_Asy, Ret_High_Asy columns.
    output_file : str
        Output file path.
    log_scale : bool
        Use logarithmic y-axis (recommended for long periods).
    """
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Compute cumulative wealth
    wealth_low = compute_cumulative_wealth(df['Ret_Low_Asy'])
    wealth_high = compute_cumulative_wealth(df['Ret_High_Asy'])
    
    # Also compute market if available
    if 'Mkt_Ret' in df.columns:
        wealth_mkt = compute_cumulative_wealth(df['Mkt_Ret'])
    else:
        wealth_mkt = None
    
    # Plot lines
    dates = df['Date']
    
    # Low Asymmetry (Blue) - Safe
    ax.plot(dates, wealth_low, 
            color='#2166AC', linewidth=2.0, 
            label='Low Asymmetry (Safe)', zorder=3)
    
    # High Asymmetry (Red) - Risky
    ax.plot(dates, wealth_high, 
            color='#B2182B', linewidth=2.0, 
            label='High Asymmetry (Risky)', zorder=3)
    
    # Market (Gray dashed) - optional
    if wealth_mkt is not None:
        ax.plot(dates, wealth_mkt, 
                color='#666666', linewidth=1.5, linestyle='--',
                label='Market', alpha=0.7, zorder=2)
    
    # Highlight crisis periods with shaded regions
    crises = [
        ('1973-10-01', '1975-03-01', '1973-74\nOil Crisis'),
        ('1987-08-01', '1987-12-01', '1987\nBlack Monday'),
        ('2000-03-01', '2002-10-01', '2000-02\nDot-com'),
        ('2007-12-01', '2009-06-01', '2008-09\nFinancial Crisis'),
    ]
    
    y_min, y_max = ax.get_ylim()
    
    for start, end, label in crises:
        try:
            start_date = pd.to_datetime(start)
            end_date = pd.to_datetime(end)
            if start_date >= dates.min() and start_date <= dates.max():
                ax.axvspan(start_date, end_date, 
                          alpha=0.15, color='gray', zorder=1)
                # Add label at top
                mid_date = start_date + (end_date - start_date) / 2
                if log_scale:
                    label_y = y_max * 0.8
                else:
                    label_y = y_max * 0.95
        except:
            pass
    
    # Formatting
    if log_scale:
        ax.set_yscale('log')
        ax.set_ylabel('Value of $1 Invested (Log Scale)', fontsize=12)
    else:
        ax.set_ylabel('Value of $1 Invested', fontsize=12)
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_title('Figure 5: Cumulative Returns of Asymmetry-Sorted Portfolios', 
                fontsize=14, fontweight='bold')
    
    # Format x-axis
    ax.xaxis.set_major_locator(mdates.YearLocator(10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_minor_locator(mdates.YearLocator(5))
    
    # Legend
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
    
    # Add annotation for final values
    final_low = wealth_low.iloc[-1]
    final_high = wealth_high.iloc[-1]
    
    # Text box with final statistics
    stats_text = (
        f"Final Value of $1:\n"
        f"  Low Asymmetry:  ${final_low:.2f}\n"
        f"  High Asymmetry: ${final_high:.2f}\n"
        f"  Premium:        ${final_high - final_low:.2f}"
    )
    
    # Position text box
    props = dict(boxstyle='round,pad=0.5', facecolor='white', 
                 edgecolor='gray', alpha=0.9)
    ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=props, family='monospace')
    
    # Grid
    ax.grid(True, alpha=0.3, which='major')
    ax.grid(True, alpha=0.15, which='minor')
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_file, dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"Saved: {output_file}")
    
    # Also save PNG version
    png_file = output_file.replace('.pdf', '.png')
    plt.savefig(png_file, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Saved: {png_file}")
    
    plt.close()


def validate_crisis_behavior(df: pd.DataFrame) -> bool:
    """
    Validate that High Asymmetry underperforms during crises.
    
    Returns
    -------
    bool
        True if validation passes.
    """
    print("\n" + "="*60)
    print("VALIDATION: Crisis Behavior")
    print("="*60)
    
    # Check 2008 Financial Crisis (if in data range)
    crisis_mask = (df['Date'] >= '2008-09-01') & (df['Date'] <= '2009-03-01')
    
    if crisis_mask.sum() > 0:
        crisis_ret_low = df.loc[crisis_mask, 'Ret_Low_Asy'].sum() * 100
        crisis_ret_high = df.loc[crisis_mask, 'Ret_High_Asy'].sum() * 100
        
        print(f"\n2008-09 Financial Crisis Returns:")
        print(f"  Low Asymmetry:  {crisis_ret_low:.1f}%")
        print(f"  High Asymmetry: {crisis_ret_high:.1f}%")
        print(f"  Spread:         {crisis_ret_high - crisis_ret_low:.1f}%")
        
        if crisis_ret_high < crisis_ret_low:
            print("\n  [PASS] High Asymmetry underperforms during crisis (as expected)")
            return True
        else:
            print("\n  [WARN] High Asymmetry did not underperform during crisis")
            return False
    else:
        print("\n  [SKIP] 2008 crisis not in sample period")
        return True


def main():
    """Main execution function."""
    print("="*70)
    print("GENERATING FIGURE 5: CUMULATIVE WEALTH (EQUITY CURVE)")
    print("="*70)
    
    OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "figures"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\nLoading time-series data...")
    df = load_timeseries_data()
    print(f"Period: {df['Date'].min().strftime('%Y-%m')} to {df['Date'].max().strftime('%Y-%m')}")
    print(f"Months: {len(df)}")
    
    # Generate figure
    print("\nGenerating equity curve...")
    output_file = str(OUTPUT_DIR / "Figure_5_Cumulative_Returns.pdf")
    plot_equity_curve(df, output_file, log_scale=True)
    
    # Validate crisis behavior
    validate_crisis_behavior(df)
    
    print("\n" + "="*70)
    print("FIGURE 5 GENERATION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
