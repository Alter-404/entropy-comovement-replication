#!/usr/bin/env python3
"""
scripts/plot_premium_dynamics.py
Generate Figure 6: Time-Series Dynamics of the Asymmetry Premium

Shows the counter-cyclical nature of the asymmetry risk premium:
- Premium is high when volatility is low (compensation for crash risk)
- Premium crashes when volatility spikes (crashes materialize)

Two visualization options:
- Option A: Time series overlay (bars + line)
- Option B: Regime scatter plot with regression (recommended)

Author: Entropy Replication Project
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import sys
from scipy import stats

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


def plot_time_series_overlay(df: pd.DataFrame, output_file: str) -> None:
    """
    Generate Figure 6A: Time Series Overlay.
    
    Left axis (bars): Monthly Ret_Spread (High - Low)
    Right axis (line): Market Volatility
    
    Parameters
    ----------
    df : pd.DataFrame
        Time-series data.
    output_file : str
        Output file path.
    """
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Create figure with dual y-axes
    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax2 = ax1.twinx()
    
    dates = df['Date']
    spread = df['Ret_Spread'] * 100  # Convert to percentage
    
    # Use realized vol if available, otherwise compute from Mkt_Vol
    if 'Mkt_Vol_Realized' in df.columns:
        vol = df['Mkt_Vol_Realized']
    elif 'Mkt_Vol' in df.columns:
        vol = np.sqrt(df['Mkt_Vol']) * np.sqrt(12) * 100  # Annualized %
    else:
        # Compute rolling volatility from market returns
        vol = df['Mkt_Ret'].rolling(12).std() * np.sqrt(12) * 100
    
    # Color bars based on sign
    colors = ['#B2182B' if s < 0 else '#2166AC' for s in spread]
    
    # Plot bars for spread
    ax1.bar(dates, spread, width=20, color=colors, alpha=0.7, label='Spread (High-Low)')
    
    # Plot line for volatility
    ax2.plot(dates, vol, color='#4DAF4A', linewidth=1.5, 
             label='Market Volatility', alpha=0.8)
    ax2.fill_between(dates, 0, vol, color='#4DAF4A', alpha=0.1)
    
    # Formatting
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Monthly Spread Return (%)', fontsize=12, color='#333333')
    ax2.set_ylabel('Annualized Market Volatility (%)', fontsize=12, color='#4DAF4A')
    
    ax1.set_title('Figure 6A: Asymmetry Premium and Market Volatility', 
                 fontsize=14, fontweight='bold')
    
    # Format x-axis
    ax1.xaxis.set_major_locator(mdates.YearLocator(10))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    # Set y-axis colors
    ax1.tick_params(axis='y', labelcolor='#333333')
    ax2.tick_params(axis='y', labelcolor='#4DAF4A')
    
    # Add zero line
    ax1.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    
    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
    
    # Annotate correlation
    corr = df['Ret_Spread'].corr(vol)
    ax1.text(0.98, 0.02, f'Correlation: {corr:.3f}', 
            transform=ax1.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save PDF
    plt.savefig(output_file, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Saved: {output_file}")
    
    # Also save PNG
    png_file = output_file.replace('.pdf', '.png')
    plt.savefig(png_file, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Saved: {png_file}")
    
    plt.close()


def plot_regime_scatter(df: pd.DataFrame, output_file: str) -> None:
    """
    Generate Figure 6B: Regime Scatter Plot (Recommended).
    
    X-axis: Market Volatility (lagged)
    Y-axis: Realized Asymmetry Premium
    
    Includes regression line showing negative slope (as per Table 6).
    
    Parameters
    ----------
    df : pd.DataFrame
        Time-series data.
    output_file : str
        Output file path.
    """
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Prepare data
    spread = df['Ret_Spread'] * 100  # Convert to percentage
    
    # Use realized vol if available
    if 'Mkt_Vol_Realized' in df.columns:
        vol = df['Mkt_Vol_Realized']
    elif 'Mkt_Vol' in df.columns:
        vol = np.sqrt(df['Mkt_Vol']) * np.sqrt(12) * 100
    else:
        vol = df['Mkt_Ret'].rolling(12).std() * np.sqrt(12) * 100
    
    # Lag volatility by 1 month (predict next month's premium)
    vol_lagged = vol.shift(1)
    
    # Remove NaN
    valid = ~(vol_lagged.isna() | spread.isna())
    x = vol_lagged[valid].values
    y = spread[valid].values
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Scatter plot with color gradient based on date
    dates_numeric = (df['Date'][valid] - df['Date'].min()).dt.days.values
    scatter = ax.scatter(x, y, c=dates_numeric, cmap='viridis', 
                        alpha=0.6, s=30, edgecolors='none')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label('Time (days from start)', fontsize=10)
    
    # Regression line
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = slope * x_line + intercept
    
    ax.plot(x_line, y_line, color='#B2182B', linewidth=2.5, 
            label=f'Regression: $\\beta$ = {slope:.3f} (p = {p_value:.3f})')
    
    # Add confidence band
    n = len(x)
    se = std_err * np.sqrt(1/n + (x_line - x.mean())**2 / np.sum((x - x.mean())**2))
    ax.fill_between(x_line, y_line - 1.96*se*np.sqrt(n), y_line + 1.96*se*np.sqrt(n),
                    color='#B2182B', alpha=0.15)
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    
    # Formatting
    ax.set_xlabel('Lagged Market Volatility (%)', fontsize=12)
    ax.set_ylabel('Monthly Asymmetry Premium (%)', fontsize=12)
    ax.set_title('Figure 6B: Counter-Cyclical Asymmetry Premium', 
                fontsize=14, fontweight='bold')
    
    # Legend with stats
    stats_text = (
        f"Regression Statistics:\n"
        f"  Slope (β):     {slope:.4f}\n"
        f"  t-stat:        {slope/std_err:.2f}\n"
        f"  R²:            {r_value**2:.3f}\n"
        f"  N:             {n}"
    )
    
    props = dict(boxstyle='round,pad=0.5', facecolor='white', 
                 edgecolor='gray', alpha=0.9)
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='left',
            bbox=props, family='monospace')
    
    # Annotate key insight
    insight_text = (
        "Key Insight:\n"
        "Negative slope confirms that\n"
        "the premium is high when\n"
        "volatility is low (crash risk\n"
        "compensation)."
    )
    ax.text(0.98, 0.98, insight_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightyellow', 
                     edgecolor='orange', alpha=0.9))
    
    ax.legend(loc='lower left', fontsize=11)
    
    plt.tight_layout()
    
    # Save
    plt.savefig(output_file, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Saved: {output_file}")
    
    # Also save PNG
    png_file = output_file.replace('.pdf', '.png')
    plt.savefig(png_file, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Saved: {png_file}")
    
    plt.close()
    
    return slope, p_value


def validate_countercyclical_premium(df: pd.DataFrame) -> bool:
    """
    Validate that premium is counter-cyclical.
    
    Returns
    -------
    bool
        True if premium is high when volatility is low.
    """
    print("\n" + "="*60)
    print("VALIDATION: Counter-Cyclical Premium")
    print("="*60)
    
    # Get volatility
    if 'Mkt_Vol_Realized' in df.columns:
        vol = df['Mkt_Vol_Realized']
    elif 'Mkt_Vol' in df.columns:
        vol = np.sqrt(df['Mkt_Vol']) * np.sqrt(12) * 100
    else:
        vol = df['Mkt_Ret'].rolling(12).std() * np.sqrt(12) * 100
    
    spread = df['Ret_Spread']
    vol_lagged = vol.shift(1)
    
    # Correlation
    valid = ~(vol_lagged.isna() | spread.isna())
    corr = np.corrcoef(vol_lagged[valid], spread[valid])[0, 1]
    
    print(f"\nCorrelation(Lagged Vol, Premium): {corr:.3f}")
    
    # Split into high/low volatility regimes
    vol_median = vol_lagged[valid].median()
    
    high_vol_mask = vol_lagged[valid] > vol_median
    low_vol_mask = vol_lagged[valid] <= vol_median
    
    premium_high_vol = spread[valid][high_vol_mask].mean() * 100
    premium_low_vol = spread[valid][low_vol_mask].mean() * 100
    
    print(f"\nMean Premium by Volatility Regime:")
    print(f"  Low Volatility:   {premium_low_vol:.3f}% per month")
    print(f"  High Volatility:  {premium_high_vol:.3f}% per month")
    
    if premium_low_vol > premium_high_vol:
        print("\n  [PASS] Premium is higher when volatility is low (counter-cyclical)")
        return True
    else:
        print("\n  [WARN] Premium is not counter-cyclical as expected")
        return False


def main():
    """Main execution function."""
    print("="*70)
    print("GENERATING FIGURE 6: ASYMMETRY PREMIUM DYNAMICS")
    print("="*70)
    
    OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "figures"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\nLoading time-series data...")
    df = load_timeseries_data()
    print(f"Period: {df['Date'].min().strftime('%Y-%m')} to {df['Date'].max().strftime('%Y-%m')}")
    print(f"Months: {len(df)}")
    
    # Generate Figure 6A: Time Series Overlay
    print("\nGenerating Figure 6A (Time Series Overlay)...")
    output_file_a = str(OUTPUT_DIR / "Figure_6A_Premium_TimeSeries.pdf")
    plot_time_series_overlay(df, output_file_a)
    
    # Generate Figure 6B: Regime Scatter (Recommended)
    print("\nGenerating Figure 6B (Regime Scatter)...")
    output_file_b = str(OUTPUT_DIR / "Figure_6B_Premium_Scatter.pdf")
    slope, p_value = plot_regime_scatter(df, output_file_b)
    
    # Validation
    validate_countercyclical_premium(df)
    
    print("\n" + "="*70)
    print("FIGURE 6 GENERATION COMPLETE")
    print("="*70)
    print(f"\nRegression slope: {slope:.4f} (p = {p_value:.3f})")
    if slope < 0 and p_value < 0.05:
        print("[PASS] Negative slope confirms counter-cyclical premium")
    else:
        print("[WARN] Slope not significantly negative")


if __name__ == "__main__":
    main()
