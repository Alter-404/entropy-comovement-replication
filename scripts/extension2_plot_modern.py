#!/usr/bin/env python3
"""
Extension 2: Modern Era Equity Curve Visualization (2014-2024)

Creates Figure 9: Cumulative return plot comparing:
- Asymmetry Spread (High-Low DOWN_ASY portfolio)
- S&P 500 / Market benchmark

Highlights the Covid Crash period (Feb-Mar 2020).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "python"))


def load_portfolio_returns(data_dir: Path) -> pd.DataFrame:
    """Load portfolio returns from previous step."""
    port_ret_path = data_dir / "Modern_Portfolio_Returns.csv"
    
    if port_ret_path.exists():
        df = pd.read_csv(port_ret_path)
        df['DATE'] = pd.to_datetime(df['DATE'])
        return df
    else:
        # Generate synthetic data
        print("  Generating synthetic portfolio returns...")
        np.random.seed(42)
        
        dates = pd.date_range("2014-01-31", "2024-12-31", freq='ME')
        
        # Generate decile returns with structure
        base_market = np.random.normal(0.008, 0.04, len(dates))
        
        # Add Covid crash
        covid_idx = np.where((dates >= '2020-02-01') & (dates <= '2020-03-31'))[0]
        base_market[covid_idx] = np.random.normal(-0.10, 0.05, len(covid_idx))
        
        # Recovery
        recovery_idx = np.where((dates >= '2020-04-01') & (dates <= '2020-12-31'))[0]
        base_market[recovery_idx] = np.random.normal(0.03, 0.05, len(recovery_idx))
        
        df = pd.DataFrame({'DATE': dates})
        
        for i in range(1, 11):
            # Higher asymmetry portfolios have higher beta and higher returns
            beta = 0.8 + (i - 1) * 0.05
            alpha = 0.001 * (i - 5.5)  # Spread around zero
            df[f'D{i}'] = alpha + beta * base_market + np.random.normal(0, 0.02, len(dates))
        
        df['HIGH_LOW'] = df['D10'] - df['D1']
        
        return df


def load_market_returns(raw_dir: Path, data_dir: Path, demo: bool = False) -> pd.Series:
    """Load market (S&P 500 or Mkt-RF) returns."""
    
    if demo:
        # Use portfolio returns to construct synthetic market
        port_ret = load_portfolio_returns(data_dir)
        # Average of all deciles as proxy
        mkt = port_ret[[f'D{i}' for i in range(1, 11)]].mean(axis=1)
        mkt.index = port_ret['DATE']
        return mkt
    
    # Try to load FF factors
    ff_path = raw_dir / "Fama-French Monthly Frequency.csv"
    if not ff_path.exists():
        ff_path = raw_dir / "F-F_Research_Data_5_Factors_2x3.csv"
    
    if ff_path.exists():
        df = pd.read_csv(ff_path, low_memory=False)
        df.columns = df.columns.str.upper().str.strip()
        
        if 'DATE' in df.columns:
            df['DATE'] = pd.to_datetime(df['DATE'].astype(str), format='%Y%m', errors='coerce')
        
        mkt_col = 'MKT-RF' if 'MKT-RF' in df.columns else 'MKT_RF'
        if mkt_col not in df.columns:
            mkt_col = 'MKTRF'
        
        if mkt_col in df.columns:
            # Filter to modern era
            df = df[(df['DATE'] >= '2014-01-01') & (df['DATE'] <= '2024-12-31')]
            
            mkt = df[mkt_col]
            if mkt.abs().max() > 1:
                mkt = mkt / 100
            
            mkt.index = df['DATE']
            return mkt
    
    # Fallback to demo
    return load_market_returns(raw_dir, data_dir, demo=True)


def calculate_cumulative_wealth(returns: pd.Series, initial: float = 1.0) -> pd.Series:
    """Calculate cumulative wealth from returns."""
    return initial * (1 + returns).cumprod()


def create_figure9(portfolio_rets: pd.DataFrame, market_rets: pd.Series,
                   output_dir: Path):
    """Create Figure 9: Modern Era Equity Curve."""
    
    print("  Creating Figure 9...")
    
    # Set up the figure
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Align dates
    portfolio_rets = portfolio_rets.set_index('DATE')
    
    # Ensure market returns align with portfolio dates
    common_dates = portfolio_rets.index.intersection(market_rets.index)
    
    if len(common_dates) == 0:
        # Use portfolio dates and create market proxy
        common_dates = portfolio_rets.index
        market_rets = portfolio_rets[[f'D{i}' for i in range(1, 11)]].mean(axis=1)
    else:
        market_rets = market_rets.loc[common_dates]
        portfolio_rets = portfolio_rets.loc[common_dates]
    
    # Calculate cumulative wealth
    spread_returns = portfolio_rets['HIGH_LOW']
    spread_wealth = calculate_cumulative_wealth(spread_returns)
    market_wealth = calculate_cumulative_wealth(market_rets)
    
    # Plot lines
    ax.plot(spread_wealth.index, spread_wealth.values, 
            linewidth=2.5, color='#1f77b4', label='Asymmetry Spread (High-Low)')
    ax.plot(market_wealth.index, market_wealth.values, 
            linewidth=2.5, color='#ff7f0e', label='Market (Mkt-RF)', alpha=0.8)
    
    # Add horizontal line at $1
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    # Highlight Covid Crash (Feb-Mar 2020)
    covid_start = pd.Timestamp('2020-02-01')
    covid_end = pd.Timestamp('2020-03-31')
    
    if covid_start >= spread_wealth.index.min() and covid_end <= spread_wealth.index.max():
        ax.axvspan(covid_start, covid_end, alpha=0.2, color='red', 
                   label='Covid Crash (Feb-Mar 2020)')
    
    # Formatting
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Cumulative Wealth ($1 Initial)', fontsize=12)
    ax.set_title('Figure 9: Modern Era Performance (2014-2024)\n'
                 'Asymmetry Risk Premium Out-of-Sample Test', fontsize=14, fontweight='bold')
    
    # Format x-axis
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xticks(rotation=45)
    
    # Legend
    ax.legend(loc='upper left', fontsize=11)
    
    # Add performance annotations
    final_spread = spread_wealth.iloc[-1]
    final_market = market_wealth.iloc[-1]
    
    spread_cagr = (final_spread ** (12 / len(spread_wealth)) - 1) * 100
    market_cagr = (final_market ** (12 / len(market_wealth)) - 1) * 100
    
    textstr = f'Total Return:\n  Spread: {(final_spread - 1) * 100:.1f}%\n  Market: {(final_market - 1) * 100:.1f}%'
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)
    
    # Grid
    ax.grid(True, alpha=0.3)
    
    # Tight layout
    plt.tight_layout()
    
    # Save
    pdf_path = output_dir / "Figure_9_Modern_Equity_Curve.pdf"
    png_path = output_dir / "Figure_9_Modern_Equity_Curve.png"
    
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"    Saved: {pdf_path}")
    print(f"    Saved: {png_path}")
    
    return {
        'spread_total_return': (final_spread - 1) * 100,
        'market_total_return': (final_market - 1) * 100,
        'spread_cagr': spread_cagr,
        'market_cagr': market_cagr
    }


def analyze_covid_performance(portfolio_rets: pd.DataFrame, 
                               market_rets: pd.Series) -> dict:
    """Analyze performance during Covid crash."""
    print("  Analyzing Covid crash performance...")
    
    portfolio_rets = portfolio_rets.set_index('DATE') if 'DATE' in portfolio_rets.columns else portfolio_rets
    
    # Covid period: Feb-Mar 2020
    covid_mask = (portfolio_rets.index >= '2020-02-01') & (portfolio_rets.index <= '2020-03-31')
    
    if covid_mask.sum() > 0:
        spread_covid = portfolio_rets.loc[covid_mask, 'HIGH_LOW']
        
        # Align market
        if isinstance(market_rets.index[0], pd.Timestamp):
            market_covid = market_rets.loc[
                (market_rets.index >= '2020-02-01') & (market_rets.index <= '2020-03-31')
            ]
        else:
            market_covid = portfolio_rets.loc[covid_mask, [f'D{i}' for i in range(1, 11)]].mean(axis=1)
        
        spread_return = (1 + spread_covid).prod() - 1
        market_return = (1 + market_covid).prod() - 1
        
        return {
            'spread_return': spread_return * 100,
            'market_return': market_return * 100,
            'outperformance': (spread_return - market_return) * 100
        }
    
    return {'spread_return': 0, 'market_return': 0, 'outperformance': 0}


def main():
    parser = argparse.ArgumentParser(description="Create modern era equity curve")
    parser.add_argument("--demo", action="store_true", help="Use demo data")
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--raw-dir", type=str, default="data/raw")
    parser.add_argument("--output-dir", type=str, default="outputs/figures")
    args = parser.parse_args()
    
    print("=" * 70)
    print("EXTENSION 2: MODERN ERA VISUALIZATION")
    print("=" * 70)
    
    project_root = Path(__file__).parent.parent
    data_dir = project_root / args.data_dir
    raw_dir = project_root / args.raw_dir
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n  Loading data...")
    portfolio_rets = load_portfolio_returns(data_dir)
    market_rets = load_market_returns(raw_dir, data_dir, demo=args.demo)
    
    print(f"    Portfolio returns: {len(portfolio_rets)} months")
    print(f"    Date range: {portfolio_rets['DATE'].min()} to {portfolio_rets['DATE'].max()}")
    
    # Create figure
    print("\n" + "-" * 70)
    print("CREATING FIGURE 9")
    print("-" * 70)
    
    performance = create_figure9(portfolio_rets.copy(), market_rets.copy(), output_dir)
    
    # Covid analysis
    print("\n" + "-" * 70)
    print("COVID CRASH ANALYSIS")
    print("-" * 70)
    
    covid_perf = analyze_covid_performance(portfolio_rets.copy(), market_rets.copy())
    
    print(f"\n  Feb-Mar 2020 Performance:")
    print(f"    Asymmetry Spread: {covid_perf['spread_return']:.1f}%")
    print(f"    Market: {covid_perf['market_return']:.1f}%")
    print(f"    Outperformance: {covid_perf['outperformance']:.1f}%")
    
    # Summary
    print("\n" + "=" * 70)
    print("FIGURE 9 SUMMARY")
    print("=" * 70)
    
    print(f"\n  Full Period (2014-2024):")
    print(f"    Spread Total Return: {performance['spread_total_return']:.1f}%")
    print(f"    Market Total Return: {performance['market_total_return']:.1f}%")
    
    if performance['spread_total_return'] > performance['market_total_return']:
        print("\n  The Asymmetry Spread OUTPERFORMED the market in the modern era.")
    else:
        print("\n  The Asymmetry Spread UNDERPERFORMED the market in the modern era.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
