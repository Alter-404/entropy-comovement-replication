#!/usr/bin/env python3
"""
Extension 1: Crisis Performance Analysis - Master Runner
=========================================================

Executes all Extension 1 analyses:
    1. Generate Crisis Indicators (Definition A, B, C)
    2. Time-Series Regime Regression (Table 8 Panel A)
    3. Panel Interaction Regression (Table 8 Panel B)
    4. Crisis Performance Visualization (Figure 8)
    5. Summary Interpretation

Usage:
    python scripts/run_extension1.py --demo  # Quick demo
    python scripts/run_extension1.py         # Full analysis
"""

import sys
from pathlib import Path
from datetime import datetime

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))


def run_extension1(demo_mode: bool = False):
    """
    Run complete Extension 1 analysis.
    
    Parameters
    ----------
    demo_mode : bool
        If True, use synthetic data for quick demonstration
    """
    start_time = datetime.now()
    
    print("\n" + "="*70)
    print("  EXTENSION 1: CRISIS PERFORMANCE ANALYSIS")
    print("  " + "="*66)
    print(f"  Mode: {'DEMO' if demo_mode else 'PRODUCTION'}")
    print(f"  Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Setup paths
    project_root = Path(__file__).parent.parent
    processed_dir = project_root / 'data' / 'processed'
    tables_dir = project_root / 'outputs' / 'tables'
    figures_dir = project_root / 'outputs' / 'figures'
    
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # Step 1: Generate Crisis Indicators
    print("\n" + "-"*70)
    print("  STEP 1/5: Generating Crisis Indicators")
    print("-"*70)
    
    try:
        from extension1_crisis_indicators import generate_crisis_flags
        crisis_flags = generate_crisis_flags(
            output_dir=tables_dir,
            demo_mode=demo_mode
        )
        results['crisis_flags'] = True
        print("\n  ✓ Crisis Flags: SUCCESS")
    except Exception as e:
        results['crisis_flags'] = False
        print(f"\n  ✗ Crisis Flags: FAILED - {e}")
    
    # Step 2: Time-Series Regime Regression
    print("\n" + "-"*70)
    print("  STEP 2/5: Time-Series Regime Regression")
    print("-"*70)
    
    try:
        from extension1_timeseries_reg import generate_table8_panelA
        table8a = generate_table8_panelA(
            processed_dir=processed_dir,
            tables_dir=tables_dir,
            demo_mode=demo_mode
        )
        results['table8_panelA'] = True
        print("\n  ✓ Table 8 Panel A: SUCCESS")
    except Exception as e:
        results['table8_panelA'] = False
        print(f"\n  ✗ Table 8 Panel A: FAILED - {e}")
    
    # Step 3: Panel Interaction Regression
    print("\n" + "-"*70)
    print("  STEP 3/5: Panel Interaction Regression")
    print("-"*70)
    
    try:
        from extension1_panel_reg import generate_table8_panelB
        table8b = generate_table8_panelB(
            processed_dir=processed_dir,
            tables_dir=tables_dir,
            demo_mode=demo_mode
        )
        results['table8_panelB'] = True
        print("\n  ✓ Table 8 Panel B: SUCCESS")
    except Exception as e:
        results['table8_panelB'] = False
        print(f"\n  ✗ Table 8 Panel B: FAILED - {e}")
    
    # Step 4: Crisis Performance Visualization
    print("\n" + "-"*70)
    print("  STEP 4/5: Crisis Performance Visualization")
    print("-"*70)
    
    try:
        from extension1_plot_drawdowns import generate_figure8
        fig8_path = generate_figure8(
            processed_dir=processed_dir,
            tables_dir=tables_dir,
            figures_dir=figures_dir,
            demo_mode=demo_mode
        )
        results['figure8'] = True
        print("\n  ✓ Figure 8: SUCCESS")
    except Exception as e:
        results['figure8'] = False
        print(f"\n  ✗ Figure 8: FAILED - {e}")
    
    # Step 5: Generate Summary
    print("\n" + "-"*70)
    print("  STEP 5/5: Generating Summary Interpretation")
    print("-"*70)
    
    try:
        from extension1_plot_drawdowns import generate_crisis_summary
        summary_path = generate_crisis_summary(
            processed_dir=processed_dir,
            tables_dir=tables_dir,
            output_dir=project_root / 'outputs',
            demo_mode=demo_mode
        )
        results['summary'] = True
        print("\n  ✓ Summary: SUCCESS")
    except Exception as e:
        results['summary'] = False
        print(f"\n  ✗ Summary: FAILED - {e}")
    
    # Final report
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print("\n" + "="*70)
    print("  EXTENSION 1 ANALYSIS COMPLETE")
    print("="*70)
    
    print(f"\n  Duration: {duration:.1f} seconds")
    print(f"\n  Results Summary:")
    
    all_success = True
    for step, success in results.items():
        status = "✓" if success else "✗"
        print(f"    [{status}] {step}")
        all_success = all_success and success
    
    print("\n  Outputs Generated:")
    print(f"    Tables:  {tables_dir}")
    print(f"      - Crisis_Flags.csv")
    print(f"      - Table_8_PanelA_TimeSeries.csv")
    print(f"      - Table_8_PanelB_Panel.csv")
    print(f"    Figures: {figures_dir}")
    print(f"      - Figure_8_Crisis_Performance.pdf")
    print(f"    Summary: {project_root / 'outputs'}")
    print(f"      - Extension1_Crisis_Summary.txt")
    
    if all_success:
        print("\n  " + "="*66)
        print("  ALL STEPS COMPLETED SUCCESSFULLY")
        print("  " + "="*66)
    else:
        print("\n  " + "="*66)
        print("  SOME STEPS FAILED - Check logs above")
        print("  " + "="*66)
    
    return all_success


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Extension 1: Crisis Performance Analysis'
    )
    parser.add_argument(
        '--demo', 
        action='store_true',
        help='Run with synthetic demo data'
    )
    args = parser.parse_args()
    
    success = run_extension1(demo_mode=args.demo)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
