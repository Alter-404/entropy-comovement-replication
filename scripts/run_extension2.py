#!/usr/bin/env python3
"""
Extension 2: Out-of-Sample Test (2014-2024) - Master Runner

Tests whether the Asymmetry Risk Premium persists in the post-publication era,
independent from the original 1963-2013 sample.

Steps:
1. Data Preparation (filter 2014-2024, apply screens)
2. Factor Construction (BM, MOM, Size, merge with entropy)
3. Portfolio Testing (decile sorts, alpha estimation)
4. Visualization (cumulative returns, Covid highlight)
5. Summary Report

Outputs:
- Table 9: Modern Era Performance (Returns & Alphas)
- Figure 9: Cumulative Return Plot (2014-2024)
- Summary interpretation
"""

import subprocess
import sys
import time
from pathlib import Path
import argparse


def run_step(step_num: int, total_steps: int, name: str, script: str, 
             args: list, project_root: Path) -> bool:
    """Run a single step and report status."""
    
    print(f"\n{'=' * 70}")
    print(f"STEP {step_num}/{total_steps}: {name}")
    print("=" * 70)
    
    script_path = project_root / "scripts" / script
    
    if not script_path.exists():
        print(f"  ERROR: Script not found: {script_path}")
        return False
    
    cmd = [sys.executable, str(script_path)] + args
    
    try:
        start_time = time.time()
        result = subprocess.run(
            cmd,
            cwd=str(project_root),
            capture_output=False,
            text=True
        )
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"\n  [✓] {name} completed in {elapsed:.1f}s")
            return True
        else:
            print(f"\n  [✗] {name} failed (exit code {result.returncode})")
            return False
            
    except Exception as e:
        print(f"\n  [✗] {name} failed with error: {e}")
        return False


def generate_summary(project_root: Path, demo: bool):
    """Generate summary text file."""
    
    print("\n" + "=" * 70)
    print("STEP 5/5: GENERATING SUMMARY")
    print("=" * 70)
    
    output_dir = project_root / "outputs" / "summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Table 9 results
    table9_path = output_dir / "Table_9_Modern_Performance.csv"
    
    summary_lines = [
        "=" * 70,
        "EXTENSION 2: OUT-OF-SAMPLE TEST SUMMARY (2014-2024)",
        "=" * 70,
        "",
        "RESEARCH QUESTION:",
        "Does the Asymmetry Risk Premium documented by Jiang, Wu, and Zhou (2018)",
        "persist in the post-publication era (2014-2024)?",
        "",
        "-" * 70,
        "METHODOLOGY:",
        "-" * 70,
        "- Sample Period: January 2014 - December 2024 (11 years)",
        "- Universe: Common US equities (SHRCD 10, 11), price >= $5",
        "- Sorting: Monthly decile sorts on DOWN_ASY",
        "- Weighting: Value-weighted portfolio returns",
        "- Benchmarks: CAPM and Fama-French 5-Factor models",
        "",
        "-" * 70,
        "RESULTS:",
        "-" * 70,
    ]
    
    if table9_path.exists():
        import pandas as pd
        table9 = pd.read_csv(table9_path)
        
        for _, row in table9.iterrows():
            summary_lines.append(f"\n{row['Period']}:")
            summary_lines.append(f"  Raw Return: {row['Raw Return (%)']}% (t = {row['Raw t-stat']})")
            summary_lines.append(f"  CAPM Alpha: {row['CAPM Alpha (%)']}% (t = {row['CAPM t-stat']})")
            summary_lines.append(f"  FF5 Alpha:  {row['FF5 Alpha (%)']}% (t = {row['FF5 t-stat']})")
        
        # Interpretation
        full_row = table9[table9['Period'].str.contains('Full')].iloc[0]
        raw_t = float(full_row['Raw t-stat'])
        raw_ret = float(full_row['Raw Return (%)'])
        
        summary_lines.extend([
            "",
            "-" * 70,
            "INTERPRETATION:",
            "-" * 70,
        ])
        
        if raw_t > 2.0:
            summary_lines.extend([
                "",
                f"The High-Low asymmetry spread earns {raw_ret:.2f}% per month",
                f"with a t-statistic of {raw_t:.2f} (highly significant).",
                "",
                "CONCLUSION: The Asymmetry Risk Premium PERSISTS in the modern era.",
                "",
                "This suggests the premium is NOT a statistical artifact or the result",
                "of data mining. The economic mechanism underlying asymmetric comovement",
                "continues to generate a risk premium in out-of-sample data.",
            ])
        elif raw_t > 1.65:
            summary_lines.extend([
                "",
                f"The High-Low asymmetry spread earns {raw_ret:.2f}% per month",
                f"with a t-statistic of {raw_t:.2f} (marginally significant).",
                "",
                "CONCLUSION: WEAK evidence of persistence.",
                "",
                "The premium shows diminished but not eliminated economic significance.",
                "This could reflect either partial arbitrage or reduced importance of",
                "the asymmetry factor in the modern market environment.",
            ])
        else:
            summary_lines.extend([
                "",
                f"The High-Low asymmetry spread earns only {raw_ret:.2f}% per month",
                f"with a t-statistic of {raw_t:.2f} (not significant).",
                "",
                "CONCLUSION: The Asymmetry Risk Premium has DECAYED post-publication.",
                "",
                "This is consistent with two hypotheses:",
                "1. Arbitrage: Sophisticated investors traded away the anomaly",
                "2. Overfitting: The original finding was sample-specific",
                "",
                "Additional research is needed to distinguish between these explanations.",
            ])
    else:
        summary_lines.append("\n  [Table 9 not found - run portfolio test first]")
    
    # Covid Analysis
    summary_lines.extend([
        "",
        "-" * 70,
        "COVID CRASH ANALYSIS (Feb-Mar 2020):",
        "-" * 70,
        "",
        "The Covid crash provides a natural experiment to test whether",
        "high-asymmetry stocks provide 'insurance' during market crashes.",
        "",
        "See Figure 9 for the equity curve visualization.",
    ])
    
    if demo:
        summary_lines.extend([
            "",
            "-" * 70,
            "NOTE: This analysis used SYNTHETIC demo data.",
            "For publication-quality results, run with real CRSP data.",
            "-" * 70,
        ])
    
    summary_lines.extend([
        "",
        "=" * 70,
        "END OF EXTENSION 2 SUMMARY",
        "=" * 70,
    ])
    
    # Save summary
    summary_path = output_dir / "Extension2_OOS_Summary.txt"
    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary_lines))
    
    print(f"\n  Saved summary: {summary_path}")
    print("\n" + "\n".join(summary_lines))
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run Extension 2: Out-of-Sample Test (2014-2024)"
    )
    parser.add_argument("--demo", action="store_true", 
                        help="Use synthetic demo data")
    parser.add_argument("--skip-entropy", action="store_true",
                        help="Skip entropy calculation (use existing scores)")
    args = parser.parse_args()
    
    print("=" * 70)
    print("EXTENSION 2: OUT-OF-SAMPLE TEST (2014-2024)")
    print("=" * 70)
    print("\nTesting Asymmetry Risk Premium persistence post-publication")
    print(f"Mode: {'DEMO (synthetic data)' if args.demo else 'PRODUCTION (real CRSP data)'}")
    
    project_root = Path(__file__).parent.parent
    
    start_time = time.time()
    steps_passed = 0
    total_steps = 5
    
    # Common args
    step_args = ["--demo"] if args.demo else []
    
    # Step 1: Data Preparation
    if run_step(1, total_steps, "Data Preparation", 
                "extension2_data_prep.py", step_args, project_root):
        steps_passed += 1
    else:
        print("\n  Data preparation failed. Aborting.")
        return 1
    
    # Step 2: Factor Construction
    if run_step(2, total_steps, "Factor Construction",
                "extension2_build_factors.py", step_args, project_root):
        steps_passed += 1
    else:
        print("\n  Factor construction failed. Aborting.")
        return 1
    
    # Step 3: Portfolio Testing
    if run_step(3, total_steps, "Portfolio Testing",
                "extension2_portfolio_test.py", step_args, project_root):
        steps_passed += 1
    else:
        print("\n  Portfolio testing failed. Continuing with remaining steps.")
    
    # Step 4: Visualization
    if run_step(4, total_steps, "Visualization",
                "extension2_plot_modern.py", step_args, project_root):
        steps_passed += 1
    else:
        print("\n  Visualization failed. Continuing with summary.")
    
    # Step 5: Summary
    if generate_summary(project_root, args.demo):
        steps_passed += 1
    
    # Final report
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("EXTENSION 2 COMPLETE")
    print("=" * 70)
    
    print(f"\n  Steps completed: {steps_passed}/{total_steps}")
    print(f"  Total time: {elapsed:.1f} seconds")
    
    print("\n  Outputs:")
    print("    - outputs/tables/Table_9_Modern_Performance.csv")
    print("    - outputs/figures/Figure_9_Modern_Equity_Curve.pdf")
    print("    - outputs/summary/Extension2_OOS_Summary.txt")
    
    if steps_passed == total_steps:
        print("\n  ALL STEPS COMPLETED SUCCESSFULLY")
        return 0
    else:
        print(f"\n  WARNING: {total_steps - steps_passed} step(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
