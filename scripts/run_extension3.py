#!/usr/bin/env python3
"""
Extension 3: ML-Based Asymmetry Prediction - Master Runner

Trains XGBoost to predict future asymmetry ranks and compares
ML-based portfolio strategy against standard historical sorting.

Steps:
1. Feature Engineering & Target Generation
2. Walk-Forward Model Training
3. Portfolio Backtest
4. Visualization & Interpretation

Outputs:
- Table 10: ML Performance Metrics
- Figure 10: Cumulative Wealth Comparison
- Figure 11: Feature Importance Plot
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
    
    tables_dir = project_root / "outputs" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    output_dir = project_root / "outputs" / "summary"
    
    summary_lines = [
        "=" * 70,
        "EXTENSION 3: ML-BASED ASYMMETRY PREDICTION SUMMARY",
        "=" * 70,
        "",
        "RESEARCH QUESTION:",
        "Can machine learning improve the Asymmetry Risk Premium strategy",
        "by predicting FUTURE asymmetry instead of using HISTORICAL asymmetry?",
        "",
        "-" * 70,
        "METHODOLOGY:",
        "-" * 70,
        "- Model: XGBoost Regressor (or GradientBoosting fallback)",
        "- Target: Cross-sectional rank of DOWN_ASY at t+1",
        "- Features:",
        "  * Micro: DOWN_ASY (lag), IVOL, TURN, SIZE, MOM, BM",
        "  * Macro: Market Volatility, Lagged Market Return",
        "  * Interactions: DOWN_ASY × MKT_VOL, IVOL × TURN",
        "- Validation: Walk-forward expanding window (no look-ahead bias)",
        "  * Initial training: 1963-1989",
        "  * First prediction: 1990",
        "  * Retrain annually with expanding window",
        "",
        "-" * 70,
        "RESULTS:",
        "-" * 70,
    ]
    
    # Load Table 10 results
    table10_path = tables_dir / "Table_10_ML_Performance.csv"
    
    if table10_path.exists():
        import pandas as pd
        table10 = pd.read_csv(table10_path)
        
        for _, row in table10.iterrows():
            summary_lines.append(f"\n{row['Strategy']}:")
            for col in table10.columns[1:]:
                if col in row and pd.notna(row[col]):
                    summary_lines.append(f"  {col}: {row[col]}")
    else:
        summary_lines.append("\n  [Table 10 not found - run backtest first]")
    
    # Load feature importance
    data_dir = project_root / "data" / "processed"
    imp_path = data_dir / "ML_Feature_Importance.csv"
    
    if imp_path.exists():
        import pandas as pd
        imp_df = pd.read_csv(imp_path)
        
        summary_lines.extend([
            "",
            "-" * 70,
            "FEATURE IMPORTANCE (Top 5):",
            "-" * 70,
        ])
        
        for _, row in imp_df.head(5).iterrows():
            summary_lines.append(f"  {row['Feature']}: {row['Importance']:.4f}")
    
    # Interpretation
    summary_lines.extend([
        "",
        "-" * 70,
        "INTERPRETATION:",
        "-" * 70,
        "",
        "If Sharpe(ML) > Sharpe(Standard):",
        "  → ML successfully extracts predictive signal from characteristics",
        "  → Future asymmetry is (partially) predictable",
        "  → Recommendation: Consider ML-enhanced strategy for live trading",
        "",
        "If Sharpe(ML) ≈ Sharpe(Standard):",
        "  → Historical asymmetry is a strong baseline",
        "  → ML adds complexity without clear improvement",
        "  → Recommendation: Stick with simpler historical approach",
        "",
        "If Sharpe(ML) < Sharpe(Standard):",
        "  → Prediction noise hurts portfolio construction",
        "  → Asymmetry persistence is the main driver, not predictability",
        "  → Recommendation: Use historical sorting only",
    ])
    
    if demo:
        summary_lines.extend([
            "",
            "-" * 70,
            "NOTE: This analysis used SYNTHETIC demo data.",
            "For publication-quality results:",
            "  1. Install XGBoost: pip install xgboost",
            "  2. (Optional) Install SHAP: pip install shap",
            "  3. Run with real CRSP data: python scripts/run_extension3.py",
            "-" * 70,
        ])
    
    summary_lines.extend([
        "",
        "=" * 70,
        "END OF EXTENSION 3 SUMMARY",
        "=" * 70,
    ])
    
    # Save summary
    summary_path = output_dir / "Extension3_ML_Summary.txt"
    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary_lines))
    
    print(f"\n  Saved summary: {summary_path}")
    print("\n" + "\n".join(summary_lines))
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run Extension 3: ML-Based Asymmetry Prediction"
    )
    parser.add_argument("--demo", action="store_true",
                        help="Use synthetic demo data")
    args = parser.parse_args()
    
    print("=" * 70)
    print("EXTENSION 3: ML-BASED ASYMMETRY PREDICTION")
    print("=" * 70)
    print("\nTraining XGBoost to predict future asymmetry ranks")
    print(f"Mode: {'DEMO (synthetic data)' if args.demo else 'PRODUCTION (real data)'}")
    
    project_root = Path(__file__).parent.parent
    
    start_time = time.time()
    steps_passed = 0
    total_steps = 5
    
    step_args = ["--demo"] if args.demo else []
    
    # Step 1: Feature Engineering
    if run_step(1, total_steps, "Feature Engineering",
                "extension3_ml_prep.py", step_args, project_root):
        steps_passed += 1
    else:
        print("\n  Feature engineering failed. Aborting.")
        return 1
    
    # Step 2: Walk-Forward Training
    if run_step(2, total_steps, "Walk-Forward Training",
                "extension3_walk_forward.py", step_args, project_root):
        steps_passed += 1
    else:
        print("\n  Training failed. Aborting.")
        return 1
    
    # Step 3: Portfolio Backtest
    if run_step(3, total_steps, "Portfolio Backtest",
                "extension3_backtest.py", step_args, project_root):
        steps_passed += 1
    else:
        print("\n  Backtest failed. Continuing with visualization.")
    
    # Step 4: Visualization
    if run_step(4, total_steps, "Visualization",
                "extension3_eval_plots.py", step_args, project_root):
        steps_passed += 1
    else:
        print("\n  Visualization failed. Continuing with summary.")
    
    # Step 5: Summary
    if generate_summary(project_root, args.demo):
        steps_passed += 1
    
    # Final report
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("EXTENSION 3 COMPLETE")
    print("=" * 70)
    
    print(f"\n  Steps completed: {steps_passed}/{total_steps}")
    print(f"  Total time: {elapsed:.1f} seconds")
    
    print("\n  Outputs:")
    print("    - outputs/tables/Table_10_ML_Performance.csv")
    print("    - outputs/figures/Figure_10_ML_Cumulative_Returns.pdf")
    print("    - outputs/figures/Figure_11_Feature_Importance.pdf")
    print("    - outputs/summary/Extension3_ML_Summary.txt")
    
    if steps_passed == total_steps:
        print("\n  ALL STEPS COMPLETED SUCCESSFULLY")
        return 0
    else:
        print(f"\n  WARNING: {total_steps - steps_passed} step(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
