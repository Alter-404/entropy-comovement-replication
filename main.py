#!/usr/bin/env python3
"""
Main entry point for Jiang, Wu, Zhou (2018) replication.

"Asymmetry in Stock Comovements: An Entropy Approach"
Journal of Financial and Quantitative Analysis

Usage:
    python main.py              # Full replication with real data
    python main.py --demo       # Demo mode (no data required)
    python main.py --help       # Show all options
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src" / "python"))


def run_extensions(extensions: list, project_root: Path, quiet: bool, demo: bool) -> bool:
    """
    Run the specified extensions.
    
    Args:
        extensions: List of extension numbers to run (1, 2, 3)
        project_root: Path to project root directory
        quiet: Suppress output
        demo: Run in demo mode
        
    Returns:
        True if all extensions succeeded, False otherwise
    """
    extension_names = {
        1: "Crisis Analysis",
        2: "Out-of-Sample 2014-2024", 
        3: "ML-Based Prediction"
    }
    
    if not quiet:
        print()
        print("=" * 70)
        print("  RUNNING EXTENSIONS")
        print("=" * 70)
        print()
    
    all_success = True
    
    for ext_num in sorted(extensions):
        script_path = project_root / "scripts" / f"run_extension{ext_num}.py"
        
        if not script_path.exists():
            if not quiet:
                print(f"  [!] Extension {ext_num} script not found: {script_path}")
            all_success = False
            continue
        
        if not quiet:
            print(f"  Running Extension {ext_num}: {extension_names.get(ext_num, 'Unknown')}...")
            print()
        
        try:
            cmd = [sys.executable, str(script_path)]
            if demo:
                cmd.append("--demo")
            
            result = subprocess.run(
                cmd,
                cwd=str(project_root),
                capture_output=quiet,
                text=True
            )
            
            if result.returncode != 0:
                if not quiet:
                    print(f"  [✗] Extension {ext_num} failed")
                    if result.stderr:
                        print(f"      Error: {result.stderr[:200]}")
                all_success = False
            else:
                if not quiet:
                    print(f"  [✓] Extension {ext_num} completed successfully")
                    print()
        except Exception as e:
            if not quiet:
                print(f"  [✗] Extension {ext_num} error: {e}")
            all_success = False
    
    if not quiet:
        print()
        print("=" * 70)
        if all_success:
            print("  ALL EXTENSIONS COMPLETE")
        else:
            print("  SOME EXTENSIONS FAILED")
        print("=" * 70)
    
    return all_success


def main():
    """Main entry point for the replication."""
    parser = argparse.ArgumentParser(
        description="Replicate Jiang, Wu, Zhou (2018) - Asymmetry in Stock Comovements",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py                      # Run full replication
    python main.py --demo               # Quick demo (no data needed)
    python main.py --phase 3            # Run up to phase 3 only
    python main.py --extensions 1 2 3   # Run replication + all extensions
    python main.py --extensions-only --extensions 1  # Run only Extension 1
    python main.py --skip-cpp           # Skip C++ compilation
    python main.py --output-dir results # Custom output directory

Phases:
    1. C++ Engine Compilation
    2. Data Loading & Preprocessing  
    3. Simulation & Validation (Table 1, Figures 1-2)
    4. Asymmetry Tests & Portfolio Construction (Tables 2, 5)
    5. Firm Characteristics (Tables 3, 4, 6)
    6. Robustness Checks (Table 7)

Extensions:
    1. Crisis Analysis (Table 8, Figure 8) - Premium during high-volatility regimes
    2. Out-of-Sample 2014-2024 (Table 9, Figure 9) - Post-publication test
    3. ML Prediction (Tables 10, Figures 10-11) - XGBoost asymmetry forecasting

Data Requirements:
    Place the following files in data/raw/:
    - CRSP Daily Stock.csv
    - CRSP Monthly Stock.csv
    - Portfolios_Formed_on_ME.csv
    - Portfolios_Formed_on_BE-ME.csv
    - 10_Portfolios_Prior_12_2.csv
    - F-F_Research_Data_Factors.csv
        """
    )
    
    parser.add_argument(
        "--demo", 
        action="store_true",
        help="Run in demo mode without real data"
    )
    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2, 3, 4, 5, 6],
        default=6,
        help="Run up to this phase (default: 6 = all phases)"
    )
    parser.add_argument(
        "--skip-cpp",
        action="store_true",
        help="Skip C++ engine compilation (use existing build)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Custom output directory (default: outputs/)"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Custom data directory (default: data/)"
    )
    parser.add_argument(
        "--n-simulations",
        type=int,
        default=1000,
        help="Number of simulations for Table 1 (default: 1000)"
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap iterations (default: 1000)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Minimal output"
    )
    parser.add_argument(
        "--extensions",
        nargs="*",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help="Run extensions after replication (1=Crisis, 2=Out-of-Sample, 3=ML)"
    )
    parser.add_argument(
        "--extensions-only",
        action="store_true",
        help="Skip replication and run only the specified extensions"
    )
    
    args = parser.parse_args()
    
    # Safety check: warn if running full replication without explicit confirmation
    if not args.demo and len(sys.argv) == 1:
        print("=" * 70)
        print("  WARNING: Full replication requires ~8GB RAM for CRSP data")
        print("=" * 70)
        print()
        print("Options:")
        print("  python main.py --demo    # Quick demo (no data, ~1 second)")
        print("  python main.py --help    # Show all options")
        print()
        response = input("Run full replication? [y/N]: ").strip().lower()
        if response != 'y':
            print("Aborted. Use --demo for a quick test.")
            return 0
        print()
    
    # Print banner
    if not args.quiet:
        print("=" * 70)
        print("  Jiang, Wu, Zhou (2018) Replication")
        print("  'Asymmetry in Stock Comovements: An Entropy Approach'")
        print("  Journal of Financial and Quantitative Analysis")
        print("=" * 70)
        print()
    
    # Import and run pipeline
    try:
        from scripts.run_full_replication import ReplicationPipeline
    except ImportError:
        # Fallback: add scripts to path
        sys.path.insert(0, str(Path(__file__).parent / "scripts"))
        from run_full_replication import ReplicationPipeline
    
    # Setup directories
    project_root = Path(__file__).parent
    
    # Create pipeline with the correct interface
    pipeline = ReplicationPipeline(
        base_dir=str(project_root),
        demo_mode=args.demo,
        verbose=args.verbose and not args.quiet
    )
    
    # Run pipeline
    if args.demo:
        if not args.quiet:
            print("Running in DEMO mode (validation only, no real data)")
            print()
    else:
        if not args.quiet:
            print(f"Running phases 1-{args.phase}")
            if args.skip_cpp:
                print("Skipping C++ compilation")
            print()
    
    # Handle extensions-only mode
    if args.extensions_only:
        extensions_to_run = args.extensions if args.extensions else [1, 2, 3]
        if not args.quiet:
            print("Running extensions only (skipping replication)")
            print()
        extension_success = run_extensions(extensions_to_run, project_root, args.quiet, args.demo)
        return 0 if extension_success else 1
    
    # Run the pipeline - demo_mode already set in constructor
    results = pipeline.run(phases=list(range(1, args.phase + 1)))
    
    # Check if all phases succeeded
    success = all(r.get('success', False) for r in results.values())
    
    # Print summary
    if not args.quiet:
        print()
        print("=" * 70)
        if success:
            print("  REPLICATION COMPLETE")
            print()
            print("  Outputs:")
            print(f"    Tables:  {project_root / 'outputs' / 'tables'}")
            print(f"    Figures: {project_root / 'outputs' / 'figures'}")
            print(f"    Report:  {project_root / 'reports' / 'replication_report.md'}")
        else:
            print("  REPLICATION FAILED")
            print("  Check the error messages above for details.")
        print("=" * 70)
    
    # Run extensions if requested
    extension_success = True
    if args.extensions is not None:
        extensions_to_run = args.extensions if args.extensions else [1, 2, 3]
        extension_success = run_extensions(extensions_to_run, project_root, args.quiet, args.demo)
    
    return 0 if (success and extension_success) else 1


if __name__ == "__main__":
    sys.exit(main())
