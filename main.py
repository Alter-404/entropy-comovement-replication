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
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src" / "python"))


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
    python main.py --skip-cpp           # Skip C++ compilation
    python main.py --output-dir results # Custom output directory

Phases:
    1. C++ Engine Compilation
    2. Data Loading & Preprocessing  
    3. Simulation & Validation (Table 1, Figures 1-2)
    4. Asymmetry Tests & Portfolio Construction (Tables 2, 5)
    5. Firm Characteristics (Tables 3, 4, 6)
    6. Robustness Checks (Table 7)

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
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
