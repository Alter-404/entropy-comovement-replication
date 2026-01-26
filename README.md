# Asymmetry in Stock Comovement : An Entropy Approach | Replication

> Academic replication of "Asymmetry in Stock Comovements: An Entropy Approach" (Jiang, Wu, Zhou 2018, JFQA) with high-performance C++ engine and comprehensive Python analysis

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![C++](https://img.shields.io/badge/C++-17-00599C?logo=cplusplus)](https://isocpp.org/)
[![CMake](https://img.shields.io/badge/CMake-3.20+-064F8C?logo=cmake)](https://cmake.org/)
[![Tests](https://img.shields.io/badge/tests-70+%20passing-brightgreen)](./tests/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

---

---

## Overview

This project provides a replication of Jiang, Wu, and Zhou's (2018) seminal paper on asymmetric stock comovements. The paper introduces an entropy-based measure to detect and quantify the empirical phenomenon that stocks move together more during market downturns than upturns—a pattern not captured by traditional correlation measures.

Built with a hybrid C++/Python architecture, the replication combines:
- **High-Performance C++ Engine** - Optimized kernel density estimation with OpenMP parallelization
- **Comprehensive Python Pipeline** - End-to-end data processing, portfolio construction, and statistical analysis
- **Publication-Quality Outputs** - 6 figures and 7 tables matching the original paper

The project demonstrates that stocks exhibiting high downside asymmetry earn a significant risk premium (~0.4% monthly), robust to standard factor controls and alternative explanations.

---

## Features

### Core Functionality
- **Entropy Calculation** - High-speed C++ implementation of Hellinger distance-based asymmetry measure
- **Bandwidth Optimization** - Silverman's rule with optional likelihood cross-validation
- **Data Pipeline** - Automated CRSP/Fama-French data loading with Parquet caching
- **Portfolio Construction** - Rolling-window DOWN_ASY calculation and quintile/decile sorting
- **Factor Models** - Carhart 4-factor regression with Newey-West HAC standard errors
- **Firm Characteristics** - 14 financial metrics (Beta, IVOL, Coskewness, Illiquidity, etc.)
- **Robustness Checks** - Subperiod analysis, double sorts, Fama-MacBeth regressions
- **Visualization** - Matplotlib/Seaborn figures with publication-ready styling
- **Report Generation** - Automated LaTeX/Markdown table formatting and summary reports

### Technical Features
- **70+ Unit Tests** - Comprehensive test coverage across all phases
- **Hybrid Architecture** - C++ for compute-intensive tasks, Python for flexibility
- **Smart Caching** - Parquet-based intermediate storage for 10-100× speedup
- **OpenMP Parallelization** - Multi-core entropy calculations
- **NumPy Integration** - Zero-copy data transfer via PyBind11
- **Demo Mode** - Synthetic data generation for testing without proprietary datasets

### Outputs

#### 6 Figures
- **Figure 1**: Symmetry vs. Asymmetry Concept (Contour Plot)
- **Figure 2**: Copula Comparison (4-panel: Clayton, Gaussian, Mixed, Real Data)
- **Figure 3**: Power Analysis (Entropy vs. HTZ Test Comparison)
- **Figure 4**: Asymmetry Distribution and Firm Size by Decile
- **Figure 5**: Cumulative Wealth of Asymmetry-Sorted Portfolios
- **Figure 6**: Premium Dynamics (Time-Series and Regime Scatter)

#### 7 Tables
- **Table 1**: Size and Power of Entropy Test (Simulation with 6 copula specifications)
- **Table 2**: Asymmetry Tests for 30 Portfolios (Size, B/M, Momentum)
- **Table 3**: Cross-Sectional Correlations with 14 Firm Characteristics
- **Table 4**: Summary Statistics by Asymmetry Deciles
- **Table 5**: Returns and Alphas (Main Asset Pricing Result - **0.44% monthly premium**)
- **Table 6**: Time-Series Regressions (Market Volatility, Liquidity, Sentiment)
- **Table 7**: Double-Sorted Portfolios (Controlling for Beta, Coskewness, etc.)

---

## Getting Started

### Prerequisites

**System Requirements:**
- **Operating System**: Linux (Ubuntu 20.04+), macOS (10.15+), or Windows
- **RAM**: 8GB minimum (16GB recommended for full dataset)
- **Storage**: 10GB free space
- **Cores**: Multi-core CPU recommended (parallelization via OpenMP)

**Required Software:**
- **Python 3.8+** ([Download](https://www.python.org/downloads/))
- **CMake 3.20+** ([Download](https://cmake.org/download/))
- **C++ Compiler** with C++17 support (GCC 9+, Clang 10+, or MSVC 2019+)
- **Git** (for cloning repository)

**Optional Dependencies:**
- **Eigen3** - Linear algebra library (auto-downloaded if not found)
- **OpenMP** - Multi-threading (usually bundled with compiler)

---

### Installation

#### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/entropy-comovement-replication.git
cd entropy-comovement-replication
```

#### 2. Install System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y \
    cmake \
    g++ \
    libeigen3-dev \
    python3-dev \
    python3-pip \
    python3-venv
```

**macOS:**
```bash
brew install cmake gcc eigen python@3.11
```

**Windows (WSL2):**
```bash
# Install WSL2 first, then use Ubuntu commands above
```

#### 3. Set Up Python Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

#### 4. Install Python Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# Verify installation
python -c "import numpy, pandas, scipy, matplotlib, seaborn; print('✓ All packages installed')"
```

**Key Packages:**
- `numpy >= 1.21.0` - Numerical computing
- `pandas >= 1.3.0` - Data manipulation
- `scipy >= 1.7.0` - Scientific computing
- `matplotlib >= 3.4.0` - Plotting
- `seaborn >= 0.11.0` - Statistical visualization
- `statsmodels >= 0.13.0` - Statistical models (for Newey-West HAC)
- `pybind11 >= 2.8.0` - C++/Python bindings
- `pytest >= 6.2.0` - Testing framework

#### 5. Build C++ Engine

```bash
# Automated build script
chmod +x scripts/build_cpp.sh
./scripts/build_cpp.sh

# Or manual build
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
cd ..
```

**Verify C++ Module:**
```bash
python -c "import entropy_cpp; print('✓ C++ engine loaded successfully')"
```

#### 6. Acquire Data (Optional - Skip for Demo Mode)

The replication requires proprietary data from CRSP and public data from Kenneth French:

**CRSP Data (Requires WRDS Subscription):**
1. Access [WRDS](https://wrds-www.wharton.upenn.edu/)
2. Download:
   - CRSP Daily Stock File (1962-2013)
   - CRSP Monthly Stock File (1965-2013)
3. Place in `data/raw/`

**Fama-French Data (Free):**
1. Visit [Kenneth French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)
2. Download:
   - Fama/French 3 Factors
   - Momentum Factor (UMD)
   - Portfolios Formed on Size, B/M, Momentum
3. Place in `data/raw/`

**Required Files:**
```
data/raw/
├── CRSP Daily Stock.csv
├── CRSP Monthly Stock.csv
├── Portfolios_Formed_on_ME.csv
├── Portfolios_Formed_on_BE-ME.csv
├── 10_Portfolios_Prior_12_2.csv
└── F-F_Research_Data_Factors.csv
```

**Note**: If data is unavailable, use `--demo` mode (synthetic data).

---

### Quick Start

#### Run Full Replication (Demo Mode)

```bash
# Complete pipeline with synthetic data (no CRSP required)
python main.py --demo

# Or use the detailed pipeline script
python scripts/run_full_replication.py --demo
```

**Expected Output:**
```
======================================================================
   ENTROPY-COMOVEMENT REPLICATION PIPELINE
   Jiang, Wu, and Zhou (2018) - JFQA
======================================================================

✓ Phase 1: Data Loading & Preprocessing [OK]
✓ Phase 2: Portfolio Construction [OK]
✓ Phase 3: Factor Model Regressions [OK]
✓ Phase 4: Firm Characteristics [OK]
✓ Phase 5: Robustness Checks [OK]
✓ Phase 6: Report Generation [OK]

Total duration: 12.3 seconds
Tables:  7/7 generated
Figures: 6/6 generated

Report: reports/replication_report.md
```

#### Run with Real Data

```bash
# Place CRSP/FF data in data/raw/ first
python main.py

# View generated tables
ls -lh outputs/tables/

# View generated figures
ls -lh outputs/figures/

# Read comprehensive report
cat reports/replication_report.md
```

#### Run Specific Phases

```bash
# Run only Tables 1-2 (simulation and tests)
python main.py --phase 3 --demo

# Run only portfolio analysis (Tables 5-7)
python main.py --phase 4 --phase 5

# Run specific table script
python scripts/replicate_table5.py
```

---

---

## Project Structure

```
entropy-comovement-replication/
│
├── CMakeLists.txt                      # Root build configuration
├── setup.py                            # Python package installer
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
├── LICENSE                             # MIT License
├── .gitignore                          # Git ignore patterns
│
├── src/
│   ├── cpp/                            # C++ High-Performance Engine
│   │   ├── entropy.h                   # Core entropy calculation
│   │   ├── entropy.cpp                 # KDE, bandwidth optimization, integration
│   │   └── bindings.cpp                # PyBind11 Python interface
│   │
│   └── python/                         # Python Analysis Modules
│       ├── __init__.py                 # Package initialization
│       ├── data_loader.py              # CRSP/FF data parsing and caching
│       ├── simulation.py               # Copula-GARCH simulation engine
│       ├── portfolio.py                # Portfolio construction and sorting
│       ├── characteristics.py          # Firm characteristic calculation
│       ├── robustness.py               # Robustness analysis framework
│       └── report_generator.py         # LaTeX/Markdown report generation
│
├── scripts/                            # Execution Scripts
│   ├── run_full_replication.py         # Master pipeline orchestrator
│   ├── build_cpp.sh                    # C++ compilation automation
│   ├── replicate_table1.py             # Table 1: Size and Power
│   ├── replicate_table2.py             # Table 2: Asymmetry Tests
│   ├── replicate_table3.py             # Table 3: Correlations
│   ├── replicate_table4.py             # Table 4: Summary Stats
│   ├── replicate_table5.py             # Table 5: Returns and Alphas
│   ├── replicate_table6.py             # Table 6: Time-Series Regressions
│   ├── replicate_table7.py             # Table 7: Double Sorts
│   ├── replicate_fig1_2.py             # Figures 1 & 2: Concept and Copulas
│   ├── plot_power_curve.py             # Figure 3: Power Analysis
│   ├── plot_asymmetry_distribution.py  # Figure 4: Asymmetry & Size
│   ├── plot_equity_curve.py            # Figure 5: Cumulative Wealth
│   ├── plot_premium_dynamics.py        # Figure 6: Premium Dynamics
│   ├── generate_timeseries_data.py     # Time-series data generator
│   ├── format_tables.py                # Table formatting utilities
│   ├── robustness_checks.py            # Comprehensive robustness tests
│
├── tests/                              # Unit & Integration Tests
│   ├── CMakeLists.txt                  # Test build configuration
│   ├── test_phase1.cpp                 # C++ engine tests
│   ├── test_phase2.py                  # Data pipeline tests
│   ├── test_phase3.py                  # Simulation tests
│   ├── test_phase4.py                  # Portfolio tests
│   ├── test_phase5.py                  # Characteristics tests
│   ├── test_phase6.py                  # Robustness tests
│   ├── test_integration.py             # End-to-end tests
│   └── test_validation.py              # Replication validation
│
├── data/                               # Data Directory (gitignored)
│   ├── raw/                            # Raw CRSP/FF CSVs (user-provided)
│   │   ├── .gitkeep
│   │   └── README.md                   # Data acquisition instructions
│   ├── processed/                      # Parquet cached data
│   │   ├── crsp_processed.parquet      # Cleaned CRSP data
│   │   ├── ff_processed.parquet        # Fama-French factors
│   │   ├── down_asy_scores.parquet     # Calculated DOWN_ASY scores
│   │   └── firm_characteristics.parquet # Firm characteristics
│   └── cache/                          # Temporary computation cache
│
├── outputs/                            # Generated Artifacts (gitignored)
│   ├── figures/                        # PDF/PNG visualizations
│   │   ├── Figure_1_Symmetry_Concept.pdf
│   │   ├── Figure_2_Copulas.pdf
│   │   ├── Figure_3_Power_Analysis.pdf
│   │   ├── Figure_4_Asymmetry_Distribution.pdf
│   │   ├── Figure_5_Cumulative_Returns.pdf
│   │   └── Figure_6[A-B]_Premium_*.pdf
│   │
│   └── tables/                         # CSV/LaTeX tables
│       ├── Table_[1-7]_*.csv           # Machine-readable tables
│       ├── Table_[1-7]_*.tex           # LaTeX formatted tables
│       └── Table_[1-7]_*_formatted.txt # Human-readable tables
│
├── reports/                            # Documentation (gitignored)
│   ├── replication_report.md           # Complete replication summary
│   └── demo_replication_report.md      # Demo mode summary
│
├── docs/                               # Static Documentation
│   ├── INSTALL.md                      # Installation instructions
│   └── TECHNICAL_DOCUMENTATION.md      # Technical documentation
│
├── build/                              # CMake build directory (gitignored)
│   ├── entropy_cpp*.so                 # Compiled C++ module
│   ├── CMakeCache.txt
│   └── tests/                          # Compiled C++ tests
│
└── notebooks/                          # Jupyter Notebooks
    └── exploratory_analysis.ipynb      # Interactive exploration
```

---

---

## Testing

### Run All Tests

```bash
# Quick test run (C++ + Python)
pytest tests/ -v

# With detailed output
pytest tests/ -v --tb=short

# Run C++ tests only
cd build && ctest --output-on-failure && cd ..

# Run Python tests only
pytest tests/*.py -v
```

### Test Coverage

Current test statistics:
- **Total Tests**: 70+
- **C++ Tests**: 6 (KDE, bandwidth, integration, edge cases)
- **Python Tests**: 64+ (data loading, simulation, portfolio, characteristics, robustness)
- **Integration Tests**: 5 (end-to-end validation)
- **Status**: All passing

Test categories:
- **Phase 1 Tests**: C++ entropy engine correctness and performance
- **Phase 2 Tests**: Data loading, caching, excess returns, standardization
- **Phase 3 Tests**: Copula simulation, power analysis, convergence
- **Phase 4 Tests**: Portfolio construction, rolling calculations, sorting logic
- **Phase 5 Tests**: Firm characteristics, correlation matrix, summary stats
- **Phase 6 Tests**: Robustness checks, Fama-MacBeth, bootstrap
- **Integration Tests**: Black-Scholes validation, end-to-end pipeline

### Run Specific Test Suites

```bash
# Test C++ entropy engine
pytest tests/test_phase1.cpp -v

# Test data pipeline
pytest tests/test_phase2.py -v

# Test simulation engine
pytest tests/test_phase3.py -v

# Test portfolio construction
pytest tests/test_phase4.py -v

# Test characteristics calculation
pytest tests/test_phase5.py -v

# Test robustness framework
pytest tests/test_phase6.py -v

# End-to-end validation
pytest tests/test_integration.py -v
```

### Validation Tests

```bash
# Verify replication accuracy
pytest tests/test_validation.py -v

# Check Table 1 power matches paper
pytest tests/test_phase3.py::test_table1_power_accuracy -v

# Verify Table 5 premium significance
pytest tests/test_phase4.py::test_table5_premium_positive -v
```

---

## Usage Guide

### Computing Asymmetry Scores

```python
import numpy as np
import entropy_cpp

# Generate or load standardized returns
np.random.seed(42)
stock_returns = np.random.randn(252)  # 1 year daily
market_returns = np.random.randn(252)

# Standardize (mean=0, std=1)
x = (stock_returns - stock_returns.mean()) / stock_returns.std(ddof=1)
y = (market_returns - market_returns.mean()) / market_returns.std(ddof=1)

# Calculate asymmetry
engine = entropy_cpp.EntropyEngine()
s_rho, down_asy = engine.calculate_metrics(x, y, c=0.0)

print(f"S_rho (unsigned): {s_rho:.6f}")
print(f"DOWN_ASY (signed): {down_asy:.6f}")
```

### Loading Market Data

```python
from src.python.data_loader import DataLoader

# Initialize with caching
loader = DataLoader(
    raw_data_dir='data/raw',
    cache_dir='data/processed'
)

# Load daily data (uses cache if available)
stocks, factors = loader.load_data(frequency='daily')

print(f"Loaded {len(stocks):,} observations")
print(f"Date range: {stocks.index.min()} to {stocks.index.max()}")
print(f"Stocks: {stocks['PERMNO'].nunique()}")

# Get standardized returns for a specific stock
apple_permno = 14593
apple_data = stocks.xs(apple_permno, level='PERMNO')
apple_std = DataLoader.get_standardized_returns(apple_data['EXRET'])
```

### Constructing Portfolios

```python
from src.python.portfolio import PortfolioConstructor

# Initialize
constructor = PortfolioConstructor(
    stocks_df=stocks,
    factors_df=factors,
    entropy_engine=entropy_cpp.EntropyEngine()
)

# Calculate DOWN_ASY scores for all stocks (rolling 12-month)
down_asy_scores = constructor.calculate_rolling_asymmetry(
    window=252,  # 252 trading days ~ 12 months
    min_obs=100  # Minimum observations per window
)

# Sort into quintiles each month
portfolios = constructor.sort_into_portfolios(
    scores=down_asy_scores,
    n_portfolios=5,
    weighting='equal'  # or 'value'
)

# Calculate portfolio returns
returns = constructor.calculate_portfolio_returns(portfolios)
print(f"High-Low Spread: {returns['High-Low'].mean():.4f} monthly")
```

### Calculating Factor Alphas

```python
from src.python.characteristics import FactorModelRegression

# Initialize
regressor = FactorModelRegression(factors_df=factors)

# Run Carhart 4-factor regression
results = regressor.estimate_alpha(
    portfolio_returns=returns['High-Low'],
    model='carhart4',  # Market, Size, Value, Momentum
    newey_west_lags=12  # HAC standard errors
)

print(f"Alpha: {results['alpha']:.4f} ({results['t_stat']:.2f})")
print(f"Market Beta: {results['beta_mkt']:.3f}")
print(f"R-squared: {results['r_squared']:.3f}")
```

### Running Robustness Checks

```python
from src.python.robustness import RobustnessRunner

# Initialize
runner = RobustnessRunner(
    returns=returns,
    down_asy=down_asy_scores,
    factors=factors,
    characteristics=firm_characteristics
)

# Subperiod analysis
subperiod_results = runner.run_subperiod_analysis(
    periods={
        'Pre-2000': ('1965-01', '1999-12'),
        'Post-2000': ('2000-01', '2013-12'),
        'Crisis': ('2007-01', '2009-12')
    }
)

# Fama-MacBeth regressions
fm_results = runner.run_fama_macbeth(
    dependent_var='returns',
    independent_vars=['DOWN_ASY', 'BETA', 'SIZE', 'B/M', 'MOM'],
    newey_west_lags=12
)

print(f"DOWN_ASY coefficient: {fm_results['DOWN_ASY']['coef']:.4f}")
print(f"t-statistic: {fm_results['DOWN_ASY']['t_stat']:.2f}")
```

---

## Tech Stack

### Core Framework
- **Python 3.8+** - Primary language for data analysis
- **C++17** - High-performance entropy engine
- **CMake 3.20+** - Cross-platform build system
- **PyBind11 2.8+** - Seamless C++/Python integration

### Numerical Computing
- **NumPy 1.21+** - Array operations and linear algebra
- **SciPy 1.7+** - Statistical functions and optimization
- **Math.NET Numerics** (C++ side) - Matrix operations
- **Eigen3 3.4+** - C++ linear algebra library
- **OpenMP 4.5+** - Multi-threading and parallelization

### Data Management
- **Pandas 1.3+** - DataFrame operations
- **PyArrow 6.0+** - Parquet file I/O (10-100× faster than CSV)
- **SQLite** (optional) - Results database

### Statistical Analysis
- **statsmodels 0.13+** - Econometric models
  - Newey-West HAC standard errors
  - Fama-MacBeth regressions
  - Time-series analysis
- **scikit-learn** (optional) - Machine learning utilities

### Visualization
- **Matplotlib 3.4+** - Publication-quality plotting
- **Seaborn 0.11+** - Statistical visualization
- **LaTeX** (optional) - Table and equation rendering

### Testing & Quality
- **pytest 6.2+** - Python testing framework
- **Catch2** (C++) - C++ unit testing (auto-downloaded)
- **pytest-cov** - Code coverage analysis

### Build & Development
- **GCC 9+ / Clang 10+ / MSVC 2019+** - C++ compiler with C++17 support
- **Git** - Version control
- **Make** - Build automation

---

## Documentation

### User Guides
- [Installation Guide](docs/INSTALL.md) - Detailed installation instructions
- [Phase Documentation](docs/) - Individual phase summaries

---

## Architecture

### Hybrid C++/Python Design

```
┌──────────────────────────────────────┐
│         Python Layer                 │
│  (Data, Portfolio, Visualization)    │
└──────────────────────────────────────┘
                  ↓
┌──────────────────────────────────────┐
│         PyBind11 Interface           │
│     (Zero-Copy Data Transfer)        │
└──────────────────────────────────────┘
                  ↓
┌──────────────────────────────────────┐
│          C++ Engine                  │
│  (KDE, Bandwidth Opt, Integration)   │
│     (OpenMP Parallelization)         │
└──────────────────────────────────────┘
```

### Processing Pipeline

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Raw CRSP   │────▶│ Data Loader │────▶│   Parquet   │
│  CSV Files  │     │  (Python)   │     │   Cache     │
└─────────────┘     └─────────────┘     └─────────────┘
                                                │
                                                ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Entropy   │◀────│  Portfolio  │◀────│  Rolling    │
│  C++ Engine │     │ Constructor │     │  Windows    │
└─────────────┘     └─────────────┘     └─────────────┘
      │
      ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ DOWN_ASY    │────▶│  Quintile   │────▶│   Factor    │
│   Scores    │     │   Sorting   │     │  Alphas     │
└─────────────┘     └─────────────┘     └─────────────┘
                                                │
                                                ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Figures    │◀────│   Report    │◀────│ Robustness  │
│   Tables    │     │  Generator  │     │   Checks    │
└─────────────┘     └─────────────┘     └─────────────┘
```

### Design Patterns
- **Strategy Pattern** - Multiple copula implementations
- **Factory Pattern** - Simulation and portfolio construction
- **Repository Pattern** - Data access abstraction
- **Pipeline Pattern** - Multi-stage processing workflow
- **Observer Pattern** - Progress reporting and logging

---

## Mathematical Background

### Entropy-Based Asymmetry Measure

The paper introduces **S_ρ**, an entropy-based measure of comovement asymmetry:

$$
S_\rho = \frac{1}{2} \int\int \left[\sqrt{f(x,y)} - \sqrt{f(-x,-y)}\right]^2 dx \, dy
$$

This is half the squared **Hellinger distance** between the joint density $f(x, y)$ and its 180° rotation $f(-x, -y)$.

**Key Properties:**
- $S_\rho = 0$ implies perfect symmetry
- $S_\rho > 0$ indicates asymmetry
- Robust to marginal distributions (uses standardized returns)
- Captures all forms of dependence asymmetry

### Signed Asymmetry Score (DOWN_ASY)

The directional asymmetry measure:

$$
\text{DOWN\_ASY} = \text{sign}(\text{LQP} - \text{UQP}) \times S_\rho
$$

where:
- **LQP** = Lower Quadrant Probability $P(X < -c, Y < -c)$
- **UQP** = Upper Quadrant Probability $P(X > c, Y > c)$
- **c** = Threshold (typically 0 for standardized data)

**Interpretation:**
- $\text{DOWN\_ASY} > 0$: Greater downside comovement (risky)
- $\text{DOWN\_ASY} < 0$: Greater upside comovement (safe)

### Implementation Details

**Kernel Density Estimation:**
- Gaussian product kernel: $K(x, y) = \phi(x) \phi(y)$
- Bandwidth selection: Silverman's rule with optional LCV
- Grid size: 100×100 over $[-4, 4]^2$

**Numerical Integration:**
- Grid-based summation (O(N²) instead of O(N⁴) for quadrature)
- Cell area: $\Delta x \times \Delta y$
- Complexity: ~10ms per calculation (252 observations)

---

## Contact

* [Mariano BENJAMIN](mailto:mariano.benjamin@dauphine.eu)
* [Noah CHIKHI](mailto:noah.chikhi@dauphine.eu)
* [Dongen LIU](mailto:gongen.liu@dauphine.eu)

For issues, discussions, or contributions, please open an issue or pull request on the project’s GitHub page.

---

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2026 Entropy Replication Project Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

---

## Acknowledgments

### Academic Paper
- **Jiang, F., Wu, J., & Zhou, G.** (2018). Asymmetry in Stock Comovements: An Entropy Approach. *Journal of Financial and Quantitative Analysis*, 53(4), 1479-1507.

### Libraries & Tools
- **NumPy & SciPy** - Foundational scientific computing
- **Pandas** - Data manipulation and analysis
- **PyBind11** - Seamless C++/Python integration
- **Eigen3** - High-performance C++ linear algebra
- **Matplotlib & Seaborn** - Visualization
- **statsmodels** - Econometric analysis
- **pytest** - Testing framework
- **CMake** - Build system

### Data Sources
- **CRSP (Wharton Research Data Services)** - Stock return data
- **Kenneth French Data Library** - Fama-French factors
- **WRDS** - Database infrastructure

### Methodology References
- **Parzen, E.** (1962). On Estimation of a Probability Density Function and Mode
- **Silverman, B. W.** (1986). Density Estimation for Statistics and Data Analysis
- **Newey, W. K., & West, K. D.** (1987). HAC Covariance Matrix Estimator
- **Hong, Y., Tu, J., & Zhou, G.** (2007). Asymmetries in Stock Returns (HTZ Test)

---

**Status**: Last Updated: January 26, 2026

