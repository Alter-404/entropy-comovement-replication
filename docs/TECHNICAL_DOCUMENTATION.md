# Technical Documentation

## Entropy-Comovement Replication Project

**Paper**: Jiang, Wu, and Zhou (2018) - "Asymmetry in Stock Comovements: An Entropy Approach"  
**Journal**: Journal of Financial and Quantitative Analysis  
**Version**: 1.0.0

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Mathematical Foundations](#mathematical-foundations)
3. [C++ Engine (`entropy_cpp`)](#c-engine)
4. [Python Modules](#python-modules)
5. [Data Pipeline](#data-pipeline)
6. [Replication Scripts](#replication-scripts)
7. [API Reference](#api-reference)
8. [Performance Optimization](#performance-optimization)
9. [Testing Framework](#testing-framework)
10. [Troubleshooting](#troubleshooting)

---

## Architecture Overview

### System Design

```
┌─────────────────────────────────────────────────────────────────┐
│                         main.py                                 │
│                    (Entry Point / CLI)                          │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│              scripts/run_full_replication.py                    │
│                   (ReplicationPipeline)                         │
│  ┌──────────┬──────────┬──────────┬──────────┬──────────┐       │
│  │ Phase 1  │ Phase 2  │ Phase 3  │ Phase 4  │ Phase 5  │ ...   │
│  └────┬─────┴────┬─────┴────┬─────┴────┬─────┴────┬─────┘       │
└───────┼──────────┼──────────┼──────────┼──────────┼─────────────┘
        │          │          │          │          │
        ▼          ▼          ▼          ▼          ▼
┌───────────────────────────────────────────────────────────────┐
│                    scripts/replicate_*.py                     │
│         (Table/Figure Generation Scripts)                     │
└───────────────────────────────────────────────────────────────┘
        │                              │
        ▼                              ▼
┌─────────────────────┐    ┌─────────────────────────────────┐
│   src/python/       │    │       src/cpp/                  │
│  ├─ data_loader.py  │    │  ├─ entropy.cpp                 │
│  ├─ simulation.py   │    │  ├─ entropy.h                   │
│  ├─ portfolio.py    │    │  ├─ bindings.cpp                │
│  └─ characteristics │    │  └─ CMakeLists.txt              │
└─────────────────────┘    └───────────────┬─────────────────┘
                                           │
                                           ▼
                           ┌───────────────────────────────┐
                           │    entropy_cpp.so             │
                           │  (Compiled Python Extension)  │
                           └───────────────────────────────┘
```

### Directory Structure

| Directory | Purpose |
|-----------|---------|
| `src/cpp/` | High-performance C++ entropy engine |
| `src/python/` | Python analysis modules |
| `scripts/` | Replication and utility scripts |
| `tests/` | Unit and integration tests |
| `data/raw/` | Input data (CRSP, Fama-French) |
| `data/processed/` | Intermediate parquet files |
| `data/cache/` | Cached computations |
| `outputs/tables/` | Generated CSV tables |
| `outputs/figures/` | Generated PDF/PNG figures |
| `reports/` | Markdown replication reports |

---

## Mathematical Foundations

### Entropy-Based Asymmetry Measure

The core innovation is an entropy-based measure of asymmetry in stock return comovements.

#### Kernel Density Estimation

The bivariate probability density function is estimated using the Parzen-Rosenblatt estimator:

$$\hat{f}(x, y) = \frac{1}{n h_1 h_2} \sum_{i=1}^{n} K\left(\frac{x - X_i}{h_1}\right) K\left(\frac{y - Y_i}{h_2}\right)$$

Where:
- $K(\cdot)$ is the Gaussian kernel: $K(z) = \frac{1}{\sqrt{2\pi}} e^{-z^2/2}$
- $h_1, h_2$ are bandwidth parameters
- $(X_i, Y_i)$ are standardized return observations

#### Bandwidth Selection (Likelihood Cross-Validation)

Optimal bandwidths are selected by maximizing the leave-one-out likelihood:

$$\text{LCV}(h_1, h_2) = \sum_{i=1}^{n} \log \hat{f}_{-i}(X_i, Y_i)$$

Where $\hat{f}_{-i}$ is the density estimate excluding observation $i$.

#### Hellinger Distance

The asymmetry metric $S_\rho$ is defined as half the squared Hellinger distance:

$$S_\rho = \frac{1}{2} \int \int \left(\sqrt{f(x,y)} - \sqrt{f(-x,-y)}\right)^2 dx\, dy$$

This measures the divergence between the joint density and its 180° rotation.

#### Downside Asymmetry (DOWN_ASY)

The signed asymmetry measure:

$$\text{DOWN\_ASY} = \text{sign}(\text{LQP} - \text{UQP}) \times S_\rho$$

Where:
- **LQP** (Lower Quadrant Probability): $P(X < c, Y < c)$
- **UQP** (Upper Quadrant Probability): $P(X > c, Y > c)$
- $c$ is the threshold (typically 0 for standardized returns)

### Statistical Tests

#### Entropy Test Statistic

$$T_n = n \cdot S_\rho \xrightarrow{d} \chi^2$$

Under the null hypothesis of symmetric comovements.

#### Bootstrap Inference

For finite samples, p-values are computed via stationary block bootstrap:
1. Resample paired observations with replacement
2. Recompute $S_\rho$ for each bootstrap sample
3. Calculate: $\text{p-value} = \frac{1}{B} \sum_{b=1}^{B} \mathbf{1}(S_\rho^{(b)} \geq S_\rho)$

---

## C++ Engine

### Module: `entropy_cpp`

The high-performance C++ engine provides the computational backbone.

#### Class: `EntropyEngine`

```cpp
class EntropyEngine {
public:
    EntropyEngine();
    
    // Main computation method
    std::pair<double, double> calculate_metrics(
        py::array_t<double> x,  // Standardized returns (asset)
        py::array_t<double> y,  // Standardized returns (market)
        double c = 0.0          // Threshold for quadrant probabilities
    );
    
    // Bandwidth optimization
    std::pair<double, double> optimize_bandwidths(
        const VectorXd& x,
        const VectorXd& y
    );
    
    // Density grid computation
    MatrixXd compute_density_grid(
        const VectorXd& x,
        const VectorXd& y,
        double h1,
        double h2
    );
    
private:
    static constexpr int GRID_SIZE = 100;
    static constexpr double GRID_MIN = -4.0;
    static constexpr double GRID_MAX = 4.0;
};
```

#### Compilation

```bash
# From project root
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Or use the build script
./scripts/build_cpp.sh
```

#### Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| Eigen3 | 3.4+ | Linear algebra operations |
| OpenMP | 4.5+ | Multi-threading parallelization |
| pybind11 | 2.10+ | Python bindings |

#### Performance Characteristics

| Operation | Time Complexity | Space Complexity |
|-----------|-----------------|------------------|
| KDE Evaluation | O(n × G²) | O(G²) |
| Bandwidth Optimization | O(n² × I) | O(n) |
| Hellinger Integration | O(G²) | O(G²) |

Where:
- n = number of observations
- G = grid size (100)
- I = optimization iterations

---

## Python Modules

### `src/python/data_loader.py`

Handles data ingestion and preprocessing.

```python
class DataLoader:
    def __init__(self, raw_data_dir: str, cache_dir: str):
        """Initialize with data directories."""
        
    def load_data(self, force_refresh: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load aligned stock and factor data.
        
        Returns:
            stocks: DataFrame with columns [PERMNO, DATE, RET, EXRET, ...]
            factors: DataFrame with columns [DATE, MKT_RF, SMB, HML, MOM, RF]
        """
        
    @staticmethod
    def get_standardized_returns(series: pd.Series) -> np.ndarray:
        """Standardize returns to mean=0, std=1."""
```

### `src/python/simulation.py`

Monte Carlo simulation for Table 1.

```python
class CopulaSimulator:
    def __init__(self, rho: float = 0.951, tau: float = 5.768):
        """Initialize copula parameters."""
        
    def simulate_mixed_copula(self, n_obs: int, kappa: float) -> np.ndarray:
        """
        Generate bivariate data from Mixed Gaussian-Clayton Copula.
        
        Args:
            n_obs: Number of observations
            kappa: Weight on Gaussian copula (0.0 to 1.0)
            
        Returns:
            Array of shape (n_obs, 2) with uniform variates
        """
        
    def apply_garch_marginals(self, uniform_data: np.ndarray) -> np.ndarray:
        """Transform uniform variates to GARCH returns."""
```

### `src/python/portfolio.py`

Portfolio construction and sorting.

```python
def create_quintile_portfolios(
    data: pd.DataFrame,
    sort_var: str,
    ret_var: str = 'RET',
    weight_var: str = None
) -> pd.DataFrame:
    """
    Create quintile portfolios sorted by a variable.
    
    Args:
        data: Stock-level data with sort variable and returns
        sort_var: Column to sort on (e.g., 'DOWN_ASY')
        ret_var: Return column
        weight_var: Optional market cap for value-weighting
        
    Returns:
        DataFrame with portfolio returns by date and quintile
    """

def double_sort_portfolios(
    data: pd.DataFrame,
    control_var: str,
    sort_var: str,
    n_portfolios: int = 5
) -> pd.DataFrame:
    """
    Sequential double-sort: first by control, then by sort_var within.
    """
```

### `src/python/characteristics.py`

Firm characteristic calculations.

```python
def compute_rolling_characteristics(
    stock_returns: pd.DataFrame,
    mkt_returns: pd.Series,
    window: int = 252
) -> pd.DataFrame:
    """
    Compute rolling firm characteristics.
    
    Returns DataFrame with columns:
        - BETA: Market beta
        - DOWNSIDE_BETA: Beta in down markets
        - UPSIDE_BETA: Beta in up markets
        - IVOL: Idiosyncratic volatility
        - SIZE: Log market cap
        - BM: Book-to-market ratio
        - MOM: 12-month momentum
        - ILLIQ: Amihud illiquidity
        - COSKEW: Coskewness with market
        - COKURT: Cokurtosis with market
    """
```

---

## Data Pipeline

### Input Data Requirements

#### CRSP Data

| File | Columns Required |
|------|------------------|
| `CRSP Daily Stock.csv` | PERMNO, date, RET, PRC, VOL, SHROUT |
| `CRSP Monthly Stock.csv` | PERMNO, date, RET, PRC, VOL, SHROUT, SHRCD |

#### Fama-French Data

| File | Columns Required |
|------|------------------|
| `F-F_Research_Data_Factors.csv` | date, Mkt-RF, SMB, HML, RF |
| `F-F_Momentum_Factor.csv` | date, Mom |
| `Portfolios_Formed_on_ME.csv` | Size decile returns |
| `Portfolios_Formed_on_BE-ME.csv` | B/M decile returns |
| `10_Portfolios_Prior_12_2.csv` | Momentum decile returns |

### Data Processing Pipeline

```
Raw CSV Files
     │
     ▼
┌─────────────────────┐
│   DataLoader        │
│  - Parse dates      │
│  - Handle missing   │
│  - Align datasets   │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   Preprocessing     │
│  - Compute EXRET    │
│  - Filter < 100 obs │
│  - Standardize      │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   Rolling Windows   │
│  - 12-month lookback│
│  - DOWN_ASY calc    │
│  - Characteristics  │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────────────────────┐
│        Parquet Cache                │
│  - down_asy_scores.parquet          │
│  - firm_characteristics.parquet     │
└─────────────────────────────────────┘
```

### Intermediate Files

| File | Schema | Description |
|------|--------|-------------|
| `down_asy_scores.parquet` | PERMNO, DATE, DOWN_ASY, S_RHO | Monthly asymmetry scores |
| `firm_characteristics.parquet` | PERMNO, DATE, BETA, IVOL, ... | 16 firm characteristics |
| `crsp_processed.parquet` | PERMNO, DATE, RET, EXRET, ... | Cleaned CRSP data |
| `ff_processed.parquet` | DATE, MKT_RF, SMB, HML, ... | Factor returns |

---

## Replication Scripts

### Table Generation

| Script | Output | Description |
|--------|--------|-------------|
| `replicate_table1.py` | `Table_1_Size_Power.csv` | Monte Carlo size/power tests |
| `replicate_table2.py` | `Table_2_Asymmetry_Tests.csv` | Asymmetry tests for 30 portfolios |
| `replicate_table3.py` | `Table_3_Correlations.csv` | Cross-sectional correlations |
| `replicate_table4.py` | `Table_4_Summary_Stats.csv` | Characteristics by DOWN_ASY decile |
| `replicate_table5.py` | `Table_5_Returns_Alphas.csv` | Portfolio returns and alphas |
| `replicate_table6.py` | `Table_6_Time_Series_Reg.csv` | Premium determinants |
| `replicate_table7.py` | `Table_7_Double_Sorts.csv` | Robustness double-sorts |

### Figure Generation

| Script | Output | Description |
|--------|--------|-------------|
| `replicate_fig1_2.py` | `Figure_1_*.pdf`, `Figure_2_*.pdf` | Symmetry concept, copula comparison |
| `plot_power_curve.py` | `Figure_3_Power_Analysis.pdf` | Power curve by sample size |
| `plot_asymmetry_distribution.py` | `Figure_4_Asymmetry_Distribution.pdf` | DOWN_ASY distribution |
| `plot_equity_curve.py` | `Figure_5_Cumulative_Returns.pdf` | Strategy performance |
| `plot_premium_dynamics.py` | `Figure_6_*.pdf` | Time-varying premium |

### Master Pipeline

```python
# scripts/run_full_replication.py

class ReplicationPipeline:
    def __init__(self, demo_mode: bool = False):
        """Initialize pipeline with mode."""
        
    def run(self) -> Dict[str, bool]:
        """
        Execute full replication pipeline.
        
        Phases:
            1. Data preprocessing
            2. Portfolio construction
            3. Factor regressions (Tables 1, 2, 5)
            4. Firm characteristics (Tables 3, 4, 6)
            5. Robustness checks (Table 7)
            6. Report generation (Figures 1-6)
            
        Returns:
            Dictionary mapping phase names to success status
        """
```

---

## API Reference

### Command-Line Interface

```bash
# Full replication (production mode)
python main.py

# Demo mode (synthetic data)
python main.py --demo

# Run specific table
python scripts/replicate_table5.py --output outputs/tables/

# Generate figures only
python scripts/replicate_fig1_2.py --output outputs/figures/
```

### Python API

```python
# Direct engine usage
import entropy_cpp

engine = entropy_cpp.EntropyEngine()
x = np.random.randn(252)  # Standardized stock returns
y = np.random.randn(252)  # Standardized market returns

s_rho, down_asy = engine.calculate_metrics(x, y, c=0.0)
print(f"S_rho: {s_rho:.4f}, DOWN_ASY: {down_asy:.4f}")
```

```python
# Pipeline usage
from scripts.run_full_replication import ReplicationPipeline

pipeline = ReplicationPipeline(demo_mode=True)
results = pipeline.run()

for phase, success in results.items():
    print(f"{phase}: {'✓' if success else '✗'}")
```

---

## Performance Optimization

### C++ Optimizations

1. **SIMD Vectorization**: Eigen's vectorized operations for KDE
2. **OpenMP Parallelization**: `#pragma omp parallel for collapse(2)` for grid computation
3. **Memory Locality**: Grid-based integration instead of numerical quadrature
4. **Stack Allocation**: Pre-allocated matrices for small kernels

### Python Optimizations

1. **Parquet Caching**: Binary format for fast I/O
2. **Vectorized Operations**: NumPy/Pandas over loops
3. **Multiprocessing**: `Pool` for independent computations
4. **Chunked Processing**: Memory-efficient large dataset handling

### Benchmarks

| Operation | Time (n=252) | Time (n=1000) |
|-----------|--------------|---------------|
| Single DOWN_ASY | ~5 ms | ~20 ms |
| Full stock-month (3000 × 600) | ~30 min | N/A |
| Table 1 (1000 sims × 6 panels) | ~10 min | ~45 min |

---

## Testing Framework

### Unit Tests

```bash
# C++ tests (Catch2)
cd build && ctest --output-on-failure

# Python tests
pytest tests/ -v
```

### Test Categories

| Test File | Coverage |
|-----------|----------|
| `test_phase1.cpp` | C++ engine correctness |
| `test_phase2.py` | Data loading pipeline |
| `test_phase3.py` | Simulation engine |
| `test_phase4.py` | Portfolio construction |
| `test_phase5.py` | Characteristic calculations |
| `test_integration.py` | End-to-end pipeline |
| `test_validation.py` | Output verification |

### Validation Criteria

1. **KDE Accuracy**: Within 1e-6 of `statsmodels.KDEMultivariate`
2. **Integration Accuracy**: Within 1e-4 of `scipy.integrate.dblquad`
3. **Symmetry Test**: $S_\rho \approx 0$ for Gaussian data
4. **Premium Sign**: High-Low spread positive (matching paper)

---

## Troubleshooting

### Common Issues

#### Build Failures

```bash
# Missing Eigen
sudo apt install libeigen3-dev  # Ubuntu
brew install eigen              # macOS

# OpenMP not found
# Add to CMakeLists.txt:
# find_package(OpenMP REQUIRED)
```

#### Import Errors

```python
# entropy_cpp not found
import sys
sys.path.insert(0, 'build')  # Add build directory

# Or rebuild:
./scripts/build_cpp.sh
```

#### Memory Issues

```python
# Large dataset processing
# Use chunked processing:
for chunk in pd.read_parquet(file, chunksize=10000):
    process(chunk)
```

#### Data Loading Errors

```python
# Check column names
df = pd.read_csv('data/raw/CRSP.csv')
print(df.columns.tolist())

# Normalize column names
df.columns = df.columns.str.upper().str.strip()
```

### Debug Mode

```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Run pipeline with debugging
pipeline = ReplicationPipeline(demo_mode=True)
pipeline.verbose = True
pipeline.run()
```

---

## Appendix

### Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Build C++ extension
./scripts/build_cpp.sh

# Verify installation
python -c "import entropy_cpp; print('OK')"
```

### Configuration

Environment variables:
- `ENTROPY_GRID_SIZE`: Override default grid size (100)
- `ENTROPY_PARALLEL`: Enable/disable OpenMP (`1`/`0`)
- `DATA_DIR`: Override default data directory

### References

1. Jiang, G. J., Wu, G., & Zhou, Y. (2018). Asymmetry in Stock Comovements: An Entropy Approach. *Journal of Financial and Quantitative Analysis*, 53(4), 1479-1507.

2. Harvey, C. R., & Siddique, A. (2000). Conditional Skewness in Asset Pricing Tests. *Journal of Finance*, 55(3), 1263-1295.

3. Ang, A., Chen, J., & Xing, Y. (2006). Downside Risk. *Review of Financial Studies*, 19(4), 1191-1239.

---

*Last updated: January 2026*
