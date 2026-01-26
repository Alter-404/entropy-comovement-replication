"""
Copula-GARCH Simulation for Entropy Test Validation

Implements the Mixed Copula-GARCH Data Generating Process (DGP)
of Jiang, Wu, and Zhou (2018).

This module generates synthetic bivariate return data with controlled
asymmetry properties for validating the entropy-based asymmetry test.
"""

import numpy as np
from scipy import stats
from scipy.special import gamma
from typing import Tuple, Optional
import warnings


class CopulaSimulator:
    """
    Generate bivariate returns using Mixed Copula-GARCH models.
    
    Implements three copula types:
    1. Gaussian Copula (symmetric)
    2. Clayton Copula (lower tail dependence)
    3. Mixed Copula (convex combination of Gaussian and Clayton)
    
    Parameters
    ----------
    rho : float
        Gaussian copula correlation parameter (default: 0.7)
    tau : float
        Clayton copula parameter (default: 2.0)
    seed : int, optional
        Random seed for reproducibility
    
    References
    ----------
    Jiang, Wu, and Zhou (2018), Section III.A, equations (11)-(13)
    """
    
    def __init__(self, rho: float = 0.7, tau: float = 2.0, seed: Optional[int] = None):
        self.rho = rho
        self.tau = tau
        
        if seed is not None:
            np.random.seed(seed)
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random.RandomState()
    
    def simulate_gaussian_copula(self, n_obs: int) -> np.ndarray:
        """
        Generate bivariate uniform samples from Gaussian copula.
        
        Parameters
        ----------
        n_obs : int
            Number of observations to generate
        
        Returns
        -------
        np.ndarray
            Shape (n_obs, 2) with uniform [0,1] marginals
        """
        # Correlation matrix
        corr_matrix = np.array([[1.0, self.rho],
                                [self.rho, 1.0]])
        
        # Generate correlated standard normals
        z = self.rng.multivariate_normal(mean=[0, 0], 
                                         cov=corr_matrix, 
                                         size=n_obs)
        
        # Transform to uniform via CDF
        u = stats.norm.cdf(z)
        
        return u
    
    def simulate_clayton_copula(self, n_obs: int) -> np.ndarray:
        """
        Generate bivariate uniform samples from Clayton copula.
        
        Clayton copula has lower tail dependence, capturing the asymmetry
        where variables move together more in downturns than upturns.
        
        Parameters
        ----------
        n_obs : int
            Number of observations
        
        Returns
        -------
        np.ndarray
            Shape (n_obs, 2) with uniform [0,1] marginals
        
        Notes
        -----
        Uses the conditional sampling method:
        - Sample v1 ~ U(0,1)
        - Sample v2 ~ U(0,1)  
        - u1 = v1
        - u2 = [v1^(-θ) * (v2^(-θ/(1+θ)) - 1) + 1]^(-1/θ)
        
        where θ = tau is the Clayton parameter.
        """
        theta = self.tau
        
        # Sample independent uniforms
        v1 = self.rng.uniform(0, 1, n_obs)
        v2 = self.rng.uniform(0, 1, n_obs)
        
        # Conditional sampling
        u1 = v1
        
        # Clayton conditional: u2 | u1
        # Handle numerical issues with small values
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            u2 = (v1**(-theta) * (v2**(-theta/(1+theta)) - 1) + 1)**(-1/theta)
        
        # Clip to [0,1] to handle numerical errors
        u2 = np.clip(u2, 0, 1)
        
        return np.column_stack([u1, u2])
    
    def simulate_mixed_copula(self, n_obs: int, kappa: float) -> np.ndarray:
        """
        Generate bivariate uniform samples from Mixed copula.
        
        With probability κ, draw from Gaussian copula.
        With probability (1-κ), draw from Clayton copula.
        
        Parameters
        ----------
        n_obs : int
            Number of observations
        kappa : float
            Mixing weight ∈ [0, 1], weight on Gaussian copula
        
        Returns
        -------
        np.ndarray
            Shape (n_obs, 2) with uniform [0,1] marginals
        """
        if not 0 <= kappa <= 1:
            raise ValueError(f"kappa must be in [0,1], got {kappa}")
        
        # Determine source for each observation
        source = self.rng.binomial(1, kappa, n_obs)  # 1 = Gaussian, 0 = Clayton
        
        # Generate from both copulas
        gaussian_samples = self.simulate_gaussian_copula(n_obs)
        clayton_samples = self.simulate_clayton_copula(n_obs)
        
        # Mix based on source
        u = np.where(source[:, np.newaxis], gaussian_samples, clayton_samples)
        
        return u
    
    def apply_garch_marginals(self, uniform_data: np.ndarray,
                             omega: float = 0.01,
                             alpha: float = 0.05,
                             beta: float = 0.90) -> np.ndarray:
        """
        Transform uniform copula variates to returns via GARCH(1,1) marginals.
        
        GARCH(1,1) model:
        r_t = σ_t * ε_t
        σ_t² = ω + α*r_{t-1}² + β*σ_{t-1}²
        ε_t ~ N(0,1)
        
        Parameters
        ----------
        uniform_data : np.ndarray
            Shape (n_obs, 2) with uniform [0,1] marginals
        omega : float
            GARCH constant term (ω)
        alpha : float
            ARCH parameter (α)
        beta : float
            GARCH parameter (β)
        
        Returns
        -------
        np.ndarray
            Shape (n_obs, 2) with GARCH returns
        """
        n_obs, n_vars = uniform_data.shape
        
        # Transform uniform to standard normal innovations
        epsilon = stats.norm.ppf(uniform_data)
        
        # Replace any inf values with large finite values
        epsilon = np.clip(epsilon, -10, 10)
        
        # Initialize GARCH process
        returns = np.zeros_like(epsilon)
        sigma_sq = np.full((n_obs, n_vars), omega / (1 - alpha - beta))  # Unconditional variance
        
        # Simulate GARCH paths
        for t in range(1, n_obs):
            sigma_sq[t] = omega + alpha * returns[t-1]**2 + beta * sigma_sq[t-1]
            returns[t] = np.sqrt(sigma_sq[t]) * epsilon[t]
        
        # First observation
        returns[0] = np.sqrt(sigma_sq[0]) * epsilon[0]
        
        return returns
    
    def generate_returns(self, n_obs: int, 
                        copula_type: str = 'gaussian',
                        kappa: float = 0.5,
                        garch: bool = True) -> np.ndarray:
        """
        Generate bivariate returns with specified copula and marginals.
        
        Parameters
        ----------
        n_obs : int
            Number of observations
        copula_type : str
            'gaussian', 'clayton', or 'mixed'
        kappa : float
            Mixing weight for mixed copula (ignored for others)
        garch : bool
            If True, apply GARCH marginals; if False, use standard normal
        
        Returns
        -------
        np.ndarray
            Shape (n_obs, 2) with standardized returns
        """
        # Generate copula samples
        if copula_type.lower() == 'gaussian':
            u = self.simulate_gaussian_copula(n_obs)
        elif copula_type.lower() == 'clayton':
            u = self.simulate_clayton_copula(n_obs)
        elif copula_type.lower() == 'mixed':
            u = self.simulate_mixed_copula(n_obs, kappa)
        else:
            raise ValueError(f"Unknown copula type: {copula_type}")
        
        # Apply marginals
        if garch:
            returns = self.apply_garch_marginals(u)
        else:
            # Simple standard normal marginals
            returns = stats.norm.ppf(u)
            returns = np.clip(returns, -10, 10)
        
        # Standardize to mean=0, std=1
        returns = (returns - returns.mean(axis=0)) / returns.std(axis=0, ddof=1)
        
        return returns
    
    def generate_analytical_density(self, copula_type: str = 'gaussian',
                                   grid_size: int = 100,
                                   x_range: Tuple[float, float] = (-4, 4),
                                   kappa: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute analytical copula density on a grid for visualization.
        
        Parameters
        ----------
        copula_type : str
            'gaussian', 'clayton', or 'mixed'
        grid_size : int
            Number of grid points per dimension
        x_range : tuple
            (min, max) range for grid
        kappa : float
            Mixing weight (for mixed copula)
        
        Returns
        -------
        X : np.ndarray
            X-coordinate grid (grid_size, grid_size)
        Y : np.ndarray
            Y-coordinate grid (grid_size, grid_size)
        density : np.ndarray
            Density values on grid (grid_size, grid_size)
        """
        # Create grid
        x = np.linspace(x_range[0], x_range[1], grid_size)
        y = np.linspace(x_range[0], x_range[1], grid_size)
        X, Y = np.meshgrid(x, y)
        
        # Transform to uniform for copula evaluation
        U = stats.norm.cdf(X)
        V = stats.norm.cdf(Y)
        
        # Evaluate copula density
        if copula_type.lower() == 'gaussian':
            density = self._gaussian_copula_density(U, V)
        elif copula_type.lower() == 'clayton':
            density = self._clayton_copula_density(U, V)
        elif copula_type.lower() == 'mixed':
            dens_gauss = self._gaussian_copula_density(U, V)
            dens_clayton = self._clayton_copula_density(U, V)
            density = kappa * dens_gauss + (1 - kappa) * dens_clayton
        else:
            raise ValueError(f"Unknown copula type: {copula_type}")
        
        # Multiply by marginal densities (standard normal)
        marginal_density = stats.norm.pdf(X) * stats.norm.pdf(Y)
        joint_density = density * marginal_density
        
        return X, Y, joint_density
    
    def _gaussian_copula_density(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Gaussian copula density function."""
        rho = self.rho
        
        # Transform to normal quantiles
        x = stats.norm.ppf(np.clip(u, 1e-10, 1-1e-10))
        y = stats.norm.ppf(np.clip(v, 1e-10, 1-1e-10))
        
        # Copula density
        exp_term = -0.5 * rho * (2*x*y - rho*(x**2 + y**2)) / (1 - rho**2)
        density = np.exp(exp_term) / np.sqrt(1 - rho**2)
        
        return density
    
    def _clayton_copula_density(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Clayton copula density function."""
        theta = self.tau
        
        # Avoid numerical issues
        u = np.clip(u, 1e-10, 1-1e-10)
        v = np.clip(v, 1e-10, 1-1e-10)
        
        # Clayton copula density
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            density = (1 + theta) * (u * v)**(-1-theta) * (u**(-theta) + v**(-theta) - 1)**(-2-1/theta)
        
        # Replace inf/nan with large/small finite values
        density = np.nan_to_num(density, nan=0.0, posinf=100.0, neginf=0.0)
        
        return density


def create_figure1_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate data for Figure 1: Symmetry Concept illustration.
    
    Figure 1 shows two examples:
    - Top-right quadrant: Symmetric (Normal with ρ = 0.7)
    - Bottom-left quadrant: Asymmetric (Mixture of circle + line)
    
    Returns
    -------
    X_sym, Y_sym : np.ndarray
        Grid coordinates for symmetric example
    density_sym : np.ndarray
        Density for symmetric example
    X_asym, Y_asym : np.ndarray
        Grid coordinates for asymmetric example  
    density_asym : np.ndarray
        Density for asymmetric example
    """
    # Symmetric example: Bivariate Normal with ρ = 0.7
    simulator_sym = CopulaSimulator(rho=0.7)
    X_sym, Y_sym, density_sym = simulator_sym.generate_analytical_density(
        copula_type='gaussian',
        grid_size=100,
        x_range=(-3, 3)
    )
    
    # Asymmetric example: Mixture of two components
    # Component A: Circular (independent, ρ = 0)
    # Component B: Diagonal line (ρ = 0.95)
    simulator_a = CopulaSimulator(rho=0.0)
    _, _, density_a = simulator_a.generate_analytical_density(
        copula_type='gaussian',
        grid_size=100,
        x_range=(-3, 3)
    )
    
    simulator_b = CopulaSimulator(rho=0.95)
    X_asym, Y_asym, density_b = simulator_b.generate_analytical_density(
        copula_type='gaussian',
        grid_size=100,
        x_range=(-3, 3)
    )
    
    # Mix with 50-50 weights
    density_asym = 0.5 * density_a + 0.5 * density_b
    
    return X_sym, Y_sym, density_sym, X_asym, Y_asym, density_asym


def create_figure2_data() -> dict:
    """
    Generate data for Figure 2: Copula comparison (4 panels).
    
    Panels:
    A. Clayton Copula (κ = 0, lower tail dependence)
    B. Gaussian Copula (κ = 1, symmetric)
    C. Mixed Copula (κ = 0.5, moderate asymmetry)
    D. Real/Simulated Data (using Mixed with empirical parameters)
    
    Returns
    -------
    dict
        Dictionary with keys 'A', 'B', 'C', 'D', each containing
        (X, Y, density) tuples
    """
    simulator = CopulaSimulator(rho=0.7, tau=2.0)
    
    panels = {}
    
    # Panel A: Clayton
    X_a, Y_a, density_a = simulator.generate_analytical_density(
        copula_type='clayton',
        grid_size=100,
        x_range=(-3, 3)
    )
    panels['A'] = (X_a, Y_a, density_a)
    
    # Panel B: Gaussian
    X_b, Y_b, density_b = simulator.generate_analytical_density(
        copula_type='gaussian',
        grid_size=100,
        x_range=(-3, 3)
    )
    panels['B'] = (X_b, Y_b, density_b)
    
    # Panel C: Mixed
    X_c, Y_c, density_c = simulator.generate_analytical_density(
        copula_type='mixed',
        kappa=0.5,
        grid_size=100,
        x_range=(-3, 3)
    )
    panels['C'] = (X_c, Y_c, density_c)
    
    # Panel D: Simulated data (KDE on generated samples)
    # For visualization, use Clayton with slight noise
    returns = simulator.generate_returns(n_obs=5000, copula_type='clayton', garch=False)
    
    # For density, just reuse mixed as proxy
    panels['D'] = (X_c, Y_c, density_c)  # Placeholder - will use KDE in plotting script
    panels['D_data'] = returns  # Actual data for KDE
    
    return panels
