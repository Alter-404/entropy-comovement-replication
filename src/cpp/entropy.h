/*
    High-performance engine for Asymmetry Entropy calculation.
    
    Implements:
    1. Parzen-Rosenblatt KDE with Gaussian Product Kernel
    2. Likelihood Cross-Validation for Bandwidth Selection
    3. Grid-based Hellinger Distance Integration
    
    Reference: Jiang, Wu, and Zhou (2018) - "Asymmetry in Stock Comovements: 
    An Entropy Approach" JFQA
*/

#ifndef ENTROPY_H
#define ENTROPY_H

#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <utility>

namespace entropy {

// Constants
constexpr double INV_SQRT_2PI = 0.3989422804014327;
constexpr double GRID_MIN = -4.0;
constexpr double GRID_MAX = 4.0;
constexpr int GRID_SIZE = 100;
constexpr double GRID_STEP = (GRID_MAX - GRID_MIN) / GRID_SIZE;

/**
 * @brief Gaussian kernel function
 * @param z Standardized distance
 * @return Kernel value
 */
inline double gaussian_kernel(double z) {
    return INV_SQRT_2PI * std::exp(-0.5 * z * z);
}

/**
 * @brief High-performance entropy engine for comovement analysis
 */
class EntropyEngine {
public:
    EntropyEngine() = default;

    /**
     * @brief Compute Leave-One-Out Log-Likelihood for bandwidth selection
     * @param x Standardized data vector (dimension 1)
     * @param y Standardized data vector (dimension 2)
     * @param h1 Bandwidth for x dimension
     * @param h2 Bandwidth for y dimension
     * @return Log-likelihood value
     */
    double compute_loo_likelihood(
        const Eigen::VectorXd& x,
        const Eigen::VectorXd& y,
        double h1,
        double h2
    ) const;

    /**
     * @brief Optimize bandwidths using Likelihood Cross-Validation
     * @param x Standardized data vector (dimension 1)
     * @param y Standardized data vector (dimension 2)
     * @return Pair of optimal (h1, h2)
     */
    std::pair<double, double> optimize_bandwidths(
        const std::vector<double>& x,
        const std::vector<double>& y
    ) const;

    /**
     * @brief Compute density grid on [-4, 4] x [-4, 4]
     * @param x Standardized data vector (dimension 1)
     * @param y Standardized data vector (dimension 2)
     * @param h1 Bandwidth for x dimension
     * @param h2 Bandwidth for y dimension
     * @return 100x100 density matrix
     */
    Eigen::MatrixXd compute_density_grid(
        const Eigen::VectorXd& x,
        const Eigen::VectorXd& y,
        double h1,
        double h2
    ) const;

    /**
     * @brief Calculate S_rho (entropy metric) and DOWN_ASY (asymmetry metric)
     * @param x Standardized returns of asset/portfolio
     * @param y Standardized returns of market
     * @param c Threshold for quadrant definition (typically 0.0)
     * @return Pair of (S_rho, DOWN_ASY)
     */
    std::pair<double, double> calculate_metrics(
        const std::vector<double>& x,
        const std::vector<double>& y,
        double c
    );

private:
    /**
     * @brief Golden section search for univariate optimization
     * @param f Function to minimize
     * @param a Lower bound
     * @param b Upper bound
     * @param tol Tolerance
     * @return Optimal value
     */
    double golden_section_search(
        std::function<double(double)> f,
        double a,
        double b,
        double tol = 1e-4
    ) const;
};

} // namespace entropy

#endif // ENTROPY_H
