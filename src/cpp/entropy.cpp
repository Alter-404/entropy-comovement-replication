/*
    Implementation of Asymmetry Entropy Engine
*/

#include "entropy.h"
#include <omp.h>
#include <limits>
#include <iostream>
#include <algorithm>

namespace entropy {

double EntropyEngine::compute_loo_likelihood(
    const Eigen::VectorXd& x,
    const Eigen::VectorXd& y,
    double h1,
    double h2
) const {
    const int n = x.size();
    double log_likelihood = 0.0;
    const double norm_factor = 1.0 / ((n - 1) * h1 * h2);

    #pragma omp parallel for reduction(+:log_likelihood)
    for (int i = 0; i < n; ++i) {
        double density_sum = 0.0;
        const double xi = x(i);
        const double yi = y(i);

        for (int j = 0; j < n; ++j) {
            if (i == j) continue; // Leave-one-out
            const double dx = (xi - x(j)) / h1;
            const double dy = (yi - y(j)) / h2;
            density_sum += gaussian_kernel(dx) * gaussian_kernel(dy);
        }
        
        // Avoid log(0)
        if (density_sum > 1e-12) {
            log_likelihood += std::log(density_sum * norm_factor);
        }
    }
    
    return log_likelihood;
}

double EntropyEngine::golden_section_search(
    std::function<double(double)> f,
    double a,
    double b,
    double tol
) const {
    const double phi = (1.0 + std::sqrt(5.0)) / 2.0;
    const double resphi = 2.0 - phi;
    
    double x1 = a + resphi * (b - a);
    double x2 = b - resphi * (b - a);
    double f1 = f(x1);
    double f2 = f(x2);
    
    while (std::abs(b - a) > tol) {
        if (f1 < f2) {
            b = x2;
            x2 = x1;
            f2 = f1;
            x1 = a + resphi * (b - a);
            f1 = f(x1);
        } else {
            a = x1;
            x1 = x2;
            f1 = f2;
            x2 = b - resphi * (b - a);
            f2 = f(x2);
        }
    }
    
    return (a + b) / 2.0;
}

std::pair<double, double> EntropyEngine::optimize_bandwidths(
    const std::vector<double>& x,
    const std::vector<double>& y
) const {
    const int n = static_cast<int>(x.size());
    
    if (n <= 0) {
        return std::make_pair(0.1, 0.1);
    }
    
    // Compute standard deviations using Eigen (internal computation is safe)
    Eigen::VectorXd x_eigen = Eigen::Map<const Eigen::VectorXd>(x.data(), n);
    Eigen::VectorXd y_eigen = Eigen::Map<const Eigen::VectorXd>(y.data(), n);
    
    const double std_x = std::sqrt((x_eigen.array() - x_eigen.mean()).square().sum() / (n - 1));
    const double std_y = std::sqrt((y_eigen.array() - y_eigen.mean()).square().sum() / (n - 1));
    
    // Silverman's rule of thumb
    const double factor = 1.06 * std::pow(static_cast<double>(n), -0.2);
    
    double h1 = factor * std_x;
    double h2 = factor * std_y;
    
    // Ensure reasonable bounds
    h1 = std::max(0.05, std::min(1.0, h1));
    h2 = std::max(0.05, std::min(1.0, h2));
    
    return std::make_pair(h1, h2);
}

Eigen::MatrixXd EntropyEngine::compute_density_grid(
    const Eigen::VectorXd& x,
    const Eigen::VectorXd& y,
    double h1,
    double h2
) const {
    const int n = x.size();
    Eigen::MatrixXd density = Eigen::MatrixXd::Zero(GRID_SIZE, GRID_SIZE);
    Eigen::VectorXd grid_axis = Eigen::VectorXd::LinSpaced(GRID_SIZE, GRID_MIN, GRID_MAX);
    
    const double norm_factor = 1.0 / (n * h1 * h2);
    
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < GRID_SIZE; ++i) {
        for (int j = 0; j < GRID_SIZE; ++j) {
            const double gx = grid_axis(i);
            const double gy = grid_axis(j);
            double sum = 0.0;
            
            for (int k = 0; k < n; ++k) {
                const double dx = (x(k) - gx) / h1;
                const double dy = (y(k) - gy) / h2;
                sum += gaussian_kernel(dx) * gaussian_kernel(dy);
            }
            
            density(i, j) = sum * norm_factor;
        }
    }
    
    return density;
}

std::pair<double, double> EntropyEngine::calculate_metrics(
    const std::vector<double>& x_vec,
    const std::vector<double>& y_vec,
    double c
) {
    // Convert std::vector to Eigen vectors for internal computations
    Eigen::VectorXd x = Eigen::Map<const Eigen::VectorXd>(x_vec.data(), x_vec.size());
    Eigen::VectorXd y = Eigen::Map<const Eigen::VectorXd>(y_vec.data(), y_vec.size());
    
    // Step 1: Optimize bandwidths
    std::pair<double, double> bandwidths = optimize_bandwidths(x_vec, y_vec);
    double h1 = bandwidths.first;
    double h2 = bandwidths.second;
    
    // Step 2: Compute density grid f(x, y)
    Eigen::MatrixXd f_xy = compute_density_grid(x, y, h1, h2);
    
    // Step 3: Compute rotated density grid f(-x, -y)
    // This is equivalent to 180-degree rotation
    Eigen::MatrixXd f_neg_x_neg_y = f_xy.colwise().reverse().rowwise().reverse();
    
    // Step 4: Compute metrics
    double integral_sq_diff = 0.0;
    double lqp = 0.0;  // Lower quadrant probability (x < -c, y < -c)
    double uqp = 0.0;  // Upper quadrant probability (x > c, y > c)
    
    Eigen::VectorXd grid = Eigen::VectorXd::LinSpaced(GRID_SIZE, GRID_MIN, GRID_MAX);
    const double cell_area = GRID_STEP * GRID_STEP;
    
    #pragma omp parallel for reduction(+:integral_sq_diff, lqp, uqp)
    for (int i = 0; i < GRID_SIZE; ++i) {
        for (int j = 0; j < GRID_SIZE; ++j) {
            const double xi = grid(i);
            const double yi = grid(j);
            const double val = f_xy(i, j);
            const double val_rot = f_neg_x_neg_y(i, j);
            
            // Integration for S_rho (Equation 7 in paper)
            // The paper integrates over BOTH tail quadrants:
            // Region 1: x > c, y > c (upper-right)
            // Region 2: x < -c, y < -c (lower-left)
            if ((xi > c && yi > c) || (xi < -c && yi < -c)) {
                const double diff = std::sqrt(std::max(0.0, val)) - std::sqrt(std::max(0.0, val_rot));
                integral_sq_diff += diff * diff;
            }
            
            // Quadrant probabilities
            if (xi > c && yi > c) {
                uqp += val;
            }
            if (xi < -c && yi < -c) {
                lqp += val;
            }
        }
    }
    
    // Apply metric normalization factors
    // Equation 7 includes 1/2 factor for S_rho
    double s_rho = 0.5 * integral_sq_diff * cell_area;
    
    // Normalize probabilities
    lqp *= cell_area;
    uqp *= cell_area;
    
    // Equation 16: sign(LQP - UQP) * S_rho
    double down_asy = (lqp >= uqp ? 1.0 : -1.0) * s_rho;
    
    return {s_rho, down_asy};
}

} // namespace entropy
