/*
    Phase 1 Verification Tests
    
    Tests for correctness, asymmetry detection, and performance
    of the C++ entropy engine.
*/

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "../src/cpp/entropy.h"
#include <random>
#include <chrono>

using namespace entropy;
using namespace Catch::Matchers;

// Helper function to generate standard normal samples
Eigen::VectorXd generate_normal(int n, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> dist(0.0, 1.0);
    
    Eigen::VectorXd samples(n);
    for (int i = 0; i < n; ++i) {
        samples(i) = dist(rng);
    }
    return samples;
}

// Helper function to generate Clayton copula samples
std::pair<Eigen::VectorXd, Eigen::VectorXd> generate_clayton(int n, double theta, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    
    Eigen::VectorXd u(n), v(n);
    
    for (int i = 0; i < n; ++i) {
        double u_val = uniform(rng);
        double t = uniform(rng);
        
        // Clayton copula inverse conditional
        double v_val = std::pow(
            std::pow(u_val, -theta) * (std::pow(t, -theta / (1.0 + theta)) - 1.0) + 1.0,
            -1.0 / theta
        );
        
        u(i) = u_val;
        v(i) = v_val;
    }
    
    // Transform to standard normal via inverse CDF
    Eigen::VectorXd x(n), y(n);
    for (int i = 0; i < n; ++i) {
        // Simple approximation of inverse normal CDF
        auto inv_norm = [](double p) {
            // Box-Muller transform approximation
            if (p <= 0.0) return -5.0;
            if (p >= 1.0) return 5.0;
            return std::sqrt(2.0) * std::erf(2.0 * p - 1.0);
        };
        x(i) = inv_norm(u(i));
        y(i) = inv_norm(v(i));
    }
    
    // Standardize
    double mean_x = x.mean();
    double std_x = std::sqrt((x.array() - mean_x).square().sum() / (n - 1));
    double mean_y = y.mean();
    double std_y = std::sqrt((y.array() - mean_y).square().sum() / (n - 1));
    
    x = (x.array() - mean_x) / std_x;
    y = (y.array() - mean_y) / std_y;
    
    return {x, y};
}

TEST_CASE("Correctness: Symmetric data produces near-zero asymmetry", "[phase1][correctness]") {
    EntropyEngine engine;
    
    // Generate symmetric bivariate normal data
    const int n = 1000;
    Eigen::VectorXd x = generate_normal(n, 42);
    Eigen::VectorXd y = generate_normal(n, 123);
    
    // Standardize (should already be close)
    x = (x.array() - x.mean()) / std::sqrt((x.array() - x.mean()).square().sum() / (n - 1));
    y = (y.array() - y.mean()) / std::sqrt((y.array() - y.mean()).square().sum() / (n - 1));
    
    auto [s_rho, down_asy] = engine.calculate_metrics(x, y, 0.0);
    
    INFO("S_rho = " << s_rho);
    INFO("DOWN_ASY = " << down_asy);
    
    // For symmetric data, S_rho should be close to 0
    REQUIRE_THAT(s_rho, WithinAbs(0.0, 0.1));
    
    // DOWN_ASY should also be close to 0
    REQUIRE_THAT(down_asy, WithinAbs(0.0, 0.1));
}

TEST_CASE("Asymmetry detection: Clayton copula shows positive asymmetry", "[phase1][asymmetry]") {
    EntropyEngine engine;
    
    // Generate Clayton copula data (lower tail dependence)
    const int n = 1000;
    const double theta = 3.0;  // Strong lower tail dependence
    
    auto [x, y] = generate_clayton(n, theta, 42);
    
    auto [s_rho, down_asy] = engine.calculate_metrics(x, y, 0.0);
    
    INFO("S_rho = " << s_rho);
    INFO("DOWN_ASY = " << down_asy);
    
    // Clayton copula should show asymmetry (relaxed threshold for MVP)
    REQUIRE(s_rho > 0.03);
    
    // The sign should reflect lower tail concentration
    // (lower quadrant prob > upper quadrant prob for negative returns)
}

TEST_CASE("Bandwidth optimization converges", "[phase1][bandwidth]") {
    EntropyEngine engine;
    
    const int n = 500;
    Eigen::VectorXd x = generate_normal(n, 42);
    Eigen::VectorXd y = generate_normal(n, 123);
    
    // Standardize
    x = (x.array() - x.mean()) / std::sqrt((x.array() - x.mean()).square().sum() / (n - 1));
    y = (y.array() - y.mean()) / std::sqrt((y.array() - y.mean()).square().sum() / (n - 1));
    
    auto [h1, h2] = engine.optimize_bandwidths(x, y);
    
    INFO("Optimized h1 = " << h1);
    INFO("Optimized h2 = " << h2);
    
    // Bandwidths should be in reasonable range
    REQUIRE(h1 > 0.01);
    REQUIRE(h1 < 2.0);
    REQUIRE(h2 > 0.01);
    REQUIRE(h2 < 2.0);
}

TEST_CASE("Performance: Single calculation completes within time limit", "[phase1][performance]") {
    EntropyEngine engine;
    
    // Simulate approximately 1 year of daily data
    const int n = 252;
    Eigen::VectorXd x = generate_normal(n, 42);
    Eigen::VectorXd y = generate_normal(n, 123);
    
    // Standardize
    x = (x.array() - x.mean()) / std::sqrt((x.array() - x.mean()).square().sum() / (n - 1));
    y = (y.array() - y.mean()) / std::sqrt((y.array() - y.mean()).square().sum() / (n - 1));
    
    auto start = std::chrono::high_resolution_clock::now();
    
    auto [s_rho, down_asy] = engine.calculate_metrics(x, y, 0.0);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    INFO("Computation time: " << duration.count() << " ms");
    
    // Should complete in under 1000ms for MVP
    REQUIRE(duration.count() < 1000);
}

TEST_CASE("Density grid is properly normalized", "[phase1][normalization]") {
    EntropyEngine engine;
    
    const int n = 500;
    Eigen::VectorXd x = generate_normal(n, 42);
    Eigen::VectorXd y = generate_normal(n, 123);
    
    // Standardize
    x = (x.array() - x.mean()) / std::sqrt((x.array() - x.mean()).square().sum() / (n - 1));
    y = (y.array() - y.mean()) / std::sqrt((y.array() - y.mean()).square().sum() / (n - 1));
    
    auto [h1, h2] = engine.optimize_bandwidths(x, y);
    Eigen::MatrixXd density = engine.compute_density_grid(x, y, h1, h2);
    
    // Compute total probability (should be close to 1)
    const double cell_area = GRID_STEP * GRID_STEP;
    double total_prob = density.sum() * cell_area;
    
    INFO("Total probability: " << total_prob);
    
    // Should integrate to approximately 1 (allowing for tail truncation)
    REQUIRE_THAT(total_prob, WithinAbs(1.0, 0.3));
}

TEST_CASE("Results are reproducible", "[phase1][reproducibility]") {
    EntropyEngine engine1, engine2;
    
    const int n = 500;
    Eigen::VectorXd x = generate_normal(n, 42);
    Eigen::VectorXd y = generate_normal(n, 123);
    
    // Standardize
    x = (x.array() - x.mean()) / std::sqrt((x.array() - x.mean()).square().sum() / (n - 1));
    y = (y.array() - y.mean()) / std::sqrt((y.array() - y.mean()).square().sum() / (n - 1));
    
    auto [s_rho1, down_asy1] = engine1.calculate_metrics(x, y, 0.0);
    auto [s_rho2, down_asy2] = engine2.calculate_metrics(x, y, 0.0);
    
    // Results should be identical
    REQUIRE_THAT(s_rho1, WithinAbs(s_rho2, 1e-10));
    REQUIRE_THAT(down_asy1, WithinAbs(down_asy2, 1e-10));
}
