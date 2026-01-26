/*
    PyBind11 bindings for the Entropy Engine
*/

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "entropy.h"

namespace py = pybind11;

PYBIND11_MODULE(entropy_cpp, m) {
    m.doc() = "High-performance C++ engine for Asymmetry Entropy calculation";
    
    py::class_<entropy::EntropyEngine>(m, "EntropyEngine")
        .def(py::init<>())
        .def("calculate_metrics", &entropy::EntropyEngine::calculate_metrics,
             py::arg("x"), py::arg("y"), py::arg("c") = 0.0,
             R"pbdoc(
                Computes S_rho and DOWN_ASY given standardized returns.
                
                Parameters
                ----------
                x : np.ndarray
                    Standardized returns of asset/portfolio (mean=0, std=1)
                y : np.ndarray
                    Standardized returns of market (mean=0, std=1)
                c : float, optional
                    Threshold for quadrant definition (default: 0.0)
                
                Returns
                -------
                tuple of (float, float)
                    (S_rho, DOWN_ASY) where:
                    - S_rho: Entropy-based comovement measure
                    - DOWN_ASY: Signed asymmetry measure
                
                References
                ----------
                Jiang, Wu, and Zhou (2018) - "Asymmetry in Stock Comovements: 
                An Entropy Approach" JFQA, pp. 469-478
             )pbdoc")
        .def("compute_loo_likelihood", &entropy::EntropyEngine::compute_loo_likelihood,
             py::arg("x"), py::arg("y"), py::arg("h1"), py::arg("h2"),
             "Compute Leave-One-Out log-likelihood for bandwidth selection")
        .def("optimize_bandwidths", &entropy::EntropyEngine::optimize_bandwidths,
             py::arg("x"), py::arg("y"),
             "Optimize bandwidths using Likelihood Cross-Validation")
        .def("compute_density_grid", &entropy::EntropyEngine::compute_density_grid,
             py::arg("x"), py::arg("y"), py::arg("h1"), py::arg("h2"),
             "Compute bivariate density on 100x100 grid");
    
    m.attr("GRID_MIN") = entropy::GRID_MIN;
    m.attr("GRID_MAX") = entropy::GRID_MAX;
    m.attr("GRID_SIZE") = entropy::GRID_SIZE;
}
