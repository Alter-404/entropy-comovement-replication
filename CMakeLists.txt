cmake_minimum_required(VERSION 3.15)
project(entropy_comovement VERSION 1.0.0 LANGUAGES CXX)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build options
option(BUILD_TESTS "Build tests" ON)
option(ENABLE_OPENMP "Enable OpenMP parallelization" ON)

# Compiler flags
if(MSVC)
    add_compile_options(/W4 /O2)
else()
    add_compile_options(-Wall -Wextra -O3 -march=native)
endif()

# Find dependencies
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Try to find pybind11, or download it
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
    message(STATUS "pybind11 not found, downloading...")
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.11.1
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# Try to find Eigen3, or download it
find_package(Eigen3 3.3 QUIET NO_MODULE)
if(NOT Eigen3_FOUND)
    message(STATUS "Eigen3 not found, downloading...")
    include(FetchContent)
    FetchContent_Declare(
        Eigen3
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.0
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(Eigen3)
endif()

if(ENABLE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    endif()
endif()

# C++ shared library (Python module)
pybind11_add_module(entropy_cpp 
    src/cpp/entropy.cpp
    src/cpp/bindings.cpp
)

target_include_directories(entropy_cpp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp
    ${EIGEN3_INCLUDE_DIR}
)

target_link_libraries(entropy_cpp PRIVATE
    Eigen3::Eigen
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(entropy_cpp PRIVATE OpenMP::OpenMP_CXX)
endif()

# Set output directory
set_target_properties(entropy_cpp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src/python"
)

# Tests
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
