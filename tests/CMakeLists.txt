cmake_minimum_required(VERSION 3.15)

# Find Catch2 or download it
find_package(Catch2 3 QUIET)

if(NOT Catch2_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.4.0
    )
    FetchContent_MakeAvailable(Catch2)
endif()

# Test executable
add_executable(test_entropy
    test_phase1.cpp
    ../src/cpp/entropy.cpp
)

target_include_directories(test_entropy PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cpp
    ${EIGEN3_INCLUDE_DIR}
)

target_link_libraries(test_entropy PRIVATE
    Catch2::Catch2WithMain
    Eigen3::Eigen
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(test_entropy PRIVATE OpenMP::OpenMP_CXX)
endif()

# Register tests
include(CTest)
include(Catch)
catch_discover_tests(test_entropy)
