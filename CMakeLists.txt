cmake_minimum_required(VERSION 3.9)

project(innokenty)

enable_language(CUDA)

set(CMAKE_BUILD_TYPE Release)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/xsimd)

set(CMAKE_CXX_FLAGS_RELEASE "-g -O3")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

add_executable(latency "src/latency.cpp" "src/topology.cpp")
target_include_directories(latency PRIVATE include)
target_link_libraries(latency PRIVATE MPI::MPI_CXX)

add_executable(throughput "src/throughput.cpp" "src/topology.cpp")
target_include_directories(throughput PRIVATE include)
target_link_libraries(throughput PRIVATE MPI::MPI_CXX)

add_executable(pi_mpi_1 "src/pi.cpp")
target_compile_definitions(pi_mpi_1 PRIVATE VERSION=0)
target_include_directories(pi_mpi_1 PRIVATE include)
target_link_libraries(pi_mpi_1 PRIVATE MPI::MPI_CXX)

add_executable(pi_mpi_openmp_2 "src/pi.cpp")
target_compile_definitions(pi_mpi_openmp_2 PRIVATE VERSION=1)
target_include_directories(pi_mpi_openmp_2 PRIVATE include)
target_link_libraries(pi_mpi_openmp_2 PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX)

add_executable(pi_mpi_openmp_xsimd_3 "src/pi.cpp")
target_compile_definitions(pi_mpi_openmp_xsimd_3 PRIVATE VERSION=2)
target_compile_options(pi_mpi_openmp_xsimd_3 PRIVATE -mavx2)
target_include_directories(pi_mpi_openmp_xsimd_3 PRIVATE include)
target_link_libraries(pi_mpi_openmp_xsimd_3 PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX xsimd)

add_executable(pi_mpi_openmp_xsimd_3_amdahl "src/pi.cpp")
target_compile_definitions(pi_mpi_openmp_xsimd_3_amdahl PRIVATE VERSION=2 AMDAHL)
target_compile_options(pi_mpi_openmp_xsimd_3_amdahl PRIVATE -mavx2) # -mavx512f -mavx512dq)
target_include_directories(pi_mpi_openmp_xsimd_3_amdahl PRIVATE include)
target_link_libraries(pi_mpi_openmp_xsimd_3_amdahl PRIVATE MPI::MPI_CXX xsimd) # OpenMP::OpenMP_CXX xsimd)

add_executable(pi_gpu "src/pi_gpu.cu")
target_include_directories(pi_gpu PRIVATE include)
set_property(TARGET pi_gpu PROPERTY CUDA_ARCHITECTURES 61)

