cmake_minimum_required(VERSION 3.10)
project(RdmaThroughputTest)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find libibverbs
find_library(IBVERBS_LIBRARY ibverbs REQUIRED)

if(NOT IBVERBS_LIBRARY)
    message(FATAL_ERROR "libibverbs not found. Please install RDMA libraries (e.g., sudo apt-get install libibverbs-dev)")
endif()

# Add executable
add_executable(rdma_throughput_test rdma_throughput_test.cpp)

# Link libraries
target_link_libraries(rdma_throughput_test
    ${IBVERBS_LIBRARY}
    pthread
)

# Compiler flags for optimization
target_compile_options(rdma_throughput_test PRIVATE
    -O3
    -Wall
    -Wextra
)

# Installation
install(TARGETS rdma_throughput_test
    RUNTIME DESTINATION bin
)

