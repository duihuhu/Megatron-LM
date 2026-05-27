cmake_minimum_required(VERSION 3.10)
project(RdmaMicrobench)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_library(IBVERBS_LIBRARY ibverbs REQUIRED)
if(NOT IBVERBS_LIBRARY)
    message(FATAL_ERROR "libibverbs not found (install libibverbs-dev)")
endif()

set(STRATEGIES_DIR "${CMAKE_SOURCE_DIR}/../../megatron/core/dist_checkpointing/strategies")

add_executable(rdma_ec_bind_bench rdma_ec_bind_bench.cpp)
target_include_directories(rdma_ec_bind_bench PRIVATE ${STRATEGIES_DIR})
target_link_libraries(rdma_ec_bind_bench ${IBVERBS_LIBRARY} pthread)
target_compile_options(rdma_ec_bind_bench PRIVATE -O3 -Wall -Wextra)

install(TARGETS rdma_ec_bind_bench RUNTIME DESTINATION bin)

