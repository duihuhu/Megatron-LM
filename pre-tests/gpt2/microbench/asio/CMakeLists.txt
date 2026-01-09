cmake_minimum_required(VERSION 3.10)
project(NetworkThroughputTest)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Boost
find_package(Boost REQUIRED COMPONENTS system program_options)

if(NOT Boost_FOUND)
    message(FATAL_ERROR "Boost not found. Please install Boost libraries.")
endif()

# Add executable
add_executable(network_throughput_test network_throughput_test.cpp)

# Link libraries
target_link_libraries(network_throughput_test
    ${Boost_LIBRARIES}
    pthread
)

target_include_directories(network_throughput_test PRIVATE
    ${Boost_INCLUDE_DIRS}
)

# Compiler flags for optimization
target_compile_options(network_throughput_test PRIVATE
    -O3
    -Wall
    -Wextra
)

# Installation
install(TARGETS network_throughput_test
    RUNTIME DESTINATION bin
)

