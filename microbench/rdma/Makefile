# Makefile for RDMA microbenchmarks

CXX = g++
CXXFLAGS = -std=c++14 -O3 -Wall -Wextra
STRATEGIES_DIR = ../../megatron/core/dist_checkpointing/strategies
INCLUDES = -I$(STRATEGIES_DIR)
LIBS = -libverbs -lpthread

.PHONY: all clean rdma_ec_bind_bench help

all: rdma_ec_bind_bench

rdma_ec_bind_bench: rdma_ec_bind_bench.cpp $(STRATEGIES_DIR)/rdma_device_utils.h
	@mkdir -p build
	@echo "Compiling rdma_ec_bind_bench..."
	$(CXX) $(CXXFLAGS) $(INCLUDES) rdma_ec_bind_bench.cpp -o build/rdma_ec_bind_bench $(LIBS)
	@echo "Build completed: build/rdma_ec_bind_bench"

clean:
	rm -rf build

help:
	@echo "Targets:"
	@echo "  make rdma_ec_bind_bench  - EC/Gemini-aligned 2-rank RDMA bench"
	@echo "  make clean"
