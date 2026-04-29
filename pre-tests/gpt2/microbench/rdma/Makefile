# Makefile for RDMA throughput test

CXX = g++
CXXFLAGS = -std=c++14 -O3 -Wall -Wextra
LIBS = -libverbs -lpthread

TARGET = rdma_throughput_test
SOURCE = rdma_throughput_test.cpp

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SOURCE)
	@echo "Compiling $(TARGET)..."
	$(CXX) $(CXXFLAGS) $(SOURCE) -o $(TARGET) $(LIBS)
	@echo "Build completed: ./$(TARGET)"

clean:
	@echo "Cleaning..."
	rm -f $(TARGET)
	rm -rf build
	@echo "Clean completed"

help:
	@echo "Makefile for RDMA throughput test"
	@echo ""
	@echo "Targets:"
	@echo "  all (default) - Build the executable"
	@echo "  clean         - Remove build artifacts"
	@echo "  help          - Show this help message"
	@echo ""
	@echo "Usage:"
	@echo "  make          - Build the program"
	@echo "  make clean    - Clean build files"
