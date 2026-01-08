# Makefile for network throughput test
# Alternative to CMake for quick compilation

CXX = g++
CXXFLAGS = -std=c++14 -O3 -Wall -Wextra
LDFLAGS = -lboost_system -lboost_program_options -lpthread

TARGET = network_throughput_test
SOURCE = network_throughput_test.cpp

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SOURCE)
	@echo "Compiling $(TARGET)..."
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCE) $(LDFLAGS)
	@echo "Build completed: ./$(TARGET)"

clean:
	@echo "Cleaning..."
	rm -f $(TARGET)
	rm -rf build
	@echo "Clean completed"

help:
	@echo "Makefile for network throughput test"
	@echo ""
	@echo "Targets:"
	@echo "  all (default) - Build the executable"
	@echo "  clean         - Remove build artifacts"
	@echo "  help          - Show this help message"
	@echo ""
	@echo "Usage:"
	@echo "  make          - Build the program"
	@echo "  make clean    - Clean build files"

