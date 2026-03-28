# Makefile for ML From Scratch Project

# Compiler and Flags
CXX = g++
CXXFLAGS = -O3 -Wall -fPIC -shared
SRC = cpp/vector_math.cpp
TARGET = cpp/libvector.so

# Default target
all: $(TARGET)

# Compile C++ shared library
$(TARGET): $(SRC)
	@echo "Compiling C++ backend..."
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)
	@echo "Compilation successful: $(TARGET)"

# Run all tests
test: all
	@echo "Running all tests..."
	@python3 test/test_math_vectors.py
	@python3 test/test_matrix_math.py
	@python3 test/validate_vectorization.py
	@python3 test/test_logistic_reg.py
	@python3 test/test_knn.py
	@echo "All tests passed!"

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -f $(TARGET)
	@echo "Done."

.PHONY: all test clean
