bCXX=g++
NVCC=nvcc
CXXFLAGS=-std=c++11
NVCCFLAGS=-std=c++11 -x cu

# Directories
BIN_DIR=./bin
SRC_DIR=./transformer-cpp
CUDA_SRC_DIR=./cuda/include
CUDA_KERNELS_DIR=./cuda/kernels
TEST_DIR=./tests
TEST_CUDA_DIR=./tests_cuda

# Find all cpp files in SRC_DIR
SOURCES := $(filter-out $(SRC_DIR)/main.cpp,$(wildcard $(SRC_DIR)/*.cpp))
CUDA_SOURCES := $(wildcard $(CUDA_SRC_DIR)/*.cpp) $(wildcard $(CUDA_SRC_DIR)/*.hpp) $(wildcard $(CUDA_KERNELS_DIR)/*.cu)
HEADERS := $(wildcard $(SRC_DIR)/*.hpp)
CUDA_HEADERS := $(wildcard $(CUDA_SRC_DIR)/*.hpp) $(wildcard $(CUDA_KERNELS_DIR)/*.cuh)

TESTS := $(wildcard $(TEST_DIR)/*.cpp)
TESTS_CUDA := $(wildcard $(TEST_CUDA_DIR)/*.cpp)
TEST_EXECUTABLES := $(patsubst $(TEST_DIR)/%.cpp,$(BIN_DIR)/%,$(TESTS))
TEST_CUDA_EXECUTABLES := $(patsubst $(TEST_CUDA_DIR)/%.cpp,$(BIN_DIR)/%,$(TESTS_CUDA))

# Main executable
MAIN_EXECUTABLE := $(BIN_DIR)/main

ifeq ($(USE_CUDA),1)
    CXX := $(NVCC)
    CXXFLAGS := $(NVCCFLAGS)
    CXXFLAGS += -DUSE_CUDA
    MAIN_EXECUTABLE := $(BIN_DIR)/main_cuda
    SOURCES += $(CUDA_SOURCES)
    HEADERS += $(CUDA_HEADERS)
endif

# Default target
all: $(MAIN_EXECUTABLE) run_main

# Rule to create the main executable
$(MAIN_EXECUTABLE): $(SRC_DIR)/main.cpp $(SOURCES)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $^ -lyaml-cpp -o $@
	@echo "   Compiled main executable successfully!"

# Rule to create test executables (CPU)
$(BIN_DIR)/%: $(TEST_DIR)/%.cpp $(SOURCES)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -I$(SRC_DIR) $< $(SOURCES) -lyaml-cpp -o $@
	@echo "   Compiled "$<" successfully!"

# Rule to create test executables (CUDA)
$(BIN_DIR)/%: $(TEST_CUDA_DIR)/%.cpp $(SOURCES)
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -I$(SRC_DIR) $< $(SOURCES) -lyaml-cpp -o $@
	@echo "   Compiled "$<" successfully!"

# Target to build test executables
test: $(TEST_EXECUTABLES)
ifeq ($(USE_CUDA),1)
test: $(TEST_CUDA_EXECUTABLES)
endif
	@$(MAKE) run_tests

# Target to run all tests
run_tests:
	@echo "Running tests..."
	@for test in $(TEST_EXECUTABLES); do \
		echo "==== Running $$test ===="; \
		./$$test; \
	done

ifeq ($(USE_CUDA),1)
	@for test in $(TEST_CUDA_EXECUTABLES); do \
		echo "==== Running $$test ===="; \
		./$$test; \
	done
endif

# Target to run main executable
run_main: $(MAIN_EXECUTABLE)
	@echo "Running main executable..."
	@./$(MAIN_EXECUTABLE)

# Clean Up
clean:
	@echo "Cleaning up..."
	@rm -rf $(BIN_DIR)/*

