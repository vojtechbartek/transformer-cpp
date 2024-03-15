CXX=g++
CXXFLAGS=-std=c++11

# Directories
BIN_DIR=./bin
SRC_DIR=./transformer-cpp
TEST_DIR=./tests

# Find all cpp files in SRC_DIR
SOURCES := $(filter-out $(SRC_DIR)/main.cpp,$(wildcard $(SRC_DIR)/*.cpp))
HEADERS := $(wildcard $(SRC_DIR)/*.hpp)
TESTS := $(wildcard $(TEST_DIR)/*.cpp)
TEST_EXECUTABLES := $(patsubst $(TEST_DIR)/%.cpp,$(BIN_DIR)/%,$(TESTS))

# Main executable
MAIN_EXECUTABLE := $(BIN_DIR)/main

# Default target
all: $(MAIN_EXECUTABLE) run_main

# Rule to create the main executable
$(MAIN_EXECUTABLE): $(SRC_DIR)/main.cpp $(SOURCES)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $^ -lyaml-cpp -o $@
	@echo "   Compiled main executable successfully!"

# Rule to create test executables
$(BIN_DIR)/%: $(TEST_DIR)/%.cpp $(SOURCES)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -I$(SRC_DIR) $< $(SOURCES) -lyaml-cpp -o $@
	@echo "   Compiled "$<" successfully!"

# Target to build test executables
test: $(TEST_EXECUTABLES) run_tests

# Target to run all tests
run_tests: test
	@echo "Running tests..."
	@for test in $(TEST_EXECUTABLES); do \
		echo "==== Running $$test ===="; \
			./$$test; \
    done

# Target to run main executable
run_main: $(MAIN_EXECUTABLE)
	@echo "Running main executable..."
	@./$(MAIN_EXECUTABLE)

# Clean Up
clean:
	@echo "Cleaning up..."
	rm -rf $(BIN_DIR)/*
