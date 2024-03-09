CXX=g++
CXXFLAGS=-std=c++11

# Directories
BIN_DIR=./bin
SRC_DIR=./transformer-cpp
TEST_DIR=./tests

# Find all cpp files in SRC_DIR, compile them to BIN_DIR
SOURCES := $(wildcard $(SRC_DIR)/*.cpp)
HEADERS := $(wildcard $(SRC_DIR)/*.hpp)
TESTS := $(wildcard $(TEST_DIR)/*.cpp)
EXECUTABLES := $(patsubst $(TEST_DIR)/%.cpp,$(BIN_DIR)/%,$(TESTS))

# Default target
all: $(EXECUTABLES)

# Rule to create executables from test source files
$(BIN_DIR)/%: $(TEST_DIR)/%.cpp $(SOURCES) $(HEADERS)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -I$(SRC_DIR) $< $(SOURCES) -o $@
	@echo "   Compiled "$<" successfully!"

# Target to run all tests
test: all
	@echo "Running tests..."
	@for test in $(EXECUTABLES); do \
		echo "==== Running $$test ===="; \
		./$$test; \
	done

# Clean Up
clean:
	@echo "Cleaning up..."
	rm -rf $(BIN_DIR)/*


