CXX = mpicxx
CXXFLAGS = -std=c++17 -Wall -O3 -fopenmp
INCLUDES = -Iinclude

SRC_DIR = src
BUILD_DIR = build
BIN_DIR = bin

# Source files
SRCS = $(SRC_DIR)/main.cpp $(SRC_DIR)/psaiim.cpp

# Object files
OBJS = $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SRCS))

# Executable
TARGET = $(BIN_DIR)/psaiim_rank

# Phony targets
.PHONY: all clean run

# Default target
all: directories $(TARGET)

# Create directories
directories:
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(BIN_DIR)
	@mkdir -p data

# Compile source files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Link object files
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Convert edgelist to METIS format
data/higgs.graph: higgs-social_network.edgelist
	python3 $(SRC_DIR)/build_edgelist.py $< $@

# Partition the graph using METIS
data/higgs.graph.part.3: data/higgs.graph
	gpmetis $< 3

# Run the program
run: all data/higgs.graph data/higgs.graph.part.3
	mpirun -np 3 $(TARGET) data/higgs.graph data/higgs.graph.part.3 10

# Clean build files
clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR) data/*.graph data/*.part.* 