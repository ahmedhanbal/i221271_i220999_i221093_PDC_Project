CXX = mpicxx
CXXFLAGS = -std=c++17 -Wall -O3 -fopenmp
INCLUDES = -Iinclude

# For serial version
CXX_SERIAL = g++
CXXFLAGS_SERIAL = -std=c++17 -Wall -O3

# For MPI-only version (no OpenMP)
CXX_MPI_ONLY = mpicxx
CXXFLAGS_MPI_ONLY = -std=c++17 -Wall -O3

SRC_DIR = src
BUILD_DIR = build
BIN_DIR = bin

# Source files
SRCS = $(SRC_DIR)/main.cpp $(SRC_DIR)/psaiim.cpp
SRCS_SERIAL = $(SRC_DIR)/main_serial.cpp $(SRC_DIR)/psaiim_serial.cpp
SRCS_MPI_ONLY = $(SRC_DIR)/main_mpi_only.cpp $(SRC_DIR)/psaiim_mpi_only.cpp

# Object files
OBJS = $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SRCS))
OBJS_SERIAL = $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%_serial.o,$(SRCS_SERIAL))
OBJS_MPI_ONLY = $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%_mpi_only.o,$(SRCS_MPI_ONLY))

# Executables
TARGET = $(BIN_DIR)/psaiim_rank
TARGET_SERIAL = $(BIN_DIR)/psaiim_serial
TARGET_MPI_ONLY = $(BIN_DIR)/psaiim_mpi_only

# Phony targets
.PHONY: all serial mpi_only clean run run_serial run_mpi_only

# Default target
all: directories $(TARGET) $(TARGET_SERIAL) $(TARGET_MPI_ONLY)

# Serial-only target
serial: directories $(TARGET_SERIAL)

# MPI-only target
mpi_only: directories $(TARGET_MPI_ONLY)

# Create directories
directories:
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(BIN_DIR)
	@mkdir -p data
	@mkdir -p benchmark_results

# Compile source files (parallel version with OpenMP)
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Compile source files (serial version)
$(BUILD_DIR)/%_serial.o: $(SRC_DIR)/%.cpp
	$(CXX_SERIAL) $(CXXFLAGS_SERIAL) $(INCLUDES) -c $< -o $@

# Compile source files (MPI-only version)
$(BUILD_DIR)/%_mpi_only.o: $(SRC_DIR)/%.cpp
	$(CXX_MPI_ONLY) $(CXXFLAGS_MPI_ONLY) $(INCLUDES) -c $< -o $@

# Link parallel version with OpenMP
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Link serial version
$(TARGET_SERIAL): $(OBJS_SERIAL)
	$(CXX_SERIAL) $(CXXFLAGS_SERIAL) -o $@ $^

# Link MPI-only version
$(TARGET_MPI_ONLY): $(OBJS_MPI_ONLY)
	$(CXX_MPI_ONLY) $(CXXFLAGS_MPI_ONLY) -o $@ $^

# Convert edgelist to METIS format
data/higgs.graph: higgs-social_network.edgelist
	python3 $(SRC_DIR)/build_edgelist.py $< $@

# Partition the graph using METIS
data/higgs.graph.part.3: data/higgs.graph
	gpmetis $< 3

# Run the parallel program with OpenMP
run: all data/higgs.graph data/higgs.graph.part.3
	mpirun -np 3 $(TARGET) data/higgs.graph data/higgs.graph.part.3 10

# Run the serial program
run_serial: serial data/higgs.graph
	$(TARGET_SERIAL) data/higgs.graph 10

# Run the MPI-only program
run_mpi_only: mpi_only data/higgs.graph data/higgs.graph.part.3
	mpirun -np 3 $(TARGET_MPI_ONLY) data/higgs.graph data/higgs.graph.part.3 10

# Clean build files
clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)

# Clean data files too
clean_all: clean
	rm -rf data/*.graph data/*.part.* 