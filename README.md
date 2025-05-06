# PDC Project Spring 2025
## Group Members
 - 22i-1271 Ahmed Ali Zahid CS-G
 - 22i-0999 Shaharyar Rizwan CS-G
 - 22i-1093 Moazzam Hafeez CS-G

## Paper : Parallel Social Behaviour Based Identifiaction of Influential Users Algorithm - PSAIIM

## Phase 1 - 20 April 2025
 - Presentation Slides
   * PSAIIM Overview
   * Parallelization Strategy

## Phase 2 - 06 May 2025
  - Implementation
    * This phase includes the complete implementation of the PSAIIM algorithm using MPI and OpenMP for parallel processing
    * Added a serial implementation for benchmarking and comparison
    * Added an MPI-only implementation (without OpenMP) for performance analysis


# PSAIIM: Parallel Social network Analysis using Influence-based Information Maximization

This project implements the PSAIIM algorithm for identifying influential nodes in large-scale social networks with three variants:

1. Parallel version using MPI and OpenMP for both inter-node and intra-node parallelism
2. MPI-only version that uses only MPI for parallelism (no OpenMP)
3. Serial version for benchmarking and comparison

The serial and MPI-only implementations follow the same algorithm logic but with different parallelization strategies.

## Prerequisites

- C++17 compiler (g++ or compatible)
- MPI implementation (OpenMPI or MPICH)
- OpenMP support
- METIS graph partitioning library
- Python 3.6+

### Installation on Ubuntu

```bash
sudo apt install mpich libmetis-dev python3 python3-pip
```

## Project Structure

- `src/` - Source code files
- `include/` - Header files
- `data/` - Generated data files (graph, partitions)
- `bin/` - Compiled binaries
- `benchmark_results/` - Performance comparison data

## Building the Project

To build all versions:
```bash
make
```

To build specific versions:
```bash
make serial        # Serial version only
make mpi_only      # MPI-only version only
```

This will compile the code and create the binaries:
- `bin/psaiim_rank` (MPI+OpenMP version)
- `bin/psaiim_mpi_only` (MPI-only version)
- `bin/psaiim_serial` (Serial version)

## Running the Algorithm

### Using the Run Script

Use the provided run script to run any version:

```bash
# Run the parallel version (MPI+OpenMP)
./run.sh -k 10 -p 3

# Run the MPI-only version
./run.sh -k 10 -p 3 --mpi-only

# Run the serial version
./run.sh -k 10 --serial
```

Options:
- `-k` Number of influential seeds to identify (default: 10)
- `-p` Number of MPI processes to use (default: 3, MPI versions only)
- `-m` Shared mirror directory (default: /mirror)
- `--mpi-only` Run the MPI-only version instead of MPI+OpenMP
- `--serial` Run the serial version

### Manual Execution

You can also run the steps manually:

1. Convert the edge list to METIS format
```bash
python3 src/build_edgelist.py higgs-social_network.edgelist data/higgs.graph
```

2. For parallel versions: Partition the graph using METIS
```bash
gpmetis data/higgs.graph 3
```

3a. Run the MPI+OpenMP version
```bash
mpirun -np 3 bin/psaiim_rank data/higgs.graph data/higgs.graph.part.3 10
```

3b. Run the MPI-only version
```bash
mpirun -np 3 bin/psaiim_mpi_only data/higgs.graph data/higgs.graph.part.3 10
```

3c. Run the serial version
```bash
bin/psaiim_serial data/higgs.graph 10
```

## Benchmarking

The project includes a benchmark script to compare the performance of all implementations:

```bash
# Run benchmarks with all implementations
./benchmark.sh --all --min-procs 3 --max-procs 12 --step 3 --runs 3

# Run benchmarks with specific implementations
./benchmark.sh --serial --mpi-only --min-procs 3 --max-procs 12

# Run only the MPI+OpenMP version
./benchmark.sh --min-procs 3 --max-procs 12
```

Options:
- `--serial` Include the serial version in benchmarks
- `--mpi-only` Include the MPI-only version in benchmarks
- `--all` Run all implementations (serial, MPI-only, MPI+OpenMP)
- `--min-procs` Minimum number of MPI processes (default: 3)
- `--max-procs` Maximum number of MPI processes (default: 12)
- `--step` Step size between process counts (default: 3)
- `--runs` Number of runs per configuration (default: 3)

Results are saved in the `benchmark_results/` directory, with a summary of execution times and speedup.

## Output

The algorithm outputs the top-k influential nodes to:
- Console output with execution time
- `results.txt` file for MPI+OpenMP version
- `results_mpi_only_[timestamp].txt` for MPI-only version
- `results_serial_[timestamp].txt` for serial version

## Algorithm Details

The PSAIIM algorithm consists of the following steps:
1. Graph partitioning into SCC/CAC components (Tarjan's algorithm)
2. Level ordering of components for DAG scheduling
3. Influence power computation (modified PageRank)
4. Seed candidate selection based on influence zones
5. BFS-tree based final seed selection

All three implementations follow the same algorithmic approach, but with different parallelization strategies:
- The MPI+OpenMP version uses MPI for inter-node communication and OpenMP for intra-node parallelism
- The MPI-only version uses only MPI for all parallelism (no OpenMP)
- The serial version runs on a single node with no parallelism

## References

- [PSAIIM research paper](PSAIIM_Paper.pdf)