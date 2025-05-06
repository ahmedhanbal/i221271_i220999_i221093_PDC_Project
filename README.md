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
    * This phase includes the complete implementation of the PSAIIM algorithm using MPI and OpenMP for parallel processing.


# PSAIIM: Parallel Social network Analysis using Influence-based Information Maximization

This project implements the PSAIIM algorithm for identifying influential nodes in large-scale social networks using MPI and OpenMP parallelism.

## Overview

PSAIIM uses a combination of:
- Graph partitioning to divide the network into components
- Parallel influence score calculation (modified PageRank)
- Component-level parallel processing via MPI
- Node-level parallel processing via OpenMP

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

## Building the Project

```bash
make
```

This will compile the code and create the binary `bin/psaiim_rank`.

## Running the Algorithm

Use the provided run script:

```bash
./run.sh -k 10 -p 3
```

Options:
- `-k` Number of influential seeds to identify (default: 10)
- `-p` Number of MPI processes to use (default: 3)
- `-m` Shared mirror directory (default: /mirror)

### Manual Execution

You can also run the steps manually:

1. Convert the edge list to METIS format
```bash
python3 src/build_edgelist.py higgs-social_network.edgelist data/higgs.graph
```

2. Partition the graph using METIS
```bash
gpmetis data/higgs.graph 3
```

3. Run the PSAIIM algorithm
```bash
mpirun -np 3 bin/psaiim_rank data/higgs.graph data/higgs.graph.part.3 10
```

## Output

The algorithm outputs the top-k influential nodes to:
- Console output with execution time
- `results.txt` file with detailed rankings

## Algorithm Details

The PSAIIM algorithm consists of the following steps:
1. Graph partitioning into SCC/CAC components (Tarjan's algorithm)
2. Level ordering of components for DAG scheduling
3. Parallel influence power computation (modified PageRank)
4. Seed candidate selection based on influence zones
5. BFS-tree based final seed selection

## References

- Original PSAIIM research paper: "Parallel Social network Analysis using Influence-based Information Maximization"
