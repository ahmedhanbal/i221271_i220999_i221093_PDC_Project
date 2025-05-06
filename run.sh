#!/bin/bash

# PSAIIM: Parallel Social network Analysis using Influence-based Information Maximization
# Run script for MPI-based execution

# Default values
K=10
PROCESSES=3
MIRROR_DIR="/mirror"
LOCAL_DATA_DIR="./data"
HOST_FILE="./hosts"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -k|--k)
      K="$2"
      shift 2
      ;;
    -p|--processes)
      PROCESSES="$2"
      shift 2
      ;;
    -m|--mirror)
      MIRROR_DIR="$2"
      shift 2
      ;;
    --hostfile)
      HOST_FILE="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [-k NUM_SEEDS] [-p NUM_PROCESSES] [-m MIRROR_DIRECTORY] [--hostfile HOST_FILE]"
      echo ""
      echo "Options:"
      echo "  -k, --k NUM_SEEDS       Number of influential seeds to identify (default: 10)"
      echo "  -p, --processes NUM     Number of MPI processes to use (default: 3)"
      echo "  -m, --mirror DIR        Shared mirror directory (default: /mirror)"
      echo "  --hostfile FILE         MPI hostfile with node information (default: ./hosts)"
      echo "  -h, --help              Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help to see available options"
      exit 1
      ;;
  esac
done

echo "=== PSAIIM Algorithm Runner ==="
echo "Number of seeds (k): $K"
echo "MPI processes: $PROCESSES"
echo "Mirror directory: $MIRROR_DIR"
echo "Host file: $HOST_FILE"
echo ""

# Step 1: Convert the edge list to METIS format
echo "[1/4] Converting edge list to METIS format..."
if [ ! -f "$LOCAL_DATA_DIR/higgs.graph" ]; then
  mkdir -p "$LOCAL_DATA_DIR"
  python3 src/build_edgelist.py higgs-social_network.edgelist "$LOCAL_DATA_DIR/higgs.graph"
else
  echo "METIS graph file already exists. Skipping conversion."
fi

# Step 2: Partition the graph using METIS
echo "[2/4] Partitioning graph with METIS (${PROCESSES} parts)..."
if [ ! -f "$LOCAL_DATA_DIR/higgs.graph.part.${PROCESSES}" ]; then
  gpmetis "$LOCAL_DATA_DIR/higgs.graph" "$PROCESSES"
else
  echo "Partition file already exists. Skipping partitioning."
fi

# Step 3: Compile the PSAIIM code
echo "[3/4] Compiling PSAIIM code..."
make

# Step 4: Run the PSAIIM algorithm
echo "[4/4] Running PSAIIM with ${PROCESSES} processes..."
mpirun --hostfile "$HOST_FILE" -np "$PROCESSES" bin/psaiim_rank "$LOCAL_DATA_DIR/higgs.graph" "$LOCAL_DATA_DIR/higgs.graph.part.${PROCESSES}" "$K"

echo ""
echo "Execution complete. Results are available in results.txt" 