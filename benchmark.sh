#!/bin/bash

# PSAIIM Benchmark Script (Modified)
# Runs the PSAIIM algorithm with different numbers of processes
# Includes dependency checking

# Default values
K=10
MIN_PROCS=3
MAX_PROCS=12
STEP=3
MIRROR_DIR="/mirror"
LOCAL_DATA_DIR="./data"
HOST_FILE="./hosts"
OUTPUT_DIR="./benchmark_results"
SKIP_PARTITIONING=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -k|--k)
      K="$2"
      shift 2
      ;;
    --min-procs)
      MIN_PROCS="$2"
      shift 2
      ;;
    --max-procs)
      MAX_PROCS="$2"
      shift 2
      ;;
    --step)
      STEP="$2"
      shift 2
      ;;
    -m|--mirror)
      MIRROR_DIR="$2"
      shift 2
      ;;
    -o|--output)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --skip-partitioning)
      SKIP_PARTITIONING=true
      shift
      ;;
    -h|--help)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  -k, --k NUM_SEEDS            Number of influential seeds to identify (default: 10)"
      echo "  --min-procs MIN_PROCESSES    Minimum number of MPI processes to use (default: 3)"
      echo "  --max-procs MAX_PROCESSES    Maximum number of MPI processes to use (default: 12)"
      echo "  --step STEP                  Step size between process counts (default: 3)"
      echo "  -m, --mirror DIR             Shared mirror directory (default: /mirror)"
      echo "  -o, --output DIR             Directory to store benchmark results (default: ./benchmark_results)"
      echo "  --skip-partitioning          Skip the METIS partitioning step (manual partitioning required)"
      echo "  -h, --help                   Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help to see available options"
      exit 1
      ;;
  esac
done

# Check dependencies
function check_dependency() {
  if ! command -v "$1" &> /dev/null; then
    echo "Error: $1 is not installed or not in PATH"
    return 1
  fi
  return 0
}

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Create CSV file for benchmark results
RESULTS_CSV="$OUTPUT_DIR/benchmark_results.csv"
echo "Processes,ExecutionTime_ms,Timestamp" > "$RESULTS_CSV"

echo "=== PSAIIM Algorithm Benchmark ==="
echo "Number of seeds (k): $K"
echo "Process range: $MIN_PROCS to $MAX_PROCS (step: $STEP)"
echo "Host file: $HOST_FILE"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Check for MPI
if ! check_dependency "mpirun"; then
  echo "Error: MPI (mpirun) is required but not found. Please install an MPI implementation."
  exit 1
fi

# Check for METIS only if we're not skipping partitioning
if [ "$SKIP_PARTITIONING" = false ]; then
  if ! check_dependency "gpmetis"; then
    echo "Warning: gpmetis (METIS partitioning tool) not found."
    echo "You can either:"
    echo "  1. Install METIS (sudo apt install libmetis-dev) and run again"
    echo "  2. Run with --skip-partitioning and create partition files manually"
    echo ""
    echo "For manual partitioning, you need to create files named:"
    echo "  data/higgs.graph.part.N (for each N number of processes)"
    echo "Each file should contain one partition number (0 to N-1) per line,"
    echo "indicating the partition for each node in the graph."
    
    read -p "Do you want to continue with a simple round-robin partitioning? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
      exit 1
    fi
    
    # Simple partitioning function
    function create_simple_partition() {
      local num_parts=$1
      local graph_file=$2
      local output_file=$3
      
      # Get number of nodes from first line of graph file
      local num_nodes=$(head -n 1 "$graph_file" | awk '{print $1}')
      echo "Creating $num_parts-way partition for $num_nodes nodes..."
      
      # Create partition file with round-robin assignment
      > "$output_file"  # Empty the file
      for ((i=0; i<num_nodes; i++)); do
        echo $((i % num_parts)) >> "$output_file"
      done
      
      echo "Created simple round-robin partition file: $output_file"
    }
  fi
fi

# Step 1: Convert the edge list to METIS format (if needed)
echo "[1/3] Converting edge list to METIS format (if needed)..."
if [ ! -f "$LOCAL_DATA_DIR/higgs.graph" ]; then
  mkdir -p "$LOCAL_DATA_DIR"
  python3 src/build_edgelist.py higgs-social_network.edgelist "$LOCAL_DATA_DIR/higgs.graph"
else
  echo "METIS graph file already exists. Skipping conversion."
fi

# Step 2: Compile the PSAIIM code
echo "[2/3] Compiling PSAIIM code..."
make

# Step 3: Run benchmarks with different numbers of processes
echo "[3/3] Running benchmarks with different process counts..."
for NUM_PROCS in $(seq $MIN_PROCS $STEP $MAX_PROCS); do
  echo ""
  echo "==== Running with $NUM_PROCS processes ===="
  
  # Create partition file if needed
  if [ ! -f "$LOCAL_DATA_DIR/higgs.graph.part.$NUM_PROCS" ]; then
    echo "Creating partition with $NUM_PROCS parts..."
    
    if [ "$SKIP_PARTITIONING" = true ]; then
      echo "Partitioning is skipped. Missing partition file for $NUM_PROCS processes."
      echo "Please create the file $LOCAL_DATA_DIR/higgs.graph.part.$NUM_PROCS manually."
      continue  # Skip this process count
    elif command -v gpmetis &> /dev/null; then
      # Use METIS if available
      gpmetis "$LOCAL_DATA_DIR/higgs.graph" "$NUM_PROCS"
    else
      # Use simple round-robin partitioning
      create_simple_partition "$NUM_PROCS" "$LOCAL_DATA_DIR/higgs.graph" "$LOCAL_DATA_DIR/higgs.graph.part.$NUM_PROCS"
    fi
  else
    echo "Partition file for $NUM_PROCS processes already exists."
  fi
  
  # Create a timestamp
  TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
  RESULT_FILE="$OUTPUT_DIR/result_${NUM_PROCS}_procs_${TIMESTAMP}.txt"
  
  # Run the PSAIIM algorithm
  echo "Starting PSAIIM with $NUM_PROCS processes..."
  mpirun --hostfile "$HOST_FILE" -np "$NUM_PROCS" bin/psaiim_rank "$LOCAL_DATA_DIR/higgs.graph" "$LOCAL_DATA_DIR/higgs.graph.part.$NUM_PROCS" "$K" | tee "$RESULT_FILE"
  
  # Extract execution time (assumes it's printed in the output)
  EXEC_TIME=$(grep "execution time:" "$RESULT_FILE" | awk '{print $4}')
  if [ -n "$EXEC_TIME" ]; then
    echo "$NUM_PROCS,$EXEC_TIME,$TIMESTAMP" >> "$RESULTS_CSV"
    echo "Execution time with $NUM_PROCS processes: $EXEC_TIME ms"
  else
    echo "Couldn't extract execution time from output"
  fi
done

echo ""
echo "Benchmark completed. Results saved to $RESULTS_CSV"
echo ""

# Generate simple report if we have results
if [ -s "$RESULTS_CSV" ] && [ "$(wc -l < "$RESULTS_CSV")" -gt 1 ]; then
  echo "=== Benchmark Summary ==="
  echo "Process count | Execution time (ms)"
  echo "-----------------------------"
  sort -t, -k2,2n "$RESULTS_CSV" | tail -n +2 | awk -F, '{printf "%-12s | %s ms\n", $1, $2}'
else
  echo "No benchmark results collected."
fi 