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
RUN_SERIAL=false     # Default: don't run serial version
RUN_MPI_ONLY=false   # Default: don't run MPI-only version
NUM_RUNS=3           # Default: run each configuration 3 times

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
    --serial)
      RUN_SERIAL=true
      shift
      ;;
    --mpi-only)
      RUN_MPI_ONLY=true
      shift
      ;;
    --all)
      RUN_SERIAL=true
      RUN_MPI_ONLY=true
      shift
      ;;
    --runs)
      NUM_RUNS="$2"
      shift 2
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
      echo "  --serial                     Also run the serial version for comparison"
      echo "  --mpi-only                   Also run the MPI-only version for comparison"
      echo "  --all                        Run all implementations (serial, MPI-only, MPI+OpenMP)"
      echo "  --runs NUM                   Number of runs for each configuration (default: 3)"
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
RESULTS_CSV="$OUTPUT_DIR/benchmark_results_$(date +%Y%m%d_%H%M%S).csv"
echo "Implementation,Processes,ExecutionTime_ms,Timestamp,RunNumber" > "$RESULTS_CSV"

echo "=== PSAIIM Algorithm Benchmark ==="
echo "Number of seeds (k): $K"
echo "Process range: $MIN_PROCS to $MAX_PROCS (step: $STEP)"
echo "Number of runs per configuration: $NUM_RUNS"
echo "Host file: $HOST_FILE"
echo "Output directory: $OUTPUT_DIR"
if [ "$RUN_SERIAL" = true ]; then
  echo "Serial version will be benchmarked"
fi
if [ "$RUN_MPI_ONLY" = true ]; then
  echo "MPI-only version will be benchmarked"
fi
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
make all

# Step 3: Run benchmarks with different configurations
echo "[3/3] Running benchmarks..."

# Run the serial version if requested
if [ "$RUN_SERIAL" = true ]; then
  echo ""
  echo "==== Running serial version ===="
  
  for ((run=1; run<=NUM_RUNS; run++)); do
    echo "Run $run/$NUM_RUNS..."
    
    # Create a timestamp
    TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
    RESULT_FILE="$OUTPUT_DIR/result_serial_run${run}_${TIMESTAMP}.txt"
    
    # Run the PSAIIM algorithm (serial)
    echo "Starting PSAIIM Serial..."
    bin/psaiim_serial "$LOCAL_DATA_DIR/higgs.graph" "$K" | tee "$RESULT_FILE"
    
    # Extract execution time (assumes it's printed in the output)
    EXEC_TIME=$(grep "execution time:" "$RESULT_FILE" | awk '{print $4}')
    if [ -n "$EXEC_TIME" ]; then
      echo "Serial,0,$EXEC_TIME,$TIMESTAMP,$run" >> "$RESULTS_CSV"
      echo "Execution time for serial run $run: $EXEC_TIME ms"
    else
      echo "Couldn't extract execution time from output"
    fi
  done
fi

# Run the MPI-only version if requested
if [ "$RUN_MPI_ONLY" = true ]; then
  # Run the MPI-only version with different numbers of processes
  for NUM_PROCS in $(seq $MIN_PROCS $STEP $MAX_PROCS); do
    echo ""
    echo "==== Running MPI-only version with $NUM_PROCS processes ===="
    
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
    
    # Run multiple times
    for ((run=1; run<=NUM_RUNS; run++)); do
      echo "Run $run/$NUM_RUNS with $NUM_PROCS processes..."
      
      # Create a timestamp
      TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
      RESULT_FILE="$OUTPUT_DIR/result_mpi_only_${NUM_PROCS}_procs_run${run}_${TIMESTAMP}.txt"
      
      # Run the PSAIIM algorithm (MPI-only)
      echo "Starting PSAIIM MPI-only with $NUM_PROCS processes..."
      mpirun --hostfile "$HOST_FILE" -np "$NUM_PROCS" bin/psaiim_mpi_only "$LOCAL_DATA_DIR/higgs.graph" "$LOCAL_DATA_DIR/higgs.graph.part.$NUM_PROCS" "$K" | tee "$RESULT_FILE"
      
      # Extract execution time (assumes it's printed in the output)
      EXEC_TIME=$(grep "execution time:" "$RESULT_FILE" | awk '{print $4}')
      if [ -n "$EXEC_TIME" ]; then
        echo "MPI-only,$NUM_PROCS,$EXEC_TIME,$TIMESTAMP,$run" >> "$RESULTS_CSV"
        echo "Execution time with $NUM_PROCS processes run $run: $EXEC_TIME ms"
      else
        echo "Couldn't extract execution time from output"
      fi
    done
  done
fi

# Run the MPI+OpenMP version (standard parallel version)
for NUM_PROCS in $(seq $MIN_PROCS $STEP $MAX_PROCS); do
  echo ""
  echo "==== Running MPI+OpenMP version with $NUM_PROCS processes ===="
  
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
  
  # Run multiple times
  for ((run=1; run<=NUM_RUNS; run++)); do
    echo "Run $run/$NUM_RUNS with $NUM_PROCS processes..."
    
    # Create a timestamp
    TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
    RESULT_FILE="$OUTPUT_DIR/result_mpi_openmp_${NUM_PROCS}_procs_run${run}_${TIMESTAMP}.txt"
    
    # Run the PSAIIM algorithm (MPI+OpenMP)
    echo "Starting PSAIIM MPI+OpenMP with $NUM_PROCS processes..."
    mpirun --hostfile "$HOST_FILE" -np "$NUM_PROCS" bin/psaiim_rank "$LOCAL_DATA_DIR/higgs.graph" "$LOCAL_DATA_DIR/higgs.graph.part.$NUM_PROCS" "$K" | tee "$RESULT_FILE"
    
    # Extract execution time (assumes it's printed in the output)
    EXEC_TIME=$(grep "execution time:" "$RESULT_FILE" | awk '{print $4}')
    if [ -n "$EXEC_TIME" ]; then
      echo "MPI+OpenMP,$NUM_PROCS,$EXEC_TIME,$TIMESTAMP,$run" >> "$RESULTS_CSV"
      echo "Execution time with $NUM_PROCS processes run $run: $EXEC_TIME ms"
    else
      echo "Couldn't extract execution time from output"
    fi
  done
done

echo ""
echo "Benchmark completed. Results saved to $RESULTS_CSV"
echo ""

# Generate simple report if we have results
echo "Generating summary..."
if [ -f "$RESULTS_CSV" ]; then
  echo "===================== BENCHMARK SUMMARY ====================="
  echo "Configuration          | Avg Time (ms) | Speedup vs Serial"
  echo "--------------------------------------------------------"
  
  # Calculate average for serial if it exists
  SERIAL_AVG=0
  if [ "$RUN_SERIAL" = true ]; then
    SERIAL_TIMES=$(grep "^Serial," "$RESULTS_CSV" | cut -d',' -f3)
    if [ -n "$SERIAL_TIMES" ]; then
      SERIAL_SUM=0
      SERIAL_COUNT=0
      for time in $SERIAL_TIMES; do
        SERIAL_SUM=$((SERIAL_SUM + time))
        SERIAL_COUNT=$((SERIAL_COUNT + 1))
      done
      SERIAL_AVG=$((SERIAL_SUM / SERIAL_COUNT))
      echo "Serial                | $SERIAL_AVG      | 1.00x"
    fi
  fi
  
  # Calculate averages for MPI-only version at different process counts
  if [ "$RUN_MPI_ONLY" = true ]; then
    for NUM_PROCS in $(seq $MIN_PROCS $STEP $MAX_PROCS); do
      PROC_TIMES=$(grep "^MPI-only,$NUM_PROCS," "$RESULTS_CSV" | cut -d',' -f3)
      if [ -n "$PROC_TIMES" ]; then
        SUM=0
        COUNT=0
        for time in $PROC_TIMES; do
          SUM=$((SUM + time))
          COUNT=$((COUNT + 1))
        done
        AVG=$((SUM / COUNT))
        
        # Calculate speedup if serial was run
        if [ "$RUN_SERIAL" = true ] && [ "$SERIAL_AVG" -gt 0 ]; then
          SPEEDUP=$(echo "scale=2; $SERIAL_AVG / $AVG" | bc)
          echo "MPI-only ($NUM_PROCS procs) | $AVG      | ${SPEEDUP}x"
        else
          echo "MPI-only ($NUM_PROCS procs) | $AVG      | N/A"
        fi
      fi
    done
  fi
  
  # Calculate averages for MPI+OpenMP version at different process counts
  for NUM_PROCS in $(seq $MIN_PROCS $STEP $MAX_PROCS); do
    PROC_TIMES=$(grep "^MPI+OpenMP,$NUM_PROCS," "$RESULTS_CSV" | cut -d',' -f3)
    if [ -n "$PROC_TIMES" ]; then
      SUM=0
      COUNT=0
      for time in $PROC_TIMES; do
        SUM=$((SUM + time))
        COUNT=$((COUNT + 1))
      done
      AVG=$((SUM / COUNT))
      
      # Calculate speedup if serial was run
      if [ "$RUN_SERIAL" = true ] && [ "$SERIAL_AVG" -gt 0 ]; then
        SPEEDUP=$(echo "scale=2; $SERIAL_AVG / $AVG" | bc)
        echo "MPI+OpenMP ($NUM_PROCS procs) | $AVG      | ${SPEEDUP}x"
      else
        echo "MPI+OpenMP ($NUM_PROCS procs) | $AVG      | N/A"
      fi
    fi
  done
  echo "=========================================================="
fi 