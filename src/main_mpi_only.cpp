#include "../include/psaiim_mpi_only.h"
#include <chrono>
#include <cstring>

int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);
    
    // Get rank and size
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Get hostname for debugging
    char hostname[256];
    gethostname(hostname, sizeof(hostname));
    
    // Check command line arguments
    if (argc != 4) {
        if (rank == 0) {
            std::cerr << "Usage: " << argv[0] << " <metis_graph_file> <metis_part_file> <k>" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }
    
    std::string metisFile = argv[1];
    std::string partFile = argv[2];
    int k = std::stoi(argv[3]);
    
    // Start timing
    auto startTime = std::chrono::high_resolution_clock::now();
    
    if (rank == 0) {
        std::cout << "PSAIIM-MPI-Only: Parallel Social network Analysis using Influence-based Information Maximization (MPI only)" << std::endl;
        std::cout << "Running with " << size << " MPI processes" << std::endl;
        std::cout << "Graph file: " << metisFile << std::endl;
        std::cout << "Partition file: " << partFile << std::endl;
        std::cout << "Number of seeds (k): " << k << std::endl;
    }
    
    // Log all hosts and their ranks
    for (int i = 0; i < size; i++) {
        if (rank == i) {
            std::cout << "Process rank " << rank << " running on host: " << hostname << std::endl;
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    // Load partitioned graph for this rank
    Graph graph = LoadPartitionedGraph(metisFile, partFile, rank, size);
    
    // Synchronize all processes
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 0) {
        std::cout << "Graph loaded. Starting PSAIIM algorithm..." << std::endl;
    }
    
    // Step 1: Graph partitioning (Tarjan's algorithm)
    auto partitionStartTime = std::chrono::high_resolution_clock::now();
    if (rank == 0) std::cout << "Step 1: Performing graph partitioning..." << std::endl;
    std::unordered_map<NodeId, CompId> partition = GraphPartition(graph);
    auto partitionEndTime = std::chrono::high_resolution_clock::now();
    auto partitionDuration = std::chrono::duration_cast<std::chrono::milliseconds>(partitionEndTime - partitionStartTime);
    if (rank == 0) std::cout << "Graph partitioning completed in " << partitionDuration.count() << " ms" << std::endl;
    
    // Step 2: Levelize partition
    auto levelizeStartTime = std::chrono::high_resolution_clock::now();
    if (rank == 0) std::cout << "Step 2: Leveling partition components..." << std::endl;
    std::vector<std::vector<CompId>> levels = LevelizePartition(partition, graph);
    auto levelizeEndTime = std::chrono::high_resolution_clock::now();
    auto levelizeDuration = std::chrono::duration_cast<std::chrono::milliseconds>(levelizeEndTime - levelizeStartTime);
    if (rank == 0) std::cout << "Partition leveling completed in " << levelizeDuration.count() << " ms" << std::endl;
    
    // Step 3: Compute influence power with MPI only
    auto influenceStartTime = std::chrono::high_resolution_clock::now();
    if (rank == 0) std::cout << "Step 3: Computing influence power..." << std::endl;
    std::unordered_map<NodeId, double> influencePower = MPIOnlyPR(graph, levels, partition);
    auto influenceEndTime = std::chrono::high_resolution_clock::now();
    auto influenceDuration = std::chrono::duration_cast<std::chrono::milliseconds>(influenceEndTime - influenceStartTime);
    if (rank == 0) std::cout << "Influence power computation completed in " << influenceDuration.count() << " ms" << std::endl;
    
    // Gather all influence powers at rank 0
    // This is simplified - in a full implementation we would need to serialize the data
    std::vector<std::pair<NodeId, double>> allInfluencePowers;
    
    // Collect local data
    for (const auto& [node, power] : influencePower) {
        allInfluencePowers.emplace_back(node, power);
    }
    
    // Only rank 0 continues with selecting candidates and seeds
    std::vector<NodeId> influentialNodes;
    
    if (rank == 0) {
        // Step 4: Select candidates
        auto candidatesStartTime = std::chrono::high_resolution_clock::now();
        std::cout << "Step 4: Selecting candidates..." << std::endl;
        std::vector<NodeId> candidates = SelectCandidates(graph, influencePower);
        auto candidatesEndTime = std::chrono::high_resolution_clock::now();
        auto candidatesDuration = std::chrono::duration_cast<std::chrono::milliseconds>(candidatesEndTime - candidatesStartTime);
        std::cout << "Selected " << candidates.size() << " candidates in " << candidatesDuration.count() << " ms" << std::endl;
        
        // Step 5: Select seeds
        auto seedsStartTime = std::chrono::high_resolution_clock::now();
        std::cout << "Step 5: Selecting seed nodes..." << std::endl;
        influentialNodes = SelectSeeds(graph, candidates, k);
        auto seedsEndTime = std::chrono::high_resolution_clock::now();
        auto seedsDuration = std::chrono::duration_cast<std::chrono::milliseconds>(seedsEndTime - seedsStartTime);
        std::cout << "Seed selection completed in " << seedsDuration.count() << " ms" << std::endl;
        
        // Calculate execution time
        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        
        // Output results
        std::cout << "\nResults:" << std::endl;
        std::cout << "Total execution time: " << duration.count() << " ms" << std::endl;
        std::cout << "Top " << k << " influential nodes:" << std::endl;
        
        for (size_t i = 0; i < influentialNodes.size(); i++) {
            NodeId node = influentialNodes[i];
            std::cout << i+1 << ". Node " << node << " (Influence Power: " 
                      << influencePower[node] << ")" << std::endl;
        }
        
        // Save results to file with timestamp for uniqueness
        std::string timestamp = std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
        std::string resultsFile = "results_mpi_only_" + timestamp + ".txt";
        std::ofstream outFile(resultsFile);
        
        if (outFile) {
            outFile << "# Top " << k << " influential nodes from PSAIIM-MPI-Only algorithm\n";
            outFile << "# MPI Processes: " << size << "\n";
            outFile << "# Rank, NodeID, InfluencePower\n";
            
            for (size_t i = 0; i < influentialNodes.size(); i++) {
                NodeId node = influentialNodes[i];
                outFile << (i+1) << ", " << node << ", " << influencePower[node] << "\n";
            }
            outFile.close();
            
            std::cout << "Results saved to " << resultsFile << std::endl;
        }
    }
    
    // Finalize MPI
    MPI_Finalize();
    return 0;
} 