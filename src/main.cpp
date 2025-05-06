#include "../include/psaiim.h"
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
        std::cout << "PSAIIM: Parallel Social network Analysis using Influence-based Information Maximization" << std::endl;
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
    std::unordered_map<NodeId, CompId> partition = GraphPartition(graph);
    
    // Synchronize data across processes (simplified for demonstration)
    // In a full implementation, we would exchange partition information among processes
    
    // Step 2: Levelize partition
    std::vector<std::vector<CompId>> levels = LevelizePartition(partition, graph);
    
    // Step 3: Compute influence power in parallel
    std::unordered_map<NodeId, double> influencePower = ParallelPR(graph, levels, partition);
    
    // Gather all influence powers at rank 0
    // (In a real implementation, we would use MPI_Gather or similar)
    std::vector<std::pair<NodeId, double>> allInfluencePowers;
    
    if (rank == 0) {
        // Collect from all processes (simplified)
        allInfluencePowers.reserve(graph.getNodeCount());
        for (const auto& [node, power] : influencePower) {
            allInfluencePowers.emplace_back(node, power);
        }
        
        // In real implementation, would receive data from other ranks here
    }
    
    // Only rank 0 continues with selecting candidates and seeds
    std::vector<NodeId> influentialNodes;
    
    if (rank == 0) {
        // Step 4: Select candidates
        std::vector<NodeId> candidates = SelectCandidates(graph, influencePower);
        
        std::cout << "Selected " << candidates.size() << " candidates" << std::endl;
        
        // Step 5: Select seeds
        influentialNodes = SelectSeeds(graph, candidates, k);
        
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
        
        // Save results to file
        std::ofstream outFile("results.txt");
        if (outFile) {
            outFile << "# Top " << k << " influential nodes from PSAIIM algorithm\n";
            outFile << "# Rank, NodeID, InfluencePower\n";
            
            for (size_t i = 0; i < influentialNodes.size(); i++) {
                NodeId node = influentialNodes[i];
                outFile << (i+1) << ", " << node << ", " << influencePower[node] << "\n";
            }
            outFile.close();
            
            std::cout << "Results saved to results.txt" << std::endl;
        }
    }
    
    // Finalize MPI
    MPI_Finalize();
    return 0;
} 