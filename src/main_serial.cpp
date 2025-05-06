#include "../include/psaiim_serial.h"
#include <chrono>
#include <cstring>

int main(int argc, char** argv) {
    // Check command line arguments
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <metis_graph_file> <k>" << std::endl;
        return 1;
    }
    
    std::string metisFile = argv[1];
    int k = std::stoi(argv[2]);
    
    // Start timing
    auto startTime = std::chrono::high_resolution_clock::now();
    
    std::cout << "PSAIIM-Serial: Serial Social network Analysis using Influence-based Information Maximization" << std::endl;
    std::cout << "Graph file: " << metisFile << std::endl;
    std::cout << "Number of seeds (k): " << k << std::endl;
    
    // Load graph
    std::cout << "Loading graph..." << std::endl;
    Graph graph = LoadGraph(metisFile);
    
    std::cout << "Graph loaded with " << graph.getNodeCount() << " nodes. Starting PSAIIM algorithm..." << std::endl;
    
    // Step 1: Graph partitioning (Tarjan's algorithm)
    auto partitionStartTime = std::chrono::high_resolution_clock::now();
    std::cout << "Step 1: Performing graph partitioning..." << std::endl;
    std::unordered_map<NodeId, CompId> partition = GraphPartition(graph);
    auto partitionEndTime = std::chrono::high_resolution_clock::now();
    auto partitionDuration = std::chrono::duration_cast<std::chrono::milliseconds>(partitionEndTime - partitionStartTime);
    std::cout << "Graph partitioning completed in " << partitionDuration.count() << " ms" << std::endl;
    
    // Step 2: Levelize partition
    auto levelizeStartTime = std::chrono::high_resolution_clock::now();
    std::cout << "Step 2: Leveling partition components..." << std::endl;
    std::vector<std::vector<CompId>> levels = LevelizePartition(partition, graph);
    auto levelizeEndTime = std::chrono::high_resolution_clock::now();
    auto levelizeDuration = std::chrono::duration_cast<std::chrono::milliseconds>(levelizeEndTime - levelizeStartTime);
    std::cout << "Partition leveling completed in " << levelizeDuration.count() << " ms" << std::endl;
    
    // Step 3: Compute influence power (serially)
    auto influenceStartTime = std::chrono::high_resolution_clock::now();
    std::cout << "Step 3: Computing influence power..." << std::endl;
    std::unordered_map<NodeId, double> influencePower = SerialPR(graph, levels, partition);
    auto influenceEndTime = std::chrono::high_resolution_clock::now();
    auto influenceDuration = std::chrono::duration_cast<std::chrono::milliseconds>(influenceEndTime - influenceStartTime);
    std::cout << "Influence power computation completed in " << influenceDuration.count() << " ms" << std::endl;
    
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
    std::vector<NodeId> influentialNodes = SelectSeeds(graph, candidates, k);
    auto seedsEndTime = std::chrono::high_resolution_clock::now();
    auto seedsDuration = std::chrono::duration_cast<std::chrono::milliseconds>(seedsEndTime - seedsStartTime);
    std::cout << "Seed selection completed in " << seedsDuration.count() << " ms" << std::endl;
    
    // Calculate total execution time
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
    std::string resultsFile = "results_serial_" + timestamp + ".txt";
    std::ofstream outFile(resultsFile);
    
    if (outFile) {
        outFile << "# Top " << k << " influential nodes from PSAIIM-Serial algorithm\n";
        outFile << "# Rank, NodeID, InfluencePower\n";
        
        for (size_t i = 0; i < influentialNodes.size(); i++) {
            NodeId node = influentialNodes[i];
            outFile << (i+1) << ", " << node << ", " << influencePower[node] << "\n";
        }
        outFile.close();
        
        std::cout << "Results saved to " << resultsFile << std::endl;
    }
    
    return 0;
} 