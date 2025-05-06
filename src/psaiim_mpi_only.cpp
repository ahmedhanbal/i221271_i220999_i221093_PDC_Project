#include "../include/psaiim_mpi_only.h"

// These functions can be reused from the parallel implementation
void Discover(NodeId u, const Graph& graph, std::unordered_map<NodeId, NodeInfo>& nodeInfo, 
              std::stack<NodeId>& stack, int& index, CompId& compId) {
    // Set the depth index for u
    nodeInfo[u].index = index;
    nodeInfo[u].lowlink = index;
    index++;
    stack.push(u);
    nodeInfo[u].onStack = true;
    
    // Consider successors of u
    for (NodeId v : graph.getNeighbors(u)) {
        if (nodeInfo[v].index == -1) {
            // Successor v has not yet been visited; recurse on it
            Discover(v, graph, nodeInfo, stack, index, compId);
            nodeInfo[u].lowlink = std::min(nodeInfo[u].lowlink, nodeInfo[v].lowlink);
        } else if (nodeInfo[v].onStack) {
            // Successor v is in stack and hence in the current SCC
            nodeInfo[u].lowlink = std::min(nodeInfo[u].lowlink, nodeInfo[v].index);
        }
    }
    
    // If u is a root node, pop the stack and generate an SCC
    if (nodeInfo[u].lowlink == nodeInfo[u].index) {
        NodeId w;
        do {
            w = stack.top();
            stack.pop();
            nodeInfo[w].onStack = false;
            nodeInfo[w].compId = compId;
        } while (w != u);
        compId++;
    }
}

// Graph Partitioning
std::unordered_map<NodeId, CompId> GraphPartition(const Graph& graph) {
    int index = 0;
    CompId compId = 0;
    std::stack<NodeId> stack;
    std::unordered_map<NodeId, NodeInfo> nodeInfo;
    
    // Initialize nodeInfo for all nodes
    for (NodeId node : graph.nodes) {
        nodeInfo[node].index = -1;
        nodeInfo[node].lowlink = -1;
        nodeInfo[node].onStack = false;
        nodeInfo[node].compId = -1;
    }
    
    // Run Tarjan's algorithm for each unvisited node
    for (NodeId node : graph.nodes) {
        if (nodeInfo[node].index == -1) {
            Discover(node, graph, nodeInfo, stack, index, compId);
        }
    }
    
    // Extract component IDs
    std::unordered_map<NodeId, CompId> partition;
    for (const auto& [node, info] : nodeInfo) {
        partition[node] = info.compId;
    }
    
    return partition;
}

// Levelize Partition
std::vector<std::vector<CompId>> LevelizePartition(const std::unordered_map<NodeId, CompId>& partition, const Graph& graph) {
    // Build component graph
    std::unordered_map<CompId, std::vector<CompId>> compGraph;
    std::unordered_map<CompId, int> inDegree;
    std::unordered_set<CompId> allComps;
    
    // Find all components and initialize in-degree
    for (const auto& [node, compId] : partition) {
        allComps.insert(compId);
        if (inDegree.find(compId) == inDegree.end()) {
            inDegree[compId] = 0;
        }
    }
    
    // Build component graph and calculate in-degrees
    for (NodeId node : graph.nodes) {
        CompId srcComp = partition.at(node);
        
        for (NodeId neighbor : graph.getNeighbors(node)) {
            CompId dstComp = partition.at(neighbor);
            
            if (srcComp != dstComp) {
                // Edge between different components
                compGraph[srcComp].push_back(dstComp);
                inDegree[dstComp]++;
            }
        }
    }
    
    // Perform topological sort (Kahn's algorithm)
    std::queue<CompId> q;
    std::vector<std::vector<CompId>> levels;
    
    // Add all nodes with in-degree 0 to the queue
    for (CompId comp : allComps) {
        if (inDegree[comp] == 0) {
            q.push(comp);
        }
    }
    
    // Process by levels
    while (!q.empty()) {
        size_t levelSize = q.size();
        std::vector<CompId> currentLevel;
        
        // Process all nodes at the current level
        for (size_t i = 0; i < levelSize; i++) {
            CompId comp = q.front();
            q.pop();
            currentLevel.push_back(comp);
            
            // Reduce in-degree of all neighbors
            for (CompId neighbor : compGraph[comp]) {
                inDegree[neighbor]--;
                if (inDegree[neighbor] == 0) {
                    q.push(neighbor);
                }
            }
        }
        
        levels.push_back(currentLevel);
    }
    
    return levels;
}

// Helper for influence computation
double ComputeBehaviorWeight(NodeId u, NodeId v, const std::unordered_map<NodeId, std::unordered_set<int>>& interests,
                            const std::unordered_map<NodeId, std::unordered_map<int, int>>& actions,
                            const std::vector<double>& actionWeights) {
    // For simplified implementation, we'll use a uniform weight if no interests/actions are provided
    double weight = 1.0;
    
    // If interest data is available, compute Jaccard similarity
    if (!interests.empty()) {
        const auto& uInterests = interests.find(u) != interests.end() ? interests.at(u) : std::unordered_set<int>();
        const auto& vInterests = interests.find(v) != interests.end() ? interests.at(v) : std::unordered_set<int>();
        
        if (!uInterests.empty() || !vInterests.empty()) {
            // Compute intersection size
            size_t intersectSize = 0;
            for (int interest : uInterests) {
                if (vInterests.count(interest) > 0) {
                    intersectSize++;
                }
            }
            
            // Compute union size
            size_t unionSize = uInterests.size() + vInterests.size() - intersectSize;
            
            // Jaccard similarity
            double sim = unionSize > 0 ? static_cast<double>(intersectSize) / unionSize : 0.0;
            weight = sim;
        }
    }
    
    // If action data is available, incorporate it
    if (!actions.empty() && !actionWeights.empty()) {
        double actionTotal = 0.0;
        
        const auto& uActions = actions.find(u) != actions.end() ? actions.at(u) : std::unordered_map<int, int>();
        
        for (size_t i = 0; i < actionWeights.size(); i++) {
            int actionCount = uActions.find(i) != uActions.end() ? uActions.at(i) : 0;
            actionTotal += actionWeights[i] * actionCount;
        }
        
        // Combine with interest similarity
        weight *= (1.0 + actionTotal);
    }
    
    return weight;
}

// MPI-only Influence Power (Modified PageRank)
std::unordered_map<NodeId, double> MPIOnlyPR(const Graph& graph, 
                                           const std::vector<std::vector<CompId>>& levels, 
                                           const std::unordered_map<NodeId, CompId>& partition) {
    // Get MPI information
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (rank == 0) {
        std::cout << "Starting influence power computation..." << std::endl;
    }
    
    // Initialize parameters
    const size_t nodeCount = graph.getNodeCount();
    const double dampingFactor = 0.85;
    const double epsilon = 1e-6;
    double diff = 1.0;
    
    // Maps for component to nodes
    std::unordered_map<CompId, std::vector<NodeId>> compToNodes;
    
    // Build compToNodes mapping
    for (const auto& [node, compId] : partition) {
        compToNodes[compId].push_back(node);
    }
    
    // Initialize influence power (PR values)
    std::unordered_map<NodeId, double> PR;
    for (NodeId node : graph.nodes) {
        PR[node] = 1.0 / nodeCount;
    }
    
    // Compute influence power (PR) iteratively by component level
    int iterations = 0;
    const int maxIterations = 20;  // Reduced maximum iterations for large graphs
    
    while (diff > epsilon && iterations < maxIterations) {
        if (rank == 0 && (iterations == 0 || iterations % 5 == 0)) {
            std::cout << "Iteration " << iterations << " with diff = " << diff << std::endl;
        }
        
        double localDiff = 0.0;
        
        // Process each level in order
        for (size_t levelIdx = 0; levelIdx < levels.size(); levelIdx++) {
            const auto& level = levels[levelIdx];
            
            // Within each level, process components in this MPI rank
            for (CompId compId : level) {
                // Check if this component should be processed by this rank
                if (compId % size != rank) {
                    continue;
                }
                
                // Get nodes in this component
                const auto it = compToNodes.find(compId);
                if (it == compToNodes.end()) {
                    continue; // Skip if no nodes for this component in this rank
                }
                
                const auto& nodes = it->second;
                
                // Create a copy of PR for this iteration
                std::unordered_map<NodeId, double> newPR;
                for (NodeId node : nodes) {
                    newPR[node] = (1.0 - dampingFactor) / nodeCount;
                }
                
                // Update PR values based on incoming links
                for (NodeId node : nodes) {
                    const auto& inNeighbors = graph.getInNeighbors(node);
                    
                    for (NodeId source : inNeighbors) {
                        // Only use information we have locally
                        if (PR.find(source) == PR.end()) continue;
                        
                        double weight = graph.getEdgeWeight(source, node);
                        size_t sourceOutDegree = graph.getOutDegree(source);
                        
                        // Skip if no outgoing links
                        if (sourceOutDegree == 0) continue;
                        
                        // Add weighted contribution
                        newPR[node] += dampingFactor * PR[source] * weight / sourceOutDegree;
                    }
                }
                
                // Calculate difference and update PR
                for (NodeId node : nodes) {
                    double nodeDiff = std::abs(newPR[node] - PR[node]);
                    localDiff = std::max(localDiff, nodeDiff);
                    PR[node] = newPR[node];
                }
            }
            
            // Synchronize after each level
            MPI_Barrier(MPI_COMM_WORLD);
        }
        
        // Synchronize diff across all MPI processes
        MPI_Allreduce(&localDiff, &diff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        
        if (rank == 0 && (iterations == 0 || iterations % 5 == 0 || diff <= epsilon || iterations == maxIterations - 1)) {
            std::cout << "Iteration " << iterations << " complete. Global diff = " << diff << std::endl;
        }
        
        iterations++;
    }
    
    if (rank == 0) {
        std::cout << "Influence power computation completed after " << iterations << " iterations" << std::endl;
    }
    
    // Normalize final scores locally
    double sum = 0.0;
    for (const auto& [node, score] : PR) {
        sum += score;
    }
    
    if (sum > 0) {
        for (auto& [node, score] : PR) {
            score /= sum;
        }
    }
    
    return PR;
}

// Nodes at Distance function
std::unordered_set<NodeId> NodesAtDistance(const Graph& graph, NodeId source, int distance) {
    std::unordered_set<NodeId> result;
    std::unordered_map<NodeId, int> distances;
    std::queue<NodeId> q;
    
    // BFS to find nodes at exactly the given distance
    q.push(source);
    distances[source] = 0;
    
    while (!q.empty()) {
        NodeId current = q.front();
        q.pop();
        
        int currentDist = distances[current];
        
        if (currentDist == distance) {
            result.insert(current);
            continue;  // Don't process neighbors beyond the target distance
        }
        
        if (currentDist > distance) {
            break;  // No need to continue if we've gone beyond the distance
        }
        
        // Explore neighbors
        for (NodeId neighbor : graph.getNeighbors(current)) {
            if (distances.find(neighbor) == distances.end()) {
                distances[neighbor] = currentDist + 1;
                q.push(neighbor);
            }
        }
    }
    
    return result;
}

// Select Candidates function
std::vector<NodeId> SelectCandidates(const Graph& graph, const std::unordered_map<NodeId, double>& influencePower) {
    // Parameters for candidate selection
    const double topPercentile = 0.1;  // Top 10% by influence power
    
    // Create sorted list of nodes by influence power
    std::vector<std::pair<NodeId, double>> sortedNodes;
    sortedNodes.reserve(influencePower.size());
    for (const auto& [node, power] : influencePower) {
        sortedNodes.emplace_back(node, power);
    }
    
    // Sort by influence power (descending)
    std::sort(sortedNodes.begin(), sortedNodes.end(),
             [](const auto& a, const auto& b) { return a.second > b.second; });
    
    // Select top percentile as candidates
    size_t numCandidates = std::max(size_t(1), size_t(sortedNodes.size() * topPercentile));
    
    std::vector<NodeId> candidates;
    candidates.reserve(numCandidates);
    
    for (size_t i = 0; i < numCandidates && i < sortedNodes.size(); i++) {
        candidates.push_back(sortedNodes[i].first);
    }
    
    return candidates;
}

// Influence BFS Tree function
TreeInfo InfluenceBFSTree(const Graph& graph, NodeId root, const std::vector<NodeId>& candidates) {
    TreeInfo tree;
    tree.root = root;
    tree.nodes.insert(root);
    
    std::queue<std::pair<NodeId, int>> q;  // Node and its depth
    std::unordered_set<NodeId> visited;
    
    q.push({root, 0});
    visited.insert(root);
    
    int totalDepth = 0;
    int numNodes = 0;
    
    while (!q.empty()) {
        auto [node, depth] = q.front();
        q.pop();
        
        totalDepth += depth;
        numNodes++;
        
        for (NodeId neighbor : graph.getNeighbors(node)) {
            if (visited.find(neighbor) == visited.end()) {
                visited.insert(neighbor);
                q.push({neighbor, depth + 1});
                tree.nodes.insert(neighbor);
            }
        }
    }
    
    tree.size = numNodes;
    tree.avgDepth = numNodes > 0 ? static_cast<double>(totalDepth) / numNodes : 0.0;
    
    return tree;
}

// Select Seeds function
std::vector<NodeId> SelectSeeds(const Graph& graph, const std::vector<NodeId>& candidates, int k) {
    if (candidates.size() <= static_cast<size_t>(k)) {
        return candidates;  // If we have fewer candidates than required seeds, return all
    }
    
    std::vector<NodeId> selectedSeeds;
    std::unordered_set<NodeId> coveredNodes;
    
    // Precompute influence trees for each candidate
    std::vector<TreeInfo> trees;
    for (NodeId candidate : candidates) {
        trees.push_back(InfluenceBFSTree(graph, candidate, candidates));
    }
    
    // Sort trees by size (descending) and average depth (ascending)
    std::sort(trees.begin(), trees.end());
    
    // Greedy selection based on maximum incremental coverage
    while (selectedSeeds.size() < static_cast<size_t>(k) && !trees.empty()) {
        // Select the first tree that provides maximal new coverage
        size_t bestIndex = 0;
        size_t bestNewCoverage = 0;
        
        for (size_t i = 0; i < trees.size(); i++) {
            // Count nodes that would be newly covered
            size_t newCovered = 0;
            for (NodeId node : trees[i].nodes) {
                if (coveredNodes.find(node) == coveredNodes.end()) {
                    newCovered++;
                }
            }
            
            if (newCovered > bestNewCoverage) {
                bestNewCoverage = newCovered;
                bestIndex = i;
            }
        }
        
        // If no new coverage, just take the largest remaining tree
        if (bestNewCoverage == 0 && !trees.empty()) {
            bestIndex = 0;
        }
        
        // Add the selected seed
        selectedSeeds.push_back(trees[bestIndex].root);
        
        // Add covered nodes to the set
        for (NodeId node : trees[bestIndex].nodes) {
            coveredNodes.insert(node);
        }
        
        // Remove the selected tree
        trees.erase(trees.begin() + bestIndex);
    }
    
    return selectedSeeds;
}

// Load partitioned graph from METIS format
Graph LoadPartitionedGraph(const std::string& metisFile, const std::string& partFile, int rank, int numProcs) {
    Graph graph;
    
    // Load partition info
    std::vector<int> nodePartitions;
    
    std::ifstream partStream(partFile);
    if (!partStream) {
        std::cerr << "Error: Could not open partition file: " << partFile << std::endl;
        return graph;
    }
    
    // Read partition assignments (one per line)
    int partId;
    while (partStream >> partId) {
        nodePartitions.push_back(partId);
    }
    partStream.close();
    
    // Open the graph file
    std::ifstream graphStream(metisFile);
    if (!graphStream) {
        std::cerr << "Error: Could not open graph file: " << metisFile << std::endl;
        return graph;
    }
    
    std::string line;
    // Skip comments
    do {
        std::getline(graphStream, line);
    } while (line[0] == '%' && graphStream.good());
    
    // Parse header: <num_nodes> <num_edges> [<format> [<ncon> [<vwgt_flag> [<ewgt_flag>]]]]
    std::istringstream iss(line);
    int numNodes, numEdges;
    iss >> numNodes >> numEdges;
    
    if (rank == 0) {
        std::cout << "Loading graph with " << numNodes << " nodes and " << numEdges << " edges" << std::endl;
    }
    
    // Check if we have enough partition info
    if (nodePartitions.size() < static_cast<size_t>(numNodes)) {
        std::cerr << "Error: Not enough partition assignments for nodes" << std::endl;
        return graph;
    }
    
    // Count how many nodes are assigned to this rank
    int nodeCount = 0;
    for (int i = 0; i < numNodes; i++) {
        if (nodePartitions[i] % numProcs == rank) {
            nodeCount++;
        }
    }
    
    if (rank == 0) {
        std::cout << "Distributing nodes across " << numProcs << " processes" << std::endl;
    }
    
    // Reset file position to header
    graphStream.clear();
    graphStream.seekg(0);
    
    // Skip comments again
    do {
        std::getline(graphStream, line);
    } while (line[0] == '%' && graphStream.good());
    
    // Skip header line
    std::getline(graphStream, line);
    
    // Read adjacency list for all nodes
    for (int i = 0; i < numNodes; i++) {
        if (!std::getline(graphStream, line)) {
            break;
        }
        
        // Check if this node is assigned to this rank
        bool isOwnNode = (nodePartitions[i] % numProcs == rank);
        
        // Process this node only if it belongs to this rank
        if (isOwnNode) {
            std::istringstream lineStream(line);
            int neighbor;
            double weight;
            
            // Read all neighbors for this node
            while (lineStream >> neighbor) {
                // METIS format is 1-indexed, convert to 0-indexed
                neighbor--;
                
                // Determine edge weight
                if (lineStream.peek() == ' ') {
                    lineStream >> weight;
                } else {
                    weight = 1.0;
                }
                
                // Add edge
                graph.addEdge(i, neighbor, weight);
            }
        }
    }
    
    if (rank == 0) {
        std::cout << "Graph loading complete" << std::endl;
    }
    
    return graph;
} 