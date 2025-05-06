#include "../include/psaiim_serial.h"

// These functions can be reused from the parallel implementation with minimal changes
void Discover(NodeId u, const Graph& graph, std::unordered_map<NodeId, NodeInfo>& nodeInfo, 
              std::stack<NodeId>& stack, int& index, CompId& compId) {
    // Set the depth index for u
    try {
        nodeInfo[u].index = index;
        nodeInfo[u].lowlink = index;
        index++;
        stack.push(u);
        nodeInfo[u].onStack = true;
        
        // Consider successors of u
        const std::vector<NodeId>& neighbors = graph.getNeighbors(u);
        for (NodeId v : neighbors) {
            if (nodeInfo.find(v) == nodeInfo.end()) {
                std::cerr << "Error: Node " << v << " not found in nodeInfo map" << std::endl;
                continue;
            }
            
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
    } catch (const std::exception& e) {
        std::cerr << "Exception in Discover: " << e.what() << " processing node " << u << std::endl;
        throw; // Rethrow to be caught by the caller
    }
}

// Iterative Tarjan's algorithm helper
void DiscoverIterative(NodeId startNode, const Graph& graph, std::unordered_map<NodeId, NodeInfo>& nodeInfo, 
                      std::stack<NodeId>& dfsStack, int& index, CompId& compId) {
    
    // Setup a non-recursive implementation of Tarjan's algorithm
    struct TaskState {
        NodeId node;
        size_t nextNeighbor;
        bool processingNeighbors;
        
        TaskState(NodeId n) : node(n), nextNeighbor(0), processingNeighbors(false) {}
    };
    
    std::stack<TaskState> taskStack;
    taskStack.push(TaskState(startNode));
    
    while (!taskStack.empty()) {
        TaskState& state = taskStack.top();
        NodeId u = state.node;
        
        if (!state.processingNeighbors) {
            // Initialize node
            nodeInfo[u].index = index;
            nodeInfo[u].lowlink = index;
            index++;
            dfsStack.push(u);
            nodeInfo[u].onStack = true;
            state.processingNeighbors = true;
        }
        
        const std::vector<NodeId>& neighbors = graph.getNeighbors(u);
        
        // Continue processing neighbors
        bool allNeighborsProcessed = true;
        for (; state.nextNeighbor < neighbors.size(); state.nextNeighbor++) {
            NodeId v = neighbors[state.nextNeighbor];
            
            if (nodeInfo.find(v) == nodeInfo.end()) {
                std::cerr << "Warning: Node " << v << " not found in nodeInfo map" << std::endl;
                continue;
            }
            
            if (nodeInfo[v].index == -1) {
                // Found an unvisited neighbor, push to stack and process it first
                taskStack.push(TaskState(v));
                allNeighborsProcessed = false;
                state.nextNeighbor++; // Move to next neighbor when we return
                break;
            } else if (nodeInfo[v].onStack) {
                // Update lowlink for nodes already on the stack
                nodeInfo[u].lowlink = std::min(nodeInfo[u].lowlink, nodeInfo[v].index);
            }
        }
        
        if (allNeighborsProcessed) {
            // Check if we've found an SCC
            if (nodeInfo[u].lowlink == nodeInfo[u].index) {
                NodeId w;
                do {
                    w = dfsStack.top();
                    dfsStack.pop();
                    nodeInfo[w].onStack = false;
                    nodeInfo[w].compId = compId;
                } while (w != u);
                compId++;
            }
            
            // Update parent's lowlink if applicable
            NodeId currentNode = u;
            taskStack.pop();
            
            if (!taskStack.empty()) {
                TaskState& parentState = taskStack.top();
                NodeId parentNode = parentState.node;
                
                // Update parent's lowlink
                nodeInfo[parentNode].lowlink = std::min(nodeInfo[parentNode].lowlink, nodeInfo[currentNode].lowlink);
            }
        }
    }
}

// Graph Partitioning with iterative approach
std::unordered_map<NodeId, CompId> GraphPartition(const Graph& graph) {
    int index = 0;
    CompId compId = 0;
    std::stack<NodeId> stack; // This stack is used for the SCC detection
    std::unordered_map<NodeId, NodeInfo> nodeInfo;
    
    std::cout << "Starting graph partitioning with " << graph.nodes.size() << " nodes..." << std::endl;
    
    // Initialize nodeInfo for all nodes
    int initCount = 0;
    for (NodeId node : graph.nodes) {
        nodeInfo[node].index = -1;
        nodeInfo[node].lowlink = -1;
        nodeInfo[node].onStack = false;
        nodeInfo[node].compId = -1;
        initCount++;
        if (initCount % 100000 == 0) {
            std::cout << "  Initialized " << initCount << " nodes..." << std::endl;
        }
    }
    
    std::cout << "Initialization complete. Starting iterative Tarjan's algorithm..." << std::endl;
    
    // Run iterative Tarjan's algorithm for each unvisited node
    int visitedCount = 0;
    int processedNodes = 0;
    
    for (NodeId node : graph.nodes) {
        if (nodeInfo[node].index == -1) {
            try {
                if (visitedCount % 1000 == 0 || visitedCount < 10) {
                    std::cout << "Processing component starting with node " << node << " (" << visitedCount << ")" << std::endl;
                }
                
                DiscoverIterative(node, graph, nodeInfo, stack, index, compId);
                
                visitedCount++;
                processedNodes++;
                
                if (processedNodes >= 10000) {
                    std::cout << "  Processed " << visitedCount << " components..." << std::endl;
                    processedNodes = 0;
                }
            } catch (const std::exception& e) {
                std::cerr << "Exception in DiscoverIterative: " << e.what() << " for node " << node << std::endl;
                // Continue to next node
            }
        }
    }
    
    std::cout << "Tarjan's algorithm complete. Found " << compId << " components." << std::endl;
    
    // Extract component IDs
    std::unordered_map<NodeId, CompId> partition;
    for (const auto& [node, info] : nodeInfo) {
        partition[node] = info.compId;
    }
    
    std::cout << "Partition extraction complete." << std::endl;
    
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

// Serial Influence Power (Modified PageRank)
std::unordered_map<NodeId, double> SerialPR(const Graph& graph, 
                                          const std::vector<std::vector<CompId>>& levels, 
                                          const std::unordered_map<NodeId, CompId>& partition) {
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
    const int maxIterations = 50;  // Maximum number of iterations
    
    while (diff > epsilon && iterations < maxIterations) {
        diff = 0.0;
        
        // Process each level in order
        for (const auto& level : levels) {
            // Within each level, process components independently
            for (CompId compId : level) {
                // Get nodes in this component
                const auto& nodes = compToNodes[compId];
                
                // Create a copy of PR for this iteration
                std::unordered_map<NodeId, double> newPR;
                for (NodeId node : nodes) {
                    newPR[node] = (1.0 - dampingFactor) / nodeCount;
                }
                
                // Update PR values based on incoming links
                for (NodeId node : nodes) {
                    const auto& inNeighbors = graph.getInNeighbors(node);
                    
                    for (NodeId source : inNeighbors) {
                        // Only consider nodes from this component or from earlier levels
                        if (partition.at(source) == compId || 
                            (std::find_if(levels.begin(), levels.end(), 
                                         [&](const std::vector<CompId>& l) { 
                                             return std::find(l.begin(), l.end(), partition.at(source)) != l.end(); 
                                         }) < 
                             std::find_if(levels.begin(), levels.end(), 
                                         [&](const std::vector<CompId>& l) { 
                                             return std::find(l.begin(), l.end(), compId) != l.end(); 
                                         }))) {
                            
                            double weight = graph.getEdgeWeight(source, node);
                            size_t sourceOutDegree = graph.getOutDegree(source);
                            
                            // Skip if no outgoing links
                            if (sourceOutDegree == 0) continue;
                            
                            // Add weighted contribution
                            newPR[node] += dampingFactor * PR[source] * weight / sourceOutDegree;
                        }
                    }
                }
                
                // Calculate difference and update PR
                for (NodeId node : nodes) {
                    double nodeDiff = std::abs(newPR[node] - PR[node]);
                    diff = std::max(diff, nodeDiff);
                    PR[node] = newPR[node];
                }
            }
        }
        
        iterations++;
    }
    
    // Normalize final scores
    double sum = 0.0;
    for (const auto& [node, score] : PR) {
        sum += score;
    }
    
    for (auto& [node, score] : PR) {
        score /= sum;
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

// Load graph from METIS format
Graph LoadGraph(const std::string& metisFile) {
    Graph graph;
    
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
    
    // Read edges
    for (int i = 0; i < numNodes; i++) {
        if (std::getline(graphStream, line)) {
            std::istringstream lineStream(line);
            int neighbor;
            double weight;
            
            while (lineStream >> neighbor) {
                // METIS format is 1-indexed, convert to 0-indexed
                neighbor--;
                
                // Check if there's a weight
                if (lineStream.peek() == ' ') {
                    lineStream >> weight;
                } else {
                    weight = 1.0;
                }
                
                // Add edge from current node to neighbor
                graph.addEdge(i, neighbor, weight);
            }
        }
    }
    
    return graph;
} 