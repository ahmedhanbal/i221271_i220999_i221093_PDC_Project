#include "../include/psaiim.h"

// Algorithm 2: Discover function for graph partitioning (Tarjan's algorithm)
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

// Algorithm 2-4: Graph Partitioning
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

// Algorithm 2.3: Levelize Partition
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

// ComputeBehaviorWeight function
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

// Algorithm 5: Parallel Influence Power (Modified PageRank)
std::unordered_map<NodeId, double> ParallelPR(const Graph& graph, 
                                              const std::vector<std::vector<CompId>>& levels, 
                                              const std::unordered_map<NodeId, CompId>& partition) {
    // Initialize parameters
    const size_t nodeCount = graph.getNodeCount();
    const double dampingFactor = 0.85;
    const double epsilon = 1e-6;
    double diff = 1.0;
    
    // Maps for component to nodes and node to component
    std::unordered_map<CompId, std::vector<NodeId>> compToNodes;
    
    // Build compToNodes mapping
    for (const auto& [node, compId] : partition) {
        compToNodes[compId].push_back(node);
    }
    
    // Initialize influence power
    std::unordered_map<NodeId, double> IP_old, IP_new;
    for (NodeId node : graph.nodes) {
        IP_old[node] = 1.0 / nodeCount;
    }
    
    // Iterative computation until convergence
    while (diff > epsilon) {
        // Process each level in sequence (MPI ranks assigned to levels)
        for (const auto& level : levels) {
            // Process components in the current level in parallel
            #pragma omp parallel for
            for (size_t i = 0; i < level.size(); i++) {
                CompId compId = level[i];
                
                // Process nodes in the component
                for (NodeId u : compToNodes[compId]) {
                    double sum = 0.0;
                    
                    // Sum contributions from incoming neighbors
                    for (NodeId v : graph.getInNeighbors(u)) {
                        size_t outDegree = graph.getOutDegree(v);
                        if (outDegree > 0) {
                            sum += graph.getEdgeWeight(v, u) * IP_old[v] / outDegree;
                        }
                    }
                    
                    // Update influence power
                    IP_new[u] = dampingFactor * sum + (1.0 - dampingFactor) / nodeCount;
                }
            }
        }
        
        // Calculate difference for convergence check
        diff = 0.0;
        for (NodeId node : graph.nodes) {
            diff += std::abs(IP_new[node] - IP_old[node]);
            IP_old[node] = IP_new[node];
        }
    }
    
    return IP_old;
}

// Get nodes at a specific distance from a source node
std::unordered_set<NodeId> NodesAtDistance(const Graph& graph, NodeId source, int distance) {
    std::unordered_set<NodeId> result;
    std::unordered_map<NodeId, int> nodeDistance;
    std::queue<NodeId> q;
    
    // Initialize BFS
    q.push(source);
    nodeDistance[source] = 0;
    
    // BFS to find nodes at the target distance
    while (!q.empty()) {
        NodeId current = q.front();
        q.pop();
        
        int currentDist = nodeDistance[current];
        
        if (currentDist == distance) {
            result.insert(current);
            continue;
        }
        
        if (currentDist > distance) {
            continue;
        }
        
        // Explore neighbors
        for (NodeId neighbor : graph.getNeighbors(current)) {
            if (nodeDistance.find(neighbor) == nodeDistance.end()) {
                nodeDistance[neighbor] = currentDist + 1;
                q.push(neighbor);
            }
        }
    }
    
    return result;
}

// Algorithm 6: Select Candidates
std::vector<NodeId> SelectCandidates(const Graph& graph, const std::unordered_map<NodeId, double>& influencePower) {
    // Convert unordered_set to vector for easier parallelization
    std::vector<NodeId> nodesList(graph.nodes.begin(), graph.nodes.end());
    std::vector<NodeId> candidates;
    
    // Process nodes in parallel
    #pragma omp parallel
    {
        std::vector<NodeId> localCandidates;
        
        #pragma omp for
        for (size_t i = 0; i < nodesList.size(); i++) {
            NodeId u = nodesList[i];
            int L = 1;
            double prev_avg = -1.0;
            
            while (true) {
                std::unordered_set<NodeId> zone = NodesAtDistance(graph, u, L);
                if (zone.empty()) {
                    break;
                }
                
                // Calculate average influence in the zone
                double sum = 0.0;
                for (NodeId x : zone) {
                    sum += influencePower.at(x);
                }
                double avg = sum / zone.size();
                
                if (avg < prev_avg) {
                    break;
                }
                
                prev_avg = avg;
                L++;
            }
            
            // Check if node's influence exceeds the average in its expanding zone
            if (influencePower.at(u) > prev_avg) {
                localCandidates.push_back(u);
            }
        }
        
        // Merge local candidates to global list
        #pragma omp critical
        {
            candidates.insert(candidates.end(), localCandidates.begin(), localCandidates.end());
        }
    }
    
    return candidates;
}

// Create influence BFS tree for a candidate
TreeInfo InfluenceBFSTree(const Graph& graph, NodeId root, const std::vector<NodeId>& candidates) {
    std::unordered_set<NodeId> visited = {root};
    std::unordered_map<NodeId, NodeId> parent = {{root, -1}};
    std::unordered_map<NodeId, int> depth = {{root, 0}};
    std::queue<NodeId> q;
    q.push(root);
    
    // Convert candidates to set for faster lookup
    std::unordered_set<NodeId> candidateSet(candidates.begin(), candidates.end());
    
    // BFS traversal
    while (!q.empty()) {
        NodeId v = q.front();
        q.pop();
        
        for (NodeId w : graph.getNeighbors(v)) {
            if (visited.find(w) == visited.end() && candidateSet.find(w) != candidateSet.end()) {
                visited.insert(w);
                parent[w] = v;
                depth[w] = depth[v] + 1;
                q.push(w);
            }
        }
    }
    
    // Calculate average depth
    double totalDepth = 0.0;
    for (const auto& [node, nodeDepth] : depth) {
        totalDepth += nodeDepth;
    }
    double avgDepth = visited.size() > 0 ? totalDepth / visited.size() : 0.0;
    
    // Create and return TreeInfo
    TreeInfo tree;
    tree.root = root;
    tree.size = visited.size();
    tree.avgDepth = avgDepth;
    tree.nodes = std::move(visited);
    
    return tree;
}

// Algorithm 7: Select Seeds
std::vector<NodeId> SelectSeeds(const Graph& graph, const std::vector<NodeId>& candidates, int k) {
    std::vector<TreeInfo> treeInfos;
    
    // Build influence trees for each candidate in parallel
    #pragma omp parallel
    {
        std::vector<TreeInfo> localTreeInfos;
        
        #pragma omp for
        for (size_t i = 0; i < candidates.size(); i++) {
            NodeId u = candidates[i];
            TreeInfo tree = InfluenceBFSTree(graph, u, candidates);
            localTreeInfos.push_back(tree);
        }
        
        // Merge local trees to global list
        #pragma omp critical
        {
            treeInfos.insert(treeInfos.end(), localTreeInfos.begin(), localTreeInfos.end());
        }
    }
    
    // Sort TreeInfos by size (desc) and avgDepth (asc)
    std::sort(treeInfos.begin(), treeInfos.end());
    
    // Select seeds
    std::vector<NodeId> seeds;
    std::unordered_set<NodeId> covered;
    
    for (const TreeInfo& tree : treeInfos) {
        // Check if this tree's nodes overlap with already covered nodes
        bool overlap = false;
        for (NodeId node : tree.nodes) {
            if (covered.find(node) != covered.end()) {
                overlap = true;
                break;
            }
        }
        
        if (!overlap) {
            seeds.push_back(tree.root);
            for (NodeId node : tree.nodes) {
                covered.insert(node);
            }
        }
        
        if (static_cast<int>(seeds.size()) == k) {
            break;
        }
    }
    
    return seeds;
}

// Load graph for a specific MPI rank
Graph LoadPartitionedGraph(const std::string& metisFile, const std::string& partFile, int rank, int numProcs) {
    Graph graph;
    
    // Read the METIS part file to determine which nodes belong to this rank
    std::unordered_set<NodeId> myNodes;
    std::ifstream partStream(partFile);
    if (!partStream) {
        std::cerr << "Error: Cannot open part file " << partFile << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    NodeId node = 1; // METIS nodes are 1-indexed
    CompId part;
    while (partStream >> part) {
        if (part % numProcs == rank) { // Simple distribution by modulo
            myNodes.insert(node);
        }
        node++;
    }
    
    // Read the METIS graph file
    std::ifstream graphStream(metisFile);
    if (!graphStream) {
        std::cerr << "Error: Cannot open METIS graph file " << metisFile << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    // Read header
    std::string line;
    std::getline(graphStream, line);
    std::istringstream iss(line);
    int numNodes, numEdges, fmt;
    iss >> numNodes >> numEdges >> fmt;
    
    // Read adjacency lists
    node = 1; // Reset node counter
    while (std::getline(graphStream, line)) {
        std::istringstream lineStream(line);
        NodeId neighbor;
        
        // Add edges for this node
        while (lineStream >> neighbor) {
            // If this node or its neighbor is in my partition, add the edge
            if (myNodes.find(node) != myNodes.end() || myNodes.find(neighbor) != myNodes.end()) {
                graph.addEdge(node, neighbor);
            }
        }
        
        node++;
    }
    
    return graph;
} 