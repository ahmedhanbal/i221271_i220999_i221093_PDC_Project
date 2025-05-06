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
    
    // Maps for component to nodes
    std::unordered_map<CompId, std::vector<NodeId>> compToNodes;
    
    // Build compToNodes mapping
    for (const auto& [node, compId] : partition) {
        compToNodes[compId].push_back(node);
    }
    
    // Initialize influence power - use thread-safe initialization
    std::unordered_map<NodeId, double> IP_old, IP_new;
    const double initialValue = 1.0 / nodeCount;
    
    for (NodeId node : graph.nodes) {
        IP_old[node] = initialValue;
        IP_new[node] = 0.0;  // Initialize to zero to avoid data races
    }
    
    // Iterative computation until convergence
    int maxIterations = 100;  // Add a maximum iterations limit
    int iteration = 0;
    
    while (diff > epsilon && iteration < maxIterations) {
        iteration++;
        
        // Reset IP_new to zero for this iteration - prevent race conditions
        for (NodeId node : graph.nodes) {
            IP_new[node] = (1.0 - dampingFactor) / nodeCount;  // Start with teleport probability
        }
        
        // Process each level in sequence (MPI ranks assigned to levels)
        for (const auto& level : levels) {
            // Process components in the current level in parallel
            #pragma omp parallel for schedule(dynamic)
            for (size_t i = 0; i < level.size(); i++) {
                CompId compId = level[i];
                
                // Create thread-local temporary storage
                std::unordered_map<NodeId, double> localUpdates;
                
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
                    
                    // Store in thread-local map first
                    localUpdates[u] = dampingFactor * sum;
                }
                
                // Update global IP_new with thread-local results
                #pragma omp critical
                {
                    for (const auto& [node, value] : localUpdates) {
                        IP_new[node] += value;  // Add to the teleport probability already there
                    }
                }
            }
        }
        
        // Calculate difference for convergence check - use thread-safe reduction
        diff = 0.0;
        
        // Convert unordered_set to vector for proper OpenMP iteration
        std::vector<NodeId> nodesVector(graph.nodes.begin(), graph.nodes.end());
        
        #pragma omp parallel reduction(+:diff)
        {
            double localDiff = 0.0;
            
            #pragma omp for nowait
            for (size_t i = 0; i < nodesVector.size(); i++) {
                NodeId node = nodesVector[i];
                localDiff += std::abs(IP_new[node] - IP_old[node]);
                
                // Update old values (thread-safe since each thread handles different nodes)
                #pragma omp critical
                {
                    IP_old[node] = IP_new[node];
                }
            }
            
            diff = localDiff;
        }
        
        if (iteration % 10 == 0) {
            char hostname[256];
            gethostname(hostname, sizeof(hostname));
            std::cout << "[Host: " << hostname << "] Iteration " << iteration << ", diff = " << diff << std::endl;
        }
    }
    
    char hostname[256];
    gethostname(hostname, sizeof(hostname));
    std::cout << "[Host: " << hostname << "] PageRank converged after " << iteration << " iterations with diff = " << diff << std::endl;
    
    return IP_old;
}

// Get nodes at a specific distance from a source node
std::unordered_set<NodeId> NodesAtDistance(const Graph& graph, NodeId source, int distance) {
    // Each call gets its own data structures to ensure thread safety
    std::unordered_set<NodeId> result;
    std::unordered_map<NodeId, int> nodeDistance;
    std::queue<NodeId> q;
    
    // Skip BFS entirely if distance is too small
    if (distance <= 0) {
        return result;
    }
    
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
            continue;  // Don't explore neighbors of nodes at target distance
        }
        
        if (currentDist > distance) {
            continue;  // Stop if we've gone too far
        }
        
        // Explore neighbors
        for (NodeId neighbor : graph.getNeighbors(current)) {
            // Avoid cycles: only process a node if we haven't seen it yet
            if (nodeDistance.find(neighbor) == nodeDistance.end()) {
                nodeDistance[neighbor] = currentDist + 1;
                q.push(neighbor);
                
                // Add to result if exactly at target distance
                if (currentDist + 1 == distance) {
                    result.insert(neighbor);
                }
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
    
    // Process nodes in parallel - use thread-local storage
    #pragma omp parallel
    {
        // Local candidates for each thread
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
        
        // Merge local candidates to global list with proper synchronization
        #pragma omp critical
        {
            // Allocate sufficient space first to avoid reallocation issues
            size_t oldSize = candidates.size();
            candidates.resize(oldSize + localCandidates.size());
            // Copy elements instead of using insert which might reallocate
            std::copy(localCandidates.begin(), localCandidates.end(), candidates.begin() + oldSize);
        }
    }
    
    return candidates;
}

// Create influence BFS tree for a candidate
TreeInfo InfluenceBFSTree(const Graph& graph, NodeId root, const std::vector<NodeId>& candidates) {
    // Each call gets its own data structures
    TreeInfo tree;
    tree.root = root;
    
    // Use sets for faster lookup and avoid duplicates
    std::unordered_set<NodeId> visited;
    std::unordered_map<NodeId, NodeId> parent;
    std::unordered_map<NodeId, int> depth;
    std::queue<NodeId> q;
    
    // Initialize with root
    visited.insert(root);
    parent[root] = -1;
    depth[root] = 0;
    q.push(root);
    
    // Convert candidates to set for faster lookup - make a copy to ensure thread safety
    std::unordered_set<NodeId> candidateSet(candidates.begin(), candidates.end());
    
    // BFS traversal with maximum depth limit to avoid infinite loops in cycles
    const int MAX_DEPTH = 100; // Reasonable limit for social networks
    
    while (!q.empty()) {
        NodeId v = q.front();
        q.pop();
        
        // Stop exploring if we've reached the maximum depth
        if (depth[v] >= MAX_DEPTH) {
            continue;
        }
        
        // Get neighbors safely
        const std::vector<NodeId>& neighbors = graph.getNeighbors(v);
        
        for (NodeId w : neighbors) {
            // Only add nodes that are candidates and haven't been visited
            if (visited.find(w) == visited.end() && candidateSet.find(w) != candidateSet.end()) {
                visited.insert(w);
                parent[w] = v;
                depth[w] = depth[v] + 1;
                q.push(w);
            }
        }
    }
    
    // Calculate average depth - guard against division by zero
    double totalDepth = 0.0;
    for (const auto& [node, nodeDepth] : depth) {
        totalDepth += nodeDepth;
    }
    
    tree.size = visited.size();
    tree.avgDepth = (tree.size > 0) ? totalDepth / tree.size : 0.0;
    tree.nodes = std::move(visited); // Move to avoid copy
    
    return tree;
}

// Algorithm 7: Select Seeds
std::vector<NodeId> SelectSeeds(const Graph& graph, const std::vector<NodeId>& candidates, int k) {
    // Vector to store tree information (thread-safe)
    std::vector<TreeInfo> treeInfos;
    
    // Pre-allocate space to avoid resizing during merge
    #pragma omp single
    {
        treeInfos.reserve(candidates.size());
    }
    
    // Build influence trees for each candidate in parallel
    #pragma omp parallel
    {
        // Local storage for each thread
        std::vector<TreeInfo> localTreeInfos;
        localTreeInfos.reserve(candidates.size() / omp_get_num_threads() + 1);
        
        #pragma omp for
        for (size_t i = 0; i < candidates.size(); i++) {
            NodeId u = candidates[i];
            TreeInfo tree = InfluenceBFSTree(graph, u, candidates);
            localTreeInfos.push_back(tree);
        }
        
        // Merge local trees to global list with proper synchronization
        #pragma omp critical
        {
            // Allocate sufficient space first to avoid reallocation issues
            size_t oldSize = treeInfos.size();
            treeInfos.resize(oldSize + localTreeInfos.size());
            // Copy elements instead of using insert which might reallocate
            std::copy(localTreeInfos.begin(), localTreeInfos.end(), treeInfos.begin() + oldSize);
        }
    }
    
    // Sort TreeInfos by size (desc) and avgDepth (asc)
    std::sort(treeInfos.begin(), treeInfos.end());
    
    // Select seeds
    std::vector<NodeId> seeds;
    seeds.reserve(k);  // Reserve space for efficiency
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
    
    try {
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
        
        partStream.close();
        
        // Read the METIS graph file
        std::ifstream graphStream(metisFile);
        if (!graphStream) {
            std::cerr << "Error: Cannot open METIS graph file " << metisFile << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        // Read header
        std::string line;
        if (!std::getline(graphStream, line)) {
            std::cerr << "Error: Empty METIS file" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        int numNodes = 0, numEdges = 0, fmt = 0;
        try {
            std::istringstream iss(line);
            iss >> numNodes >> numEdges >> fmt;
        } catch (const std::exception& e) {
            std::cerr << "Error parsing METIS header: " << e.what() << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        // Read adjacency lists
        node = 1; // Reset node counter
        while (std::getline(graphStream, line)) {
            if (line.empty() || line[0] == '#') {
                continue;  // Skip empty lines and comments
            }
            
            try {
                std::istringstream lineStream(line);
                NodeId neighbor;
                
                // Add edges for this node
                while (lineStream >> neighbor) {
                    // If this node or its neighbor is in my partition, add the edge
                    if (myNodes.find(node) != myNodes.end() || myNodes.find(neighbor) != myNodes.end()) {
                        graph.addEdge(node, neighbor);
                    }
                }
            } catch (const std::exception& e) {
                std::cerr << "Error parsing METIS line " << node << ": " << e.what() << std::endl;
                // Continue with the next line instead of aborting
            }
            
            node++;
            
            // Safety check to ensure we don't exceed declared number of nodes
            if (node > numNodes) {
                break;
            }
        }
        
        graphStream.close();
        
        // Get hostname for debugging
        char hostname[256];
        gethostname(hostname, sizeof(hostname));
        
        // Output some stats
        std::cout << "Rank " << rank << " on host " << hostname 
                  << " loaded " << graph.nodes.size() 
                  << " nodes and " << graph.adjList.size() << " edges." << std::endl;
                  
    } catch (const std::exception& e) {
        std::cerr << "Error in LoadPartitionedGraph: " << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    return graph;
} 