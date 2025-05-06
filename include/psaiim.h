#ifndef PSAIIM_H
#define PSAIIM_H

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <stack>
#include <algorithm>
#include <cmath>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <mpi.h>
#include <omp.h>
#include <unistd.h> // For gethostname

// Types and Structures
using NodeId = int;
using Weight = double;
using CompId = int;
using Level = int;

// Graph representation
struct Graph {
    std::unordered_map<NodeId, std::vector<NodeId>> adjList;            // Node -> outgoing neighbors
    std::unordered_map<NodeId, std::vector<NodeId>> inAdjList;          // Node -> incoming neighbors
    std::unordered_map<NodeId, std::unordered_map<NodeId, Weight>> edgeWeights; // Edge weights
    std::unordered_set<NodeId> nodes;
    
    // Helper function to add an edge
    void addEdge(NodeId src, NodeId dst, Weight weight = 1.0) {
        adjList[src].push_back(dst);
        inAdjList[dst].push_back(src);
        edgeWeights[src][dst] = weight;
        nodes.insert(src);
        nodes.insert(dst);
    }
    
    // Get outgoing neighbors of a node
    const std::vector<NodeId>& getNeighbors(NodeId node) const {
        static const std::vector<NodeId> empty;
        auto it = adjList.find(node);
        return (it != adjList.end()) ? it->second : empty;
    }
    
    // Get incoming neighbors of a node
    const std::vector<NodeId>& getInNeighbors(NodeId node) const {
        static const std::vector<NodeId> empty;
        auto it = inAdjList.find(node);
        return (it != inAdjList.end()) ? it->second : empty;
    }
    
    // Get out-degree of a node
    size_t getOutDegree(NodeId node) const {
        auto it = adjList.find(node);
        return (it != adjList.end()) ? it->second.size() : 0;
    }
    
    // Get in-degree of a node
    size_t getInDegree(NodeId node) const {
        auto it = inAdjList.find(node);
        return (it != inAdjList.end()) ? it->second.size() : 0;
    }
    
    // Get weight of an edge
    Weight getEdgeWeight(NodeId src, NodeId dst) const {
        auto srcIt = edgeWeights.find(src);
        if (srcIt != edgeWeights.end()) {
            auto dstIt = srcIt->second.find(dst);
            if (dstIt != srcIt->second.end()) {
                return dstIt->second;
            }
        }
        return 0.0;
    }
    
    // Get number of nodes
    size_t getNodeCount() const {
        return nodes.size();
    }
};

// Component info for graph partitioning
struct NodeInfo {
    int index = -1;        // DFS index
    int lowlink = -1;      // Lowest reached index
    bool onStack = false;  // Is node currently on the stack
    CompId compId = -1;    // Component ID
};

// Structure to store influence tree information
struct TreeInfo {
    NodeId root;
    size_t size;
    double avgDepth;
    std::unordered_set<NodeId> nodes;
    
    // For sorting by size (desc) then avg depth (asc)
    bool operator<(const TreeInfo& other) const {
        if (size != other.size)
            return size > other.size;
        return avgDepth < other.avgDepth;
    }
};

// Pseudocode reference:
/*
Algorithm 1: PSAIIM (Main)
1. Partition ← GraphPartition(G)
2. Levels    ← LevelizePartition(Partition)
3. IP        ← ParallelPR(G, A, I, Levels)
4. Candidates ← SelectCandidates(G, IP)
5. S         ← SelectSeeds(G, Candidates, k)
6. return S
*/

// Function declarations
std::unordered_map<NodeId, CompId> GraphPartition(const Graph& graph);
std::vector<std::vector<CompId>> LevelizePartition(const std::unordered_map<NodeId, CompId>& partition, const Graph& graph);
std::unordered_map<NodeId, double> ParallelPR(const Graph& graph, const std::vector<std::vector<CompId>>& levels, const std::unordered_map<NodeId, CompId>& partition);
std::vector<NodeId> SelectCandidates(const Graph& graph, const std::unordered_map<NodeId, double>& influencePower);
std::vector<NodeId> SelectSeeds(const Graph& graph, const std::vector<NodeId>& candidates, int k);

// Helper functions
void Discover(NodeId u, Graph& graph, std::unordered_map<NodeId, NodeInfo>& nodeInfo, 
              std::stack<NodeId>& stack, int& index, CompId& compId);
              
std::unordered_set<NodeId> NodesAtDistance(const Graph& graph, NodeId source, int distance);
TreeInfo InfluenceBFSTree(const Graph& graph, NodeId root, const std::vector<NodeId>& candidates);

// MPI utility functions
Graph LoadPartitionedGraph(const std::string& metisFile, const std::string& partFile, int rank, int numProcs);

#endif // PSAIIM_H 