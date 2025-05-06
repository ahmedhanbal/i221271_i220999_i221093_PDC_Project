#!/usr/bin/env python3
"""
Convert higgs-social_network.edgelist to METIS graph format.
"""
import sys
import os

def convert_to_metis(input_file, output_file):
    # Dictionary to store adjacency list
    adj_list = {}
    
    # Step 1: Read the edge list and build adjacency list
    with open(input_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            
            src = int(parts[0])
            dst = int(parts[1])
            
            # Add to adjacency list
            if src not in adj_list:
                adj_list[src] = []
            adj_list[src].append(dst)
            
            # Ensure destination node exists in adj_list (even if it has no outgoing edges)
            if dst not in adj_list:
                adj_list[dst] = []
    
    # Step 2: Reindex nodes to be 1-indexed consecutive integers
    node_map = {}
    for idx, node in enumerate(sorted(adj_list.keys()), 1):
        node_map[node] = idx
    
    # Step 3: Write to METIS format
    # Format:
    # num_nodes num_edges 0 (0 for unweighted)
    # for each node i, list its adjacent nodes
    with open(output_file, 'w') as f:
        num_nodes = len(adj_list)
        # Count total edges (each edge is counted once in METIS format)
        num_edges = sum(len(neighbors) for neighbors in adj_list.values())
        
        # Write header
        f.write(f"{num_nodes} {num_edges} 0\n")
        
        # Write adjacency list for each node
        for i in range(1, num_nodes + 1):
            orig_node = next(k for k, v in node_map.items() if v == i)
            neighbors = [str(node_map[n]) for n in adj_list[orig_node]]
            f.write(' '.join(neighbors) + '\n')
    
    print(f"Converted {input_file} to METIS format in {output_file}")
    print(f"Nodes: {num_nodes}, Edges: {num_edges}")
    
    # Create node mapping file
    map_file = os.path.splitext(output_file)[0] + ".nodemap"
    with open(map_file, 'w') as f:
        for orig_node, new_node in node_map.items():
            f.write(f"{orig_node} {new_node}\n")
    
    print(f"Node mapping saved to {map_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} input_edgelist output_metis")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    convert_to_metis(input_file, output_file) 