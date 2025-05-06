#!/usr/bin/env python3
"""
Convert higgs-social_network.edgelist to METIS graph format with multithreading for efficiency.
"""
import sys
import os
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import time
from collections import defaultdict

def read_chunk(file_obj, chunk_size=1000000):
    """Read a chunk of lines from the file."""
    lines = []
    for _ in range(chunk_size):
        line = file_obj.readline()
        if not line:
            break
        lines.append(line)
    return lines

def process_chunk(chunk):
    """Process a chunk of lines into a partial adjacency list."""
    local_adj_list = defaultdict(list)
    nodes = set()
    
    for line in chunk:
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        
        try:
            src = int(parts[0])
            dst = int(parts[1])
            
            # Add to adjacency list
            local_adj_list[src].append(dst)
            
            # Track all nodes (including those with no outgoing edges)
            nodes.add(src)
            nodes.add(dst)
        except ValueError:
            continue  # Skip lines with non-integer values
            
    # Ensure all nodes exist in the adjacency list
    for node in nodes:
        if node not in local_adj_list:
            local_adj_list[node] = []
            
    return local_adj_list, nodes

def convert_to_metis(input_file, output_file, num_threads=None):
    if num_threads is None:
        num_threads = multiprocessing.cpu_count()
    
    print(f"Starting conversion using {num_threads} threads...")
    start_time = time.time()
    
    # Step 1: Read the edge list and build adjacency list in parallel
    chunk_size = 1000000  # Adjust chunk size based on memory constraints
    partial_adj_lists = []
    all_nodes = set()
    
    with open(input_file, 'r') as f:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            while True:
                chunk = read_chunk(f, chunk_size)
                if not chunk:
                    break
                future = executor.submit(process_chunk, chunk)
                futures.append(future)
            
            # Collect results
            for future in futures:
                adj_list, nodes = future.result()
                partial_adj_lists.append(adj_list)
                all_nodes.update(nodes)
    
    # Merge adjacency lists
    adj_list = defaultdict(list)
    for partial in partial_adj_lists:
        for node, neighbors in partial.items():
            adj_list[node].extend(neighbors)
    
    adj_build_time = time.time() - start_time
    print(f"Adjacency list built in {adj_build_time:.2f} seconds")
    print(f"Found {len(all_nodes)} unique nodes")
    
    # Step 2: Reindex nodes to be 1-indexed consecutive integers
    mapping_start = time.time()
    print("Creating node mapping...")
    
    # Create mapping directly from sorted nodes
    node_map = {node: idx for idx, node in enumerate(sorted(all_nodes), 1)}
    
    mapping_time = time.time() - mapping_start
    print(f"Node mapping created in {mapping_time:.2f} seconds")
    
    # Step 3: Write to METIS format
    writing_start = time.time()
    print("Writing METIS file...")
    
    num_nodes = len(node_map)
    # Count total edges - METIS counts each edge only once (undirected graph format)
    # So we divide by 2 for undirected graph interpretation
    total_edges = sum(len(neighbors) for neighbors in adj_list.values())
    metis_edges = total_edges // 2  # For METIS format, count each edge only once
    
    print(f"Total directed edges: {total_edges}, METIS undirected edges: {metis_edges}")
    
    # Create batched node processing for writing
    def process_node_batch(node_batch):
        batch_results = []
        for node in node_batch:
            new_idx = node_map[node]
            if node in adj_list:  # Check if node has outgoing edges
                neighbors = [str(node_map[n]) for n in adj_list[node]]
                batch_results.append((new_idx, ' '.join(neighbors)))
            else:
                batch_results.append((new_idx, ''))
        return batch_results
    
    # Process nodes in batches
    batch_size = max(1000, num_nodes // (num_threads * 10))  # Heuristic for batch size
    node_batches = []
    nodes_list = list(all_nodes)
    
    for i in range(0, len(nodes_list), batch_size):
        node_batches.append(nodes_list[i:i+batch_size])
    
    # Process batches in parallel
    results = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(process_node_batch, batch) for batch in node_batches]
        for future in futures:
            results.extend(future.result())
    
    # Sort results by new node index
    results.sort(key=lambda x: x[0])
    
    # Write to file
    with open(output_file, 'w') as f:
        # Write header - METIS format expects undirected edge count
        f.write(f"{num_nodes} {metis_edges} 0\n")
        
        # Write each node's adjacency list
        for _, adjacency in results:
            f.write(adjacency + '\n')
    
    writing_time = time.time() - writing_start
    print(f"METIS file written in {writing_time:.2f} seconds")
    
    # Create node mapping file
    map_file = os.path.splitext(output_file)[0] + ".nodemap"
    print(f"Writing node mapping to {map_file}...")
    
    map_writing_start = time.time()
    
    # Write node mapping in batches
    def write_node_map_batch(batch, file_obj):
        for orig_node, new_node in batch:
            file_obj.write(f"{orig_node} {new_node}\n")
    
    # Create batches of node mapping entries
    map_entries = list(node_map.items())
    map_batches = []
    
    for i in range(0, len(map_entries), batch_size):
        map_batches.append(map_entries[i:i+batch_size])
    
    with open(map_file, 'w') as f:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            for batch in map_batches:
                executor.submit(write_node_map_batch, batch, f)
    
    map_writing_time = time.time() - map_writing_start
    print(f"Node mapping written in {map_writing_time:.2f} seconds")
    
    end_time = time.time()
    print(f"Converted {input_file} to METIS format in {output_file}")
    print(f"Nodes: {num_nodes}, Edges: {metis_edges}")
    print(f"Total conversion time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} input_edgelist output_metis [num_threads]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    # Optional number of threads parameter
    num_threads = None
    if len(sys.argv) >= 4:
        try:
            num_threads = int(sys.argv[3])
        except ValueError:
            print("Invalid number of threads, using default (CPU count)")
    
    convert_to_metis(input_file, output_file, num_threads) 