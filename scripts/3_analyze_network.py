"""
Usage:
    python scripts/3_analyze_network.py --input data/processed/building_network_graph.pkl --output data/processed/network_metrics.csv
"""

import argparse
import sys
import os
from pathlib import Path
import pandas as pd
import pickle
import json
import networkx as nx
import logging
from typing import Dict, List


sys.path.append(str(Path(__file__).parent.parent))


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_shortest_paths(G: nx.Graph, max_paths: int = 100) -> Dict:
    """
    Calculate shortest paths between all node pairs.
    
    Args:
        G: NetworkX graph
        max_paths: Maximum number of paths to calculate (for large graphs)
        
    Returns:
        Dictionary with shortest paths
    """
    logger.info("Calculating shortest paths...")
    
    nodes = list(G.nodes())
    num_nodes = len(nodes)
    total_pairs = num_nodes * (num_nodes - 1) // 2
    
    logger.info(f"Total node pairs: {total_pairs}")
    
    paths = {}
    path_lengths = {}
    
    if total_pairs > max_paths:
        logger.info(f"Large graph detected. Calculating paths for first {max_paths} pairs...")
        # For large graphs, calculate paths for a sample
        import random
        pairs = [(nodes[i], nodes[j]) for i in range(num_nodes) for j in range(i + 1, num_nodes)]
        pairs = random.sample(pairs, min(max_paths, len(pairs)))
    else:
        pairs = [(nodes[i], nodes[j]) for i in range(num_nodes) for j in range(i + 1, num_nodes)]
    
    calculated = 0
    for source, target in pairs:
        try:
            if nx.has_path(G, source, target):
                path = nx.shortest_path(G, source, target, weight='weight')
                length = nx.shortest_path_length(G, source, target, weight='weight')
                paths[f"{source}->{target}"] = path
                path_lengths[f"{source}->{target}"] = length
                calculated += 1
        except Exception as e:
            logger.warning(f"Error calculating path from {source} to {target}: {e}")
            continue
    
    logger.info(f"Calculated {calculated} shortest paths")
    
    return {
        'paths': paths,
        'path_lengths': path_lengths,
        'total_pairs': total_pairs,
        'calculated_pairs': calculated
    }


def calculate_centrality_metrics(G: nx.Graph) -> pd.DataFrame:
    """
    Calculate centrality metrics for all nodes.
    
    Metrics:
    - Degree Centrality: Number of connections
    - Betweenness Centrality: How many shortest paths pass through this node
    - Closeness Centrality: Average distance to all other nodes
    - PageRank: Importance score
    
    Args:
        G: NetworkX graph
        
    Returns:
        DataFrame with centrality metrics
    """
    logger.info("Calculating centrality metrics...")
    
    nodes = list(G.nodes())
    metrics = []
    
    
    logger.info("  Calculating degree centrality...")
    degree_centrality = nx.degree_centrality(G)
    
    # Betweenness Centrality (may be slow for large graphs)
    logger.info("  Calculating betweenness centrality...")
    try:
        # Use approximation for large graphs
        if len(nodes) > 50:
            logger.info("    Large graph detected, using approximation...")
            betweenness_centrality = nx.betweenness_centrality(G, weight='weight', k=min(50, len(nodes)))
        else:
            betweenness_centrality = nx.betweenness_centrality(G, weight='weight')
    except Exception as e:
        logger.warning(f"    Error calculating betweenness centrality: {e}")
        betweenness_centrality = {node: 0.0 for node in nodes}
    
    # Closeness Centrality
    logger.info("  Calculating closeness centrality...")
    try:
        if nx.is_connected(G):
            closeness_centrality = nx.closeness_centrality(G, distance='weight')
        else:
            # For disconnected graphs, calculate for each component
            logger.warning("    Graph is not connected, calculating closeness for each component...")
            closeness_centrality = {}
            for component in nx.connected_components(G):
                subgraph = G.subgraph(component)
                if len(component) > 1:
                    component_closeness = nx.closeness_centrality(subgraph, distance='weight')
                    closeness_centrality.update(component_closeness)
                else:
                    # Single node component
                    node = list(component)[0]
                    closeness_centrality[node] = 0.0
            # Fill missing nodes with 0
            for node in nodes:
                if node not in closeness_centrality:
                    closeness_centrality[node] = 0.0
    except Exception as e:
        logger.warning(f"    Error calculating closeness centrality: {e}")
        closeness_centrality = {node: 0.0 for node in nodes}
    
    
    logger.info("  Calculating PageRank...")
    try:
        pagerank = nx.pagerank(G, weight='weight')
    except Exception as e:
        logger.warning(f"    Error calculating PageRank: {e}")
        pagerank = {node: 1.0 / len(nodes) for node in nodes}
    
    
    for node in nodes:
        metrics.append({
            'building_id': node,
            'degree': G.degree(node),
            'degree_centrality': degree_centrality.get(node, 0.0),
            'betweenness_centrality': betweenness_centrality.get(node, 0.0),
            'closeness_centrality': closeness_centrality.get(node, 0.0),
            'pagerank': pagerank.get(node, 0.0)
        })
    
    df = pd.DataFrame(metrics)
    logger.info(f"Calculated metrics for {len(df)} nodes")
    
    return df


def analyze_network(
    input_graph_path: str,
    output_metrics_path: str,
    output_paths_path: str = None
):
    """
    Analyze network graph and calculate metrics.
    
    Args:
        input_graph_path: Path to input graph pickle file
        output_metrics_path: Path to output metrics CSV file
        output_paths_path: Path to output paths JSON file (optional)
    """
    logger.info("=" * 60)
    logger.info("Step 3: Network Analysis")
    logger.info("=" * 60)
    
    
    logger.info(f"Loading graph from: {input_graph_path}")
    if not os.path.exists(input_graph_path):
        logger.error(f"Graph file not found: {input_graph_path}")
        raise FileNotFoundError(f"Graph file not found: {input_graph_path}")
    
    with open(input_graph_path, 'rb') as f:
        G = pickle.load(f)
    
    logger.info(f"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Calculate centrality metrics
    metrics_df = calculate_centrality_metrics(G)
    
    # Save metrics
    logger.info(f"Saving metrics to: {output_metrics_path}")
    output_dir = Path(output_metrics_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(output_metrics_path, index=False)
    logger.info(f"Successfully saved metrics to {output_metrics_path}")
    
    # Calculate shortest paths
    if output_paths_path:
        logger.info("Calculating shortest paths...")
        paths_data = calculate_shortest_paths(G, max_paths=1000)
        
        # Save paths
        logger.info(f"Saving paths to: {output_paths_path}")
        paths_output_dir = Path(output_paths_path).parent
        paths_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert paths to JSON-serializable format
        paths_serializable = {
            'paths': {k: list(v) for k, v in paths_data['paths'].items()},
            'path_lengths': paths_data['path_lengths'],
            'total_pairs': paths_data['total_pairs'],
            'calculated_pairs': paths_data['calculated_pairs']
        }
        
        with open(output_paths_path, 'w') as f:
            json.dump(paths_serializable, f, indent=2)
        logger.info(f"Successfully saved paths to {output_paths_path}")
    
    
    logger.info("\nCentrality Metrics Summary:")
    logger.info(f"  Highest degree: {metrics_df.loc[metrics_df['degree'].idxmax(), 'building_id']} (degree={metrics_df['degree'].max()})")
    logger.info(f"  Highest betweenness: {metrics_df.loc[metrics_df['betweenness_centrality'].idxmax(), 'building_id']} (value={metrics_df['betweenness_centrality'].max():.4f})")
    logger.info(f"  Highest closeness: {metrics_df.loc[metrics_df['closeness_centrality'].idxmax(), 'building_id']} (value={metrics_df['closeness_centrality'].max():.4f})")
    logger.info(f"  Highest PageRank: {metrics_df.loc[metrics_df['pagerank'].idxmax(), 'building_id']} (value={metrics_df['pagerank'].max():.4f})")
    
    logger.info("=" * 60)
    logger.info("Step 3 completed!")
    logger.info("=" * 60)
    
    return metrics_df


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Analyze network graph and calculate metrics'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input graph pickle file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/processed/network_metrics.csv',
        help='Path to output metrics CSV file'
    )
    parser.add_argument(
        '--paths',
        type=str,
        default=None,
        help='Path to output paths JSON file (optional)'
    )
    
    args = parser.parse_args()
    
    try:
        analyze_network(
            args.input,
            args.output,
            args.paths or args.output.replace('.csv', '_paths.json')
        )
    except Exception as e:
        logger.error(f"Error in Adım 3: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

