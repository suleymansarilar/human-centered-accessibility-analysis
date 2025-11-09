"""
Usage:
    python scripts/2_build_network.py --input data/processed/buildings_geodata.csv --output data/processed/building_network_graph.pkl --threshold 200
"""

import argparse
import sys
import os
from pathlib import Path
import pandas as pd
import pickle
import logging
import networkx as nx


sys.path.append(str(Path(__file__).parent.parent))

from utils.network_utils import build_distance_graph, build_delaunay_graph, calculate_network_statistics


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def build_network(
    input_csv_path: str,
    output_graph_path: str,
    distance_threshold: float = 200.0,
    method: str = 'distance',
    use_edge_distance: bool = False
):
    """
    Build network graph from building data.
    
    Args:
        input_csv_path: Path to buildings_geodata.csv
        output_graph_path: Path to output graph pickle file
        distance_threshold: Maximum distance (meters) for edge creation
        method: 'distance' (threshold-based) or 'delaunay' (triangulation-based)
        use_edge_distance: If True, use edge-to-edge distance
    """
    logger.info("=" * 60)
    logger.info("Step 2: Network Graph Construction")
    logger.info("=" * 60)
    
    
    logger.info(f"Loading building data from: {input_csv_path}")
    if not os.path.exists(input_csv_path):
        logger.error(f"Input file not found: {input_csv_path}")
        raise FileNotFoundError(f"Input file not found: {input_csv_path}")
    
    df = pd.read_csv(input_csv_path)
    logger.info(f"Loaded {len(df)} buildings")
    
    
    footprints_path = input_csv_path.replace('.csv', '_footprints.pkl')
    if not os.path.exists(footprints_path):
        logger.warning(f"Footprints file not found: {footprints_path}")
        logger.warning("Will use centroid-based distance only")
        footprints = {}
    else:
        logger.info(f"Loading footprints from: {footprints_path}")
        with open(footprints_path, 'rb') as f:
            footprints = pickle.load(f)
        logger.info(f"Loaded {len(footprints)} building footprints")
    
    
    if method == 'delaunay':
        logger.info("Using Delaunay triangulation method")
        G = build_delaunay_graph(df, footprints)
    else:
        logger.info("Using distance-based method")
        G = build_distance_graph(
            df,
            footprints,
            distance_threshold=distance_threshold,
            distance_method='haversine',
            use_edge_distance=use_edge_distance
        )
    
    
    logger.info("Calculating network statistics...")
    stats = calculate_network_statistics(G)
    
    logger.info("\nNetwork Statistics:")
    logger.info(f"  Number of nodes: {stats['num_nodes']}")
    logger.info(f"  Number of edges: {stats['num_edges']}")
    logger.info(f"  Density: {stats['density']:.4f}")
    logger.info(f"  Average degree: {stats['average_degree']:.2f}")
    logger.info(f"  Is connected: {stats['is_connected']}")
    logger.info(f"  Number of connected components: {stats['num_connected_components']}")
    if stats['average_shortest_path_length']:
        logger.info(f"  Average shortest path length: {stats['average_shortest_path_length']:.2f}m")
    if stats['average_clustering']:
        logger.info(f"  Average clustering coefficient: {stats['average_clustering']:.4f}")
    
   
    logger.info(f"Saving graph to: {output_graph_path}")
    output_dir = Path(output_graph_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_graph_path, 'wb') as f:
        pickle.dump(G, f)
    logger.info(f"Successfully saved graph to {output_graph_path}")
    
    
    stats_path = output_graph_path.replace('.pkl', '_stats.json')
    import json
    
    stats_serializable = {}
    for key, value in stats.items():
        if value is None:
            stats_serializable[key] = None
        elif isinstance(value, (int, float, str, bool)):
            stats_serializable[key] = value
        else:
            stats_serializable[key] = str(value)
    
    with open(stats_path, 'w') as f:
        json.dump(stats_serializable, f, indent=2)
    logger.info(f"Saved network statistics to: {stats_path}")
    
    
    edges_path = output_graph_path.replace('.pkl', '_edges.csv')
    edges_data = []
    for u, v, data in G.edges(data=True):
        edges_data.append({
            'source': u,
            'target': v,
            'distance_m': data.get('distance_m', data.get('weight', 0)),
            'weight': data.get('weight', 0)
        })
    
    edges_df = pd.DataFrame(edges_data)
    edges_df.to_csv(edges_path, index=False)
    logger.info(f"Saved edge list to: {edges_path}")
    logger.info(f"Total edges: {len(edges_df)}")
    
    return G


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Build network graph from building data'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input buildings_geodata.csv'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/processed/building_network_graph.pkl',
        help='Path to output graph pickle file'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=200.0,
        help='Distance threshold (meters) for edge creation'
    )
    parser.add_argument(
        '--method',
        type=str,
        choices=['distance', 'delaunay'],
        default='distance',
        help='Network construction method'
    )
    parser.add_argument(
        '--use-edge-distance',
        action='store_true',
        help='Use edge-to-edge distance instead of centroid distance'
    )
    
    args = parser.parse_args()
    
    try:
        build_network(
            args.input,
            args.output,
            distance_threshold=args.threshold,
            method=args.method,
            use_edge_distance=args.use_edge_distance
        )
        logger.info("=" * 60)
        logger.info("Adım 2 tamamlandı!")
        logger.info("=" * 60)
    except Exception as e:
        logger.error(f"Error in Adım 2: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

