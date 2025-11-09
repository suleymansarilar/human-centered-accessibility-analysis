"""
Usage:
    python scripts/4_calculate_accessibility.py --input data/processed/all_buildings.csv --graph data/processed/building_network_graph.pkl --output data/processed/accessibility_metrics.csv
"""

import argparse
import sys
import os
from pathlib import Path
import pandas as pd
import pickle
import networkx as nx
import numpy as np
from typing import Dict, List
import logging


sys.path.append(str(Path(__file__).parent.parent))

from utils.network_utils import haversine_distance
from shapely.geometry import Point


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_distance_accessibility(
    buildings_df: pd.DataFrame,
    distance_threshold: float = 500.0
) -> pd.Series:
    """
    Calculate distance-based accessibility.
    
    For each building, count how many buildings are within distance_threshold.
    
    Args:
        buildings_df: DataFrame with building data
        distance_threshold: Distance threshold in meters
        
    Returns:
        Series with accessibility counts
    """
    logger.info(f"Calculating distance-based accessibility (threshold: {distance_threshold}m)...")
    
    accessibility = {}
    num_buildings = len(buildings_df)
    
    for idx, row in buildings_df.iterrows():
        building_id = row['building_id']
        point1 = Point(row['centroid_lon'], row['centroid_lat'])
        
        count = 0
        for idx2, row2 in buildings_df.iterrows():
            if idx == idx2:
                continue
            
            point2 = Point(row2['centroid_lon'], row2['centroid_lat'])
            distance = haversine_distance(point1, point2)
            
            if distance <= distance_threshold:
                count += 1
        
        accessibility[building_id] = count
    
    logger.info(f"Calculated distance accessibility for {len(accessibility)} buildings")
    return pd.Series(accessibility)


def calculate_network_accessibility(
    G: nx.Graph,
    buildings_df: pd.DataFrame,
    distance_threshold: float = 500.0
) -> pd.Series:
    """
    Calculate network-based accessibility.
    
    For each building, count how many buildings are reachable via network
    within distance_threshold.
    
    Args:
        G: NetworkX graph
        buildings_df: DataFrame with building data
        distance_threshold: Distance threshold in meters
        
    Returns:
        Series with network accessibility counts
    """
    logger.info(f"Calculating network-based accessibility (threshold: {distance_threshold}m)...")
    
    accessibility = {}
    nodes = list(G.nodes())
    
    for node in nodes:
        # Use BFS to find all reachable nodes within distance threshold
        reachable_count = 0
        
        try:
            # Calculate shortest paths to all other nodes
            shortest_paths = nx.single_source_dijkstra_path_length(G, node, weight='weight', cutoff=distance_threshold)
            reachable_count = len(shortest_paths) - 1  # Exclude self
        except Exception as e:
            logger.warning(f"Error calculating network accessibility for {node}: {e}")
            reachable_count = 0
        
        accessibility[node] = reachable_count
    
    logger.info(f"Calculated network accessibility for {len(accessibility)} buildings")
    return pd.Series(accessibility)


def calculate_average_path_distance(
    G: nx.Graph,
    buildings_df: pd.DataFrame
) -> pd.Series:
    """
    Calculate average shortest path distance to all other buildings.
    
    Args:
        G: NetworkX graph
        buildings_df: DataFrame with building data
        
    Returns:
        Series with average path distances
    """
    logger.info("Calculating average path distances...")
    
    avg_distances = {}
    nodes = list(G.nodes())
    
    for node in nodes:
        try:
            if nx.is_connected(G):
                # Calculate shortest paths to all other nodes
                shortest_paths = nx.single_source_dijkstra_path_length(G, node, weight='weight')
                # Exclude self
                distances = [d for n, d in shortest_paths.items() if n != node]
                if distances:
                    avg_distance = np.mean(distances)
                else:
                    avg_distance = 0.0
            else:
                # For disconnected graphs, calculate for connected component
                component = next(nx.connected_components(G), None)
                if component and node in component:
                    subgraph = G.subgraph(component)
                    shortest_paths = nx.single_source_dijkstra_path_length(subgraph, node, weight='weight')
                    distances = [d for n, d in shortest_paths.items() if n != node]
                    if distances:
                        avg_distance = np.mean(distances)
                    else:
                        avg_distance = 0.0
                else:
                    avg_distance = np.inf
        except Exception as e:
            logger.warning(f"Error calculating average path distance for {node}: {e}")
            avg_distance = np.inf
        
        avg_distances[node] = avg_distance
    
    logger.info(f"Calculated average path distances for {len(avg_distances)} buildings")
    return pd.Series(avg_distances)


def calculate_weighted_accessibility(
    buildings_df: pd.DataFrame,
    distance_accessibility: pd.Series,
    area_weight: bool = True
) -> pd.Series:
    """
    Calculate weighted accessibility.
    
    Weight accessibility by building area (larger buildings are more important).
    
    Args:
        buildings_df: DataFrame with building data
        distance_accessibility: Distance-based accessibility counts
        area_weight: Whether to weight by area
        
    Returns:
        Series with weighted accessibility scores
    """
    logger.info("Calculating weighted accessibility...")
    
    
    buildings_df_indexed = buildings_df.set_index('building_id')
    distance_accessibility_aligned = distance_accessibility.reindex(buildings_df_indexed.index, fill_value=0)
    
    if area_weight:
        # Normalize area (0-1)
        max_area = buildings_df_indexed['area_m2'].max()
        min_area = buildings_df_indexed['area_m2'].min()
        area_range = max_area - min_area
        
        if area_range > 0:
            normalized_area = (buildings_df_indexed['area_m2'] - min_area) / area_range
        else:
            normalized_area = pd.Series([1.0] * len(buildings_df_indexed), index=buildings_df_indexed.index)
        
        # Weight accessibility by normalized area
        weighted_accessibility = distance_accessibility_aligned * (1 + normalized_area)
    else:
        weighted_accessibility = distance_accessibility_aligned
    
    logger.info(f"Calculated weighted accessibility for {len(weighted_accessibility)} buildings")
    return weighted_accessibility


def calculate_accessibility(
    input_csv_path: str,
    input_graph_path: str,
    output_csv_path: str,
    distance_threshold: float = 500.0
):
    """
    Calculate accessibility metrics for all buildings.
    
    Args:
        input_csv_path: Path to buildings CSV file
        input_graph_path: Path to network graph pickle file
        output_csv_path: Path to output accessibility metrics CSV file
        distance_threshold: Distance threshold in meters
    """
    logger.info("=" * 60)
    logger.info("Step 4: Human-Centered Accessibility Scoring")
    logger.info("=" * 60)
    
    
    logger.info(f"Loading building data from: {input_csv_path}")
    if not os.path.exists(input_csv_path):
        logger.error(f"Input file not found: {input_csv_path}")
        raise FileNotFoundError(f"Input file not found: {input_csv_path}")
    
    buildings_df = pd.read_csv(input_csv_path)
    logger.info(f"Loaded {len(buildings_df)} buildings")
    
    
    logger.info(f"Loading graph from: {input_graph_path}")
    if not os.path.exists(input_graph_path):
        logger.error(f"Graph file not found: {input_graph_path}")
        raise FileNotFoundError(f"Graph file not found: {input_graph_path}")
    
    with open(input_graph_path, 'rb') as f:
        G = pickle.load(f)
    
    logger.info(f"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Calculate accessibility metrics
    distance_accessibility = calculate_distance_accessibility(buildings_df, distance_threshold)
    network_accessibility = calculate_network_accessibility(G, buildings_df, distance_threshold)
    avg_path_distance = calculate_average_path_distance(G, buildings_df)
    weighted_accessibility = calculate_weighted_accessibility(buildings_df, distance_accessibility, area_weight=True)
    
    # Combine metrics - align all series by building_id
    buildings_indexed = buildings_df.set_index('building_id')
    
    # Align all series to building_id index
    distance_accessibility_aligned = distance_accessibility.reindex(buildings_indexed.index, fill_value=0)
    network_accessibility_aligned = network_accessibility.reindex(buildings_indexed.index, fill_value=0)
    avg_path_distance_aligned = avg_path_distance.reindex(buildings_indexed.index, fill_value=np.nan)
    weighted_accessibility_aligned = weighted_accessibility.reindex(buildings_indexed.index, fill_value=0)
    
    # Create DataFrame
    accessibility_df = pd.DataFrame({
        'building_id': buildings_indexed.index,
        'distance_500m_count': distance_accessibility_aligned.values,
        'network_reachable_count': network_accessibility_aligned.values,
        'avg_path_distance_m': avg_path_distance_aligned.values,
        'weighted_accessibility': weighted_accessibility_aligned.values
    })
    
    # Replace inf with NaN for better handling
    accessibility_df['avg_path_distance_m'] = accessibility_df['avg_path_distance_m'].replace([np.inf, -np.inf], np.nan)
    
    # Save metrics
    logger.info(f"Saving accessibility metrics to: {output_csv_path}")
    output_dir = Path(output_csv_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    accessibility_df.to_csv(output_csv_path, index=False)
    logger.info(f"Successfully saved accessibility metrics to {output_csv_path}")
    
    # Display summary
    logger.info("\nHuman-Centered Accessibility Metrics Summary:")
    logger.info(f"  Average distance accessibility (500m): {accessibility_df['distance_500m_count'].mean():.2f}")
    logger.info(f"  Average network accessibility: {accessibility_df['network_reachable_count'].mean():.2f}")
    avg_path_dist = accessibility_df['avg_path_distance_m'].mean()
    if pd.notna(avg_path_dist):
        logger.info(f"  Average path distance: {avg_path_dist:.2f}m")
    else:
        logger.info(f"  Average path distance: N/A (disconnected graph)")
    
    # Check for valid weighted accessibility values
    valid_weighted = accessibility_df['weighted_accessibility'].dropna()
    if len(valid_weighted) > 0:
        max_idx = valid_weighted.idxmax()
        max_val = valid_weighted.max()
        max_building = accessibility_df.loc[max_idx, 'building_id']
        logger.info(f"  Highest weighted accessibility: {max_building} (value={max_val:.2f})")
    else:
        logger.info(f"  Highest weighted accessibility: N/A")
    
    logger.info("=" * 60)
    logger.info("Step 4 completed!")
    logger.info("=" * 60)
    
    return accessibility_df


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Calculate human-centered accessibility metrics for buildings'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input buildings CSV file'
    )
    parser.add_argument(
        '--graph',
        type=str,
        required=True,
        help='Path to input graph pickle file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/processed/accessibility_metrics.csv',
        help='Path to output accessibility metrics CSV file'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=500.0,
        help='Distance threshold in meters'
    )
    
    args = parser.parse_args()
    
    try:
        calculate_accessibility(
            args.input,
            args.graph,
            args.output,
            args.threshold
        )
    except Exception as e:
        logger.error(f"Error in Adım 4: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

