"""
Network Analysis Utilities

"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional
from shapely.geometry import Point, Polygon
from geopy.distance import geodesic
import logging

logger = logging.getLogger(__name__)


def haversine_distance(point1: Point, point2: Point) -> float:
    """
    Calculate Haversine distance between two points (lat/lon).
    
    Haversine formula calculates the great-circle distance between two points
    on a sphere given their latitudes and longitudes.
    
    Args:
        point1: Shapely Point (lon, lat)
        point2: Shapely Point (lon, lat)
        
    Returns:
        Distance in meters
    """
    # Extract coordinates
    lat1, lon1 = point1.y, point1.x
    lat2, lon2 = point2.y, point2.x
    
    # Calculate using geopy
    try:
        distance = geodesic((lat1, lon1), (lat2, lon2)).meters
        return distance
    except Exception as e:
        logger.warning(f"Error calculating geodesic distance: {e}")
        
        return _simple_haversine(lat1, lon1, lat2, lon2)


def _simple_haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Simple Haversine distance calculation.
    
    Args:
        lat1, lon1: Coordinates of first point
        lat2, lon2: Coordinates of second point
        
    Returns:
        Distance in meters
    """
    from math import radians, sin, cos, sqrt, atan2
    
    # Earth radius in meters
    R = 6371000
    
    # Convert to radians
    phi1 = radians(lat1)
    phi2 = radians(lat2)
    delta_phi = radians(lat2 - lat1)
    delta_lambda = radians(lon2 - lon1)
    
    # Haversine formula
    a = sin(delta_phi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(delta_lambda / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
    distance = R * c
    return distance


def euclidean_distance(point1: Point, point2: Point) -> float:
    """
    Calculate Euclidean distance between two points.
    
    Use this for projected coordinates (e.g., UTM).
    
    Args:
        point1: Shapely Point
        point2: Shapely Point
        
    Returns:
        Distance in same units as coordinates
    """
    return point1.distance(point2)


def edge_to_edge_distance(polygon1: Polygon, polygon2: Polygon) -> float:
    """
    Calculate minimum edge-to-edge distance between two polygons.
    
    This is more accurate than centroid-to-centroid for buildings that are
    close together or have irregular shapes.
    
    Args:
        polygon1: First polygon
        polygon2: Second polygon
        
    Returns:
        Minimum distance between polygons
    """
    return polygon1.distance(polygon2)


def build_distance_graph(
    buildings_df,
    footprints: Dict[str, Polygon],
    distance_threshold: float = 200.0,
    distance_method: str = 'haversine',
    use_edge_distance: bool = False
) -> nx.Graph:
    """
    Build network graph from building data.
    
    Creates an undirected weighted graph where:
    - Nodes: Buildings (node_id = building_id)
    - Edges: Connections between buildings within distance threshold
    - Edge weights: Distance between buildings
    
    Args:
        buildings_df: DataFrame with building data (must have 'building_id', 
                     'centroid_lon', 'centroid_lat')
        footprints: Dictionary mapping building_id to Shapely Polygon
        distance_threshold: Maximum distance (meters) for edge creation
        distance_method: 'haversine' (lat/lon) or 'euclidean' (projected)
        use_edge_distance: If True, use edge-to-edge distance instead of centroid
        
    Returns:
        NetworkX undirected weighted graph
    """
    logger.info(f"Building network graph with {len(buildings_df)} buildings")
    logger.info(f"Distance threshold: {distance_threshold}m")
    logger.info(f"Distance method: {distance_method}")
    logger.info(f"Use edge distance: {use_edge_distance}")
    
   
    G = nx.Graph()
    
    
    for _, row in buildings_df.iterrows():
        building_id = row['building_id']
        G.add_node(building_id, **row.to_dict())
    
    logger.info(f"Added {len(G.nodes())} nodes to graph")
    
    
    building_ids = buildings_df['building_id'].tolist()
    num_buildings = len(building_ids)
    edges_added = 0
    
    for i in range(num_buildings):
        building_id1 = building_ids[i]
        row1 = buildings_df[buildings_df['building_id'] == building_id1].iloc[0]
        
        # Create point for building 1
        point1 = Point(row1['centroid_lon'], row1['centroid_lat'])
        polygon1 = footprints.get(building_id1)
        
        for j in range(i + 1, num_buildings):
            building_id2 = building_ids[j]
            row2 = buildings_df[buildings_df['building_id'] == building_id2].iloc[0]
            
            # Create point for building 2
            point2 = Point(row2['centroid_lon'], row2['centroid_lat'])
            polygon2 = footprints.get(building_id2)
            
            
            if use_edge_distance and polygon1 and polygon2:
                distance = edge_to_edge_distance(polygon1, polygon2)
            else:
                if distance_method == 'haversine':
                    distance = haversine_distance(point1, point2)
                else:
                    distance = euclidean_distance(point1, point2)
            
            # Add edge if within threshold
            if distance <= distance_threshold:
                G.add_edge(building_id1, building_id2, weight=distance, distance_m=distance)
                edges_added += 1
    
    logger.info(f"Added {edges_added} edges to graph")
    logger.info(f"Graph density: {nx.density(G):.4f}")
    logger.info(f"Average degree: {sum(dict(G.degree()).values()) / len(G.nodes()):.2f}")
    
    return G


def build_delaunay_graph(
    buildings_df,
    footprints: Dict[str, Polygon]
) -> nx.Graph:
    """
    Build network graph using Delaunay triangulation.
    
    Delaunay triangulation connects each building to its nearest neighbors,
    creating a more realistic connectivity pattern than distance-based threshold.
    
    Args:
        buildings_df: DataFrame with building data
        footprints: Dictionary mapping building_id to Shapely Polygon
        
    Returns:
        NetworkX undirected weighted graph
    """
    from scipy.spatial import Delaunay
    
    logger.info("Building network graph using Delaunay triangulation")
    
    # Create empty graph
    G = nx.Graph()
    
    # Add nodes
    for _, row in buildings_df.iterrows():
        building_id = row['building_id']
        G.add_node(building_id, **row.to_dict())
    
    # Extract coordinates
    coords = []
    building_ids = buildings_df['building_id'].tolist()
    for building_id in building_ids:
        row = buildings_df[buildings_df['building_id'] == building_id].iloc[0]
        coords.append([row['centroid_lon'], row['centroid_lat']])
    
    coords = np.array(coords)
    
    # Create Delaunay triangulation
    try:
        tri = Delaunay(coords)
    except Exception as e:
        logger.error(f"Error creating Delaunay triangulation: {e}")
        return G
    
    # Add edges from triangulation
    edges_added = 0
    for simplex in tri.simplices:
        for i in range(3):
            node1 = building_ids[simplex[i]]
            node2 = building_ids[simplex[(i + 1) % 3]]
            
            if not G.has_edge(node1, node2):
                # Calculate distance
                point1 = Point(coords[simplex[i]])
                point2 = Point(coords[simplex[(i + 1) % 3]])
                distance = haversine_distance(point1, point2)
                
                G.add_edge(node1, node2, weight=distance, distance_m=distance)
                edges_added += 1
    
    logger.info(f"Added {edges_added} edges from Delaunay triangulation")
    logger.info(f"Graph density: {nx.density(G):.4f}")
    
    return G


def calculate_network_statistics(G: nx.Graph) -> Dict:
    """
    Calculate network statistics.
    
    Args:
        G: NetworkX graph
        
    Returns:
        Dictionary with network statistics
    """
    stats = {
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        'density': nx.density(G),
        'average_degree': sum(dict(G.degree()).values()) / len(G.nodes()) if len(G.nodes()) > 0 else 0,
        'is_connected': nx.is_connected(G),
        'num_connected_components': nx.number_connected_components(G),
    }
    
    # Average shortest path length (only for connected graphs)
    if stats['is_connected']:
        try:
            stats['average_shortest_path_length'] = nx.average_shortest_path_length(G, weight='weight')
        except Exception as e:
            logger.warning(f"Error calculating average shortest path length: {e}")
            stats['average_shortest_path_length'] = None
    else:
        stats['average_shortest_path_length'] = None
    
    # Clustering coefficient
    try:
        stats['average_clustering'] = nx.average_clustering(G)
    except Exception as e:
        logger.warning(f"Error calculating clustering coefficient: {e}")
        stats['average_clustering'] = None
    
    return stats

