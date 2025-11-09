"""
Flow Simulation Utilities

Functions for simulating human flow patterns between buildings.
"""

import json
import pandas as pd
import networkx as nx
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def load_flow_scenarios(scenario_file: str = 'config/flow_scenarios.json') -> Dict:
    """
    Load flow scenarios from JSON file.
    
    Args:
        scenario_file: Path to flow scenarios JSON file
        
    Returns:
        Dictionary with flow scenarios data
    """
    scenario_path = Path(scenario_file)
    if not scenario_path.exists():
        logger.error(f"Flow scenarios file not found: {scenario_file}")
        raise FileNotFoundError(f"Flow scenarios file not found: {scenario_file}")
    
    with open(scenario_path, 'r', encoding='utf-8') as f:
        scenarios_data = json.load(f)
    
    logger.info(f"Loaded {len(scenarios_data.get('flow_scenarios', []))} flow scenarios")
    return scenarios_data


def create_od_matrix(
    buildings_df: pd.DataFrame,
    scenario: Dict
) -> Dict[Tuple[str, str], float]:
    """
    Create origin-destination (OD) matrix for a flow scenario.
    
    Args:
        buildings_df: DataFrame with building data
        scenario: Flow scenario dictionary
        
    Returns:
        Dictionary mapping (origin, destination) to flow volume
    """
    origin_types = scenario.get('origin_building_types', [])
    destination_types = scenario.get('destination_building_types', [])
    flow_multiplier = scenario.get('flow_multiplier', 1.0)
    
    od_matrix = {}
    building_ids = buildings_df['building_id'].tolist()
    
    # Get origin and destination buildings based on type
    origin_buildings = []
    destination_buildings = []
    
    for _, row in buildings_df.iterrows():
        building_id = row['building_id']
        building_type = str(row.get('building_type', '')).lower()
        usage = str(row.get('usage', '')).lower()
        
        # Check if building matches origin types
        for origin_type in origin_types:
            if origin_type.lower() in building_type or origin_type.lower() in usage:
                origin_buildings.append(building_id)
                break
        
        # Check if building matches destination types
        for dest_type in destination_types:
            if dest_type.lower() in building_type or dest_type.lower() in usage:
                destination_buildings.append(building_id)
                break
    
    # If no specific types, use all buildings
    if not origin_buildings:
        origin_buildings = building_ids
    if not destination_buildings:
        destination_buildings = building_ids
    
    # Create OD matrix (simplified: equal distribution)
    # In a real implementation, we'd use building area, occupancy, etc.
    for origin in origin_buildings:
        for destination in destination_buildings:
            if origin != destination:
                # Simple flow calculation: based on building area
                origin_area = buildings_df[buildings_df['building_id'] == origin]['area_m2'].values[0]
                dest_area = buildings_df[buildings_df['building_id'] == destination]['area_m2'].values[0]
                
                # Flow volume: proportional to building areas
                flow_volume = (origin_area * dest_area) / 1000.0 * flow_multiplier
                od_matrix[(origin, destination)] = flow_volume
    
    return od_matrix


def simulate_flow(
    G: nx.Graph,
    od_matrix: Dict[Tuple[str, str], float],
    scenario: Dict,
    distribution: str = 'equal_split'
) -> Dict[str, Dict]:
    """
    Simulate flow on network graph.
    
    Args:
        G: NetworkX graph
        od_matrix: Origin-destination matrix
        scenario: Flow scenario dictionary
        distribution: Flow distribution method
        
    Returns:
        Dictionary with flow simulation results
    """
    flow_intensity = scenario.get('flow_intensity', 'medium')
    congestion_factor = scenario.get('congestion_factor', 1.0)
    
    # Initialize edge flows
    edge_flows = {}
    for u, v in G.edges():
        edge_flows[(u, v)] = 0.0
        edge_flows[(v, u)] = 0.0  # Undirected graph
    
    # Distribute flow from OD matrix
    for (origin, destination), flow_volume in od_matrix.items():
        if origin not in G.nodes() or destination not in G.nodes():
            continue
        
        try:
            # Find shortest path
            if nx.has_path(G, origin, destination):
                path = nx.shortest_path(G, origin, destination, weight='weight')
                
                # Distribute flow along path edges
                if distribution == 'equal_split':
                    # Equal flow on all edges
                    flow_per_edge = flow_volume / max(len(path) - 1, 1)
                else:
                    # Weighted by edge distance (shorter edges get more flow)
                    path_length = nx.shortest_path_length(G, origin, destination, weight='weight')
                    if path_length > 0:
                        flow_per_edge = flow_volume / path_length
                    else:
                        flow_per_edge = flow_volume
                
                # Add flow to each edge in path
                for i in range(len(path) - 1):
                    u, v = path[i], path[i + 1]
                    edge_flows[(u, v)] += flow_per_edge
                    edge_flows[(v, u)] += flow_per_edge  # Undirected
        except Exception as e:
            logger.warning(f"Error simulating flow from {origin} to {destination}: {e}")
            continue
    
    return edge_flows


def calculate_congestion(
    G: nx.Graph,
    edge_flows: Dict[Tuple[str, str], float],
    scenario: Dict,
    congestion_model: str = 'linear'
) -> Dict[Tuple[str, str], float]:
    """
    Calculate congestion levels for each edge.
    
    Args:
        G: NetworkX graph
        edge_flows: Dictionary with edge flows
        scenario: Flow scenario dictionary
        congestion_model: Congestion model type
        
    Returns:
        Dictionary mapping edges to congestion levels
    """
    congestion_threshold = scenario.get('congestion_threshold', 100)
    congestion_levels = {}
    
    for (u, v), flow_volume in edge_flows.items():
        if (u, v) not in G.edges():
            continue
        
        # Simple linear congestion model
        congestion = min(flow_volume / congestion_threshold, 1.0) if congestion_threshold > 0 else 0.0
        congestion_levels[(u, v)] = congestion
    
    return congestion_levels


def calculate_travel_time_with_congestion(
    G: nx.Graph,
    edge_flows: Dict[Tuple[str, str], float],
    congestion_levels: Dict[Tuple[str, str], float],
    base_walking_speed: float = 1.4
) -> Dict[Tuple[str, str], float]:
    """
    Calculate travel time with congestion effects.
    
    Args:
        G: NetworkX graph
        edge_flows: Dictionary with edge flows
        congestion_levels: Dictionary with congestion levels
        base_walking_speed: Base walking speed in m/s
        
    Returns:
        Dictionary mapping edges to travel times
    """
    travel_times = {}
    
    for (u, v), congestion in congestion_levels.items():
        if (u, v) not in G.edges():
            continue
        
        # Get edge distance
        distance = G[u][v].get('distance_m', G[u][v].get('weight', 0))
        
        # Simple congestion penalty: reduce speed by congestion percentage
        effective_speed = base_walking_speed * (1 - congestion * 0.3)  # Max 30% speed reduction
        travel_time = distance / max(effective_speed, 0.1)  # Minimum speed
        
        travel_times[(u, v)] = travel_time
    
    return travel_times


def identify_bottlenecks(
    edge_flows: Dict[Tuple[str, str], float],
    congestion_levels: Dict[Tuple[str, str], float],
    threshold: float = 0.6
) -> List[Dict]:
    """
    Identify bottleneck edges.
    
    Args:
        edge_flows: Dictionary with edge flows
        congestion_levels: Dictionary with congestion levels
        threshold: Congestion threshold for bottleneck identification
        
    Returns:
        List of bottleneck dictionaries
    """
    bottlenecks = []
    
    for (u, v), congestion in congestion_levels.items():
        if congestion >= threshold:
            flow_volume = edge_flows.get((u, v), 0.0)
            bottlenecks.append({
                'edge': f"{u}->{v}",
                'source': u,
                'target': v,
                'congestion_level': congestion,
                'flow_volume': flow_volume,
                'recommendation': 'Widen path or add alternative route' if congestion > 0.8 else 'Monitor congestion'
            })
    
    # Sort by congestion level (highest first)
    bottlenecks.sort(key=lambda x: x['congestion_level'], reverse=True)
    
    return bottlenecks

