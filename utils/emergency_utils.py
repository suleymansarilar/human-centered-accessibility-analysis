"""
Emergency Scenarios Utilities

Functions for analyzing emergency evacuation scenarios.
"""

import json
import pandas as pd
import networkx as nx
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def load_emergency_scenarios(scenario_file: str = 'config/emergency_scenarios.json') -> Dict:
    """
    Load emergency scenarios from JSON file.
    
    Args:
        scenario_file: Path to emergency scenarios JSON file
        
    Returns:
        Dictionary with emergency scenarios data
    """
    scenario_path = Path(scenario_file)
    if not scenario_path.exists():
        logger.error(f"Emergency scenarios file not found: {scenario_file}")
        raise FileNotFoundError(f"Emergency scenarios file not found: {scenario_file}")
    
    with open(scenario_path, 'r', encoding='utf-8') as f:
        scenarios_data = json.load(f)
    
    logger.info(f"Loaded {len(scenarios_data.get('emergency_scenarios', []))} emergency scenarios")
    return scenarios_data


def find_safe_zones(
    buildings_df: pd.DataFrame,
    scenario: Dict
) -> List[str]:
    """
    Find safe zones for evacuation.
    
    Args:
        buildings_df: DataFrame with building data
        scenario: Emergency scenario dictionary
        
    Returns:
        List of safe zone building IDs
    """
    safe_zone_types = scenario.get('safe_zones', [])
    safe_zones = []
    
    # For now, we'll use a simplified approach
    # In a real implementation, we'd check building metadata for safe zone indicators
    # (e.g., open space, parking, etc.)
    
    for _, row in buildings_df.iterrows():
        building_id = row['building_id']
        building_type = str(row.get('building_type', '')).lower()
        usage = str(row.get('usage', '')).lower()
        area = row.get('area_m2', 0)
        
        # Check if building matches safe zone types
        for safe_type in safe_zone_types:
            if safe_type.lower() in building_type or safe_type.lower() in usage:
                safe_zones.append(building_id)
                break
        
        # Large open areas can be safe zones
        if 'open_space' in safe_zone_types and area > 500:
            safe_zones.append(building_id)
    
    # If no safe zones found, use all buildings as potential safe zones
    if not safe_zones:
        safe_zones = buildings_df['building_id'].tolist()
    
    return safe_zones


def find_evacuation_routes(
    G: nx.Graph,
    emergency_buildings: List[str],
    safe_zones: List[str],
    k: int = 2
) -> Dict[str, Dict]:
    """
    Find evacuation routes from emergency buildings to safe zones.
    
    Args:
        G: NetworkX graph
        emergency_buildings: List of emergency building IDs
        safe_zones: List of safe zone building IDs
        k: Number of alternative routes
        
    Returns:
        Dictionary mapping emergency building to evacuation routes
    """
    evacuation_routes = {}
    
    for emergency_building in emergency_buildings:
        if emergency_building not in G.nodes():
            continue
        
        routes = {
            'primary_route': None,
            'alternative_routes': [],
            'evacuation_time_minutes': np.inf,
            'route_capacity': 0,
            'safety_score': 0.0
        }
        
        # Find shortest route to any safe zone
        best_route = None
        best_distance = np.inf
        
        for safe_zone in safe_zones:
            if safe_zone not in G.nodes() or safe_zone == emergency_building:
                continue
            
            try:
                if nx.has_path(G, emergency_building, safe_zone):
                    path_length = nx.shortest_path_length(G, emergency_building, safe_zone, weight='weight')
                    if path_length < best_distance:
                        best_distance = path_length
                        best_route = nx.shortest_path(G, emergency_building, safe_zone, weight='weight')
            except Exception as e:
                logger.warning(f"Error finding route from {emergency_building} to {safe_zone}: {e}")
                continue
        
        if best_route:
            routes['primary_route'] = best_route
            routes['evacuation_time_minutes'] = best_distance / (1.4 * 60)
            routes['safety_score'] = 0.5 if best_route else 0.0
        
        evacuation_routes[emergency_building] = routes
    
    return evacuation_routes


def calculate_evacuation_time(
    G: nx.Graph,
    route: List[str],
    walking_speed: float = 1.4,
    panic_factor: float = 1.0
) -> float:
    """
    Calculate evacuation time for a route.
    
    Args:
        G: NetworkX graph
        route: List of building IDs in the route
        walking_speed: Walking speed in m/s
        panic_factor: Panic factor (multiplier for speed)
        
    Returns:
        Evacuation time in minutes
    """
    if len(route) < 2:
        return 0.0
    
    total_distance = 0.0
    for i in range(len(route) - 1):
        u, v = route[i], route[i + 1]
        if G.has_edge(u, v):
            distance = G[u][v].get('distance_m', G[u][v].get('weight', 0))
            total_distance += distance
    
    # Apply panic factor (higher panic = faster walking, but also more congestion)
    effective_speed = walking_speed * panic_factor
    
    # Convert to minutes
    time_minutes = (total_distance / effective_speed) / 60.0
    
    return time_minutes


def analyze_safety(
    evacuation_routes: Dict[str, Dict],
    min_alternative_routes: int = 1
) -> Dict[str, Dict]:
    """
    Analyze safety metrics for evacuation routes.
    
    Args:
        evacuation_routes: Dictionary with evacuation routes
        min_alternative_routes: Minimum number of routes for safety
        
    Returns:
        Dictionary with safety metrics
    """
    safety_analysis = {}
    
    for building_id, routes in evacuation_routes.items():
        has_route = routes.get('primary_route') is not None
        evacuation_time = routes.get('evacuation_time_minutes', np.inf)
        safety_score = routes.get('safety_score', 0.0)
        
        # Simple risk assessment
        is_at_risk = not has_route or evacuation_time > 15.0
        
        safety_analysis[building_id] = {
            'evacuation_routes_count': 1 if has_route else 0,
            'min_evacuation_time_minutes': evacuation_time,
            'safety_score': safety_score,
            'critical_paths': 1 if has_route else 0,
            'is_at_risk': is_at_risk,
            'recommendation': 'Needs improvement' if is_at_risk else 'Acceptable'
        }
    
    return safety_analysis

