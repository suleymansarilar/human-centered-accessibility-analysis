"""
User Profile Analysis Utilities

Functions for analyzing accessibility for different user profiles.
"""

import json
import pandas as pd
import networkx as nx
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging
import sys

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from utils.network_utils import haversine_distance
from shapely.geometry import Point

logger = logging.getLogger(__name__)


def load_user_profiles(profile_file: str = 'config/user_profiles.json') -> Dict:
    """
    Load user profiles from JSON file.
    
    Args:
        profile_file: Path to user profiles JSON file
        
    Returns:
        Dictionary with user profiles data
    """
    profile_path = Path(profile_file)
    if not profile_path.exists():
        logger.error(f"User profiles file not found: {profile_file}")
        raise FileNotFoundError(f"User profiles file not found: {profile_file}")
    
    with open(profile_path, 'r', encoding='utf-8') as f:
        profiles_data = json.load(f)
    
    logger.info(f"Loaded {len(profiles_data.get('user_profiles', []))} user profiles")
    return profiles_data


def calculate_profile_accessibility(
    buildings_df: pd.DataFrame,
    G: nx.Graph,
    profile: Dict,
    distance_threshold: float = 500.0
) -> Dict[str, float]:
    """
    Calculate accessibility metrics for a specific user profile.
    
    Args:
        buildings_df: DataFrame with building data
        G: NetworkX graph
        profile: User profile dictionary
        distance_threshold: Distance threshold in meters
        
    Returns:
        Dictionary with accessibility metrics for each building
    """
    profile_id = profile['profile_id']
    walking_speed = profile.get('walking_speed_mps', 1.4)
    max_distance = profile.get('max_distance_m', 1000.0)
    barrier_types = profile.get('barrier_types', [])
    
    # Adjust distance threshold based on profile
    profile_threshold = min(distance_threshold, max_distance)
    
    accessibility = {}
    nodes = list(G.nodes())
    
    for node in nodes:
        building_row = buildings_df[buildings_df['building_id'] == node]
        if building_row.empty:
            continue
        
        building_id = node
        
        # Calculate reachable buildings via network
        reachable_count = 0
        total_travel_time = 0.0
        barrier_impact = 0.0
        
        try:
            # Calculate shortest paths to all other nodes within threshold
            # Adjust edge weights based on walking speed
            # Create a modified graph with adjusted weights
            G_modified = G.copy()
            
            # Adjust edge weights based on walking speed
            for u, v, data in G_modified.edges(data=True):
                distance = data.get('distance_m', data.get('weight', 0))
                # Travel time = distance / walking_speed
                travel_time = distance / walking_speed
                G_modified[u][v]['travel_time'] = travel_time
                # Use travel time as weight for accessibility calculation
                G_modified[u][v]['weight'] = travel_time
            
            # Calculate reachable nodes within time threshold
            time_threshold = profile_threshold / walking_speed
            
            shortest_paths = nx.single_source_dijkstra_path_length(
                G_modified, 
                node, 
                weight='weight', 
                cutoff=time_threshold
            )
            
            reachable_count = len(shortest_paths) - 1  # Exclude self
            
            # Calculate average travel time to reachable buildings
            if reachable_count > 0:
                travel_times = [time for n, time in shortest_paths.items() if n != node]
                total_travel_time = np.mean(travel_times) if travel_times else 0.0
            else:
                total_travel_time = np.inf
            
            if barrier_types:
                barrier_penalty = len(barrier_types) * 0.2
                total_travel_time = total_travel_time * (1 + barrier_penalty)
            
        except Exception as e:
            logger.warning(f"Error calculating accessibility for {node} (profile: {profile_id}): {e}")
            reachable_count = 0
            total_travel_time = np.inf
            barrier_impact = 1.0
        
        if total_travel_time > 0 and total_travel_time != np.inf:
            accessibility_score = 1.0 / (1.0 + total_travel_time / 60.0)
        else:
            accessibility_score = 0.0
        
        accessibility[building_id] = {
            'reachable_count': reachable_count,
            'avg_travel_time': total_travel_time,
            'accessibility_score': accessibility_score,
            'barrier_impact': barrier_impact
        }
    
    return accessibility


def identify_barriers(
    buildings_df: pd.DataFrame,
    G: nx.Graph,
    profile: Dict
) -> Dict[str, List[str]]:
    """
    Identify barriers for a specific user profile.
    
    Args:
        buildings_df: DataFrame with building data
        G: NetworkX graph
        profile: User profile dictionary
        
    Returns:
        Dictionary mapping building_id to list of barrier types
    """
    barrier_types = profile.get('barrier_types', [])
    barriers = {}
    
    for _, row in buildings_df.iterrows():
        building_id = row['building_id']
        building_barriers = []
        building_type = row.get('building_type', '')
        area = row.get('area_m2', 0)
        
        if 'stairs' in barrier_types:
            if building_type and 'old' in str(building_type).lower():
                building_barriers.append('stairs')
        
        if 'narrow_paths' in barrier_types:
            if area < 100:
                building_barriers.append('narrow_paths')
        
        barriers[building_id] = building_barriers
    
    return barriers


def compare_profiles(
    profiles_data: Dict,
    accessibility_results: Dict[str, Dict]
) -> pd.DataFrame:
    """
    Compare accessibility across different user profiles.
    
    Args:
        profiles_data: User profiles data
        accessibility_results: Dictionary mapping profile_id to accessibility results
        
    Returns:
        DataFrame with comparison results
    """
    comparison_data = []
    
    for profile in profiles_data.get('user_profiles', []):
        profile_id = profile['profile_id']
        profile_name = profile['name']
        
        if profile_id not in accessibility_results:
            continue
        
        results = accessibility_results[profile_id]
        
        # Calculate summary statistics
        reachable_counts = [r['reachable_count'] for r in results.values()]
        travel_times = [r['avg_travel_time'] for r in results.values() if r['avg_travel_time'] != np.inf]
        accessibility_scores = [r['accessibility_score'] for r in results.values()]
        
        comparison_data.append({
            'profile_id': profile_id,
            'profile_name': profile_name,
            'avg_reachable_buildings': np.mean(reachable_counts) if reachable_counts else 0,
            'max_reachable_buildings': np.max(reachable_counts) if reachable_counts else 0,
            'min_reachable_buildings': np.min(reachable_counts) if reachable_counts else 0,
            'avg_travel_time_minutes': np.mean(travel_times) / 60.0 if travel_times else np.inf,
            'avg_accessibility_score': np.mean(accessibility_scores) if accessibility_scores else 0,
            'buildings_with_good_accessibility': sum(1 for score in accessibility_scores if score > 0.5)
        })
    
    return pd.DataFrame(comparison_data)


def calculate_accessibility_disparities(
    accessibility_results: Dict[str, Dict],
    reference_profile: str = 'able_bodied_adult'
) -> Dict:
    """
    Calculate accessibility disparities between profiles.
    
    Args:
        accessibility_results: Dictionary mapping profile_id to accessibility results
        reference_profile: Reference profile for comparison
        
    Returns:
        Dictionary with disparity metrics
    """
    if reference_profile not in accessibility_results:
        logger.warning(f"Reference profile {reference_profile} not found")
        return {}
    
    reference_results = accessibility_results[reference_profile]
    disparities = {}
    
    for profile_id, results in accessibility_results.items():
        if profile_id == reference_profile:
            continue
        
        # Calculate disparity for each building
        building_disparities = []
        for building_id in reference_results.keys():
            if building_id in results:
                ref_score = reference_results[building_id]['accessibility_score']
                profile_score = results[building_id]['accessibility_score']
                
                if ref_score > 0:
                    disparity = (ref_score - profile_score) / ref_score
                    building_disparities.append(disparity)
        
        if building_disparities:
            disparities[profile_id] = {
                'avg_disparity': np.mean(building_disparities),
                'max_disparity': np.max(building_disparities),
                'min_disparity': np.min(building_disparities),
                'buildings_affected': sum(1 for d in building_disparities if d > 0.1)
            }
    
    return disparities

