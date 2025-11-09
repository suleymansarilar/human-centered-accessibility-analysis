"""
User Profile Analysis Script

Usage:
    python scripts/5_user_profiles.py --buildings data/processed/all_buildings.csv --graph data/processed/building_network_graph.pkl --output data/processed/user_accessibility.csv
"""

import argparse
import sys
import os
from pathlib import Path
import pandas as pd
import pickle
import networkx as nx
import json
import numpy as np
import logging

sys.path.append(str(Path(__file__).parent.parent))

from utils.user_profiles import (
    load_user_profiles,
    calculate_profile_accessibility,
    identify_barriers,
    compare_profiles,
    calculate_accessibility_disparities
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def analyze_user_profiles(
    buildings_csv_path: str,
    graph_pkl_path: str,
    output_csv_path: str,
    profile_file: str = 'config/user_profiles.json',
    distance_threshold: float = 500.0
):
    """
    Analyze accessibility for different user profiles.
    
    Args:
        buildings_csv_path: Path to buildings CSV file
        graph_pkl_path: Path to network graph pickle file
        output_csv_path: Path to output CSV file
        profile_file: Path to user profiles JSON file
        distance_threshold: Distance threshold in meters
    """
    logger.info("=" * 60)
    logger.info("Step 5: User Profile Analysis")
    logger.info("=" * 60)
    
    
    logger.info(f"Loading building data from: {buildings_csv_path}")
    if not os.path.exists(buildings_csv_path):
        logger.error(f"Buildings file not found: {buildings_csv_path}")
        raise FileNotFoundError(f"Buildings file not found: {buildings_csv_path}")
    
    buildings_df = pd.read_csv(buildings_csv_path)
    logger.info(f"Loaded {len(buildings_df)} buildings")
    
    
    logger.info(f"Loading graph from: {graph_pkl_path}")
    if not os.path.exists(graph_pkl_path):
        logger.error(f"Graph file not found: {graph_pkl_path}")
        raise FileNotFoundError(f"Graph file not found: {graph_pkl_path}")
    
    with open(graph_pkl_path, 'rb') as f:
        G = pickle.load(f)
    
    logger.info(f"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    
    logger.info(f"Loading user profiles from: {profile_file}")
    profiles_data = load_user_profiles(profile_file)
    user_profiles = profiles_data.get('user_profiles', [])
    logger.info(f"Loaded {len(user_profiles)} user profiles")
    
    
    logger.info("Calculating accessibility for each user profile...")
    accessibility_results = {}
    user_accessibility_data = []
    
    for profile in user_profiles:
        profile_id = profile['profile_id']
        profile_name = profile['name']
        
        logger.info(f"  Processing profile: {profile_name} ({profile_id})")
        
        # Calculate accessibility for this profile
        profile_accessibility = calculate_profile_accessibility(
            buildings_df,
            G,
            profile,
            distance_threshold=distance_threshold
        )
        
        accessibility_results[profile_id] = profile_accessibility
        
        # Identify barriers
        barriers = identify_barriers(buildings_df, G, profile)
        
        # Store results
        for building_id, metrics in profile_accessibility.items():
            building_barriers = barriers.get(building_id, [])
            
            user_accessibility_data.append({
                'building_id': building_id,
                'profile_id': profile_id,
                'profile_name': profile_name,
                'reachable_count': metrics['reachable_count'],
                'avg_travel_time_minutes': metrics['avg_travel_time'] / 60.0 if metrics['avg_travel_time'] != float('inf') else None,
                'accessibility_score': metrics['accessibility_score'],
                'barrier_impact': metrics['barrier_impact'],
                'barriers': ', '.join(building_barriers) if building_barriers else 'none'
            })
        
        logger.info(f"    Calculated accessibility for {len(profile_accessibility)} buildings")
        logger.info(f"    Average reachable buildings: {np.mean([m['reachable_count'] for m in profile_accessibility.values()]):.2f}")
        logger.info(f"    Average accessibility score: {np.mean([m['accessibility_score'] for m in profile_accessibility.values()]):.2f}")
    
    
    user_accessibility_df = pd.DataFrame(user_accessibility_data)
    
    # Save results
    logger.info(f"Saving user accessibility data to: {output_csv_path}")
    output_dir = Path(output_csv_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    user_accessibility_df.to_csv(output_csv_path, index=False)
    logger.info(f"Successfully saved user accessibility data to {output_csv_path}")
    
    
    logger.info("Comparing profiles...")
    comparison_df = compare_profiles(profiles_data, accessibility_results)
    
    comparison_path = output_csv_path.replace('.csv', '_comparison.csv')
    comparison_df.to_csv(comparison_path, index=False)
    logger.info(f"Saved profile comparison to: {comparison_path}")
    
    
    logger.info("Calculating accessibility disparities...")
    disparities = calculate_accessibility_disparities(accessibility_results)
    
    disparities_path = output_csv_path.replace('.csv', '_disparities.json')
    with open(disparities_path, 'w') as f:
        json.dump(disparities, f, indent=2)
    logger.info(f"Saved disparities to: {disparities_path}")
    
    
    logger.info("\nUser Profile Analysis Summary:")
    logger.info("\nProfile Comparison:")
    for _, row in comparison_df.iterrows():
        logger.info(f"  {row['profile_name']}:")
        logger.info(f"    Average reachable buildings: {row['avg_reachable_buildings']:.2f}")
        logger.info(f"    Average travel time: {row['avg_travel_time_minutes']:.2f} minutes")
        logger.info(f"    Average accessibility score: {row['avg_accessibility_score']:.2f}")
        logger.info(f"    Buildings with good accessibility: {row['buildings_with_good_accessibility']}")
    
    logger.info("\nAccessibility Disparities:")
    for profile_id, disparity_data in disparities.items():
        logger.info(f"  {profile_id}:")
        logger.info(f"    Average disparity: {disparity_data['avg_disparity']:.2%}")
        logger.info(f"    Buildings affected: {disparity_data['buildings_affected']}")
    
    logger.info("=" * 60)
    logger.info("Step 5 completed!")
    logger.info("=" * 60)
    
    return user_accessibility_df


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Analyze accessibility for different user profiles'
    )
    parser.add_argument(
        '--buildings',
        type=str,
        required=True,
        help='Path to buildings CSV file'
    )
    parser.add_argument(
        '--graph',
        type=str,
        required=True,
        help='Path to network graph pickle file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/processed/user_accessibility.csv',
        help='Path to output CSV file'
    )
    parser.add_argument(
        '--profiles',
        type=str,
        default='config/user_profiles.json',
        help='Path to user profiles JSON file'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=500.0,
        help='Distance threshold in meters'
    )
    
    args = parser.parse_args()
    
    try:
        analyze_user_profiles(
            args.buildings,
            args.graph,
            args.output,
            args.profiles,
            args.threshold
        )
    except Exception as e:
        logger.error(f"Error in Step 5: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

