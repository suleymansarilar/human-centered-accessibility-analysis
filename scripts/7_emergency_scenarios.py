"""
Emergency Scenarios Script

Usage:
    python scripts/7_emergency_scenarios.py --buildings data/processed/all_buildings.csv --graph data/processed/building_network_graph.pkl --output data/processed/emergency_scenarios.json
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

from utils.emergency_utils import (
    load_emergency_scenarios,
    find_safe_zones,
    find_evacuation_routes,
    calculate_evacuation_time,
    analyze_safety
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def analyze_emergency_scenarios(
    buildings_csv_path: str,
    graph_pkl_path: str,
    output_json_path: str,
    scenario_file: str = 'config/emergency_scenarios.json'
):
    """
    Analyze emergency evacuation scenarios.
    
    Args:
        buildings_csv_path: Path to buildings CSV file
        graph_pkl_path: Path to network graph pickle file
        output_json_path: Path to output JSON file
        scenario_file: Path to emergency scenarios JSON file
    """
    logger.info("=" * 60)
    logger.info("Step 7: Emergency Scenarios Analysis")
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
    
    
    logger.info(f"Loading emergency scenarios from: {scenario_file}")
    scenarios_data = load_emergency_scenarios(scenario_file)
    emergency_scenarios = scenarios_data.get('emergency_scenarios', [])
    evacuation_params = scenarios_data.get('evacuation_parameters', {})
    logger.info(f"Loaded {len(emergency_scenarios)} emergency scenarios")
    
    
    logger.info("Analyzing emergency scenarios...")
    emergency_analysis = {}
    all_safety_analysis = {}
    
    for scenario in emergency_scenarios:
        scenario_id = scenario['scenario_id']
        scenario_name = scenario['name']
        
        logger.info(f"  Processing scenario: {scenario_name} ({scenario_id})")
        
        # Determine emergency buildings
        emergency_building_ids = scenario.get('emergency_building_ids', [])
        emergency_building_types = scenario.get('emergency_building_types', [])
        
        if emergency_building_ids == ['all'] or emergency_building_types == ['all']:
            emergency_buildings = buildings_df['building_id'].tolist()
        else:
            emergency_buildings = []
            for _, row in buildings_df.iterrows():
                building_id = row['building_id']
                building_type = str(row.get('building_type', '')).lower()
                
                if building_id in emergency_building_ids:
                    emergency_buildings.append(building_id)
                elif any(emergency_type.lower() in building_type for emergency_type in emergency_building_types):
                    emergency_buildings.append(building_id)
        
        logger.info(f"    Emergency buildings: {len(emergency_buildings)}")
        
        # Find safe zones
        safe_zones = find_safe_zones(buildings_df, scenario)
        logger.info(f"    Safe zones: {len(safe_zones)}")
        
        # Find evacuation routes
        k = evacuation_params.get('min_alternative_routes', 2)
        evacuation_routes = find_evacuation_routes(G, emergency_buildings, safe_zones, k=k)
        logger.info(f"    Evacuation routes calculated for {len(evacuation_routes)} buildings")
        
        # Calculate evacuation times with panic factor
        walking_speed_multiplier = scenario.get('walking_speed_multiplier', 1.0)
        panic_factor = scenario.get('panic_factor', 1.0)
        base_walking_speed = 1.4 * walking_speed_multiplier  # m/s
        
        for building_id, routes in evacuation_routes.items():
            if routes['primary_route']:
                evacuation_time = calculate_evacuation_time(
                    G, routes['primary_route'], 
                    walking_speed=base_walking_speed,
                    panic_factor=panic_factor
                )
                routes['evacuation_time_minutes'] = evacuation_time
        
        # Analyze safety
        safety_analysis = analyze_safety(evacuation_routes, min_alternative_routes=k)
        all_safety_analysis[scenario_id] = safety_analysis
        
        # Calculate total evacuation time (max of all buildings)
        max_evacuation_time = max(
            [r.get('evacuation_time_minutes', np.inf) for r in evacuation_routes.values()],
            default=0.0
        )
        
        # Store scenario analysis
        emergency_analysis[scenario_id] = {
            'name': scenario_name,
            'emergency_buildings': emergency_buildings,
            'safe_zones': safe_zones,
            'evacuation_routes': {
                building_id: {
                    'primary_route': routes['primary_route'],
                    'alternative_routes': routes['alternative_routes'],
                    'evacuation_time_minutes': routes.get('evacuation_time_minutes', np.inf),
                    'route_capacity': routes.get('route_capacity', 0),
                    'safety_score': routes.get('safety_score', 0.0)
                }
                for building_id, routes in evacuation_routes.items()
            },
            'total_evacuation_time_minutes': max_evacuation_time,
            'safety_analysis': safety_analysis
        }
        
        logger.info(f"    Total evacuation time: {max_evacuation_time:.2f} minutes")
        logger.info(f"    Buildings at risk: {sum(1 for s in safety_analysis.values() if s.get('is_at_risk', False))}")
    
    
    # Save results
    logger.info(f"Saving emergency scenarios analysis to: {output_json_path}")
    output_dir = Path(output_json_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to JSON-serializable format
    emergency_analysis_serializable = {}
    for scenario_id, analysis in emergency_analysis.items():
        emergency_analysis_serializable[scenario_id] = {
            'name': analysis['name'],
            'emergency_buildings': analysis['emergency_buildings'],
            'safe_zones': analysis['safe_zones'],
            'evacuation_routes': {
                building_id: {
                    'primary_route': routes['primary_route'],
                    'alternative_routes': routes['alternative_routes'],
                    'evacuation_time_minutes': float(routes['evacuation_time_minutes']) if routes['evacuation_time_minutes'] != np.inf else None,
                    'route_capacity': int(routes['route_capacity']),
                    'safety_score': float(routes['safety_score'])
                }
                for building_id, routes in analysis['evacuation_routes'].items()
            },
            'total_evacuation_time_minutes': float(analysis['total_evacuation_time_minutes']) if analysis['total_evacuation_time_minutes'] != np.inf else None,
            'safety_analysis': {
                building_id: {
                    'evacuation_routes_count': int(safety['evacuation_routes_count']),
                    'min_evacuation_time_minutes': float(safety['min_evacuation_time_minutes']) if safety['min_evacuation_time_minutes'] != np.inf else None,
                    'safety_score': float(safety['safety_score']),
                    'critical_paths': int(safety['critical_paths']),
                    'is_at_risk': bool(safety['is_at_risk']),
                    'recommendation': safety['recommendation']
                }
                for building_id, safety in analysis['safety_analysis'].items()
            }
        }
    
    with open(output_json_path, 'w') as f:
        json.dump(emergency_analysis_serializable, f, indent=2)
    logger.info(f"Successfully saved emergency scenarios analysis to {output_json_path}")
    
    
    safety_metrics_path = output_json_path.replace('.json', '_safety_metrics.json')
    with open(safety_metrics_path, 'w') as f:
        json.dump(all_safety_analysis, f, indent=2)
    logger.info(f"Saved safety metrics to: {safety_metrics_path}")
    
    
    evacuation_analysis_path = output_json_path.replace('.json', '_evacuation_analysis.csv')
    evacuation_analysis_data = []
    for scenario_id, analysis in emergency_analysis.items():
        for building_id, safety in analysis['safety_analysis'].items():
            evacuation_analysis_data.append({
                'scenario_id': scenario_id,
                'scenario_name': analysis['name'],
                'building_id': building_id,
                'evacuation_routes_count': safety['evacuation_routes_count'],
                'min_evacuation_time_minutes': safety['min_evacuation_time_minutes'],
                'safety_score': safety['safety_score'],
                'is_at_risk': safety['is_at_risk'],
                'recommendation': safety['recommendation']
            })
    
    evacuation_analysis_df = pd.DataFrame(evacuation_analysis_data)
    evacuation_analysis_df.to_csv(evacuation_analysis_path, index=False)
    logger.info(f"Saved evacuation analysis to: {evacuation_analysis_path}")
    
    
    logger.info("\nEmergency Scenarios Summary:")
    for scenario_id, analysis in emergency_analysis.items():
        logger.info(f"  {analysis['name']}:")
        logger.info(f"    Emergency buildings: {len(analysis['emergency_buildings'])}")
        logger.info(f"    Safe zones: {len(analysis['safe_zones'])}")
        logger.info(f"    Total evacuation time: {analysis['total_evacuation_time_minutes']:.2f} minutes")
        at_risk = sum(1 for s in analysis['safety_analysis'].values() if s.get('is_at_risk', False))
        logger.info(f"    Buildings at risk: {at_risk}")
        avg_safety = np.mean([s['safety_score'] for s in analysis['safety_analysis'].values()])
        logger.info(f"    Average safety score: {avg_safety:.2f}")
    
    logger.info("=" * 60)
    logger.info("Step 7 completed!")
    logger.info("=" * 60)
    
    return emergency_analysis


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Analyze emergency evacuation scenarios'
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
        default='data/processed/emergency_scenarios.json',
        help='Path to output JSON file'
    )
    parser.add_argument(
        '--scenarios',
        type=str,
        default='config/emergency_scenarios.json',
        help='Path to emergency scenarios JSON file'
    )
    
    args = parser.parse_args()
    
    try:
        analyze_emergency_scenarios(
            args.buildings,
            args.graph,
            args.output,
            args.scenarios
        )
    except Exception as e:
        logger.error(f"Error in Adım 7: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

