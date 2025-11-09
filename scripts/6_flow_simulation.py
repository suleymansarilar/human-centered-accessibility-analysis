"""
Flow Simulation Script

Usage:
    python scripts/6_flow_simulation.py --buildings data/processed/all_buildings.csv --graph data/processed/building_network_graph.pkl --output data/processed/flow_simulation.csv
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

from utils.flow_simulation import (
    load_flow_scenarios,
    create_od_matrix,
    simulate_flow,
    calculate_congestion,
    calculate_travel_time_with_congestion,
    identify_bottlenecks
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def analyze_flow_simulation(
    buildings_csv_path: str,
    graph_pkl_path: str,
    output_csv_path: str,
    scenario_file: str = 'config/flow_scenarios.json'
):
    """
    Analyze flow simulation for different scenarios.
    
    Args:
        buildings_csv_path: Path to buildings CSV file
        graph_pkl_path: Path to network graph pickle file
        output_csv_path: Path to output CSV file
        scenario_file: Path to flow scenarios JSON file
    """
    logger.info("=" * 60)
    logger.info("Step 6: Flow Simulation")
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
    
    
    logger.info(f"Loading flow scenarios from: {scenario_file}")
    scenarios_data = load_flow_scenarios(scenario_file)
    flow_scenarios = scenarios_data.get('flow_scenarios', [])
    flow_model = scenarios_data.get('flow_model', {})
    logger.info(f"Loaded {len(flow_scenarios)} flow scenarios")
    
    
    logger.info("Simulating flow for each scenario...")
    flow_simulation_data = []
    all_bottlenecks = {}
    
    for scenario in flow_scenarios:
        scenario_id = scenario['scenario_id']
        scenario_name = scenario['name']
        
        logger.info(f"  Processing scenario: {scenario_name} ({scenario_id})")
        
        # Create OD matrix
        od_matrix = create_od_matrix(buildings_df, scenario)
        logger.info(f"    Created OD matrix with {len(od_matrix)} origin-destination pairs")
        
        # Simulate flow
        distribution = flow_model.get('distribution', 'equal_split')
        edge_flows = simulate_flow(G, od_matrix, scenario, distribution=distribution)
        logger.info(f"    Simulated flow on {len([f for f in edge_flows.values() if f > 0])} edges")
        
        # Calculate congestion
        congestion_model = flow_model.get('congestion_model', 'linear')
        congestion_levels = calculate_congestion(G, edge_flows, scenario, congestion_model=congestion_model)
        
        # Calculate travel time with congestion
        base_walking_speed = flow_model.get('base_walking_speed_mps', 1.4)
        travel_times = calculate_travel_time_with_congestion(
            G, edge_flows, congestion_levels, base_walking_speed=base_walking_speed
        )
        
        # Identify bottlenecks
        bottlenecks = identify_bottlenecks(edge_flows, congestion_levels, threshold=0.6)
        all_bottlenecks[scenario_id] = bottlenecks
        logger.info(f"    Identified {len(bottlenecks)} bottlenecks")
        
        # Store results
        for (u, v), flow_volume in edge_flows.items():
            if flow_volume > 0 and (u, v) in G.edges():
                congestion = congestion_levels.get((u, v), 0.0)
                travel_time = travel_times.get((u, v), 0.0)
                
                flow_simulation_data.append({
                    'edge_id': f"{u}->{v}",
                    'source': u,
                    'target': v,
                    'scenario_id': scenario_id,
                    'scenario_name': scenario_name,
                    'flow_volume': flow_volume,
                    'congestion_level': congestion,
                    'travel_time_seconds': travel_time,
                    'travel_time_minutes': travel_time / 60.0
                })
        
        logger.info(f"    Total flow volume: {sum(edge_flows.values()) / 2:.2f} (undirected)")  # Divide by 2 for undirected
        logger.info(f"    Average congestion: {np.mean(list(congestion_levels.values())):.2f}")
        logger.info(f"    Max congestion: {np.max(list(congestion_levels.values())):.2f}")
    
    
    flow_simulation_df = pd.DataFrame(flow_simulation_data)
    
    # Save results
    logger.info(f"Saving flow simulation data to: {output_csv_path}")
    output_dir = Path(output_csv_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    flow_simulation_df.to_csv(output_csv_path, index=False)
    logger.info(f"Successfully saved flow simulation data to {output_csv_path}")
    
    
    bottlenecks_path = output_csv_path.replace('.csv', '_bottlenecks.json')
    with open(bottlenecks_path, 'w') as f:
        json.dump(all_bottlenecks, f, indent=2)
    logger.info(f"Saved bottlenecks to: {bottlenecks_path}")
    
    
    logger.info("\nFlow Simulation Summary:")
    for scenario in flow_scenarios:
        scenario_id = scenario['scenario_id']
        scenario_name = scenario['name']
        scenario_data = flow_simulation_df[flow_simulation_df['scenario_id'] == scenario_id]
        
        if not scenario_data.empty:
            logger.info(f"  {scenario_name}:")
            logger.info(f"    Total flow volume: {scenario_data['flow_volume'].sum():.2f}")
            logger.info(f"    Average congestion: {scenario_data['congestion_level'].mean():.2f}")
            logger.info(f"    Max congestion: {scenario_data['congestion_level'].max():.2f}")
            logger.info(f"    Bottlenecks: {len(all_bottlenecks.get(scenario_id, []))}")
    
    logger.info("=" * 60)
    logger.info("Step 6 completed!")
    logger.info("=" * 60)
    
    return flow_simulation_df


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Simulate human flow patterns between buildings'
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
        default='data/processed/flow_simulation.csv',
        help='Path to output CSV file'
    )
    parser.add_argument(
        '--scenarios',
        type=str,
        default='config/flow_scenarios.json',
        help='Path to flow scenarios JSON file'
    )
    
    args = parser.parse_args()
    
    try:
        analyze_flow_simulation(
            args.buildings,
            args.graph,
            args.output,
            args.scenarios
        )
    except Exception as e:
        logger.error(f"Error in Adım 6: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

