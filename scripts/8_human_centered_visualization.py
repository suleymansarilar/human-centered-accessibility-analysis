"""
Human-Centered Visualization Script

Usage:
    python scripts/8_human_centered_visualization.py --buildings data/processed/all_buildings.csv --graph data/processed/building_network_graph.pkl --user-accessibility data/processed/user_accessibility.csv --flow-simulation data/processed/flow_simulation.csv --emergency-scenarios data/processed/emergency_scenarios.json --output data/output/
"""

import argparse
import sys
import os
from pathlib import Path
import pandas as pd
import pickle
import networkx as nx
import logging

sys.path.append(str(Path(__file__).parent.parent))

from utils.human_centered_viz import (
    plot_user_accessibility_heatmap,
    plot_user_accessibility_comparison,
    plot_accessibility_disparities,
    plot_flow_simulation,
    plot_emergency_evacuation_routes,
    plot_emergency_safety_analysis
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_human_centered_visualizations(
    buildings_csv_path: str,
    graph_pkl_path: str,
    user_accessibility_csv_path: str,
    flow_simulation_csv_path: str,
    emergency_scenarios_json_path: str,
    disparities_json_path: str,
    emergency_analysis_csv_path: str,
    output_dir: str
):
    """
    Create human-centered visualizations.
    
    Args:
        buildings_csv_path: Path to buildings CSV file
        graph_pkl_path: Path to network graph pickle file
        user_accessibility_csv_path: Path to user accessibility CSV file
        flow_simulation_csv_path: Path to flow simulation CSV file
        emergency_scenarios_json_path: Path to emergency scenarios JSON file
        disparities_json_path: Path to disparities JSON file
        emergency_analysis_csv_path: Path to emergency analysis CSV file
        output_dir: Output directory for visualizations
    """
    logger.info("=" * 60)
    logger.info("Step 8: Human-Centered Visualization")
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
    
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    
    logger.info("\n1. Creating user accessibility visualizations...")
    
    if os.path.exists(user_accessibility_csv_path):
        user_accessibility_df = pd.read_csv(user_accessibility_csv_path)
        logger.info(f"Loaded user accessibility data: {len(user_accessibility_df)} records")
        
        # User accessibility heatmap
        heatmap_path = output_path / 'user_accessibility_heatmap.png'
        plot_user_accessibility_heatmap(buildings_df, user_accessibility_df, str(heatmap_path))
        
        # User accessibility comparison
        comparison_path = output_path / 'user_accessibility_comparison.png'
        plot_user_accessibility_comparison(user_accessibility_df, str(comparison_path))
    else:
        logger.warning(f"User accessibility file not found: {user_accessibility_csv_path}")
    
    
    # Skip disparities visualization (combined with comparison)
    # logger.info("\n2. Creating accessibility disparities visualization...")
    
    logger.info("\n3. Creating flow simulation visualizations...")
    
    if os.path.exists(flow_simulation_csv_path):
        flow_simulation_df = pd.read_csv(flow_simulation_csv_path)
        logger.info(f"Loaded flow simulation data: {len(flow_simulation_df)} records")
        
        # Only create visualization for rush hour (most important scenario)
        if 'rush_hour_morning' in flow_simulation_df['scenario_id'].values:
            flow_viz_path = output_path / 'flow_simulation_rush_hour.png'
            plot_flow_simulation(buildings_df, G, flow_simulation_df, 'rush_hour_morning', str(flow_viz_path))
            logger.info(f"  Created flow visualization for: Rush Hour")
        else:
            # If rush hour not available, use first scenario
            scenarios = flow_simulation_df['scenario_id'].unique()
            if len(scenarios) > 0:
                scenario_id = scenarios[0]
                flow_viz_path = output_path / f'flow_simulation.png'
                plot_flow_simulation(buildings_df, G, flow_simulation_df, scenario_id, str(flow_viz_path))
                logger.info(f"  Created flow visualization for: {scenario_id}")
    else:
        logger.warning(f"Flow simulation file not found: {flow_simulation_csv_path}")
    
    
    logger.info("\n4. Creating emergency evacuation route visualizations...")
    
    if os.path.exists(emergency_scenarios_json_path):
        import json
        with open(emergency_scenarios_json_path, 'r') as f:
            emergency_scenarios = json.load(f)
        
        # Only create visualization for fire evacuation (most common scenario)
        if 'fire_evacuation' in emergency_scenarios:
            evacuation_path = output_path / 'emergency_evacuation_fire.png'
            plot_emergency_evacuation_routes(
                buildings_df, G, emergency_scenarios_json_path, 'fire_evacuation', str(evacuation_path)
            )
            logger.info(f"  Created evacuation route visualization for: Fire Evacuation")
        else:
            # If fire not available, use first scenario
            scenario_ids = list(emergency_scenarios.keys())
            if len(scenario_ids) > 0:
                scenario_id = scenario_ids[0]
                evacuation_path = output_path / 'emergency_evacuation.png'
                plot_emergency_evacuation_routes(
                    buildings_df, G, emergency_scenarios_json_path, scenario_id, str(evacuation_path)
                )
                logger.info(f"  Created evacuation route visualization for: {scenario_id}")
    else:
        logger.warning(f"Emergency scenarios file not found: {emergency_scenarios_json_path}")
    
    
    # Skip safety analysis visualization (simplified output)
    # logger.info("\n5. Creating emergency safety analysis visualization...")
    
    
    logger.info("\n" + "=" * 60)
    logger.info("Step 8 completed!")
    logger.info("=" * 60)
    logger.info(f"All visualizations saved to: {output_dir}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Create human-centered visualizations'
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
        '--user-accessibility',
        type=str,
        default='data/processed/user_accessibility.csv',
        help='Path to user accessibility CSV file'
    )
    parser.add_argument(
        '--flow-simulation',
        type=str,
        default='data/processed/flow_simulation.csv',
        help='Path to flow simulation CSV file'
    )
    parser.add_argument(
        '--emergency-scenarios',
        type=str,
        default='data/processed/emergency_scenarios.json',
        help='Path to emergency scenarios JSON file'
    )
    parser.add_argument(
        '--disparities',
        type=str,
        default='data/processed/user_accessibility_disparities.json',
        help='Path to disparities JSON file'
    )
    parser.add_argument(
        '--emergency-analysis',
        type=str,
        default='data/processed/emergency_scenarios_evacuation_analysis.csv',
        help='Path to emergency analysis CSV file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/output',
        help='Output directory for visualizations'
    )
    
    args = parser.parse_args()
    
    try:
        create_human_centered_visualizations(
            args.buildings,
            args.graph,
            args.user_accessibility,
            args.flow_simulation,
            args.emergency_scenarios,
            args.disparities,
            args.emergency_analysis,
            args.output
        )
    except Exception as e:
        logger.error(f"Error in Adım 8: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

