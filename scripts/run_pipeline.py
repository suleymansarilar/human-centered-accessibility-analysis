"""
Complete Pipeline Runner - Human-Centered Accessibility & Flow Analysis

Usage:
    python scripts/run_pipeline.py --input data/input/*.gml --output data/output/ --threshold 500
"""

import argparse
import sys
import os
from pathlib import Path
import logging
import glob


sys.path.append(str(Path(__file__).parent.parent))


import importlib.util

def load_module(module_path, module_name):
    """Load a module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


extract_module = load_module(
    Path(__file__).parent / "1_extract_buildings.py",
    "extract_buildings"
)
network_module = load_module(
    Path(__file__).parent / "2_build_network.py",
    "build_network"
)
analyze_module = load_module(
    Path(__file__).parent / "3_analyze_network.py",
    "analyze_network"
)
accessibility_module = load_module(
    Path(__file__).parent / "4_calculate_accessibility.py",
    "calculate_accessibility"
)
visualize_module = load_module(
    Path(__file__).parent / "5_visualize_network.py",
    "visualize_network"
)

extract_buildings = extract_module.extract_buildings
build_network = network_module.build_network
analyze_network = analyze_module.analyze_network
calculate_accessibility = accessibility_module.calculate_accessibility
visualize_network = visualize_module.visualize_network


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_pipeline(
    input_pattern: str,
    output_dir: str = 'data/output/',
    threshold: float = 500.0,
    distance_threshold: float = 500.0
):
    """
    Run the complete human-centered accessibility analysis pipeline.
    
    Args:
        input_pattern: Input GML file pattern (e.g., 'data/input/*.gml')
        output_dir: Output directory
        threshold: Network construction distance threshold (meters)
        distance_threshold: Accessibility distance threshold (meters)
    """
    logger.info("=" * 60)
    logger.info("Human-Centered Accessibility & Flow Analysis Pipeline")
    logger.info("=" * 60)
    logger.info("Phase 1: Building Extraction & Network Analysis")
    logger.info("Phase 2: User Profiles, Flow Simulation, Emergency Scenarios")
    logger.info("Phase 3: Human-Centered Visualization")
    logger.info("=" * 60)
    
    # Create directories
    output_path = Path(output_dir)
    processed_dir = output_path.parent / 'processed'
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find GML files
    gml_files = glob.glob(input_pattern)
    if not gml_files:
        logger.error(f"No GML files found matching pattern: {input_pattern}")
        return
    
    logger.info(f"Found {len(gml_files)} GML files")
    
    # Step 1: Extract buildings
    logger.info("\n" + "=" * 60)
    logger.info("Step 1: Extract Buildings")
    logger.info("=" * 60)
    
    all_buildings = []
    for gml_file in gml_files:
        logger.info(f"Processing: {gml_file}")
        output_csv = processed_dir / f"{Path(gml_file).stem}_buildings.csv"
        try:
            df = extract_buildings(str(gml_file), str(output_csv))
            if df is not None and not df.empty:
                all_buildings.append(df)
        except Exception as e:
            logger.error(f"Error processing {gml_file}: {e}")
            continue
    
    if not all_buildings:
        logger.error("No buildings extracted. Pipeline stopped.")
        return
    
    # Combine buildings
    import pandas as pd
    combined_df = pd.concat(all_buildings, ignore_index=True)
    combined_csv = processed_dir / "all_buildings.csv"
    combined_df.to_csv(combined_csv, index=False)
    logger.info(f"Combined {len(combined_df)} buildings")
    
    # Combine footprints
    from scripts.combine_footprints import combine_footprints
    footprints_pkl = processed_dir / "all_buildings_footprints.pkl"
    combine_footprints(str(processed_dir), str(footprints_pkl))
    
    
    logger.info("\n" + "=" * 60)
    logger.info("Step 2: Build Network Graph")
    logger.info("=" * 60)
    
    graph_pkl = processed_dir / "building_network_graph.pkl"
    try:
        G = build_network(
            str(combined_csv),
            str(graph_pkl),
            distance_threshold=threshold,
            method='distance',
            use_edge_distance=False
        )
        logger.info(f"Network graph created: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    except Exception as e:
        logger.error(f"Error building network: {e}")
        return
    
    
    logger.info("\n" + "=" * 60)
    logger.info("Step 3: Analyze Network")
    logger.info("=" * 60)
    
    metrics_csv = processed_dir / "network_metrics.csv"
    paths_json = processed_dir / "network_paths.json"
    try:
        metrics_df = analyze_network(
            str(graph_pkl),
            str(metrics_csv),
            str(paths_json)
        )
        logger.info(f"Network analysis completed: {len(metrics_df)} buildings analyzed")
    except Exception as e:
        logger.error(f"Error analyzing network: {e}")
        metrics_csv = None
        paths_json = None
    
    # Calculate accessibility
    logger.info("\n" + "=" * 60)
    logger.info("Step 4: Calculate Human-Centered Accessibility")
    logger.info("=" * 60)
    
    accessibility_csv = processed_dir / "accessibility_metrics.csv"
    try:
        accessibility_df = calculate_accessibility(
            str(combined_csv),
            str(graph_pkl),
            str(accessibility_csv),
            distance_threshold=distance_threshold
        )
        logger.info(f"Accessibility calculation completed: {len(accessibility_df)} buildings analyzed")
    except Exception as e:
        logger.error(f"Error calculating accessibility: {e}")
        accessibility_csv = None
    
    # Visualize
    logger.info("\n" + "=" * 60)
    logger.info("Step 5: Visualize Results")
    logger.info("=" * 60)
    
    try:
        visualize_network(
            str(combined_csv),
            str(graph_pkl),
            str(metrics_csv) if metrics_csv else None,
            str(accessibility_csv) if accessibility_csv else None,
            str(paths_json) if paths_json else None,
            str(output_path)
        )
        logger.info("Visualization completed")
    except Exception as e:
        logger.error(f"Error creating visualizations: {e}")
    
    all_buildings_csv = str(combined_csv)
    
    # Phase 2: User Profiles, Flow Simulation, Emergency Scenarios
    logger.info("\n" + "=" * 60)
    logger.info("Phase 2: User Profiles, Flow Simulation, Emergency Scenarios")
    logger.info("=" * 60)
    
    # Step 5: User Profile Analysis
    logger.info("\n" + "=" * 60)
    logger.info("Step 5: User Profile Analysis")
    logger.info("=" * 60)
    try:
        step5_module = load_module(
            Path(__file__).parent / "5_user_profiles.py",
            "user_profiles"
        )
        step5_module.analyze_user_profiles(
            buildings_csv_path=all_buildings_csv,
            graph_pkl_path=str(graph_pkl),
            output_csv_path=str(processed_dir / 'user_accessibility.csv'),
            profile_file='config/user_profiles.json',
            distance_threshold=threshold
        )
    except Exception as e:
        logger.error(f"Error in Step 5: {e}")
        import traceback
        traceback.print_exc()
    
    # Step 6: Flow Simulation
    logger.info("\n" + "=" * 60)
    logger.info("Step 6: Flow Simulation")
    logger.info("=" * 60)
    try:
        step6_module = load_module(
            Path(__file__).parent / "6_flow_simulation.py",
            "flow_simulation"
        )
        step6_module.analyze_flow_simulation(
            buildings_csv_path=all_buildings_csv,
            graph_pkl_path=str(graph_pkl),
            output_csv_path=str(processed_dir / 'flow_simulation.csv'),
            scenario_file='config/flow_scenarios.json'
        )
    except Exception as e:
        logger.error(f"Error in Step 6: {e}")
        import traceback
        traceback.print_exc()
    
    # Step 7: Emergency Scenarios
    logger.info("\n" + "=" * 60)
    logger.info("Step 7: Emergency Scenarios")
    logger.info("=" * 60)
    try:
        step7_module = load_module(
            Path(__file__).parent / "7_emergency_scenarios.py",
            "emergency_scenarios"
        )
        step7_module.analyze_emergency_scenarios(
            buildings_csv_path=all_buildings_csv,
            graph_pkl_path=str(graph_pkl),
            output_json_path=str(processed_dir / 'emergency_scenarios.json'),
            scenario_file='config/emergency_scenarios.json'
        )
    except Exception as e:
        logger.error(f"Error in Step 7: {e}")
        import traceback
        traceback.print_exc()
    
    # Phase 3: Human-Centered Visualization
    logger.info("\n" + "=" * 60)
    logger.info("Phase 3: Human-Centered Visualization")
    logger.info("=" * 60)
    
    # Step 8: Human-Centered Visualization
    logger.info("\n" + "=" * 60)
    logger.info("Step 8: Human-Centered Visualization")
    logger.info("=" * 60)
    try:
        step8_module = load_module(
            Path(__file__).parent / "8_human_centered_visualization.py",
            "human_centered_visualization"
        )
        step8_module.create_human_centered_visualizations(
            buildings_csv_path=all_buildings_csv,
            graph_pkl_path=str(graph_pkl),
            user_accessibility_csv_path=str(processed_dir / 'user_accessibility.csv'),
            flow_simulation_csv_path=str(processed_dir / 'flow_simulation.csv'),
            emergency_scenarios_json_path=str(processed_dir / 'emergency_scenarios.json'),
            disparities_json_path=str(processed_dir / 'user_accessibility_disparities.json'),
            emergency_analysis_csv_path=str(processed_dir / 'emergency_scenarios_evacuation_analysis.csv'),
            output_dir=str(output_path)
        )
    except Exception as e:
        logger.error(f"Error in Step 8: {e}")
        import traceback
        traceback.print_exc()
    
    logger.info("\n" + "=" * 60)
    logger.info("Pipeline completed successfully!")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {output_path}")
    logger.info("\nPhase 1: Building Extraction & Network Analysis - Complete")
    logger.info("Phase 2: User Profiles, Flow Simulation, Emergency Scenarios - Complete")
    logger.info("Phase 3: Human-Centered Visualization - Complete")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Run complete human-centered accessibility analysis pipeline'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input GML file pattern (e.g., "data/input/*.gml")'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/output/',
        help='Output directory'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=500.0,
        help='Network construction distance threshold (meters)'
    )
    parser.add_argument(
        '--distance-threshold',
        type=float,
        default=500.0,
        help='Accessibility distance threshold (meters)'
    )
    
    args = parser.parse_args()
    
    try:
        run_pipeline(
            args.input,
            args.output,
            args.threshold,
            args.distance_threshold
        )
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

