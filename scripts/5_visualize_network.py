"""
Usage:
    python scripts/5_visualize_network.py --buildings data/processed/all_buildings.csv --graph data/processed/building_network_graph.pkl --metrics data/processed/network_metrics.csv --accessibility data/processed/accessibility_metrics.csv --output data/output/
"""

import argparse
import sys
import os
import json
from pathlib import Path
import pandas as pd
import pickle
import networkx as nx
import logging


sys.path.append(str(Path(__file__).parent.parent))

from utils.visualization_utils import (
    plot_network_graph,
    plot_accessibility_heatmap,
    plot_optimal_paths
)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def visualize_network(
    buildings_csv_path: str,
    graph_pkl_path: str,
    metrics_csv_path: str = None,
    accessibility_csv_path: str = None,
    paths_json_path: str = None,
    output_dir: str = 'data/output/'
):
    """
    Create visualizations for network analysis.
    
    Args:
        buildings_csv_path: Path to buildings CSV file
        graph_pkl_path: Path to network graph pickle file
        metrics_csv_path: Path to network metrics CSV file (optional)
        accessibility_csv_path: Path to accessibility metrics CSV file (optional)
        paths_json_path: Path to paths JSON file (optional)
        output_dir: Output directory for visualization files
    """
    logger.info("=" * 60)
    logger.info("Step 5: Network Visualization")
    logger.info("=" * 60)
    
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    
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
    
    # Load metrics (optional)
    metrics_df = None
    if metrics_csv_path and os.path.exists(metrics_csv_path):
        logger.info(f"Loading metrics from: {metrics_csv_path}")
        metrics_df = pd.read_csv(metrics_csv_path)
        logger.info(f"Loaded metrics for {len(metrics_df)} buildings")
    
    # Load accessibility (optional)
    accessibility_df = None
    if accessibility_csv_path and os.path.exists(accessibility_csv_path):
        logger.info(f"Loading accessibility from: {accessibility_csv_path}")
        accessibility_df = pd.read_csv(accessibility_csv_path)
        logger.info(f"Loaded accessibility for {len(accessibility_df)} buildings")
    
    # 1. Network Graph Visualization
    logger.info("\n1. Creating network graph visualization...")
    network_graph_path = output_path / 'network_graph.png'
    try:
        plot_network_graph(
            G,
            buildings_df,
            metrics_df=metrics_df,
            output_path=str(network_graph_path),
            node_size_column='degree',
            node_color_column='betweenness_centrality',
            figsize=(12, 10),
            dpi=300
        )
        logger.info(f"✓ Network graph saved to: {network_graph_path}")
    except Exception as e:
        logger.error(f"Error creating network graph: {e}")
        import traceback
        traceback.print_exc()
    
    # 2. Accessibility Heatmap
    if accessibility_df is not None:
        logger.info("\n2. Creating accessibility heatmap...")
        heatmap_path = output_path / 'accessibility_heatmap.png'
        try:
            plot_accessibility_heatmap(
                buildings_df,
                accessibility_df,
                output_path=str(heatmap_path),
                color_column='weighted_accessibility',
                figsize=(12, 10),
                dpi=300
            )
            logger.info(f"✓ Accessibility heatmap saved to: {heatmap_path}")
        except Exception as e:
            logger.error(f"Error creating accessibility heatmap: {e}")
            import traceback
            traceback.print_exc()
    else:
        logger.warning("Accessibility data not provided, skipping heatmap")
    
    # 3. Optimal Paths Visualization
    if paths_json_path and os.path.exists(paths_json_path):
        logger.info("\n3. Creating optimal paths visualization...")
        paths_path = output_path / 'optimal_paths.png'
        try:
            with open(paths_json_path, 'r') as f:
                paths_data = json.load(f)
            
            plot_optimal_paths(
                G,
                buildings_df,
                paths_data,
                output_path=str(paths_path),
                max_paths=10,
                figsize=(12, 10),
                dpi=300
            )
            logger.info(f"✓ Optimal paths saved to: {paths_path}")
        except Exception as e:
            logger.error(f"Error creating optimal paths visualization: {e}")
            import traceback
            traceback.print_exc()
    else:
        logger.info("Paths data not provided, skipping optimal paths visualization")
    
    # 4. Additional visualizations (if metrics available)
    if metrics_df is not None:
        # Degree distribution
        logger.info("\n4. Creating degree distribution plot...")
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(metrics_df['degree'], bins=20, edgecolor='black', alpha=0.7)
            ax.set_xlabel('Degree', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.set_title('Degree Distribution', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            degree_dist_path = output_path / 'degree_distribution.png'
            plt.savefig(degree_dist_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"✓ Degree distribution saved to: {degree_dist_path}")
        except Exception as e:
            logger.warning(f"Error creating degree distribution: {e}")
        
        
        logger.info("\n5. Creating centrality comparison plot...")
        try:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Degree centrality
            axes[0, 0].scatter(metrics_df['degree'], metrics_df['degree_centrality'], alpha=0.7)
            axes[0, 0].set_xlabel('Degree', fontsize=10)
            axes[0, 0].set_ylabel('Degree Centrality', fontsize=10)
            axes[0, 0].set_title('Degree vs Degree Centrality', fontsize=12)
            axes[0, 0].grid(True, alpha=0.3)
            
            # Betweenness vs Closeness
            axes[0, 1].scatter(metrics_df['betweenness_centrality'], metrics_df['closeness_centrality'], alpha=0.7)
            axes[0, 1].set_xlabel('Betweenness Centrality', fontsize=10)
            axes[0, 1].set_ylabel('Closeness Centrality', fontsize=10)
            axes[0, 1].set_title('Betweenness vs Closeness', fontsize=12)
            axes[0, 1].grid(True, alpha=0.3)
            
            # PageRank
            axes[1, 0].bar(range(len(metrics_df)), metrics_df['pagerank'], alpha=0.7)
            axes[1, 0].set_xlabel('Building Index', fontsize=10)
            axes[1, 0].set_ylabel('PageRank', fontsize=10)
            axes[1, 0].set_title('PageRank Values', fontsize=12)
            axes[1, 0].grid(True, alpha=0.3)
            
            # All centralities comparison
            if len(metrics_df) > 1:
                metrics_normalized = metrics_df[['degree_centrality', 'betweenness_centrality', 'closeness_centrality', 'pagerank']].copy()
                for col in metrics_normalized.columns:
                    min_val = metrics_normalized[col].min()
                    max_val = metrics_normalized[col].max()
                    if max_val > min_val:
                        metrics_normalized[col] = (metrics_normalized[col] - min_val) / (max_val - min_val)
                
                axes[1, 1].plot(metrics_normalized.index, metrics_normalized['degree_centrality'], label='Degree', marker='o')
                axes[1, 1].plot(metrics_normalized.index, metrics_normalized['betweenness_centrality'], label='Betweenness', marker='s')
                axes[1, 1].plot(metrics_normalized.index, metrics_normalized['closeness_centrality'], label='Closeness', marker='^')
                axes[1, 1].plot(metrics_normalized.index, metrics_normalized['pagerank'], label='PageRank', marker='d')
                axes[1, 1].set_xlabel('Building Index', fontsize=10)
                axes[1, 1].set_ylabel('Normalized Centrality', fontsize=10)
                axes[1, 1].set_title('Centrality Comparison', fontsize=12)
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            centrality_comp_path = output_path / 'centrality_comparison.png'
            plt.savefig(centrality_comp_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"✓ Centrality comparison saved to: {centrality_comp_path}")
        except Exception as e:
            logger.warning(f"Error creating centrality comparison: {e}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Adım 5 tamamlandı!")
    logger.info("=" * 60)
    logger.info(f"All visualizations saved to: {output_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Create network visualizations'
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
        '--metrics',
        type=str,
        default=None,
        help='Path to network metrics CSV file (optional)'
    )
    parser.add_argument(
        '--accessibility',
        type=str,
        default=None,
        help='Path to accessibility metrics CSV file (optional)'
    )
    parser.add_argument(
        '--paths',
        type=str,
        default=None,
        help='Path to paths JSON file (optional)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/output/',
        help='Output directory for visualization files'
    )
    
    args = parser.parse_args()
    
    try:
        visualize_network(
            args.buildings,
            args.graph,
            args.metrics,
            args.accessibility,
            args.paths,
            args.output
        )
    except Exception as e:
        logger.error(f"Error in Adım 5: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

