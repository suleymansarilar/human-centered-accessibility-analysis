"""
Visualization Utilities

"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def plot_network_graph(
    G: nx.Graph,
    buildings_df: pd.DataFrame,
    metrics_df: Optional[pd.DataFrame] = None,
    output_path: str = 'network_graph.png',
    node_size_column: str = 'degree',
    node_color_column: str = 'betweenness_centrality',
    figsize: Tuple[int, int] = (12, 10),
    dpi: int = 300
):
    """
    Plot network graph with buildings as nodes.
    
    Args:
        G: NetworkX graph
        buildings_df: DataFrame with building data (centroid coordinates)
        metrics_df: DataFrame with network metrics (optional)
        output_path: Path to save the plot
        node_size_column: Column name for node size
        node_color_column: Column name for node color
        figsize: Figure size
        dpi: DPI for output image
    """
    logger.info(f"Plotting network graph...")
    
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    
    pos = {}
    for _, row in buildings_df.iterrows():
        building_id = row['building_id']
        if building_id in G.nodes():
            pos[building_id] = (row['centroid_lon'], row['centroid_lat'])
    
    
    if metrics_df is not None:
        
        if node_size_column in metrics_df.columns:
            node_sizes = metrics_df.set_index('building_id')[node_size_column]
            # Normalize to reasonable size range (50-500)
            min_size = node_sizes.min()
            max_size = node_sizes.max()
            if max_size > min_size:
                node_sizes = 50 + (node_sizes - min_size) / (max_size - min_size) * 450
            else:
                node_sizes = pd.Series([250] * len(node_sizes), index=node_sizes.index)
        else:
            node_sizes = [300] * len(G.nodes())
        
        
        if node_color_column in metrics_df.columns:
            node_colors = metrics_df.set_index('building_id')[node_color_column]
            
            min_color = node_colors.min()
            max_color = node_colors.max()
            if max_color > min_color:
                node_colors = (node_colors - min_color) / (max_color - min_color)
            else:
                node_colors = pd.Series([0.5] * len(node_colors), index=node_colors.index)
        else:
            node_colors = [0.5] * len(G.nodes())
    else:
        node_sizes = [300] * len(G.nodes())
        node_colors = [0.5] * len(G.nodes())
    
    
    edges = G.edges()
    edge_weights = [G[u][v].get('weight', 1.0) for u, v in edges]
    
    
    if edge_weights:
        min_weight = min(edge_weights)
        max_weight = max(edge_weights)
        if max_weight > min_weight:
            edge_widths = [1 + (w - min_weight) / (max_weight - min_weight) * 3 for w in edge_weights]
        else:
            edge_widths = [2] * len(edge_weights)
    else:
        edge_widths = [1] * len(edges)
    
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        width=edge_widths,
        alpha=0.5,
        edge_color='gray'
    )
    
    
    node_list = list(G.nodes())
    node_sizes_list = [node_sizes.get(node, 300) if isinstance(node_sizes, pd.Series) else node_sizes[i] for i, node in enumerate(node_list)]
    node_colors_list = [node_colors.get(node, 0.5) if isinstance(node_colors, pd.Series) else node_colors[i] for i, node in enumerate(node_list)]
    
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_size=node_sizes_list,
        node_color=node_colors_list,
        cmap=plt.cm.viridis,
        alpha=0.8,
        edgecolors='black',
        linewidths=1
    )
    
    # Draw labels (optional, can be disabled for large graphs)
    if len(G.nodes()) <= 20:
        labels = {node: node[:10] + '...' if len(str(node)) > 10 else str(node) for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=8)
    
    
    ax.set_title('Building Network Graph', fontsize=14, fontweight='bold')
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    
    if metrics_df is not None and node_color_column in metrics_df.columns:
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=node_colors.min(), vmax=node_colors.max()))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label(node_color_column.replace('_', ' ').title(), fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    logger.info(f"Saved network graph to: {output_path}")
    plt.close()


def plot_accessibility_heatmap(
    buildings_df: pd.DataFrame,
    accessibility_df: pd.DataFrame,
    output_path: str = 'accessibility_heatmap.png',
    color_column: str = 'weighted_accessibility',
    figsize: Tuple[int, int] = (12, 10),
    dpi: int = 300
):
    """
    Plot accessibility heatmap.
    
    Args:
        buildings_df: DataFrame with building data (centroid coordinates)
        accessibility_df: DataFrame with accessibility metrics
        output_path: Path to save the plot
        color_column: Column name for heatmap color
        figsize: Figure size
        dpi: DPI for output image
    """
    logger.info(f"Plotting accessibility heatmap...")
    
    
    merged_df = buildings_df.merge(accessibility_df, on='building_id', how='left')
    
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    
    if color_column in merged_df.columns:
        color_values = merged_df[color_column].fillna(0)
    else:
        logger.warning(f"Column {color_column} not found, using default")
        color_values = pd.Series([0] * len(merged_df))
    
    
    min_color = color_values.min()
    max_color = color_values.max()
    if max_color > min_color:
        normalized_colors = (color_values - min_color) / (max_color - min_color)
    else:
        normalized_colors = pd.Series([0.5] * len(color_values))
    
    # Create scatter plot
    scatter = ax.scatter(
        merged_df['centroid_lon'],
        merged_df['centroid_lat'],
        c=normalized_colors,
        s=merged_df['area_m2'] * 10,  # Size based on area
        cmap=plt.cm.RdYlGn,
        alpha=0.7,
        edgecolors='black',
        linewidths=1
    )
    
    
    ax.set_title('Accessibility Heatmap', fontsize=14, fontweight='bold')
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(color_column.replace('_', ' ').title(), fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    logger.info(f"Saved accessibility heatmap to: {output_path}")
    plt.close()


def plot_optimal_paths(
    G: nx.Graph,
    buildings_df: pd.DataFrame,
    paths_data: Dict,
    output_path: str = 'optimal_paths.png',
    max_paths: int = 5,
    figsize: Tuple[int, int] = (12, 10),
    dpi: int = 300
):
    """
    Plot optimal paths between buildings.
    
    Args:
        G: NetworkX graph
        buildings_df: DataFrame with building data
        paths_data: Dictionary with paths data
        output_path: Path to save the plot
        max_paths: Maximum number of paths to plot
        figsize: Figure size
        dpi: DPI for output image
    """
    logger.info(f"Plotting optimal paths...")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Get node positions
    pos = {}
    for _, row in buildings_df.iterrows():
        building_id = row['building_id']
        if building_id in G.nodes():
            pos[building_id] = (row['centroid_lon'], row['centroid_lat'])
    
    # Draw all edges (light gray)
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.2, edge_color='gray', width=1)
    
    # Draw highlighted paths
    paths = paths_data.get('paths', {})
    path_items = list(paths.items())[:max_paths]
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(path_items)))
    
    for (path_key, path), color in zip(path_items, colors):
        # Draw path edges
        for i in range(len(path) - 1):
            nx.draw_networkx_edges(
                G, pos, ax=ax,
                edgelist=[(path[i], path[i + 1])],
                edge_color=color,
                width=3,
                alpha=0.7,
                style='dashed'
            )
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_size=500,
        node_color='lightblue',
        alpha=0.8,
        edgecolors='black',
        linewidths=2
    )
    
    # Draw labels
    labels = {node: node[:10] + '...' if len(str(node)) > 10 else str(node) for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=8)
    
    # Set title and labels
    ax.set_title(f'Optimal Paths (showing {len(path_items)} paths)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    logger.info(f"Saved optimal paths plot to: {output_path}")
    plt.close()

