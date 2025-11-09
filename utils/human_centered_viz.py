"""
Human-Centered Visualization Utilities

Functions for visualizing user profiles, flow simulation, and emergency scenarios.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300


def plot_user_accessibility_heatmap(
    buildings_df: pd.DataFrame,
    user_accessibility_df: pd.DataFrame,
    output_path: str
):
    """
    Plot accessibility heatmap for different user profiles.
    
    Args:
        buildings_df: DataFrame with building data
        user_accessibility_df: DataFrame with user accessibility data
        output_path: Path to save the plot
    """
    logger.info("Plotting user accessibility heatmap...")
    
    # Pivot table: buildings x profiles
    pivot_data = user_accessibility_df.pivot_table(
        index='building_id',
        columns='profile_name',
        values='accessibility_score',
        aggfunc='mean'
    )
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot heatmap
    sns.heatmap(
        pivot_data.T,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn',
        center=0.5,
        vmin=0,
        vmax=1,
        ax=ax,
        cbar_kws={'label': 'Accessibility Score'}
    )
    
    ax.set_title('User Profile Accessibility Heatmap', fontsize=14, fontweight='bold')
    ax.set_xlabel('Building ID', fontsize=12)
    ax.set_ylabel('User Profile', fontsize=12)
    ax.set_xticklabels([bid[:8] + '...' if len(bid) > 8 else bid for bid in pivot_data.index], rotation=45, ha='right')
    ax.set_yticklabels(pivot_data.columns, rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved user accessibility heatmap to: {output_path}")


def plot_user_accessibility_comparison(
    user_accessibility_df: pd.DataFrame,
    output_path: str
):
    """
    Plot accessibility comparison across user profiles.
    
    Args:
        user_accessibility_df: DataFrame with user accessibility data
        output_path: Path to save the plot
    """
    logger.info("Plotting user accessibility comparison...")
    
    # Calculate average metrics per profile
    profile_metrics = user_accessibility_df.groupby('profile_name').agg({
        'accessibility_score': 'mean',
        'avg_travel_time_minutes': 'mean',
        'reachable_count': 'mean'
    }).reset_index()
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Accessibility Score
    axes[0].barh(profile_metrics['profile_name'], profile_metrics['accessibility_score'], color='steelblue')
    axes[0].set_xlabel('Accessibility Score', fontsize=11)
    axes[0].set_title('Average Accessibility Score', fontsize=12, fontweight='bold')
    axes[0].grid(axis='x', alpha=0.3)
    
    # 2. Travel Time
    axes[1].barh(profile_metrics['profile_name'], profile_metrics['avg_travel_time_minutes'], color='coral')
    axes[1].set_xlabel('Travel Time (minutes)', fontsize=11)
    axes[1].set_title('Average Travel Time', fontsize=12, fontweight='bold')
    axes[1].grid(axis='x', alpha=0.3)
    
    # 3. Reachable Buildings
    axes[2].barh(profile_metrics['profile_name'], profile_metrics['reachable_count'], color='mediumseagreen')
    axes[2].set_xlabel('Reachable Buildings', fontsize=11)
    axes[2].set_title('Average Reachable Buildings', fontsize=12, fontweight='bold')
    axes[2].grid(axis='x', alpha=0.3)
    
    plt.suptitle('User Profile Accessibility Comparison', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved user accessibility comparison to: {output_path}")


def plot_accessibility_disparities(
    disparities_file: str,
    output_path: str
):
    """
    Plot accessibility disparities between user profiles.
    
    Args:
        disparities_file: Path to disparities JSON file
        output_path: Path to save the plot
    """
    logger.info("Plotting accessibility disparities...")
    
    # Load disparities
    with open(disparities_file, 'r') as f:
        disparities = json.load(f)
    
    # Prepare data
    profile_names = []
    avg_disparities = []
    buildings_affected = []
    
    for profile_id, data in disparities.items():
        # Convert profile_id to readable name
        profile_name = profile_id.replace('_', ' ').title()
        profile_names.append(profile_name)
        avg_disparities.append(data['avg_disparity'] * 100)  # Convert to percentage
        buildings_affected.append(data['buildings_affected'])
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Average Disparity
    ax1.barh(profile_names, avg_disparities, color='tomato')
    ax1.set_xlabel('Average Disparity (%)', fontsize=11)
    ax1.set_title('Accessibility Disparity', fontsize=12, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    ax1.axvline(x=0, color='black', linewidth=0.8)
    
    # 2. Buildings Affected
    ax2.barh(profile_names, buildings_affected, color='darkorange')
    ax2.set_xlabel('Buildings Affected', fontsize=11)
    ax2.set_title('Buildings with Accessibility Issues', fontsize=12, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    plt.suptitle('Accessibility Disparities Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved accessibility disparities plot to: {output_path}")


def plot_flow_simulation(
    buildings_df: pd.DataFrame,
    G: nx.Graph,
    flow_simulation_df: pd.DataFrame,
    scenario_id: str,
    output_path: str
):
    """
    Plot flow simulation for a specific scenario.
    
    Args:
        buildings_df: DataFrame with building data
        G: NetworkX graph
        flow_simulation_df: DataFrame with flow simulation data
        scenario_id: Scenario ID to plot
        output_path: Path to save the plot
    """
    logger.info(f"Plotting flow simulation for scenario: {scenario_id}")
    
    # Filter data for scenario
    scenario_data = flow_simulation_df[flow_simulation_df['scenario_id'] == scenario_id]
    
    if scenario_data.empty:
        logger.warning(f"No data found for scenario: {scenario_id}")
        return
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # 1. Flow Volume
    edge_flows = {}
    for _, row in scenario_data.iterrows():
        edge = (row['source'], row['target'])
        edge_flows[edge] = row['flow_volume']
    
    # Plot network with flow
    pos = {}
    # Try different column name variations
    lon_col = 'centroid_longitude' if 'centroid_longitude' in buildings_df.columns else 'centroid_lon'
    lat_col = 'centroid_latitude' if 'centroid_latitude' in buildings_df.columns else 'centroid_lat'
    for _, row in buildings_df.iterrows():
        building_id = row['building_id']
        pos[building_id] = (row[lon_col], row[lat_col])
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, ax=ax1, node_color='lightblue', node_size=500, alpha=0.7)
    
    # Draw edges with flow width
    max_flow = max(edge_flows.values()) if edge_flows else 1
    for (u, v), flow in edge_flows.items():
        width = (flow / max_flow) * 5 if max_flow > 0 else 1
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], ax=ax1, width=width, alpha=0.6, edge_color='steelblue')
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, ax=ax1, font_size=8)
    
    ax1.set_title(f'Flow Simulation: {scenario_data.iloc[0]["scenario_name"]}', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # 2. Simple congestion bar chart
    if len(scenario_data) > 0:
        congestion_data = scenario_data.groupby('edge_id').agg({
            'congestion_level': 'mean'
        }).reset_index()
        
        ax2.barh(range(len(congestion_data)), congestion_data['congestion_level'], color='coral')
        ax2.set_yticks(range(len(congestion_data)))
        ax2.set_yticklabels([eid[:20] + '...' if len(eid) > 20 else eid for eid in congestion_data['edge_id']], fontsize=8)
        ax2.set_xlabel('Congestion Level', fontsize=11)
        ax2.set_title('Edge Congestion Levels', fontsize=12, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved flow simulation plot to: {output_path}")


def plot_emergency_evacuation_routes(
    buildings_df: pd.DataFrame,
    G: nx.Graph,
    emergency_scenarios_file: str,
    scenario_id: str,
    output_path: str
):
    """
    Plot emergency evacuation routes for a specific scenario.
    
    Args:
        buildings_df: DataFrame with building data
        G: NetworkX graph
        emergency_scenarios_file: Path to emergency scenarios JSON file
        scenario_id: Scenario ID to plot
        output_path: Path to save the plot
    """
    logger.info(f"Plotting emergency evacuation routes for scenario: {scenario_id}")
    
    # Load emergency scenarios
    with open(emergency_scenarios_file, 'r') as f:
        emergency_scenarios = json.load(f)
    
    if scenario_id not in emergency_scenarios:
        logger.warning(f"Scenario {scenario_id} not found in emergency scenarios")
        return
    
    scenario = emergency_scenarios[scenario_id]
    evacuation_routes = scenario.get('evacuation_routes', {})
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Get positions
    pos = {}
    for _, row in buildings_df.iterrows():
        building_id = row['building_id']
        # Try different column name variations
        lon_col = 'centroid_longitude' if 'centroid_longitude' in buildings_df.columns else 'centroid_lon'
        lat_col = 'centroid_latitude' if 'centroid_latitude' in buildings_df.columns else 'centroid_lat'
        pos[building_id] = (row[lon_col], row[lat_col])
    
    # Draw nodes
    emergency_buildings = scenario.get('emergency_buildings', [])
    safe_zones = scenario.get('safe_zones', [])
    
    # Draw all nodes
    all_nodes = list(G.nodes())
    non_emergency_nodes = [n for n in all_nodes if n not in emergency_buildings and n not in safe_zones]
    
    nx.draw_networkx_nodes(G, pos, nodelist=non_emergency_nodes, ax=ax, 
                          node_color='lightgray', node_size=300, alpha=0.5)
    nx.draw_networkx_nodes(G, pos, nodelist=emergency_buildings, ax=ax, 
                          node_color='red', node_size=500, alpha=0.8, label='Emergency Building')
    nx.draw_networkx_nodes(G, pos, nodelist=safe_zones, ax=ax, 
                          node_color='green', node_size=500, alpha=0.8, label='Safe Zone')
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='lightgray', width=1, alpha=0.3)
    
    # Draw evacuation routes
    route_colors = plt.cm.Set3(np.linspace(0, 1, len(evacuation_routes)))
    for i, (building_id, routes) in enumerate(evacuation_routes.items()):
        if building_id not in emergency_buildings:
            continue
        
        # Primary route
        primary_route = routes.get('primary_route', [])
        if len(primary_route) > 1:
            route_edges = [(primary_route[j], primary_route[j+1]) for j in range(len(primary_route)-1)]
            nx.draw_networkx_edges(G, pos, edgelist=route_edges, ax=ax, 
                                  edge_color=route_colors[i], width=3, alpha=0.7, 
                                  style='solid', label=f'Route from {building_id[:8]}')
        
        # Alternative routes
        alternative_routes = routes.get('alternative_routes', [])
        for alt_route in alternative_routes:
            if len(alt_route) > 1:
                route_edges = [(alt_route[j], alt_route[j+1]) for j in range(len(alt_route)-1)]
                nx.draw_networkx_edges(G, pos, edgelist=route_edges, ax=ax, 
                                      edge_color=route_colors[i], width=2, alpha=0.4, 
                                      style='dashed')
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=7)
    
    ax.set_title(f'Emergency Evacuation Routes: {scenario["name"]}', fontsize=14, fontweight='bold')
    ax.axis('off')
    ax.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved emergency evacuation routes plot to: {output_path}")


def plot_emergency_safety_analysis(
    emergency_analysis_file: str,
    output_path: str
):
    """
    Plot emergency safety analysis - simplified version.
    
    Args:
        emergency_analysis_file: Path to emergency evacuation analysis CSV file
        output_path: Path to save the plot
    """
    logger.info("Plotting emergency safety analysis...")
    
    # Load data
    df = pd.read_csv(emergency_analysis_file)
    
    # Simple summary: evacuation time and risk status
    scenario_summary = df.groupby('scenario_name').agg({
        'min_evacuation_time_minutes': 'mean',
        'is_at_risk': 'sum'
    }).reset_index()
    
    # Create simple figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. Evacuation Time
    ax1.barh(scenario_summary['scenario_name'], scenario_summary['min_evacuation_time_minutes'], 
             color='steelblue')
    ax1.set_xlabel('Evacuation Time (minutes)', fontsize=11)
    ax1.set_title('Average Evacuation Time', fontsize=12, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # 2. Buildings at Risk
    ax2.barh(scenario_summary['scenario_name'], scenario_summary['is_at_risk'], 
             color='tomato')
    ax2.set_xlabel('Buildings at Risk', fontsize=11)
    ax2.set_title('Buildings at Risk', fontsize=12, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    plt.suptitle('Emergency Safety Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved emergency safety analysis plot to: {output_path}")

