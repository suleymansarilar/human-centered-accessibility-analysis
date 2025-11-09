# Quick Start Guide

This guide will help you get started with the Human-Centered Accessibility & Flow Analysis project quickly.

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/suleymansarilar/human-centered-accessibility-analysis.git
cd human-centered-accessibility-analysis
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

**Note:** If you encounter errors, try installing packages individually or check the error messages for guidance.

### 3. Prepare Data
Place your CityGML files in the `data/input/` directory.

## Running the Pipeline

### Basic Usage
```bash
python scripts/run_pipeline.py --input "data/input/*.gml" --output data/output/ --threshold 500
```

### Parameters
- `--input`: Input GML file pattern (e.g., "data/input/*.gml")
- `--output`: Output directory for visualizations
- `--threshold`: Network construction distance threshold in meters (default: 500)

### Example
```bash
# Process all GML files in data/input/
python scripts/run_pipeline.py --input "data/input/*.gml" --output data/output/ --threshold 500

# Process a specific file
python scripts/run_pipeline.py --input "data/input/building1.gml" --output data/output/ --threshold 500
```

## Expected Outputs

After running the pipeline, you should see:

### Data Files (in `data/processed/`)
- `all_buildings.csv` - Extracted building data
- `building_network_graph.pkl` - Network graph
- `network_metrics.csv` - Network analysis metrics
- `accessibility_metrics.csv` - Accessibility scores
- `user_accessibility.csv` - User profile accessibility data
- `flow_simulation.csv` - Flow simulation data
- `emergency_scenarios.json` - Emergency scenario analysis

### Visualizations (in `data/output/`)
- `network_graph.png` - Network visualization
- `accessibility_heatmap.png` - Accessibility heatmap
- `user_accessibility_heatmap.png` - User accessibility heatmap
- `user_accessibility_comparison.png` - User profile comparison
- `flow_simulation_rush_hour.png` - Flow simulation (rush hour)
- `emergency_evacuation_fire.png` - Emergency evacuation (fire scenario)

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError**
   - Solution: Install missing packages with `pip install <package-name>`
   - Check `requirements.txt` for all required packages

2. **FileNotFoundError**
   - Solution: Make sure GML files are in `data/input/` directory
   - Check file paths in the command

3. **Memory Error**
   - Solution: Process files one at a time or use smaller datasets
   - Reduce the number of buildings if possible

4. **Visualization Errors**
   - Solution: Some visualizations might fail - this is normal for a learning project
   - Check the log messages for specific errors

### Getting Help

- Check GitHub issues (if available)
- Review documentation files

## Next Steps

1. Review the outputs in `data/output/`
2. Check the processed data in `data/processed/`
3. Experiment with different threshold values
4. Try different GML files
5. Modify configuration files in `config/`

## Notes

- Some features might not work perfectly
- Performance might be slow with large datasets
- Some visualizations are experimental

---

**Last Updated:** 2024-12-19

