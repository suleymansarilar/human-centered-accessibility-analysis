"""
Usage:
    python scripts/1_extract_buildings.py --input data/input/sample_buildings.gml --output data/processed/buildings_geodata.csv
"""

import argparse
import sys
import os
from pathlib import Path
import pandas as pd
import logging

sys.path.append(str(Path(__file__).parent.parent))

from utils.gml_parser import CityGMLParser


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_buildings(input_gml_path: str, output_csv_path: str):
    """
    Extract buildings from CityGML file and save to CSV.
    
    Args:
        input_gml_path: Path to input CityGML file
        output_csv_path: Path to output CSV file
    """
    logger.info("=" * 60)
    logger.info("Step 1: Building Footprint Extraction")
    logger.info("=" * 60)
    
    if not os.path.exists(input_gml_path):
        logger.error(f"Input file not found: {input_gml_path}")
        raise FileNotFoundError(f"Input file not found: {input_gml_path}")
    
    
    output_dir = Path(output_csv_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    
    logger.info(f"Parsing CityGML file: {input_gml_path}")
    parser = CityGMLParser(input_gml_path)
    
    try:
        parser.parse()
    except Exception as e:
        logger.error(f"Error parsing CityGML file: {e}")
        raise
    
   
    logger.info("Converting to DataFrame...")
    df = parser.to_dataframe()
    
    if df.empty:
        logger.warning("No buildings extracted from CityGML file")
        return
    
   
    logger.info(f"Extracted {len(df)} buildings")
    logger.info(f"Building ID range: {df['building_id'].min()} to {df['building_id'].max()}")
    logger.info(f"Area range: {df['area_m2'].min():.2f} to {df['area_m2'].max():.2f} m²")
    logger.info(f"Centroid longitude range: {df['centroid_lon'].min():.6f} to {df['centroid_lon'].max():.6f}")
    logger.info(f"Centroid latitude range: {df['centroid_lat'].min():.6f} to {df['centroid_lat'].max():.6f}")
    
   
    logger.info(f"Saving to CSV: {output_csv_path}")
    df.to_csv(output_csv_path, index=False)
    logger.info(f"Successfully saved {len(df)} buildings to {output_csv_path}")
    
  
    footprints_path = output_csv_path.replace('.csv', '_footprints.pkl')
    import pickle
    footprints = parser.get_footprints()
    with open(footprints_path, 'wb') as f:
        pickle.dump(footprints, f)
    logger.info(f"Saved footprints to: {footprints_path}")
    
    
    logger.info("\nFirst 5 buildings:")
    print(df.head())
    
    return df


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Extract building footprints from CityGML file'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input CityGML file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/processed/buildings_geodata.csv',
        help='Path to output CSV file'
    )
    
    args = parser.parse_args()
    
    try:
        extract_buildings(args.input, args.output)
        logger.info("=" * 60)
        logger.info("Step 1 completed!")
        logger.info("=" * 60)
    except Exception as e:
        logger.error(f"Error in Step 1: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()

