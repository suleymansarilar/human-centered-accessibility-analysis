"""
Combine footprints from multiple GML files into a single pickle file.
"""

import pickle
from pathlib import Path
import sys

def combine_footprints(input_dir, output_path):
    """Combine all footprint pickle files into one."""
    input_path = Path(input_dir)
    output_path = Path(output_path)
    
    
    footprint_files = list(input_path.glob("*_footprints.pkl"))
    
    if not footprint_files:
        print(f"No footprint files found in {input_dir}")
        return None
    
    print(f"Found {len(footprint_files)} footprint files")
    
    # Combine footprints
    all_footprints = {}
    for fp_file in footprint_files:
        print(f"Loading {fp_file.name}...")
        with open(fp_file, 'rb') as f:
            footprints = pickle.load(f)
            all_footprints.update(footprints)
    
    # Save combined
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(all_footprints, f)
    
    print(f"Combined {len(all_footprints)} footprints into {output_path}")
    return all_footprints

if __name__ == '__main__':
    input_dir = sys.argv[1] if len(sys.argv) > 1 else 'data/processed'
    output_path = sys.argv[2] if len(sys.argv) > 2 else 'data/processed/all_buildings_footprints.pkl'
    combine_footprints(input_dir, output_path)

