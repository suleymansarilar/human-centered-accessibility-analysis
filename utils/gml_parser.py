"""
CityGML Parser Utilities

CityGML Schema:
- Namespace: http://www.opengis.net/citygml/2.0
- Building: bldg:Building
- GroundSurface: bldg:GroundSurface
- Geometry: gml:Polygon, gml:LinearRing, gml:posList
"""

import xml.etree.ElementTree as ET
from typing import List, Dict, Optional, Tuple
import pandas as pd
from shapely.geometry import Polygon, Point
from shapely.ops import transform
import pyproj
from functools import partial
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CityGML namespaces
CITYGML_NS = {
    'citygml': 'http://www.opengis.net/citygml/2.0',
    'core': 'http://www.opengis.net/citygml/2.0',
    'bldg': 'http://www.opengis.net/citygml/building/2.0',
    'gml': 'http://www.opengis.net/gml',
    'app': 'http://www.opengis.net/citygml/appearance/2.0',
    'gen': 'http://www.opengis.net/citygml/generics/2.0',
    'xAL': 'urn:oasis:names:tc:ciq:xsdschema:xAL:2.0',
    'xlink': 'http://www.w3.org/1999/xlink',
    'xsi': 'http://www.w3.org/2001/XMLSchema-instance'
}


class CityGMLParser:
    """
    CityGML file parser.
    
    Attributes:
        tree: XML element tree
        root: Root element
        buildings: Extracted building data list
    """
    
    def __init__(self, gml_file_path: str):
        """
        Initialize parser with CityGML file.
        
        Args:
            gml_file_path: Path to CityGML file
        """
        self.gml_file_path = gml_file_path
        self.tree = None
        self.root = None
        self.buildings = []
        
        # Coordinate system (default: WGS84)
        self.source_crs = None
        self.target_crs = 'EPSG:4326'  # WGS84 (lat/lon)
        
    def parse(self):
        """Parse CityGML file and extract buildings."""
        logger.info(f"Parsing CityGML file: {self.gml_file_path}")
        
        # Parse XML
        try:
            self.tree = ET.parse(self.gml_file_path)
            self.root = self.tree.getroot()
        except ET.ParseError as e:
            logger.error(f"XML parsing error: {e}")
            raise
        except FileNotFoundError:
            logger.error(f"File not found: {self.gml_file_path}")
            raise
        
        # Detect coordinate system
        self._detect_crs()
        
        # Extract buildings
        self._extract_buildings()
        
        logger.info(f"Extracted {len(self.buildings)} buildings")
        
    def _detect_crs(self):
        """
        Detect coordinate reference system from CityGML file.
        
        CityGML files may have CRS information in:
        - gml:boundedBy/gml:Envelope/@srsName
        - gml:pos/@srsName
        - app:srsName attribute
        """
        # Default: assume WGS84 if not specified
        self.source_crs = 'EPSG:4326'
        
        # Try to find CRS in boundedBy
        bounded_by = self.root.find('.//gml:boundedBy', CITYGML_NS)
        if bounded_by is not None:
            envelope = bounded_by.find('gml:Envelope', CITYGML_NS)
            if envelope is not None:
                srs_name = envelope.get('srsName')
                if srs_name:
                    self.source_crs = srs_name
                    logger.info(f"Detected CRS: {self.source_crs}")
        
        # Try to find CRS in first pos element
        pos = self.root.find('.//gml:pos', CITYGML_NS)
        if pos is not None:
            srs_name = pos.get('srsName')
            if srs_name:
                self.source_crs = srs_name
                logger.info(f"Detected CRS from pos: {srs_name}")
    
    def _extract_buildings(self):
        """
        Extract all buildings from CityGML file.
        
        For each building:
        1. Extract building ID
        2. Extract ground surface (footprint)
        3. Extract metadata (height, type, usage)
        4. Calculate centroid and area
        """
        # Try different ways to find buildings
        # Method 1: Direct bldg:Building elements
        building_elements = self.root.findall('.//bldg:Building', CITYGML_NS)
        
        # Method 2: core:cityObjectMember/bldg:Building
        if not building_elements:
            city_object_members = self.root.findall('.//core:cityObjectMember', CITYGML_NS)
            for member in city_object_members:
                building = member.find('bldg:Building', CITYGML_NS)
                if building is not None:
                    building_elements.append(building)
        
        if not building_elements:
            logger.warning("No buildings found in CityGML file")
            return
        
        logger.info(f"Found {len(building_elements)} building elements")
        
        for idx, building_elem in enumerate(building_elements):
            try:
                building_data = self._extract_building_data(building_elem, idx)
                if building_data:
                    self.buildings.append(building_data)
            except Exception as e:
                logger.warning(f"Error extracting building {idx}: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                continue
    
    def _extract_building_data(self, building_elem, idx: int) -> Optional[Dict]:
        """
        Extract data from a single building element.
        
        Args:
            building_elem: XML element for building
            idx: Building index (if no ID available)
            
        Returns:
            Dictionary with building data
        """
        
        building_id = building_elem.get('{http://www.opengis.net/gml}id')
        if not building_id:
            building_id = f"building_{idx}"
        
        
        footprint = self._extract_ground_surface(building_elem)
        if not footprint:
            logger.warning(f"No ground surface found for building {building_id}")
            return None
        
        
        height = self._extract_height(building_elem)
        building_type = self._extract_building_type(building_elem)
        usage = self._extract_usage(building_elem)
        
        # Calculate centroid and area
        # For area calculation, we need to work in projected CRS (meters)
        # For centroid, we'll transform to WGS84
        
        # Calculate area in source CRS (should be in meters for projected CRS)
        area = footprint.area
        
        # Calculate centroid
        centroid = footprint.centroid
        
        # Transform centroid to WGS84 for lat/lon coordinates
        if self.source_crs != self.target_crs:
            centroid_wgs84 = self._transform_coordinates(centroid, self.source_crs, self.target_crs)
            # Use transformed centroid for lat/lon
            centroid = centroid_wgs84
        else:
            # If already WGS84, area needs to be recalculated
            # For WGS84, we need to project to a metric CRS for area calculation
            if 'EPSG:4326' in str(self.source_crs) or 'WGS84' in str(self.source_crs):
                try:
                    utm_crs = 'EPSG:32636'
                    source_proj = pyproj.CRS('EPSG:4326')
                    target_proj = pyproj.CRS(utm_crs)
                    transformer = pyproj.Transformer.from_crs(source_proj, target_proj, always_xy=True)
                    
                    from shapely.ops import transform as shapely_transform
                    footprint_utm = shapely_transform(
                        lambda x, y: transformer.transform(x, y)[:2],
                        footprint
                    )
                    area = footprint_utm.area
                except Exception as e:
                    logger.warning(f"Could not calculate area in UTM: {e}")
                    area = footprint.area * 111000 * 111000
        
        building_data = {
            'building_id': building_id,
            'centroid_lon': centroid.x,
            'centroid_lat': centroid.y,
            'area_m2': area,
            'height_m': height,
            'building_type': building_type,
            'usage': usage,
            'footprint': footprint  # Shapely Polygon
        }
        
        return building_data
    
    def _extract_ground_surface(self, building_elem) -> Optional[Polygon]:
        """
        Extract ground surface polygon from building element.
        
        Ground surface is typically in:
        - bldg:lod0FootPrint/gml:Polygon (direct)
        - bldg:lod0FootPrint/gml:MultiSurface/gml:surfaceMember/gml:Polygon
        - bldg:GroundSurface/bldg:lod0FootPrint/gml:Polygon
        - bldg:GroundSurface/bldg:lod1FootPrint/gml:Polygon
        
        Args:
            building_elem: Building XML element
            
        Returns:
            Shapely Polygon of ground surface, or None if not found
        """
        # Try lod0FootPrint first
        lod0_footprint = building_elem.find('.//bldg:lod0FootPrint', CITYGML_NS)
        if lod0_footprint is not None:
            # Try MultiSurface 
            multi_surface = lod0_footprint.find('.//gml:MultiSurface', CITYGML_NS)
            if multi_surface is not None:
                # Get first surface member polygon
                surface_member = multi_surface.find('.//gml:surfaceMember', CITYGML_NS)
                if surface_member is not None:
                    polygon = surface_member.find('.//gml:Polygon', CITYGML_NS)
                    if polygon is not None:
                        return self._parse_polygon(polygon)
            
            # Try direct Polygon
            polygon = lod0_footprint.find('.//gml:Polygon', CITYGML_NS)
            if polygon is not None:
                return self._parse_polygon(polygon)
        
        # Try GroundSurface element
        ground_surface = building_elem.find('.//bldg:GroundSurface', CITYGML_NS)
        if ground_surface is not None:
            # Try lod0FootPrint in GroundSurface
            lod0_footprint = ground_surface.find('.//bldg:lod0FootPrint', CITYGML_NS)
            if lod0_footprint is not None:
                polygon = lod0_footprint.find('.//gml:Polygon', CITYGML_NS)
                if polygon is not None:
                    return self._parse_polygon(polygon)
            
            # Try lod1FootPrint
            lod1_footprint = ground_surface.find('.//bldg:lod1FootPrint', CITYGML_NS)
            if lod1_footprint is not None:
                polygon = lod1_footprint.find('.//gml:Polygon', CITYGML_NS)
                if polygon is not None:
                    return self._parse_polygon(polygon)
        
        # Try alternative: building directly contains polygon
        polygon = building_elem.find('.//gml:Polygon', CITYGML_NS)
        if polygon is not None:
            return self._parse_polygon(polygon)
        
        return None
    
    def _parse_polygon(self, polygon_elem) -> Optional[Polygon]:
        """
        Parse GML Polygon element to Shapely Polygon.
        
        GML Polygon structure:
        - gml:Polygon
          - gml:exterior
            - gml:LinearRing
              - gml:posList (space-separated coordinates)
              - OR gml:pos (individual points)
        
        Args:
            polygon_elem: GML Polygon XML element
            
        Returns:
            Shapely Polygon, or None if parsing fails
        """
        
        exterior = polygon_elem.find('.//gml:exterior', CITYGML_NS)
        if exterior is None:
            return None
        
        
        linear_ring = exterior.find('.//gml:LinearRing', CITYGML_NS)
        if linear_ring is None:
            return None
        
        
        coords = self._extract_coordinates(linear_ring)
        if not coords or len(coords) < 3:
            return None
        
        # Close polygon if not closed
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        
        try:
            polygon = Polygon(coords)
            
            if not polygon.is_valid:
                logger.warning(f"Invalid polygon, attempting to fix...")
                polygon = polygon.buffer(0)  # Fix self-intersections
            return polygon
        except Exception as e:
            logger.warning(f"Error creating polygon: {e}")
            return None
    
    def _extract_coordinates(self, linear_ring) -> List[Tuple[float, float]]:
        """
        Extract coordinates from LinearRing element.
        
        Coordinates can be in:
        - gml:posList (space-separated: "x1 y1 x2 y2 ..." or "x1 y1 z1 x2 y2 z2 ...")
        - gml:pos (individual points: "x y" or "x y z")
        
        Args:
            linear_ring: GML LinearRing XML element
            
        Returns:
            List of (x, y) coordinate tuples
        """
        coords = []
        
        # Try posList
        pos_list = linear_ring.find('.//gml:posList', CITYGML_NS)
        if pos_list is not None:
            text = pos_list.text.strip()
            values = text.split()
            
            
            if len(values) % 2 == 0:
                # 2D coordinates
                for i in range(0, len(values), 2):
                    x = float(values[i])
                    y = float(values[i + 1])
                    coords.append((x, y))
            elif len(values) % 3 == 0:
                # 3D coordinates, take only x, y
                for i in range(0, len(values), 3):
                    x = float(values[i])
                    y = float(values[i + 1])
                    coords.append((x, y))
            
            if coords:
                return coords
        
        # Fallback: individual pos elements
        pos_elements = linear_ring.findall('.//gml:pos', CITYGML_NS)
        if pos_elements:
            for pos in pos_elements:
                text = pos.text.strip()
                values = text.split()
                if len(values) >= 2:
                    x = float(values[0])
                    y = float(values[1])
                    coords.append((x, y))
        
        return coords
    
    def _extract_height(self, building_elem) -> Optional[float]:
        """Extract building height from building element."""
        height_attrs = building_elem.findall('.//gen:doubleAttribute[@name="buildingHeight"]', CITYGML_NS)
        if height_attrs:
            value_elem = height_attrs[0].find('gen:value', CITYGML_NS)
            if value_elem is not None and value_elem.text:
                try:
                    return float(value_elem.text.strip())
                except (ValueError, AttributeError):
                    pass
        
        # Try measuredHeight
        measured_height = building_elem.find('.//bldg:measuredHeight', CITYGML_NS)
        if measured_height is not None:
            try:
                return float(measured_height.text)
            except (ValueError, AttributeError):
                pass
        
        # Try storeysAboveGround * average storey height (typical: 3m)
        storeys = building_elem.find('.//bldg:storeysAboveGround', CITYGML_NS)
        if storeys is not None:
            try:
                return float(storeys.text) * 3.0  # Estimate
            except (ValueError, AttributeError):
                pass
        
        return None
    
    def _extract_building_type(self, building_elem) -> Optional[str]:
        """Extract building type/function from building element."""
        type_attrs = building_elem.findall('.//gen:intAttribute[@name="buildingType"]', CITYGML_NS)
        if type_attrs:
            value_elem = type_attrs[0].find('gen:value', CITYGML_NS)
            if value_elem is not None and value_elem.text:
                return value_elem.text.strip()
        
        bldg_class = building_elem.find('.//bldg:class', CITYGML_NS)
        if bldg_class is not None and bldg_class.text:
            class_text = bldg_class.text.strip()
            translations = {
                'MimariBina': 'ArchitecturalBuilding',
                'Mimari Bina': 'ArchitecturalBuilding',
                'Yapı': 'Building',
                'Bina': 'Building'
            }
            return translations.get(class_text, class_text)
        
        # Try buildingFunction
        function = building_elem.find('.//bldg:function', CITYGML_NS)
        if function is not None:
            code = function.get('code')
            if code:
                return code
        
        # Try usage
        usage = building_elem.find('.//bldg:usage', CITYGML_NS)
        if usage is not None:
            code = usage.get('code')
            if code:
                return code
        
        return None
    
    def _extract_usage(self, building_elem) -> Optional[str]:
        """Extract building usage information."""
        construction_id_attrs = building_elem.findall('.//gen:stringAttribute[@name="constructionID"]', CITYGML_NS)
        if construction_id_attrs:
            value_elem = construction_id_attrs[0].find('gen:value', CITYGML_NS)
            if value_elem is not None and value_elem.text:
                return value_elem.text.strip()
        
        name_elem = building_elem.find('.//gml:name', CITYGML_NS)
        if name_elem is not None and name_elem.text:
            name = name_elem.text.strip()
            translations = {
                'Mimari Bina': 'Architectural Building',
                'MimariBina': 'Architectural Building'
            }
            return translations.get(name, name)
        
        # Try bldg:usage
        usage = building_elem.find('.//bldg:usage', CITYGML_NS)
        if usage is not None and usage.text:
            return usage.text.strip()
        
        return None
    
    def _transform_coordinates(self, point: Point, source_crs: str, target_crs: str) -> Point:
        """
        Transform coordinates from source CRS to target CRS.
        
        Args:
            point: Shapely Point in source CRS
            source_crs: Source coordinate reference system
            target_crs: Target coordinate reference system
            
        Returns:
            Transformed Point
        """
        try:
            if source_crs.startswith('EPSG:'):
                source_crs = source_crs
            elif 'EPSG' in source_crs:
                import re
                match = re.search(r'EPSG:(\d+)', source_crs)
                if match:
                    source_crs = f"EPSG:{match.group(1)}"
                else:
                    source_crs = 'EPSG:4326'
            else:
                source_crs = 'EPSG:4326'
            
            if target_crs.startswith('EPSG:'):
                target_crs = target_crs
            else:
                target_crs = 'EPSG:4326'
            
            if source_crs == target_crs:
                return point
            
            source_proj = pyproj.CRS(source_crs)
            target_proj = pyproj.CRS(target_crs)
            transformer = pyproj.Transformer.from_crs(source_proj, target_proj, always_xy=True)
            
            x, y = transformer.transform(point.x, point.y)
            return Point(x, y)
        except Exception as e:
            logger.warning(f"Coordinate transformation error: {e} (source: {source_crs}, target: {target_crs})")
            return point
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert extracted buildings to pandas DataFrame.
        
        Returns:
            DataFrame with building data (without footprint geometry)
        """
        if not self.buildings:
            return pd.DataFrame()
        
        # Extract data without footprint (for DataFrame)
        data = []
        for building in self.buildings:
            data.append({
                'building_id': building['building_id'],
                'centroid_lon': building['centroid_lon'],
                'centroid_lat': building['centroid_lat'],
                'area_m2': building['area_m2'],
                'height_m': building['height_m'],
                'building_type': building['building_type'],
                'usage': building['usage']
            })
        
        return pd.DataFrame(data)
    
    def get_footprints(self) -> Dict[str, Polygon]:
        """
        Get building footprints as dictionary.
        
        Returns:
            Dictionary mapping building_id to Shapely Polygon
        """
        return {b['building_id']: b['footprint'] for b in self.buildings}


if __name__ == '__main__':
    # Test parser
    parser = CityGMLParser('data/input/sample_buildings.gml')
    parser.parse()
    
    df = parser.to_dataframe()
    print(df)
    
    footprints = parser.get_footprints()
    print(f"Extracted {len(footprints)} building footprints")

