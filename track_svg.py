"""Generate SVG representation of track for D3 visualization from NWB files."""

import numpy as np
import pandas as pd
from typing import Tuple, Dict
from pathlib import Path
import argparse
import sys
from util_load_data import load_data, filter_df_by_dict
import pynwb
import h5py


def validate_nwb_file(nwb_path: str) -> Path:
    """Validate that the NWB file exists and is accessible.
    
    Args:
        nwb_path: Path to NWB file
        
    Returns:
        Path object for the NWB file
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file doesn't have .nwb extension
    """
    path = Path(nwb_path)
    if not path.exists():
        raise FileNotFoundError(f"NWB file not found: {path}")
    if path.suffix.lower() != '.nwb':
        raise ValueError(f"File must have .nwb extension: {path}")
    return path


def get_dataframe_object_ids(nwb_path: str) -> Dict[str, str]:
    """Extract object IDs for DataFrames stored in the NWB file.
    
    Args:
        nwb_path: Path to NWB file
        
    Returns:
        Dictionary mapping DataFrame names to their object IDs
    """
    with pynwb.NWBHDF5IO(nwb_path, 'r') as io:
        nwb = io.read()
        
        # Get DataFrames from scratch
        df_objects = {}
        if hasattr(nwb, 'scratch'):
            for name, obj in nwb.scratch.items():
                if isinstance(obj, pynwb.core.DynamicTable):
                    df_objects[name] = str(obj.object_id)
        
        # Ensure we have the required DataFrames
        required_dfs = {'position_df', 'events_per_epoch_df'}
        found_dfs = set(df_objects.keys())
        
        if not required_dfs.issubset(found_dfs):
            missing = required_dfs - found_dfs
            print(f"Warning: Missing required DataFrames: {missing}")
            print(f"Available DataFrames in scratch: {found_dfs}")
        
        return df_objects


def load_position_data(nwb_path: str, df_obj_ids: Dict[str, str] = None) -> Tuple[pd.Series, pd.Series]:
    """Load position data from NWB file using utility functions.
    
    Args:
        nwb_path: Path to NWB file
        df_obj_ids: Dictionary mapping dataframe names to their object IDs.
                   If None, will attempt to extract from file.
        
    Returns:
        Tuple of (x_positions, y_positions) as pandas Series
    """
    # Get object IDs if not provided
    if df_obj_ids is None:
        df_obj_ids = get_dataframe_object_ids(nwb_path)
    
    # Load data using utility function
    indexed_dataframes, _ = load_data(nwb_path, df_obj_ids)
    
    # Extract position data - assuming it's in the position_df
    if 'position_df' not in indexed_dataframes:
        raise KeyError("position_df not found in loaded dataframes")
    
    position_df = indexed_dataframes['position_df']
    
    # Extract x and y positions
    x_pos = position_df['projected_x_position']
    y_pos = position_df['projected_y_position']
    
    return x_pos, y_pos


def rdp_simplify(points: np.ndarray, epsilon: float) -> np.ndarray:
    """Simplify a curve using the Ramer-Douglas-Peucker algorithm.
    
    Args:
        points: Array of points shape (N, 2) for x,y coordinates
        epsilon: Maximum distance for point simplification
        
    Returns:
        Simplified array of points
    """
    def point_line_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
        """Calculate perpendicular distance of point to line segment."""
        if np.all(start == end):
            return np.linalg.norm(point - start)
        
        line_vec = end - start
        point_vec = point - start
        line_len = np.linalg.norm(line_vec)
        line_unit_vec = line_vec / line_len
        
        # Project point onto line
        proj_len = np.dot(point_vec, line_unit_vec)
        if proj_len < 0:
            return np.linalg.norm(point - start)
        elif proj_len > line_len:
            return np.linalg.norm(point - end)
        
        # Calculate perpendicular distance
        proj = start + line_unit_vec * proj_len
        return np.linalg.norm(point - proj)
    
    if len(points) <= 2:
        return points
    
    # Find point with maximum distance
    dists = [point_line_distance(points[i], points[0], points[-1]) 
             for i in range(1, len(points) - 1)]
    max_dist = max(dists)
    max_idx = dists.index(max_dist) + 1
    
    if max_dist > epsilon:
        # Recursively simplify both segments
        left = rdp_simplify(points[:max_idx + 1], epsilon)
        right = rdp_simplify(points[max_idx:], epsilon)
        return np.vstack((left[:-1], right))
    else:
        return np.vstack((points[0], points[-1]))


def create_track_svg(x_positions: pd.Series, 
                    y_positions: pd.Series,
                    epsilon: float = 2.0,
                    stroke_width: float = 1.0,
                    stroke_color: str = "#lightgrey") -> str:
    """Create SVG representation of track from position data.
    
    Args:
        x_positions: Series of x coordinates
        y_positions: Series of y coordinates
        epsilon: Simplification tolerance for RDP algorithm
        stroke_width: Width of track path
        stroke_color: Color of track path
        
    Returns:
        SVG string representation of track
    """
    # Combine coordinates and simplify
    points = np.column_stack((x_positions.values, y_positions.values))
    simplified_points = rdp_simplify(points, epsilon)
    
    # Calculate viewBox parameters with padding
    padding = 10
    x_min, y_min = simplified_points.min(axis=0) - padding
    x_max, y_max = simplified_points.max(axis=0) + padding
    width = x_max - x_min
    height = y_max - y_min
    
    # Create SVG string with metadata
    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg"',
        f'    viewBox="{x_min} {y_min} {width} {height}"',
        f'    data-x-min="{x_min:.2f}"',
        f'    data-x-max="{x_max:.2f}"',
        f'    data-y-min="{y_min:.2f}"',
        f'    data-y-max="{y_max:.2f}"',
        f'    preserveAspectRatio="xMidYMid meet">',
        
        '    <!-- Background track -->',
        '    <g class="track-background">',
        f'        <path d="M {simplified_points[0,0]:.2f} {simplified_points[0,1]:.2f}',
    ]
    
    # Add line segments
    for point in simplified_points[1:]:
        svg_lines.append(f'             L {point[0]:.2f} {point[1]:.2f}')
    
    # Close path and add style
    svg_lines.extend([
        f'"          stroke="{stroke_color}"',
        f'           stroke-width="{stroke_width}"',
        '           fill="none"/>',
        '    </g>',
        
        '    <!-- Layer for rat positions -->',
        '    <g class="rat-positions"></g>',
        
        '    <!-- Layer for nonlocal positions -->',
        '    <g class="nonlocal-positions"></g>',
        '</svg>'
    ])
    
    return '\n'.join(svg_lines)


def save_track_svg(x_positions: pd.Series,
                  y_positions: pd.Series,
                  output_path: str,
                  **kwargs) -> None:
    """Save track SVG to file.
    
    Args:
        x_positions: Series of x coordinates
        y_positions: Series of y coordinates
        output_path: Path to save SVG file
        **kwargs: Additional arguments passed to create_track_svg
    """
    svg_content = create_track_svg(x_positions, y_positions, **kwargs)
    with open(output_path, 'w') as f:
        f.write(svg_content)


def process_nwb_file(nwb_path: str, 
                    df_obj_ids: Dict[str, str] = None,
                    output_path: str = None, 
                    epsilon: float = 2.0):
    """Process NWB file and create SVG track representation.
    
    Args:
        nwb_path: Path to NWB file
        df_obj_ids: Dictionary mapping dataframe names to their object IDs.
                   If None, will attempt to extract from file.
        output_path: Path to save SVG file (optional)
        epsilon: Simplification tolerance for RDP algorithm
    """
    # Load position data
    x_pos, y_pos = load_position_data(nwb_path, df_obj_ids)
    
    # Generate output path if not provided
    if output_path is None:
        nwb_file = Path(nwb_path)
        output_path = nwb_file.with_suffix('.svg')
    
    # Save SVG
    save_track_svg(
        x_positions=x_pos,
        y_positions=y_pos,
        output_path=output_path,
        epsilon=epsilon
    )
    
    print(f"Track SVG saved to: {output_path}")
    print(f"Number of original points: {len(x_pos)}")
    
    # Load SVG to count simplified points
    with open(output_path, 'r') as f:
        svg_content = f.read()
    num_points = svg_content.count(' L ')
    print(f"Number of simplified points: {num_points + 1}")  # +1 for initial point


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate SVG track from NWB file',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'nwb_filepath', 
        type=str,
        help='Path to NWB file'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output SVG path (optional)'
    )
    parser.add_argument(
        '--epsilon', '-e',
        type=float,
        default=2.0,
        help='Simplification tolerance'
    )
    parser.add_argument(
        '--list-dfs',
        action='store_true',
        help='List available DataFrames in the NWB file and exit'
    )
    
    args = parser.parse_args()
    
    try:
        # Validate NWB file path
        nwb_path = validate_nwb_file(args.nwb_filepath)
        
        if args.list_dfs:
            df_objects = get_dataframe_object_ids(str(nwb_path))
            print("\nAvailable DataFrames in NWB file:")
            for name, obj_id in df_objects.items():
                print(f"  {name}: {obj_id}")
            sys.exit(0)
        
        process_nwb_file(
            nwb_path=str(nwb_path),
            output_path=args.output,
            epsilon=args.epsilon
        )
        
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1) 