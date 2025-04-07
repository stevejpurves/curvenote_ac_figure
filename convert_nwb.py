"""Convert NWB file to SVG track representation."""

import argparse
import sys
from pathlib import Path
from pynwb import NWBHDF5IO
from hdmf_zarr.nwb import NWBZarrIO
from contextlib import suppress
import logging
import time
from datetime import timedelta

logging.basicConfig(level=logging.INFO)  # or DEBUG for even more


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


def confirm_operation(nwb_path: Path, output_dir: Path) -> bool:
    """Ask user to confirm the operation with file details.
    
    Args:
        nwb_path: Path to input NWB file
        output_dir: Path to output directory
        
    Returns:
        bool: True if user confirms, False otherwise
    """
    print("\nOperation Summary:")
    print("-----------------")
    print(f"Input NWB file: {nwb_path}")
    print(f"File size: {nwb_path.stat().st_size / (1024*1024):.1f} MB")
    print(f"Output directory: {output_dir}")
    print(f"Output directory exists: {output_dir.exists()}")
    
    if not output_dir.exists():
        print(f"\nNote: Output directory will be created at: {output_dir}")
    
    response = input("\nProceed with conversion? [y/N]: ").lower().strip()
    return response in ('y', 'yes')


def format_time(seconds: float) -> str:
    """Format time duration in a human-readable format.
    
    Args:
        seconds: Time duration in seconds
        
    Returns:
        Formatted string with appropriate units
    """
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    else:
        td = timedelta(seconds=seconds)
        hours = td.seconds // 3600
        minutes = (td.seconds % 3600) // 60
        seconds = td.seconds % 60
        parts = []
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes}m")
        if seconds > 0 or not parts:
            parts.append(f"{seconds}s")
        return " ".join(parts)


def main(nwb_path: Path, output_dir: Path):
    """Main conversion function.
    
    Args:
        nwb_path: Path to input NWB file
        output_dir: Path to output directory
    """
    start_time = time.time()
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    zarr_filename = output_dir / f"{nwb_path.stem}.zarr"

    print(f"Converting {nwb_path} to ZARR...")
    print(f"Output will be saved to {output_dir}")
    
    with NWBHDF5IO(nwb_path, 'r', load_namespaces=True) as read_io:  # Create HDF5 IO object for read
        with NWBZarrIO(zarr_filename, mode='w') as export_io:         # Create Zarr IO object for write
            export_io.export(src_io=read_io, write_args=dict(link_data=False, chunking="auto"))   # Export from HDF5 to Zarr
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\nConversion completed in {format_time(duration)}")
    print(f"Output saved to: {zarr_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Convert NWB file to SVG track representation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'nwb_filepath',
        type=str,
        help='Path to input NWB file'
    )
    parser.add_argument(
        '--outdir', '-o',
        type=str,
        default='./output',
        help='Path to output directory'
    )
    
    args = parser.parse_args()
    
    try:
        # Validate NWB file path
        nwb_path = validate_nwb_file(args.nwb_filepath)
        output_dir = Path(args.outdir)
        
        # Get confirmation before proceeding
        if not confirm_operation(nwb_path, output_dir):
            print("Operation cancelled by user")
            sys.exit(0)
        
        # Run main conversion
        main(nwb_path, output_dir)
        
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)