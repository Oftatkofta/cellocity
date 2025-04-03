"""Module for handling TIFF file loading and metadata extraction."""

import numpy as np
from pathlib import Path
import json
import xml.etree.ElementTree as ET
import tifffile


class TiffLoader:
    """Class for loading and handling TIFF files with metadata."""
    
    def __init__(self, file_path, debug=False):
        """Initialize TiffLoader with a file path."""
        self.debug = debug
        self.file_path = Path(file_path)
        self.tiff = None
        self.metadata = None
        self.intended_frame_interval_ms = None
        self.pixel_size_um = None
        self.channel_names = None
        self.position = None
        self.n_channels = None
        self.n_frames = None
        self.n_slices = None
        self.z_interval_um = None

        # Load the file
        self._load_file()
        
        # Extract metadata
        self.metadata = self._extract_metadata()
        
        if self.debug:
            print(f"\nInitialized TiffLoader with:")
            print(f"  Channels: {self.n_channels}")
            print(f"  Frames: {self.n_frames}")
            print(f"  Slices: {self.n_slices}")
            print(f"  Image size: {self.tiff.pages[0].shape}")
            print(f"  Pixel size: {self.pixel_size_um} µm")
            print(f"  Frame interval: {self.intended_frame_interval_ms} ms")
            print(f"  Z interval: {self.z_interval_um} µm")

    def _load_file(self):
        """Load the TIFF file."""
        try:
            self.tiff = tifffile.TiffFile(self.file_path)
        except Exception as e:
            raise IOError(f"Failed to load TIFF file: {e}")

    def _extract_metadata(self):
        """Extract metadata from the TIFF file."""
        if self.debug:
            print("\nExamining file structure:")
            print(f"Number of pages: {len(self.tiff.pages)}")
            if hasattr(self.tiff, 'series'):
                print(f"Number of series: {len(self.tiff.series)}")
                for i, series in enumerate(self.tiff.series):
                    print(f"Series {i} shape: {series.shape}")
                    print(f"Series {i} axes: {series.axes}")

        # Get MicroManager metadata using tifffile's built-in handling
        if hasattr(self.tiff, 'micromanager_metadata'):
            if self.debug:
                print("\nFound MicroManager metadata:")
                print(self.tiff.micromanager_metadata)
            
            mm_metadata = self.tiff.micromanager_metadata
            metadata = {}
            
            # Get metadata from Summary
            if 'Summary' in mm_metadata:
                summary = mm_metadata['Summary']
                
                # Frame interval - use Interval_ms from Summary
                if 'Interval_ms' in summary:
                    metadata['frame_interval_ms'] = float(summary['Interval_ms'])
                
                # Dimensions
                metadata['n_frames'] = int(summary.get('Frames', 1))
                metadata['n_channels'] = int(summary.get('Channels', 1))
                metadata['n_slices'] = int(summary.get('Slices', 1))
                metadata['z-step_um'] = abs(float(summary.get('z-step_um', 1)))

                # Channel names
                if 'ChNames' in summary:
                    metadata['channel_names'] = summary['ChNames']
                
                # Position info
                if 'StagePositions' in summary:
                    for pos in summary['StagePositions']:
                        if pos['Label'] in self.file_path.stem:
                            metadata['position'] = pos['Label']
                            break
                    
            
            # Get the first page
            page = self.tiff.pages[0]
            
            # Try to get pixel size from different sources
            pixel_size = None
            
            # 1. Try MicroManager metadata first
            if hasattr(page, 'description'):
                mm_data = page.tags['MicroManagerMetadata'].value
                
                if 'PixelSize_um' in mm_data:
                    pixel_size = float(mm_data['PixelSize_um'])
                elif 'PixelSizeUm' in mm_data:
                    pixel_size = float(mm_data['PixelSizeUm'])
            
            # 2. Try OME metadata if MicroManager didn't have it
            if pixel_size is None and hasattr(self.tiff, 'ome_metadata'):
                ome = self.tiff.ome_metadata
                if ome:
                    # Parse XML string
                    root = ET.fromstring(ome)
                    # Find Pixels element
                    pixels = root.find('.//{http://www.openmicroscopy.org/Schemas/OME/2016-06}Pixels')
                    if pixels is not None:
                        # Try PhysicalSizeX first
                        size_x = pixels.get('PhysicalSizeX')
                        if size_x:
                            pixel_size = float(size_x)
            
            # 3. Try TIFF resolution tags as last resort
            if pixel_size is None and hasattr(page, 'tags'):
                try:
                    x_res = page.tags['XResolution'].value
                    if x_res[1] != 0:  # Avoid division by zero
                        # Convert from resolution to size in micrometers
                        pixel_size = (x_res[1] / x_res[0]) * 1e-3  # assuming resolution is in pixels/mm
                except (KeyError, AttributeError):
                    pass
            metadata['pixel_size_um'] = pixel_size

            print(f"\nPixel size found: {pixel_size} µm")
            

            # Update instance attributes
            self.intended_frame_interval_ms = metadata.get('frame_interval_ms')
            self.pixel_size_um = metadata.get('pixel_size_um')
            self.channel_names = metadata.get('channel_names')
            self.n_channels = metadata.get('n_channels')
            self.n_frames = metadata.get('n_frames')
            self.n_slices = metadata.get('n_slices')
            self.z_interval_um = metadata.get('z-step_um')

            if self.debug:
                print("\nExtracted metadata:")
                for key, value in metadata.items():
                    print(f"  {key}: {value}")
            
            return metadata
            
        return None

    def extract_channel(self, channel_idx, slice_idx=0):
        """Extract a specific channel and z-slice, along with its elapsed times."""
        try:
            # Get the pages for this channel/slice
            pages = []
            elapsed_times = []
            
            if self.debug:
                print(f"\nExtracting frames for channel {channel_idx}, slice {slice_idx}")
            
            # Calculate the correct page indices for this channel
            for t in range(self.n_frames):
                # For TCYX format, the page index is:
                # t * (n_channels * n_slices) + channel_idx * n_slices + slice_idx
                page_idx = t * (self.n_channels * self.n_slices) + channel_idx * self.n_slices + slice_idx
                
                if page_idx < len(self.tiff.pages):
                    page = self.tiff.pages[page_idx]
                    pages.append(page)
                    
                    # Get elapsed time from page's MicroManager metadata
                    try:
                        # Access the frame-specific metadata
                        frame_meta = page.tags['MicroManagerMetadata'].value
                        if 'ElapsedTime-ms' in frame_meta:
                            elapsed_time = float(frame_meta['ElapsedTime-ms'])
                            elapsed_times.append(elapsed_time)
                            if self.debug and t == 0:
                                print(f"Found elapsed time in frame metadata: {elapsed_time} ms")
                        else:
                            # Fallback to calculated time only if actual time not available
                            elapsed_time = t * self.intended_frame_interval_ms
                            elapsed_times.append(elapsed_time)
                            if self.debug and t == 0:
                                print(f"Using calculated elapsed time: {elapsed_time} ms")
                    except Exception as e:
                        if self.debug:
                            print(f"Error getting elapsed time for frame {t}: {e}")
                        elapsed_time = t * self.intended_frame_interval_ms
                        elapsed_times.append(elapsed_time)
            
            if self.debug:
                print(f"Reading {len(pages)} pages")
                print(f"Elapsed times: {elapsed_times}")
            
            # Stack the pages into a 3D array
            array = np.stack([page.asarray() for page in pages])
            
            # Convert elapsed times to numpy array
            elapsed_times = np.array(elapsed_times)
            

            return array, elapsed_times
            
        except Exception as e:
            if self.debug:
                print(f"Error extracting channel: {e}")
                import traceback
                traceback.print_exc()
            raise
    
    def extract_channel_3d(self, channel_idx):
        """Extract a specific channel and retain all z-slices, along with its elapsed times.
        
        Returns a 4D array with dimensions (T, Z, Y, X) for the specified channel.
        """
        try:
            if self.debug:
                print(f"\nExtracting 4D stack for channel {channel_idx}")
            
            # Initialize lists to store data for each timepoint
            timepoint_stacks = []
            elapsed_times = []
            
            # Process each timepoint
            for t in range(self.n_frames):
                # Initialize list to store all z-slices for this timepoint
                z_slices = []
                
                # Get all z-slices for this timepoint and channel
                for z in range(self.n_slices):
                    # Calculate page index: t * (channels * slices) + channel * slices + z
                    page_idx = t * (self.n_channels * self.n_slices) + channel_idx * self.n_slices + z
                    
                    if page_idx < len(self.tiff.pages):
                        page = self.tiff.pages[page_idx]
                        z_slices.append(page.asarray())
                
                # Only record elapsed time once per timepoint (using first z-slice)
                first_z_page_idx = t * (self.n_channels * self.n_slices) + channel_idx * self.n_slices
                
                if first_z_page_idx < len(self.tiff.pages):
                    page = self.tiff.pages[first_z_page_idx]
                    
                    # Get elapsed time from page's MicroManager metadata
                    try:
                        frame_meta = page.tags['MicroManagerMetadata'].value
                        if 'ElapsedTime-ms' in frame_meta:
                            elapsed_time = float(frame_meta['ElapsedTime-ms'])
                            if self.debug and t == 0:
                                print(f"Found elapsed time in frame metadata: {elapsed_time} ms")
                        else:
                            elapsed_time = t * self.intended_frame_interval_ms
                            if self.debug and t == 0:
                                print(f"Using calculated elapsed time: {elapsed_time} ms")
                    except Exception as e:
                        if self.debug:
                            print(f"Error getting elapsed time for frame {t}: {e}")
                        elapsed_time = t * self.intended_frame_interval_ms
                    
                    elapsed_times.append(elapsed_time)
                    
                    # Stack all z-slices for this timepoint into a 3D array (Z, Y, X)
                    if z_slices:
                        timepoint_stack = np.stack(z_slices)
                        timepoint_stacks.append(timepoint_stack)
            
            if self.debug:
                print(f"Extracted {len(timepoint_stacks)} timepoints with {self.n_slices} z-slices each")
                if timepoint_stacks:
                    print(f"Stack shape: {timepoint_stacks[0].shape}")
                print(f"Elapsed times: {elapsed_times[:5]}...")
            
            # Stack all timepoints into a 4D array (T, Z, Y, X)
            array = np.stack(timepoint_stacks)
            
            # Convert elapsed times to numpy array
            elapsed_times = np.array(elapsed_times)
            
            return array, elapsed_times
            
        except Exception as e:
            if self.debug:
                print(f"Error extracting channel: {e}")
                import traceback
                traceback.print_exc()
            raise


    def close(self):
        """Close the TIFF file."""
        if self.tiff is not None:
            self.tiff.close() 