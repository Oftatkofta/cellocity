"""Module for handling channel data."""

import statistics
import numpy as np
import tifffile
import os
import warnings
import gc
import json



class Channel:
    """Class representing a single channel of data."""
    
    def __init__(self, channel_idx, tiff_loader, slice_idx=0, debug=True):
        """Initialize Channel object."""
        self.channel_idx = channel_idx
        self.tiff_loader = tiff_loader
        self.slice_idx = slice_idx
        self.debug = debug
        self.array = None
        self.name = None
        self.frame_interval_ms = None
        self.pxSize_um = None
        self.elapsedTimes_ms = None
        self.frame_interval_ms = tiff_loader.intended_frame_interval_ms
        self.pxSize_um = tiff_loader.pixel_size_um
        self.name = tiff_loader.channel_names[channel_idx] if tiff_loader.channel_names else f"ch{channel_idx}"
        self.actualFrameInterval_ms = None

        if self.debug:
            print(f"Channel name: {self.name}")
        
        # Extract the array data and elapsed times
        self.array, self.elapsedTimes_ms = self._extract_channel(self.channel_idx, self.tiff_loader, slice_idx)
        self.actualFrameInterval_ms = self.getActualFrameInterval_ms()

        if self.debug:
            print(f"Loaded channel {channel_idx}, slice {slice_idx}")
            print(f"Array shape: {self.array.shape}")
            print(f"Pixel size: {self.pxSize_um} µm")
            print(f"Frame interval: {self.frame_interval_ms} ms")
            print(f"Elapsed times: {self.elapsedTimes_ms[:5]}...")

    def _extract_channel(self, channel_idx, tiff_loader, slice_idx):
        """Extract channel data from tiff_loader."""
        if self.debug:
            print(f"Extracting channel {channel_idx}, slice {slice_idx}")
        
        return tiff_loader.extract_channel(channel_idx, slice_idx)

    def getActualFrameInterval_ms(self):
        """Return the actual frame intervals."""
        if self.actualFrameInterval_ms is None:
            self.actualFrameInterval_ms = np.diff(self.elapsedTimes_ms).mean()
        
        return self.actualFrameInterval_ms

    def getArray(self):
        """Return the array data."""
        return self.array

    def getElapsedTimes_ms(self):
        """Return the elapsed times."""
        return self.elapsedTimes_ms

    def getIntendedFrameInterval_ms(self):
        """Return the intended frame interval."""
        return self.frame_interval_ms

    def getPixelSize_um(self):
        """Return the pixel size."""
        return self.pxSize_um
    
    def close(self):
        """Clean up resources."""
        # Add any cleanup needed
        pass


class MedianChannel(Channel):
    """
    A subclass of ``Channel`` where the channel array has been temporal median filtered.

    Temporal median filtering is very useful when performing optical flow based analysis of time lapse microscopy data,
    because it filters out fast moving free-floating debris from the dataset. Note that the median array will be shorter
    than the original array. In the default case, if a temporal median of 3 frames is applied, then the the output
    array will contain 3-1 = 2 frames less than the input if a gliding projection (default) is performed.
    
    """
    
    def __init__(self, channel, doGlidingProjection=True, frameSamplingInterval=3, startFrame=0, stopFrame=None, debug=False):
        """Initialize MedianChannel."""
        # Copy attributes from parent channel
        self.channel_idx = channel.channel_idx
        self.tiff_loader = channel.tiff_loader
        self.slice_idx = channel.slice_idx
        self.debug = debug
        self.name = f"{channel.name}_MED"
        self.frame_interval_ms = channel.frame_interval_ms
        self.pxSize_um = channel.pxSize_um
        self.actualFrameInterval_ms = channel.actualFrameInterval_ms
        # Fields specific for MedianChannel
        self.parent_channel = channel
        self.doGlidingProjection = doGlidingProjection
        self.frameSamplingInterval = frameSamplingInterval
        self.startFrame = startFrame
        

        # Get number of frames from array
        n_frames = channel.getArray().shape[0]
        self.stopFrame = n_frames if stopFrame is None else stopFrame
        
        if self.debug:
            print(f"\nInitializing MedianChannel:")
            print(f"  Parent channel: {channel.name}")
            print(f"  Frame sampling interval: {frameSamplingInterval}")
            print(f"  Start frame: {startFrame}")
            print(f"  Stop frame: {self.stopFrame}")
            print(f"  Gliding projection: {doGlidingProjection}")
            print(f"  Parent array shape: {channel.getArray().shape}")
            print(f"  Parent elapsed times: {channel.getElapsedTimes_ms()}")
        
        # Calculate median array
        self.array = self.getTemporalMedianFilter(
            doGlidingProjection=self.doGlidingProjection,
            startFrame=self.startFrame,
            stopFrame=self.stopFrame,
            frameSamplingInterval=self.frameSamplingInterval
        )
        
        if self.debug:
            print(f"  Median array shape: {self.array.shape}")
        
        # Update timing info
        self.frame_interval_ms = self._recalculate_finterval()
        self.elapsedTimes_ms = self._recalculate_elapsed_times()
        
        
        if self.debug:
            print(f"  Frame interval: {self.frame_interval_ms} ms")
            print(f"  Elapsed times: {self.elapsedTimes_ms}")
            print(f"  Actual intervals: {self.actualFrameInterval_ms}")


    def getTemporalMedianFilter(self, doGlidingProjection, startFrame, stopFrame, frameSamplingInterval):
        """Return temporal median filter of the parent Channel."""
        # Input validation
        if stopFrame > self.parent_channel.getArray().shape[0]:
            raise ValueError("stopFrame cannot be larger than number of frames!")
        if startFrame >= stopFrame:
            raise ValueError("startFrame cannot be larger than or equal to stopFrame!")
        if stopFrame - startFrame < frameSamplingInterval:
            raise ValueError("Not enough frames selected for median projection!")
            
        # Get input array
        arr = self.parent_channel.getArray()
        
        # Calculate output shape
        if doGlidingProjection:
            n_frames = stopFrame - startFrame - (frameSamplingInterval - 1)
        else:
            n_frames = (stopFrame - startFrame) // frameSamplingInterval
            
        out_shape = (n_frames, arr.shape[1], arr.shape[2])
        out = np.empty(out_shape, dtype=np.float32)
        
        # Fill output array
        frame_idx = 0
        if doGlidingProjection:
            for i in range(startFrame, stopFrame - frameSamplingInterval + 1):
                out[frame_idx] = np.median(arr[i:i + frameSamplingInterval], axis=0)
                frame_idx += 1
        else:
            for i in range(startFrame, stopFrame, frameSamplingInterval):
                if frame_idx >= n_frames:
                    break
                out[frame_idx] = np.median(arr[i:i + frameSamplingInterval], axis=0)
                frame_idx += 1
                
        return out

    
    def _recalculate_elapsed_times(self):
        """Recalculate elapsed times based on projection type."""
        if self.array is None:
            return []
            
        # Get base times or calculate from interval
        base_times = self.parent_channel.getElapsedTimes_ms()
        if len(base_times) == 0:  # Check array length instead of truth value
            interval = self.parent_channel.getIntendedFrameInterval_ms()
            base_times = np.array([i * interval for i in range(self.parent_channel.getArray().shape[0])])
        
        out = []
        if self.doGlidingProjection:
            # For gliding projection, use the center frame's timestamp
            n_frames = self.stopFrame - self.startFrame - (self.frameSamplingInterval - 1)
            interval = self.parent_channel.getIntendedFrameInterval_ms()
            
            for i in range(n_frames):
                center_idx = self.startFrame + i + self.frameSamplingInterval // 2
                if center_idx < len(base_times):
                    out.append(base_times[center_idx])
                else:
                    # Calculate time from interval if beyond base timestamps
                    frame_time = center_idx * interval
                    out.append(frame_time)
        else:
            # For block projection, use median of timestamps in each block
            n_frames = (self.stopFrame - self.startFrame) // self.frameSamplingInterval
            interval = self.parent_channel.getIntendedFrameInterval_ms()
            
            for i in range(self.startFrame, self.stopFrame, self.frameSamplingInterval):
                if len(out) >= n_frames:
                    break
    
                # Get timestamps for this block
                end_idx = min(i + self.frameSamplingInterval, len(base_times))
                block_times = base_times[i:end_idx]
                
                if len(block_times) > 0:  # Check array length instead of truth value
                    # Use median of available timestamps
                    out.append(statistics.median(block_times))
                else:
                    # Calculate time from interval if beyond base timestamps
                    frame_time = i * interval
                    out.append(frame_time)
        
        return out
    
    def _recalculate_finterval(self):
        """Recalculate frame interval based on projection type."""
        base_interval = self.parent_channel.getIntendedFrameInterval_ms()
        if base_interval is None:
            base_interval = 1  # Default to 1s if not available
    
        if self.doGlidingProjection:
            # Gliding projection keeps the same interval
            return base_interval
        else:
            # Block projection increases interval by sampling factor
            return base_interval * self.frameSamplingInterval


def normalization_to_8bit(image_stack, lowPcClip = 0.175, highPcClip = 0.175):
    """
    Function to rescale 16/32/64 bit arrays to 8-bit for visualizing output

    Defaults to saturate 0.35% of pixels, 0.175% in each end by default, which often produces nice results. This
    is the same as pressing 'Auto' in the ImageJ contrast manager. `numpy.interp()` linear interpolation is used
    for the mapping.
    
    :param image_stack: 3D Numpy array to be rescaled
    :type image_stack: Numpy array
    :param lowPcClip: Fraction for black clipping bound
    :type lowPcClip: float
    :param highPcClip: Fraction for white/saturated clipping bound
    :type highPcClip: float
    :return: 8-bit numpy array of the same shape as :param image_stack:
    :rtype: numpy.dtype('uint8')
    
    """


    #clip image to saturate 0.35% of pixels 0.175% in each end by default.
    low = int(np.percentile(image_stack, lowPcClip))
    high = int(np.percentile(image_stack, 100 - highPcClip))
    
    # use linear interpolation to find new pixel values
    image_equalized = np.interp(image_stack.flatten(), (low, high), (0, 255))
    
    return image_equalized.reshape(image_stack.shape).astype('uint8')

def rehape3DArrayTo6D(array_3d):
    """
    reshapes 3D (t, x, y) array to (t, 1, 1, x, y, 1).

    Used when saving ImageJ compatible tifs using ``Tifffile`` where dimensions have to be in TZCYXS order.
    :param array_3d: 3D numpy array
    :return:
    
    """
    assert len(array_3d.shape) == 3, "Must use 3D array!"
    array_3d.shape = (array_3d.shape[0], 1, 1, array_3d.shape[1], array_3d.shape[2], 1)

def reshape6DArrayTo3D(array_6D):
    """
    Undoes what reshape3DArrayTo6D does to the shape of the array.

    :param array_6D: 6D numpy array
    :return:
    
    """
    assert len(array_6D.shape) == 6, "Must use 6D array!"
    array_6D.shape = (array_6D.shape[0], array_6D.shape[3], array_6D.shape[4])








