import napari
from qtpy.QtWidgets import QWidget, QVBoxLayout, QPushButton, QFileDialog
from pathlib import Path
import numpy as np

from cellocity.channel import Channel
from cellocity.analysis import FarenbackAnalyzer
from cellocity.tiffloader import TiffLoader

import os
from magicgui import magicgui

# Define a list of distinct colormaps for different Z-slices
COLORMAPS = ['red', 'green', 'blue', 'magenta', 'cyan', 'yellow', 'gray', 'viridis', 'plasma', 'inferno']

class CellocityNapariGUI(QWidget):
    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer = viewer
        self.loader = None
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the minimal GUI elements."""
        layout = QVBoxLayout()
        
        # Just a load button for now
        self.load_button = QPushButton("Load File")
        self.load_button.clicked.connect(self.load_file)
        layout.addWidget(self.load_button)
        
        self.setLayout(layout)
        
    def load_file(self):
        """Load a TIFF file and analyze all Z-slices."""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open TIFF File",
            "",
            "TIFF Files (*.tif *.tiff);;All Files (*)"
        )
        if filename:
            try:
                # Load the file
                self.loader = TiffLoader(filename)
                
                # For each Z-slice
                for z in range(self.loader.n_slices):
                    # Get the first channel for this Z-slice
                    channel = Channel(self.loader, channel_idx=0, slice_idx=z)
                    array = channel.getArray()
                    
                    # Add raw data to viewer
                    self.viewer.add_image(
                        array,
                        name=f"Z{z}_Raw",
                        scale=[1, self.loader.pixel_size_um, self.loader.pixel_size_um]
                    )
                    
                    # Do Farneback analysis
                    analyzer = FarenbackAnalyzer(channel, unit="um/s")
                    flows = analyzer._getFlows()
                    
                    # Add U and V components to viewer with color coding
                    u = flows[..., 0]  # U component
                    v = flows[..., 1]  # V component
                    
                    # Use a different colormap for each Z-slice
                    colormap = COLORMAPS[z % len(COLORMAPS)]
                    
                    self.viewer.add_image(
                        u,
                        name=f"Z{z}_U",
                        colormap=colormap,
                        scale=[1, self.loader.pixel_size_um, self.loader.pixel_size_um]
                    )
                    self.viewer.add_image(
                        v,
                        name=f"Z{z}_V",
                        colormap=colormap,
                        scale=[1, self.loader.pixel_size_um, self.loader.pixel_size_um]
                    )
                    
            except Exception as e:
                print(f"Error loading file: {str(e)}")

def main():
    # Create napari viewer with 3D display
    viewer = napari.Viewer(ndisplay=3)
    
    # Create and add our widget
    widget = CellocityNapariGUI(viewer)
    viewer.window.add_dock_widget(widget, name="Cellocity")
    
    # Start napari
    napari.run()

if __name__ == "__main__":
    main()


