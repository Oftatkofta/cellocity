from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QPushButton, QLabel, QFileDialog, QComboBox,
                            QSpinBox, QDoubleSpinBox, QApplication, QScrollArea,
                            QProgressBar, QGroupBox, QCheckBox, QMessageBox, QFormLayout)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
import sys
import pathlib
import tifffile
import numpy as np
import os
import warnings
import re
import statistics
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cellocity.channel import Channel, normalization_to_8bit
from cellocity.analysis import FarenbackAnalyzer, OpenPivAnalyzer, FlowSpeedAnalysis
from tiffloader import TiffLoader

class CellocityGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cellocity Analysis")
        self.setMinimumSize(800, 600)
        
        # Initialize variables
        self.channel = None
        self.analyzer = None
        self.analysis = None
        self.loader = None
        self.position_files = None
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        
        # Left panel for controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        layout.addWidget(left_panel)
        
        # File selection
        file_layout = QHBoxLayout()
        self.file_label = QLabel("No file selected")
        select_button = QPushButton("Select File")
        select_button.clicked.connect(self.load_file)
        file_layout.addWidget(self.file_label)
        file_layout.addWidget(select_button)
        left_layout.addLayout(file_layout)
        
        # Analysis parameters
        self.setupUI()
        
        # Analysis buttons
        button_layout = QHBoxLayout()
        analyze_button = QPushButton("Run Analysis")
        analyze_button.clicked.connect(self.runAnalysis)
        save_button = QPushButton("Save Results")
        save_button.clicked.connect(self.save_results)
        button_layout.addWidget(analyze_button)
        button_layout.addWidget(save_button)
        left_layout.addLayout(button_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        left_layout.addWidget(self.progress_bar)
        
        # Status label
        self.status_label = QLabel("")
        left_layout.addWidget(self.status_label)
        
        # Right panel for preview
        right_panel = QScrollArea()
        right_panel.setWidgetResizable(True)
        right_panel.setMinimumWidth(400)
        layout.addWidget(right_panel)
        
        # Preview label
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignCenter)
        right_panel.setWidget(self.preview_label)

    def setupUI(self):
        """Setup the GUI elements."""
        # Create central widget first and set it
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # Create main layout with parent
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # Create all widgets with proper parent
        self.metadata_label = QLabel(self.central_widget)
        self.metadata_label.setWordWrap(True)
        
        # File selection
        file_layout = QHBoxLayout()
        self.load_button = QPushButton("Load File", self.central_widget)
        self.load_button.clicked.connect(self.load_file)
        file_layout.addWidget(self.load_button)
        self.main_layout.addLayout(file_layout)
        
        # Channel and slice selection
        select_layout = QHBoxLayout()
        
        # Channel selector
        channel_label = QLabel("Channel:", self.central_widget)
        self.channel_select = QComboBox(self.central_widget)
        select_layout.addWidget(channel_label)
        select_layout.addWidget(self.channel_select)
        
        # Z-slice selector
        self.slice_label = QLabel("Z-Slice:", self.central_widget)
        self.slice_select = QSpinBox(self.central_widget)
        self.slice_select.setMinimum(0)
        self.slice_select.setValue(0)
        select_layout.addWidget(self.slice_label)
        select_layout.addWidget(self.slice_select)
        
        # Hide slice selector initially
        self.slice_label.hide()
        self.slice_select.hide()
        
        self.main_layout.addLayout(select_layout)
        
        # Frame range
        self.range_group = QGroupBox("Frame Range", self.central_widget)
        range_layout = QFormLayout()
        
        self.use_range = QCheckBox("Use Range", self.range_group)
        range_layout.addRow(self.use_range)
        
        self.range_start = QSpinBox(self.range_group)
        self.range_start.setMinimum(0)
        range_layout.addRow("Start:", self.range_start)
        
        self.range_stop = QSpinBox(self.range_group)
        self.range_stop.setMinimum(1)
        range_layout.addRow("Stop:", self.range_stop)
        
        self.range_group.setLayout(range_layout)
        self.main_layout.addWidget(self.range_group)
        
        # Add metadata label to layout
        self.main_layout.addWidget(self.metadata_label)
        
        # Progress and status
        self.progress_bar = QProgressBar(self.central_widget)
        self.main_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel(self.central_widget)
        self.main_layout.addWidget(self.status_label)
        
        # Run button
        self.run_button = QPushButton("Run Analysis", self.central_widget)
        self.run_button.clicked.connect(self.runAnalysis)
        self.main_layout.addWidget(self.run_button)

    def load_file(self):
        """Load a TIFF file and update the GUI."""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open TIFF File",
            "",
            "TIFF Files (*.tif *.tiff);;All Files (*)"
        )
        if filename:
            try:
                self.loader = TiffLoader(filename)
                self.updateMetadata()
                self.status_label.setText(f"Loaded: {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error loading file: {str(e)}")
                self.loader = None

    def get_channel_count(self):
        """Get the number of channels in the loaded file."""
        if not self.loader:
            return 0
        return self.loader.n_channels

    def get_channel_names(self):
        """Get list of channel names if available."""
        if not self.loader:
            return []
        if self.loader.channel_names:
            return self.loader.channel_names
        return [f"Channel {i}" for i in range(self.get_channel_count())]

    def read_metadata(self):
        """Read metadata from the loaded file."""
        if not self.loader:
            return {}
        
        metadata = {}
        metadata['Channels'] = self.loader.n_channels
        metadata['Frames'] = self.loader.n_frames
        metadata['Slices'] = self.loader.n_slices
        metadata['Frame Interval (ms)'] = self.loader.intended_frame_interval_ms
        metadata['Pixel Size (µm)'] = self.loader.pixel_size_um
        
        # Update frame range controls
        self.range_start.setMaximum(self.loader.n_frames - 1)
        self.range_stop.setMaximum(self.loader.n_frames)
        self.range_stop.setValue(self.loader.n_frames)
        
        # Update slice selector
        if self.loader.n_slices > 1:
            self.slice_label.show()
            self.slice_select.show()
            self.slice_select.setMaximum(self.loader.n_slices - 1)
            self.slice_select.setValue(0)  # Reset to first slice
        else:
            self.slice_label.hide()
            self.slice_select.hide()
            self.slice_select.setValue(0)
        
        return metadata

    def create_channel(self):
        """Create a Channel object from the current settings."""
        if not self.loader:
            raise ValueError("No file loaded")
            
        print(f"Debug - loader type: {type(self.loader)}")
        
        channel_idx = self.channel_select.currentIndex()
        slice_idx = self.slice_select.value() if self.loader.n_slices > 1 else 0
        
        return Channel(channel_idx,self.loader, slice_idx)

    def update_preview(self):
        """Update the preview image."""
        try:
            if not self.loader:
                return
                
            channel = self.create_channel()
            array = channel.getArray()
            
            if array.size == 0:
                return
                
            frame = min(self.frame_select.value(), array.shape[0] - 1)
            image = array[frame]
            
            # Update preview display...
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error updating preview: {str(e)}")

    def toggle_range_selection(self, state):
        """Enable/disable range selection spinboxes"""
        enabled = state == Qt.Checked
        self.range_start.setEnabled(enabled)
        self.range_stop.setEnabled(enabled)

    def runAnalysis(self):
        """Run Farneback analysis on the current channel."""
        try:
            channel = self.create_channel()
            
            # Get frame range if specified
            if self.use_range.isChecked():
                start = self.range_start.value()
                stop = self.range_stop.value()
                channel.trim(start, stop)
            
            # Create Farneback analyzer
            self.analyzer = FarenbackAnalyzer(channel)
            
            self.progress_bar.setValue(100)
            self.status_label.setText("Analysis complete")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error in analysis: {str(e)}")

    def save_results(self):
        """Save analysis results"""
        if not self.analysis:
            self.status_label.setText("Please run analysis first")
            return
            
        try:
            # Get save directory
            save_dir = QFileDialog.getExistingDirectory(
                self,
                "Select Save Directory"
            )
            
            if save_dir:
                save_path = pathlib.Path(save_dir)
                
                # Create base filename from channel name and parameters
                base_name = pathlib.Path(self.loader.file_path).stem
                if self.use_range.isChecked():
                    base_name += f"_frames_{self.range_start.value()}-{self.range_stop.value()}"
                base_name += f"_Farneback"
                base_name += f"_{self.unit_select.currentText().replace('/', '_per_')}"
                
                # Save flow array as TIFF
                flow_path = save_path / f"{base_name}_flow.tif"
                self.analyzer.saveArrayAsTif(flow_path)
                
                # Save speed data as CSV
                csv_path = save_path / f"{base_name}_speeds.csv"
                self.analysis.saveCSV(csv_path)
                
                self.status_label.setText(f"Results saved to {save_dir}")
                
        except Exception as e:
            self.status_label.setText(f"Error saving results: {str(e)}")
            print(f"Save error: {str(e)}")
            import traceback
            traceback.print_exc()

    def updateMetadata(self):
        """Update metadata display and controls based on loaded file."""
        if not self.loader:
            return
        
        # Update channel selector with proper names
        self.channel_select.clear()
        if self.loader.channel_names:
            self.channel_select.addItems(self.loader.channel_names)
        else:
            self.channel_select.addItems([f"Channel {i}" for i in range(self.loader.n_channels)])
        
        # Update slice selector visibility and range
        if self.loader.n_slices > 1:
            self.slice_select.show()
            self.slice_label.show()
            self.slice_select.setMaximum(self.loader.n_slices - 1)
            self.slice_select.setValue(0)
        else:
            self.slice_select.hide()
            self.slice_label.hide()
            self.slice_select.setValue(0)
        
        # Update frame range
        self.range_start.setMinimum(0)
        self.range_start.setMaximum(self.loader.n_frames - 1)
        self.range_start.setValue(0)
        
        self.range_stop.setMinimum(1)
        self.range_stop.setMaximum(self.loader.n_frames)
        self.range_stop.setValue(self.loader.n_frames)
        
        # Update metadata text
        metadata_text = [
            f"File: {self.loader.file_path}",
            f"Channels: {self.loader.n_channels}",
            f"Channel Names: {', '.join(self.loader.channel_names) if self.loader.channel_names else 'Not specified'}",
            f"Frames: {self.loader.n_frames}",
            f"Slices: {self.loader.n_slices}",
            f"Frame Interval (ms): {self.loader.intended_frame_interval_ms}",
            f"Pixel Size (µm): {self.loader.pixel_size_um}"
        ]
        
        self.metadata_label.setText("\n".join(metadata_text))

def main():
    app = QApplication(sys.argv)
    window = CellocityGUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 