from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QPushButton, QLabel, QFileDialog, QComboBox,
                            QSpinBox, QDoubleSpinBox, QApplication, QScrollArea,
                            QProgressBar, QGroupBox, QCheckBox)
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

class CellocityGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cellocity Analysis")
        self.setMinimumSize(800, 600)
        
        # Initialize variables
        self.channel = None
        self.analyzer = None
        self.analysis = None
        self.tif = None
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
        select_button.clicked.connect(self.select_file)
        file_layout.addWidget(self.file_label)
        file_layout.addWidget(select_button)
        left_layout.addLayout(file_layout)
        
        # Analysis parameters
        param_layout = QVBoxLayout()
        
        # Channel selection
        channel_layout = QHBoxLayout()
        channel_layout.addWidget(QLabel("Channel:"))
        self.channel_select = QComboBox()
        self.channel_select.currentIndexChanged.connect(self.create_channel)
        channel_layout.addWidget(self.channel_select)
        param_layout.addLayout(channel_layout)
        
        # Frame selection
        frame_layout = QHBoxLayout()
        frame_layout.addWidget(QLabel("Frame:"))
        self.frame_select = QSpinBox()
        self.frame_select.setMinimum(0)
        self.frame_select.valueChanged.connect(self.update_preview)
        frame_layout.addWidget(self.frame_select)
        param_layout.addLayout(frame_layout)
        
        # Analysis type selection
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Analysis Type:"))
        self.analysis_type = QComboBox()
        self.analysis_type.addItems(["Farenback", "OpenPIV"])
        type_layout.addWidget(self.analysis_type)
        param_layout.addLayout(type_layout)
        
        # Unit selection
        unit_layout = QHBoxLayout()
        unit_layout.addWidget(QLabel("Unit:"))
        self.unit_select = QComboBox()
        self.unit_select.addItems(["um/s", "um/min", "um/h"])
        unit_layout.addWidget(self.unit_select)
        param_layout.addLayout(unit_layout)
        
        # Add frame range selection after frame preview selection
        range_group = QGroupBox("Analysis Range")
        range_layout = QVBoxLayout()
        
        # Checkbox to enable/disable range selection
        self.use_range = QCheckBox("Analyze subset of frames")
        self.use_range.stateChanged.connect(self.toggle_range_selection)
        range_layout.addWidget(self.use_range)
        
        # Range selection
        range_controls = QHBoxLayout()
        range_controls.addWidget(QLabel("Start:"))
        self.range_start = QSpinBox()
        self.range_start.setMinimum(0)
        self.range_start.setEnabled(False)
        range_controls.addWidget(self.range_start)
        
        range_controls.addWidget(QLabel("Stop:"))
        self.range_stop = QSpinBox()
        self.range_stop.setMinimum(1)
        self.range_stop.setEnabled(False)
        range_controls.addWidget(self.range_stop)
        
        range_layout.addLayout(range_controls)
        range_group.setLayout(range_layout)
        param_layout.addWidget(range_group)
        
        left_layout.addLayout(param_layout)
        
        # Analysis buttons
        button_layout = QHBoxLayout()
        analyze_button = QPushButton("Run Analysis")
        analyze_button.clicked.connect(self.run_analysis)
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

    def get_channel_count(self):
        """Get number of channels in the file"""
        if self.tif is None:
            return 0
            
        if self.tif.is_micromanager:
            # Get from Summary if available
            if 'Summary' in self.tif.micromanager_metadata:
                summary = self.tif.micromanager_metadata['Summary']
                if 'Channels' in summary:
                    return summary['Channels']
            # Fallback to IndexMap
            return len(set(self.tif.micromanager_metadata["IndexMap"]["Channel"]))
        elif self.tif.is_imagej:
            return self.tif.imagej_metadata.get('channels', 1)
        return 1

    def get_channel_names(self):
        """Get channel names from metadata if available"""
        if self.tif is None:
            return []
            
        if self.tif.is_micromanager:
            if 'Summary' in self.tif.micromanager_metadata:
                summary = self.tif.micromanager_metadata['Summary']
                if 'ChNames' in summary:
                    return summary['ChNames']
        return [f"Channel {i}" for i in range(self.get_channel_count())]

    def read_metadata(self, tif):
        """Read and print metadata from tif file"""
        if tif.is_micromanager:
            mm_meta = tif.micromanager_metadata
            mm_keys = list(mm_meta.keys())
            print("\nMicroManager Metadata:")
            print("Keys:", mm_keys)
            for key in mm_keys:
                print(f"{key}::", mm_meta[key])
            
        if tif.is_imagej:
            ij_meta = tif.imagej_metadata
            ij_keys = list(ij_meta.keys())
            print("\nImageJ Metadata:")
            print("Keys:", ij_keys)
            for key in ij_keys:
                print(f"{key}:::", ij_meta[key])

    def select_file(self):
        """Open file dialog to select input image file"""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image File",
            "",
            "Image Files (*.tif *.tiff);;All Files (*)"
        )
        
        if filename:
            try:
                self.file_label.setText(os.path.basename(filename))
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.tif = tifffile.TiffFile(filename)
                
                # Update channel selection
                n_channels = self.get_channel_count()
                channel_names = self.get_channel_names()
                
                self.channel_select.clear()
                self.channel_select.addItems(channel_names)
                
                # Channel will be created by the combobox signal
                self.channel_select.setCurrentIndex(0)
                
            except Exception as e:
                self.status_label.setText(f"Error loading file: {str(e)}")
                print(f"Error loading file: {str(e)}")

    def create_channel(self):
        """Create channel object from current selection"""
        try:
            if self.tif is None:
                return
            
            # Create new channel with debug enabled
            ch_index = self.channel_select.currentIndex()
            label = pathlib.Path(self.tif.filename).stem
            name = f"{label}_Ch{ch_index + 1}"
            
            self.channel = Channel(
                chIndex=ch_index,
                tiffFile=self.tif,
                name=name,
                debug=True  # Enable debug output
            )
            
            # Force array loading and verify
            array = self.channel.getArray()
            if array.size == 0:
                raise ValueError("Failed to load image data")
            
            # Update UI
            self.frame_select.setMaximum(array.shape[0] - 1)
            self.range_stop.setMaximum(array.shape[0])
            self.update_preview()
            
        except Exception as e:
            self.status_label.setText(f"Error: {str(e)}")
            print(f"Error creating channel: {str(e)}")

    def update_preview(self):
        """Update preview when frame changes"""
        if self.channel is None:
            return
        
        try:
            # Get array for current frame
            array = self.channel.getArray()
            if array.size > 0:
                frame_idx = self.frame_select.value()
                if frame_idx < array.shape[0]:
                    frame = array[frame_idx]
                    frame_8bit = normalization_to_8bit(frame)
                    
                    # Convert to QImage
                    height, width = frame_8bit.shape
                    bytes_per_line = width
                    image = QImage(frame_8bit.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
                    
                    # Scale to fit while maintaining aspect ratio
                    pixmap = QPixmap.fromImage(image)
                    scaled_pixmap = pixmap.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    
                    # Display
                    self.preview_label.setPixmap(scaled_pixmap)
                else:
                    self.status_label.setText(f"Frame {frame_idx} out of range")
                    self.preview_label.clear()
            else:
                self.status_label.setText("No image data available")
                self.preview_label.clear()
            
        except Exception as e:
            self.status_label.setText(f"Error updating preview: {str(e)}")
            print(f"Preview error: {str(e)}")
            self.preview_label.clear()

    def toggle_range_selection(self, state):
        """Enable/disable range selection spinboxes"""
        enabled = state == Qt.Checked
        self.range_start.setEnabled(enabled)
        self.range_stop.setEnabled(enabled)

    def run_analysis(self):
        """Run the selected analysis"""
        if not self.channel:
            self.status_label.setText("Please select a file first")
            return
            
        try:
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            
            # If using range, trim the channel
            if self.use_range.isChecked():
                start = self.range_start.value()
                stop = self.range_stop.value()
                if start >= stop:
                    raise ValueError("Start frame must be less than stop frame")
                if start >= self.channel.getArray().shape[0]:
                    raise ValueError("Start frame exceeds number of frames")
                if stop > self.channel.getArray().shape[0]:
                    raise ValueError("Stop frame exceeds number of frames")
                
                self.channel.trim(start, stop)
                if self.channel.getArray().size == 0:
                    raise ValueError("No frames left after trimming")
            
            # Validate metadata
            if self.channel.pxSize_um is None or self.channel.pxSize_um <= 0:
                self.channel.pxSize_um = 1.0
                self.status_label.setText("Warning: Using default pixel size of 1 μm")
            
            if self.channel.finterval_ms is None or self.channel.finterval_ms <= 0:
                self.channel.finterval_ms = 1000.0
                self.status_label.setText("Warning: Using default frame interval of 1s")
            
            # Create analyzer based on selection
            if self.analysis_type.currentText() == "Farenback":
                self.analyzer = FarenbackAnalyzer(
                    self.channel,
                    self.unit_select.currentText()
                )
                self.analyzer.doFarenbackFlow()
                self.progress_bar.setValue(50)
            else:
                self.analyzer = OpenPivAnalyzer(
                    self.channel,
                    self.unit_select.currentText()
                )
                self.analyzer.doOpenPIV()
                self.progress_bar.setValue(50)
                
            # Create speed analysis
            self.analysis = FlowSpeedAnalysis(self.analyzer)
            self.analysis.calculateSpeeds()
            
            self.progress_bar.setValue(100)
            self.status_label.setText("Analysis completed successfully")
            
        except Exception as e:
            self.status_label.setText(f"Error during analysis: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            self.progress_bar.setVisible(False)

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
                base_name = self.channel.name
                if self.use_range.isChecked():
                    base_name += f"_frames_{self.range_start.value()}-{self.range_stop.value()}"
                base_name += f"_{self.analysis_type.currentText()}"
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

def main():
    app = QApplication(sys.argv)
    window = CellocityGUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 