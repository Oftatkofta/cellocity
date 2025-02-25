from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QPushButton, QLabel, QFileDialog, QComboBox,
                            QSpinBox, QDoubleSpinBox, QApplication)
from PyQt5.QtCore import Qt
import sys
import pathlib
from cellocity.channel import Channel
from cellocity.analysis import FarenbackAnalyzer, OpenPivAnalyzer, FlowSpeedAnalysis

class CellocityGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cellocity Analysis")
        self.setMinimumSize(600, 400)
        
        # Initialize variables
        self.channel = None
        self.analyzer = None
        self.analysis = None
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # File selection
        file_layout = QHBoxLayout()
        self.file_label = QLabel("No file selected")
        select_button = QPushButton("Select File")
        select_button.clicked.connect(self.select_file)
        file_layout.addWidget(self.file_label)
        file_layout.addWidget(select_button)
        layout.addLayout(file_layout)
        
        # Analysis parameters
        param_layout = QVBoxLayout()
        
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
        
        # Channel selection
        channel_layout = QHBoxLayout()
        channel_layout.addWidget(QLabel("Channel:"))
        self.channel_select = QSpinBox()
        self.channel_select.setMinimum(0)
        channel_layout.addWidget(self.channel_select)
        param_layout.addLayout(channel_layout)
        
        layout.addLayout(param_layout)
        
        # Analysis buttons
        button_layout = QHBoxLayout()
        analyze_button = QPushButton("Run Analysis")
        analyze_button.clicked.connect(self.run_analysis)
        save_button = QPushButton("Save Results")
        save_button.clicked.connect(self.save_results)
        button_layout.addWidget(analyze_button)
        button_layout.addWidget(save_button)
        layout.addLayout(button_layout)
        
        # Status label
        self.status_label = QLabel("")
        layout.addWidget(self.status_label)

    def select_file(self):
        """Open file dialog to select input image file"""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image File",
            "",
            "Image Files (*.tif *.tiff);;All Files (*)"
        )
        
        if filename:
            self.file_label.setText(filename)
            try:
                # Create channel from file
                self.channel = Channel(
                    chIndex=self.channel_select.value(),
                    tiffFile=filename,
                    name=pathlib.Path(filename).stem
                )
                self.status_label.setText("File loaded successfully")
            except Exception as e:
                self.status_label.setText(f"Error loading file: {str(e)}")

    def run_analysis(self):
        """Run the selected analysis"""
        if not self.channel:
            self.status_label.setText("Please select a file first")
            return
            
        try:
            # Create analyzer based on selection
            if self.analysis_type.currentText() == "Farenback":
                self.analyzer = FarenbackAnalyzer(
                    self.channel,
                    self.unit_select.currentText()
                )
                self.analyzer.doFarenbackFlow()
            else:
                self.analyzer = OpenPivAnalyzer(
                    self.channel,
                    self.unit_select.currentText()
                )
                self.analyzer.doOpenPIV()
                
            # Create speed analysis
            self.analysis = FlowSpeedAnalysis(self.analyzer)
            self.analysis.calculateSpeeds()
            
            self.status_label.setText("Analysis completed successfully")
            
        except Exception as e:
            self.status_label.setText(f"Error during analysis: {str(e)}")

    def save_results(self):
        """Save analysis results"""
        if not self.analysis:
            self.status_label.setText("Please run analysis first")
            return
            
        save_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Save Directory"
        )
        
        if save_dir:
            try:
                save_path = pathlib.Path(save_dir)
                self.analysis.saveArrayAsTif(save_path)
                self.analysis.saveCSV(save_path)
                self.status_label.setText("Results saved successfully")
            except Exception as e:
                self.status_label.setText(f"Error saving results: {str(e)}")

def main():
    app = QApplication(sys.argv)
    window = CellocityGUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 