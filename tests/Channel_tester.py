import pytest
import numpy as np
import os
from pathlib import Path
import tifffile
from cellocity.channel import Channel, MedianChannel, normalization_to_8bit
from cellocity.tiffloader import TiffLoader
from cellocity.analysis import FarenbackAnalyzer, FlowAnalysis, FlowSpeedAnalysis, AlignmentIndexAnalysis, IopAnalysis, FiveSigmaAnalysis
import napari

TEST_DATA_DIR = r"C:\Users\Jens\Documents\_Microscopy\FrankenScope2\Calibration stuff\MM_2-0-3_testfiles\subset"
TEST_FILE = TEST_DATA_DIR + r"\5TP-10s_1Z_3Ch_4pos_1_MMStack_Pos-2-000_001.ome.tif"

def test_tiff_loader():
    loader = TiffLoader(TEST_FILE, debug=True)
    chan = loader.extract_channel(0)
    #print(chan)
    
def test_channel():
    loader = TiffLoader(TEST_FILE, debug=True)
    chan = Channel(0, loader, debug=True)
    print("\nTesting MedianChannel creation...")
    median_chan = MedianChannel(chan, debug=True)
    
    # Create Farneback analyzer
    farenback_analyzer = FarenbackAnalyzer(chan, unit="um/s")
    chan_analysis = FlowAnalysis(farenback_analyzer)
    drawn_flow= chan_analysis.draw_all_flow_frames_superimposed()
    speed_analysis = FlowSpeedAnalysis(farenback_analyzer)
    speed_analysis.calculateSpeeds()
    print(speed_analysis.getAvgSpeeds())
    alignment_index_analysis = AlignmentIndexAnalysis(farenback_analyzer)
    alignment_index_analysis.calculateAverage()
    print(alignment_index_analysis.getAvgAlignIdxs())
    iop_analysis = IopAnalysis(farenback_analyzer)
    iop_analysis.calculateIops()
    print(iop_analysis.getIops())
    five_sigma_analysis = FiveSigmaAnalysis(farenback_analyzer)
    five_sigma_analysis.calculateFiveSigma()
    print(five_sigma_analysis.getFiveSigma())

if __name__ == "__main__":
    test_channel()


