from tifffile import TiffFile
import numpy as np
import os
from cellocity.channel import Channel
from cellocity.tiffloader import TiffLoader
from cellocity.analysis import FlowAnalysis, FarenbackAnalyzer, FlowSpeedAnalysis
import tifffile as tif

infile = os.path.abspath(r"C:\Users\Jens\Documents\_Microscopy\FrankenScope2\Calibration stuff\MM_2-0-3_testfiles\5TP-0ms_1Z_3Ch_4pos_1_MMStack_Pos-2-000_000.ome.tif")
outdir = os.path.abspath(r"D:\Napari\flows2")


kwargs = {"scale": 1, "step": 40, "line_thicknes": 2}

def process_channel(infile, outdir, ch_idx = 0):
    file_loader = TiffLoader(infile, debug=False)
    channel = Channel(ch_idx, file_loader)
    analyzer = FarenbackAnalyzer(channel,  "um/min")
    analysis = FlowAnalysis(analyzer)
    analysis.draw_all_flow_frames(scalebarFlag=True, scalebarLength=10, **kwargs)
    analysis.saveFlowAsTif(os.path.join(outdir, "ch_"+str(ch_idx)+".tif"), **kwargs)
    speed_analysis = FlowSpeedAnalysis(analyzer)
    speed_analysis.calculateSpeeds()
    speed_analysis.saveArrayAsTif(outdir)
    file_loader.close()
    return None


if __name__ == "__main__":
    process_channel(infile, outdir)













