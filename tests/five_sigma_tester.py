from pathlib import Path
import tifffile
from cellocity.channel import Channel, MedianChannel
from cellocity.analysis import FarenbackAnalyzer, OpenPivAnalyzer, FiveSigmaAnalysis
from cellocity.validation import make_channels, make_fb_flow_analyzer, make_piv_analyzer
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

def main():

    inpath = Path(r"C:\Users\Jens\Desktop\S-BSST461")
    outpath = Path(r"C:\Users\Jens\Desktop\temp")
    ch_list = make_channels(inpath)
    for ch in ch_list:
        print(ch.name)
        a1 = make_fb_flow_analyzer(ch)
        fsig = FiveSigmaAnalysis(a1)
        fsig.calculateCorrelationOneFrame(5)
        fsig.saveCSV(outdir=outpath, fname=fsig.analyzer.channel.name+"opt_Cvv.csv")
        a2 = make_piv_analyzer(ch)
        fsig = FiveSigmaAnalysis(a2)
        fsig.calculateCorrelationOneFrame(5)
        fsig.saveCSV(outdir=outpath, fname=fsig.analyzer.channel.name + "piv_Cvv.csv")

if __name__ == "__main__":
    main()
