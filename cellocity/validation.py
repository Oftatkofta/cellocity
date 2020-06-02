import time, os
from pathlib import Path
import tifffile
from cellocity.channel import Channel
from cellocity.analysis import FarenbackAnalyzer, OpenPivAnalyzer
from matplotlib import pyplot as plt
import numpy as np

"""
A quick sanity check on the flow analyzer using images of a fixed monolayer
translated 1 um in x, y, or xy between frames. It's not a time lapse stack, so
some massaging of the Channel objects will have to be done 
"""
inpath = Path(r"C:\Users\Jens\Documents\_Microscopy\FrankenScope2\Calibration stuff\DIC_truth")

def convertChannel(fname, finterval=1):
    """
    Converts a mulitiposition MM file to a fake timelapse Channel with finterval second frame interval.

    :param fname: Path to file
    :param finterval: desired frame interval in output ``Channel``, defaults to 1 second
    :return: Channel
    :rtype: cellocity.Channel
    """

    with tifffile.TiffFile(fname, multifile=False) as tif:
        name = str(fname).split(".")[0]
        name = name.split("\\")[-1][:-8]
        ch = Channel(0, tif, name)
        ch.finterval_ms = finterval * 1000

    return ch

def make_channels(inpath):
    """
    Creates a list of Channel objects from files in inPath.

    :param inpath: Path
    :return: list of Channels
    :rtype: list
    """

    out=[]

    for f in inpath.iterdir():
        if (f.suffix == ".tif") and f.is_file():
            out.append(convertChannel(f))

    return out

ch_list = make_channels(inpath)

for ch in ch_list:
    a1 = FarenbackAnalyzer(ch, "um/s")
    a2 = OpenPivAnalyzer(ch, "um/s")
    a2.doOpenPIV()
    print(a1, a2)

def analyzeFiles(fnamelist, outdir, flowkwargs, scalebarFlag, scalebarLength):
    """
    Automatically analyzes tifffiles annd saves ouput in outfolder. If input has two channels, analysis is run on
    channel index 1
    """
    for fname in fnamelist:

        with tifffile.TiffFile(fname, multifile=False) as tif:
            lab = os.path.split(fname)[1][:-8] #lable string
            print("Working on: {} as {}".format(fname, lab))
            t1 = time.time()
            ij_metadata = tif.imagej_metadata
            n_channels = int(tif.micromanager_metadata['Summary']['Channels'])

            Ch0 = Channel(0, tif, name=lab + "_Ch1")
            print("Elapsed for file load and Channel creation {:.2f} s.".format(time.time() - t1))
            print("Actual frame interval is: {:.2f} s".format(Ch0.finterval_ms))
            print("Corrected frame interval is: {:.2f} s".format(Ch0.getActualFrameIntevals_ms().mean()/1000))

            print("Start median filter of Channel 1...")
            Ch0.getTemporalMedianFilterArray()
            print("Elapsed for file {:.2f} s, now calculating Channel 1 flow...".format(time.time() - t1))

            Analysis_Ch0 = FarenbackAnalyzer(Ch0, "um/2")
            Analysis_Ch0.doFarenbackFlow()

            print("flow finished, calculating speeds...")
            Analysis_Ch0.doFlowsToSpeed(doHist=True, nbins=100, hist_range=None)


            bins = Analysis_Ch0.histograms[1]
            width = 0.7 * (bins[1] - bins[0])
            center = (bins[:-1] + bins[1:]) / 2
            for i in range(len(Analysis_Ch0.histograms[0])):
                hist = Analysis_Ch0.histograms[0][i]
                plt.bar(center, hist, align='center', width=width)
                savename = Analysis_Ch0.channel.name+"_"+str(i)
                plt.savefig(os.path.join(outdir, savename))
                plt.close()

            print("File done!")
    return True
