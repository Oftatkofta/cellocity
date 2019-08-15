import time, os
import tifffile
from MM_flow_analyzer import *
from matplotlib import pyplot as plt
import numpy as np
"""
A quick sanity check on the flow analyzer using images of a fixed monlayer
translated 1 um in x, y, or xy between frames. It's not a time lapse stack, so
some massaging of the Channel objects will have to be done 
"""

indir = r"C:\Users\Jens\Microscopy\FrankenScope2\_Pilar\DIC_truth"
outdir = r"C:\Users\Jens\Microscopy\FrankenScope2\_Pilar\DIC_truth\out2"
filelist = []
for fname in os.listdir(indir):
    if os.path.isfile(os.path.join(indir, fname)):
        filelist.append(os.path.join(indir, fname))

def makeFakeChannel(tif, chidx, desiredinterval):
    #creates a fake channel object from a stack to emulate time lapse images
    Ch0 = Channel(chidx, tif, name=lab + "_Ch1")
    return None

flowkwargs = {"step": 15, "scale": 20, "line_thicknes": 2}
scalebarFlag = True
scalebarLength = 1


def analyzeFiles(fnamelist, outdir, flowkwargs, scalebarFlag, scalebarLength):
    """
    Automatically analyzes tifffiles annd saves ouput in outfolder. If input has two channels, analysis is run on
    channel index 1
    :param tif: Tifffile objekt
    :return:
    """
    for fname in fnamelist:

        with tifffile.TiffFile(fname, multifile=False) as tif:
            lab = os.path.split(fname)[1][:-8]
            print("Working on: {} as {}".format(fname, lab))
            t1 = time.time()
            ij_metadata = tif.imagej_metadata
            n_channels = int(tif.micromanager_metadata['Summary']['Channels'])

            Ch0 = Channel(0, tif, name=lab + "_Ch1")
            print("Elapsed for file load and Channel creation {:.2f} s.".format(time.time() - t1))

            print("Actual frame interval is: {:.2f} s".format(Ch0.getActualFrameIntevals_ms().mean()/1000))

            print("Start median filter of Channel 1...")
            Ch0.getTemporalMedianFilterArray()
            print("Elapsed for file {:.2f} s, now calculating Channel 1 flow...".format(time.time() - t1))
            Analysis_Ch0 = FarenbackAnalyzer(Ch0)
            #Analysis_Ch0.scaler = Ch0.pxSize_um * frames_per_min #Cheat
            Analysis_Ch0.doFarenbackFlow()

            print("flow finished, calculating speeds...")
            Analysis_Ch0.doFlowsToSpeed()

            arr = np.average(Analysis_Ch0.speeds, axis=(1, 2))
            plt.plot(arr, label=lab[20:-10]+"-"+str(Ch0.getActualFrameIntevals_ms().mean()))
            print("Elapsed for file {:.2f} s, now drawing flow...".format(time.time() - t1))

            print("File done!")
    plt.legend()
    plt.show()
    return True

analyzeFiles(filelist, outdir, flowkwargs, scalebarFlag, scalebarLength)