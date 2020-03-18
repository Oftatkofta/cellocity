import time, os
import tifffile
from cellocity.channel import Channel,
from matplotlib import pyplot as plt
import numpy as np
"""
A quick sanity check on the flow analyzer using images of a fixed monlayer
translated 1 um in x, y, or xy between frames. It's not a time lapse stack, so
some massaging of the Channel objects will have to be done 
"""


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

analyzeFiles(filelist[0:2], outdir, flowkwargs, scalebarFlag, scalebarLength)

if __name__ == '__main__':
    gamma = r"C:\Users\Jens\Microscopy\MMgamma_demodata\dummydata_2\dummydata_2_MMStack_Position_a.ome.tif"
    beta = r"C:\Users\Jens\Microscopy\MMgamma_demodata\dummydata_beta_1\dummydata_beta_1_MMStack_Pos0.ome.tif"
    onefour = r"C:\Users\Jens\Microscopy\MMgamma_demodata\dummy1_4_1\dummy1_4_1_MMStack_Pos0.ome.tif"
    fs_beta = r"C:\Users\Jens\Microscopy\FrankenScope2\_Pilar\Multi channel\plate3_Mss109_to15min_every20sec_1_MMStack_A_b.ome.tif"
    fs_beta2 = r"C:\Users\Jens\Microscopy\FrankenScope2\_Pilar\Multi channel\raw_STm infection_10x__1_MMStack_MOI2.ome.tif"
    fs_onefour = r"C:\Users\Jens\Downloads\SG_Mitotracker-green_Lysotracker-red_cellROX-deepRed_post_1_MMStack_3-Pos_001_001.ome.tif"
    imageJ = r"C:\Users\Jens\Desktop\laminin_2_MMStack_Control_2-5E4.ome.tif"
    ij2 = r"C:\Users\Jens\Desktop\MC_1.tif"
    hs = r"C:\Users\Jens\Desktop\HyperStack_345.tif"


    # filelist = [gamma, beta, onefour, fs_beta]
    filelist = [imageJ, hs]

    outdir = r"C:\Users\Jens\Desktop\temp"

    flowkwargs = {"step": 15, "scale": 20, "line_thicknes": 2}
    scalebarFlag = True
    scalebarLength = 1

    analyzeFiles(filelist, outdir, flowkwargs, scalebarFlag, scalebarLength)