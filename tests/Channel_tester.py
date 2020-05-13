from cellocity.channel import Channel, MedianChannel
import cellocity.analysis as analysis
from tifffile import TiffFile
import os
from pathlib import Path
from matplotlib import pyplot as plt

onefour = r"C:\\Users\\Jens\\Documents\\_Microscopy\\FrankenScope2\\_Pilar\\fucci_GFP-geminin_RFP-cdt1_7-5min_1_MMStack_Pos0.ome.tif"
beta = r"C:\Users\Jens\Documents\_Microscopy\FrankenScope2\_Pilar\raw_STm infection_10x__1_MMStack_MOI2.ome.tif"
ij = r"C:\\Users\\Jens\\Documents\\_Microscopy\\FrankenScope2\\_Pilar\\fucci_GFP-geminin_RFP-cdt1_7-5min_IJ.tif"

testfiles={"one4" : onefour,
          "ij" : ij,
          "beta" : beta}

savepath = Path(r"C:\Users\Jens\Desktop\temp")

for testcase in testfiles.keys():
    ch0.trim(4, 16)
    tif = TiffFile(testfiles[testcase])
    ch0 = Channel(0, tif, testcase)
    ch0.trim(4,16)

    n = ch0.getArray()

    for glideFlag in [False, True]:
        print("Glideflag:", glideFlag)
        ch0_median = MedianChannel(ch0,doGlidingProjection=glideFlag, frameSamplingInterval=4)
        print(len(ch0_median.elapsedTimes_ms), ch0_median.elapsedTimes_ms)
        print(ch0_median.getActualFrameIntevals_ms(), ch0_median.getIntendedFrameInterval_ms(), ch0_median.doFrameIntervalSanityCheck())

        a_ch0 = analysis.FarenbackAnalyzer(ch0_median, "um/h")
        a_ch0.doFarenbackFlow()
        speeds_ch0 = analysis.FlowSpeedAnalysis(a_ch0)
        speeds_ch0.calculateSpeeds()
        speeds_ch0.calculateAverageSpeeds()
        speeds_ch0.calculateHistograms()
        #speeds_ch0.saveSpeedArray(savepath)
        speeds_ch0.saveSpeedCSV(savepath,fname=speeds_ch0.getChannelName()+"glide-"+str(glideFlag)+".csv")





tif.close()


