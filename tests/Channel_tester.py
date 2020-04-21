from cellocity.channel import Channel, MedianChannel
import cellocity.analysis as analysis
from tifffile import TiffFile
from pympler import asizeof
import os

onefour = r"C:\\Users\\Jens\\Microscopy\\FrankenScope2\\_Pilar\\fucci_GFP-geminin_RFP-cdt1_7-5min_1_MMStack_Pos0.ome.tif"
beta = r"C:\Users\Jens\Microscopy\FrankenScope2\_Pilar\raw_STm infection_10x__1_MMStack_MOI2.ome.tif"
ij = r"C:\\Users\\Jens\\Microscopy\\FrankenScope2\\_Pilar\\fucci_GFP-geminin_RFP-cdt1_7-5min_IJ.tif"

testfiles=[onefour, ij, beta]

#tif = TiffFile(beta)
#for key in tif.micromanager_metadata.keys():
#    print(key, tif.micromanager_metadata[key])

for fh in testfiles:
    print(os.path.getsize(fh), fh)
    tif = TiffFile(fh)
    ch0 = Channel(0, tif, "test_1")
    #ch1 = Channel(1, tif, "test_2")
    n = ch0.getArray()
    #n = ch1.getArray()
    ch0_median = MedianChannel(ch0,doGlidingProjection=True, frameSamplingInterval=4,startFrame=0, stopFrame=6)
    ch0_median = ch0.getTemporalMedianChannel(doGlidingProjection=True, frameSamplingInterval=4,startFrame=0, stopFrame=6)
#a_ch0 = analysis.FarenbackAnalyzer(ch0, "um/h")
#a_ch0.doFarenbackFlow()
#print(a_ch0.flows.shape)


tif.close()


