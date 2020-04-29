from cellocity.channel import Channel, MedianChannel
import cellocity.analysis as analysis
from tifffile import TiffFile
from pympler import asizeof
import os

onefour = r"C:\\Users\\Jens\\Microscopy\\FrankenScope2\\_Pilar\\fucci_GFP-geminin_RFP-cdt1_7-5min_1_MMStack_Pos0.ome.tif"
beta = r"C:\Users\Jens\Microscopy\FrankenScope2\_Pilar\raw_STm infection_10x__1_MMStack_MOI2.ome.tif"
ij = r"C:\\Users\\Jens\\Microscopy\\FrankenScope2\\_Pilar\\fucci_GFP-geminin_RFP-cdt1_7-5min_IJ.tif"

testfiles=[onefour, ij, beta]

tif = TiffFile(beta)
for key in tif.micromanager_metadata.keys():
    print(key, tif.micromanager_metadata[key])

for fh in testfiles:
    print(os.path.getsize(fh), fh)
    tif = TiffFile(fh)
    ch0 = Channel(0, tif, "test_1")
    ch0.trim(4,14)

    n = ch0.getArray()

    for glideFlag in [False, True]:
        print("Glideflag:", glideFlag)
        ch0_median = MedianChannel(ch0,doGlidingProjection=glideFlag)
        print(ch0_median.getArray().shape)
        a_ch0 = analysis.FarenbackAnalyzer(ch0_median, "um/h")
        a_ch0.doFarenbackFlow()
        print(a_ch0.flows.shape)


tif.close()


