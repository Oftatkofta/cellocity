from cellocity.channel import Channel
import cellocity.analysis as analysis
from tifffile import TiffFile
from pympler import asizeof
import os

onefour = r"C:\\Users\\Jens\\Microscopy\\FrankenScope2\\_Pilar\\fucci_GFP-geminin_RFP-cdt1_7-5min_1_MMStack_Pos0.ome.tif"
ij = r"C:\\Users\\Jens\\Microscopy\\FrankenScope2\\_Pilar\\fucci_GFP-geminin_RFP-cdt1_7-5min_IJ.tif"

mm_filesize_bytes = os.path.getsize(onefour)
ij_filesize_bytes = os.path.getsize(ij)

print(mm_filesize_bytes, ij_filesize_bytes)

tif = TiffFile(onefour)

ch0 = Channel(0, tif, "test_1")
ch1 = Channel(1, tif, "test_2")
ch2 = Channel(2, tif, "test_3")

n = ch0.getArray()
n = ch1.getArray()
n = ch2.getArray()

a_ch0 = analysis.FarenbackAnalyzer(ch0, "um/h")
a_ch0.doFarenbackFlow()
print(a_ch0.flows.shape)


tif.close()


