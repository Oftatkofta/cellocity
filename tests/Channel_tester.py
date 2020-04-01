from cellocity.channel import Channel
from tifffile import TiffFile
from pympler import asizeof
import os

onefour = r"C:\\Users\\Jens\\Microscopy\\FrankenScope2\\_Pilar\\fucci_GFP-geminin_RFP-cdt1_7-5min_1_MMStack_Pos0.ome.tif"
ij = r"C:\\Users\\Jens\\Microscopy\\FrankenScope2\\_Pilar\\fucci_GFP-geminin_RFP-cdt1_7-5min_IJ.tif"

mm_filesize_bytes = os.path.getsize(onefour)
ij_filesize_bytes = os.path.getsize(ij)

print(mm_filesize_bytes, ij_filesize_bytes)

tif = TiffFile(onefour)
tif_ij = TiffFile(ij)

print(tif.pages[0].axes)
print(tif.pages[0].tags, "\n")
print(tif_ij.pages[0].tags)




print(tif.pages.useframes)

print(asizeof.asizeof(tif))

ch0 = Channel(0, tif, "test_1")
ch1 = Channel(1, tif, "test_2")
ch2 = Channel(2, tif, "test_3")
print(asizeof.asizeof(ch0), asizeof.asizeof(ch1), asizeof.asizeof(ch2),
      asizeof.asizeof(ch0) + asizeof.asizeof(ch1) + asizeof.asizeof(ch2))

n = ch0.getArray()
n = ch1.getArray()
n = ch2.getArray()

print(asizeof.asizeof(ch0), asizeof.asizeof(ch1), asizeof.asizeof(ch2),
      asizeof.asizeof(ch0) + asizeof.asizeof(ch1) + asizeof.asizeof(ch2))


print(asizeof.asizeof(tif))

#ch1_ij =Channel(1, tif_ij, "testIJ_2")
#print(asizeof.asizeof(ch0), asizeof.asizeof(ch0_ij), asizeof.asizeof(ch1), asizeof.asizeof(ch1_ij))
#print(asizeof.asizeof(tif), asizeof.asizeof(tif_ij))

tif_ij.close()
tif.close()


