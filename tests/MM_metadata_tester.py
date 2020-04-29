from cellocity.channel import Channel, MedianChannel
import cellocity.analysis as analysis
from tifffile import TiffFile
from pympler import asizeof
import os

onefour_split = r"C:\\Users\\Jens\\Downloads\\_1_MMStack_laminin_TGFbeta2.ome.tif"


tif = TiffFile(onefour_split)
for key in tif.micromanager_metadata.keys():
    if key == "Summary":
        for key2 in tif.micromanager_metadata[key].keys():
            print(key, key2, tif.micromanager_metadata[key][key2])

    print(key, tif.micromanager_metadata[key])