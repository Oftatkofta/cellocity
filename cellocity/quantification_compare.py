import channel as ch
from tifffile import TiffFile
import xml
myfile = r"F:\bactunet_val\OGM\EXP-20-BT0350\hOGM_1_MMStack_Default.ome.tif\hOGM_1_MMStack_Default.ome.tif"


def make_channel(filename, ch_idx, name=None):
    with TiffFile(filename) as tif:
        if name is None:
            label = tif.filename.split(".")[0]
            name = label + "_Ch" + str(ch_idx + 1)
            out = ch.Channel(ch_idx, tif, name=name)
            out.getArray()
            return out


with TiffFile(myfile) as tif:
    print(tif)
    arr = tif.asarray()[:,1,:,:]
    print(arr.shape)

    #channel_1 = ch.Channel(0, tif, "DIC")  # 0-indexed channels, meaning ch1 in ImageJ
    #channel_2 = ch.Channel(1, tif, "mCherry")
    #print(channel_1.getArray().shape, channel_2.getArray().shape)



##3-frame gliding temporal median projection by default
# channel_2_median = MedianChannel(channel_1)
# print(channel_1.finterval_ms, channel_2.pxSize_um)
# print(channel_2.pages)