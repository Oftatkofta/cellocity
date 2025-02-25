# -*- coding: utf-8 -*-
"""
Created on Fri May 20 14:56:37 2022

@author: Jens
"""

import channel as ch
from tifffile import TiffFile
import tifffile

mm2_gamma = r"C:\Users\Jens\Desktop\hOGM_1_MMStack_Default.ome.tif\hOGM_1_MMStack_Default.ome.tif"
mm2_beta = r"C:\Users\Jens\Documents\_Microscopy\FrankenScope2\_Pilar\EXP-18-BP4241 Live imaging STm invasion monolayer different stages\Day5_WellD_afterinfection__1\Day5_WellD_afterinfection__1_MMStack_Pos0.ome.tif"
mm_14 = r"C:\Users\Jens\Documents\_Microscopy\FrankenScope2\Calibration stuff\180405 Hela 100X Zstack\100X_Ap2.5_HeLa_1X_disk_in_1\100X_Ap2.5_HeLa_1X_disk_in_1_MMStack.ome.tif"
mm2 = r"F:\anisa\220601 transformation screen\PEBr-1_4_1\PEBr-1_4_1_MMStack_B2-Site_0.ome.tif"
mm2_flex = r"F:\convallaria\flexoscope\live_5pc_glucose_PBS_1\live_5pc_glucose_PBS_1_MMStack_Default.ome.tif"
testfiles = [mm2_gamma, mm2, mm2_flex]

def read_metadata(filename):
    with TiffFile(filename) as tif:
        mm_meta = tif.micromanager_metadata
        ij_meta = tif.imagej_metadata
        #pages = tif.pages
        #mm_keys = ['Summary', 'MajorVersion', 'Comments'] # skip , 'IndexMap'
        mm_keys = list(mm_meta.keys())
        ij_keys = list(ij_meta.keys())
        print(filename, mm_keys, ij_keys)
        if len(mm_keys) > 0:
            for key in mm_keys:
                print(key+"::", mm_meta[key])
        if len(ij_keys) > 0:
            for key in ij_keys:
                print(key+":::", ij_meta[key])

def make_channel(filename, ch_idx, name=None):
    with TiffFile(filename) as tif:
        if name is None:
            label = tif.filename.split(".")[0]
            name = label + "_Ch" + str(ch_idx + 1)
            out = ch.Channel(ch_idx, tif, name=name)
            out.getArray()
            return out


if __name__ == "__main__":
    for file in testfiles:
        #read_metadata(file)
        channel = make_channel(file, 0)
        print(channel.name)
        print(channel.getArray().shape)