import time, os
import tifffile
from MM_flow_analyzer import *
from matplotlib import pyplot as plt
import numpy as np

indir = r"C:\Users\Jens\Desktop\temp"
outdir = r"C:\Users\Jens\Desktop\temp\out"




def analyzeFiles(fnamelist, outdir, flowkwargs, scalebarFlag, scalebarLength, chToAnalyze = 0):
    """
    Automatically analyzes tifffiles annd saves ouput in outfolder. If input has two channels, analysis is run on
    channel index 1 by default, but can be changed.
    :param tif: Tifffile objekt
    :return:
    """
    for fname in fnamelist:

        with tifffile.TiffFile(fname, multifile=False) as tif:

            if tif.is_micromanager:
                lab = os.path.split(fname)[1][:-8]
            else:
                lab = os.path.split(fname)[1][:-4]
            print("Working on: {} as {}".format(fname, lab))
            t1 = time.time()
            ij_metadata = tif.imagej_metadata

            if tif.is_micromanager:
                n_channels = int(tif.micromanager_metadata['Summary']['Channels'])
            else:
                n_channels = int(ij_metadata['channels'])

            Ch0 = Channel(chToAnalyze, tif, name=lab + "_Ch"+str(chToAnalyze+1))

            print("Elapsed for file load and Channel creation {:.2f} s.".format(time.time() - t1))
            finterval_s = round(Ch0.finterval_ms / 1000, 2)
            frames_per_min = round(60 / finterval_s, 2)
            tunit = 's'
            print(
                "Intended dimensions: frame interval {:.2f}s, {:.2f} frames/min, pixel size: {:.2f} um ".format(finterval_s, frames_per_min, Ch0.pxSize_um))

            print("Actual frame interval is: {:.2f} s".format(Ch0.getActualFrameIntevals_ms().mean()/1000))

            if not Ch0.doFrameIntervalSanityCheck():
                print("Replacing intended interval with actual!")
                finterval_ms = Ch0.getActualFrameIntevals_ms().mean()
                finterval_s = round(finterval_ms / 1000, 2)
                frames_per_min = round(60 / finterval_s, 2)
                tunit = 's'
                Ch0.scaler = Ch0.pxSize_um * frames_per_min  # um/px * frames/min * px/frame = um/min
                Ch0.finterval_ms = finterval_ms

            print(
                    "Using dimensions: frame interval {:.2f}s, {:.2f} frames/min, pixel size: {:.2f} um ".format(
                        finterval_s, frames_per_min, Ch0.pxSize_um))


            print("Start median filter of Channel 1...")
            Ch0.getTemporalMedianFilterArray()
            print("Elapsed for file {:.2f} s, now calculating Channel 1 flow...".format(time.time() - t1))
            Analysis_Ch0 = FarenbackAnalyzer(Ch0)
            Analysis_Ch0.doFarenbackFlow()

            print("flow finished, calculating speeds...")
            Analysis_Ch0.doFlowsToSpeed()
            print("Saving speeds...as {}_speeds.tif".format(Analysis_Ch0.channel.name))
            Analysis_Ch0.saveSpeedArray(outdir)
            Analysis_Ch0.saveSpeedCSV(outdir)
            print("Elapsed for file {:.2f} s, now drawing flow...".format(time.time() - t1))
            Analysis_Ch0.draw_all_flow_frames(scalebarFlag, scalebarLength, **flowkwargs)

            Analysis_Ch0.rehapeDrawnFramesTo6d()

            ij_metadatasave = {'unit': 'um', 'finterval': finterval_s,
                               'tunit': tunit, 'Info': ij_metadata.get('Info', "None"), 'frames': Analysis_Ch0.flows.shape[0],
                               'slices': 1, 'channels': n_channels}

            if n_channels == 2:
                print("Loading Channel 2")
                if chToAnalyze == 0:
                    Ch1 = Channel(1, tif, name=lab + "_Ch2")
                else:
                    Ch1 = Channel(0, tif, name=lab + "_Ch1")

                print("Start median filter of the other channel ...")
                Ch1.getTemporalMedianFilterArray()
                Ch1.medianArray = normalization_to_8bit(Ch1.medianArray, lowPcClip=10, highPcClip=0)
                Ch1.rehapeMedianFramesTo6d()

                savename = os.path.join(outdir, lab + "_2Chan_flow.tif")
                #print(Analysis_Ch0.drawnFrames.shape, Ch1.medianArray[:stopframe-3].shape)
                arr_to_save = np.concatenate((Analysis_Ch0.drawnFrames, Ch1.medianArray[:-1]), axis=2)


            else:
                savename = os.path.join(outdir, lab + "_flow.tif")

                arr_to_save = Analysis_Ch0.drawnFrames

            print("Saving flow...")
            tifffile.imwrite(savename, arr_to_save.astype(np.uint8),
                                 imagej=True, resolution=(1 / Ch0.pxSize_um, 1 / Ch0.pxSize_um),
                                 metadata=ij_metadatasave
                                 )
            print("File done!")

    return True
