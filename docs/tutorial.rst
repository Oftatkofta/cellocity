Cellocity Tutorial
==================

Step-by-step guide
------------------

Do like this:

1. load
2. Process
3. Analyse data

 .. code-block:: python
    :linenos:

    from cellocity.channel import Channel
    import tifffile
    import os
    
    my_filename = "2_channel_micromanager_data.ome.tif"
    chToAnalyze = 0 #0-based indexing of channels
    
    def main():
        with tifffile.TiffFile(my_filename, multifile=False) as tif:

            label = os.path.split(my_filename)[1][:-4]
            
            #report to user
            print("Working on: {} as {}".format(my_filename, label))
            ij_metadata = tif.imagej_metadata
            n_channels = int(ij_metadata.get('channels', 1))

            Ch0 = Channel(chToAnalyze, tif, name=label + "_Ch" + str(chToAnalyze + 1))

            finterval_s = round(Ch0.finterval_ms / 1000, 2)
            frames_per_min = round(60 / finterval_s, 2)
            tunit = 's'
            print(
                "Intended dimensions: frame interval {:.2f}s, {:.2f} frames/min, pixel size: {:.2f} um ".format(
                    finterval_s, frames_per_min, Ch0.pxSize_um))
                    
            actual_interval=Ch0.getActualFrameIntevals_ms().mean() / 1000)
            print("Actual frame interval is: {:.2f} s".format(actual_interval)

            if not Ch0.doFrameIntervalSanityCheck():
                print("Replacing intended interval with actual!")
                finterval_ms = Ch0.getActualFrameIntevals_ms().mean()
                finterval_s = round(finterval_ms / 1000, 2)
                frames_per_min = round(60 / finterval_s, 2)
                tunit = 's'
                Ch0.scaler = Ch0.pxSize_um * frames_per_min  # um/px * frames/min * px/frame = um/min
                Ch0.finterval_ms = finterval_ms

            print(
                "Using dimensions: frame interval {:.2f}s, {:.2f} frames/min, pixel size: {:.2f} um. Output unit: {}".format(
                    finterval_s, frames_per_min, Ch0.pxSize_um, unit))

            print("Start median filter of Channel 1...")
            Ch0.getTemporalMedianFilterArray()
            print("Elapsed for file {:.2f} s, now calculating Channel 1 flow...".format(time.time() - t1))
            Analysis_Ch0 = FarenbackAnalyzer(Ch0, unit)
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
                               'tunit': tunit, 'Info': ij_metadata.get('Info', "None"),
                               'frames': Analysis_Ch0.flows.shape[0],
                               'slices': 1, 'channels': n_channels}