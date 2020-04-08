Cellocity Tutorial
==================

Step-by-step guide
------------------

This turorial will show how to:

1. Load a file and create a Channel object. 
2. Process Channel object
3. Perform analysis by creating an Analysis object from Channel object 

Load a file and create a Channel object
+++++++++++++++++++++++++++++++++++++++
 .. code-block:: python
    
    from cellocity.channel import Channel
    import tifffile

    my_filename = "2_channel_micromanager_timelapse.ome.tif"
    chToAnalyze = 0  # 0-based indexing of channels

    #safely load file
    with tifffile.TiffFile(my_filename, multifile=False) as tif:

        #strips ome.tif from filename
        label = my_filename.split(".")[0]
        channelName = label + "_Ch" + str(chToAnalyze + 1)
        channel_0 = Channel(chToAnalyze, tif, name=channelName)
        
 .. warning::
 
    Cellocity assumes that it can hold all Channel data in RAM.
    
TiffFile & Channel objects and RAM
++++++++++++++++++++++++++++++++++

A Tifffile does not load all its image data in to RAM when created, however
upon accessing data during Channel creation some of it will be cached, thus
increasing its size somewhat. Channel objects store all image data in RAM and
can get quite hefty for long timelapses.
    
    
Process Channel object
++++++++++++++++++++++

 .. code-block:: python
 
    channel_0.do_stuff()



