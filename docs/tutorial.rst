Cellocity Tutorial
==================

Step-by-step guide
------------------

This turorial will show how to:

1. Load a file and create a :class:'cellocity.channel.Channel' object. 
2. Preprocess Channel object
3. Perform analysis by creating an Analysis object from Channel object 
4. Extract data from the Analysis object by creating an Analyzer object.

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

A ``Tifffile`` does not load all its image data in to RAM when created, however
upon accessing data during ``Channel`` creation some of it will be cached, thus
increasing its size somewhat. ``Channel`` objects store all image data in RAM and
can get quite hefty for long timelapses.
    
    
Preprocess ``Channel`` object
+++++++++++++++++++++++++++++

First, we will check if the frame interval stated in the metadata is in agreement with
the time stamps of the individual frames in the channel (within 1%). This is done with the



Channel objects have convenient preprocessing methods, such as trimming frames
and temporal median filtering. Let's start by trimming our newly created channel to
frames 10-60, meaning we discard frames 0-9 and 60-end.

 .. code-block:: python
	
	#Trim channel to include frame 10-59
    channel_0.trim(10,60)





Now let's perform a temporal median filter, meaning we do a median filter over time.
This will have the effect of filtering out fast moving free-floating debrees, thus 
greatly reducing the noise in the final analysis. This is done by creating a child ``MedianChannel``
object. ``MedianChannel`` takes care of properly

.. code-block:: python

	



