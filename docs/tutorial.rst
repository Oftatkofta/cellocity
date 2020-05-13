Cellocity Tutorial
==================

Step-by-step guide
------------------

This turorial will show how to:

1. Load a file and create a :class:`cellocity.channel.Channel` object. 
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
    
	A ``Tifffile`` does not load all its image data in to RAM when created, however
	upon accessing data during ``Channel`` creation some of it will be cached, thus
	increasing its size somewhat. ``Channel`` objects store all image data in RAM and
	can get quite hefty for long timelapses.
    
    
Preprocess ``Channel`` object
+++++++++++++++++++++++++++++

First, we will check if the frame interval stated in the metadata is in agreement with
the time stamps of the individual frames in the channel (within 1%). This is done with the
``Channel.doFrameIntervalSanityCheck(maxDiff=0.01)`` method. If there is an discrepancy between
the actual frame intervals and the intended, if can be fixed by calling the 
``Channel.fixFrameInterval()`` method, which overwrites the intended frame interval with the actual
average frame interval.

  .. code-block:: python
  
	if not channel_0.doFrameIntervalSanityCheck():
		channel_0.fixFrameInterval()

  .. note::
	Checking and fixing the frame interval is currently only possible on MicroManager ome.tif files.

Channel objects have convenient preprocessing methods, such as trimming frames
and temporal median filtering. Let's start by trimming our newly created channel to
frames 10-60, meaning we discard frames 0-9 and from frame 60 onward to the end.

 .. code-block:: python
	
	#Trim channel to include frame 10-59
	channel_0.trim(10,60)


Now let's perform a temporal median filter, meaning we do a median filter over time.
This will have the effect of filtering out fast moving free-floating debrees, thus 
greatly reducing the noise in the final analysis. This is done by creating a child ``MedianChannel``
object. Median filitering can be done with a gliding window (default), or by binning the frames.
``MedianChannel`` takes care of properly recalculating frame intervals in either case. The default 
frame sampling interval is 3.

.. code-block:: python
	
	from cellocity.channel import MedianChannel

	gliding_median_channel_0 = MedianChannel(channel_0)
	binned_4frame_median_channel_0 = MedianChannel(channel_0, doGlidingProjection=False, frameSamplingInterval=4)
	

``MedianChannel`` objects can also be created by calling the ``.getTemporalMedianChannel()`` method on a ``Channel``.
The following code gives identical results to the above example:

.. code-block:: python
	
	arguments ={
			doGlidingProjection = True,
			frameSamplingInterval=3,
			startFrame=0,
			stopFrame=None
			}

	
	gliding_median_channel_0 = channel_0.getTemporalMedianChannel(arguments)
	
	arguments = {doGlidingProjection = False,
				frameSamplingInterval=4,
				startFrame=0,
				stopFrame=None}

	binned_4frame_median_channel_0 = channel_0.getTemporalMedianChannel(arguments)

Analysis of ``Channel`` object
++++++++++++++++++++++++++++++

Now let's do an optical flow analysis of our prepocessed ``Channel``. This is done
by instatiating an ``Analyzer`` object with a ``Channel`` as argument. In this case we
will perform an optical flow analysis using the Farenback flow analysis from OpenCV. This
is handled by a ``FarenbackAnalyzer``, which is a specific subtype ``FlowAnalyzer`` of ``Analyzer``.

``FarenbackAnalyzer`` takes two arguments, one ``Channel`` and one **unit**. **unit** is a string
indicating the unit that we want the output to be in. Currently only "um/s", "um/min", and "um/h" are
implemented. Cellocity handles all unit conversions automatically in the background.


.. code-block:: python

	from cellocity.analysis import FarenbackAnalyzer
	
	fb_analyzer_ch0 = FarenbackAnalyzer(channel = gliding_median_channel_0, unit = "um/h")
	fb_analyzer_ch0.doFarenbackFlow()
	
Great, now we have calculated the optical flow of channel_0 with the default parameters. Now its
time to extract data. This is done by creating ``Analysis`` objects. In our case we want to analyse
the flow speeds of our channel. To do this we can utilise the ``FlowSpeedAnalysis`` class, which works on
``FlowAnalyzer`` objects.

.. code-block:: python
	
	from cellocity.analysis import FlowSpeedAnalysis
	
	speed_analysis_ch0 = FlowSpeedAnalysis(fb_analyzer_ch0)
	speed_analysis_ch0.calculateSpeeds()
	speed_analysis_ch0.calculateAverageSpeeds()
	
When speeds have been calculated the results can be stored either as a 32-bit tif, where pixel values represent
flow speeds in the location of the pixel, or the average speed of each frame can be saved as a .csv file for further
processing.

.. code-block:: python

	from pathlib import Path
	
	savepath = Path("path/to/save/folder")
	
	speed_analysis_ch0.saveSpeedArray(outdir=savepath):
	speed_analysis_ch0.saveSpeedCSV(outdir=savepath, fname="mySpeeds.csv", tunit="s")
	
That's it! If you want more detailed information, please check the :doc:`api`
	
	



