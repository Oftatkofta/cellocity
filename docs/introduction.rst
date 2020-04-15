Introduction to Cellocity
=========================

A 30 second pitch
-----------------

Cellocity is an bioimage analysis tool for quantifying confluent cell dynamics. The main advantages of Cellocity is its ability work on unlabeled Brightfield time lapse microscopy data, and its ability to both quantify and visualize abstract optical flow analysis to the user.

.. figure:: _static/convergence.png
    :align: left
    :alt: Example output
    
    Figure showing one frame of raw data (left), a vector field visualization (center), and a heat map encoding speeds (right).


Cellocity development history
-----------------------------

Cellocity has been developed over multiple years and several projects. The nucleus was developed at `Stig Ove Bøe's <https://ous-research.no/home/boe/Group+members/10831>`_ research group at Oslo University Hospital and at the `Nanoscopy Gaustad <https://www.med.uio.no/english/research/core-facilities/advanced-light-microscopy-gaustad/>`_ imaging core facility at the University of Oslo. Many of Cellocity's core algorithm implementations and methods, such as the 5-sigma correlation length analysis, were presented in a `Nature Communications <https://www.nature.com/articles/s41467-018-05578-7>`_ publication in 2018.


Cellocity architecture
----------------------

Cellocity is built on top of  Christoph Gohlke's `Tifffile library <https://pypi.org/project/tifffile/>`_ and uses the ``Tifffile`` object to read input and to write output files. Cellocity also relies heavily on `OpenCV <https://opencv.org/>`_ and `OpenPIV <http://www.openpiv.net/>`_ for optical flow analysis and output visualizations. `NumPy <https://numpy.org/>`_ is used internally for image data manipulation in the form of ``numpy.ndarrays``.

The core element in Cellocity is the ``Channel`` object, which represents one Z-plane of one time lapse image channel. ``Channel`` objects also handle image pre-processing, such as temporal or spatial median filtering. ``Channel`` objects are then given as input to ``Analysis`` objects, which perform specific analysis on the data. ``Analysis`` objects can then, in turn, be given to ``Analyzer`` objects, which take care of performing further analysis, such as calculating the alignment index, instantaneous order parameter (:math:`{\psi}`), and correlation length.

Examples of algorithms implemented
----------------------------------
Instantaneous Order Parameter (:math:`{\psi}`)
   :math:`{\psi}` = 1 corresponds to a perfectly uniform velocity field, where all the cells move in the same direction and with the same speed, while :math:`{\psi}` = 0 is expected for a randomly oriented velocity field. See [Malinverno `et. al` (2017)] for details.

Alignment Index



Examples
--------

Simple file loading example::

    from cellocity.channel import Channel
    from tiffile import Tiffile

    tif = Tifffile(myFile)
    channel_1 = Channel(0, tif, "channel name") #0-indexed channels

Simple pre-processing example::
    
    channel_1.doTemporalMedianFilter()
    
Simple optical flow calculation example::
    
    from cellocity.analysis import FarenbackAnalyzer
    
    analysis_Ch1 = FarenbackAnalyzer(channel_1, "um/min")
    analysis_Ch1.doFarenbackFlow()

References
----------

.. [Malinverno `et. al` (2017)] Malinverno, C., Corallino, S., Giavazzi, F. et al. Endocytic reawakening of motility in jammed epithelia. Nature   Mater 16, 587–596 (2017). https://doi.org/10.1038/nmat4848
[#] Natuer et al