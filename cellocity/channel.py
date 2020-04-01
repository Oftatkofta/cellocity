import numpy as np
import tifffile.tifffile as tifffile
import re
import cv2 as cv
import time
import os
import pandas as pd
import warnings


class Channel(object):
    """
    Base Class to keep track of one channel (x,y,t) of microscopy data.

    Channel Objects are created from tifffile.Tifffile and act as shallow copies of the
    TiffPage objects making up the channel,
    until a Nupy array is generated by 'getArray'. Then self.array is populated by a Numpy array from the
    raw image data, using the 'asarray' function in 'tiffile.Pages'. Only a single z-slice
    and channel are handled per Channel object. A reference to the base 'tifffile.Tifffile' is stored in
    self.tif.

    There are currently two very similar subclasses of Channel, MM_Channel, and IJ_Channel to handle
    Micromanager OME-TIFFs and ImageJ hyperstacks, respectively.


    """
    def __init__(self, chIndex, tiffFile, name, sliceIndex=0):
        """
        :param chIndex: index of channel to create, 0-based.
        :type chIndex: int
        :param tiffFile: TiffFile object to extract channel from
        :type tiffFile: :class:'tifffile'
        :param name: name of channel, used in Analysis output
        :type name: str
        :param sliceIndex: z-slice to extract, defaults to 0
        :type sliceIndex: int

        """
        self.chIndex = chIndex
        self.sliceIdx = sliceIndex
        self.tif = tiffFile
        self.name = name
        self.tif_ij_metadata = tifffile.imagej_metadata
        self.pxSize_um, self.finterval_ms = self._read_px_size_and_finteval() #frame interval from settings, not actual
        self.elapsedTimes_ms = [] #_page_extractor method populates this
        self.pages = self._page_extractor()
        self.array = np.empty((0)) # getArray populates this when called
        self.actualFrameIntervals_ms = None #getActualFrameIntervals_ms populates this when called
        self.medianArray = np.empty((0)) # getTemporalMedianFilterArray populates when called
        self.frameSamplingInterval = None # getTemporalMedianFilterArray populates when called


    def _page_extractor(self):
        """
        Tifffile objects reads the actual TIF image data together with the TIF-tags from disk and encapsulates them in
        TiffPage objecs. The TiffPages that make up the channel data are read in this method.

        Tifffile has an option to read TiffFrame objects instead of TiffPage objects. TiffFrames are light weight
        versions of TiffPages, but they do not contain any TIF-tags. This method ensures that TiffPages are returned by
        setting Tifffile.pages.useframes to _False_.

        :return: (list) TiffPage objects corresponding to the chosen slice and channel

        """

        self.tif.pages.useframes = False  # TiffFrames can't be used for extracting metadata
        out = []

        if self.tif.is_micromanager:

            sliceMap = self.tif.micromanager_metadata["IndexMap"]["Slice"]
            channelMap = self.tif.micromanager_metadata["IndexMap"]["Channel"]

        elif self.tif.is_imagej:
            elapsed = 0
            indexMap = self._ij_pagemapper()
            sliceMap = indexMap[1]
            channelMap = indexMap[0]

        for i in range(len(self.tif.pages)):

            if (sliceMap[i] == self.sliceIdx) and (channelMap[i] == self.chIndex):
                page = self.tif.pages[i]
                out.append(page)
                if self.tif.is_micromanager:
                    self.elapsedTimes_ms.append(page.tags["MicroManagerMetadata"].value["ElapsedTime-ms"])
                else:
                    self.elapsedTimes_ms.append(elapsed)
                    elapsed += self.finterval_ms
        return out

    def _ij_pagemapper(self):

        """
        Helper function to make maps for sorting IJ Pages in to slices and channels.

        :returns: (channelMap, sliceMap, frameMap) which are lists of integers describing which channel,
        slice, frame each Tiff-Page belongs to. Indexes start at 0.

        """
        nChannels = self.tif.imagej_metadata.get('channels', 1)
        nSlices = self.tif.imagej_metadata.get('slices', 1)
        nFrames = self.tif.imagej_metadata.get('frames', None)

        channelMap = []
        sliceMap = []
        frameMap = []

        for i in range(len(self.tif.pages)):
            chIdx = (i % nChannels)
            slIdx = (i // nChannels) % nSlices
            frIdx = ( i // (nChannels*nSlices))


            channelMap.append(int(chIdx))
            sliceMap.append(int(slIdx))
            frameMap.append(int(frIdx))

        return channelMap, sliceMap, frameMap

    def _read_px_size_and_finteval(self):
        """
        Determines which version of MM that was used to acquire the data, or if it is an ImageJ file.
        MM versions 1.4 and 2.0-gamma, share Metadata structure, but 2.0.0-beta is slightly different
        in where the frame interval and pixel sizes can be read from. In 2.0-beta the
        frame interval is read from tif.micromanager_metadata['Summary']['WaitInterval'],
        and in 1.4/2.0-gamma it is read from tif.micromanager_metadata['Summary']['Interval_ms']

        Pixel size is read from tif.micromanager_metadata['Summary']['PixelSize_um'] in 1.4/2.0-gamma, but from
        tif.micromanager_metadata['PixelSize_um'] in 2.0-beta

        MM versions used for testing>
        MicroManagerVersion 1.4.23 20180220
        MicroManagerVersion 2.0.0-gamma1 20190527
        MicroManagerVersion 2.0.0-beta3 20180923

        :param mm_metadata: (dict) MicroManager metadata dictionary

        :return: (tuple) (pixel_size_um, frame_interval)
        """
        if self.tif.is_micromanager:
            # if the file is a MM file this branch determines which version
            one4_regex = re.compile("1\.4\.[\d]")  # matches 1.4.d
            gamma_regex = re.compile("gamma")
            beta_regex = re.compile("beta")

            version = self.tif.micromanager_metadata["Summary"]["MicroManagerVersion"]
            px_size_um = self.tif.micromanager_metadata['PixelSizeUm']

            if (re.search(beta_regex, version) != None):
                finterval_ms = self.tif.micromanager_metadata['Summary']['WaitInterval']

                return px_size_um, finterval_ms

            elif (re.search(one4_regex, version) != None):
                finterval_ms = self.tif.micromanager_metadata['Summary']['Interval_ms']

                return px_size_um, finterval_ms

            elif (re.search(gamma_regex, version) != None):
                finterval_ms = self.tif.micromanager_metadata['Summary']['Interval_ms']

                return px_size_um, finterval_ms

        elif self.tif.is_imagej:
            #this is not as clean due to the undocumated nature of imageJ metadata

            #time
            finterval = self.tif.imagej_metadata.get('finterval', 1)
            tunit = self.tif.imagej_metadata.get('tunit', 's')

            if (tunit == 'min') or (tunit == 'm'):
                finterval_ms = 60 * 1000 * finterval

            elif (tunit == 'hour') or (tunit == 'h') or (tunit == 'hours'):
                finterval_ms = 60 * 60 * 1000 * finterval

            elif (tunit == 'sec') or (tunit == 's') or (tunit == 'seconds'):
                finterval_ms = 1000 * finterval

            #space
            divisor, dividend = self.tif.pages[0].tags['XResolution'].value #TODO check for 0
            px_size = float(dividend/divisor)
            sz_unit = self.tif.imagej_metadata.get('unit', '\\u00B5m')

            if (sz_unit == 'cm'):
                px_size_um = px_size * 10 * 1000 #10 mm/cm * 1000 um/mm = 10000 um

            elif (sz_unit == 'mm'):
                px_size_um = px_size * 1000  # 1000 um/mm

            elif (sz_unit == '\\u00B5m') or (sz_unit == 'um') or (sz_unit == 'micrometer') or (sz_unit == 'micron'):
                px_size_um = px_size

            return px_size_um, finterval_ms

        else:
            return 1, 1 #defaults to pixels/frame

    def getPages(self):

        return self.pages

    def getElapsedTimes_ms(self):

        return self.elapsedTimes_ms

    def getArray(self):

        if len(self.array) != 0:

            return self.array

        else:
            outshape = (len(self.pages),
                        self.pages[0].shape[0],
                        self.pages[0].shape[1])

            outType = self.pages[0].asarray().dtype

        out = np.empty(outshape, outType)

        for i in range(len(self.pages)):
            out[i] = self.pages[i].asarray()

        return out

    def getTemporalMedianFilterArray(self, startFrame=0, stopFrame=None,
                               frameSamplingInterval=3, recalculate=False):
        """
        The function first runs a gliding N-frame temporal median on every pixel to
        smooth out noise and to remove fast moving debris that is not migrating
        cells. Recalculates the median array if recalculate is True.

        :param arr: (3d numpy array) with a shape of (t, y, x)
        :param stopFrame: (int) Last frame to analyze, defaults to analyzing all frames if None
        :param startFrame: (int) First frame to analyze
        :param frameSamplingInterval: (int) do median projection every N frames
        :param recalculate: (bool) Should the median projection be recalculated?

        :return: An Nupy array of the type float32

        """

        if len(self.medianArray) != 0:
            if not recalculate:

                return self.medianArray

        if (stopFrame == None) or (stopFrame > len(self.pages)):
            stopFrame = len(self.pages)

        if (startFrame >= stopFrame):
            raise ValueError("StartFrame cannot be larger than Stopframe!")

        if (stopFrame-startFrame < frameSamplingInterval):
            raise ValueError("Not enough frames selected to do median projection! ")


        self.frameSamplingInterval = frameSamplingInterval
        arr = self.getArray()
        # nr_out_frames = n_in-(samplingInterval-1)
        nr_outframes = (stopFrame - startFrame) - (frameSamplingInterval - 1)

        outshape = (nr_outframes, arr.shape[1], arr.shape[2])

        self.medianArray = np.ndarray(outshape, dtype=np.float32)

        outframe = 0

        for inframe in range(startFrame, stopFrame-frameSamplingInterval+1):

            # median of frames n1,n2,n3...
            frame_to_store = np.median(arr[inframe:inframe + frameSamplingInterval], axis=0).astype(np.float32)

            self.medianArray[outframe] = frame_to_store
            outframe += 1

        return self.medianArray

    def getTiffFile(self):
        '''

        Returns the 'Tifffile' that the 'Channel' is based on.

        :return: Tifffile-object used in Channel creation
        :rtype: object tifffile.Tifffile
        '''
        return self.tif

    def getActualFrameIntevals_ms(self):
        # the intervals between frames in ms as a 1D numpy array
        # returns None if only one frame exists or if the file is not from MicroManager

        if (self.actualFrameIntervals_ms != None):

            return self.actualFrameIntervals_ms

        elif len(self.pages) == 1:

            return None

        else:
            out = []
            t0 = self.elapsedTimes_ms[0]
            for t in self.elapsedTimes_ms[1:]:
                out.append(t-t0)
                t0 = t
            return np.asarray(out)

    def getIntendedFrameInterval_ms(self):

        return self.finterval_ms

    def doFrameIntervalSanityCheck(self, maxDiff=0.01):
        #Checks if the intended frame interval matches the actual within maxDiff defaults to allowing 1% difference


        if len(self.pages) == 1:
            return None

        elif (self.getIntendedFrameInterval_ms() == 0):

            return False
        else:
            fract = self.getActualFrameIntevals_ms().mean()/self.getIntendedFrameInterval_ms()
            out = abs(1-fract) < maxDiff

            return out

    def rehapeMedianFramesTo6d(self):
        #reshapes 3D (t, x, y) array to (t, 1, 1, x, y, 1) for saving dimensions in TZCYXS order
        shape = self.medianArray.shape
        self.medianArray.shape = (shape[0], 1, 1, shape[1], shape[2], 1)


def normalization_to_8bit(image_stack, lowPcClip = 0.175, highPcClip = 0.175):
    """
    Function to rescale 16/32/64 bit arrays to 8-bit for visualizing output

    Defaults to saturate 0.35% of pixels, 0.175% in each end by default, which often produces nice results. This
    is the same as pressing 'Auto' in the ImageJ contrast manager. numpy.interp() linear interpolation is used
    for the mapping.

    :param image_stack: Numpy array to be rescaled
    :type image_stack: Numpy array
    :param lowPcClip: Fraction for black clipping bound
    :type lowPcClip: float
    :param highPcClip: Fraction for white/saturated clipping bound
    :type highPcClip: float
    :return: 8-bit numpy array of the same shape as :param image_stack:
    :rtype: numpy.dtype('uint8')
    """


    #clip image to saturate 0.35% of pixels 0.175% in each end by default.
    low = int(np.percentile(image_stack, lowPcClip))
    high = int(np.percentile(image_stack, 100 - highPcClip))

    # use linear interpolation to find new pixel values
    image_equalized = np.interp(image_stack.flatten(), (low, high), (0, 255))

    return image_equalized.reshape(image_stack.shape).astype('uint8')


def read_micromanager(tif):
    """
    returns metadata from a micromanager file
    """
    pass




