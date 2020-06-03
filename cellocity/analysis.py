import numpy as np
import cv2 as cv
import os, time
import pandas as pd
import tifffile
import warnings
import cellocity.channel as channel
from matplotlib import pyplot as plt

class Analyzer(object):
    """
    Base object for all Analysis object types, handles progress updates.

    """

    def __init__(self, channel):
        """
        :param channel: A Channel object
        :type channel: class:`channel.Channel`

        """

        self.channel = channel
        self.progress = 0  # 0-100 for pyQt5 progressbar
        self.process_time = 0 #time taken to process

    def getProgress(self):
        """
        Returnas current progress in the interval 0-100.

        :return: Percentage progress of analysis
        :rtype: float

        """
        return self.progress

    def updateProgress(self, increment):
        """
        Updates self.progress by increment

        :param increment:
        :return:

        """

        self.progress += increment
        print("Progress: {:.1f} % on {}".format(self.progress, self.channel.name))

    def resetProgress(self):
        """
        Resets progressbar to 0

        :return:

        """
        self.progress = 0


class FlowAnalyzer(Analyzer):
    """
    Base object for all optical flow analysis object types.

    Stores UV vector components in self.flows as a (t, x, y, uv) numpy array.
    Also calculates and stores a scaling factor that converts flow from pixels per frame to distance/time.


    """

    def __init__(self, channel, unit):
        """
        :param unit: must be one of ["um/s", "um/min", "um/h"]
        :type unit: str
        """

        super().__init__(channel)

        self.allowed_units = ["um/s", "um/min", "um/h"]
        assert unit in self.allowed_units, "unit has to be one of "+ str(self.allowed_units)
        self.unit = unit
        self.scaler = self._getScaler()  # value to multiply vector lengths by to get selected unit from px/frame
        self.flows = None  # (t, x, y, uv) numpy array
        self.drawnFrames = None  # for output visualization

    def _getScaler(self):
        """
        Calculates a scalar value by which to scale from px/frame to um/min, um/h or um/s
        in the unit um*frame/px*(min/h/s)

        example:
        um/px * frames/min * px/frame = um/min

        :return: scaler
        :rtype: float

        """

        finterval_s = self.channel.finterval_ms / 1000

        if self.unit == "um/min":
            frames_per_min = round(60 / finterval_s, 2)
            return self.channel.pxSize_um * frames_per_min

        if self.unit == "um/h":
            frames_per_h = round(60 * 60 / finterval_s, 2)
            return self.channel.pxSize_um * frames_per_h

        if self.unit == "um/s":
            return self.channel.pxSize_um * finterval_s

    def get_u_array(self, frame):
        """
        Returns the u-component array of self.flows at frame

        :param frame: frame to extract u-component matrix from
        :type frame: int
        :return: u-component of velocity vectors as a 2D NumPy array
        :rtype: numpy.ndarray
        """

        return self.flows[frame, :, :, 0]

    def get_v_array(self, frame):
        """
        Returns the v-component array of self.flows

        :param frame: frame to extract v-component matrix from
        :type frame: int
        :return: v-component of velocity vectors as a 2D NumPy array
        :rtype: numpy.ndarray
        """

        return self.flows[frame, :, :, 1]

    def _getFlows(self):
        if self.flows is None:
            warnings.warn("No flow has been calculated!")
        return self.flows

    def alignment_index(self, u, v):
        """
        Returns an array of the same shape as u and v with the alignment index (ai).

        Alignment index is defined as in Malinverno et. al 2017, for every frame the ai is the average of the dot
        products of the mean velocity vector with each individual vector, all divided by the product of their
        magnitudes.

        :param u: 2D numpy array with u component of velocity vectors
        :param v: 2D numpy array with v component of velocity vectors
        :return: nunpy array with size=input.size where every entry is the alignment index in that pixel

        """

        assert (u.shape == v.shape) and (len(u.shape) == 2), "Only single frames are processed"

        vector_0 = np.array((np.mean(u), np.mean(v)))
        v0_magnitude = np.linalg.norm(vector_0)

        vector_magnitudes = np.sqrt((np.square(u) + np.square(v)))  # a^2 + b^2 = c^2
        magnitude_products = vector_magnitudes * v0_magnitude
        dot_products = u * vector_0[0] + v * vector_0[1]  # Scalar multiplication followed by array addition

        ai = np.divide(dot_products, magnitude_products)

        return ai

    def rms(self, frame): #Root Mean Square Velocity
        """
        Calculates the root mean square velocity of the input frame number from optical flow data in self.flows.

        rms is the speed, or vector magnitudes, in the unit pixels/frame. This is equivalent to taking the
        square root of the mean square velocity. rms is used in the calculation of IOP.

        :param frame: the number of the frame to be analyzed
        :type frame: int
        :return: the root mean square velocity of the velocity vectors in the frame
        :rtype: float
        """
        u=self.get_u_array()
        rms = np.sqrt(np.mean(np.square(u)+np.square(v))) #sqrt(u^2+v^2)

        return rms

    def smvvm(self, u, v):  # square_mean_vectorial_velocity_magnitude
        """
        Array addition of the squared average vector components, used in calculating the instantaneous order parameter

        :param u:
            2D numpy array with the u component of velocity vectors
        :param v:
            2D numpy array with the u component of velocity vectors
        :return:
            2D numpy array with the u component of velocity vectors

        """

        return np.square(np.mean(u)) + np.square(np.mean(v))

    def instantaneous_order_parameter(self, u, v):
        """
        Calculates the instantaneous order parameter (iop) in one PIV frame see  Malinverno et. al 2017 for a more detailed
        explanation. The iop is a measure of how similar the vectors in a field are, which takes in to account both the
        direcions and magnitudes of the vectors. iop always between 0 and 1, with iop = 1 being a perfectly uniform field
        of identical vectors, and iop = 0 for a perfectly random field.

        :param u:
            2D numpy array with the u component of velocity vectors
        :param v:
            2D numpy array with the u component of velocity vectors
        :return:
            (float) iop of vector field
        """
        return smvvm(u, v) / msv(u, v) #square_mean_vectorial_velocity_magnitude/Mean Square Velocity


class FarenbackAnalyzer(FlowAnalyzer):
    """
    Performs OpenCV's Farenbäck optical flow anaysis.

    """
    def __init__(self, channel, unit):
        """
        :param channel: Channel object
        :param unit: (str) "um/s", "um/min", or "um/h"

        """
        super().__init__(channel, unit)

    def doFarenbackFlow(self, pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0):
        """
        Calculates Farenback flow for a single channel time lapse

        returns numpy array of dtype int32 with flow in the unit px/frame
        Output values need to be multiplied by a scalar to be converted to speeds.

        """
        t0 = time.time()
        arr = self.channel.getArray()

        # Create empty array for speed
        self.flows = np.empty((arr.shape[0] - 1, arr.shape[1], arr.shape[2], 2), dtype=np.float32)

        #Setup progress reporting
        self.resetProgress()

        assert self.flows.shape[0] >= 1, "0 flow frames!"
        progress_increment = 100 / self.flows.shape[0]

        for i in range(self.flows.shape[0]):
            flow = cv.calcOpticalFlowFarneback(arr[i],
                                               arr[i + 1],
                                               None,
                                               pyr_scale,
                                               levels,
                                               winsize,
                                               iterations,
                                               poly_n,
                                               poly_sigma,
                                               flags)

            self.flows[i] = flow.astype(np.float32)
            self.updateProgress(progress_increment)


        self.process_time = time.time() - t0

        return self.flows




class OpenPivAnalyzer(FlowAnalyzer):
    """
    Implements OpenPIV's optical flow anaysis.

    """

    def __init__(self, channel, unit):
        """
        :param channel: Channel object
        :param unit: (str) "um/s", "um/min", or "um/h"

        """

        super().__init__(channel, unit)
        self.flow_coordinates = None
        self.default_piv_params =  dict(window_size=64,
                                        overlap=32,
                                        dt=1,
                                        search_area_size=70,
                                        sig2noise_method="peak2peak")


    def doOpenPIV(self, **piv_params):
        """
        The function does PIV analysis between every frame in input ``Channel``.

        It populates self.flows with the u and v components of the velocity vectors as two (smaller)
        numpy arrays. An additional array, self.flow_coorinates, with the x and y coordinates
        corresponding to the centers of the search windows in the original input
        array is also also populated.

        :param piv_params: parameters for the openPIV function extended_search_area_piv
        :type piv_params: dict
        :return: (u_component_array, v_component_array, original_x_coord_array, original_y_coord_array)
        :rtype: tuple

        """
        from openpiv import process

        t0 = time.time()

        if piv_params.get("window_size", None) is None:

            piv_params = self.default_piv_params


        arr = self.channel.getArray()
        n_frames = arr.shape[0] - 1

        #Setup progress reporting
        self.resetProgress()

        assert n_frames >= 1, "0 flow frames!"
        progress_increment = 100 / n_frames

        # original x/y coordinates
        x, y = process.get_coordinates(image_size=arr[0].shape,
                                            window_size=piv_params["window_size"],
                                            overlap=piv_params["overlap"],
                                       )

        # Zero-filled output arrays are created beforehand for maximal performance
        out_u = np.zeros((n_frames, x.shape[0], x.shape[1]))
        out_v = np.zeros_like(out_u)

        for i in range(n_frames):

            #openPIV works on 32bit images
            frame_a = arr[i].astype(np.int32)
            frame_b = arr[i + 1].astype(np.int32)

            out_u[i], out_v[i], s2n = process.extended_search_area_piv(frame_a, frame_b,
                                                                  window_size=piv_params["window_size"],
                                                                  overlap=piv_params["overlap"],
                                                                  dt=piv_params["dt"],
                                                                  search_area_size=piv_params["search_area_size"],
                                                                  sig2noise_method=piv_params["sig2noise_method"] )

            self.updateProgress(progress_increment)

        #all calculated arrays have the same shape
        shape = out_u.shape

        out_u = out_u.reshape((shape[0], shape[1], shape[2], 1))
        out_v = out_v.reshape((shape[0], shape[1], shape[2], 1))
        x = x.reshape((shape[1], shape[2], 1))
        y = y.reshape((shape[1], shape[2], 1))

        self.flows = np.concatenate([out_u, out_v], axis=3).astype(np.float32)
        self.flow_coordinates = np.concatenate([x, y], axis=2).astype(np.int16)
        self.process_time = time.time() - t0

        return self.flows, self.flow_coordinates


class Analysis(object):
    """
    Base object for analysis of Analysis classes

    """

    def __init__(self, analyzer):
        """
        :param analyzer: Analyzer object

        """
        assert isinstance(analyzer, Analyzer), "Analysis needs an Analyzer object to initialize!"
        self.analyzer = analyzer

    def getChannelName(self):
        """
        Returns the name of the channel that the Analyzer is based on.

        :return: self.name of the Channel that the base Analyzer is based on.
        :rtype: str
        """
        return self.analyzer.channel.name

class FlowAnalysis(Analysis):
    """
    Base object for analysis of optical flow and PIV.

    Works on FlowAnalyzer objects, such as FarenbackAnalyzer and OpenPIVAnalyzer. Needs a 4D (t, x, y, uv) numpy array
    representing a time lapse of a vector field to initialize.

    """
    def __init__(self, analyzer):
        assert isinstance(analyzer, FlowAnalyzer), "FlowAnalysis works on FlowAnalyzer objects!"
        super().__init__(analyzer)

    def _draw_flow_frame(self, img, flow, step=15, scale=20, line_thicknes=2):
        """
        Helper function to draw flow arrows on an singe image frame

        :param img: Background image (2D) of same xy shape as flow
        :type img: numpy.ndarray
        :param flow: 2D uv flow array (1 frame)
        :param step: pixels between arrows
        :param scale: length scaling of arrows
        :param line_thicknes: thickenss of lines
        :return: image
        :rtype: numpy.ndarray
        """
        h, w = img.shape[:2]
        y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
        fx, fy = flow[y, x].T * scale
        lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
        lines = np.int32(lines + 0.5)
        vis = img.copy()
        cv.polylines(vis, lines, 0, 255, line_thicknes)
        # for (x1, y1), (_x2, _y2) in lines:
        # radius = int(math.sqrt((x1-_x2)**2+(y1-_y2)**2))
        # cv.circle(vis, (x1, y1), 1, 255, 1)

        return vis

    def _draw_scalebar(self, img, pxlength):
        """
        Draws a white scale bar in the bottom right corner

        :param img: 2D 8-bit np array to draw on
        :param pxlength: (int) length of scale bar in pixels
        :return: 2D 8-bit np array image with scale bar drawn on.

        """

        h, w = img.shape[:2]
        from_x = w - 32
        from_y = h - 50
        to_x = from_x - pxlength
        to_y = from_y
        vis = img.copy()
        cv.line(vis, (from_x, from_y), (to_x, to_y), 255, 5)

        return vis

    def draw_all_flow_frames_superimposed(self, scalebarFlag=False, scalebarLength=10, **kwargs):
        """
        Draws flow superimposed on the background channel as an 8-bit array.

        Draws a subset of the flow as lines on top of the background channel. Because the flow represents what happens
        between frames, the flow is not drawn on the las frame of the channel, which is discarded. Creates and populates
        self.drawnframes to store the drawn array. If the underlying channel object is 16-bit, it will converted to 8bit
        with the `channel.normailzation_to_8bit()` function.

        :param scalebarFlag: Should a scale bar be drawn on the output?
        :type scalebarFlag: bool
        :param scalebarLength: What speed should the scale bar represent with its length the unit is set by the unit given to the Analyzer
        :param kwargs: Additional arguments passed to self._draw_flow_frame()
        :type kwargs: dict

        :return: 8bit numpy array
        """
        flows = self.analyzer._getFlows()
        bg = self.analyzer.channel.getArray()
        outshape = (flows.shape[0], flows.shape[1], flows.shape[2])
        out = np.empty(outshape, dtype='uint8')
        scale = kwargs.get("scale", 1)
        scalebar_px = int(scale * scalebarLength / self.analyzer.scaler)

        if bg.dtype != np.dtype('uint8'):
            bg = channel.normalization_to_8bit(bg)

        for i in range(out.shape[0]):

            out[i] = self._draw_flow_frame(bg[i], flows[i], **kwargs)
            if scalebarFlag:
                out[i] = self._draw_scalebar(out[i], scalebar_px)

        self.drawnFrames = out

        return out

    def draw_all_flow_frames(self, scalebarFlag=False, scalebarLength=10, **kwargs):
        """
        Draws flow superimposed on the background channel as an 8-bit array.

        Draws a subset of the flow as lines on top of the background channel. Because the flow represents what happens
        between frames, the flow is not drawn on the las frame of the channel, which is discarded. Creates and populates
        self.drawnframes to store the drawn array. If the underlying channel object is 16-bit, it will converted to 8bit
        with the `channel.normailzation_to_8bit()` function.

        :param scalebarFlag: Should a scale bar be drawn on the output?
        :type scalebarFlag: bool
        :param scalebarLength: What speed should the scale bar represent with its length the unit is set by the unit given to the Analyzer
        :param kwargs: Additional arguments passed to self._draw_flow_frame()
        :type kwargs: dict

        :return: 8bit numpy array
        """

        flows = self.analyzer._getFlows()
        bg = np.zeros_like(self.analyzer.channel.getArray())
        outshape = (flows.shape[0], flows.shape[1], flows.shape[2])
        out = np.empty(outshape, dtype='uint8')
        scale = kwargs.get("scale", 1)
        scalebar_px = int(scale * scalebarLength / self.analyzer.scaler)

        if bg.dtype != np.dtype('uint8'):
            bg = channel.normalization_to_8bit(bg)

        for i in range(out.shape[0]):

            out[i] = self._draw_flow_frame(bg[i], flows[i], **kwargs)
            if scalebarFlag:
                out[i] = self._draw_scalebar(out[i], scalebar_px)

        self.drawnFrames = out

        return out

    def _rehapeDrawnFramesTo6d(self):
        # reshapes 3D (t, x, y) array to (t, 1, 1, x, y, 1) for saving dimensions in TZCYXS order

        if (len(self.drawnFrames.shape) == 6):
            return None

        shape = self.drawnFrames.shape
        self.drawnFrames.shape = (shape[0], 1, 1, shape[1], shape[2], 1)

    def saveFlowAsTif(self, outpath):
        """
        Saves the drawn frames as an imageJ compatible tif with rudimentary metadata.

        :param outpath: Path to savefolder
        :type outpath: Path object
        :return: None
        """
        assert self.drawnFrames is not None, "No frames drawn!"
        fname = self.getChannelName()+"_flow.tif"
        savename = outpath / fname

        self._rehapeDrawnFramesTo6d()
        arr_to_save = self.drawnFrames

        print("Saving flow...")

        finterval_s = self.analyzer.channel.finterval_ms / 1000
        ij_metadatasave = {'unit': 'um', 'finterval': finterval_s,
                           'tunit': 's', 'Info': "None",
                           'frames': self.analyzer.flows.shape[0],
                           'slices': 1, 'channels': 1}

        tifffile.imwrite(savename, arr_to_save.astype(np.uint8),
                     imagej=True, resolution=(1 / self.analyzer.channel.pxSize_um, 1 / self.analyzer.channel.pxSize_um),
                     metadata=ij_metadatasave
                     )

        print("File done!")

        return




class FlowSpeedAnalysis(FlowAnalysis):
    """
    Handles all analysis and data output of speeds from FlowAnalyzers.

    Calculates pixel-by pixel speeds from flow vectors.

    """
    def __init__(self, analyzer):
        super().__init__(analyzer)
        self.speeds = None  # (t ,x, y) 3D numpy-array
        self.avg_speeds = None  # 1D numpy array of frame average speeds
        self.histograms = None  # populated by calculateHistograms

    def calculateSpeeds(self, scaler=None):
        """
        Calculates speeds from the flows in parent Anlayzer

        Turns a (t, x, y, uv) flow numpy array with u/v component vectors in to a (t, x, y) speed array. Populates
        self.speeds. Scales all the output by multiplying with scaler, defaluts to using the self.scaler from the base
        FlowAnalyzer object if the scaler argument is ``None`` .

        :returns None

        """

        if scaler is None:
            scaler = self.analyzer.scaler

        assert isinstance(scaler, (int, float)), "scaler has to be int or float!"
        flows = self.analyzer.flows

        out = np.square(flows)
        out = out.sum(axis=3)
        out = np.sqrt(out) * scaler
        self.speeds = out

    def calculateAverageSpeeds(self):
        """
        Calculates the average speed for each time point in self.speeds

        :return: self.avg_speeds
        :rtype: 1D numpy.ndarray of the same length as self.speeds

        """
        if self.speeds is None:
            self.calculateSpeeds()

        #sometiimes OpenPIV genereates NaN values
        if np.isnan(self.speeds).any():
            self.avg_speeds = np.nanmean(self.speeds, axis=(1, 2))

        else:
            self.avg_speeds = self.speeds.mean(axis=(1,2))

        self.avg_speeds.shape = self.avg_speeds.shape[0] #make sure array is 1D

        return self.avg_speeds

    def calculateHistograms(self, hist_range=None, nbins=100, density=True):
        """
        Calculates a histogram for each frame in self.speeds

        :param hist_range: Range of histogram, defaults to 0-max
        :type hist_range: tuple
        :param nbins: Number of bins in histogram, defaults to 100
        :type nbins: int
        :param density: If ``False``, the result will contain the number of samples in each bin. If ``True`` (default),
                        the result is the value of the probability density function at the bin, normalized such that the
                        integral over the range is 1.

        :type density: bool
        :return: self.histograms
        :rtype: tuple (numpy.ndarray, bins)

        """

        if self.speeds is None:
            self.calculateSpeeds()

        if hist_range == None:
            hist_range = (0, self.speeds.max())

        print("Histogram range: {}".format(hist_range))

        hists = np.empty((self.speeds.shape[0], nbins), dtype=np.float32)

        for i in range(self.speeds.shape[0]):
            hist = np.histogram(self.speeds[i], bins=nbins, range=hist_range, density=True)
            hists[i] = hist[0]

        #bins are only stored once, because they are identical for all timepoints
        bins = hist[1]

        self.histograms = (hists, bins)

        return self.histograms

    def getAvgSpeeds(self):
        """
        Returns average speed per frame as a 1D Numpy array.

        :return: average speed per frame
        :rtype: numpy.ndarray (1D)

        """
        if self.avg_speeds is None:
            self.calculateAverageSpeeds()

        return self.avg_speeds

    def getSpeeds(self):
        """
        Returns self.speeds.

        Calculates self.speeds with default values if it has not already been calculated.

        :return: self.speeds as a 3D Numpy array
        :rtype: numpy.ndarray (3D)

        """
        if self.speeds is None:
            self.calculateSpeeds()

        return self.speeds

    def plotHistogram(self, frame):
        """
        Plots the histogram for the supplied frame.

        Uses Pyplot to create a histogram plot and displays it to the user.

        :param frame: frame to plot
        :type frame: int

        :return: Pyplot object
        """
        assert self.histograms is not None, "speed histograms have not been calculated!"

        hist = self.histograms[0][frame]
        bins = self.histograms[1]
        width = 0.7 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2
        plt.bar(center, hist, align='center', width=width)
        plt.show()

    def saveSpeedArray(self, outdir, fname=None):
        """
        Saves the speed array as a 32-bit tif with imageJ metadata.

        Pixel intensities encode speeds in the chosen analysis unit

        :param outdir: Directory to store file in
        :type outdir: pathlib.Path
        :param fname: Filename, defaults to Analysis channel name with appended tags +_speeds-SizeUnit-per-TimeUnit.tif
                      if ``None``

        :return: None
        """

        original_shape = self.speeds.shape
        #imageJ hyperstacks need 6D arrays for saving
        channel.rehape3DArrayTo6D(self.speeds)

        if fname == None:
            #Replace slash character in unit with space_per_time
            unit =self.analyzer.unit.replace("/", "-per-")
            fname = self.analyzer.channel.name + "_speeds-"+unit+".tif"


        saveme = outdir / fname

        ij_metadatasave = {'unit': 'um', 'finterval': round(self.analyzer.channel.finterval_ms / 1000, 2),
                           'tunit': "s", 'frames': self.speeds.shape[0],
                           'slices': 1, 'channels': 1}

        tifffile.imwrite(file=saveme,
                         data=self.speeds.astype(np.float32),
                         imagej=True,
                         resolution=(1 / self.analyzer.channel.pxSize_um, 1 / self.analyzer.channel.pxSize_um),
                         metadata=ij_metadatasave)

        #restore original array shape in case further analysis is performed
        self.speeds.shape = original_shape

    def saveSpeedCSV(self, outdir, fname=None, tunit = "s"):
        """
        Saves a csv of average speeds per frame in outdir.

        :param outdir: Directory where output is stored
        :type outdir: pathlib.Path
        :param fname: filename, defaults to channel name + speeds.csv
        :type fname: str
        :param tunit: Time unit in output one of: "s", "min", "h", "days"
        :type tunit: str
        :return:
        """
        # print("Saving csv of mean speeds...")

        if fname is None:
            fname = self.analyzer.channel.name + "_speeds.csv"

        arr = self.getAvgSpeeds()

        time_multipliers = {
            "s": 1,
            "min": 1/60,
            "h": 1/(60*60),
            "days": 1/(24*60*60)
        }
        assert tunit in time_multipliers.keys(), "tunit has to be one of: " + str(time_multipliers.keys())

        fr_interval_multiplier = time_multipliers.get(tunit) * (self.analyzer.channel.finterval_ms/1000)

        timepoints_abs = np.arange(0, arr.shape[0], dtype='float32') * fr_interval_multiplier

        df = pd.DataFrame(arr, index=timepoints_abs, columns=["AVG_frame_flow_" + self.analyzer.unit])
        df.index.name = "Time("+tunit+")"

        saveme = outdir / fname
        df.to_csv(saveme)

    def saveSpeedCSV(self, outdir, fname=None, tunit = "s"):
        """
        Saves a csv of average speeds per frame in outdir.

        :param outdir: Directory where output is stored
        :type outdir: pathlib.Path
        :param fname: filename, defaults to channel name + speeds.csv
        :type fname: str
        :param tunit: Time unit in output one of: "s", "min", "h", "days"
        :type tunit: str
        :return:
        """
        # print("Saving csv of mean speeds...")

        if fname is None:
            fname = self.analyzer.channel.name + "_speeds.csv"

        arr = self.getAvgSpeeds()

        time_multipliers = {
            "s": 1,
            "min": 1/60,
            "h": 1/(60*60),
            "days": 1/(24*60*60)
        }
        assert tunit in time_multipliers.keys(), "tunit has to be one of: " + str(time_multipliers.keys())

        fr_interval_multiplier = time_multipliers.get(tunit) * (self.analyzer.channel.finterval_ms/1000)

        timepoints_abs = np.arange(0, arr.shape[0], dtype='float32') * fr_interval_multiplier

        df = pd.DataFrame(arr, index=timepoints_abs, columns=["AVG_frame_flow_" + self.analyzer.unit])
        df.index.name = "Time("+tunit+")"

        saveme = outdir / fname
        df.to_csv(saveme)





class AlignmentIndex(Analysis):
    """
    Calculates the alignment index as defined as in Malinverno et. al 2017.

    The alignment index is 1 when the local velocity is parallel to the mean direction of migration  (-1 if antiparallel).

    """

    def __init__(self, analyzer, returnMagnitudesFlag=False):
        super().__init__(analyzer)
        self.returnMagnitudesFlag = returnMagnitudesFlag




def analyzeFiles(fnamelist, outdir, flowkwargs, scalebarFlag, scalebarLength, chToAnalyze=0, unit="um/min"):
    """
    Automatically analyzes tifffiles and saves ouput in outfolder. If input has two channels, analysis is run on
    channel index 1 by default, but can be changed.

    :param tif: Tifffile objekt
    :return:

    """

    for fname in fnamelist:

        with tifffile.TiffFile(fname, multifile=False) as tif:

            lab = os.path.split(fname)[1][:-4]
            print("Working on: {} as {}".format(fname, lab))
            t1 = time.time()
            ij_metadata = tif.imagej_metadata
            n_channels = int(ij_metadata.get('channels', 1))

            Ch0 = Channel(chToAnalyze, tif, name=lab + "_Ch" + str(chToAnalyze + 1))

            print("Elapsed for file load and Channel creation {:.2f} s.".format(time.time() - t1))
            finterval_s = round(Ch0.finterval_ms / 1000, 2)
            frames_per_min = round(60 / finterval_s, 2)
            tunit = 's'
            print(
                "Intended dimensions: frame interval {:.2f}s, {:.2f} frames/min, pixel size: {:.2f} um ".format(
                    finterval_s, frames_per_min, Ch0.pxSize_um))

            print("Actual frame interval is: {:.2f} s".format(Ch0.getActualFrameIntevals_ms().mean() / 1000))

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

            if n_channels >= 2:

                if chToAnalyze == 0:
                    print("Loading Channel {}".format(2))
                    Ch1 = Channel(1, tif, name=lab + "_Ch2")
                else:
                    print("Loading Channel {}".format(1))
                    Ch1 = Channel(0, tif, name=lab + "_Ch1")

                print("Start median filter of the other channel ...")
                Ch1.getTemporalMedianFilterArray()
                Ch1.medianArray = normalization_to_8bit(Ch1.medianArray, lowPcClip=10, highPcClip=0)
                Ch1.rehapeArrayTo6D()

                savename = os.path.join(outdir, lab + "_2Chan_flow.tif")
                # print(Analysis_Ch0.drawnFrames.shape, Ch1.medianArray[:stopframe-3].shape)
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
