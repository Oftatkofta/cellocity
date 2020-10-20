import numpy as np
import cv2 as cv
import os, time, math
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
        Returns current progress in the interval 0-100.

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
        self.pxSize_um = None #PIV analysis changes pixel size
        self.flows = None  # (t, x, y, uv) numpy array
    
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

    def get_flow_shape(self):
        """
        Returns the shape of self.flows

        :return: the shape of self.flows
        :rtype: tuple
        """

        return self.flows.shape

    def get_pixel_size(self):
        """
        Returns the pixel size in um of the Analyzer.

        Some type of Analyzers, such as the OpenPivAnalyzer change the pixel size of the array by downsampling.

        :return: pixel size in um of the Analyzer
        :rtype: float
        """

        return self.pxSize_um

    def _getFlows(self):
        if self.flows is None:
            warnings.warn("No flow has been calculated!")
        return self.flows


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
        self.pxSize_um = self.channel.pxSize_um #optical flow does not change pixel size
    
    def doFarenbackFlow(self, pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0):
        """
        Calculates Farenback flow for a single channel time lapse with validated default parameters.
    
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
        #OpenCV places (0, 0) in upper left corner, so y-values needs to be flipped
        y = arr.shape[2] - y
    
        # Zero-filled output arrays are created beforehand for maximal performance
        out_u = np.zeros((n_frames, x.shape[0], x.shape[1]))
        out_v = np.zeros_like(out_u)

        self.pxSize_um = arr.shape[1]/x.shape[0] * self.channel.pxSize_um
    
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
            #v-array needs to be flipped
            out_v[i] = -out_v[i]
    
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
    Base object for handling data output and analysis and  of Analyzer classes.

    """
    
    def __init__(self, analyzer):
        """
        :param analyzer: Analyzer object
    
        """
        assert isinstance(analyzer, Analyzer), "Analysis needs an Analyzer object to initialize!"
        self.analyzer = analyzer
    
    def getChannelName(self):
        """
        Returns the name of the channel that the base Analyzer, in turn, is based on.
    
        :return: self.name of the Channel that the base Analyzer is based on.
        :rtype: str
        """
        return self.analyzer.channel.name

    def getAnalyzer(self):
        """
        Returns the Analyzer that the Analysis is based on.

        :return: the Analyzer that the Analyser is based on.
        :rtype: Analyzer
        """
        return self.analyzer

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
        Helper function to draw flow arrows on an singe image frame.
    
        If the flow was generated with OpenPIV, `_draw_open_piv_frame()´ will be called instead.
    
        :param img: Background image (2D) of same xy shape as flow
        :type img: numpy.ndarray
        :param flow: 2D uv flow array (1 frame)
        :param step: pixels between arrows
        :param scale: length scaling of arrows
        :param line_thicknes: thickenss of lines
        :return: image
        :rtype: numpy.ndarray
        """
        if type(self.analyzer) is OpenPivAnalyzer:
    
            return self._draw_open_piv_frame(img, flow, scale, line_thicknes)
    
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
    
    def _draw_open_piv_frame(self, bg, flow, scale, line_thicknes):
        """
        Draws scaled optical from an OpenPIVAnalyser on background image. Visualizes the entire flow.
    
        :param img:
        :param flow:
        :param scale:
        :param line_thicknes:
        :return:
        """
        #scale = kwargs.get("scale",1)
        #line_thicknes = kwargs.get("line_thicknes", 2)
    
        from_coord = self.analyzer.flow_coordinates
        to_coord = np.multiply(flow, scale)
        #replace NaN with 0, because sometimes OpenPIV gives NaNs
        to_coord = np.nan_to_num(to_coord, 0)
        to_coord = to_coord + from_coord
    
        vis = bg.copy()
       # assert (to_coord.shape == from_coord.shape) and (len(to_coord) == 2), "Only single frames supported!"
    
        for row in range(from_coord.shape[0]):
            for col in range(from_coord.shape[1]):
                fromX, fromY = from_coord[row][col]
                toX, toY = to_coord[row][col]
                cv.line(vis, (fromX, fromY), (toX, toY), (255,255,255), line_thicknes)
    
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
        between frames, the flow is not drawn on the last frame of the channel, which is discarded. Creates and populates
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
        outshape = (flows.shape[0], bg.shape[1], bg.shape[2])
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
        Draws flow on a black background as an 8-bit array.
    
        Draws a subset of the flow as lines on top of a black background. Because the flow represents what happens
        between frames, the flow is not drawn on the last frame of the channel, which is discarded. Creates and populates
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
        outshape = (flows.shape[0], bg.shape[1], bg.shape[2])
        out = np.empty(outshape, dtype='uint8')
        scale = kwargs.get("scale", 1)
        scalebar_px = int(scale * scalebarLength / self.analyzer.scaler)
    
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
        if type(self.analyzer)==FarenbackAnalyzer:
            suffix = "_flow.tif"
        if type(self.analyzer)==OpenPivAnalyzer:
            suffix = "_PIV.tif"
    
        fname = self.getChannelName()+suffix
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

    Calculates pixel-by-pixel speeds from flow vectors.
    
    """
    def __init__(self, analyzer):
        super().__init__(analyzer)
        self.speeds = None  # (t ,x, y) 3D numpy-array
        self.avg_speeds = None  # 1D numpy array of frame average speeds
        self.histograms = None  # populated by calculateHistograms
    
    def calculateSpeeds(self, scaler=None):
        """
        Calculates speeds from the flows in parent Analyzer
        
        Turns a (t, x, y, uv) flow numpy array with u/v component vectors in to a (t, x, y) speed array. Populates
        self.speeds. Scales all the output by multiplying with scaler, defaults to using the self.scaler from the base
        FlowAnalyzer object if the scaler argument is ``None``.
    
        self.scaler is the scalar quantity that converts flow vectors from the general unit of pixels/frame in to the
        desired output unit, such as um/s.
    
        :returns self.speeds
        :rtype: numpy.ndarray
    
        """
    
        if scaler is None:
            scaler = self.analyzer.scaler
    
        assert isinstance(scaler, (int, float)), "scaler has to be int or float!"
        flows = self.analyzer.flows
    
        out = np.square(flows)
        out = out.sum(axis=3)
        out = np.sqrt(out) * scaler
        self.speeds = out
    
        return self.speeds
    
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
            hist = np.histogram(self.speeds[i], bins=nbins, range=hist_range, density=density)
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
    
    def getAvgSpeedsAsDf(self):
        """
        Returns frame and average speed for the frame as a Pandas DataFrame.
    
        :return: DataFrame with 1 column for average speed and index = frame number
        :rtype: pandas.DataFrame
        """
    
        if self.avg_speeds is None:
            self.calculateAverageSpeeds()
    
        arr = self.getAvgSpeeds()
    
        df = pd.DataFrame(arr, columns=["AVG_speed_" + self.analyzer.unit])
    
        return df
    
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
    
    def _plotHistogram(self, frame):
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
    
    def saveArrayAsTif(self, outdir, fname=None):
        """
        Saves the speed array as a 32-bit tif with imageJ metadata.
    
        Pixel intensities encode speeds in the chosen analysis unit
    
        :param outdir: Directory to store file in
        :type outdir: pathlib.Path
        :param fname: Filename, defaults to Analysis channel name with appended tags +_speeds-SizeUnit-per-TimeUnit.tif
                      if ``None``
    
        :return: None
        """
        assert self.speeds is not None, "Speeds not calculated!"
    
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
    
    def saveCSV(self, outdir, fname=None, tunit ="s"):
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

class AlignmentIndexAnalysis(FlowAnalysis):
    """
    Calculates the alignment index for the flow vectors in a FlowAnalyzer object.

    Alignment index (AI) is defined as in Malinverno et. al 2017. For every frame the AI is the average of the dot
    products of the mean velocity vector with each individual vector, all divided by the product of their
    magnitudes.
    
    The alignment index is 1 when the local velocity is parallel to the mean direction of migration  (-1 if antiparallel).
    
    """
    def __init__(self, analyzer):
        super().__init__(analyzer)
        self.alignment_idxs = None
        self.avg_alignment_idxs = None
    
    def calculateAlignIdxs(self):
        """
        Calculates the aligment index for each pixel in base FlowAnalyzer flow array and populates self.alignment_idxs
        
        :return: numpy array with same size as analyzer flows, where every entry is the alignment index in that pixel
        :rtype: numpy.ndarray
        """
        flows = self.analyzer._getFlows()
        self.alignment_idxs = np.empty((flows.shape[0], flows.shape[1], flows.shape[2]))
    
        for frame in range(flows.shape[0]):
            u = self.analyzer.get_u_array(frame)
            v = self.analyzer.get_v_array(frame)
            self.alignment_idxs[frame] = self._alignment_index(u, v)
    
        return self.alignment_idxs
    
    def _alignment_index(self, u, v):
        """
        Returns an array of the same shape as u and v with the alignment index (ai).
    
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
    
    def saveArrayAsTif(self, outdir, fname=None):
        """
        Saves the alignment index array as a 32-bit tif with imageJ metadata.
    
        Pixel intensities encode alignment indexes.
    
        :param outdir: Directory to store file in
        :type outdir: pathlib.Path
        :param fname: Filename, defaults to Analysis channel name with appended tags +_ai.tif
                      if ``None``
    
        :return: None
        """
        assert self.alignment_idxs is not None, "Alignment indexes not calculated!"
    
        original_shape = self.alignment_idxs.shape
        #imageJ hyperstacks need 6D arrays for saving
        channel.rehape3DArrayTo6D(self.alignment_idxs)
    
        if fname == None:
            fname = self.analyzer.channel.name + "_ai.tif"


        saveme = outdir / fname
    
        ij_metadatasave = {'unit': 'um', 'finterval': round(self.analyzer.channel.finterval_ms / 1000, 2),
                           'tunit': "s", 'frames': self.alignment_idxs.shape[0],
                           'slices': 1, 'channels': 1}
    
        tifffile.imwrite(file=saveme,
                         data=self.alignment_idxs.astype(np.float32),
                         imagej=True,
                         resolution=(1 / self.analyzer.channel.pxSize_um, 1 / self.analyzer.channel.pxSize_um),
                         metadata=ij_metadatasave)
    
        #restore original array shape in case further analysis is performed
        self.alignment_idxs.shape = original_shape
    
    def calculateAverage(self):
        """
        Calculates the average alignment index for each time point in self.alignment_idxs
    
        :return: self.avg_alignment_idxs, 1D numpy.ndarray of the same length as self.alignment_idxs
        :rtype: numpy.ndarray
    
        """
        if self.alignment_idxs is None:
            self.calculateAlignIdxs()
    
        #sometiimes OpenPIV genereates NaN values
        if np.isnan(self.alignment_idxs.any()):
            self.avg_alignment_idxs = np.nanmean(self.alignment_idxs, axis=(1, 2))
    
        else:
            self.avg_alignment_idxs = self.alignment_idxs.mean(axis=(1,2))
    
        self.avg_alignment_idxs.shape = self.avg_alignment_idxs.shape[0] #make sure array is 1D
    
        return self.avg_alignment_idxs
    
    def getAvgAlignIdxs(self):
        """
        Returns average alignment indexes for Analyzer
    
        :return:
        """
        if self.avg_alignment_idxs is None:
            self.calculateAverage()
    
        return self.avg_alignment_idxs
    
    def getAvgAlignIdxAsDf(self):
        """
        Returns frame and average alignment index for the frame as a Pandas DataFrame.
    
        :return: DataFrame with 1 column for average aligmnent index and index = frame number
        :rtype: pandas.DataFrame
        """
    
        if self.avg_alignment_idxs is None:
            self.calculateAverage()
    
        arr = self.getAvgAlignIdxs()
    
        df = pd.DataFrame(arr, columns=["AVG_alignment_index"])
    
        return df
    
    def saveCSV(self, outdir, fname=None, tunit="s"):
    
        """
        Saves a csv of average aligmnent indexes per frame in outdir.
    
        :param outdir: Directory where output is stored
        :type outdir: pathlib.Path
        :param fname: filename, defaults to channel name + ai.csv
        :type fname: str
        :param tunit: Time unit in output one of: "s", "min", "h", "days"
        :type tunit: str
        :return:
        """
    
        if fname is None:
            fname = self.analyzer.channel.name + "_ai.csv"
    
        arr = self.getAvgAlignIdxs()
    
        time_multipliers = {
            "s": 1,
            "min": 1 / 60,
            "h": 1 / (60 * 60),
            "days": 1 / (24 * 60 * 60)
        }
        assert tunit in time_multipliers.keys(), "tunit has to be one of: " + str(time_multipliers.keys())
    
        fr_interval_multiplier = time_multipliers.get(tunit) * (self.analyzer.channel.finterval_ms / 1000)
    
        timepoints_abs = np.arange(0, arr.shape[0], dtype='float32') * fr_interval_multiplier
    
        df = pd.DataFrame(arr, index=timepoints_abs, columns=["AVG_alignment_idx_" + self.analyzer.unit])
        df.index.name = "Time(" + tunit + ")"
    
        saveme = outdir / fname
        df.to_csv(saveme)

class IopAnalysis(FlowAnalysis):
    """
    Calculates the instantaneous order parameter (iop) for each frame of flow (see Malinverno et. al 2017 for a more
    detailed explanation).

    The iop is a measure of how similar the vectors in a field are, which takes in to account both the
    direction and magnitudes of the vectors. iop is always between 0 and 1, with iop = 1 being a perfectly uniform field
    of identical vectors, and iop = 0 for a perfectly random field.
    """
    def __init__(self, flowanalyzer):
        """
        :param flowanalyzer: a FlowAnalyzer object
        :type flowanalyzer: analysis.FlowAnalyzer
        """
        super().__init__(flowanalyzer)
        self.iops = None

    def _msv(self, u, v):  # Mean Square Velocity
        """
         Calculates the mean square velocity of one frame from flow data

        :param u:
            2D numpy array with the u component of velocity vectors
        :param v:
            2D numpy array with the v component of velocity vectors
        :return:
            (float) the mean square velocity of the velocity vectors in the frame
        """

        msv = np.mean(np.square(u) + np.square(v))

        return msv

    
    def _smvvm(self, u, v):  # Square Mean Vectorial Velocity Magnitude
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
    
    def _instantaneous_order_parameter(self, u, v):
        """
        Calculates the instantaneous order parameter (iop) in one flow frame see  Malinverno et. al 2017 for a more detailed
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
        return self._smvvm(u, v) / self._msv(u, v) #square_mean_vectorial_velocity_magnitude/Mean Square Velocity

    def calculateIops(self):
        """
        Calculates the IOP for each frame in base FlowAnalyzer flow array and populates self.iops

        :return: list of the IOP from each frame
        :rtype: list
        """

        flows = self.analyzer._getFlows()
        self.iops = []

        for frame in range(flows.shape[0]):
            u = self.analyzer.get_u_array(frame)
            v = self.analyzer.get_v_array(frame)
            self.iops.append(self._instantaneous_order_parameter(u, v))

        return self.iops

    def getIops(self):
        """
        Returns the instantaneous order parameter for Analyzer

        :return: list of instantaneous order parameters
        :rtype: list

        """
        if self.iops is None:
            self.calculateIops()

        return self.iops

    def getIopsAsDf(self):
        """
        Returns frame and iop for the frame as a Pandas DataFrame.

        :return: DataFrame with 1 column for iop and index = frame number
        :rtype: pandas.DataFrame
        """

        if self.iops is None:
            self.calculateIops()

        arr = self.getIops()

        df = pd.DataFrame(arr, columns=["IOP"])

        return df

    def saveCSV(self, outdir, fname=None, tunit="s"):

        """
        Saves a csv of the iop per frame in outdir.

        :param outdir: Directory where output is stored
        :type outdir: pathlib.Path
        :param fname: filename, defaults to channel name + iop.csv
        :type fname: str
        :param tunit: Time unit in output one of: "s", "min", "h", "days"
        :type tunit: str
        :return:
        """

        if fname is None:
            fname = self.analyzer.channel.name + "_iop.csv"

        arr = self.getIops()

        time_multipliers = {
            "s": 1,
            "min": 1 / 60,
            "h": 1 / (60 * 60),
            "days": 1 / (24 * 60 * 60)
        }
        assert tunit in time_multipliers.keys(), "tunit has to be one of: " + str(time_multipliers.keys())

        fr_interval_multiplier = time_multipliers.get(tunit) * (self.analyzer.channel.finterval_ms / 1000)

        timepoints_abs = np.arange(0, arr.shape[0], dtype='float32') * fr_interval_multiplier

        df = pd.DataFrame(arr, index=timepoints_abs, columns=["instantaneous_order_parameter"])
        df.index.name = "Time(" + tunit + ")"

        saveme = outdir / fname

        df.to_csv(saveme)


class FiveSigmaAnalysis(FlowAnalysis):
    """
    Calculates the 5-sigma corrleation length for each frame of flow (see Lång et. al 2018 or the documentation for a
    more detailed explanation).

    The 5-:math:`{\sigma}` correlation length was defined as the largest distance, :math:`r`, where the average angle
    between two velocity vectors :math:`r` micrometers apart was :math:`<90°` with a statistical significance level of
    5 :math:`\sigma` :math:`(p=3×10^{−7})`.

    """

    def __init__(self, flowanalyzer, maxdist=None):
        """
        :param flowanalyzer: a FlowAnalyzer object
        :type flowanalyzer: analysis.FlowAnalyzer
        :param maxdist: Maximum distance (in pixels) to test if None defaults to max(flow width, height)
        :type maxdist: int
        """
        super().__init__(flowanalyzer)
        self.flow_shape = self.getAnalyzer().get_flow_shape()
        self.diagonalCoordinates = self._calculate_diagonal_coordinates() #list of tuples
        self.distanceAngleDict = {}

        if maxdist is None:
            self.r = max(self.flow_shape[1], self.flow_shape[2])
        else:
            self.r = maxdist

        self.lcorrs = {}  # frame:correlation length in um
        self._process_times = {} # frame:time_to_process_lcorr in seconds

    def _calculate_diagonal_coordinates(self):
        """
        creates a list of coordinates that run down the shortest diagonal of the flow array

        :return: list of coordinate tuples
        """

        out = []
        nrows = self.flow_shape[1]
        ncols = self.flow_shape[2]

        for diagonal in range(0, min(nrows, ncols)):  # follow the shortest diagonal
            out.append((diagonal, diagonal))

        return out

    def _create_distanceAngleDict(self):

        nframes = self.flow_shape[0]
        distanceAngleDict = {}

        for f in range(nframes):
            distanceAngleDict[f] = {}

        return  distanceAngleDict

    def _get_v0_plus_r_coordinates_cardinal(self, v0_cord, r):
        """
        Gets the 4 coordinates of the pixels that are r pixels away from the coordinate (v0_x, v0_y) along the four
        cardinal directions. Coordinates are returned in the order: up/top, right, down/bottom, left.

        The coordinates are returned in Matrix/numpy form (row,col), i.e. (y,x) when compared to traditional image
        coordinate numbering. The input coordinate for v0 is also expected to be in this format (y_coord, x_coord).

        If a coordinate falls outside of the image it is not returned.

        :param v0_cord: (tuple) of (ints), (row, col) coordinates of v0 in matrix notation.
        :type v0_cord: tuple
        :param r: distance in pixels from v0 along the cardinal directions
        :type r: int
        :return: list coordinates in martix notation, if no valid coordinates are found an empty list is returned.
        :rtype: list

        """

        array_width = self.flow_shape[2]  # number of columns
        array_height = self.flow_shape[1]  # number of rows
        v0_r = v0_cord[0]  # row number of v0
        v0_c = v0_cord[1]  # column number of v0

        assert r > 0, "r needs to be positive and >0 !"
        assert array_width > v0_c, "v0_y needs to be less than array_width!"
        assert v0_c >= 0, "v0_y needs to be positive!"
        assert array_height > v0_r, "v0_y needs to be < array_height!"
        assert v0_r >= 0, "v0_y needs to be positive and < array_height!"

        top_r = v0_r - r #top row
        right_c = v0_c + r #right column
        bottom_r = v0_r + r
        left_c = v0_c - r

        out = []

        if (top_r >= 0):
            out.append((top_r, v0_c))

        if (right_c < array_width):
            out.append((v0_r, right_c))

        if (bottom_r < array_height):
            out.append((bottom_r, v0_c))

        if (left_c >= 0):
            out.append((v0_r, left_c))

        return out


    def _get_all_angles(self, frame, v0_coord, r_step=1, r_min=1):
        """
        Gets the vector, v0 from the given frame and the coordinate (v0_coord) from a velocity vector field. Grows the distance r from r_min to r_max
        in increments of r_step. For each distance r calculates the average of the (cos) angles between v0 and v0+r in the
        four cardinal directions. The result is stored in a dictionary where the keys are distances and the values are lists
        of average angles for that distance. The updated resultsDict is returned.

        :param frame: frame to get v0 from
        :param v0_coord: (tuple) Coordinates (row, column) of vector_0
        :param r_max: (int) maximum distance in pixels from v0 to calculate the angels for
        :param r_step: (int) step size to grow the distance r by
        :param r_min: (int) starting distance from v0 to calculate the angels for
        :return: (dict) input resultDict updated with key = radius, value = list of means for cos(theta) v_0-v_r {radius:[list of mean angles]}
        """
        u_array = self.getAnalyzer().get_u_array(frame)
        v_array = self.getAnalyzer().get_v_array(frame)
        r_max = self.r

        v0_u = u_array[v0_coord]
        v0_v = v_array[v0_coord]

        magnitudes = np.sqrt(np.square(u_array) + np.square(v_array))
        v0_magnitude = magnitudes[v0_coord]

        dot_products = u_array * v0_u + v_array * v0_v  # Computes ALL the dot products with v0
        magnitudes = magnitudes * v0_magnitude  # Multiplies all magnitudes by the magnitude of v0

        if frame not in self.distanceAngleDict.keys():
            self.distanceAngleDict[frame]={}

        for r in range(r_min, r_max, r_step):
            if r not in self.distanceAngleDict[frame].keys():
                self.distanceAngleDict[frame][r] = []

            coords = self._get_v0_plus_r_coordinates_cardinal(v0_coord, r)

            if len(coords) == 0:
                break  # stop when we run out of valid coordinates
            for coordinate in coords:
                if (magnitudes[coordinate] == 0) or (magnitudes[coordinate] is None):  # avoid masked or erroneous values
                    pass
                else:
                    cos_theta = dot_products[coordinate] / magnitudes[coordinate]
                    self.distanceAngleDict[frame][r].append(cos_theta)

        return self.distanceAngleDict[frame]

    def _calculate_angels_one_frame(self, frame):
        """
        Gets all valid angles for the diagonal for the given frame.

        Populates self.distanceAngleCoordinates[frame]
        """
        for coorinate in self.diagonalCoordinates:  # follow the diagonal
            self._get_all_angles(frame, coorinate)

        #cleanup results
        for r in self.distanceAngleDict[frame].keys():
            if len(self.distanceAngleDict[frame][r]) == 0:
                self.distanceAngleDict[frame].pop(r, None)  # No need to save empty data lists, it breaks the statistics


    def _calculate_angles_all_frames(self):
        """

        :return:
        """
        nframes = self.flow_shape[0]

        for frame in range(nframes):
            self._calculate_angels_one_frame(frame)


    def calculateCorrelationOneFrame(self, frame, n_sigma=5):
        """

        :param frame: (flow) frame to calculate correlation length for
        :param n_sigma: Number of standard deviations to consider significant

        :return:
        """
        t0 = time.time()
        if frame not in self.distanceAngleDict.keys():
            self._calculate_angels_one_frame(frame)

        r_dist_um = []
        avg_angle = []
        significantfFlag = False
        px_scale = self.analyzer.get_pixel_size()

        for radius in sorted(self.distanceAngleDict[frame].keys()):
            cos_theta_list = self.distanceAngleDict[frame][radius]

            # Sometimes openPIV outputs strange values
            sanitized_cos_theta_list = [a for a in cos_theta_list if a <= 1.0]
            #if len(sanitized_cos_theta_list) != len(cos_theta_list):
            #    print("Bad angles at frame {} and radius {}, number ok: {}, not ok: {}".format(
            #        frame, radius, len(sanitized_cos_theta_list), len(cos_theta_list) - len(sanitized_cos_theta_list)))

            if len(sanitized_cos_theta_list) == 0:
                print("No acceptable angles left, aborting!")
                pass

            mean_cos_theta = np.nanmean(sanitized_cos_theta_list)
            mean_angle_degrees = math.acos(mean_cos_theta) * (180 / math.pi)
            sd_cos_theta = np.nanstd(sanitized_cos_theta_list)
            sd_angles_degrees = math.acos(sd_cos_theta) * (180 / math.pi)
            SEM_angles = sd_angles_degrees / math.sqrt(len(sanitized_cos_theta_list))

            r_dist_um.append(radius * px_scale)
            avg_angle.append(mean_angle_degrees)

            #is the mean angle significantly lower than 90?
            significantfFlag = mean_angle_degrees + n_sigma * SEM_angles <= 90

            if not significantfFlag:
                if len(r_dist_um) == 1:
                    self.lcorrs[frame] = 0
                else:
                    self.lcorrs[frame] = r_dist_um[-1]
                print("{}-sigma reached at r={} on frame {}, last significant distance was {} um".format(
                    n_sigma, radius, frame, r_dist_um[-1]))
                break

        if significantfFlag and (len(r_dist_um) == len(self.distanceAngleDict[frame].keys())):
            #still significant after entire diagonal?
            self.lcorrs[frame] = r_dist_um[-1]
            print("{}-sigma not reached at r={} on frame {}, maximum significant distance was {} um".format(
                n_sigma, radius, frame, r_dist_um[-1]))

        self._process_times[frame] = round(time.time()-t0, 2)

    def calculateCorrelationAllFrames(self, n_sigma=5):
        """
        Calculates correlation length for all flow frames

        :param n_sigma:
        :return:
        """
        for frame in range(self.flow_shape[0]):
            print("Calculating correlation for frame {}".format(frame))
            self.calculateCorrelationOneFrame(frame, n_sigma)

    def getCorrelationLengths(self):
        """
        Returns correlation lengths as a dictionary frame:correlation_length_in_um

        :return:
        """

        return self.lcorrs

    def getCorrelationLengthsAsDf(self, tunit="s"):
        """
        Returns a Pandas DataFrame with index:time in "tunit" and Correlation length.

        :param tunit: Time unit in output one of: "s", "min", "h", "days"
        :return: pandas.Dataframe
        """
        time_multipliers = {
            "s": 1,
            "min": 1 / 60,
            "h": 1 / (60 * 60),
            "days": 1 / (24 * 60 * 60)}

        assert tunit in time_multipliers.keys(), "tunit has to be one of: " + str(time_multipliers.keys())

        df = pd.DataFrame.from_dict(self.lcorrs, orient='index', columns=["Cvv_um"])
        df.sort_index(inplace=True)
        fr_interval_multiplier = time_multipliers.get(tunit) * (self.analyzer.channel.finterval_ms / 1000)
        timepoints_abs = df.index.to_numpy() * fr_interval_multiplier
        df.index = timepoints_abs
        df.index.name = "Time(" + tunit + ")"

        return df

    def saveCSV(self, outdir, fname=None, tunit="s"):
        """
        Saves a csv of correlation lengths per frame in outdir.

        :param outdir: Directory where output is stored
        :type outdir: pathlib.Path
        :param fname: filename, defaults to channel name + _Cvv.csv
        :type fname: str
        :param tunit: Time unit in output one of: "s", "min", "h", "days"
        :type tunit: str
        :return:
        """

        if fname is None:
            fname = self.analyzer.channel.name + "_Cvv.csv"

        saveme = outdir / fname
        df = self.getCorrelationLengthsAsDf(tunit)

        df.to_csv(saveme)
