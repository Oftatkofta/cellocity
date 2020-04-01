import numpy as np
import cv2 as cv
import os
import pandas as pd
import tifffile

class Analyzer(object):
    """
    Base object for all Analysis object types

    """

    def __init__(self, channel, unit):
        """
        :param channel: A Channel object
        :param unit: (str) output unit, depends on analysis
        
        """
        self.channel = channel
        self.progress = 0  # 0-100 for pyQt5 progressbar
        self.unit = unit


class FlowAnalyzer(Analyzer):
    """
    Base object for all optical flow analysis object types.

    Stores UV vector components in self.flows as a (t, x, y, uv) numpy array.
    Stores pixel speeds, scaling factors, histograms etc.

    """

    def __init__(self, channel, unit):
        super().__init__(channel, unit)
        self.flows = None  # (t, x, y, uv) numpy array
        self.speeds = None  # (t ,x, y) 3D numpy-array
        self.avg_speeds = None  # 1D numpy array of frame average speeds
        self.histograms = None  # populated by doFlowsToAvgSpeed if doHist = True
        self.drawnFrames = None  # for output visualization
        self.scaler = self._getScaler()  # value to multiply vector lengths by to get selected unit from px/frame


class FarenbackAnalyzer(FlowAnalyzer):
    """
    Implements OpenCV's Farenbäck optical flow anaysis.

    """

    def __init__(self, channel, unit):
        """
        :param channel: Channel object
        :param unit: (str) "um/s", "um/min", or "um/h"

        """
        super().__init__(channel, unit)

    def _getScaler(self):
        """
        Calculates a scalar value by which to scale from px/frame to um/min, um/h or um/s
        in the unit um*frame/px*(min/h/s)

        example:
        um/px * frames/min * px/frame = um/min

        :return: (float) scaler

        """

        if not self.channel.doFrameIntervalSanityCheck():  # Are actual and intended frame intervals within 1%?
            print("Replacing intended interval with actual!")
            finterval_ms = self.channel.getActualFrameIntevals_ms().mean()
            finterval_s = round(finterval_ms / 1000, 2)
            frames_per_min = round(60 / finterval_s, 2)
            self.channel.finterval_ms = finterval_ms

            return self.channel.pxSize_um * frames_per_min  # um/px * frames/min * px/frame = um/min

        finterval_s = self.channel.finterval_ms / 1000
        frames_per_min = round(60 / finterval_s, 2)

        if self.unit == "um/min":
            return self.channel.pxSize_um * frames_per_min

        if self.unit == "um/h":
            frames_per_h = round(60 * 60 / finterval_s, 2)
            return self.channel.pxSize_um * frames_per_h

        if self.unit == "um/s":
            return self.channel.pxSize_um * finterval_s

    def _getFlows(self):
        if (self.flows == None):
            warnings.warn("No flow has been calculated!")
        return self.flows

    def doFarenbackFlow(self, pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0):
        """
        Calculates Farenback flow for a single channel time lapse

        returns numpy array of dtype int32 with flow in the unit px/frame
        Output values need to be multiplied by a scalar to be converted to speeds.

        """

        arr = self.channel.getTemporalMedianFilterArray()

        # Create empty array for speed
        self.flows = np.empty((arr.shape[0] - 1, arr.shape[1], arr.shape[2], 2), dtype=np.float32)

        for i in range(arr.shape[0] - 1):
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
            self.progress = 100 * i / (arr.shape[0] - 1)
            print("Progress: {:.2f} % on {}".format(self.progress, self.channel.name))

        return self.flows

    def doFlowsToSpeed(self, scaler=None, doAvgSpeed=False, doHist=False, nbins=10, hist_range=None):
        """
        Turns a (t, x, y, uv) flow numpy array with u/v component vectors in to a (t, x, y) speed array
        populates self.speeds.
        If doAvgSpeed is True the 1D array self.avg_speeds is also populated.
        If doHist is True the tuple self.histograms is populated (histograms, bins) histograms are calculated with
        nbins bins. hist_range defaults to (0, max_speed) if None

        Scales all the output by multiplying with scaler, defalut output is in um/min if scaler is None

        :returns self.speeds

        """

        if (scaler == None):
            scaler = self.scaler

        try:
            if self.flows == None:
                raise Exception("No flow calculated, please calculate flow first!")

        except ValueError:
            pass

        out = np.square(self.flows)
        out = out.sum(axis=3)
        out = np.sqrt(out) * scaler
        self.speeds = out

        if doHist:

            if (hist_range == None):
                hist_range = (0, out.max())

            print("Histogram range: {}".format(hist_range))
            hists = np.ones((self.flows.shape[0], nbins), dtype=np.float32)

            for i in range(self.flows.shape[0]):
                hist = np.histogram(out[i], bins=nbins, range=hist_range, density=True)
                hists[i] = hist[0]

            bins = hist[1]

            self.histograms = (hists, bins)

        if doAvgSpeed:
            self.avg_speeds = out.mean(axis=(1, 2))

        return self.speeds

    def _draw_flow_frame(self, img, flow, step=15, scale=20, line_thicknes=2):
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

    def draw_all_flow_frames(self, scalebarFlag=False, scalebarLength=10, **kwargs):
        """
        Draws the flow on all the frames in bg with standard settings

        """

        flows = self.flows
        bg = self.channel.getTemporalMedianFilterArray()
        outshape = (flows.shape[0], flows.shape[1], flows.shape[2])
        out = np.empty(outshape, dtype='uint8')
        scale = kwargs["scale"]
        scalebar_px = int(scale * scalebarLength / self.scaler)

        if bg.dtype != np.dtype('uint8'):
            bg = normalization_to_8bit(bg)

        for i in range(out.shape[0]):

            out[i] = self._draw_flow_frame(bg[i], flows[i], **kwargs)
            if scalebarFlag:
                out[i] = self._draw_scalebar(out[i], scalebar_px)

        self.drawnFrames = out

        return out

    def rehapeDrawnFramesTo6d(self):
        # reshapes 3D (t, x, y) array to (t, 1, 1, x, y, 1) for saving dimensions in TZCYXS order

        if (len(self.drawnFrames.shape) == 6):
            return None

        shape = self.drawnFrames.shape
        self.drawnFrames.shape = (shape[0], 1, 1, shape[1], shape[2], 1)

    def saveSpeedArray(self, outdir, fname=None):
        # Saves the speeds as a 32-bit tif
        shape = self.speeds.shape
        self.speeds.shape = (shape[0], 1, 1, shape[1], shape[2], 1)  # dimensions in TZCYXS order

        if fname == None:
            saveme = os.path.join(outdir, self.channel.name + "_speeds.tif")

        else:
            saveme = os.path.join(outdir, fname)

        ij_metadatasave = {'unit': 'um', 'finterval': round(self.channel.finterval_ms / 1000, 2),
                           'tunit': "s", 'frames': shape[0],
                           'slices': 1, 'channels': 1}

        tifffile.imwrite(saveme, self.speeds.astype(np.float32),
                         imagej=True, resolution=(1 / self.channel.pxSize_um, 1 / self.channel.pxSize_um),
                         metadata=ij_metadatasave
                         )

    def saveSpeedCSV(self, outdir):
        # print("Saving csv of mean speeds...")
        if (len(self.speeds.shape) == 6):
            arr = np.average(self.speeds, axis=(3, 4))

        else:
            arr = np.average(self.speeds, axis=(1, 2))

        fr_interval = self.channel.frameSamplingInterval
        arr.shape = arr.shape[0]  # make 1D

        timepoints_abs = np.arange(fr_interval - 1, arr.shape[0] + fr_interval - 1,
                                   dtype='float32') * self.channel.finterval_ms / 1000

        df = pd.DataFrame(arr, index=timepoints_abs, columns=["AVG_frame_flow_" + self.unit])
        df.index.name = "Time(s)"
        saveme = os.path.join(outdir, self.channel.name + "_speeds.csv")
        df.to_csv(saveme)


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

    # TODO everything


class Analysis(object):
    """
    Base object for analysis of Analyzer classes
    """

    def __init__(self, analyzer):
        """

        :param analyzer:
        Analysis object
        """
        self.analyzer = analyzer


class AlignmentIndex(Analysis):
    """
    Calculates the alignment index as defined as in Malinverno et. al 2017.

    """

    def __init__(self, analyzer, returnMagnitudesFlag=False):
        super().__init__(analyzer)
        self.returnMagnitudesFlag = returnMagnitudesFlag

    def alignment_index(self, u, v, alsoReturnMagnitudes=False):
        """
        Returns an array of the same shape as u and v with the alignment index (ai), defined as in Malinverno et. al 2017.
        For every frame the ai is the average of the dot products of the mean velocity vector with each individual
        vector, all divided by the product of their magnitudes.

        If alsoReturnMagnitudes is set to True, then an additional array with the vector magnitudes, i.e, speeds in
        pixels/frame is also returned.

        :param u:
            2D numpy array with u component of velocity vectors
        :param v:
            2D numpy array with v component of velocity vectors
        :param alsoReturnMagnitudes:
            (bool) Should the function also return the vector magnitudes
        :return:
            nunpy array with size=input.size where every entry is the alignment index in that pixel

        """

        assert (u.shape == v.shape) and (len(u.shape) == 2)  # Only single frames are processed

        vector_0 = np.array((np.mean(u), np.mean(v)))
        v0_magnitude = np.linalg.norm(vector_0)

        vector_magnitudes = np.sqrt((np.square(u) + np.square(v)))  # a^2 + b^2 = c^2
        magnitude_products = vector_magnitudes * v0_magnitude
        dot_products = u * vector_0[0] + v * vector_0[1]  # Scalar multiplication followed by array addition

        ai = np.divide(dot_products, magnitude_products)

        if alsoReturnMagnitudes:
            return ai, vector_magnitudes

        else:
            return ai


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
                Ch1.rehapeMedianFramesTo6d()

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
