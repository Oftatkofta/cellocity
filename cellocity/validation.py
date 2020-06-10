import time, os
from pathlib import Path
import tifffile
from cellocity.channel import Channel, MedianChannel
from cellocity.analysis import FarenbackAnalyzer, OpenPivAnalyzer, FlowSpeedAnalysis, AlignmentIndexAnalysis
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import cv2 as cv


"""
A quick sanity check on the flow analyzer using images of a fixed monolayer
translated 1 um in x, y, or xy between frames. It's not a time lapse stack, so
some massaging of the Channel objects will have to be done 
"""
inpath = Path(r"C:\Users\Jens\Documents\_Microscopy\FrankenScope2\Calibration stuff\DIC_truth")
outpath = Path(r"C:\Users\Jens\Desktop\temp")
outpath2 = Path(r"C:\Users\Jens\Desktop\temp2")
outpath3 = Path(r"C:\Users\Jens\Desktop\temp3")

def convertChannel(fname, finterval=1):
    """
    Converts a mulitiposition MM file to a fake timelapse Channel with finterval second frame interval.

    :param fname: Path to file
    :param finterval: desired frame interval in output ``Channel``, defaults to 1 second
    :return: Channel
    :rtype: channel.Channel
    """

    with tifffile.TiffFile(fname, multifile=False) as tif:
        name = str(fname).split(".")[0]
        name = name.split("\\")[-1][:-8]
        ch = Channel(0, tif, name)
        ch.finterval_ms = finterval * 1000

    return ch

def convertMedianChannel(fname, finterval=1):
    """
    Converts a mulitiposition ome.tif MM file to a fake timelapse MedianChannel with finterval second frame interval.

    :param fname: Path to file
    :param finterval: desired frame interval in output ``Channel``, defaults to 1 second
    :return: MedianChannel
    :rtype: channel.MedianChannel
    """

    with tifffile.TiffFile(fname, multifile=False) as tif:
        name = str(fname).split(".")[0]
        name = name.split("\\")[-1][:-8]
        ch = Channel(0, tif, name)
        ch.finterval_ms = finterval * 1000
        ch = MedianChannel(ch)

    return ch

def make_channels(inpath):
    """
    Creates a list of Channel objects from files in inPath.

    :param inpath: Path
    :return: list of Channels
    :rtype: list
    """

    out=[]

    for f in inpath.iterdir():
        if (f.suffix == ".tif") and f.is_file():
            chan = convertChannel(f)
            #TODO remove trim
            #chan.trim(0,3)
            out.append(chan)
            m_chan = convertMedianChannel(f, 1)
            out.append(m_chan)

    return out


def processAndSave(ch, outpath, **kwargs):
    a1 = FarenbackAnalyzer(ch, "um/s")
    a1.doFarenbackFlow()

    t1 = str(round(a1.process_time, 2))
    speed1 = FlowSpeedAnalysis(a1)
    speed1.calculateAverageSpeeds()
    speed1.saveCSV(outpath, fname="FLOW_" + ch.name + "_" + t1 + ".csv")
    speed1.saveArrayAsTif(outpath, fname="FLOW_" + ch.name + "_speeds.tif")
    speed1.calculateHistograms()
    speed1.draw_all_flow_frames_superimposed(scalebarFlag=True, scalebarLength=1, **kwargs)
    speed1.saveFlowAsTif(outpath)
    ai1 = AlignmentIndexAnalysis(a1)
    ai1.saveCSV(outpath, fname="FLOW_" + ch.name + "_ai.csv")

    a2 = OpenPivAnalyzer(ch, "um/s")
    a2.doOpenPIV()
    t2 = str(round(a2.process_time, 2))
    speed2 = FlowSpeedAnalysis(a2)
    speed2.calculateAverageSpeeds()
    speed2.saveCSV(outpath, fname="PIV_"+ch.name+"_"+t2+".csv")
    speed2.saveArrayAsTif(outpath, fname="PIV_" + ch.name + "_speeds.tif")
    speed1.calculateHistograms()
    speed2.draw_all_flow_frames_superimposed(scalebarFlag=True, scalebarLength=1, **kwargs)
    speed2.saveFlowAsTif(outpath)
    ai2 = AlignmentIndexAnalysis(a2)
    ai2.saveCSV(outpath, fname="PIV_" + ch.name + "_ai.csv")


def speedCsvToDataFrame(inpath):
    """
    Generates and returns a dataframe from FlowSpeedAnalysis CSVs in inpath.
    """
    alldata = pd.DataFrame()

    for f in inpath.iterdir():

        if (f.suffix == ".csv") and f.is_file():
            df = pd.read_csv(f, index_col=0)
            fields = f.name.split("_")
            analyzer = fields[0]
            process_time = float(fields[-1][:-4])
            magnification = fields[4]
            displacemet = fields[5] + " " +fields[6]

            if "MED" in f.name:
                filter = "Median"
            else:
                filter = "None"

            df["analyzer"]=analyzer
            df["filter"]=filter
            df["magnification"]=magnification
            df["process_time"]=process_time
            df["displacement"]=displacemet
            df["filename"]=f.name

            alldata = alldata.append(df)

    return alldata

def toDataFrame(ch, df):
    """
    Generates and returns a dataframe from FlowSpeedAnalysis CSVs in inpath.
    """
    alldata = pd.DataFrame()

    for f in inpath.iterdir():

        if (f.suffix == ".csv") and f.is_file():
            df = pd.read_csv(f, index_col=0)
            fields = f.name.split("_")
            analyzer = fields[0]
            process_time = float(fields[-1][:-4])
            magnification = fields[4]
            displacemet = fields[5] + " " +fields[6]

            if "MED" in f.name:
                filter = "Median"
            else:
                filter = "None"

            df["analyzer"]=analyzer
            df["filter"]=filter
            df["magnification"]=magnification
            df["process_time"]=process_time
            df["displacement"]=displacemet
            df["filename"]=f.name

            alldata = alldata.append(df)

    return alldata


def make_proces_time_plot(df):
    """
    Generates a bar plot comparing processing times for the two analyzers.

    """
    sns_plot = sns.catplot(x="analyzer", y="process_time",
                    hue="filter",
                    data=df, kind="violin",
                    height=8, aspect=.7)


    return sns_plot

def make_speed_plot(df):
    """
    Generates a plot comparing average frame flow speeds from dataframe

    """
    sns_plot = sns.catplot(x="analyzer", y="AVG_frame_flow_um/s",
                    hue="displacement", col="filter",
                    data=df, kind="box",
                    height=8, aspect=.7)
    return sns_plot

def test_flow(ch):
    """
    Creates a FarenbackAnalyzer and performs optical flow calculations with default settings in um/s.

    :param ch: channel.Channel
    :return: anlysis.FarenbackAnalyzer
    """
    analyzer = FarenbackAnalyzer(ch, "um/s")
    analyzer.doFarenbackFlow()

    return analyzer

def test_piv(ch):
    """
    Creates an openPivAnalyzer and performs optical flow calculations with default settings in um/s.

    :param ch: channel.Channel
    :return: anlysis.OpenPivAnalyzer

    """
    analyzer = OpenPivAnalyzer(ch, "um/s")
    analyzer.doOpenPIV()

    return analyzer

def get_ai_as_df(analyzer):

    ai = AlignmentIndexAnalysis(analyzer)
    ai.calculateAlignIdxs()

    return ai.getAvgAlignIdxAsDf()

def get_speed_as_df(analyzer):

     = AlignmentIndexAnalysis(analyzer)
    ai.calculateAlignIdxs()

    return ai.getAvgAlignIdxAsDf()

def run_validation(inpath, outpath, **kvargs):
    ch_list = make_channels(inpath)
    for ch in ch_list:
        processAndSave(ch, outpath, **kvargs)
        #test_ai(ch, outpath)

if __name__ == "__Main__":
    finterval = 1
    kvargs = {'step': 60, 'scale': 10, 'line_thicknes': 2}
    run_validation(inpath, outpath)
    df = speedCsvToDataFrame(outpath)
    timeplot = make_proces_time_plot(df)
    speedplot = make_speed_plot(df)

kvargs = {'step': 32, 'scale': 10, 'line_thicknes': 2}
run_validation(inpath, outpath, **kvargs)
#fname = r"C:\Users\Jens\Documents\_Microscopy\FrankenScope2\Calibration stuff\DIC_truth\fixed_monolayer_DIC_40X_dX-1um_dY-1um_1_MMStack.ome.tif"
#testchannel = convertChannel(fname, 1)
#testchannel.trim(0,4)
test_ai(testchannel, outpath3)