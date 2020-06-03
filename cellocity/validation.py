import time, os
from pathlib import Path
import tifffile
from cellocity.channel import Channel, MedianChannel
from cellocity.analysis import FarenbackAnalyzer, OpenPivAnalyzer, FlowSpeedAnalysis
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np


"""
A quick sanity check on the flow analyzer using images of a fixed monolayer
translated 1 um in x, y, or xy between frames. It's not a time lapse stack, so
some massaging of the Channel objects will have to be done 
"""
inpath = Path(r"C:\Users\Jens\Documents\_Microscopy\FrankenScope2\Calibration stuff\DIC_truth")
outpath = Path(r"C:\Users\Jens\Desktop\temp")

def convertChannel(fname, finterval=1):
    """
    Converts a mulitiposition MM file to a fake timelapse Channel with finterval second frame interval.

    :param fname: Path to file
    :param finterval: desired frame interval in output ``Channel``, defaults to 1 second
    :return: Channel
    :rtype: cellocity.Channel
    """

    with tifffile.TiffFile(fname, multifile=False) as tif:
        name = str(fname).split(".")[0]
        name = name.split("\\")[-1][:-8]
        ch = Channel(0, tif, name)
        ch.finterval_ms = finterval * 1000

    return ch

def convertMedianChannel(fname, finterval=1):
    """
    Converts a mulitiposition MM file to a fake timelapse Channel with finterval second frame interval.

    :param fname: Path to file
    :param finterval: desired frame interval in output ``Channel``, defaults to 1 second
    :return: Channel
    :rtype: cellocity.Channel
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


def processAndSave(ch, outpath):
    a1 = FarenbackAnalyzer(ch, "um/s")
    a2 = OpenPivAnalyzer(ch, "um/s")
    a2.doOpenPIV()
    a1.doFarenbackFlow()
    speed1 = FlowSpeedAnalysis(a1)
    speed2 = FlowSpeedAnalysis(a2)
    speed1.calculateAverageSpeeds()
    speed2.calculateAverageSpeeds()
    t1 = str(round(a1.process_time, 2))
    t2 = str(round(a2.process_time, 2))
    speed1.saveSpeedCSV(outpath, fname="FLOW_"+ch.name+"_"+t1+".csv")
    speed2.saveSpeedCSV(outpath, fname="PIV_"+ch.name+"_"+t2+".csv")




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

def run_validation(inpath, outpath):
    ch_list = make_channels(inpath)
    for ch in ch_list:
        processAndSave(ch)


if __name__ == "__Main__":
    df = speedCsvToDataFrame(outpath)
    timeplot = make_proces_time_plot(df)
    speedplot = make_speed_plot(df)


#sns.stripplot(x="analyzer", y="process_time", data=df)
#sns.boxplot(x="analyzer", y="process_time", data=df)


#
#sns.distplot(df.)
#gdf = df.groupby(["analyzer", "filter", "displacement"])
#time_df = gdf["process_time"]
#df.boxplot(column=["process_time"], by="analyzer", grid=False)
#df.boxplot(column='AVG_frame_flow_um/s', by=["analyzer","displacement", "filter"], grid=False)
    #plt.show()