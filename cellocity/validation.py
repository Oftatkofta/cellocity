from pathlib import Path
import tifffile
from cellocity.channel import Channel, MedianChannel
from cellocity.analysis import FarenbackAnalyzer, OpenPivAnalyzer, FlowSpeedAnalysis, AlignmentIndexAnalysis
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

# This module contains the code used for the "Validaton of the Cellocity software" section of the documentation.
# The code herein performs a sanity check on your Cellocity installation by running a series of test functions
# on all files in the inpath.
# Cellocity has been validated on a set of images of a fixed monolayer translated 1 um in x, y, or x+y between frames.
# It's natively not a time lapse stack data set, so some custom manipulation of the Channel objects will have to be done
# in order to make it appear as though the image stacks come from a time lapse set with a 1 Hz imaging frame rate.

inpath = Path(r"C:\Users\Jens\Documents\_Microscopy\FrankenScope2\Calibration stuff\DIC_truth")
outpath3 = Path(r"C:\Users\Jens\Desktop\temp3")

def convertChannel(fname, finterval=1):
    """
    Converts a mulitiposition MM file to a timelapse Channel with finterval second frame interval.

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
    Converts a mulitiposition ome.tif MM file to a timelapse MedianChannel with finterval second frame interval.

    :param fname: Path to file
    :param finterval: desired frame interval in output ``Channel``, defaults to 1 second
    :return: MedianChannel with default 3-frame gliding window
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
            out.append(chan)
            m_chan = convertMedianChannel(f)
            out.append(m_chan)

    return out

def processAndMakeDf(ch_list):
    """
    Creates analyzers from and runs test functions on a list of Channels.


    :param ch_list: List of Channel objects
    :type ch_list: list
    :param outpath: Path where to save output
    :type outpath: pathlib.Path
    :return: Pandas DataFrame with data from analysis
    :rtype: pandas.DataFrame
    """

    alldata = pd.DataFrame()

    for ch in ch_list:
        a1 = make_fb_flow_analyzer(ch)
        alldata = alldata.append(get_data_as_df(a1, "optical_flow"))
        a2 = make_piv_analyzer(ch)
        alldata = alldata.append(get_data_as_df(a2, "PIV"))


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
    sns_plot = sns.catplot(x="analyzer", y="AVG_speed_um/s",
                    hue="displacement", col="filter",
                    data=df, kind="box",
                    height=8, aspect=.7)

    return sns_plot

def make_ai_plot(df):
    """
    Generates a plot comparing average frame flow speeds from dataframe

    """
    sns_plot = sns.catplot(x="analyzer", y="aligmnent_index",
                    hue="displacement", col="filter",
                    data=df, kind="box",
                    height=8, aspect=.7)

    return sns_plot


def make_fb_flow_analyzer(ch):
    """
    Creates a FarenbackAnalyzer and performs optical flow calculations with default settings in um/s.

    :param ch: channel.Channel
    :return: anlysis.FarenbackAnalyzer
    """
    analyzer = FarenbackAnalyzer(ch, "um/s")
    analyzer.doFarenbackFlow()

    return analyzer

def make_piv_analyzer(ch):
    """
    Creates an openPivAnalyzer and performs optical flow calculations with default settings in um/s.

    :param ch: channel.Channel
    :return: anlysis.OpenPivAnalyzer

    """
    analyzer = OpenPivAnalyzer(ch, "um/s")
    analyzer.doOpenPIV()

    return analyzer

def get_data_as_df(analyzer, analyzername):
    """
    Creates FlowSpeedAnalysis() and AlignmentIndexAnalysis() from a FlowAnalyzer.

    Calculates average frame speeds and alignment indexes and returns a DataFrame with the results.


    :param analyzer: FlowAnalyzer
    :type analyzer: analysis.FlowAnalyzer
    :param analyzername: Name of FlowAnalyzer
    :type analyzername: str

    :return: pd.DataFrame containing results and information derived from channel.name
    :rtype: pandas.DataFrame
    """

    speed_analysis = FlowSpeedAnalysis(analyzer)
    speed_analysis.calculateAverageSpeeds()
    ai_analysis = AlignmentIndexAnalysis(analyzer)
    ai_analysis.calculateAverage()

    df = speed_analysis.getAvgSpeedsAsDf()
    df["aligmnent_index"] = ai_analysis.getAvgAlignIdxAsDf()

    df["analyzer"] = analyzername
    df["process_time"] = str(round(analyzer.process_time, 2))
    df["file_name"] = analyzer.channel.name

    fields = analyzer.channel.name.split("_")
    magnification = fields[3]
    displacemet = fields[4] + " " + fields[5]
    df["magnification"] = magnification
    df["displacement"] = displacemet

    if "MED" in analyzer.channel.name:
        filter = "Median"
    else:
        filter = "None"
    df["filter"] = filter

    return df


if __name__ == "__main__":
    finterval = 1
    kvargs = {'step': 60, 'scale': 10, 'line_thicknes': 2}
    saveme = outpath3 / "alldata.csv"
    ch_list = make_channels(inpath)
    df = processAndMakeDf(ch_list)
    df.to_csv(saveme)
    df = pd.read_csv(saveme)

    timeplot = make_proces_time_plot(df)
    plt.show()

    speed_plot = make_speed_plot(df)
    plt.show()

    ai_plot = make_ai_plot(df)
    plt.show()


