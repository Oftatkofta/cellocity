from pathlib import Path
import tifffile
from cellocity.channel import Channel, MedianChannel
from cellocity.analysis import FarenbackAnalyzer, OpenPivAnalyzer, FlowSpeedAnalysis, AlignmentIndexAnalysis
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

#A quick sanity check on the flow analyzer using images of a fixed monolayer
#translated 1 um in x, y, or x+y between frames. It's not a time lapse stack, so
#some custom manipulation of the Channel objects will have to be done.

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


if __name__ == "__Main__":
    finterval = 1
    kvargs = {'step': 60, 'scale': 10, 'line_thicknes': 2}
    saveme = outpath3 / "alldata.csv"
    ch_list = make_channels(inpath)
    df = processAndMakeDf(ch_list)
    df.to_csv(saveme)
    df = pd.read_csv(r"C:\Users\Jens\Desktop\temp3\alldata.csv")
    print(df.columns)
    timeplot = make_proces_time_plot(df)
    plt.show()
    # plt.savefig(avg_speed_compare.png)
    speedplot = make_speed_plot(df)
    plt.show()
    aiplot = make_ai_plot(df)
    plt.show()


