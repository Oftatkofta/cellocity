from pathlib import Path
import tifffile
from cellocity.channel import Channel, MedianChannel
from cellocity.analysis import FarenbackAnalyzer, OpenPivAnalyzer, FiveSigmaAnalysis
from cellocity.validation import make_channels, make_fb_flow_analyzer, make_piv_analyzer
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns


def combine_lcorr_and_process_time_to_df(lcorrdf, processtimedict, file_name, analyzer_name):
    """
    Performs combination and mutation of correlation length dataframe to simplify visualization and plotting
    Used by 5sigma_validation.
    """

    df1 = pd.DataFrame.from_dict(processtimedict, orient='index', columns=["process_time"])
    df1.sort_index(inplace=True)
    lcorrdf["process_time"] = df1["process_time"]
    lcorrdf["file_name"] = file_name
    lcorrdf["analyzer"] = analyzer_name

    fields = file_name.split("_")
    magnification = fields[3]
    displacemet = fields[4] + " " + fields[5]
    lcorrdf["magnification"] = magnification
    lcorrdf["displacement"] = displacemet

    if "MED" in file_name:
        filter_name = "Median"
    else:
        filter_name = "None"

    lcorrdf["filter"] = filter_name

    #max_lcorrs are given by pixel size * pixels per image column -1 (1607 for the test data)
    max_lcorrs = {
        "10X": 1805.46,
        "15X": 1203.64,
        "40X": 459.60,
        "60Xopt": 306.94,
        "60X": 200.88}

    lcorrdf = lcorrdf.assign(
        max_lcorr=lambda dataframe: dataframe['magnification'].map(lambda max: max_lcorrs.get(max)))
    lcorrdf = lcorrdf.assign(
        fraction_of_max_lcorr=lambda df: df['Cvv_um'] / df['max_lcorr'])

    return lcorrdf

def make_lcorr_plot(lcorrdf):
    """
    Generates a plot comparing correlation lengths between analyzers and magnifications

    """
    sns_plot = sns.catplot(x="magnification", y="fraction_of_max_lcorr",
                           hue="filter", col = "analyzer",
                           data=lcorrdf, kind="box",
                           height=8, aspect=.7, )
    sns_plot.set_axis_labels("Magnification", "Fraction of theoretical maximum correlation length")


    return sns_plot

def make_proces_time_plot(lcorrdf):
    """
    Generates a bar plot comparing processing times for the two analyzers.

    """
    sns_plot = sns.catplot(x="analyzer", y="process_time",
                    data=lcorrdf, kind="box",
                    height=8, aspect=.7)

    sns_plot.set_axis_labels("Analyzer", "Process time (s)")

    return sns_plot

def run_5sigma_validation(inpath, outpath):
    print("Creating Channel objects...")
    ch_list = make_channels(inpath)
    alldata = pd.DataFrame()
    saveme = outpath / "5sigma_analysis.csv"
    for ch in ch_list:
        print("Farenback flow analyzer starting on: ", ch.name)
        a1 = make_fb_flow_analyzer(ch)
        print("5-sigma analysis starting on: ", ch.name)
        fsig = FiveSigmaAnalysis(a1)
        fsig.calculateCorrelationAllFrames()
        lcorrdf = fsig.getCorrelationLengthsAsDf()
        processtimedict = fsig._process_times
        alldata = alldata.append(combine_lcorr_and_process_time_to_df(lcorrdf,
                                                                      processtimedict,
                                                                      ch.name,
                                                                      "optical_flow"
                                                                      ))
        print("PIV analyzer starting on: ", ch.name)
        a2 = make_piv_analyzer(ch)
        print("5-sigma analysis starting on: ", ch.name)
        fsig = FiveSigmaAnalysis(a2)
        fsig.calculateCorrelationAllFrames()
        lcorrdf = fsig.getCorrelationLengthsAsDf()
        processtimedict = fsig._process_times
        alldata = alldata.append(combine_lcorr_and_process_time_to_df(lcorrdf,
                                                                      processtimedict,
                                                                      ch.name,
                                                                      "PIV"
                                                                      ))

    alldata.to_csv(saveme)

    return alldata


