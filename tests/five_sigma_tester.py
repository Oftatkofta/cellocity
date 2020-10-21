from pathlib import Path
import tifffile
from cellocity.channel import Channel, MedianChannel
from cellocity.analysis import FarenbackAnalyzer, OpenPivAnalyzer, FiveSigmaAnalysis
from cellocity.validation import make_channels, make_fb_flow_analyzer, make_piv_analyzer
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns


def combine_lcorr_and_process_time_to_df(lcorrdf, processtimedict, file_name, analyzer_name):

    df1 = pd.DataFrame.from_dict(processtimedict, orient='index', columns=["process_time"])
    df1.sort_index(inplace=True)
    lcorrdf["process_time"] = df1["process_time"]
    lcorrdf["file_name"] = file_name
    lcorrdf["analyzer"] = analyzer_name

    fields = file_name.split("_")
    magnification = fields[3]
    displacemet = fields[4] + " " + fields[5]
    df["magnification"] = magnification
    df["displacement"] = displacemet

    if "MED" in file_name:
        filter_name = "Median"
    else:
        filter_name = "None"

    lcorrdf["filter"] = filter_name

    return lcorrdf



def main():

    inpath = Path(r"C:\Users\Jens\Desktop\S-BSST461")
    outpath = Path(r"C:\Users\Jens\Desktop\temp")
    ch_list = make_channels(inpath)
    alldata = pd.DataFrame()
    saveme = outpath / "5sigma.csv"
    for ch in ch_list:
        print(ch.name)

        a1 = make_fb_flow_analyzer(ch)
        fsig = FiveSigmaAnalysis(a1)
        fsig.calculateCorrelationAllFrames()
        lcorrdf = fsig.getCorrelationLengthsAsDf()
        processtimedict = fsig._process_times
        alldata = alldata.append(combine_lcorr_and_process_time_to_df(lcorrdf,
                                                                      processtimedict,
                                                                      ch.name,
                                                                      "optical_flow"
                                                                      ))
        a2 = make_piv_analyzer(ch)
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

if __name__ == "__main__":
    main()
