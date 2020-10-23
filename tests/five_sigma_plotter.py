from pathlib import Path

from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns



def make_lcorr_plot(df):
    """
    Generates a plot comparing correlation lengths between analyzers and magnifications

    """
    sns_plot = sns.catplot(x="magnification", y="fraction_of_max_lcorr",
                           hue="filter", col = "analyzer",
                           data=df, kind="box",
                           height=8, aspect=.7, )
    sns_plot.set_axis_labels("Magnification", "Fraction of theoretical maximum correlation length")


    return sns_plot

def make_proces_time_plot(df):
    """
    Generates a bar plot comparing processing times for the two analyzers.

    """
    sns_plot = sns.catplot(x="analyzer", y="process_time",
                    data=df, kind="box",
                    height=8, aspect=.7)

    sns_plot.set_axis_labels("Analyzer", "Process time (s)")

    return sns_plot


inpath = Path(r"C:\Users\Jens\Desktop\temp")
openme = inpath / "5sigma.csv"
alldata = pd.read_csv(openme)
alldata = alldata.assign(
    filter = lambda dataframe: dataframe['file_name'].map(lambda fname: "Median" if "MED" in fname else "None"))
alldata = alldata.assign(
    displacement = lambda dataframe: dataframe['file_name'].map(lambda fname: fname.split("_")[4] + " " + fname.split("_")[5]))
alldata = alldata.assign(
    magnification = lambda dataframe: dataframe['file_name'].map(lambda fname: fname.split("_")[3]))
max_lcorrs = {
    "10X": 1805.46,
    "15X": 1203.64,
    "40X": 459.60,
    "60Xopt": 306.94,
    "60X": 200.88}

alldata = alldata.assign(
    max_lcorr = lambda dataframe: dataframe['magnification'].map(lambda max: max_lcorrs.get(max)))
alldata = alldata.assign(
    fraction_of_max_lcorr = lambda df: df['Cvv_um']/df['max_lcorr']
)

saveme = inpath / "5sigma_.csv"
alldata.to_csv(saveme)
print(alldata)

timeplot = make_proces_time_plot(alldata)
savename = inpath / "5sigma_process_time_compare.png"
plt.savefig(savename)
plt.show()

lcorr_plot = make_lcorr_plot(alldata)
savename = inpath / "5sigma_lcorr_compare.png"
plt.savefig(savename)

plt.show()