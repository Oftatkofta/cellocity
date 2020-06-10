from cellocity.channel import Channel, MedianChannel
import cellocity.analysis as analysis
from tifffile import TiffFile
from pathlib import Path
from matplotlib import pyplot as plt
from celluloid import Camera

#Read and write paths
ij_file_path = r"C:\Users\Jens\Desktop\cellocity_logo\cellocity_logo2_noisy.tif"
savepath = Path(r"C:\Users\Jens\Desktop\cellocity_logo")

#open ij file as Tifffile object
tif = TiffFile(ij_file_path)

#create Channel object
my_channel = Channel(0, tif, "noisy logo")
#my_channel.trim(80, 89)

#perform temporal median filter
my_median_channel = MedianChannel(my_channel)

#initialize analyzer objects and define unit
analyzer = analysis.FarenbackAnalyzer(my_channel, "um/s")
median_analyzer = analysis.FarenbackAnalyzer(my_median_channel, "um/s")

#calculate the flows
analyzer.doFarenbackFlow()
median_analyzer.doFarenbackFlow()

#initialize the annalysis objects
regular_analysis = analysis.FlowSpeedAnalysis(analyzer)
median_analysis = analysis.FlowSpeedAnalysis(median_analyzer)


def animateAndSaveLogo(flow_analysis):
    """
    Draws, ainimates, and saves flow vectors

    :param flow_analysis: a FlowAnalysis object
    :return: None

    """

    drawn_frames = flow_analysis.draw_all_flow_frames_superimposed(scalebarFlag=True, scalebarLength=10, scale=10,
                                                                      line_thicknes=1)
    channel_name = flow_analysis.getChannelName()

    fig = plt.figure()
    camera = Camera(fig)
    plt.title("A logo made from " + channel_name)
    plt.style.use('seaborn-dark')
    plt.axis('off')

    for i in range(drawn_frames.shape[0]):
        plt.imshow(drawn_frames[i])
        camera.snap()

    animation = camera.animate()
    file_name = channel_name + ".mp4"
    saveme = savepath / file_name

    animation.save(str(saveme), writer='ffmpeg')

a_list_of_analysis = [regular_analysis, median_analysis]

for a in a_list_of_analysis:
    animateAndSaveLogo(a)
    a.calculateAverageSpeeds()
    a.saveArrayAsTif(savepath)
    a.saveFlowAsTif(savepath)

tif.close()

