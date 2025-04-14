import napari
from tifffile import TiffFile
import numpy as np
import os
import cv2 as cv
from cellocity.channel import Channel
from cellocity.tiffloader import TiffLoader
from cellocity.analysis import FlowAnalysis, FarenbackAnalyzer
import tifffile as tif
from magicgui import magicgui
import matplotlib.cm as cm  # For colormap

infile = os.path.abspath(r"D:\NT_cropped250410\crop.tif")

flowdir = os.path.abspath(r"D:\NT_cropped250410\flows")

kwargs = {"scale": 1, "step": 40, "line_thicknes": 2}

def process_file_draw(infile, outdir):
    file_loader = TiffLoader(infile, debug=False)
    ch_idx = 0
    for z in range(file_loader.n_slices):
        channel = Channel(ch_idx, file_loader, z)
        #print(channel.getArray().shape, channel.getElapsedTimes_ms())
        analyzer = FarenbackAnalyzer(channel,  "um/s")
        analysis = FlowAnalysis(analyzer)
        analysis.draw_all_flow_frames(scalebarFlag=True, scalebarLength=10, **kwargs)
        analysis.saveFlowAsTif(os.path.join(outdir, str(z)+".tif"), **kwargs)
    file_loader.close()
    return None

def process_file_flow(infile, outdir, save_flow=False):
    file_loader = TiffLoader(infile, debug=False)
    ch_idx = 0
    flows = []
    for z in range(file_loader.n_slices):
        channel = Channel(ch_idx, file_loader, z)
        #print(channel.getArray().shape, channel.getElapsedTimes_ms())
        analyzer = FarenbackAnalyzer(channel,  "um/s")
        flows.append(analyzer.flows)
    flows = np.stack(flows, axis=0)
    if save_flow:
        tif.imwrite(os.path.join(outdir, "flows.tif"), flows)
    file_loader.close()
    
    return flows



def load_processed_files(indir):
    files = os.listdir(indir)
    files = [os.path.join(indir, f) for f in files if f.endswith(".tif")]
    return files

def load_files_and_concatenate(files):
    arrays = [tif.imread(f) for f in files]
    return np.stack(arrays, axis=1)

def load_flow_files_and_concatenate(files):
    arrays = [tif.imread(f) for f in files]
    return np.stack(arrays, axis=0)

def subsample_flow_array(flow_array, t, z, sample_interval=50):
    """
    Subsample the flow array to the desired time and z slice.
    Reformats the output to Napari-frendly (N,2,D) format.
    sample_interval is the number of pixels between each vector.
    """
    sub_arr = flow_array[t, z, :, :, :]
    idx = np.linspace(0, sub_arr.shape[0]-1, sample_interval, dtype='int')
    xx = np.meshgrid(idx, idx)
    sub_arr = sub_arr[xx,xx]
    start_positions = np.stack(xx,xx, axis=1)
    end_positions = start_positions + sub_arr
    n_vectors = start_positions.shape[0] * start_positions.shape[1]
    start_flat = start_positions.reshape(n_vectors, 2)
    end_flat = end_positions.reshape(n_vectors, 2)
    vectors = np.stack((start_flat, end_flat), axis=1) 
    
    return vectors




def display_drawn_data():
    #process_file(infile, outdir)
    #files = load_processed_files(outdir)
    #flow_array = load_files_and_concatenate(files)
    file_loader = TiffLoader(infile, debug=True)

    dic_array, dic_elapsed_times = file_loader.extract_channel_3d(0)
    draq7_array, draq7_elapsed_times = file_loader.extract_channel_3d(1)
    
    z_scale = file_loader.z_interval_um/file_loader.pixel_size_um


    viewer = napari.Viewer(ndisplay=3, axis_labels=("z", "t"))
    dic_layer = viewer.add_image(dic_array[:-1], name="DIC", colormap="gray", blending="translucent", scale=[1, z_scale, 1, 1])
    #viewer.add_image(data[:,:,1,:,:], name="GFP", colormap="green", blending="translucent", scale=[1, z_scale, 1, 1])
    draq7_layer = viewer.add_image(draq7_array[:-1], name="DRAQ7", colormap="magenta", blending="translucent", scale=[1, z_scale, 1, 1])
    
    #flow_layer = viewer.add_image(flow_array, name="Flow", colormap="red", blending="translucent", scale=[1, z_scale, 1, 1])
    # Add channel 1 as 2D image with custom Z control
    # Start with z = 0
    initial_z = 0
    # Extract the initial Z slice with singleton Z-dim: shape (T, 1, Y, X)
    dic_slice_data = dic_array[:-1, initial_z:initial_z+1, :, :]
    #vector_slice_data = flow_array[:, initial_z:initial_z+1, :, :]
    # Add the DIC slice layer at its Z position using `translate`
    dic_slice = viewer.add_image(
        dic_slice_data,
        name='DIC (Single Slice)',
        colormap='gray',
        scale=[1, z_scale, 1, 1],  # time, z, y, x
        translate=[0, initial_z * z_scale, 0, 0],
        blending='translucent',
    )
    
# Z slider to update the slice and its position
    @magicgui(z={"label": "Slice Z index", "max": dic_array.shape[1] - 1}, auto_call=True)
    def update_z(z: int = 0):
        new_data = dic_array[:-1, z:z+1, :, :]
        dic_slice.data = new_data
        dic_slice.translate = [0, z * z_scale, 0, 0]
        #new_flow_data = flow_array[:, z:z+1, :, :]
        #flow_slice.data = new_flow_data
        #flow_slice.translate = [0, z * z_scale, 0, 0]

    viewer.window.add_dock_widget(update_z, area='right')

    napari.run()
    file_loader.close()

def display_flow_data():
    flows = process_file_flow(infile, flowdir, save_flow=False)
    
if __name__ == "__main__":
    flows = process_file_flow(infile, flowdir, save_flow=False)
    print(flows.shape)
    display_data()
    #files = load_processed_files(flowdir)
    #flow_array = load_flow_files_and_concatenate(files)
    #print(flow_array.shape)












