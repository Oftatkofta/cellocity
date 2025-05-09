import napari
from tifffile import TiffFile
import numpy as np
import os
from channel import Channel
from tiffloader import TiffLoader
#from analysis import FlowAnalysis, FarenbackAnalyzer
import tifffile as tif
from magicgui import magicgui


infile = os.path.abspath(r"D:\NT_cropped250410\crop.tif")
flowdir = os.path.abspath(r"D:\NT_cropped250410\flows")
labelfile = os.path.abspath(r"D:\NT_cropped250410\labels\Labels.tif")

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
    files = [os.path.join(indir, f) for f in files if f.endswith("flow.tif")]
    return files

def load_files_and_concatenate(files):
    arrays = [tif.imread(f) for f in files]
    return np.stack(arrays, axis=1)

def load_flow_files_and_concatenate(files):
    arrays = [tif.imread(f) for f in files]
    return np.stack(arrays, axis=0)

def load_flow_file(file, debug=False):
    flow_array = tif.imread(file)
    if debug:
        print(flow_array.shape)

    return flow_array

import numpy as np

def flow_to_napari_vectors(flow, step=50):
    """
    Convert (T, Z, X, Y, 2) flow (u,v) into sparse Napari vectors (N, 2, 4).

    Parameters
    ----------
    flow : np.ndarray, shape (T, Z, X, Y, 2)
        flow[...,0]=u (X‐disp), flow[...,1]=v (Y‐disp)
    step : int
        spatial subsampling step

    Returns
    -------
    vectors : np.ndarray, shape (N, 2, 4)
        each entry is [start, direction] in (t, z, y, x) coords
    """
    T, Z, X, Y, _ = flow.shape
    vecs = []

    for t in range(T):
        for z in range(Z):
            for x in range(0, X, step):
                for y in range(0, Y, step):
                    u, v = flow[t, z, x, y]                  # keep as float
                    start = np.array([t, z, y, x], dtype=float)
                    # displacement in (t, z, y, x) → only y and x move
                    direction = np.array([0, 0, v, u], dtype=float)
                    vecs.append([start, direction])

    return np.stack(vecs, axis=0)



def display_drawn_data():
    #process_file(infile, outdir)
    files = load_processed_files(flowdir)
    flow_array = load_files_and_concatenate(files)
    file_loader = TiffLoader(infile, debug=False)

    dic_array, dic_elapsed_times = file_loader.extract_channel_3d(0)
    draq7_array, draq7_elapsed_times = file_loader.extract_channel_3d(1)
    
    z_scale = file_loader.z_interval_um/file_loader.pixel_size_um


    viewer = napari.Viewer(ndisplay=3, axis_labels=("z", "t"))
    dic_layer = viewer.add_image(dic_array[:-1], name="DIC", colormap="gray", blending="translucent", scale=[1, z_scale, 1, 1])
    #viewer.add_image(data[:,:,1,:,:], name="GFP", colormap="green", blending="translucent", scale=[1, z_scale, 1, 1])
    draq7_layer = viewer.add_image(draq7_array[:-1], name="DRAQ7", colormap="magenta", blending="translucent", scale=[1, z_scale, 1, 1])
    
    flow_layer = viewer.add_image(flow_array, name="Flow", colormap="red", blending="translucent", scale=[1, z_scale, 1, 1])
    # Add channel 1 as 2D image with custom Z control
    # Start with z = 0
    initial_z = 0
    # Extract the initial Z slice with singleton Z-dim: shape (T, 1, Y, X)
    dic_slice_data = dic_array[:-1, initial_z:initial_z+1, :, :]
    vector_slice_data = flow_array[:, initial_z:initial_z+1, :, :]
    # Add the DIC slice layer at its Z position using `translate`
    dic_slice = viewer.add_image(
        dic_slice_data,
        name='DIC (Single Slice)',
        colormap='gray',
        scale=[1, z_scale, 1, 1],  # time, z, y, x
        translate=[0, initial_z * z_scale, 0, 0],
        blending='translucent',
    )
    vector_slice = viewer.add_image(
        vector_slice_data,
        name='Flow (Single Slice)',
        colormap='red',
        scale=[1, z_scale, 1, 1],  # time, z, y, x
        translate=[0, initial_z * z_scale, 0, 0],
        blending='translucent',
    )
    
# Z slider to update the slice and its position
    @magicgui(z={"label": "Slice Z index", "max": dic_array.shape[1] - 1}, auto_call=True)
    def update_z(z: int = 0):
        dic_slice.data = dic_array[:-1, z:z+1, :, :]
        dic_slice.translate = [0, z * z_scale, 0, 0]
        #new_flow_data = flow_array[:, z:z+1, :, :]
        vector_slice.data = flow_array[:, z:z+1, :, :]
        vector_slice.translate = [0, z * z_scale, 0, 0]

    viewer.window.add_dock_widget(update_z, area='right')

    napari.run()
    file_loader.close()

def display_flow_data(infile, flowfile = None):
    if flowfile is None:
        flows = process_file_flow(infile, flowdir, save_flow=False)
    else:
        flows = tif.imread(flowfile)
    
    vectors = flow_to_napari_vectors(flows, step=20)
    print(vectors.shape)

    file_loader = TiffLoader(infile, debug=False)
    dic_array, dic_elapsed_times = file_loader.extract_channel_3d(0)
    dic_array = dic_array[:-1]
    draq7_array, draq7_elapsed_times = file_loader.extract_channel_3d(1)
    draq7_array = draq7_array[:-1]

    
    z_scale = file_loader.z_interval_um/file_loader.pixel_size_um
    #print(z_scale, flows.shape, dic_array.shape)

    viewer = napari.Viewer(ndisplay=3)

    dic_layer = viewer.add_image(dic_array, name="DIC", colormap="gray", blending="translucent", scale=[1, z_scale, 1, 1], visible=False)
    draq7_layer = viewer.add_image(draq7_array, name="DRAQ7", colormap="magenta", blending="translucent", scale=[1, z_scale, 1, 1], contrast_limits=[180, 277])
    #label_layer = viewer.add_labels(labels[0,0,:,:], name="Labels", blending="translucent")
    # Add the vectors layer
    vectors_layer = viewer.add_vectors(
        vectors, 
        name="Flow", 
        edge_width=0.2, 
        edge_color="red",
        length=2.5,
        scale=[1, z_scale, 1, 1]
    )
    dic_slice_data = dic_array[:, 0:1, :, :]
    dic_slice = viewer.add_image(
        dic_slice_data,
        name='DIC (Single Slice)',
        colormap='gray',
        scale=[1, z_scale, 1, 1],  # time, z, y, x
        translate=[0, 0, 0, 0],
        blending='translucent',
        depiction='plane'

    )
    # Z slider to update the slice and its position
    @magicgui(z={"label": "Slice Z index", "max": dic_array.shape[1] - 1}, auto_call=True)
    def update_z(z: int = 0):
        dic_slice.data = dic_array[:, z:z+1, :, :]
        dic_slice.translate = [0, z * z_scale, 0, 0]
    
    viewer.window.add_dock_widget(update_z, area='right')
    
    napari.run()
    file_loader.close()


def display_labels():
    file_loader = TiffLoader(infile, debug=False)
    dic_array, dic_elapsed_times = file_loader.extract_channel_3d(0)
    draq7_array, draq7_elapsed_times = file_loader.extract_channel_3d(1)
    
    z_scale = file_loader.z_interval_um/file_loader.pixel_size_um
    flows = tif.imread(os.path.join(flowdir, "flows.tif"))
    labels = tif.imread(labelfile)
    
    viewer = napari.Viewer(ndisplay=3)
    dic_layer = viewer.add_image(dic_array, name="DIC", colormap="gray", blending="translucent", scale=[1, z_scale, 1, 1])
    labels_layer = viewer.add_labels(labels, name="Labels", blending="translucent", scale=[1, z_scale, 1, 1])
    draq7_layer = viewer.add_image(draq7_array, name="DRAQ7", colormap="magenta", blending="translucent", scale=[1, z_scale, 1, 1])
    napari.run()


if __name__ == "__main__":
    
    #display_labels()
    #display_drawn_data()
    display_flow_data(infile, flowfile=os.path.join(flowdir, "flows.tif"))
    
    #flow_array = load_flow_file(os.path.join(flowdir, "flows.tif"), debug=False)
    #files = load_processed_files(flowdir)
    #flow_array = load_flow_files_and_concatenate(files)










