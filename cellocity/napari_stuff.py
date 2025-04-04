import napari
from tifffile import TiffFile
import numpy as np
import os
from magicgui import magicgui
import scipy
import skimage

infile = os.path.abspath(r"D:\HujejX1_ODMd1_MSS109_75uM-DRAQ7_start14.01_1_1_MMStack_Pos0.ome.tif")


with TiffFile(infile) as tif:
    data = tif.asarray()
    mm_metadata = tif.micromanager_metadata
    page_0 = tif.pages[0]
    z_scale = abs(float(mm_metadata['Summary']['z-step_um'])/page_0.tags['MicroManagerMetadata'].value['PixelSizeUm'])
    print(z_scale)
    dic_channel = data[:,:,0,:,:]
    draq7_channel = data[:,:,2,:,:]


viewer = napari.Viewer(ndisplay=3, axis_labels=("z", "t"))
#dic_layer = viewer.add_image(dic_channel, name="DIC", colormap="gray", blending="translucent", scale=[1, z_scale, 1, 1])
#viewer.add_image(data[:,:,1,:,:], name="GFP", colormap="green", blending="translucent", scale=[1, z_scale, 1, 1])
draq7_layer = viewer.add_image(draq7_channel, name="DRAQ7", colormap="magenta", blending="translucent", scale=[1, z_scale, 1, 1])
# Add channel 1 as 2D image with custom Z control
# Start with z = 0
initial_z = 0
# Extract the initial Z slice with singleton Z-dim: shape (T, 1, Y, X)
dic_slice_data = dic_channel[:, initial_z:initial_z+1, :, :]

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
@magicgui(z={"label": "DIC Z index", "max": dic_channel.shape[1] - 1}, auto_call=True)
def update_z(z: int = 0):
    new_data = dic_channel[:, z:z+1, :, :]
    dic_slice.data = new_data
    dic_slice.translate = [0, z * z_scale, 0, 0]

viewer.window.add_dock_widget(update_z, area='right')

napari.run()

