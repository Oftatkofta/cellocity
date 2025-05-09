from scipy import ndimage as ndi
from napari_animation import Animation
import napari
import skimage
import scipy

viewer = napari.Viewer(ndisplay=3)

nuclei = skimage.data.cells3d()[:,1,...]


animation = Animation(viewer)

image_layer = viewer.add_image(nuclei, name="nuclei", depiction="plane",
                               blending='translucent')


viewer.camera.angles = (-40, 42, 142)
viewer.camera.zoom *= 0.5



image_layer.plane.position = (0, 0, 0)
animation.capture_keyframe(steps=30)

image_layer.plane.position = (59, 0, 0)
animation.capture_keyframe(steps=30)

image_layer.plane.position = (0, 0, 0)

animation.capture_keyframe(steps=30)


animation.animate("layer_planes.mp4", canvas_only=True)
