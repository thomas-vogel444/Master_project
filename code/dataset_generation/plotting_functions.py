import numpy as np
from matplotlib import pyplot

def plot_dicom_image(image_3d, z):
	h, w, d = image_3d.shape
	x_grid = np.arange(h)
	y_grid = np.arange(w)
	pyplot.figure(dpi=80)
	pyplot.axes().set_aspect('equal', 'datalim')
	pyplot.set_cmap(pyplot.gray())
	pyplot.pcolormesh(x_grid, y_grid, image_3d[:,:,z])
	pyplot.show()

def plot_2d_image(image):
	h, w   = image.shape
	x_grid = np.arange(h)
	y_grid = np.arange(w)
	pyplot.figure(dpi=80)
	pyplot.axes().set_aspect('equal', 'datalim')
	pyplot.set_cmap(pyplot.gray())
	pyplot.pcolormesh(x_grid, y_grid, np.transpose(image))
	pyplot.show()