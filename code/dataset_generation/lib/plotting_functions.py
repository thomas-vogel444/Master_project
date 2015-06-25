import numpy as np
from matplotlib import pyplot

def generate_grids(h, w):
	x_grid = np.arange(h)
	y_grid = np.arange(w)
	return x_grid, y_grid

def plot_2d_image(fig, image_2d, show=True):
	x_grid, y_grid = generate_grids(image_2d.shape[0], image_2d.shape[1])
	# pyplot.figure(dpi=80)
	pyplot.axes().set_aspect('equal', 'datalim')
	pyplot.set_cmap(pyplot.gray())
	pyplot.pcolormesh(x_grid, y_grid, np.transpose(image_2d))
	if show == True: 
		pyplot.show()

def plot_dicom_image(fig, image_3d, z, show=True):
	plot_2d_image(fig, image_3d[:,:,z], show=False)
	if show == True: 
		pyplot.show()

def plot_nrrd_image(fig, image_3d, z, show=True):
	plot_2d_image(np.transpose(fig, image_3d[:,:,z]), show=False)
	if show == True: 
		pyplot.show()


# def dataset_example_plot(x, y, z, image_3d, patches):
# 	fig = plt.figure()
# 	a   = fig.add_subplot(1,2,1)  

