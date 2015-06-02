import dicom
import h5py
import os
import re
import pprint as pp
import numpy as np

# Canonical path names for use by the various functions
data_directory = "../../ct_atrium"
CT_scan_path = "../../ct_atrium/CTScan_name"
DICOM_path = "../../ct_atrium/CTScan_name/DICOMS/DICOM_name"

#************************************************************
# For each CT scan folder get all the DICOM names
#************************************************************
def get_DICOM_names(CT_scan_directory):
	"""
		Get the DICOM file names following a given pattern, i.e. a name with 8 digits in it.
	"""
	DICOM_directory = os.path.join(CT_scan_directory, "DICOMS")
	DICOM_name_pattern = re.compile("^[0-9]{8}")
	DICOM_names = []
	for root, directory, files in os.walk(DICOM_directory):
		DICOM_names = [myfile for myfile in files if DICOM_name_pattern.match(myfile)]

	return DICOM_names

#*****************************************************************************************
# For a given dicom file, produce a 4D numpy array Batch Channel Width Height
# with:
#	Batch   : number of training examples, i.e. number of voxels in the image 480 * 480
#	Channel : 3 planes centred at the voxel
#	Width   : width of the image path, i.e. 32
#	Height  : height of the image patch, i.e. 32
#*****************************************************************************************
def get_CT_scan_array(CT_scan_name, CT_scan_dicom_filenames, dicom_height = 480, dicom_width = 480):	
	"""
		Get the 3D image from a given CT scan into a numpy array.
	"""
	# Get the dimensions of a given DICOM file
	dicom_file_path  = DICOM_path.replace("CTScan_name", CT_scan_name).replace("DICOM_name", CT_scan_dicom_filenames[0])
	ref 			 = dicom.read_file(dicom_file_path)
	DICOM_dimensions = (int(ref.Rows), int(ref.Columns), len(CT_scan_dicom_filenames))

	# Loop through all the DICOM files for a given CT scan and get all the values into a single 3D numpy array
	CT_scan_array = np.zeros(DICOM_dimensions, dtype="uint16")
	for dicom_filename in CT_scan_dicom_filenames:
	    # read the file
	    dicom_file_path = DICOM_path.replace("CTScan_name", CT_scan_name).replace("DICOM_name", dicom_filename)
	    ds = dicom.read_file(dicom_file_path)
	    # store the raw image data
	    CT_scan_array[:, :, CT_scan_dicom_filenames.index(dicom_filename)] = ds.pixel_array

	return CT_scan_array

#*****************************************************************************************
#									GENERATING PATCHES
#*****************************************************************************************
def tri_planar_patch_generator(x,y,z,image_3d,patch_size):
	""" 
		Generates a 3 * patch_size * patch_size numpy matrix centred at (x,y,z) containing the 
		three perpendicular patches from the 3D tensor image. 
	"""
	height, width, depth = image_3d.shape
	patches = np.zeros((3, patch_size, patch_size))

	x_min = 0
	x_max = patch_size
	y_min = 0
	y_max = patch_size
	z_min = 0
	z_max = patch_size

	if x < patch_size/2:
		x_min = patch_size/2 - x

	if x > height - patch_size/2:
		x_max = patch_size/2 + height - x

	if y < patch_size/2:
		y_min = patch_size/2 - y

	if y > width - patch_size/2:
		y_max = patch_size/2 + width - y

	if z < patch_size/2:
		z_min = patch_size/2 - z

	if z > depth - patch_size/2:
		z_max = patch_size/2 + depth - z

	patches[0, x_min:x_max, y_min:y_max] = image_3d[np.maximum(x-patch_size/2, 0):np.minimum(x+patch_size/2, height), 
						 np.maximum(y-patch_size/2, 0):np.minimum(y+patch_size/2, width), 
						 z]

	patches[1, x_min:x_max, z_min:z_max] = image_3d[np.maximum(x-patch_size/2, 0):np.minimum(x+patch_size/2, height), 
						 y, 
						 np.maximum(z-patch_size/2, 0):np.minimum(z+patch_size/2, depth)]

	patches[2, y_min:y_max, z_min:z_max] = image_3d[x, 
						 np.maximum(y-patch_size/2, 0):np.minimum(y+patch_size/2, width), 
						 np.maximum(z-patch_size/2, 0):np.minimum(z+patch_size/2, depth)]

	return patches

#************************************************************************************************************
# 					I NEED TO START PLOTTING STUFF AND UNDERSTAND MATPLOTLIB PROPERLY!!!
#************************************************************************************************************
# Testing for an edge voxel
# patch_size = 100
# x = 200
# y = 200
# z = 0

# patches = tri_planar_patch_generator(x,y,z,CT_scan_array,patch_size)

# print patches[0,:,:]
# print patches[1,:,:]
# print patches[2,:,:]

# x_grid = np.arange(dicom_height)
# y_grid = np.arange(dicom_width)
# z_grid = np.arange(number_dicoms) 

# # Let's do it!
# # I first want to plot a full 2D slices
# from matplotlib import pyplot
# pyplot.figure(dpi=80)
# pyplot.axes().set_aspect('equal', 'datalim')
# pyplot.set_cmap(pyplot.gray())
# pyplot.pcolormesh(x_grid, y_grid, CT_scan_array[:,:,z])
# pyplot.plot(x,y,"ro")
# pyplot.show()

# # Then I want to plot a patch from that slice
# from matplotlib import pyplot
# pyplot.figure(dpi=80)
# pyplot.subplot(311)
# pyplot.axes().set_aspect('equal', 'datalim')
# pyplot.set_cmap(pyplot.gray())
# pyplot.pcolormesh(np.arange(patch_size), np.arange(patch_size), patches[0,:,:])

# pyplot.show()
# pyplot.subplot(312)
# pyplot.axes().set_aspect('equal', 'datalim')
# pyplot.set_cmap(pyplot.gray())
# pyplot.pcolormesh(np.arange(patch_size), np.arange(patch_size), patches[1,:,:])
# pyplot.show()

# pyplot.subplot(313)
# pyplot.axes().set_aspect('equal', 'datalim')
# pyplot.set_cmap(pyplot.gray())
# pyplot.pcolormesh(np.arange(patch_size), np.arange(patch_size), patches[2,:,:])
# pyplot.show()

#************************************************************************************************************
# 											GENERATING THE FULL DATASET
#************************************************************************************************************
# For each voxel, produce 3 32*32 perpendicular patches with it as their centre. 
# patch_size = 32
# tri_planar_dataset = np.zeros((CT_scan_array.size, 3, patch_size, patch_size))

# dicom_height  = 480
# dicom_width   = 480
# number_dicoms = 2
# # number_dicoms = len(CT_scan_dicom_filenames)

# x_grid = np.arange(dicom_height)
# y_grid = np.arange(dicom_width)
# z_grid = np.arange(2)
# # z_grid = np.arange(number_dicoms)

# for z in z_grid:
# 	print "Generating patches for the %i th dicom file..." %(z)
# 	for y in y_grid:
# 		for x in x_grid:
# 			tri_planar_dataset[x + dicom_width*y + dicom_height*dicom_width*z, :, :, :] = tri_planar_patch_generator(x,y,z,CT_scan_array,patch_size)

#************************************************************************************************************
# 									SAVING THE FULL DATASET INTO AN HDF5 File
#************************************************************************************************************
# Each data element is saved in 1 byte of memory as it takes values between 0 and 255.
# import h5py
# f = h5py.File("mytestfile.hdf5", "w")
# dataset = f.create_dataset("my_first_dataset", tri_planar_dataset.shape, dtype="uint8")

#************************************************************************************************************
# 											LOADING THE NRRD DATA
#************************************************************************************************************
# Loading the NRRD data
# import nrrd
# nrrd_path = "../../ct_atrium/14022803/14022803.nrrd"

# nrrd_data, nrrd_header = nrrd.read(nrrd_path)



if __name__ == "__main__":
	# Get the names of all the CT scans
	ct_directory_pattern = re.compile("[0-9]{8}")
	list_CT_scans = [directory for directory in os.listdir(data_directory) if ct_directory_pattern.match(directory)]

	# For every CT scan, get a list of their associated DICOM files
	CT_scan_dictionary = {}
	for CT_scan in list_CT_scans:
		CT_scan_directory = CT_scan_path.replace("CTScan_name", CT_scan)
		CT_scan_dictionary[CT_scan] = get_DICOM_names(CT_scan_directory)


	# For every CT scan, produce a dataset
	dataset_directory = os.path.join(data_directory, "datasets")
	dataset_path      = os.path.join(dataset_directory, "dataset.hdf5")
	f 			      = h5py.File(dataset_path, "w")
	for CT_scan, DICOM_list in CT_scan_dictionary.items()[0:3]:
		# Extract the 3d image into a numpy array
		print "Extracting the data from the DICOM files for CT scan %s" % CT_scan
		CT_scan_3d_image = get_CT_scan_array(CT_scan, DICOM_list)

		# Generate 3 perpendicular patches for each data point
		# For each voxel, produce 3 32*32 perpendicular patches with it as their centre. 
		patch_size    = 32
		dicom_height, dicom_width, number_dicoms = CT_scan_3d_image.shape

		x_grid = np.arange(dicom_height)
		y_grid = np.arange(dicom_width)
		z_grid = np.arange(number_dicoms)

		# tri_planar_dataset = np.zeros((CT_scan_3d_image.size, 3, patch_size, patch_size))
		# for z in z_grid:
		# 	print "Generating patches for the %i th dicom file..." %(z)
		# 	for y in y_grid:
		# 		for x in x_grid:
		# 			tri_planar_dataset[x + dicom_width*y + dicom_height*dicom_width*z, :, :, :] = tri_planar_patch_generator(x,y,z,CT_scan_3d_image,patch_size)

		z = 0
		tri_planar_dataset = np.zeros((CT_scan_3d_image[:,:,z].size, 3, patch_size, patch_size))
		print "Generating patches for the %i th dicom file..." %(z)
		for y in y_grid:
			for x in x_grid:
				tri_planar_dataset[x + dicom_width*y + dicom_height*dicom_width*z, :, :, :] = tri_planar_patch_generator(x,y,z,CT_scan_3d_image,patch_size)		

		# Save the dataset. Each data element is saved in 1 byte of memory as it takes values between 0 and 255.
		print "Saving the dataset..."
		dataset 	 = f.create_dataset(CT_scan, tri_planar_dataset.shape, dtype="uint8")
		dataset[...] = tri_planar_dataset

	print "The datasets have been saved into %s" % (dataset_path)


































