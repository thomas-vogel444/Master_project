import dicom
import h5py
import os
import re
import pprint as pp
import numpy as np

# Get all the CT scan folder names
data_directory = "../../ct_atrium"
ct_scan_path = "../../ct_atrium/CTScan_name"
DICOM_path = "../../ct_atrium/CTScan_name/DICOMS/DICOM_name"

ct_directory_pattern = re.compile("[0-9]{8}")
ListCTScans = [directory for directory in os.listdir(data_directory) if ct_directory_pattern.match(directory)]

#************************************************************
# For each CT scan folder get all the DICOM names
#************************************************************
ct_scan_dictionary = {}
for CT_scan_name in ListCTScans:

	# Get the path of a specific CT scan data directory
	ct_scan_directory = ct_scan_path.replace("CTScan_name", CT_scan_name)

	# Get the DICOM file names into a dictionary with the CT scan name as value and a list of DICOM image names as values
	dicom_directory = os.path.join(ct_scan_directory, "DICOMS")
	dicom_name_pattern = re.compile("^[0-9]{8}")
	dicom_files = []
	for root, directory, files in os.walk(dicom_directory):
		dicom_files = [myfile for myfile in files if dicom_name_pattern.match(myfile)]

	ct_scan_dictionary[CT_scan_name] = dicom_files
	
#*****************************************************************************************
# For a given dicom file, produce a 4D numpy array Batch Channel Width Height
# with:
#	Batch   : number of training examples, i.e. number of voxels in the image 480 * 480
#	Channel : 3 planes centred at the voxel
#	Width   : width of the image path, i.e. 32
#	Height  : height of the image patch, i.e. 32
#*****************************************************************************************
# Get the path for a given CT scan
CT_scan_names           = ct_scan_dictionary.keys()
CT_scan_name 			= CT_scan_names[0]
CT_scan_dicom_filenames = ct_scan_dictionary[CT_scan_name]

dicom_height  = 480
dicom_width   = 480
number_dicoms = len(CT_scan_dicom_filenames)

# Loop through all the DICOM files for a given CT scan and get all the values into a single 3D numpy array
CT_scan_array = np.zeros((dicom_height, dicom_width, number_dicoms), dtype="uint16")
for dicom_filename in CT_scan_dicom_filenames:
    # read the file
    dicom_file_path = DICOM_path.replace("CTScan_name", CT_scan_name).replace("DICOM_name", dicom_filename)
    print "Reading %s" %(dicom_file_path)
    ds = dicom.read_file(dicom_file_path)
    # store the raw image data
    CT_scan_array[:, :, CT_scan_dicom_filenames.index(dicom_filename)] = ds.pixel_array  

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
tri_planar_dataset = np.zeros((CT_scan_array.size, 3, patch_size, patch_size))

dicom_height  = 480
dicom_width   = 480
number_dicoms = len(CT_scan_dicom_filenames)

x_grid = np.arange(dicom_height)
y_grid = np.arange(dicom_width)
z_grid = np.arange(number_dicoms)

for z in z_grid:
	for y in y_grid:
		for x in x_grid:
			tri_planar_dataset[x + dicom_width*y + dicom_height*dicom_width*z, :, :, :] = tri_planar_patch_generator(x,y,z,CT_scan_array,patch_size)


#************************************************************************************************************
# 											LOADING THE NRRD DATA
#************************************************************************************************************
# Loading the NRRD data
import nrrd
nrrd_path = "../../ct_atrium/14022803/14022803.nrrd"

nrrd_data, nrrd_header = nrrd.read(nrrd_path)









