import os
import re
import numpy as np
import dicom
import nrrd

def get_CT_scan_names(data_directory, ct_directory_pattern):
	"""
		Get the names of all the CT scans
	"""
	return [directory for directory in os.listdir(data_directory) if ct_directory_pattern.match(directory)]

def get_DICOMs(CT_scan_directory):
	"""
		Get the DICOM file names following a given pattern for a given CT scan, i.e. a name with 8 digits in it.
	"""
	DICOM_directory = os.path.join(CT_scan_directory, "DICOMS")
	DICOM_name_pattern = re.compile("^[0-9]{8}")
	DICOM_names = []
	for root, directory, files in os.walk(DICOM_directory):
		DICOM_names = [myfile for myfile in files if DICOM_name_pattern.match(myfile)]

	return DICOM_names

def get_all_DICOMs(CT_scan_path_template, CT_scans):
	"""
		For every CT scan, get a list of their associated DICOM files.
		This returns a dictionary.
	"""
	CT_scan_dictionary = {}
	for CT_scan in CT_scans:
		CT_scan_dictionary[CT_scan] = get_DICOMs(CT_scan_path_template.replace("CTScan_name", CT_scan))
	return CT_scan_dictionary


def get_CT_scan_array(CT_scan_name, CT_scan_dicom_filenames, DICOM_dimensions, DICOM_path_template):	
	"""
		For a given dicom file, produce a 4D numpy array Batch Channel Width Height
		with:
			Batch   : number of training examples, i.e. number of voxels in the image 480 * 480
			Channel : 3 planes centred at the voxel
			Width   : width of the image path, i.e. 32
			Height  : height of the image patch, i.e. 32
	"""
	# Loop through all the DICOM files for a given CT scan and get all the values into a single 3D numpy array
	CT_scan_array = np.zeros(DICOM_dimensions, dtype="uint16")
	for dicom_filename in CT_scan_dicom_filenames:
	    # read the file
	    dicom_file_path = DICOM_path_template.replace("CTScan_name", CT_scan_name).replace("DICOM_name", dicom_filename)
	    ds = dicom.read_file(dicom_file_path)
	    # store the raw image data
	    CT_scan_array[:, :, CT_scan_dicom_filenames.index(dicom_filename)] = ds.pixel_array

	return CT_scan_array

def random_3d_indices(CT_scan_labels, n, target_label):
	"""
		Randomly selects n data points from the set of points of a given label and returns their indices.
	"""
	atrium_3d_indices = np.where(CT_scan_labels == 1)
	atrium_training_1d_indices = np.random.choice(xrange(len(np.where(CT_scan_labels == 1)[0])), n, replace=False)
	
	return np.dstack((atrium_3d_indices[0][atrium_training_1d_indices], 
											atrium_3d_indices[1][atrium_training_1d_indices], 
											atrium_3d_indices[2][atrium_training_1d_indices]))[0]

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

def generate_random_tri_planar_dataset(CT_scans, CT_scan_dictionary, n_examples_per_CT_scan, parameters):
	tri_planar_dataset = np.zeros((n_examples_per_CT_scan * len(CT_scans), 3, parameters["patch_size"], parameters["patch_size"]))
	for i_CT_scan, CT_scan in enumerate(CT_scans):
		# Extract the NRRD file into a numpy array
		nrrd_path = parameters["NRRD_path_template"].replace("CTScan_name", CT_scan)
		CT_scan_labels, CT_scan_nrrd_header = nrrd.read(nrrd_path)

		# Get the 3D image into a numpy array
		print "Extracting 3d image from the DICOM files for CT scan %s" % CT_scan
		DICOMs = CT_scan_dictionary[CT_scan]
		CT_scan_3d_image = get_CT_scan_array(CT_scan, DICOMs, CT_scan_nrrd_header["sizes"], parameters["DICOM_path_template"])

		# Sample indexes from the atrium
		atrium_3d_indices = random_3d_indices(CT_scan_labels, n_examples_per_CT_scan/2, 1)

		# For each index sampled generate 3 patches centred at the voxel of interest
		for i, atrium_3d_index in enumerate(atrium_3d_indices):
			x, y, z = atrium_3d_index
			tri_planar_dataset[i + n_examples_per_CT_scan/2 * i_CT_scan] = tri_planar_patch_generator(x, y, z ,CT_scan_3d_image,parameters["patch_size"])

		# Sample indexes from the non-atrium
		non_atrium_3d_indices = random_3d_indices(CT_scan_labels, n_examples_per_CT_scan/2, 0)

		# For each index sampled generate 3 patches centred at the voxel of interest
		for i, non_atrium_3d_index in enumerate(non_atrium_3d_indices):
			x, y, z = non_atrium_3d_index
			tri_planar_dataset[len(CT_scans)*n_examples_per_CT_scan/2 + i + n_examples_per_CT_scan/2 * i_CT_scan] = \
					tri_planar_patch_generator(x, y, z ,CT_scan_3d_image,parameters["patch_size"])
	return tri_planar_dataset
