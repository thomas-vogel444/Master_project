import os
import re
import numpy as np
import dicom
import nrrd
import Image


def get_CT_scan_names(data_directory, ct_directory_pattern):
	"""
		Get the names of all the CT scans
	"""
	return [directory for directory in os.listdir(data_directory) if ct_directory_pattern.match(directory)]

def get_NRRD_array(nrrd_path):
	"""
		Wrapper around nrrd.read so as to return the transpose of the array.
	"""
	CT_scan_labels, CT_scan_nrrd_header = nrrd.read(nrrd_path)
	CT_scan_labels = np.transpose(CT_scan_labels, (1,0,2))
	return CT_scan_labels, CT_scan_nrrd_header

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

def random_3d_indices(CT_scan_labels, n, target_label, z):
	"""
		Randomly selects n data points from the set of points of a given label and returns their indices.
	"""
	if z is None:
		indices_3d = np.where(CT_scan_labels == target_label)
	else:
		indices_3d = list(np.where(CT_scan_labels[:,:,z] == target_label))
		indices_3d.append(np.ones(len(indices_3d[0]))*z)

	indices_1d = np.random.choice(xrange(len(indices_3d[0])), n, replace=False)

	return np.dstack((indices_3d[0][indices_1d], 
											indices_3d[1][indices_1d], 
											indices_3d[2][indices_1d]))[0]


def generate_patch(x,y,image_2d, patch_size):
	height, width = image_2d.shape
	patch = np.zeros((patch_size, patch_size))

	x_min = 0
	x_max = patch_size
	y_min = 0
	y_max = patch_size

	if x < patch_size/2:
		x_min = patch_size/2 - x

	if x > height - patch_size/2:
		x_max = patch_size/2 + height - x

	if y < patch_size/2:
		y_min = patch_size/2 - y

	if y > width - patch_size/2:
		y_max = patch_size/2 + width - y

	patch[x_min:x_max, y_min:y_max] = image_2d[np.maximum(x-patch_size/2, 0):np.minimum(x+patch_size/2, height), 
						 np.maximum(y-patch_size/2, 0):np.minimum(y+patch_size/2, width)]

	return patch


def generate_patches(x,y,z,image_3d,patch_size):
	""" 
		Generates a 3 * patch_size * patch_size numpy matrix centred at (x,y,z) containing the 
		three perpendicular patches from the 3D tensor image. 
	"""
	height, width, depth = image_3d.shape
	patches = np.zeros((4, patch_size, patch_size))

	patches[0] = generate_patch(x,y,image_3d[:,:,z], patch_size)
	patches[1] = generate_patch(x,z,image_3d[:,y,:], patch_size)
	patches[2] = generate_patch(y,z,image_3d[x,:,:], patch_size)
	patches[3] = resize_patch(generate_patch(x,y,image_3d[:,:,z], 4*patch_size))
	return patches

def generate_random_dataset(CT_scans, CT_scan_dictionary, n_examples_per_CT_scan, parameters, z=None):
	tri_planar_dataset = np.zeros((n_examples_per_CT_scan * len(CT_scans), 4, parameters["patch_size"], parameters["patch_size"]))
	tri_planar_labels  = np.zeros(n_examples_per_CT_scan * len(CT_scans))
	for i_CT_scan, CT_scan in enumerate(CT_scans):
		# Extract the NRRD file into a numpy array
		nrrd_path = parameters["NRRD_path_template"].replace("CTScan_name", CT_scan)
		CT_scan_labels, CT_scan_nrrd_header = get_NRRD_array(nrrd_path)

		# Get the 3D image into a numpy array
		print "Extracting 3d image from the DICOM files for CT scan %s" % CT_scan
		DICOMs = CT_scan_dictionary[CT_scan]
		CT_scan_3d_image = get_CT_scan_array(CT_scan, DICOMs, CT_scan_nrrd_header["sizes"], parameters["DICOM_path_template"])

		# Sample indexes from the atrium and non-atrium
		atrium_3d_indices     = random_3d_indices(CT_scan_labels, n_examples_per_CT_scan/2, 1, z)
		non_atrium_3d_indices = random_3d_indices(CT_scan_labels, n_examples_per_CT_scan/2, 0, z)

		# For each index sampled generate 3 patches centred at the voxel of interest
		for i, atrium_3d_index in enumerate(atrium_3d_indices):
			x, y, z = atrium_3d_index
			tri_planar_dataset[i + n_examples_per_CT_scan/2 * i_CT_scan] = generate_patches(x, y, z ,CT_scan_3d_image,parameters["patch_size"])
			tri_planar_labels[i + n_examples_per_CT_scan/2 * i_CT_scan] = 2

		# For each index sampled generate 3 patches centred at the voxel of interest
		for i, non_atrium_3d_index in enumerate(non_atrium_3d_indices):
			x, y, z = non_atrium_3d_index
			tri_planar_dataset[len(CT_scans)*n_examples_per_CT_scan/2 + i + n_examples_per_CT_scan/2 * i_CT_scan] = \
					generate_patches(x, y, z ,CT_scan_3d_image,parameters["patch_size"])
			tri_planar_labels[len(CT_scans)*n_examples_per_CT_scan/2 + i + n_examples_per_CT_scan/2 * i_CT_scan] = 1
	return tri_planar_dataset, tri_planar_labels

def resize_patch(patch):
	"""
		resizes the image to 32*32.
	"""
	im = Image.fromarray(patch)
	out = im.resize((32, 32))
	return np.array(out)


