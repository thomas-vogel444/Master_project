from CTScanImage import CTScanImage
import utils
from functools import partial
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
import itertools
import numpy as np

def generate_example_inputs(voxel_location, CT_scan, patch_size):
	""" 
		Generates a 3 * patch_size * patch_size numpy matrix centred at (x,y,z) containing the 
		three perpendicular patches from the 3D tensor image. 
	"""
	x,y,z = voxel_location
	patches = np.zeros((6, patch_size, patch_size))

	# Get the 3 normal patches
	patches[0] = utils.padded_square_image_crop((x,y),CT_scan.image[:,:,z], patch_size)
	patches[1] = utils.padded_square_image_crop((x,z),CT_scan.image[:,y,:], patch_size)
	patches[2] = utils.padded_square_image_crop((y,z),CT_scan.image[x,:,:], patch_size)

	# Get the 3 compressed patches
	patches[3] = utils.resize_image_2d_array(utils.padded_square_image_crop((x,y), CT_scan.image[:,:,z], 5*patch_size), patch_size, patch_size)
	patches[4] = utils.resize_image_2d_array(utils.padded_square_image_crop((x,z), CT_scan.image[:,y,:], 5*patch_size), patch_size, patch_size)
	patches[5] = utils.resize_image_2d_array(utils.padded_square_image_crop((y,z), CT_scan.image[x,:,:], 5*patch_size), patch_size, patch_size)

	return patches

def generate_dataset_from_CT_scan(voxel_locations, CT_scan, patch_size):
	"""
		Generates a dataset from a CT scan given a list of voxel locations.
	"""
	# For each index sampled generate 3 patches centred at the voxel of interest
	n_examples = len(voxel_locations)

	tri_planar_dataset = np.zeros((n_examples, 6, patch_size, patch_size))
	for i, voxel_location in enumerate(voxel_locations):
		# progress bar
		utils.drawProgressBar((i + 1)/n_examples, bar_length = 20)

		# Generating the dataset
		tri_planar_dataset[i,:,:,:] = generate_example_inputs(voxel_location, CT_scan, patch_size)
	
	return tri_planar_dataset

def generate_full_segmentation_dataset(CT_scan, patch_size):
	"""
		Generates a full dataset for all the voxels in the CT scan.
	"""
	# Get the 3d indices
	height, width, depth = CT_scan.image.shape
	full_indices_3d 	 = list(itertools.product(range(height), range(width), range(depth)))

	# Generate a set of inputs for each voxel
	tri_planar_dataset 	= generate_dataset_from_CT_scan(full_indices_3d, CT_scan, patch_size)

	return tri_planar_dataset

def generate_random_dataset(CT_scan_names, n_examples_per_label, CT_scan_parameters_template, patch_size, sampling_type, dicom_index=None, xy_padding=0, z_padding=0):
	"""
		Generates a random dataset from a set of CT scans.
	"""
	dataset = np.zeros((sum(n_examples_per_label)*len(CT_scan_names), 6, patch_size, patch_size))
	labels  = np.zeros(sum(n_examples_per_label)*len(CT_scan_names))

	for i, CT_scan_name in enumerate(CT_scan_names):
		print "Generating datasets from CT scan %s" %CT_scan_name
		CT_scan 			= CTScanImage(CT_scan_name, CT_scan_parameters_template, xy_padding, z_padding)

		label_types = range(1, len(n_examples_per_label) + 1)	# Should be (1,2,3) or (1,2) if the sampling type is "With_Atrium" or "Without_Atrium" respectively
		random_indices = list(itertools.chain.from_iterable(
				[CT_scan.sample_CT_scan_indices(sampling_type, n_examples_per_label[label-1], label, dicom_index) for label in label_types]))

		n_examples = len(CT_scan_labels)
		dataset[(i*n_examples):((i+1)*n_examples)] = generate_dataset_from_CT_scan(random_indices, CT_scan, patch_size)
		labels[(i*n_examples):((i+1)*n_examples)]  = np.array(map(CT_scan.get_label, random_indices))

	return dataset, labels




