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

def generate_dataset_from_CT_scan(CT_scan, patch_size, n_examples_per_label, sampling_type, dicom_index=None, multithreaded=True):
	"""
		Generates a random dataset from a CT scan.
	"""
	# For each index sampled generate 3 patches centred at the voxel of interest
	labels = range(1, len(n_examples_per_label) + 1)	# Should be (1,2,3) or (1,2) if the sampling type is "With_Atrium" or "Without_Atrium" respectively
	random_indices = [CT_scan.sample_CT_scan_indices(sampling_type, n_examples_per_label[label-1], label, dicom_index) for label in labels]

	#*****************************************************************
	# Set up the multiprocessing stuff	
	#*****************************************************************
	# First attempt... Good for now but later I want to try to make something with dividing the workload among processes.
	map_function = partial(generate_example_inputs, CT_scan=CT_scan, patch_size=patch_size)
	if multithreaded == True:
		pool = Pool() 
		tri_planar_dataset 	= pool.map(map_function, itertools.chain.from_iterable(random_indices))
		pool.close() 
		pool.join() 
	else:
		tri_planar_dataset 	= map(map_function, itertools.chain.from_iterable(random_indices))

	tri_planar_labels	= map(CT_scan.get_label, itertools.chain.from_iterable(random_indices))

	return np.array(tri_planar_dataset), np.array(tri_planar_labels)

