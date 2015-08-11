import utils
import itertools
import numpy as np


class DatasetGenerator:
	def __init__(self, CT_scan, patch_size):
		self.CT_scan 	= CT_scan
		self.patch_size = patch_size

	def generate_compressed_patch(self, center_coordinates, fixed_dimension, slice_number):
		"""
			Generates a compressed patch on a given 2D slice of the CT scan image located at location_2d
		"""
		if fixed_dimension == "x":
			CT_scan_slice = self.CT_scan.image[slice_number,:,:]
		if fixed_dimension == "y":
			CT_scan_slice = self.CT_scan.image[:,slice_number,:]
		if fixed_dimension == "z":
			CT_scan_slice = self.CT_scan.image[:,:,slice_number]

		large_patch = utils.padded_square_image_crop(center_coordinates, CT_scan_slice, 5*self.patch_size)
		return utils.resize_image_2d_array(large_patch, self.patch_size, self.patch_size)

	def generate_example_inputs(self, voxel_location):
		""" 
			Generates a 3 * patch_size * patch_size numpy matrix centred at (x,y,z) containing the 
			three perpendicular patches from the 3D tensor image. 
		"""
		x,y,z = voxel_location
		patches = np.zeros((6, self.patch_size, self.patch_size))

		# Get the 3 normal patches
		patches[0] = utils.padded_square_image_crop((x,y),self.CT_scan.image[:,:,z], self.patch_size)
		patches[1] = utils.padded_square_image_crop((x,z),self.CT_scan.image[:,y,:], self.patch_size)
		patches[2] = utils.padded_square_image_crop((y,z),self.CT_scan.image[x,:,:], self.patch_size)

		# Get the 3 compressed patches
		patches[3] = self.generate_compressed_patch((x,y), "z", z)
		patches[4] = self.generate_compressed_patch((x,z), "y", y)
		patches[5] = self.generate_compressed_patch((y,z), "x", x)
		return patches

	def generate_random_dataset(self, n_examples_per_label, sampling_type, dicom_index=None):
		"""
			Generates a random dataset from a CT scan.
		"""
		tri_planar_dataset    = np.zeros((sum(n_examples_per_label), 6, self.patch_size, self.patch_size))
		tri_planar_labels     = np.zeros(sum(n_examples_per_label))

		# For each index sampled generate 3 patches centred at the voxel of interest
		labels = range(len(n_examples_per_label))
		random_indices = [self.CT_scan.sample_CT_scan_indices(sampling_type, n_examples_per_label[label], label, dicom_index) for label in labels]

		# Generates the datasets and labels of the sample points
		for i, index in enumerate(itertools.chain.from_iterable(random_indices)):
			# Progress bar
			utils.drawProgressBar(float(i)/(sum(n_examples_per_label)-1), 100)

			# Generate the triplanar dataset and labels
			tri_planar_dataset[i] 	= self.generate_example_inputs(index)
			x,y,z = index
			tri_planar_labels[i]	= self.CT_scan.labels[x,y,z] + 1   # The training algorithm requires class labels to be 1 or 2 not 0 and 1

		return tri_planar_dataset, tri_planar_labels

