from CTScanImage import CTScanImage
import utils
import itertools
import numpy as np


class DatasetGenerator:
	def __init__(self, CT_scan, patch_size):
		self.CT_scan 	= CT_scan
		self.patch_size	= patch_size

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
		patches[3] = utils.resize_image_2d_array(	
							utils.padded_square_image_crop((x,y), self.CT_scan.image[:,:,z], 5*self.patch_size), 
							self.patch_size, 
							self.patch_size)
		patches[4] = utils.resize_image_2d_array(	
							utils.padded_square_image_crop((x,z), self.CT_scan.image[:,y,:], 5*self.patch_size), 
							self.patch_size, 
							self.patch_size)
		patches[5] = utils.resize_image_2d_array(	
							utils.padded_square_image_crop((y,z), self.CT_scan.image[x,:,:], 5*self.patch_size), 
							self.patch_size, 
							self.patch_size)

		return patches

	def generate_dataset_from_CT_scan(self, voxel_locations):
		"""
			Generates a dataset from a CT scan given a list of voxel locations.
		"""
		# For each index sampled generate 3 patches centred at the voxel of interest
		tri_planar_dataset = np.zeros((len(voxel_locations), 6, self.patch_size, self.patch_size))
		for i, voxel_location in enumerate(voxel_locations):
			# progress bar
			utils.drawProgressBar((i + 1)/float(len(voxel_locations)), bar_length = 100)

			# Generating the dataset
			tri_planar_dataset[i,:,:,:] = self.generate_example_inputs(voxel_location)
		
		return tri_planar_dataset

	def generate_full_transversal_segmentation_dataset(self, z):
		"""
			Generates a full dataset of a transversal slice (x-y plane) of a CT scan.
		"""
		# Get the 3d indices
		height, width, depth = self.CT_scan.image.shape
		full_indices_3d 	 = [list(xy_coordinate) + [z] for xy_coordinate in itertools.product(range(height), range(width))]

		# Generate a set of inputs for each voxel
		dataset = self.generate_dataset_from_CT_scan(full_indices_3d)

		return dataset




