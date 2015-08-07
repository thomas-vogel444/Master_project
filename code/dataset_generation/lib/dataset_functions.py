from CTScanImage import CTScanImage
import utils
import os
import itertools
import re
import numpy as np
import dicom
import nrrd
import Image

def random_3d_indices(CT_scan_labels, n, target_label, z):
	"""
		Randomly selects n data points from the set of points of a given label and returns their indices.
	"""
	if z is None:
		indices_3d = np.where(CT_scan_labels == target_label)
	else:
		indices_3d = list(np.where(CT_scan_labels[:,:,z] == target_label))
		indices_3d.append(np.ones(len(indices_3d[0]), dtype=np.int)*z)

	indices_1d = np.random.choice(xrange(len(indices_3d[0])), min(n,len(indices_3d[0])), replace=False)

	return np.dstack((indices_3d[0][indices_1d], 
											indices_3d[1][indices_1d], 
											indices_3d[2][indices_1d]))[0]

def generate_patch(x,y,image_2d, patch_size):
	"""
		Generate a single patch of size patch_size*patch_size.
	"""
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

def resize_patch(patch, patch_size):
	"""
		resizes the image to patch_size*patch_size.
	"""
	im = Image.fromarray(patch)
	out = im.resize((patch_size, patch_size))
	return np.array(out)

def generate_patches(voxel_location, image_3d, patch_size=32):
	""" 
		Generates a 3 * patch_size * patch_size numpy matrix centred at (x,y,z) containing the 
		three perpendicular patches from the 3D tensor image. 
	"""
	x,y,z = voxel_location
	patches = np.zeros((6, patch_size, patch_size))

	patches[0] = generate_patch(x,y,image_3d[:,:,z], patch_size)
	patches[1] = generate_patch(x,z,image_3d[:,y,:], patch_size)
	patches[2] = generate_patch(y,z,image_3d[x,:,:], patch_size)
	patches[3] = resize_patch(generate_patch(x,y,image_3d[:,:,z], 5*patch_size), patch_size)
	patches[4] = resize_patch(generate_patch(x,z,image_3d[:,y,:], 5*patch_size), patch_size)
	patches[5] = resize_patch(generate_patch(y,z,image_3d[x,:,:], 5*patch_size), patch_size)
	return patches

def generate_random_dataset_from_CT_scan(CT_scan, n_examples_per_label, patch_size, sampling_type, dicom_index=None):
	"""
		Generates a random dataset from a CT scan.
	"""
	tri_planar_dataset    = np.zeros((sum(n_examples_per_label), 6, patch_size, patch_size))
	tri_planar_labels     = np.zeros(sum(n_examples_per_label))

	# For each index sampled generate 3 patches centred at the voxel of interest
	labels = range(len(n_examples_per_label))
	random_indices = [CT_scan.sample_CT_scan_indices(sampling_type, n_examples_per_label[label], label, dicom_index) for label in labels]

	# Generates the datasets and labels of the sample points
	for i, index in enumerate(itertools.chain.from_iterable(random_indices)):
		utils.drawProgressBar(float(i)/(sum(n_examples_per_label)-1), 100)
		tri_planar_dataset[i] 	= generate_patches(index, CT_scan.image, patch_size)
		x,y,z = index
		tri_planar_labels[i]	= CT_scan.labels[x,y,z] + 1   # The training algorithm requires class labels to be 1 or 2 not 0 and 1

	return tri_planar_dataset, tri_planar_labels

def generate_random_dataset(CT_scan_names, n_examples_per_label, CT_scan_parameters_template, patch_size, sampling_type, dicom_index=None, xy_padding=0, z_padding=0):
	"""
		Generates a full dataset from a set of CT scans.
	"""
	n_examples_per_CT_scan = sum(n_examples_per_label)
	dataset = np.zeros((n_examples_per_CT_scan*len(CT_scan_names), 6, patch_size, patch_size))
	labels  = np.zeros(n_examples_per_CT_scan*len(CT_scan_names))

	for i, CT_scan_name in enumerate(CT_scan_names):
		print "Generating datasets from CT scan %s" %CT_scan_name
		CT_scan = CTScanImage(CT_scan_name, CT_scan_parameters_template, xy_padding, z_padding)
		dataset[(i*n_examples_per_CT_scan):((i+1)*n_examples_per_CT_scan)], labels[(i*n_examples_per_CT_scan):((i+1)*n_examples_per_CT_scan)] = \
				generate_random_dataset_from_CT_scan(CT_scan, n_examples_per_label, patch_size, sampling_type, dicom_index)

	return dataset, labels

