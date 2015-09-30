import dataset_generation.CTScanImage as CTScanImage
import dataset_generation.DatasetGenerator as DatasetGenerator
import dataset_generation.utils as utils
import os
import re
import numpy as np
import h5py


def generate_random_dataset(CT_scan_names, n_examples_per_label, CT_scan_parameters_template, patch_size, sampling_type, dicom_index=None, xy_padding=0, z_padding=0):
	"""
		Generates a random dataset from a set of CT scans.
	"""
	dataset = np.zeros((sum(n_examples_per_label)*len(CT_scan_names), 6, patch_size, patch_size))
	labels  = np.zeros(sum(n_examples_per_label)*len(CT_scan_names))

	for i, CT_scan_name in enumerate(CT_scan_names):
		print "Generating datasets from CT scan %s" %CT_scan_name
		CT_scan 			= CTScanImage(CT_scan_name, CT_scan_parameters_template, xy_padding, z_padding)
		dataset_generator 	= DatasetGenerator(CT_scan, patch_size)

		# Randomly generate the pixel indices 
		random_indices = list(itertools.chain.from_iterable(
					[	CT_scan.sample_CT_scan_indices(sampling_type, n_examples_per_label[label-1], label, dicom_index) 
						for label in range(1, len(n_examples_per_label) + 1)])
				)

		n_examples = sum(n_examples_per_label)

		# Get the dataset for a given CT scan
		dataset[(i*n_examples):((i+1)*n_examples)] = dataset_generator.generate_dataset_from_CT_scan(random_indices)

		# Get the respective labels
		labels[(i*n_examples):((i+1)*n_examples)]  = np.array(map(CT_scan.get_label, random_indices))

	return dataset, labels

if __name__ == "__main__":
	# Setting Parameters
	data_directory 		= "../ct_atrium/training"
	dataset_directory 	= "../datasets/"
	random_dataset_name = "nothing_dataset.hdf5"

	patch_size 							= 32
	sampling_type						= "Without_Atrium_Box"		# options: Random, With_Atrium_Box, Without_Atrium_Box
	n_examples_per_CT_scan_per_label 	= (100, 100) 	# (n_non_bd_non_atrium, n_bd_non_atrium, n_atrium)
	xy_padding, z_padding  				= 5, 1

	CT_scan_parameters_template = {
			"CT_scan_path_template" : os.path.join(data_directory, "CTScan_name"),
			"NRRD_path_template"    : os.path.join(data_directory, "CTScan_name/CTScan_name.nrrd"),
			"DICOM_directory"		: os.path.join(data_directory, "CTScan_name/DICOMS"),
			"DICOM_path_template"   : os.path.join(data_directory, "CTScan_name/DICOMS/DICOM_name"),
			"CT_directory_pattern"  : re.compile("[0-9]{8}")
		}

	# Find the names of the CT scan directories in the data directory
	CT_scan_names = [directory for directory in os.listdir(data_directory) if CT_scan_parameters_template["CT_directory_pattern"].match(directory)]

	# Generate the dataset from the list of CT scans
	print "=======> Generating the dataset <======="
	dataset, labels = generate_random_dataset(	CT_scan_names, 
											n_examples_per_CT_scan_per_label, 
											CT_scan_parameters_template, 
											patch_size, 
											sampling_type, 
											xy_padding=xy_padding, 
											z_padding=z_padding, 
										)

	# Saving the dataset
	dataset_path 	= os.path.join(dataset_directory, random_dataset_name)
	print "=======> Saving dataset in %s <=======" %dataset_path
	utils.save_dataset(dataset_path, dataset, labels)






