from dataset_generation.CTScanImage import CTScanImage
import dataset_generation.dataset_functions as df
import os
import re
import numpy as np
import h5py


def generate_dataset(CT_scan_names, n_examples_per_label, CT_scan_parameters_template, patch_size, sampling_type, dicom_index=None, xy_padding=0, z_padding=0):
	"""
		Generates a full dataset from a set of CT scans.
	"""
	dataset = np.zeros((sum(n_examples_per_label)*len(CT_scan_names), 6, patch_size, patch_size))
	labels  = np.zeros(sum(n_examples_per_label)*len(CT_scan_names))

	for i, CT_scan_name in enumerate(CT_scan_names):
		print "Generating datasets from CT scan %s" %CT_scan_name
		CT_scan 			= CTScanImage(CT_scan_name, CT_scan_parameters_template, xy_padding, z_padding)
		CT_scan_dataset, CT_scan_labels = df.generate_dataset_from_CT_scan(CT_scan, patch_size, n_examples_per_label, sampling_type, dicom_index)

		n_examples = len(CT_scan_labels)
		dataset[(i*n_examples):((i+1)*n_examples)], labels[(i*n_examples):((i+1)*n_examples)] = CT_scan_dataset, CT_scan_labels		

	return dataset, labels

if __name__ == "__main__":
	# Setting Parameters
	data_directory 		= "../ct_atrium/"
	dataset_directory 	= "../datasets/"
	random_dataset_name = "CNN_small_atrium_box_datasets.hdf5"

	patch_size 									= 32
	sampling_type								= "With_Atrium_Box"		# options: Random, With_Atrium_Box, Without_Atrium_Box
	n_training_examples_per_CT_scan_per_label 	= (3750, 3750, 7500) 	# (n_non_bd_non_atrium, n_bd_non_atrium, n_atrium)
	n_testing_examples_per_CT_scan_per_label	= [100000]
	xy_padding, z_padding  						= 5, 1

	# ********************************************************************************************
	# 						Generate the training and testing datasets
	# ********************************************************************************************
	CT_scan_types = ["training", "testing"]
	dataset_path 	= os.path.join(dataset_directory, random_dataset_name)

	print "=======> Saving the training and testing datasets in %s <=======" %dataset_path
	f 				= h5py.File(dataset_path, "w")

	for CT_scan_type in CT_scan_types:
		CT_scans_directory	= os.path.join(data_directory, CT_scan_type)

		CT_scan_parameters_template = {
				"CT_scan_path_template" : os.path.join(CT_scans_directory, "CTScan_name"),
				"NRRD_path_template"    : os.path.join(CT_scans_directory, "CTScan_name/CTScan_name.nrrd"),
				"DICOM_directory"		: os.path.join(CT_scans_directory, "CTScan_name/DICOMS"),
				"DICOM_path_template"   : os.path.join(CT_scans_directory, "CTScan_name/DICOMS/DICOM_name"),
				"CT_directory_pattern"  : re.compile("[0-9]{8}")
			}

		print "=======> Generating the %s dataset <======="%CT_scan_type
		CT_scan_names		= [directory for directory in os.listdir(CT_scans_directory) if CT_scan_parameters_template["CT_directory_pattern"].match(directory)]
		
		if CT_scan_type == "training":
			dataset, labels = generate_dataset(CT_scan_names, n_training_examples_per_CT_scan_per_label, CT_scan_parameters_template, patch_size, sampling_type, xy_padding=xy_padding, z_padding=z_padding)
		elif CT_scan_type == "testing":
			dataset, labels = generate_dataset(CT_scan_names, n_testing_examples_per_CT_scan_per_label, CT_scan_parameters_template, patch_size, "Random")

		dataset_hdf5 	  	  = f.create_dataset("%s_dataset"%CT_scan_type, dataset.shape, dtype="uint32")
		dataset_hdf5[...]     = dataset
		labels_hdf5 	  	  = f.create_dataset("%s_labels"%CT_scan_type, labels.shape, dtype="uint8")
		labels_hdf5[...]  	  = labels








