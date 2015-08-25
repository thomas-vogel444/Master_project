import dataset_generation.dataset_functions as df
import dataset_generation.utils as utils
import os
import re
import numpy as np
import h5py

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
	dataset, labels = df.generate_random_dataset(	CT_scan_names, 
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






