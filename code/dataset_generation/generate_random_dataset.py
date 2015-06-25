import os
import lib.dataset_functions as df
import re
import math
import numpy as np
import h5py
import lib.nrrd as nrrd

# - Generate a training dataset with 100000 example voxels half in the atrium, half not in the atrium randomly across 22 CT scans.
# - Generate a testing dataset with 20000 example voxels half in the atrium, half not in the atrium randomly across the remaining 5 CT scans.

if __name__ == "__main__":
	# Setting Parameters
	parameters = {
		"data_directory" 		: "../../ct_atrium",
		"CT_scan_path_template" : "../../ct_atrium/CTScan_name",
		"NRRD_path_template"    : "../../ct_atrium/CTScan_name/CTScan_name.nrrd",
		"DICOM_path_template"   : "../../ct_atrium/CTScan_name/DICOMS/DICOM_name",
		"ct_directory_pattern"  : re.compile("[0-9]{8}"),
		"patch_size"    		: 32,
		"n_training_CT_scans"   : 22,
		"n_testing_CT_scans"	: 5,
		"n_training_examples_per_CT_scan" : 2500,
		"n_testing_examples_per_CT_scan"  : 2500
	}

	# Separate the CT scans into a training and testing set
	CT_scans = df.get_CT_scan_names(parameters["data_directory"], parameters["ct_directory_pattern"])
	CT_scan_dictionary = df.get_all_DICOMs(parameters["CT_scan_path_template"], CT_scans)

	training_CT_scans = np.random.choice(CT_scans, parameters["n_training_CT_scans"], replace=False)
	testing_CT_scans  = [CT_scan for CT_scan in CT_scans if CT_scan not in training_CT_scans]

	# Generate the datasets
	print "=======> Generating the training dataset <======="
	tri_planar_training_dataset = df.generate_random_tri_planar_dataset(training_CT_scans, CT_scan_dictionary, parameters["n_training_examples_per_CT_scan"], parameters)
	print "=======> Generating the testing dataset <======="
	tri_planar_testing_dataset  = df.generate_random_tri_planar_dataset(testing_CT_scans, CT_scan_dictionary, parameters["n_testing_examples_per_CT_scan"], parameters)
	
	# Stick all this stuff into a dataset
	dataset_directory 	  = os.path.join(parameters["data_directory"], "datasets")
	dataset_path  		  = os.path.join(dataset_directory, "CNN_datasets.hdf5")
	print "=======> Saving the datasets in %s <=======" %dataset_path
	f 			  		  = h5py.File(dataset_path, "w")
	training_dataset 	  = f.create_dataset("training_dataset", tri_planar_training_dataset.shape, dtype="uint8")
	training_dataset[...] = tri_planar_training_dataset
	testing_dataset 	  = f.create_dataset("testing_dataset", tri_planar_testing_dataset.shape, dtype="uint8")
	testing_dataset[...]  = tri_planar_testing_dataset
	f.close()









