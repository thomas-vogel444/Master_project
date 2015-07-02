import lib.dataset_functions as df
import lib.utils as utils
import os
import re
import math
import numpy as np
import h5py

# - Generate a training dataset with 100000 example voxels half in the atrium, half not in the atrium randomly across 22 CT scans.
# - Generate a testing dataset with 20000 example voxels half in the atrium, half not in the atrium randomly across the remaining 5 CT scans.
# - Generate a full dataset for the dicom file of 1 CT scan for segmentation

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

	# ********************************************************************************************
	# 					Separate the CT scans into a training and testing set
	# ********************************************************************************************
	CT_scans = df.get_CT_scan_names(parameters["data_directory"], parameters["ct_directory_pattern"])
	CT_scan_dictionary = df.get_all_DICOMs(parameters["CT_scan_path_template"], CT_scans)

	np.random.seed(12)
	training_CT_scans = np.random.choice(CT_scans, parameters["n_training_CT_scans"], replace=False)
	testing_CT_scans  = [CT_scan for CT_scan in CT_scans if CT_scan not in training_CT_scans]

	# ********************************************************************************************
	# 						Generate the training and tetsing datasets
	# ********************************************************************************************
	print "=======> Generating the training dataset <======="
	tri_planar_training_dataset = df.generate_random_tri_planar_dataset(training_CT_scans, CT_scan_dictionary, parameters["n_training_examples_per_CT_scan"], parameters)
	print "=======> Generating the testing dataset <======="
	tri_planar_testing_dataset  = df.generate_random_tri_planar_dataset(testing_CT_scans, CT_scan_dictionary, parameters["n_testing_examples_per_CT_scan"], parameters)

	# ********************************************************************************************
	# 	  Generating the segmentation dataset from a dicom file of one of the testing CT scans
	# ********************************************************************************************
	segmented_CT_scan 		 = testing_CT_scans[0]
	segmented_CT_scan_DICOMS = df.get_DICOMs(parameters["CT_scan_path_template"].replace("CTScan_name", segmented_CT_scan))
	nrrd_path 		  		 = parameters["NRRD_path_template"].replace("CTScan_name", segmented_CT_scan)

	# CT_scan_labels, CT_scan_nrrd_header 	 = nrrd.read(nrrd_path)	
	CT_scan_labels, CT_scan_nrrd_header 	 = df.get_NRRD_array(nrrd_path)
	CT_scan_3d_image  						 = df.get_CT_scan_array(segmented_CT_scan, segmented_CT_scan_DICOMS, 
																	CT_scan_nrrd_header["sizes"], parameters["DICOM_path_template"])
	dicom_height, dicom_width, number_dicoms = CT_scan_3d_image.shape
	x_grid, y_grid = utils.generate_grids(dicom_height, dicom_width)

	tri_planar_segmentation_dataset = np.zeros((CT_scan_3d_image[:,:,0].size, 3, parameters["patch_size"], parameters["patch_size"]))

	z = 30
	print "Generating the segmentation dataset from the DICOM file %i from CT scan %s..." %(z, segmented_CT_scan)
	for x in y_grid:
		for y in x_grid:
			tri_planar_segmentation_dataset[y + dicom_width*x, :, :, :] = df.tri_planar_patch_generator(x,y,z,CT_scan_3d_image,parameters["patch_size"])

	# ********************************************************************************************
	# 								Stick all this stuff into a dataset
	# ********************************************************************************************
	dataset_directory = os.path.join(parameters["data_directory"], "datasets")
	dataset_path      = os.path.join(dataset_directory, "CNN_datasets.hdf5")

	print "=======> Saving the datasets in %s <=======" %dataset_path
	f 			  		  	  = h5py.File(dataset_path, "w")
	training_dataset 	  	  = f.create_dataset("training_dataset", tri_planar_training_dataset.shape, dtype="uint8")
	training_dataset[...]  	  = tri_planar_training_dataset
	testing_dataset 	  	  = f.create_dataset("testing_dataset", tri_planar_testing_dataset.shape, dtype="uint8")
	testing_dataset[...]  	  = tri_planar_testing_dataset
	segmentation_dataset_name = "segmentation_dataset"
	segmentation_dataset  	  = f.create_dataset(segmentation_dataset_name, tri_planar_segmentation_dataset.shape, dtype="uint8")
	segmentation_dataset[...] = tri_planar_segmentation_dataset
	segmentation_label_name   = "segmentation_labels"
	segmentation_label  	  = f.create_dataset(segmentation_label_name, CT_scan_labels[:,:,z].shape, dtype="uint8")
	segmentation_label[...]   = CT_scan_labels[:,:,z]
	segmentation_values_name  = "segmentation_values"
	segmentation_values  	  = f.create_dataset(segmentation_values_name, CT_scan_3d_image[:,:,z].shape, dtype="uint8")
	segmentation_values[...]  = CT_scan_3d_image[:,:,z]
	f.close()









