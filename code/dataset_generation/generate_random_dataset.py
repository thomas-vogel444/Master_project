from lib.CTScanImage import CTScanImage
import lib.dataset_functions as df
import os
import re
import math
import numpy as np
import h5py
import argparse

if __name__ == "__main__":
	# Setting Parameters
	data_directory = "../../ct_atrium/"
	dataset_directory = "../../datasets/"
	
	CT_scan_parameters_template = {
		"CT_scan_path_template" : data_directory + "CTScan_name",
		"NRRD_path_template"    : data_directory + "CTScan_name/CTScan_name.nrrd",
		"DICOM_directory"		: data_directory + "CTScan_name/DICOMS",
		"DICOM_path_template"   : data_directory + "CTScan_name/DICOMS/DICOM_name",
		"CT_directory_pattern"  : re.compile("[0-9]{8}")
		}
	patch_size 			= 32
	n_training_CT_scans = 22
	n_testing_CT_scans  = 5
	# Labels: 
	# 			Atrium: 				 2
	# 			Boundary Non-Atrium: 	 1
	# 			Non-Boundary Non-Atrium: 0
	n_examples_per_CT_scan_per_label = (3500, 3500, 3500) # (n_non_bd_non_atrium, n_bd_non_atrium, n_atrium)

	# ********************************************************************************************
	# 					Separate the CT scans into a training and testing set
	# ********************************************************************************************
	# Get the names of all the CT scans
	CT_scan_names = [directory for directory in os.listdir(data_directory) if CT_scan_parameters_template["CT_directory_pattern"].match(directory)]

	np.random.seed(12)
	training_CT_scan_names = np.random.choice(CT_scan_names, n_training_CT_scans, replace=False)
	testing_CT_scan_names  = [CT_scan_name for CT_scan_name in CT_scan_names if CT_scan_name not in training_CT_scan_names]

	generate_random_dataset = False
	generate_segmentation_dataset = True
	# ********************************************************************************************
	# 						Generate the training and testing datasets
	# ********************************************************************************************
	if generate_random_dataset == True:
		print "=======> Generating the training dataset <======="
		training_dataset, training_labels = df.generate_random_dataset(training_CT_scan_names, n_examples_per_CT_scan_per_label, CT_scan_parameters_template, patch_size)

		print "=======> Generating the testing dataset <======="
		testing_dataset, testing_labels  = df.generate_random_dataset(testing_CT_scan_names, n_examples_per_CT_scan_per_label, CT_scan_parameters_template, patch_size)
		dataset_path      = os.path.join(dataset_directory, "CNN_datasets.hdf5")

		print "=======> Saving the training and testing datasets in %s <=======" %dataset_path
		f 			  		  	  = h5py.File(dataset_path, "w")
		# For the training dataset
		training_dataset_hdf5 	  	  = f.create_dataset("training_dataset", training_dataset.shape, dtype="uint32")
		training_dataset_hdf5[...]    = np.int16(training_dataset)
		training_labels_hdf5 	  	  = f.create_dataset("training_labels", training_labels.shape, dtype="uint8")
		training_labels_hdf5[...]  	  = np.int16(training_labels)
		# For the testing dataset
		testing_dataset_hdf5 	  	  = f.create_dataset("testing_dataset", testing_dataset.shape, dtype="uint32")
		testing_dataset_hdf5[...]  	  = np.int16(testing_dataset)
		testing_labels_hdf5 	  	  = f.create_dataset("testing_labels", testing_labels.shape, dtype="uint8")
		testing_labels_hdf5[...]  	  = testing_labels
		f.close()

	# # ********************************************************************************************
	# # 	  Generating the segmentation dataset from a dicom file of one of the testing CT scans
	# # ********************************************************************************************
	if generate_segmentation_dataset == True:
		segmented_CT_scan_name 	 = testing_CT_scan_names[0]
		segmented_CT_scan 		 = CTScanImage(segmented_CT_scan_name, CT_scan_parameters_template)

		dicom_height, dicom_width, number_dicoms = segmented_CT_scan.image.shape
		x_grid, y_grid = range(dicom_height), range(dicom_width)

		tri_planar_segmentation_dataset = np.zeros((segmented_CT_scan.image[:,:,0].size, 6, patch_size, patch_size))

		z = 30
		print "=======> Generating the segmentation dataset from the DICOM file %i from CT scan %s... <=======" %(z, segmented_CT_scan.name)
		for x in y_grid:
			for y in x_grid:
				tri_planar_segmentation_dataset[y + dicom_width*x, :, :, :] = df.generate_patches((x,y,z),segmented_CT_scan.image, patch_size)

		segmentation_dataset_path = os.path.join(dataset_directory, "segmentation_datasets.hdf5")
		print "=======> Saving the segmentation dataset in %s <=======" %segmentation_dataset_path
		f 			  		  	  = h5py.File(segmentation_dataset_path, "w")
		segmentation_dataset_name = "segmentation_dataset"
		segmentation_dataset  	  = f.create_dataset(segmentation_dataset_name, tri_planar_segmentation_dataset.shape, dtype="uint32")
		segmentation_dataset.attrs["CT_scan"] 	   = segmented_CT_scan.name
		segmentation_dataset.attrs["DICOM_number"] = z
		segmentation_dataset[...] = tri_planar_segmentation_dataset
		segmentation_label_name   = "segmentation_labels"
		segmentation_label  	  = f.create_dataset(segmentation_label_name, segmented_CT_scan.labels[:,:,z].shape, dtype="uint8")
		segmentation_label[...]   = segmented_CT_scan.labels[:,:,z]
		segmentation_values_name  = "segmentation_values"
		segmentation_values  	  = f.create_dataset(segmentation_values_name, segmented_CT_scan.image[:,:,z].shape, dtype="uint32")
		segmentation_values[...]  = segmented_CT_scan.image[:,:,z]








