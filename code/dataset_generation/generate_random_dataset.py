from lib.CTScanImage import CTScanImage
import lib.dataset_functions as df
import lib.utils as utils
import os
import sys
import re
import math
import numpy as np
import h5py
import argparse

if __name__ == "__main__":
	generate_random_dataset 	  = False
	generate_segmentation_dataset = True

	# Setting Parameters
	data_directory 		= "../../ct_atrium/"
	dataset_directory 	= "../../datasets/"
	random_dataset_name = "CNN_box_atrium_datasets.hdf5"
	
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
	n_examples_per_CT_scan_per_label = (3500, 3500, 7000) # (n_non_bd_non_atrium, n_bd_non_atrium, n_atrium)
	xy_padding = 5
	z_padding  = 5
	# ********************************************************************************************
	# 					Separate the CT scans into a training and testing set
	# ********************************************************************************************
	# Get the names of all the CT scans
	CT_scan_names = [directory for directory in os.listdir(data_directory) if CT_scan_parameters_template["CT_directory_pattern"].match(directory)]

	np.random.seed(12)
	training_CT_scan_names = np.random.choice(CT_scan_names, n_training_CT_scans, replace=False)
	testing_CT_scan_names  = [CT_scan_name for CT_scan_name in CT_scan_names if CT_scan_name not in training_CT_scan_names]

	# ********************************************************************************************
	# 						Generate the training and testing datasets
	# ********************************************************************************************
	if generate_random_dataset == True:
		print "=======> Generating the training dataset <======="
		training_dataset, training_labels = df.generate_random_dataset(training_CT_scan_names, n_examples_per_CT_scan_per_label, CT_scan_parameters_template, patch_size, xy_padding=xy_padding, z_padding=z_padding)

		print "=======> Generating the testing dataset <======="
		testing_dataset, testing_labels  = df.generate_random_dataset(testing_CT_scan_names, n_examples_per_CT_scan_per_label, CT_scan_parameters_template, patch_size, xy_padding=xy_padding, z_padding=z_padding)
		dataset_path      = os.path.join(dataset_directory, random_dataset_name)

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
		x_grid, y_grid, z_grid 					 = range(dicom_height), range(dicom_width), range(number_dicoms)
		x_slice, y_slice, z_slice 				 = (250, 250, 30)


		print "=======> Generating the segmentation dataset for fixed z = %i from CT scan %s... <=======" %(z_slice, segmented_CT_scan.name)
		tri_planar_segmentation_dataset_fixed_z = np.zeros((segmented_CT_scan.image[:,:,0].size, 6, patch_size, patch_size))
		for x in x_grid:
			utils.drawProgressBar(float(x)/(dicom_height-1), 100)
			for y in y_grid:
				tri_planar_segmentation_dataset_fixed_z[y + dicom_width*x, :, :, :] = df.generate_patches((x,y,z_slice),segmented_CT_scan.image, patch_size)

		print "=======> Generating the segmentation dataset for fixed y = %i from CT scan %s... <=======" %(y_slice, segmented_CT_scan.name)
		tri_planar_segmentation_dataset_fixed_y = np.zeros((segmented_CT_scan.image[:,0,:].size, 6, patch_size, patch_size))
		for x in x_grid:
			utils.drawProgressBar(float(x)/(dicom_height-1), 100)
			for z in z_grid:
				tri_planar_segmentation_dataset_fixed_y[z + number_dicoms*x, :, :, :] = df.generate_patches((x,y_slice,z),segmented_CT_scan.image, patch_size)

		print "=======> Generating the segmentation dataset for fixed x = %i from CT scan %s... <=======" %(x_slice, segmented_CT_scan.name)
		tri_planar_segmentation_dataset_fixed_x = np.zeros((segmented_CT_scan.image[0,:,:].size, 6, patch_size, patch_size))
		for z in z_grid:
			utils.drawProgressBar(float(z)/(number_dicoms-1), 100)
			for y in y_grid:
				tri_planar_segmentation_dataset_fixed_x[y + dicom_width*z, :, :, :] = df.generate_patches((x_slice,y,z),segmented_CT_scan.image, patch_size)

		segmentation_dataset_path = os.path.join(dataset_directory, "segmentation_datasets.hdf5")
		f 			  		  	  = h5py.File(segmentation_dataset_path, "w")
		print "=======> Saving the segmentation datasets for fixed z in %s <=======" %segmentation_dataset_path
		segmentation_dataset_fixed_z_name = "segmentation_dataset_fixed_z"
		segmentation_dataset_fixed_z 				  = f.create_dataset(segmentation_dataset_fixed_z_name, tri_planar_segmentation_dataset_fixed_z.shape, dtype="uint32")
		segmentation_dataset_fixed_z.attrs["CT_scan"] = segmented_CT_scan.name
		segmentation_dataset_fixed_z.attrs["z_slice"] = z_slice
		segmentation_dataset_fixed_z[...] 			  = tri_planar_segmentation_dataset_fixed_z

		print "=======> Saving the segmentation datasets for fixed y in %s <=======" %segmentation_dataset_path
		segmentation_dataset_fixed_y_name = "segmentation_dataset_fixed_y"
		segmentation_dataset_fixed_y 				  = f.create_dataset(segmentation_dataset_fixed_y_name, tri_planar_segmentation_dataset_fixed_y.shape, dtype="uint32")
		segmentation_dataset_fixed_y.attrs["CT_scan"] = segmented_CT_scan.name
		segmentation_dataset_fixed_y.attrs["y_slice"] = y_slice
		segmentation_dataset_fixed_y[...] 			  = tri_planar_segmentation_dataset_fixed_y

		print "=======> Saving the segmentation datasets for fixed x in %s <=======" %segmentation_dataset_path
		segmentation_dataset_fixed_x_name = "segmentation_dataset_fixed_x"
		segmentation_dataset_fixed_x 				  = f.create_dataset(segmentation_dataset_fixed_x_name, tri_planar_segmentation_dataset_fixed_x.shape, dtype="uint32")
		segmentation_dataset_fixed_x.attrs["CT_scan"] = segmented_CT_scan.name
		segmentation_dataset_fixed_x.attrs["x_slice"] = x_slice
		segmentation_dataset_fixed_x[...] 			  = tri_planar_segmentation_dataset_fixed_x
		f.close()








