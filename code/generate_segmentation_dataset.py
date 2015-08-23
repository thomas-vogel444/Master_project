from dataset_generation.CTScanImage import CTScanImage
import dataset_generation.utils as utils
import dataset_generation.dataset_functions as df
import os
import re
import numpy as np
import h5py


if __name__ == "__main__":
	# ********************************************************************************************
	# 										Setting parameters
	# ********************************************************************************************
	data_directory 				= "../ct_atrium/testing"
	dataset_directory 			= "../datasets/"
	patch_size 					= 32

	CT_scan_parameters_template = {
					"CT_scan_path_template" : os.path.join(data_directory, "CTScan_name"),
					"NRRD_path_template"    : os.path.join(data_directory, "CTScan_name/CTScan_name.nrrd"),
					"DICOM_directory"		: os.path.join(data_directory, "CTScan_name/DICOMS"),
					"DICOM_path_template"   : os.path.join(data_directory, "CTScan_name/DICOMS/DICOM_name"),
					"CT_directory_pattern"  : re.compile("[0-9]{8}")
				}

	# Find all the CT scan names in the data_directory
	testing_CT_scan_names = [directory for directory in os.listdir(data_directory) if CT_scan_parameters_template["CT_directory_pattern"].match(directory)]

	for testing_CT_scan_name in testing_CT_scan_names:
		segmentation_directory 	= os.path.join(dataset_directory, "segmentation_dataset_for_%s"%testing_CT_scan_name)
		segmented_CT_scan 		= CTScanImage(testing_CT_scan_name, CT_scan_parameters_template)
		_, _, number_of_slices 	= segmented_CT_scan.image.shape

		for z in range(number_of_slices):
			print "Generating a dataset for transversal slice number: %i"%z

			# Generating the transversal dataset
			transversal_dataset = df.generate_full_transversal_segmentation_dataset(segmented_CT_scan, patch_size, z)

			# Saving the transversal dataset
			transversal_dataset_path = os.path.join(segmentation_directory, "segmentation_dataset_#%i.hdf5")%z
			print "Saving the dataset in %s"%s
			utils.save_dataset(transversal_dataset_path, transversal_dataset)

