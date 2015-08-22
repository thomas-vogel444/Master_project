from dataset_generation.CTScanImage import CTScanImage
import dataset_generation.dataset_functions as df
import dataset_generation.utils as utils
import os
import re


if __name__ == "__main__":
	# Setting Parameters
	data_directory 		= "../ct_atrium/testing"
	dataset_directory 	= "../datasets/"
	random_dataset_name = "nothing_dataset.hdf5"

	patch_size 							= 32
	multithreaded 						= False

	CT_scan_parameters_template = {
			"CT_scan_path_template" : os.path.join(data_directory, "CTScan_name"),
			"NRRD_path_template"    : os.path.join(data_directory, "CTScan_name/CTScan_name.nrrd"),
			"DICOM_directory"		: os.path.join(data_directory, "CTScan_name/DICOMS"),
			"DICOM_path_template"   : os.path.join(data_directory, "CTScan_name/DICOMS/DICOM_name"),
			"CT_directory_pattern"  : re.compile("[0-9]{8}")
		}

	# Find the names of the CT scan directories in the data directory
	CT_scan_names = [directory for directory in os.listdir(data_directory) if CT_scan_parameters_template["CT_directory_pattern"].match(directory)]

	for CT_scan_name in [CT_scan_names[0]]:
		# Generate the dataset from the list of CT scans
		print "=======> Generating the dataset for CT scan %s <======="%CT_scan_name
		CT_scan = CTScanImage(CT_scan_name, CT_scan_parameters_template)

		dataset = df.generate_full_segmentation_dataset(CT_scan, patch_size)

		# Saving the dataset
		dataset_path = os.path.join(dataset_directory, "full_3d_segmentation_dataset_%s.hdf5"%CT_scan_name)
		print "=======> Saving dataset in %s <=======" %dataset_path
		utils.save_dataset(dataset_path, dataset)