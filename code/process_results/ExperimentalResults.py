from dataset_generation.CTScanImage import CTScanImage
import numpy as np
import h5py
import re
import os


data_directory 				= "../ct_atrium/testing"
CT_scan_parameters_template = {
		"CT_scan_path_template" : os.path.join(data_directory, "CTScan_name"),
		"NRRD_path_template"    : os.path.join(data_directory, "CTScan_name/CTScan_name.nrrd"),
		"DICOM_directory"		: os.path.join(data_directory, "CTScan_name/DICOMS"),
		"DICOM_path_template"   : os.path.join(data_directory, "CTScan_name/DICOMS/DICOM_name"),
		"CT_directory_pattern"  : re.compile("[0-9]{8}")
	}

def get_slice_number(segmentation_dataset):
	p = re.compile("predicted_file_(.*).hdf5")
	return int(p.search(segmentation_dataset).group(1))

def read_values(filename):
    with open(filename) as f:
    	f.readline()
        return [float(x.strip()) for x in f]

class ExperimentalResults:
	"""
		ExperimentalResults is responsible for extracting the results from an experiment base directory.
	"""

	def __init__(self, experiment_base_directory):
		self.experiment_base_directory 	= experiment_base_directory
		self.segmented_CT_scans 		= self.get_CT_scans()
		self.predicted_labels 			= self.get_predicted_labels()
		self.testing_dice_coefficients 	= self.get_dice_coefficients("test")
		self.training_dice_coefficients = self.get_dice_coefficients("train")

	def get_predicted_labels(self):
		"""
			Get the predicted labels.
		"""
		predictions = []
		for segmented_CT_scan in self.segmented_CT_scans: 
			# Get all the hdf5 files
			predicted_files_directory 	= os.path.join(self.experiment_base_directory, "predicted_files_%s"%segmented_CT_scan.name)
			predicted_files 		= sorted([predicted_file for predicted_file in os.listdir(predicted_files_directory)], key=get_slice_number)

			# Get the dimensions
			predicted_file_path = os.path.join(predicted_files_directory, predicted_files[0])
			f = h5py.File(predicted_file_path, "r")
			height, width = np.array(f["predicted_labels"]).shape
			f.close()

			# Extract the 3D labeling image
			prediction = np.zeros((height, width, len(predicted_files)))
			for index, predicted_file in enumerate(predicted_files):
				predicted_file_path = os.path.join(predicted_files_directory, predicted_file)

				f = h5py.File(predicted_file_path, "r")
				prediction[:,:,index] = np.array(f["predicted_labels"]) + 1 # prediction comes with labels 0 or 1. We want labels 1 or 2
				f.close()
			predictions.append(prediction)
		return predictions

	def get_CT_scans(self):
		return [CTScanImage("14040204", CT_scan_parameters_template), 
				CTScanImage("14031201", CT_scan_parameters_template)]

	def get_dice_coefficients(self, type):
		"""	
			Extracts the dice coefficients from test.log or train.log
		"""
		filename = os.path.join(self.experiment_base_directory, "%s.log"%type)
		return read_values(filename)

	def get_classification_statistics(self):
		"""
			Produce classification error statistics from the predicted labels and true labels.
		"""
		classification_statistics = {}

		true_positives 	= np.zeros(self.predicted_labels[0].shape, 'uint8')
		true_negatives 	= np.zeros(self.predicted_labels[0].shape, 'uint8')
		errors 			= np.zeros(self.predicted_labels[0].shape, 'uint8')

		number_of_atrium_voxels 				= len(np.where(self.segmented_CT_scans[0].labels == 2)[0])
		number_of_non_atrium_voxels				= len(np.where(self.segmented_CT_scans[0].labels == 1)[0])
		number_correctly_classified_atrium 		= np.sum(np.where((self.predicted_labels[0] == self.segmented_CT_scans[0].labels) & (self.segmented_CT_scans[0].labels == 2), 1, 0))
		number_correctly_classified_non_atrium 	= np.sum(np.where((self.predicted_labels[0] == self.segmented_CT_scans[0].labels) & (self.segmented_CT_scans[0].labels == 1), 1, 0))
		
		errors = np.where(self.predicted_labels[0] != self.segmented_CT_scans[0].labels, 1, 0)

		classification_statistics["Sensitivity"] = round(number_correctly_classified_atrium/float(number_of_atrium_voxels), 3)

		classification_statistics["Specificity"] = round(number_correctly_classified_non_atrium/float(number_of_non_atrium_voxels), 3)

		classification_statistics["Dice coefficient over test CT scan"] = round(	(number_correctly_classified_atrium + 
															number_correctly_classified_non_atrium) /
															float(self.predicted_labels[0].size), 3)

		classification_statistics["Dice coefficient over training set"] = round(self.training_dice_coefficients[-1], 3)
		classification_statistics["Dice coefficient over testing set"] = round(self.testing_dice_coefficients[-1], 3)
		return classification_statistics







