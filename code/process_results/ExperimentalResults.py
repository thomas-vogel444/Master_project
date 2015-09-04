from SegmentedCTScan import SegmentedCTScan
import numpy as np
import h5py
import re
import os

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

	def __init__(self, experiment_base_directory, CT_scans):
		self.experiment_base_directory 	= experiment_base_directory
		self.predicted_labels 			= self.get_predicted_labels(CT_scans)
		self.segmented_CT_scans 		= self.get_segmented_CT_scans(CT_scans)

		self.testing_dice_coefficients 	= self.get_dice_coefficients("test")
		self.training_dice_coefficients = self.get_dice_coefficients("train")

	def get_predicted_labels(self, CT_scans):
		"""
			Get the predicted labels.
		"""
		predicted_labels = []
		for segmented_CT_scan in CT_scans: 
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
			predicted_labels.append(prediction)
		return predicted_labels

	def get_segmented_CT_scans(self, CT_scans):
		segmented_CT_scans = []
		for CT_scan, predicted_label in zip(CT_scans, self.predicted_labels): 
			segmented_CT_scans.append(SegmentedCTScan(CT_scan, predicted_label))

		return segmented_CT_scans 

	def get_dice_coefficients(self, type):
		"""	
			Extracts the dice coefficients from test.log or train.log
		"""
		filename = os.path.join(self.experiment_base_directory, "%s.log"%type)
		return read_values(filename)







