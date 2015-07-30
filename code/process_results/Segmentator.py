import subprocess
import os

class Segmentator:
	def __init__(self, segmentation_parameters):
		self.segmentation_parameters = segmentation_parameters
		if os.path.isfile(segmentation_parameters["predictedPath"]):
			os.remove(segmentation_parameters["predictedPath"])

	def segment(self):
		segmentation_command = "th segment.lua -GPU %(GPU)i -segmentationFile %(segmentationFile)s -segmentationLabels %(segmentationLabels)s "\
								"-segmentationValues %(segmentationValues)s -segmentationDataset %(segmentationDataset)s "\
								"-predictedPath %(predictedPath)s -predictedDataset %(predictedDataset)s -modelPath %(modelPath)s "\
								"-type %(type)s" %segmentation_command_options

		print "******************** Running the following command ********************"
		print segmentation_command
		print 
		subprocess.call(segmentation_command, shell=True)

	def get_segmentation_command_options(self, segmentation_particular):
		return dict(self.segmentation_parameters.items() + 
					{"segmentationDataset"	: segmentation_particulars[0]}.items() + 
					{"segmentationLabels"	: segmentation_particulars[1]}.items() + 
					{"segmentationValues"	: segmentation_particulars[2]}.items() + 
					{"predictedDataset"		: segmentation_particulars[3]}.items())


	"segmentationDataset"	: "segmentation_dataset_fixed_z",
	"segmentationLabels"	: "segmentation_labels_fixed_z",
	"segmentationValues"	: "segmentation_values_fixed_z",
	"predictedPath"			: "predicted_labels.hdf5",
	"predictedDataset"		: "predicted_labels_fixed_z"