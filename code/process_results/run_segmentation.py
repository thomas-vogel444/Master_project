import subprocess
import os

def segment(segmentation_parameters):
	segmentation_command = "th segment.lua -segmentationFile %(segmentationFile)s -segmentationDataset %(segmentationDataset)s "\
							"-save %(save)s -modelPath %(modelPath)s -type %(type)s" %segmentation_parameters

	print "******************** Running the following command ********************"
	print segmentation_command
	print 
	subprocess.call(segmentation_command, shell=True)

#***************************************************************************************************************
if __name__ == "__main__":
	segmentation_parameters = {
		"segmentationFile" 		: "../../datasets/small_segmentation_datasets.hdf5",
		"segmentationDataset"	: "segmentation_dataset_fixed_z",
		"save"					: "predicted_labels.hdf5",
		"modelPath"				: "model.net",
		"type"					: "float"
	}

	segment(segmentation_parameters)




















