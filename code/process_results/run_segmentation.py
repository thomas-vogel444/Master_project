import subprocess
import os

def segment(segmentation_parameters):
	segmentation_command = "th segment.lua -segmentationFile %(segmentationFile)s -segmentationDataset %(segmentationDataset)s "\
							"-predictedPath %(predictedPath)s -predictedDataset -%(predictedDataset)s -modelPath %(modelPath)s "\
							"-type %(type)s" %segmentation_parameters

	print "******************** Running the following command ********************"
	print segmentation_command
	print 
	subprocess.call(segmentation_command, shell=True)

#***************************************************************************************************************
if __name__ == "__main__":
	segmentation_parameters = {
		"segmentationFile" 		: "../../datasets/segmentation_datasets.hdf5",
		"segmentationDataset"	: "segmentation_dataset_fixed_z",
		"predictedPath"			: "predicted_labels.hdf5",
		"predictedDataset"		: "predicted_labels_fixed_z",
		"modelPath"				: "model.net",
		"type"					: "float"
	}

	datasets = {
		"segmentation_dataset_fixed_z" : "predicted_labels_fixed_z",
		"segmentation_dataset_fixed_y" : "predicted_labels_fixed_y",
		"segmentation_dataset_fixed_x" : "predicted_labels_fixed_x"
	}

	if os.path.isfile(segmentation_parameters["predictedPath"]):
		os.remove(segmentation_parameters["predictedPath"])

	for segmentation_dataset, predicted_dataset in datasets.items():
		segmentation_parameters["segmentationDataset"]  = segmentation_dataset
		segmentation_parameters["predictedDataset"] 	= predicted_dataset
		segment(segmentation_parameters)




















