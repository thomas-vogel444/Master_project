import os
from Segmentator import Segmentator

segmentation_directory 	= "../../experimental_results/varying_learning_rate/0"
name_extensions 		= ["fixed_z", "fixed_y", "fixed_x"]

segmentation_parameters_template = {
			"GPU"					: 2,
			"segmentationFile" 		: "../../datasets/segmentation_datasets.hdf5",
			"segmentationDataset"	: "segmentation_dataset_NAME",
			"segmentationLabels"	: "labels_NAME",
			"segmentationValues"	: "values_NAME",
			"predictedPath"			: os.path.join(segmentation_directory, "predicted_labels.hdf5"),
			"predictedDataset"		: "predicted_labels_NAME",
			"modelPath"				: os.path.join(segmentation_directory, "model.net"),
			"imagePath"				: os.path.join(segmentation_directory, "image_NAME.png"),
			"type"					: "cuda"
		}

segmentator 			= Segmentator(segmentation_parameters_template)

for name_extension in name_extensions:
	segmentator.segment(name_extension)