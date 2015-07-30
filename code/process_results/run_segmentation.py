from Segmentator import Segmentator

segmentation_parameters = {
	"GPU"					: 2,
	"segmentationFile" 		: "../../datasets/segmentation_datasets.hdf5",
	"segmentationDataset"	: "segmentation_dataset_fixed_z",
	"segmentationLabels"	: "segmentation_labels_fixed_z",
	"segmentationValues"	: "segmentation_values_fixed_z",
	"predictedPath"			: "predicted_labels.hdf5",
	"predictedDataset"		: "predicted_labels_fixed_z"
	"modelPath"				: "model.net",
	"type"					: "cuda"
}

segmentation_particulars = [["segmentation_dataset_fixed_z", "segmentation_labels_fixed_z", "segmentation_values_fixed_z", "predicted_labels_fixed_z"],
		["segmentation_dataset_fixed_y", "segmentation_labels_fixed_y", "segmentation_values_fixed_y", "predicted_labels_fixed_y"],
		["segmentation_dataset_fixed_x", "segmentation_labels_fixed_x", "segmentation_values_fixed_x", "predicted_labels_fixed_x"]]

segmentator = Segmentator(segmentation_parameters)

for segmentation_particular in segmentation_particulars:
	segmentator.segment(segmentation_particular)
