from Segmentator import Segmentator

segmentation_parameters = {
	"segmentationFile" 		: "../../datasets/small_segmentation_datasets.hdf5",
	"predictedPath"			: "predicted_labels.hdf5",
	"modelPath"				: "model.net",
	"type"					: "float"
}

datasets = {
	"segmentation_dataset_fixed_z" : "predicted_labels_fixed_z",
	"segmentation_dataset_fixed_y" : "predicted_labels_fixed_y",
	"segmentation_dataset_fixed_x" : "predicted_labels_fixed_x"
}

segmentator = Segmentator(segmentation_parameters)

for segmentation_dataset, predicted_dataset in datasets.items():
	segmentator.segment(segmentation_dataset, predicted_dataset)