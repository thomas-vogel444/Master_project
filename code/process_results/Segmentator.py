import pprint as pp
import subprocess
import os

class Segmentator:
	"""
		Segmentator is responsible for running the segmentation routine segment.lua using a given model.
	"""
	name_extensions = ["fixed_z", "fixed_y", "fixed_x"]

	def __init__(self, model_directory, segmentation_dataset):
		self.segmentation_parameter_template = {
				"GPU"					: 2,
				"segmentationFile" 		: segmentation_dataset,
				"segmentationDataset"	: "segmentation_dataset_NAME",
				"segmentationLabels"	: "labels_NAME",
				"segmentationValues"	: "values_NAME",
				"predictedPath"			: os.path.join(model_directory, "predicted_labels.hdf5"),
				"predictedDataset"		: "predicted_labels_NAME",
				"modelPath"				: os.path.join(model_directory, "model.net"),
				"imagePath"				: os.path.join(model_directory, "image_NAME.png"),
				"type"					: "cuda"
			}

		if os.path.isfile(self.segmentation_parameter_template["predictedPath"]):
			os.remove(self.segmentation_parameter_template["predictedPath"])

	def segment(self):
		for name_extension in self.name_extensions:
			segmentation_command_options = self.get_segmentation_command_options(name_extension)
			segmentation_command = "th segment.lua -GPU %(GPU)i -segmentationFile %(segmentationFile)s -segmentationLabels %(segmentationLabels)s "\
									"-segmentationValues %(segmentationValues)s -segmentationDataset %(segmentationDataset)s "\
									"-predictedPath %(predictedPath)s -predictedDataset %(predictedDataset)s -imagePath %(imagePath)s "\
									"-modelPath %(modelPath)s -type %(type)s" %segmentation_command_options

			print "******************** Running the following command ********************"
			print segmentation_command
			print 
			subprocess.call(segmentation_command, shell=True)

	def get_segmentation_command_options(self, name_extension):
		command_options = dict(self.segmentation_parameter_template)

		command_options["segmentationDataset"] 	= command_options["segmentationDataset"].replace("NAME", name_extension)
		command_options["segmentationLabels"] 	= command_options["segmentationLabels"].replace("NAME", name_extension)
		command_options["segmentationValues"] 	= command_options["segmentationValues"].replace("NAME", name_extension)
		command_options["segmentationLabels"] 	= command_options["segmentationLabels"].replace("NAME", name_extension)
		command_options["predictedDataset"] 	= command_options["predictedDataset"].replace("NAME", name_extension)
		command_options["modelPath"] 			= command_options["modelPath"].replace("NAME", name_extension)
		command_options["imagePath"] 			= command_options["imagePath"].replace("NAME", name_extension)

		return command_options