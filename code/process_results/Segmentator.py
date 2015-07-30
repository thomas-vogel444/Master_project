import pprint as pp
import subprocess
import os

class Segmentator:
	"""
		Segmentator is responsible for running the segmentation routine segment.lua using a given model.
	"""
	name_extensions = ["fixed_z", "fixed_y", "fixed_x"]

	def __init__(self, segmentation_dataset):
		self.segmentation_parameter_template = {
				"GPU"					: 2,
				"segmentationFile" 		: segmentation_dataset,
				"segmentationDataset"	: "segmentation_dataset_NAME",
				"segmentationLabels"	: "labels_NAME",
				"segmentationValues"	: "values_NAME",
				"predictedPath"			: "MODEL_DIRECTORY/predicted_labels.hdf5",
				"predictedDataset"		: "predicted_labels_NAME",
				"modelPath"				: "MODEL_DIRECTORY/model.net",
				"imagePath"				: "MODEL_DIRECTORY/image_NAME.png",
				"type"					: "cuda"
			}

	def segment(self, model_directory):
		"""
			Segments the three segmentation datasets in the segmentation hdf5 file.
		"""
		segmentation_parameters 	 = self.get_segmentation_parameters(model_directory)
		if os.path.isfile(self.segmentation_parameter_template["predictedPath"]):
			os.remove(self.segmentation_parameter_template["predictedPath"])
		
		for name_extension in self.name_extensions:	
			segmentation_command_options = self.get_segmentation_command_options(segmentation_parameters, name_extension)

			segmentation_command = "th segment.lua -GPU %(GPU)i -segmentationFile %(segmentationFile)s -segmentationLabels %(segmentationLabels)s "\
									"-segmentationValues %(segmentationValues)s -segmentationDataset %(segmentationDataset)s "\
									"-predictedPath %(predictedPath)s -predictedDataset %(predictedDataset)s -imagePath %(imagePath)s "\
									"-modelPath %(modelPath)s -type %(type)s" %segmentation_command_options

			print "******************** Running the following command ********************"
			print segmentation_command
			print 
			subprocess.call(segmentation_command, shell=True)

	def get_segmentation_parameters(self, model_directory):
		"""
			Fill in the model directory in the parameter template.
		"""
		segmentation_parameters = dict(self.segmentation_parameter_template)
		segmentation_parameters["predictedPath"] 	= segmentation_parameters["predictedPath"].replace("MODEL_DIRECTORY", model_directory)
		segmentation_parameters["modelPath"] 		= segmentation_parameters["modelPath"].replace("MODEL_DIRECTORY", model_directory)
		segmentation_parameters["imagePath"] 		= segmentation_parameters["imagePath"].replace("MODEL_DIRECTORY", model_directory)
		return segmentation_parameters

	def get_segmentation_command_options(self, segmentation_parameters, name_extension):
		"""
			Customises the command option values for each segmentation dataset.
		"""
		command_options = segmentation_parameters

		command_options["segmentationDataset"] 	= command_options["segmentationDataset"].replace("NAME", name_extension)
		command_options["segmentationLabels"] 	= command_options["segmentationLabels"].replace("NAME", name_extension)
		command_options["segmentationValues"] 	= command_options["segmentationValues"].replace("NAME", name_extension)
		command_options["segmentationLabels"] 	= command_options["segmentationLabels"].replace("NAME", name_extension)
		command_options["predictedDataset"] 	= command_options["predictedDataset"].replace("NAME", name_extension)
		command_options["modelPath"] 			= command_options["modelPath"].replace("NAME", name_extension)
		command_options["imagePath"] 			= command_options["imagePath"].replace("NAME", name_extension)

		return command_options