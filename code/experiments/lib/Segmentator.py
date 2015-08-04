import subprocess
import os

class Segmentator:
	"""
		Segmentator is responsible for running the segmentation routine segment.lua using a given model.
	"""
	name_extensions = ["fixed_z", "fixed_y", "fixed_x"]

	def __init__(self, segmentation_parameters):
		self.segmentation_command_line_options = {
				"GPU"					: segmentation_parameters["GPU"],
				"segmentationCode"		: segmentation_parameters["segmentationCode"],
				"segmentationFile" 		: segmentation_parameters["segmentationFile"],
				"segmentationDataset"	: "segmentation_dataset_NAME",
				"segmentationLabels"	: "labels_NAME",
				"segmentationValues"	: "values_NAME",
				"predictedPath"			: os.path.join(segmentation_parameters["modelDirectory"], "predicted_labels.hdf5"),
				"predictedDataset"		: "predicted_labels_NAME",
				"modelPath"				: os.path.join(segmentation_parameters["modelDirectory"], "model.net"),
				"imagePath"				: os.path.join(segmentation_parameters["modelDirectory"], "image_NAME.png"),
				"type"					: "cuda"
			}

	def segment(self):
		"""
			Segments the three segmentation datasets in the segmentation hdf5 file.
		"""
		if os.path.isfile(segmentation_parameters["predictedPath"]):
			os.remove(segmentation_parameters["predictedPath"])

		for name_extension in self.name_extensions:	
			segmentation_command_options = self.get_segmentation_command_options(segmentation_parameters, name_extension)

			segmentation_command = "th %(segmentationCode)s -GPU %(GPU)i -segmentationFile %(segmentationFile)s -segmentationLabels %(segmentationLabels)s "\
									"-segmentationValues %(segmentationValues)s -segmentationDataset %(segmentationDataset)s "\
									"-predictedPath %(predictedPath)s -predictedDataset %(predictedDataset)s -imagePath %(imagePath)s "\
									"-modelPath %(modelPath)s -type %(type)s" %segmentation_command_options

			print "******************** Running the following command ********************"
			print segmentation_command
			print 
			subprocess.call(segmentation_command, shell=True)

	def get_segmentation_command_options(self, segmentation_parameters, name_extension):
		"""
			Customises the command option values for each segmentation dataset.
		"""
		command_options = dict(segmentation_parameters)

		command_options["segmentationDataset"] 	= command_options["segmentationDataset"].replace("NAME", name_extension)
		command_options["segmentationLabels"] 	= command_options["segmentationLabels"].replace("NAME", name_extension)
		command_options["segmentationValues"] 	= command_options["segmentationValues"].replace("NAME", name_extension)
		command_options["segmentationLabels"] 	= command_options["segmentationLabels"].replace("NAME", name_extension)
		command_options["predictedDataset"] 	= command_options["predictedDataset"].replace("NAME", name_extension)
		command_options["modelPath"] 			= command_options["modelPath"].replace("NAME", name_extension)
		command_options["imagePath"] 			= command_options["imagePath"].replace("NAME", name_extension)

		return command_options