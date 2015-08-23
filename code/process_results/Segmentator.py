import subprocess
import os

class Segmentator:
	"""
		Segmentator is responsible for running the segmentation routine segment.lua using a given model.
	"""
	def __init__(self, segmentation_parameters):
		self.segmentation_parameters = {
				"GPU_id"				: segmentation_parameters["GPU_id"],
				"number_of_GPUs"		: segmentation_parameters["number_of_GPUs"],
				"segmentationCode"		: segmentation_parameters["segmentationCode"],
				"segmentationDataset"	: "dataset",
				"predictedDataset"		: "predicted_labels",
				"modelPath"				: os.path.join(segmentation_parameters["modelDirectory"], "model.net"),
			}

	def segment(self, segmentation_path, predicted_path, height, width):
		"""
			Segments the three segmentation datasets in the segmentation hdf5 file.
		"""
		if os.path.isfile(predicted_path):
			os.remove(predicted_path)

		segmentation_command_options = self.get_segmentation_command_options(segmentation_path, predicted_path, height, width)
		
		segmentation_command = 	"th %(segmentationCode)s "\
								"-GPU_id %(GPU_id)i "\
								"-number_of_GPUs %(number_of_GPUs)i "\
								"-segmentationPath %(segmentationPath)s "\
								"-segmentationDataset %(segmentationDataset)s "\
								"-height %(height)i "\
								"-width %(width)i "\
								"-predictedPath %(predictedPath)s "\
								"-predictedDataset %(predictedDataset)s "\
								"-modelPath %(modelPath)s" %segmentation_command_options

		subprocess.call(segmentation_command, shell=True)

	def get_segmentation_command_options(self, segmentation_path, predicted_path, width, height):
		"""
			Customises the command option values for each segmentation dataset.
		"""
		command_options = dict(self.segmentation_parameters)

		command_options["segmentationPath"] = segmentation_path
		command_options["predictedPath"] 	= predicted_path
		command_options["width"] 			= width
		command_options["height"] 			= height

		return command_options
