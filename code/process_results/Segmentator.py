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

	def segment(self, segmentation_file, predicted_file, height, width):
		"""
			Segments the three segmentation datasets in the segmentation hdf5 file.
		"""
		if os.path.isfile(self.segmentation_parameters["predictedPath"]):
			os.remove(self.segmentation_parameters["predictedPath"])

		segmentation_command_options = self.get_segmentation_command_options(segmentationFile, predicted_file, height, width)

		segmentation_command = 	"th %(segmentationCode)s "\
								"-GPU_id %(GPU_id)i "\
								"-number_of_GPUs %(number_of_GPUs)i "\
								"-segmentationFile %(segmentationFile)s "\
								"-segmentationDataset %(segmentationDataset)s "\
								"-height %i"\
								"-width %i"\
								"-predictedPath %(predictedPath)s "\
								"-predictedDataset %(predictedDataset)s "\
								"-modelPath %(modelPath)s" %segmentation_command_options

		subprocess.call(segmentation_command, shell=True)

	def get_segmentation_command_options(self, segmentation_path, predicted_file, width, height):
		"""
			Customises the command option values for each segmentation dataset.
		"""
		command_options = dict(self.segmentation_parameters)

		command_options["segmentationPath"] = segmentation_path
		command_options["predicted_path"] 	= predicted_file
		command_options["width"] 			= width
		command_options["height"] 			= height

		return command_options
