import pprint as pp
import subprocess
import os

class Segmentator:
	def __init__(self, segmentation_parameter_template):
		self.segmentation_parameter_template = segmentation_parameter_template
		if os.path.isfile(self.segmentation_parameter_template["predictedPath"]):
			os.remove(self.segmentation_parameter_template["predictedPath"])

	def segment(self, name_extension):
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