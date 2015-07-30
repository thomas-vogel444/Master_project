import subprocess
import os

class Segmentator:
	def __init__(self, segmentation_parameters):
		self.segmentation_parameters = segmentation_parameters
		if os.path.isfile(segmentation_parameters["predictedPath"]):
			os.remove(segmentation_parameters["predictedPath"])

	def segment(self, segmentation_particular):
		segmentation_command_options = self.get_segmentation_command_options(segmentation_particular)
		segmentation_command = "th segment.lua -GPU %(GPU)i -segmentationFile %(segmentationFile)s -segmentationLabels %(segmentationLabels)s "\
								"-segmentationValues %(segmentationValues)s -segmentationDataset %(segmentationDataset)s "\
								"-predictedPath %(predictedPath)s -predictedDataset %(predictedDataset)s -modelPath %(modelPath)s "\
								"-type %(type)s" %segmentation_command_options

		print "******************** Running the following command ********************"
		print segmentation_command
		print 
		subprocess.call(segmentation_command, shell=True)

	def get_segmentation_command_options(self, segmentation_particular):
		return dict(self.segmentation_parameters.items() + 
					{"segmentationDataset"	: segmentation_particular[0]}.items() + 
					{"segmentationLabels"	: segmentation_particular[1]}.items() + 
					{"segmentationValues"	: segmentation_particular[2]}.items() + 
					{"predictedDataset"		: segmentation_particular[3]}.items())
