import subprocess
import os
from string import Template
import shutil
import subprocess
import os
import pprint as pp
"""
	Template run.py script to show you how it's done...
"""

def train_model(training_parameters):
	training_command = "th main.lua -seed %(seed)i -threads %(threads)i -identifier %(identifier)i -dataset %(dataset)s -size %(size)s "\
			"-model %(model)s -maxepoch %(maxepoch)i -save %(save)s -learningRate %(learningRate)f -batchSize %(batchSize)i -weightDecay %(weightDecay)f "\
			"-momentum %(momentum)f -type %(type)s" %training_parameters

	print "******************** Running the following command ********************"
	print training_command
	print "******************** Running the following command ********************"
	subprocess.call(training_command, shell=True)

#***************************************************************************************************************
if __name__ == "__main__":
	# Parameters for the set of experiments to be conducted
	experiment_directory = os.getcwd()
	code_directory = "../code/CNN"
	os.chdir(code_directory)

	training_parameters = {
		"seed"			: 1, 
		"threads"		: 2, 
		"identifier" 	: 1, 
		"dataset" 		: os.path.abspath("../../datasets/small_CNN_datasets.hdf5"), 
		"size" 			: "full", 
		"model" 		: "convnet", 
		"maxepoch"		: 1, 
		"save"			: os.path.abspath(os.path.join(experiment_directory, "test_run")), 
		"learningRate"	: 0.1, 
		"batchSize"		: 512, 
		"weightDecay"	: 0.0, 
		"momentum"		: 0.0, 
		"type"			: "float"
	}

	# A bunch of experiments
	learningRates	= (0.1, 0.05, 0.01)

	for identifier, learningRate in enumerate(learningRates):
		training_parameters["identifier"] 	= identifier
		training_parameters["save"] 		= os.path.abspath(os.path.join(experiment_directory, str(identifier)))
		training_parameters["learningRate"] = learningRate

		train_model(training_parameters)

	os.chdir(experiment_directory)
