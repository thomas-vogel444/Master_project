import jinja2
import subprocess
import os
"""
	Template run.py script to show you how it's done...
"""

def train_model(training_parameters):
	training_command = "th main.lua -seed %(seed)i -threads %(threads)i -identifier %(identifier)i -dataset %(dataset)s -size %(size)s "\
			"-model %(model)s -maxepoch %(maxepoch)i -save %(save)s -learningRate %(learningRate)f -batchSize %(batchSize)i -weightDecay %(weightDecay)f "\
			"-momentum %(momentum)f -type %(type)s" %training_parameters

	print "******************** Running the following command ********************"
	print training_command
	print
	subprocess.call(training_command, shell=True)

#***************************************************************************************************************
if __name__ == "__main__":
	# Experiment parameters
	experiment_directory = os.getcwd()
	code_directory = "../../code/CNN"
	model_template = 'model_template.lua'
	os.chdir(code_directory)

	training_parameters = {
		"seed"			: 1, 
		"threads"		: 2, 
		"identifier" 	: 1, 
		"dataset" 		: os.path.abspath("../../datasets/CNN_box_atrium_datasets.hdf5"), 
		"size" 			: "full", 
		"model" 		: "convnet", 
		"maxepoch"		: 30, 
		"save"			: os.path.abspath(os.path.join(experiment_directory, "test_run")), 
		"learningRate"	: 0.1, 
		"batchSize"		: 512, 
		"weightDecay"	: 0.0, 
		"momentum"		: 0.0, 
		"type"			: "cuda"
	}

	model_parameters = {
		"nfeats"  		: 6,
		"patchsize"  	: 32,
		"nfeaturemaps"  : [32,64,1000,1000],
		"filtsize" 	  	: 5,
		"poolsize" 	  	: [3,2],
		"featuremaps_h" : 2,
		"featuremaps_w" : 2,
		"noutputs" 	  	: 2
	}

	# Loading the model template and rendering
	templateEnv = jinja2.Environment(loader = jinja2.FileSystemLoader('.'))
	template 	= templateEnv.get_template(model_template)
	model_text  = template.render(model_parameters)

	# Saving the model in model.lua
	with open('model.lua', 'w') as f:
		f.write(model_text)

	# A bunch of experiments
	momentums	= (0.5, 0.1, 0.05, 0.01)

	for identifier, momentum in enumerate(momentums):
		training_parameters["identifier"] 	= identifier
		training_parameters["save"] 		= os.path.abspath(os.path.join(experiment_directory, str(identifier)))
		training_parameters["momentum"] = momentum

		train_model(training_parameters)

	os.chdir(experiment_directory)

