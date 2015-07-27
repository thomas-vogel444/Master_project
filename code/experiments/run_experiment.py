import json
import subprocess
import os
import pprint as pp	
import shutil

class Experiment:
	def __init__(self, experiment_parameters, base_model_generator, varying_parameter, varying_parameter_values):
		self.experiment_parameters		= experiment_parameters
		self.base_model_generator		= base_model_generator
		self.varying_parameter 			= varying_parameter
		self.varying_parameter_values 	= varying_parameter_values

	def run_experiment(self):
		# Create the experiment directory
		if os.path.exists(self.experiment_parameters["experiment_results_directory"]):
			shutil.rmtree(self.experiment_parameters["experiment_results_directory"])

		os.makedirs(self.experiment_parameters["experiment_results_directory"])

		# Change directory to the code directory
		os.chdir(self.experiment_parameters["NN_code_directory"])

		# Train a new model for every parameter value that needs varying
		for identifier, varying_parameter_value in enumerate(self.varying_parameter_values):
			saving_directory = self.get_model_saving_directory(identifier)
			
			model = base_model_generator.new_model(saving_directory, varying_parameter=varying_parameter, varying_parameter_value=varying_parameter_value)
			model.train()

			self.save_as_json(model.training_parameters, "training_parameters.json", saving_directory)
			self.save_as_json(model.model_parameters, "model_parameters.json", saving_directory)

		# Come back to the experiment directory
		os.chdir(self.experiment_parameters["experiment_code_directory"])

	def get_model_saving_directory(self, identifier):
		return os.path.join(self.experiment_parameters["experiment_results_directory"], str(identifier))

	def save_as_json(self, parameters, filename, saving_directory):
		json_path = os.path.join(saving_directory, filename)
		with open(json_path, 'w') as file:
			json.dump(parameters, file, indent=4, separators=(',', ': '))


class BaseModelGenerator:
	"""
		BaseModelGenerator is responsible for generating model objects with only one parameter modified from a basic set of parameters.
	"""
	def __init__(self, training_parameters, model_parameters):
		self.training_parameters 	= training_parameters
		self.model_parameters		= model_parameters

	def new_model(self, saving_directory, varying_parameter=None, varying_parameter_value=None):
		if varying_parameter is not None:
			new_training_parameters, new_model_parameters = self.generate_new_parameters(varying_parameter, varying_parameter_value)
		else:
			new_training_parameters, new_model_parameters = self.training_parameters, self.model_parameters

		return Model(new_training_parameters, new_model_parameters, saving_directory)	

	def generate_new_parameters(self, varying_parameter, varying_parameter_value):
		new_training_parameters	= self.training_parameters
		new_model_parameters	= self.model_parameters

		if varying_parameter in new_model_parameters:
			new_model_parameters[varying_parameter] = varying_parameter_value
		elif varying_parameter in new_training_parameters:
			new_training_parameters[varying_parameter] = varying_parameter_value

		return new_training_parameters, new_model_parameters

class Model:
	"""
		Model is responsable for training a Neural Network given training and model parameters using the torch code.
	"""
	def __init__(self, training_parameters, model_parameters, saving_directory):
		self.training_parameters = training_parameters
		self.model_parameters 	 = model_parameters
		self.saving_directory 	 = saving_directory

	def train(self):
		"""
			Calls the linux command to train a neural network.
		"""
		command_options = self.get_training_command_options()
		training_command = "th main.lua -dataset %(dataset)s -maxepoch %(maxepoch)i -savingDirectory %(savingDirectory)s "\
							"-learningRate %(learningRate)f -batchSize %(batchSize)i "\
							"-momentum %(momentum)f -type %(type)s" %command_options
		print "******************** Training with the following parameters ********************"
		pp.pprint(training_parameters)
		subprocess.call(training_command, shell=True)

	def get_training_command_options(self):
		"""
			The linux command invoking the lua code to train the NN requires the inclusion of the saving directory in its options.
		"""
		return dict(self.training_parameters.items() + {"savingDirectory": self.saving_directory}.items())

#***************************************************************************************************************
if __name__ == "__main__":
	# Parameters for the set of experiments to be conducted
	experiment_name = "test_experiment"

	experiment_parameters = {
		"experiment_code_directory"		: os.getcwd(),
		"experiment_results_directory" 	: os.path.join(os.path.abspath("../../experimental_results"), experiment_name),
		"NN_code_directory" 			: os.path.abspath("../CNN")
	}

	training_parameters = {
		"maxepoch"		: 1, 
		"learningRate"	: 0.1, 
		"batchSize"		: 512, 
		"momentum"		: 0.0, 
		"dataset" 		: os.path.abspath("../../datasets/small_CNN_datasets.hdf5"),
		"type"			: "float"
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

	varying_parameter 			= "learningRate"
	varying_parameter_values 	= [0.1, 0.05, 0.01]
	base_model_generator 		= BaseModelGenerator(training_parameters, model_parameters)
	experiment 					= Experiment(experiment_parameters, base_model_generator, varying_parameter, varying_parameter_values)
	experiment.run_experiment()



