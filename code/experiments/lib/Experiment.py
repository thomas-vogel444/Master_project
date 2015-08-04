import jinja2
import json
import subprocess
import os
import pprint as pp	
import shutil

class Experiment:
	"""
		This class is responsible for handling all the logistics involved in training a Neural Network.
	"""
	def __init__(self, NN_code_directory, model):
		self.NN_code_directory	= NN_code_directory
		self.model 				= model

	def run_experiment(self):
		"""
			Sets up the logistics for training a model, trains the model, and finish with some post-training logistics.
		"""
		current_working_directory = os.getcwd()

		# Generate the model source file
		self.generate_model_source_file()

		# Change directory to the code directory
		os.chdir(self.NN_code_directory)
		
		# Train the model
		model.train()

		# Post training step
		self.save_as_json(model.training_parameters, "training_parameters.json", saving_directory)
		self.save_as_json(model.model_parameters, "model_parameters.json", saving_directory)

		# Come back to the original directory
		os.chdir(current_working_directory)

	def generate_model_source_file(self):
		"""
			Generates the model source file from a model template filling in the model values with those in model parameters.
		"""
		template_environment = jinja2.Environment(loader = jinja2.FileSystemLoader(self.model.model_parameters["modelTemplateDirectory"]))
		model_template 		 = template_environment.get_template(self.model.model_parameters["modelTemplate"])

		with open(self.model.model_parameters["modelPath"], 'w+') as f:
		    f.write(model_template.render(self.model.model_parameters))

	def save_as_json(self, parameters, filename, saving_directory):
		"""
			Saves a dictionary of parameters into a json file in a given saving directory.
		"""
		json_path = os.path.join(saving_directory, filename)
		with open(json_path, 'w') as file:
			json.dump(parameters, file, indent=4, separators=(',', ': '))

class Model:
	"""
		Model is responsable for calling the lua code to train a Neural Network given training and model parameters.
	"""
	def __init__(self, training_parameters, model_parameters):
		self.training_parameters = training_parameters
		self.model_parameters 	 = model_parameters

	def train(self):
		"""
			Calls the linux command to train a neural network.
		"""
		command_options = self.get_training_command_options()
		training_command = "th main.lua -GPU %(GPU_identifier)i -dataset %(dataset)s -modelPath %(modelPath)s -maxepoch %(maxepoch)i "\
							"-savingDirectory %(savingDirectory)s -learningRate %(learningRate)f -batchSize %(batchSize)i "\
							"-momentum %(momentum)f -type %(type)s" %command_options
		subprocess.call(training_command, shell=True)

