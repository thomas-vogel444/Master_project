import os
import subprocess

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
		# Change directory to the code directory
		current_working_directory = os.getcwd()
		os.chdir(self.model_parameters["NN_code_directory"])

		self.call_training_routine()

		# Come back to the original directory
		os.chdir(current_working_directory)

	def call_training_routine(self):
		"""
			Calls the linux command to train a neural network.
		"""
		training_command  = "th main.lua -GPU_id %(GPU_identifier)i -number_of_GPUs %(number_of_GPUs)i -training_dataset %(training_dataset)s "\
							"-testing_dataset %(testing_dataset)s -modelFilePath %(modelFilePath)s "\
							"-maxepoch %(maxepoch)i -savingDirectory %(savingDirectory)s -learningRate %(learningRate)f "\
							"-batchSize %(batchSize)i -momentum %(momentum)f" %self.training_parameters

		if self.training_parameters["presavedModelPath"] != "":
			training_command += " -presavedModelPath %s"%self.training_parameters["presavedModelPath"]

		# Call the training command
		subprocess.call(training_command, shell=True)
