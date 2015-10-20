import jinja2
import json
import subprocess
import os
import shutil


class Experiment:
	"""
		This class is responsible for handling all the logistics involved in training a Neural Network.
	"""
    def __init__(self, model):
        self.model = model

    def run_experiment(self):
		"""
			Sets up the logistics for training a model, trains the model, and finish with some post-training logistics.
		"""
		# Generate the model source file
		self.generate_model_source_file()
		
		# Train the model
		self.model.train()

		# Post training step
		self.save_as_json(self.model.training_parameters, "training_parameters.json")
		self.save_model()

	def generate_model_source_file(self):
		"""
			Generates the model source file from a model template filling in the model values with those in model parameters.
		"""
		template_environment = jinja2.Environment(loader = jinja2.FileSystemLoader(self.model.model_parameters["modelTemplateDirectory"]))
		model_template 		 = template_environment.get_template(self.model.model_parameters["modelTemplate"])

		with open(self.model.model_parameters["modelFilePath"], 'w+') as f:
		    f.write(model_template.render(self.model.model_parameters))

	def save_as_json(self, parameters, filename):
		"""
			Saves a dictionary of parameters into a json file in a given saving directory.
		"""
		json_path = os.path.join(self.model.training_parameters["savingDirectory"], filename)
		with open(json_path, 'w') as file:
			json.dump(parameters, file, indent=4, separators=(',', ': '))

	def save_model(self):
		"""
			Saves the model code in the saving directory of the experiment.
		"""
		model_filename = os.path.basename(self.model.model_parameters["modelFilePath"])
		if os.path.isfile(os.path.join(self.model.training_parameters["savingDirectory"], model_filename)):
			os.remove(os.path.join(self.model.training_parameters["savingDirectory"], model_filename))

		shutil.move(self.model.model_parameters["modelFilePath"], self.model.training_parameters["savingDirectory"])
