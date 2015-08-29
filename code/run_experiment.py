from experiments.Segmentator import Segmentator
from experiments.Experiment import Experiment
from experiments.Model import Model
import threading
import os
import shutil

#***************************************************************************************************************
if __name__ == "__main__":
	# ************************************************************************************************
	# 						Base parameters for the set of experiments to be conducted
	# ************************************************************************************************
	base_project_path 				= os.path.abspath("..")
	NN_code_directory 				= os.path.join(base_project_path, "code/CNN")
	dataset_directory				= os.path.join(base_project_path, "datasets")
	experimental_results_directory  = os.path.join(base_project_path, "experimental_results")

	def get_base_parameters(experiment_name, model_template):
		model_name 						= model_template.replace("_template", "")
		base_training_parameters = {
			"GPU_identifier"	: 1,
			"number_of_GPUs"	: 4,
			"savingDirectory"	: os.path.join(experimental_results_directory, experiment_name),
			"presavedModelPath"	: "",
			"modelFilePath"		: os.path.join(os.path.join(NN_code_directory, "models"), model_name),
			"maxepoch"			: 100, 
			"learningRate"		: 0.5, 
			"batchSize"			: 1500*4, 
			"momentum"			: 0, 
			"training_dataset" 	: os.path.join(dataset_directory,"no_atrium_box_training_dataset.hdf5"),
			"testing_dataset" 	: os.path.join(dataset_directory,"testing_dataset.hdf5")
		}

		base_model_parameters = {
			"NN_code_directory"		: NN_code_directory,
			"modelTemplateDirectory": os.path.join(NN_code_directory, "model_templates"),
			"modelTemplate" 		: model_template,
			"modelFilePath"			: os.path.join(os.path.join(NN_code_directory, "models"), model_name),
			"activation_function"	: "ReLU",
			"pooling_function"		: "SpatialMaxPooling",
			"nfeaturemaps"  		: [32,64,100],
			"filtsize" 	  			: 5,
			"poolsize" 	  			: [2,2],
			"featuremaps_h" 		: 5,
			"featuremaps_w" 		: 5,
			"noutputs" 	  			: 2
		}

		return dict(base_training_parameters), dict(base_model_parameters)

	def start_experiment(training_parameters, model_parameters):
		# Train the model
		model 		= Model(training_parameters, model_parameters)
		experiment 	= Experiment(model)
		experiment.run_experiment()

	# ************************************************************************************************
	# 										Run the experiments for varying parameters
	# ************************************************************************************************
	model_template 		= "model_template_2_conv_1_conn_layers.lua"

	experiment_name 	= "varying_momentum/0"
	training_parameters, model_parameters = get_base_parameters(experiment_name, model_template)
	training_parameters["momentum"]  = 0

	start_experiment(training_parameters, model_parameters)

	experiment_name 	= "varying_momentum/0_05"
	training_parameters, model_parameters = get_base_parameters(experiment_name, model_template)
	training_parameters["momentum"]  = 0.05

	start_experiment(training_parameters, model_parameters)

	experiment_name 	= "varying_momentum/0_1"
	training_parameters, model_parameters = get_base_parameters(experiment_name, model_template)
	training_parameters["momentum"]  = 0.1

	start_experiment(training_parameters, model_parameters)

	experiment_name 	= "varying_momentum/0_5"
	training_parameters, model_parameters = get_base_parameters(experiment_name, model_template)
	training_parameters["momentum"]  = 0.5

	start_experiment(training_parameters, model_parameters)
