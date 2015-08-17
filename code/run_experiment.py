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
	def get_base_parameters(base_project_path, experiment_name, model_template):
		NN_code_directory 				= os.path.join(base_project_path, "code/CNN")
		dataset_directory				= os.path.join(base_project_path, "datasets")
		experimental_results_directory  = os.path.join(base_project_path, "experimental_results")

		model_name 						= model_template.replace("_template", "")
		base_training_parameters = {
			"GPU_identifier"	: 1,
			"number_of_GPUs"	: 4,
			"savingDirectory"	: os.path.join(experimental_results_directory, experiment_name),
			"presavedModelPath"	: "",
			"modelFilePath"		: os.path.join(os.path.join(NN_code_directory, "models"), model_name),
			"maxepoch"			: 100, 
			"learningRate"		: 0.1, 
			"batchSize"			: 1500*4, 
			"momentum"			: 0.0, 
			"training_dataset" 	: os.path.join(dataset_directory,"small_atrium_box_training_dataset.hdf5"),
			"testing_dataset" 	: os.path.join(dataset_directory,"testing_dataset.hdf5")
		}

		base_model_parameters = {
			"NN_code_directory"		: NN_code_directory,
			"modelTemplateDirectory": os.path.join(NN_code_directory, "model_templates"),
			"modelTemplate" 		: model_template,
			"modelFilePath"			: os.path.join(os.path.join(NN_code_directory, "models"), model_name),
			"activation_function"	: "ReLU",
			"nfeaturemaps"  		: [64,1000],
			"filtsize" 	  			: 5,
			"poolsize" 	  			: [2,2],
			"featuremaps_h" 		: 14,
			"featuremaps_w" 		: 14,
			"noutputs" 	  			: 2
		}

		base_segmentation_parameters = {
			"GPU_id"			: 1,
			"number_of_GPUs"	: 4,
			"segmentationCode"	: os.path.join(NN_code_directory, "segment.lua"),
			"segmentationFile" 	: os.path.join(dataset_directory,"segmentation_datasets.hdf5"),
			"modelDirectory"	: os.path.join(experimental_results_directory, experiment_name)
		}
		return dict(base_training_parameters), dict(base_model_parameters), dict(base_segmentation_parameters)

	def start_experiment(training_parameters, model_parameters, segmentation_parameters):
		# Train the model
		model 		= Model(training_parameters, model_parameters)
		experiment 	= Experiment(model)
		experiment.run_experiment()

		# Produce some segmentation results
		segmentator = Segmentator(segmentation_parameters)
		segmentator.segment()

	# ************************************************************************************************
	# 										Run the experiments for varying parameters
	# ************************************************************************************************
	base_project_path 			= os.path.abspath("..")
	dataset_directory			= os.path.join(base_project_path, "datasets")

	experiment_name 	= "varying_activation_functions/ReLU"
	model_template 		= "model_template_1_conn_layers.lua"
	training_parameters, model_parameters, segmentation_parameters = get_base_parameters(base_project_path, experiment_name, model_template)
	model_parameters["activation_function"] = "ReLU"
	start_experiment(training_parameters, model_parameters, segmentation_parameters)

	experiment_name 	= "varying_activation_functions/Tanh"
	model_template 		= "model_template_1_conn_layers.lua"
	training_parameters, model_parameters, segmentation_parameters = get_base_parameters(base_project_path, experiment_name, model_template)
	model_parameters["activation_function"] = "Tanh"
	start_experiment(training_parameters, model_parameters, segmentation_parameters)

	experiment_name 	= "varying_activation_functions/Sigmoid"
	model_template 		= "model_template_1_conn_layers.lua"
	training_parameters, model_parameters, segmentation_parameters = get_base_parameters(base_project_path, experiment_name, model_template)
	model_parameters["activation_function"] = "Sigmoid"
	start_experiment(training_parameters, model_parameters, segmentation_parameters)
	















