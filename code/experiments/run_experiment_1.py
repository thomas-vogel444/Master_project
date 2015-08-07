from lib.Segmentator import Segmentator
from lib.Experiment import Experiment, Model
import threading
import os

#***************************************************************************************************************
if __name__ == "__main__":
	# ************************************************************************************************
	# 						Base parameters for the set of experiments to be conducted
	# ************************************************************************************************
	NN_code_directory 				= os.path.abspath("../CNN")
	dataset_directory				= os.path.abspath("../../datasets")
	experimental_results_directory  = os.path.abspath("../../experimental_results")

	def get_base_parameters(experiment_name, model_template, model_identifier = ""):
		model_name 						= model_template.replace("_template", model_identifier)
		base_training_parameters = {
			"type"				: "cuda",
			"GPU_identifier"	: 1,
			"number_of_GPUs"	: 1,
			"savingDirectory"	: os.path.join(experimental_results_directory, experiment_name),
			"modelPath"			: os.path.join(os.path.join(NN_code_directory, "models"), model_name),
			"maxepoch"			: 15, 
			"learningRate"		: 0.1, 
			"batchSize"			: 512, 
			"momentum"			: 0.0, 
			"dataset" 			: os.path.join(dataset_directory,"CNN_small_atrium_box_datasets.hdf5")
		}

		base_model_parameters = {
			"NN_code_directory"		: NN_code_directory,
			"modelTemplateDirectory": os.path.join(NN_code_directory, "model_templates"),
			"modelTemplate" 		: model_template,
			"modelPath"				: os.path.join(os.path.join(NN_code_directory, "models"), model_name),
			"nfeats"  				: 6,
			"patchsize"  			: 32,
			"nfeaturemaps"  		: [32,64,1000,1000],
			"filtsize" 	  			: 5,
			"poolsize" 	  			: [3,2],
			"featuremaps_h" 		: 2,
			"featuremaps_w" 		: 2,
			"noutputs" 	  			: 2
		}

		base_segmentation_parameters = {
			"GPU"				: 1,
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
	# 										Run the experiments 
	# ************************************************************************************************
	experiment_name = "varying_datasets/small_atrium_box"
	model_template 	= "model_template.lua"	
	training_parameters, model_parameters, segmentation_parameters = get_base_parameters(experiment_name, model_template, "_small_atrium_box_")
	training_parameters["dataset"] 		  	= os.path.join(dataset_directory, "CNN_small_atrium_box_datasets.hdf5")
	start_experiment(training_parameters, model_parameters, segmentation_parameters)



