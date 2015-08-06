from Segmentator import Segmentator
from Experiment import Experiment, Model
import os

#***************************************************************************************************************
if __name__ == "__main__":
	# ************************************************************************************************
	# 						Base parameters for the set of experiments to be conducted
	# ************************************************************************************************
	experiment_name 				= "test_experiment/0"
	model_name 						= "test_model.lua"
	NN_code_directory 				= os.path.abspath("../../CNN")
	dataset_directory				= os.path.abspath("../../../datasets")
	experimental_results_directory  = os.path.abspath("../../../experimental_results")
	
	training_parameters = {
		"type"				: "cuda",
		"GPU_identifier"	: 1,
		"savingDirectory"	: os.path.join(experimental_results_directory, experiment_name),
		"modelPath"			: os.path.join(os.path.join(NN_code_directory, "models"), model_name),
		"maxepoch"			: 5, 
		"learningRate"		: 0.1, 
		"batchSize"			: 512, 
		"momentum"			: 0.0, 
		"dataset" 			: os.path.join(dataset_directory, "CNN_small_atrium_box_datasets_44000.hdf5")
	}

	model_parameters = {
		"modelTemplateDirectory": os.path.join(NN_code_directory, "model_templates"),
		"modelTemplate" 		: "model_template.lua",
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

	# ************************************************************************************************
	# 									Parameters for the segmentation
	# ************************************************************************************************
	segmentation_parameters = {
		"GPU"				: 2,
		"segmentationCode"	: os.path.join(NN_code_directory, "segment.lua"),
		"segmentationFile" 	: os.path.join(dataset_directory, "datasets/segmentation_datasets.hdf5"),
		"modelDirectory"	: os.path.join(experimental_results_directory, experiment_name)
	}

	# ************************************************************************************************
	# 										Run the experiment
	# ************************************************************************************************
	model 		= Model(training_parameters, model_parameters)
	experiment 	= Experiment(NN_code_directory, model)
	experiment.run_experiment()

	# Produce some segmentation results
	segmentator = Segmentator(segmentation_parameters)
	segmentator.segment()

