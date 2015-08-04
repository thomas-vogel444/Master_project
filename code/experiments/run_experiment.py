from lib.Segmentator import Segmentator
from lib.Experiment import Experiment, Model
from jinja2 import Environment, FileSystemLoader
import os

#***************************************************************************************************************
if __name__ == "__main__":
	# ************************************************************************************************
	# 						Base parameters for the set of experiments to be conducted
	# ************************************************************************************************
	experiment_name 			= "test_experiment/0"
	experiment_results_directory= os.path.join(os.path.abspath("../../experimental_results"), experiment_name)
	
	NN_code_directory 			= os.path.abspath("../CNN")
	model_template_directory 	= os.path.join(NN_code_directory, "model_templates")
	models_directory			= os.path.join(NN_code_directory, "models")

	segmentation_code_path		= os.path.join(NN_code_directory, "segment.lua")
	segmentation_file_path 		= os.path.abspath("../../datasets/segmentation_datasets.hdf5")

	training_parameters = {
		"type"				: "cuda",
		"GPU_identifier"	: 1,
		"savingDirectory"	: experiment_results_directory
		"modelPath"			: 
		"maxepoch"			: 1, 
		"learningRate"		: 0.1, 
		"batchSize"			: 512, 
		"momentum"			: 0.0, 
		"dataset" 			: os.path.abspath("../../datasets/CNN_small_atrium_box_datasets_110000.hdf5.hdf5"),
	}

	model_parameters = {
		"modelTemplate" : os.path.join(model_template_directory, "model_template.lua")
		"modelPath"		: os.path.join(models_directory, "test_model.lua")
		"nfeats"  		: 6,
		"patchsize"  	: 32,
		"nfeaturemaps"  : [32,64,1000,1000],
		"filtsize" 	  	: 5,
		"poolsize" 	  	: [3,2],
		"featuremaps_h" : 2,
		"featuremaps_w" : 2,
		"noutputs" 	  	: 2
	}

	# Generate the model source file
	template_environment = Environment(loader = FileSystemLoader(model_parameters["modelTemplate"]))
	with open(model_parameters["modelPath"], 'w') as f:
	    f.write(template_environment.render(model_parameters))

	# ************************************************************************************************
	# 									Parameters for the segmentation
	# ************************************************************************************************
	segmentation_parameters = {
		"GPU"				: 2,
		"segmentationCode"	: segmentation_code_path,
		"segmentationFile" 	: segmentation_file_path,
		"modelPath" 		: os.path.join(experiment_results_directory, "model.net")
	}

	# ************************************************************************************************
	# 										Run the experiment
	# ************************************************************************************************
	model 		= Model(training_parameters, model_parameters)
	experiment 	= Experiment(NN_code_directory, model)
	experiment.run_experiment()

	# Produce some segmentation results
	for segmentation_type in segmentation_types:
		segmentation_parameters["segmentationDataset"]	= segmentation_parameters["segmentationDataset"].replace("NAME", segmentation_type)
		segmentation_parameters["segmentationLabels"]	= segmentation_parameters["segmentationLabels"].replace("NAME", segmentation_type)
		segmentation_parameters["segmentationValues"]	= segmentation_parameters["segmentationValues"].replace("NAME", segmentation_type)
		segmentation_parameters["predictedDataset"]		= segmentation_parameters["predictedDataset"].replace("NAME", segmentation_type)

		segmentator = Segmentator(segmentation_parameters)
		segmentator.segment()
