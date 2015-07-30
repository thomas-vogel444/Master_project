from lib.Segmentator import Segmentator
from lib.Experiment import Experiment, BaseModelGenerator
import os

#***************************************************************************************************************
if __name__ == "__main__":
	# Parameters for the set of experiments to be conducted
	experiment_name 			= "test_experiment"
	varying_parameter 			= "learningRate"
	varying_parameter_values 	= [0.1, 0.05, 0.01]
	training_dataset_path		= os.path.abspath("../../datasets/small_CNN_datasets.hdf5")
	segmentation_code_path		= "../CNN/segment.lua"
	segmentation_file_path 		= "../../datasets/small_segmentation_datasets.hdf5"

	experiment_parameters = {
		"experiment_code_directory"		: os.getcwd(),
		"experiment_results_directory" 	: os.path.join(os.path.abspath("../../experimental_results"), experiment_name),
		"NN_code_directory" 			: os.path.abspath("../CNN")
	}

	training_parameters = {
		"maxepoch"		: 15, 
		"learningRate"	: 0.1, 
		"batchSize"		: 512, 
		"momentum"		: 0.0, 
		"dataset" 		: training_dataset_path,
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

	base_model_generator = BaseModelGenerator(training_parameters, model_parameters)
	segmentator 		 = Segmentator(segmentation_file_path, segmentation_code_path)
	experiment 			 = Experiment(experiment_parameters, base_model_generator, segmentator, varying_parameter, varying_parameter_values)
	experiment.run_experiment()



