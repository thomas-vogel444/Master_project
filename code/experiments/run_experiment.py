from lib.Experiment import Experiment, BaseModelGenerator
import os

#***************************************************************************************************************
if __name__ == "__main__":
	# Parameters for the set of experiments to be conducted
	experiment_name 			= "varying_first_layer_nfeaturemaps"
	varying_parameter 			= "nfeaturemaps"
	varying_parameter_values 	= [[32,64,1000,1000], [64,64,1000,1000], [92,64,1000,1000], [128,64,1000,1000]]

	experiment_parameters = {
		"experiment_code_directory"		: os.getcwd(),
		"experiment_results_directory" 	: os.path.join(os.path.abspath("../../experimental_results"), experiment_name),
		"NN_code_directory" 			: os.path.abspath("../CNN")
	}

	training_parameters = {
		"maxepoch"		: 30, 
		"learningRate"	: 0.1, 
		"batchSize"		: 512, 
		"momentum"		: 0.0, 
		"dataset" 		: os.path.abspath("../../datasets/CNN_box_atrium_datasets.hdf5"),
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

	base_model_generator 		= BaseModelGenerator(training_parameters, model_parameters)
	experiment 					= Experiment(experiment_parameters, base_model_generator, varying_parameter, varying_parameter_values)
	experiment.run_experiment()



